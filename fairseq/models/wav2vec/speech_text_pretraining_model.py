# @Time    : 2021/11/24
# @Author  : Minglun Han
# @File    : speech_text_pretraining_model.py

import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange, index_put, is_xla_tensor
from fairseq.distributed import fsdp_wrap
from fairseq.tasks import FairseqTask

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass
class Wav2Vec2Config(FairseqDataclass):
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    quantize_targets: bool = field(
        default=True, metadata={"help": "use quantized targets"}
    )
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )
    same_quantizer: bool = field(
        default=False, metadata={"help": "use same quantizer for inputs and targets"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )
    quantizer_depth: int = field(
        default=1,
        metadata={"help": "number of quantizer layers"},
    )
    quantizer_factor: int = field(
        default=3,
        metadata={
            "help": "dimensionality increase for inner quantizer layers (if depth > 1)"
        },
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_before: bool = False
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )


@dataclass
class SpeechTextPretrainingConfig(Wav2Vec2Config):
    # Add configurations of cross_modal_encoder
    cross_modal_encoder_layers: int = field(
        default=6, metadata={"help": "the number of layers of the cross modal encoder."}
    )
    cross_modal_encoder_embed_dim: int = field(
        default=768,
        metadata={"help": "the embedding dimension of the cross modal encoder."},
    )
    cross_modal_encoder_ffn_dim: int = field(
        default=3072,
        metadata={"help": "the feed forward dimension of the cross modal encoder."},
    )
    cross_modal_encoder_num_heads: int = field(
        default=12, metadata={"help": "the number of heads of the cross modal encoder."}
    )
    ce_encoder_layer_norm_first: bool = field(default=False)
    ce_encoder_inputs_dropout: float = field(default=0.1)
    disable_ce_encoder: bool = field(
        default=False, metadata={"help": "whether to disable cross modal encoder"}
    )

    # Add configurations of text encoder module
    no_scale_text_embedding: bool = field(
        default=True, metadata={"help": "whether to scale text embeddings."}
    )
    max_text_seq_positions: int = field(
        default=1024,
        metadata={"help": "Maximum input length supported by the positional encoding."},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={"help": "whether to use position embeddings for token inputs"},
    )
    learned_pos: bool = field(
        default=False,
        metadata={
            "help": "whether to employ positional embedding, otherwise employ positional encoding."
        },
    )
    text_input_dropout: float = field(
        default=0.1,
        metadata={
            "help": "the dropout rate for the input text embeddings of text encoder."
        },
    )
    text_encoder_layers: int = field(
        default=6, metadata={"help": "the number of layers of the cross modal encoder."}
    )
    text_encoder_embed_dim: int = field(
        default=768,
        metadata={"help": "the embedding dimension of the cross modal encoder."},
    )
    text_encoder_ffn_dim: int = field(
        default=3072,
        metadata={"help": "the feed forward dimension of the cross modal encoder."},
    )
    text_encoder_num_heads: int = field(
        default=12, metadata={"help": "the number of heads of the cross modal encoder."}
    )
    text_encoder_layer_norm_first: bool = field(default=False)

    # cif settings
    cif_input_embed_dim: int = field(
        default=768, metadata={"help": "encoder output embedding dimension"}
    )
    cif_embedding_dim: int = field(
        default=512, metadata={"help": "cif output embedding dimension"}
    )
    produce_weight_type: str = field(
        default="conv",
        metadata={"help": "the style of weight generation from encoder outputs"},
    )
    cif_threshold: float = field(
        default=0.99,
        metadata={"help": "the threshold of accumulated weight for firing"},
    )
    conv_cif_layer_num: int = field(
        default=1, metadata={"help": "the number of cif convolution layers"}
    )
    conv_cif_width: int = field(
        default=3, metadata={"help": "the width of kernel of CIF convolution layer"}
    )
    conv_cif_output_channels_num: int = field(
        default=768, metadata={"help": "the number of CIF convolution output channels"}
    )
    conv_cif_dropout: float = field(
        default=0.0,
        metadata={"help": "the dropout rate of the final convolutional layer"},
    )
    dense_cif_units_num: int = field(
        default=768,
        metadata={"help": "the projection size of dense cif weight projection"},
    )
    apply_scaling: bool = field(
        default=True, metadata={"help": "scale the summation of all weights"}
    )
    apply_tail_handling: bool = field(
        default=True,
        metadata={"help": "handle the tails of cif weights with special strategy"},
    )
    tail_handling_firing_threshold: float = field(
        default=0.5, metadata={"help": "the firing threshold of tail handling"}
    )
    add_cif_ctxt_layers: bool = field(
        default=False,
        metadata={
            "help": "whether use extra encoding layers to contextualize cif outputs"
        },
    )
    cif_ctxt_layers: int = field(
        default=2, metadata={"help": "the number of context layers for cif outputs"}
    )
    cif_ctxt_embed_dim: int = field(
        default=768,
        metadata={"help": "the embedding dimension of context layers for cif outputs"},
    )
    cif_ctxt_ffn_embed_dim: int = field(
        default=3072,
        metadata={
            "help": "the feed forward network dimension of context layers for cif outputs"
        },
    )
    cif_ctxt_attention_heads: int = field(
        default=8,
        metadata={
            "help": "the number of attention heads of context layers for cif outputs"
        },
    )
    cif_ctxt_dropout: float = field(
        default=0.1, metadata={"help": "the dropout of context layers for cif outputs"}
    )
    cif_ctxt_activation_dropout: float = field(
        default=0.0,
        metadata={"help": "the actiavtion dropout of context layers for cif outputs"},
    )
    cif_ctxt_attention_dropout: float = field(
        default=0.1,
        metadata={"help": "the attention dropout of context layers for cif outputs"},
    )
    cif_ctxt_normalize_before: bool = field(
        default=True,
        metadata={
            "help": "whether to conduct nromalization before get into next sub-block"
        },
    )

    # nar decoder settings
    nar_asr_decoder_mode: str = field(
        default="nar_decoder",
        metadata={
            "help": "the mode of decoder, there are three options: ar_decoder, nar_decoder, proj"
        },
    )
    pre_final_proj_dim: int = field(default=768)
    nar_decoder_layers: int = field(default=2)
    nar_decoder_embed_dim: int = field(default=512)
    nar_decoder_ffn_dim: int = field(default=2048)
    nar_decoder_num_heads: int = field(default=8)
    nar_decoder_dropout: float = field(default=0.1)
    nar_decoder_activation_dropout: float = field(default=0.1)
    nar_decoder_attention_dropout: float = field(default=0.1)

    # settings about the masking stratetgies for text inputs
    learned_text_mask_emb: bool = field(
        default=True,
        metadata={
            "help": "whether to employ a trainable mask embedding for mask positions."
        },
    )
    mlm_text_mask_prob: float = field(
        default=0.15,
        metadata={
            "help": "the masking probability for masked language modeling over text."
        },
    )
    mlm_text_mask_span_length: int = field(
        default=1, metadata={"help": "the length of masked span for mlm."}
    )
    tlm_text_mask_prob: float = field(
        default=0.70,
        metadata={
            "help": "the masking probability for translation language modeling over text."
        },
    )
    tlm_text_mask_span_length: int = field(
        default=1, metadata={"help": "the length of masked span for tlm."}
    )
    tlm_spec_mask_prob: float = field(
        default=0.70,
        metadata={
            "help": "the masking probability for translation language modeling over speech."
        },
    )
    tlm_spec_mask_span_length: int = field(
        default=10, metadata={"help": "the length of masked span for tlm over speech."}
    )


@register_model("speech_text_pretraining", dataclass=SpeechTextPretrainingConfig)
class SpeechTextPretrainingModel(BaseFairseqModel):
    def __init__(self, cfg: SpeechTextPretrainingConfig, task: FairseqTask):
        super().__init__()
        self.cfg = cfg
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        # Obtain default_dictionary
        self.default_dict = task.default_dictionary

        # build conv feat extractor
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        # Register mask configurations for audios
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        # Register mask configurations for text inputs
        self.learned_text_mask_emb = cfg.learned_text_mask_emb
        self.mlm_text_mask_prob = cfg.mlm_text_mask_prob
        self.mlm_text_mask_span_length = cfg.mlm_text_mask_span_length
        self.tlm_text_mask_prob = cfg.tlm_text_mask_prob
        self.tlm_text_mask_span_length = cfg.tlm_text_mask_span_length

        # Register mask configuration for acoustics in TLM
        self.tlm_spec_mask_prob = cfg.tlm_spec_mask_prob
        self.tlm_spec_mask_span_length = cfg.tlm_spec_mask_span_length

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere
        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,  # V
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,  # G
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        # A shared trainable vector for the replacement of mask locations
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )
        if cfg.learned_text_mask_emb:
            self.text_mask_emb = nn.Parameter(
                torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
            )
        else:
            self.text_mask_emb = torch.zeros(cfg.encoder_embed_dim)

        self.speech_encoder = TransformerEncoder(cfg)
        self.conv_layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )
        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        ## build position encoding or embedding for text encoder layers
        self.text_encoder_layers = cfg.text_encoder_layers
        self.text_encoder_embed_dim = cfg.text_encoder_embed_dim
        self.text_encoder_ffn_dim = cfg.text_encoder_ffn_dim
        self.text_encoder_num_heads = cfg.text_encoder_num_heads
        self.text_encoder_layer_norm_first = cfg.text_encoder_layer_norm_first
        self.embed_scale = (
            1.0
            if cfg.no_scale_text_embedding
            else math.sqrt(self.text_encoder_embed_dim)
        )
        self.text_embedding_layer = self.build_embedding(
            self.default_dict, self.text_encoder_embed_dim, path=None
        )
        self.text_embed_positions = (
            PositionalEmbedding(
                cfg.max_text_seq_positions,
                self.text_encoder_embed_dim,
                self.default_dict.pad(),
                learned=cfg.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        self.dropout_text_inputs = nn.Dropout(cfg.text_input_dropout)
        self.cls_emb = nn.Parameter(
            torch.FloatTensor(cfg.text_encoder_embed_dim).uniform_()
        )
        self.sep_emb = nn.Parameter(
            torch.FloatTensor(cfg.text_encoder_embed_dim).uniform_()
        )
        self.text_encoder_stacks = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.text_encoder_embed_dim,
                    ffn_embedding_dim=self.text_encoder_ffn_dim,
                    num_attention_heads=self.text_encoder_num_heads,
                    dropout=cfg.dropout,
                    activation_dropout=cfg.activation_dropout,
                    attention_dropout=cfg.attention_dropout,
                    layer_norm_first=cfg.text_encoder_layer_norm_first,
                )
                for _ in range(self.text_encoder_layers)
            ]
        )
        self.text_encoder_layer_norm = LayerNorm(self.text_encoder_embed_dim)

        ## build cross-modal encoder (CE encoder)
        self.disable_ce_encoder = cfg.disable_ce_encoder
        self.cross_modal_encoder_layers = cfg.cross_modal_encoder_layers
        self.cross_modal_encoder_embed_dim = cfg.cross_modal_encoder_embed_dim
        self.cross_modal_encoder_ffn_dim = cfg.cross_modal_encoder_ffn_dim
        self.cross_modal_encoder_num_heads = cfg.cross_modal_encoder_num_heads
        self.ce_encoder_layer_norm_first = cfg.ce_encoder_layer_norm_first
        self.dropout_ce_inputs = nn.Dropout(cfg.ce_encoder_inputs_dropout)
        self.cross_modal_encoder_stacks = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.cross_modal_encoder_embed_dim,
                    ffn_embedding_dim=self.cross_modal_encoder_ffn_dim,
                    num_attention_heads=self.cross_modal_encoder_num_heads,
                    dropout=cfg.dropout,
                    activation_dropout=cfg.activation_dropout,
                    attention_dropout=cfg.attention_dropout,
                    layer_norm_first=cfg.ce_encoder_layer_norm_first,
                )
                for _ in range(self.cross_modal_encoder_layers)
            ]
        )
        self.ce_encoder_layer_norm = LayerNorm(self.cross_modal_encoder_embed_dim)

        # build cls projection for speech-text matching
        self.cls_proj = nn.Linear(self.cross_modal_encoder_embed_dim, 1)

        # build masked language modeling projection for masked prediction
        self.latent_vars = cfg.latent_vars
        self.latent_groups = cfg.latent_groups
        self.quantized_vocab_size = self.latent_vars**self.latent_groups
        self.text_proj = nn.Linear(
            self.cross_modal_encoder_embed_dim, len(self.default_dict)
        )
        self.spec_proj = nn.Linear(
            self.cross_modal_encoder_embed_dim, self.latent_vars**self.latent_groups
        )

        # build asr decoder part
        self.cif = CifMiddleware(cfg=cfg)
        if cfg.nar_asr_decoder_mode == "proj":
            self.nar_asr_decoder = NarProjAsrDecoder(cfg, self.default_dict)
        elif cfg.nar_asr_decoder_mode == "nar_decoder":
            self.nar_asr_decoder = NarTransformerAsrDecoder(cfg, self.default_dict)
        else:
            self.nar_asr_decoder = NarProjAsrDecoder(cfg, self.default_dict)

        # build tts decoder part
        # TODO: build tts decoder part

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def build_embedding(self, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_model(cls, cfg: SpeechTextPretrainingConfig, task: FairseqTask = None):
        """Build a new model instance."""
        return cls(cfg, task)

    def apply_spec_channel_temporal_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def apply_spec_temporal_mask(
        self,
        x,
        padding_mask,
    ):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.tlm_spec_mask_prob,
                self.tlm_spec_mask_span_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        return x, mask_indices

    def apply_text_temporal_mask(
        self,
        x,  # x is the text embeddings after text embedding layer
        padding_mask,
        text_mask_prob=None,
        text_mask_length=None,
    ):
        B, T, C = x.shape
        mask_indices = None

        if text_mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                text_mask_prob,
                text_mask_length,
                min_masks=1,
                no_overlap=self.no_mask_overlap,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
        x = index_put(x, mask_indices, self.text_mask_emb)
        return x, mask_indices

    def sample_negatives(self, y, num, padding_count=None):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat(
            [y, negatives], dim=0
        )  # combine target with negative distractors

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits = logits / self.logit_temp

        if is_xla_tensor(logits) or neg_is_pos.any():
            fillval = -float(2**30)
            if not hasattr(self, "_inftensor"):
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def select_samples(self, samples, indices):
        selected_samples = dict()
        for key in samples.keys():
            if type(samples[key]) == torch.Tensor and samples[key].numel() > 1:
                selected_samples[key] = torch.index_select(samples[key], 0, indices)
            else:
                continue

        return selected_samples

    def combine_samples(self, items, ordered_keys, input_samples_dict):
        """
        :param items: the items wanna to combine;
        :param ordered_keys: the ordered data labels list, all samples will be organized follow this data order list;
        :param input_samples_dict: the input samples dictionary with data labe as keys;
        :return: combined samples
        """
        # Recombine all samples for convienence
        recom_sample = dict()
        for item in items:
            if item == "ntokens":
                continue
            if item == "id":
                continue
            recom_sample[item] = torch.cat(
                [input_samples_dict[data_label][item] for data_label in ordered_keys],
                dim=0,
            )
        return recom_sample

    def forward(
        self,
        sample,
        mask_audio=True,
        mask_text=True,
        features_only=False,
        mode=1,
    ):
        # Collect all keys name
        items = sample.keys()

        # Split samples according to data labels
        data_labels = sample["data_labels"]
        onehot_text_indices = (data_labels == 0).int()
        onehot_speech_indices = (data_labels == 1).int()
        onehot_paired_indices = (data_labels == 2).int()
        text_indices = torch.nonzero(onehot_text_indices).squeeze(-1)
        speech_indices = torch.nonzero(onehot_speech_indices).squeeze(-1)
        paired_indices = torch.nonzero(onehot_paired_indices).squeeze(-1)
        num_text_samples = text_indices.size(0)
        num_spec_samples = speech_indices.size(0)
        num_pair_samples = paired_indices.size(0)
        text_samples = self.select_samples(sample, text_indices)
        speech_samples = self.select_samples(sample, speech_indices)
        paired_samples = self.select_samples(sample, paired_indices)

        stm_labels = None
        num_pos_paired_samples = None
        num_total_paired_samples = None
        if mode == 2:
            num_pos_paired_samples = num_pair_samples
            num_total_paired_samples = num_pos_paired_samples * 2
            assert (
                num_pair_samples == num_spec_samples
            ), "please ensure that the number of speech samples equals that of paired samples"

            negative_paired_samples = dict()
            negative_paired_samples["data_labels"] = (
                2
                * torch.ones([num_total_paired_samples - num_pos_paired_samples])
                .int()
                .cuda()
            )
            for i in range(num_pos_paired_samples):
                spec_sample_id = torch.tensor(
                    [random.randint(0, num_spec_samples - 1)]
                ).cuda()
                text_sample_id = torch.tensor(
                    [random.randint(0, num_text_samples - 1)]
                ).cuda()
                spec_sample = self.select_samples(speech_samples, spec_sample_id)
                text_sample = self.select_samples(text_samples, text_sample_id)

                # Combine samples with negative match label
                for key in items:
                    if key == "ntokens":
                        continue
                    if key == "data_labels":
                        continue
                    if key == "id":
                        continue

                    if key == "text" or key == "target_lengths" or key == "target":
                        if key not in negative_paired_samples.keys():
                            negative_paired_samples[key] = [text_sample[key]]
                        else:
                            # negative_paired_samples[key] = negative_paired_samples[key].append(text_sample[key])
                            negative_paired_samples[key].append(text_sample[key])
                    else:
                        if key not in negative_paired_samples.keys():
                            negative_paired_samples[key] = [spec_sample[key]]
                        else:
                            # negative_paired_samples[key] = negative_paired_samples[key].append(spec_sample[key])
                            negative_paired_samples[key].append(spec_sample[key])

            # Combine all single samples
            for key in negative_paired_samples.keys():
                if key == "data_labels":
                    continue
                negative_paired_samples[key] = torch.cat(
                    negative_paired_samples[key], dim=0
                )

            # Combine positive and negative samples
            for key in items:
                if key == "ntokens":
                    continue
                if key == "id":
                    continue
                paired_samples[key] = torch.cat(
                    [paired_samples[key], negative_paired_samples[key]], dim=0
                )

            # Produce speech and text matching labels
            stm_labels = torch.cat(
                [
                    torch.ones([num_pos_paired_samples]),
                    torch.zeros([num_total_paired_samples - num_pos_paired_samples]),
                ],
                dim=0,
            ).cuda()
            num_pair_samples = num_total_paired_samples

        all_samples = None
        data_borders = None
        if mode == 1:
            all_samples = {
                "text": text_samples,
                "spec": speech_samples,
            }
            data_borders = {
                "text": (0, num_text_samples),
                "spec": (num_text_samples, num_text_samples + num_spec_samples),
            }
            del paired_samples
        elif mode == 2:
            all_samples = {
                "text": text_samples,
                "spec": speech_samples,
                "pair": paired_samples,
            }
            data_borders = {
                "text": (0, num_text_samples),
                "spec": (num_text_samples, num_text_samples + num_spec_samples),
                "pair": (
                    num_text_samples + num_spec_samples,
                    num_text_samples + num_spec_samples + num_pair_samples,
                ),
            }
        elif mode == 3:
            all_samples = {
                "pair": paired_samples,
            }
            data_borders = {
                "pair": (0, num_pair_samples),
            }
            del speech_samples
            del text_samples
        elif mode == 4:
            all_samples = {
                "spec": speech_samples,
            }
            data_borders = {
                "spec": (0, num_spec_samples),
            }
        elif mode == 5:
            all_samples = {
                "text": text_samples,
            }
            data_borders = {
                "text": (0, num_text_samples),
            }
        else:
            all_samples = None
            data_borders = None

        # Release some space for future forward
        del data_labels
        del onehot_text_indices
        del onehot_speech_indices
        del onehot_paired_indices
        del text_indices
        del speech_indices
        del paired_indices

        # Forward speech encoder part
        spec_enc_data_borders = None
        spec_enc_samples = None
        if mode == 1 or mode == 4:  # Single modal training
            spec_enc_data_borders = {"spec": (0, num_spec_samples)}
            spec_enc_samples = all_samples["spec"]
        elif mode == 2:  # Paired and unpaired training
            spec_enc_data_borders = {
                "spec": (0, num_spec_samples),
                "pair": (num_spec_samples, num_spec_samples + num_pair_samples),
            }
            spec_enc_samples = self.combine_samples(
                items, ["spec", "pair"], all_samples
            )
        elif mode == 3:  # Only paired trainings
            spec_enc_data_borders = {"pair": (0, num_pair_samples)}
            spec_enc_samples = all_samples["pair"]
        else:
            spec_enc_data_borders = None
            spec_enc_samples = None

        speech_encoder_outputs = None
        if spec_enc_samples is not None:
            speech_encoder_outputs = self.forward_speech_encoder(
                spec_enc_samples,
                spec_enc_data_borders,
                mask_audio=mask_audio,
            )

        # Forward text encoder part
        text_enc_samples = None
        text_enc_data_borders = None
        if mode == 1 or mode == 5:  # Single modal training
            text_enc_data_borders = {"text": (0, num_text_samples)}
            text_enc_samples = all_samples["text"]
        elif mode == 2:  # Paired and unpaired training
            text_enc_data_borders = {
                "text": (0, num_text_samples),
                "pair": (num_text_samples, num_text_samples + num_pair_samples),
            }
            text_enc_samples = self.combine_samples(
                items, ["text", "pair"], all_samples
            )
        elif mode == 3:  # Only paired trainings
            text_enc_data_borders = {"pair": (0, num_pair_samples)}
            text_enc_samples = all_samples["pair"]
        else:
            text_enc_samples = None
            text_enc_data_borders = None

        text_encoder_outputs = None
        if text_enc_samples is not None:
            text_encoder_outputs = self.forward_text_encoder(
                text_enc_samples, text_enc_data_borders, mask_text
            )

        # Prepare inputs for cross modal encoder
        (
            cse_inputs,
            cse_padding_mask,
            text_max_len,
            spec_max_len,
        ) = self.prepare_cse_inputs(
            text_encoder_outputs,
            speech_encoder_outputs,
            text_enc_data_borders,
            spec_enc_data_borders,
            data_borders,
        )

        # Forward the cross-modal encoder part
        # Forward the pure text inputs
        joint_outputs = self.forward_cross_modal_encoder(
            cse_inputs, cse_padding_mask, text_max_len, spec_max_len, data_borders
        )

        cls_outputs = joint_outputs[:, 0, :].squeeze(1)  # B x C
        final_text_outputs = joint_outputs[:, 1 : (text_max_len + 1), :]  # B x T_t x C
        final_speech_outputs = joint_outputs[:, (text_max_len + 2) :, :]  # B x T_s x C

        result = {
            "text_outputs": text_encoder_outputs["text_outputs"]
            if text_encoder_outputs is not None
            else None,
            "text_enc_padding_mask": text_encoder_outputs["text_padding_mask"]
            if text_encoder_outputs is not None
            else None,
            "speech_outputs": speech_encoder_outputs["speech_outputs"]
            if speech_encoder_outputs is not None
            else None,
            "spec_enc_padding_mask": speech_encoder_outputs["speech_padding_mask"]
            if speech_encoder_outputs is not None
            else None,
            "final_text_outputs": final_text_outputs,
            "final_speech_outputs": final_speech_outputs,
            "text_max_len": text_max_len,
            "spec_max_len": spec_max_len,
            "cse_padding_mask": cse_padding_mask,
            "joint_outputs": joint_outputs,
            "data_borders": data_borders,
        }

        if "pair" in data_borders.keys():
            # Speech-text matching (STM)
            cls_outputs = cls_outputs[
                data_borders["pair"][0] : data_borders["pair"][1]
            ]  # B x C
            stm_logits = self.cls_proj(cls_outputs).squeeze(-1)  # B x C --> B x 1 --> B
            result["stm_logits"] = stm_logits
            result["stm_labels"] = stm_labels

            # Translation Language Modeling (TLM) for text part
            paired_text_mask_indices = text_encoder_outputs[
                "paired_text_mask_indices"
            ]  # B x T
            pos_paired_text_mask_indices = paired_text_mask_indices[
                :num_pos_paired_samples, :
            ]
            paired_text_tlm_outputs = final_text_outputs[
                data_borders["pair"][0] : data_borders["pair"][1]
            ]  # B x T x C
            paired_text_tlm_outputs = paired_text_tlm_outputs[
                :num_pos_paired_samples, :, :
            ][
                pos_paired_text_mask_indices
            ]  # B x T x C
            paired_text_tlm_logits = self.text_proj(paired_text_tlm_outputs)
            paired_text_tlm_targets = text_encoder_outputs["tlm_targets"]
            paired_text = text_encoder_outputs["paired_text"]
            result["paired_text_tlm_logits"] = paired_text_tlm_logits
            result["paired_text_tlm_targets"] = paired_text[:num_pos_paired_samples, :][
                pos_paired_text_mask_indices
            ]

            # Translation Language Modeling (TLM) for speech part
            paired_spec_mask_indices = speech_encoder_outputs[
                "paired_spec_mask_indices"
            ]
            paired_spec_tlm_outputs = final_speech_outputs[
                data_borders["pair"][0] : data_borders["pair"][1]
            ]  # B x T x C
            paired_spec_tlm_outputs = paired_spec_tlm_outputs[
                :num_pos_paired_samples, :, :
            ]  # B/2 x T x C
            paired_spec_tlm_outputs = paired_spec_tlm_outputs[
                paired_spec_mask_indices[:num_pos_paired_samples, :]
            ]
            paired_spec_tlm_logits = self.spec_proj(
                paired_spec_tlm_outputs
            )  # B x T x V

            pair_quantized_target_ids = speech_encoder_outputs[
                "pair_quantized_target_ids"
            ]
            if self.latent_groups == 1:
                paired_spec_tlm_targets = pair_quantized_target_ids.squeeze(
                    -1
                )  # B x T_mask
                paired_spec_tlm_targets = paired_spec_tlm_targets[
                    :num_pos_paired_samples
                ]
            elif self.latent_groups == 2:
                one_hot_raw_ids = (
                    torch.nn.functional.one_hot(
                        pair_quantized_target_ids[:, :, 0], self.latent_vars
                    )
                    .unsqueeze(2)
                    .repeat(1, 1, self.latent_vars, 1)
                )
                one_hot_col_ids = (
                    torch.nn.functional.one_hot(
                        pair_quantized_target_ids[:, :, -1], self.latent_vars
                    )
                    .unsqueeze(2)
                    .repeat(1, 1, self.latent_vars, 1)
                )
                ind = (
                    one_hot_raw_ids.transpose(-1, -2) * one_hot_col_ids
                )  # B x T x V x V
                targets_id_pool = (
                    torch.tensor(list(range(0, self.quantized_vocab_size)))
                    .view([self.latent_vars] * self.latent_groups)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )  # 1 x 1 x V x V
                paired_spec_tlm_targets = (
                    (ind * targets_id_pool).sum(-1).sum(-1)
                )  # B x T
                paired_spec_tlm_targets = paired_spec_tlm_targets[
                    :num_pos_paired_samples
                ]
            else:
                raise NotImplementedError("This is not supported until now.")

            result["paired_spec_tlm_logits"] = paired_spec_tlm_logits
            result["paired_spec_tlm_targets"] = paired_spec_tlm_targets
            result["num_pair_samples"] = num_pair_samples
            result["num_pos_paired_samples"] = num_pos_paired_samples

        if "text" in data_borders.keys():
            # Masked Language modeling (MLM)
            unpaired_text_mask_indices = text_encoder_outputs[
                "unpaired_text_mask_indices"
            ]
            text_mlm_outputs = final_text_outputs[
                data_borders["text"][0] : data_borders["text"][1]
            ]  # B x T x C
            text_mlm_outputs = text_mlm_outputs[unpaired_text_mask_indices]
            text_mlm_logits = self.text_proj(text_mlm_outputs)  # B x T x V
            text_mlm_targets = text_encoder_outputs["mlm_targets"]

            result["text_mlm_logits"] = text_mlm_logits
            result["text_mlm_targets"] = text_mlm_targets
            result["num_text_samples"] = num_text_samples

        if "spec" in data_borders.keys():
            # W2v-BERT speech masked language modeling (MLM)
            unpaired_spec_mask_indices = speech_encoder_outputs[
                "unpaired_spec_mask_indices"
            ]
            spec_mlm_outputs = final_speech_outputs[
                data_borders["spec"][0] : data_borders["spec"][1]
            ]  # B x T x C

            # Obtain w2v-bert speech mlm logits
            spec_mlm_outputs = spec_mlm_outputs[
                unpaired_spec_mask_indices
            ]  # (B x T_mask) x V
            spec_mlm_logits = self.spec_proj(spec_mlm_outputs)  # (B x T_mask) x V
            result["spec_mlm_logits"] = spec_mlm_logits  # (B x T_mask) x V

            # Obtain w2v-bert speech mlm targets
            spec_quantized_target_ids = speech_encoder_outputs[
                "spec_quantized_target_ids"
            ]  # B x T_mask x G
            if self.latent_groups == 1:
                result["spec_mlm_targets"] = spec_quantized_target_ids.squeeze(
                    -1
                )  # B x T_mask
            elif self.latent_groups == 2:
                one_hot_raw_ids = (
                    torch.nn.functional.one_hot(
                        spec_quantized_target_ids[:, :, 0], self.latent_vars
                    )
                    .unsqueeze(2)
                    .repeat(1, 1, self.latent_vars, 1)
                )
                one_hot_col_ids = (
                    torch.nn.functional.one_hot(
                        spec_quantized_target_ids[:, :, -1], self.latent_vars
                    )
                    .unsqueeze(2)
                    .repeat(1, 1, self.latent_vars, 1)
                )
                ind = (
                    one_hot_raw_ids.transpose(-1, -2) * one_hot_col_ids
                )  # B x T_mask x V x V
                targets_id_pool = (
                    torch.tensor(list(range(0, self.quantized_vocab_size)))
                    .view([self.latent_vars] * self.latent_groups)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .cuda()
                )  # 1 x 1 x V x V
                spec_mlm_targets = (ind * targets_id_pool).sum(-1).sum(-1)  # B x T_mask
                result["spec_mlm_targets"] = spec_mlm_targets  # B x T_mask
            else:
                raise NotImplementedError("This is not supported until now.")

            # Contrastive Loss
            contrastive_spec_logits = speech_encoder_outputs["contrastive_spec_logits"]
            result["contrastive_spec_logits"] = contrastive_spec_logits

            # Diversity Loss and L1 Loss for feature encoder outputs
            result["features_pen"] = speech_encoder_outputs["features_pen"]
            result["prob_perplexity"] = speech_encoder_outputs["prob_perplexity"]
            result["code_perplexity"] = speech_encoder_outputs["code_perplexity"]
            result["num_vars"] = speech_encoder_outputs["num_vars"]
            result["temp"] = speech_encoder_outputs["temp"]

            result["num_spec_samples"] = num_spec_samples

        ## Forward ASR decoder
        # cif_inputs = {
        #     "encoder_raw_out": speech_outputs,
        #     "encoder_padding_mask": padding_mask,
        # }
        # target_lengths = (text_padding_mask != 0).int().sum(-1)
        # cif_outputs = self.cif(encoder_outputs=cif_inputs, target_lengths=target_lengths)
        # decoder_outputs = self.nar_asr_decoder(prev_output_tokens=text, cif_out=cif_outputs)

        return result

    def prepare_cse_inputs(
        self,
        text_encoder_outputs=None,
        speech_encoder_outputs=None,
        text_enc_data_borders=None,
        spec_enc_data_borders=None,
        data_borders=None,
    ):

        all_keys = data_borders.keys()

        text_max_len = 0
        if text_encoder_outputs is not None:
            text_outputs = text_encoder_outputs["text_outputs"]
            text_padding_mask = text_encoder_outputs["text_padding_mask"]
            text_max_len = text_padding_mask.size(-1)

        spec_max_len = 0
        if speech_encoder_outputs is not None:
            speech_outputs = speech_encoder_outputs["speech_outputs"]
            spec_padding_mask = speech_encoder_outputs["speech_padding_mask"]
            spec_max_len = spec_padding_mask.size(-1)

        text_part_list = []
        text_padding_mask_list = []
        spec_part_list = []
        spec_padding_mask_list = []

        # Prepare the text part of cse inputs
        if "text" in all_keys and text_encoder_outputs is not None:
            text_text_outputs = text_outputs[
                text_enc_data_borders["text"][0] : text_enc_data_borders["text"][1]
            ]
            text_spec_outputs = torch.randn(
                [text_text_outputs.size(0), spec_max_len, text_outputs.size(-1)]
            ).cuda()
            text_text_padding_mask = text_padding_mask[
                text_enc_data_borders["text"][0] : text_enc_data_borders["text"][1]
            ]
            text_spec_padding_mask = (
                torch.ones([text_text_outputs.size(0), spec_max_len]).cuda().bool()
            )

            text_part_list.append(text_text_outputs)
            text_padding_mask_list.append(text_text_padding_mask)
            spec_part_list.append(text_spec_outputs)
            spec_padding_mask_list.append(text_spec_padding_mask)

        # Prepare the spec part of cse inputs
        if "spec" in all_keys and speech_encoder_outputs is not None:
            spec_spec_outputs = speech_outputs[
                spec_enc_data_borders["spec"][0] : spec_enc_data_borders["spec"][1]
            ]
            spec_text_outputs = torch.randn(
                [spec_spec_outputs.size(0), text_max_len, speech_outputs.size(-1)]
            ).cuda()
            spec_spec_padding_mask = spec_padding_mask[
                spec_enc_data_borders["spec"][0] : spec_enc_data_borders["spec"][1]
            ]
            spec_text_padding_mask = (
                torch.ones([spec_spec_outputs.size(0), text_max_len]).cuda().bool()
            )

            text_part_list.append(spec_text_outputs)
            text_padding_mask_list.append(spec_text_padding_mask)
            spec_part_list.append(spec_spec_outputs)
            spec_padding_mask_list.append(spec_spec_padding_mask)

        # Prepare the pair part of cse inputs
        if (
            "pair" in all_keys
            and speech_encoder_outputs is not None
            and text_encoder_outputs is not None
        ):
            paired_text_outputs = text_outputs[
                text_enc_data_borders["pair"][0] : text_enc_data_borders["pair"][1]
            ]
            paired_spec_outputs = speech_outputs[
                spec_enc_data_borders["pair"][0] : spec_enc_data_borders["pair"][1]
            ]
            paired_text_padding_mask = text_padding_mask[
                text_enc_data_borders["pair"][0] : text_enc_data_borders["pair"][1]
            ]
            paired_spec_padding_mask = spec_padding_mask[
                spec_enc_data_borders["pair"][0] : spec_enc_data_borders["pair"][1]
            ]

            text_part_list.append(paired_text_outputs)
            text_padding_mask_list.append(paired_text_padding_mask)
            spec_part_list.append(paired_spec_outputs)
            spec_padding_mask_list.append(paired_spec_padding_mask)

        text_inputs = torch.cat(text_part_list, dim=0)
        modified_text_padding_mask = torch.cat(text_padding_mask_list, dim=0)
        spec_inputs = torch.cat(spec_part_list, dim=0)
        modified_spec_padding_mask = torch.cat(spec_padding_mask_list, dim=0)

        assert text_inputs.size(0) == spec_inputs.size(0)
        total_bsz = text_inputs.size(0)
        cls_padding_mask = torch.zeros([total_bsz, 1]).cuda()
        sep_padding_mask = torch.zeros([total_bsz, 1]).cuda()

        joint_inputs = torch.cat(
            [
                self.cls_emb.unsqueeze(0).repeat([total_bsz, 1]).unsqueeze(1),
                text_inputs,
                self.sep_emb.unsqueeze(0).repeat([total_bsz, 1]).unsqueeze(1),
                spec_inputs,
            ],
            dim=1,
        )
        joint_padding_mask = torch.cat(
            [
                cls_padding_mask,
                modified_text_padding_mask,
                sep_padding_mask,
                modified_spec_padding_mask,
            ],
            dim=1,
        )

        return joint_inputs, joint_padding_mask, text_max_len, spec_max_len

    def forward_cross_modal_encoder(
        self, inputs, input_padding_mask, text_max_len, spec_max_len, data_borders
    ):

        inputs = inputs.half()

        if self.disable_ce_encoder or self.cross_modal_encoder_layers == 0:
            return inputs

        if not self.ce_encoder_layer_norm_first:
            inputs = self.ce_encoder_layer_norm(inputs)
        inputs = self.dropout_ce_inputs(inputs)

        if "text" in data_borders and (
            "pair" in data_borders or "spec" in data_borders
        ):
            ## 1.Forward pure text inputs
            pure_textual_inputs = inputs[
                data_borders["text"][0] : data_borders["text"][1]
            ]  # B x T x C
            pure_textual_inputs = pure_textual_inputs[:, : (text_max_len + 2), :]
            # (text_max_len + 2) cause there are [CLS] and [SEP]
            pure_textual_input_padding_mask = input_padding_mask[
                data_borders["text"][0] : data_borders["text"][1]
            ]
            pure_textual_input_padding_mask = pure_textual_input_padding_mask[
                :, : (text_max_len + 2)
            ]
            pure_textual_inputs = pure_textual_inputs.transpose(
                0, 1
            ).half()  # T x B x C
            for layer in self.text_encoder_stacks:
                pure_textual_inputs, _ = layer(
                    pure_textual_inputs,
                    self_attn_padding_mask=pure_textual_input_padding_mask,
                    need_weights=False,
                )
            pure_textual_inputs = pure_textual_inputs.transpose(0, 1)  # B x T x C

            ## 2.Forward other parts with only speech or paired data
            num_text_samples = pure_textual_inputs.size(0)
            other_inputs = inputs[num_text_samples:, :, :]
            other_input_padding_mask = input_padding_mask[num_text_samples:, :]
            other_inputs = other_inputs.transpose(0, 1).half()
            for layer in self.text_encoder_stacks:
                other_inputs, _ = layer(
                    other_inputs,
                    self_attn_padding_mask=other_input_padding_mask,
                    need_weights=False,
                )
            other_inputs = other_inputs.transpose(0, 1)  # B x T x C

            ## 3.Combine all of them
            pure_textual_inputs = torch.cat(
                [
                    pure_textual_inputs,  # num_text_samples x (text_max_len + 2) x C
                    torch.zeros(
                        num_text_samples, spec_max_len, pure_textual_inputs.size(-1)
                    )
                    .half()
                    .cuda(),
                ],
                dim=1,
            )
            outputs = torch.cat([pure_textual_inputs, other_inputs], dim=0)
        else:
            # Forward cross-modal encoder
            inputs = inputs.transpose(0, 1).half()
            for layer in self.text_encoder_stacks:
                inputs, _ = layer(
                    inputs,
                    self_attn_padding_mask=input_padding_mask,
                    need_weights=False,
                )
            outputs = inputs.transpose(0, 1)  # B x T x C

        if self.ce_encoder_layer_norm_first:
            outputs = self.ce_encoder_layer_norm(outputs)

        return outputs

    def forward_text_embedding_module(self, text):
        x = self.text_embedding_layer(text)
        x = self.embed_scale * x
        if self.text_embed_positions is not None:
            x = x + self.text_embed_positions(text)
        if not self.text_encoder_layer_norm_first:
            x = self.text_encoder_layer_norm(x)
        x = self.dropout_text_inputs(x)

        return x

    def forward_text_encoder(self, text_enc_samples, text_enc_data_borders, mask_text):
        text = text_enc_samples["text"]
        text_padding_mask = (text == self.default_dict.pad()).bool()

        # Forward text embedding layers
        text_embeds = self.forward_text_embedding_module(text)

        # Forward masking
        unpaired_text_mask_indices = None
        paired_text_mask_indices = None
        if mask_text:
            masked_text_embeds_list = []
            text_mask_indices_list = []

            if "text" in text_enc_data_borders.keys():
                # For unpaired text
                unpaired_text_embeds = text_embeds[
                    text_enc_data_borders["text"][0] : text_enc_data_borders["text"][1]
                ]
                unpaired_text_padding_mask = text_padding_mask[
                    text_enc_data_borders["text"][0] : text_enc_data_borders["text"][1]
                ]
                (
                    unpaired_masked_text_embeds,
                    unpaired_text_mask_indices,
                ) = self.apply_text_temporal_mask(
                    unpaired_text_embeds,
                    unpaired_text_padding_mask,
                    text_mask_prob=self.mlm_text_mask_prob,
                    text_mask_length=self.mlm_text_mask_span_length,
                )
                masked_text_embeds_list.append(unpaired_masked_text_embeds)
                text_mask_indices_list.append(unpaired_text_mask_indices)

                if unpaired_text_mask_indices.numel() == 0:
                    print(unpaired_text_mask_indices)
                    print(unpaired_text_mask_indices.size())
                    raise ValueError("unpaired_text_mask_indices has no elements.")

            if "pair" in text_enc_data_borders.keys():
                # For paired text
                paired_text_embeds = text_embeds[
                    text_enc_data_borders["pair"][0] : text_enc_data_borders["pair"][1]
                ]
                paired_text_padding_mask = text_padding_mask[
                    text_enc_data_borders["pair"][0] : text_enc_data_borders["pair"][1]
                ]
                (
                    paired_masked_text_embeds,
                    paired_text_mask_indices,
                ) = self.apply_text_temporal_mask(
                    paired_text_embeds,
                    paired_text_padding_mask,
                    text_mask_prob=self.tlm_text_mask_prob,
                    text_mask_length=self.tlm_text_mask_span_length,
                )
                masked_text_embeds_list.append(paired_masked_text_embeds)
                text_mask_indices_list.append(paired_text_mask_indices)

            # Combine each outputs
            masked_text_embeds = torch.cat(masked_text_embeds_list, dim=0)
            masked_text_indices = torch.cat(text_mask_indices_list, dim=0)
        else:
            masked_text_embeds = text_embeds
            masked_text_indices = None

        # Forward transformer layers
        x = masked_text_embeds.transpose(0, 1)  # T x B x C
        for layer in self.text_encoder_stacks:
            x, _ = layer(
                x, self_attn_padding_mask=text_padding_mask, need_weights=False
            )
        x = x.transpose(0, 1)  # B x T x C
        if self.text_encoder_layer_norm_first:
            x = self.text_encoder_layer_norm(x)  # B x T x C

        result = {
            "text_outputs": x,
            "text_padding_mask": text_padding_mask,
            "text_mask_indices": masked_text_indices,
        }

        if "text" in text_enc_data_borders.keys():
            if unpaired_text_mask_indices is not None:
                unpaired_text = text[
                    text_enc_data_borders["text"][0] : text_enc_data_borders["text"][1]
                ]
                result["unpaired_text"] = unpaired_text
                result["mlm_targets"] = unpaired_text[unpaired_text_mask_indices]
                result["unpaired_text_mask_indices"] = unpaired_text_mask_indices

        if "pair" in text_enc_data_borders.keys():
            if paired_text_mask_indices is not None:
                paired_text = text[
                    text_enc_data_borders["pair"][0] : text_enc_data_borders["pair"][1]
                ]
                result["paired_text"] = paired_text
                result["tlm_targets"] = paired_text[paired_text_mask_indices]
                result["paired_text_mask_indices"] = paired_text_mask_indices

        return result

    def forward_speech_encoder(
        self, spec_enc_samples, spec_enc_data_borders, mask_audio=False
    ):

        # Get all speech samples
        source = spec_enc_samples["source"]
        padding_mask = spec_enc_samples["padding_mask"]

        # Forward conv feat extractor
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = (
            features[
                spec_enc_data_borders["spec"][0] : spec_enc_data_borders["spec"][1]
            ]
            .float()
            .pow(2)
            .mean()
            if "spec" in spec_enc_data_borders.keys()
            else None
        )

        features = features.transpose(1, 2)
        features = self.conv_layer_norm(features)
        unmasked_features = features.clone()
        # if padding_mask is not None and padding_mask.any():
        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        # Forward masking operation
        y_spec = None
        unpaired_spec_mask_indices = None
        unpaired_spec_unmasked_features = None
        y_pair = None
        paired_mask_indices = None
        pair_unmasked_features = None

        if mask_audio:
            masked_spec_feats_list = []
            spec_mask_indices_list = []

            if "spec" in spec_enc_data_borders.keys():
                # For unpaired speech
                unpaired_spec_feats = features[
                    spec_enc_data_borders["spec"][0] : spec_enc_data_borders["spec"][1]
                ]
                unpaired_spec_padding_mask = padding_mask[
                    spec_enc_data_borders["spec"][0] : spec_enc_data_borders["spec"][1]
                ]
                (
                    unpaired_masked_spec_feats,
                    unpaired_spec_mask_indices,
                ) = self.apply_spec_channel_temporal_mask(
                    unpaired_spec_feats, unpaired_spec_padding_mask
                )
                unpaired_spec_unmasked_features = unmasked_features[
                    spec_enc_data_borders["spec"][0] : spec_enc_data_borders["spec"][1]
                ]

                masked_spec_feats_list.append(unpaired_masked_spec_feats)
                spec_mask_indices_list.append(unpaired_spec_mask_indices)

                if (
                    not is_xla_tensor(unpaired_masked_spec_feats)
                    and unpaired_spec_mask_indices is not None
                ):
                    # tpu-comment: reducing the size in a dynamic way causes
                    # too many recompilations on xla.
                    y_spec = unpaired_spec_unmasked_features[
                        unpaired_spec_mask_indices
                    ].view(
                        unpaired_spec_unmasked_features.size(0),
                        -1,
                        unpaired_spec_unmasked_features.size(-1),
                    )  # y is the real values of the masked locations
                else:
                    y_spec = unpaired_spec_unmasked_features

            if "pair" in spec_enc_data_borders.keys():
                # Paired data
                paired_spec_feats = features[
                    spec_enc_data_borders["pair"][0] : spec_enc_data_borders["pair"][1]
                ]
                paired_spec_padding_mask = padding_mask[
                    spec_enc_data_borders["pair"][0] : spec_enc_data_borders["pair"][1]
                ]
                (
                    paired_masked_spec_feats,
                    paired_mask_indices,
                ) = self.apply_spec_temporal_mask(
                    paired_spec_feats, paired_spec_padding_mask
                )
                pair_unmasked_features = unmasked_features[
                    spec_enc_data_borders["pair"][0] : spec_enc_data_borders["pair"][1]
                ]

                masked_spec_feats_list.append(paired_masked_spec_feats)
                spec_mask_indices_list.append(paired_mask_indices)

                if (
                    not is_xla_tensor(paired_masked_spec_feats)
                    and paired_mask_indices is not None
                ):
                    # tpu-comment: reducing the size in a dynamic way causes
                    # too many recompilations on xla.
                    y_pair = pair_unmasked_features[paired_mask_indices].view(
                        pair_unmasked_features.size(0),
                        -1,
                        pair_unmasked_features.size(-1),
                    )  # y is the real values of the masked locations
                else:
                    y_pair = pair_unmasked_features

            masked_spec_feats = torch.cat(masked_spec_feats_list, dim=0)
            spec_mask_indices = torch.cat(spec_mask_indices_list, dim=0)
        else:
            # All feats after masking
            masked_spec_feats = features
            spec_mask_indices = None

            # For contrastive learning
            if "spec" in spec_enc_data_borders.keys():
                y_spec = unmasked_features[
                    spec_enc_data_borders["spec"][0] : spec_enc_data_borders["spec"][1]
                ]
                unpaired_spec_unmasked_features = unmasked_features[
                    spec_enc_data_borders["spec"][0] : spec_enc_data_borders["spec"][1]
                ]
            if "pair" in spec_enc_data_borders.keys():
                y_pair = unmasked_features[
                    spec_enc_data_borders["pair"][0] : spec_enc_data_borders["pair"][1]
                ]
                pair_unmasked_features = unmasked_features[
                    spec_enc_data_borders["pair"][0] : spec_enc_data_borders["pair"][1]
                ]

        # Forward contrastive module of speech encoder
        x, layer_results = self.speech_encoder(
            masked_spec_feats, padding_mask=padding_mask
        )

        # Forward quantizer module
        def forward_quantizer(x, y, unmasked_features, mask_indices, return_all=False):
            # Forward quantizer part with convolutional layers outputs
            if self.quantizer:
                q = self.quantizer(y, produce_targets=True)
                y = q["x"]  # B x T x C
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                quantized_target_ids = q["targets"]  # B x T x G

                y = self.project_q(y)

                # Obtain negtive samples for contrastive loss
                if self.negatives_from_everywhere:
                    neg_cands = self.quantizer(
                        unmasked_features, produce_targets=False
                    )["x"]
                    negs, _ = self.sample_negatives(
                        neg_cands, y.size(1), padding_count=None
                    )
                    negs = self.project_q(negs)
                else:
                    negs, _ = self.sample_negatives(
                        y,
                        y.size(1),
                        padding_count=None,
                    )  # N_negs x B x T x C

                # Obtain some negtive samples from codebooks
                if self.codebook_negatives > 0:
                    cb_negs = self.quantizer.sample_from_codebook(
                        y.size(0) * y.size(1), self.codebook_negatives
                    )
                    cb_negs = cb_negs.view(
                        self.codebook_negatives, y.size(0), y.size(1), -1
                    )  # order doesnt matter
                    cb_negs = self.project_q(cb_negs)
                    negs = torch.cat([negs, cb_negs], dim=0)
            else:
                y = self.project_q(y)
                num_vars = None
                code_ppl = None
                prob_ppl = None
                curr_temp = None
                quantized_target_ids = None  # B x T x G

                if self.negatives_from_everywhere:
                    negs, _ = self.sample_negatives(
                        unmasked_features, y.size(1), padding_count=None
                    )
                    negs = self.project_q(negs)
                else:
                    negs, _ = self.sample_negatives(y, y.size(1), padding_count=None)

            # Take out the masked locations final outputs
            if not is_xla_tensor(x):
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                x = x[mask_indices].view(x.size(0), -1, x.size(-1))  # B x T_mask x C

            # Unavailable for now
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # y shape = B x T_mask x C
            # negs shape = n_negs x B x T x C

            x = self.final_proj(
                x
            )  # Project x to the dimension of latent variables, B x T_mask x C_final
            x = self.compute_preds(x, y, negs)

            if return_all:
                return x, quantized_target_ids, num_vars, code_ppl, prob_ppl, curr_temp
            else:
                return x, quantized_target_ids

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        logits_spec = None
        spec_quantized_target_ids = None
        if "spec" in spec_enc_data_borders.keys():
            (
                logits_spec,
                spec_quantized_target_ids,
                num_vars,
                code_ppl,
                prob_ppl,
                curr_temp,
            ) = forward_quantizer(
                x=x[
                    spec_enc_data_borders["spec"][0] : spec_enc_data_borders["spec"][1]
                ],
                y=y_spec,
                unmasked_features=unpaired_spec_unmasked_features,
                mask_indices=unpaired_spec_mask_indices,
                return_all=True,
            )

        logits_pair = None
        pair_quantized_target_ids = None
        if "pair" in spec_enc_data_borders.keys():
            logits_pair, pair_quantized_target_ids = forward_quantizer(
                x=x[
                    spec_enc_data_borders["pair"][0] : spec_enc_data_borders["pair"][1]
                ],
                y=y_pair,
                unmasked_features=pair_unmasked_features,
                mask_indices=paired_mask_indices,
                return_all=False,
            )

        # General outputs
        result = {
            "speech_outputs": x,
            "speech_padding_mask": padding_mask,
            "spec_mask_indices": spec_mask_indices,
            "features_pen": features_pen,
        }

        if "spec" in spec_enc_data_borders.keys():
            if unpaired_spec_mask_indices is not None:
                result["unpaired_spec_mask_indices"] = unpaired_spec_mask_indices
            if spec_quantized_target_ids is not None:
                result["spec_quantized_target_ids"] = spec_quantized_target_ids
            if logits_spec is not None:
                result["contrastive_spec_logits"] = logits_spec
                # print("logits_spec: ")
                # print(logits_spec.size())
            if prob_ppl is not None:
                result["prob_perplexity"] = prob_ppl
                result["code_perplexity"] = code_ppl
                result["num_vars"] = num_vars
                result["temp"] = curr_temp

        if "pair" in spec_enc_data_borders.keys():
            if paired_mask_indices is not None:
                result["paired_spec_mask_indices"] = paired_mask_indices
            if pair_quantized_target_ids is not None:
                result["pair_quantized_target_ids"] = pair_quantized_target_ids
            if logits_pair is not None:
                result["tlm_spec_logits"] = logits_pair

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, sample, padding_mask, mask=False, layer=None):
        res = self.forward(
            sample, mask_audio=mask, mask_text=mask, features_only=True, mode=1
        )
        return res

    def get_infonce_logits(self, net_output):
        logits = net_output["contrastive_spec_logits"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_infonce_targets(self, net_output):
        x = net_output["contrastive_spec_logits"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    # noinspection PyStatementEffect
    def get_extra_losses(self, net_output):
        pen = {}

        if "prob_perplexity" in net_output.keys():
            pen["prob_perplexity"] = (
                net_output["num_vars"] - net_output["prob_perplexity"]
            ) / net_output["num_vars"]
            # (net_output["num_vars"] - net_output["prob_perplexity"]) / net_output["num_vars"]

        if "features_pen" in net_output.keys():
            pen["features_pen"] = net_output["features_pen"]

        return pen

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

    @staticmethod
    def get_probs_from_logits(logits, log_probs=False):
        """Get normalized probabilities (or log probs) from logits."""
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)


class CifMiddleware(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Get configurations related to continuous integrate-and-fire
        self.cif_threshold = cfg.cif_threshold
        self.cif_output_dim = cfg.cif_embedding_dim
        self.encoder_embed_dim = cfg.cif_input_embed_dim
        self.produce_weight_type = cfg.produce_weight_type
        self.apply_scaling = cfg.apply_scaling
        self.apply_tail_handling = cfg.apply_tail_handling
        self.tail_handling_firing_threshold = cfg.tail_handling_firing_threshold

        # Build weight projection layer to compute weight from encoder outputs
        if self.produce_weight_type == "dense":
            self.dense_proj = Linear(
                self.encoder_embed_dim, cfg.dense_cif_units_num
            ).cuda()
            self.weight_proj = Linear(cfg.dense_cif_units_num, 1).cuda()
        elif self.produce_weight_type == "conv":
            self.cif_conv_layer_num = cfg.conv_cif_layer_num
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim,
                cfg.conv_cif_output_channels_num,
                cfg.conv_cif_width,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ).cuda()
            self.conv_dropout = torch.nn.Dropout(p=cfg.conv_cif_dropout).cuda()
            self.weight_proj = Linear(cfg.conv_cif_output_channels_num, 1).cuda()
        else:
            self.weight_proj = Linear(self.encoder_embed_dim, 1).cuda()

        # Build the final projection layer for cif outputs
        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = Linear(
                self.encoder_embed_dim, self.cif_output_dim, bias=False
            ).cuda()

    def forward(self, encoder_outputs, target_lengths, **kwargs):
        # Collect inputs
        encoder_raw_outputs = encoder_outputs["encoder_raw_out"]  # B x T x C
        encoder_padding_mask = encoder_outputs["encoder_padding_mask"]  # B x T
        # Convert boolean value to integer type
        # encoder_raw_outputs should have shape [batch_size, Length, encoder_embed_dim]
        # targets_length should have shape [batch_size]
        # encoder_padding_mask should have shape [batch_size, length]
        not_padding_mask = ~encoder_padding_mask  # non_padding_mask has shape B x T

        # Produce weights
        if self.produce_weight_type == "dense":
            x = self.dense_proj(encoder_raw_outputs)
            x = torch.relu(x)
            x = self.weight_proj(x)
        elif self.produce_weight_type == "conv":
            x = encoder_raw_outputs.permute(
                0, 2, 1
            )  # Adjust the shape of convolution layer input [B, C_in, T]
            x = self.conv(x)  # conv_out has shape [B, C_out, T]
            x = x.permute(0, 2, 1)
            x = self.conv_dropout(x)  # Adjust conv output to shape [B, T, C_cif]
            x = self.weight_proj(x)
        else:
            x = self.weight_proj(encoder_raw_outputs)

        # Calculate weights
        weight = torch.sigmoid(x)  # weight has shape B x T x 1
        weight = weight.squeeze(-1) * not_padding_mask.int()  # weight has shape B x T
        org_weight = weight

        # Sum weights
        if self.training and self.apply_scaling and target_lengths is not None:
            # Conduct scaling when training
            # (target_lengths + 1 because this target_lengths does not take <eos> into consideration)
            weight_sum = weight.sum(-1)  # weight_sum has shape [batch_size]
            normalize_scalar = torch.unsqueeze(target_lengths / weight_sum, -1)
            weight = weight * normalize_scalar

        # Integrate and fire
        batch_size = encoder_raw_outputs.size(0)
        max_length = encoder_raw_outputs.size(1)
        encoder_embed_dim = encoder_raw_outputs.size(2)
        padding_start_id = not_padding_mask.sum(-1)  # shape B

        # Initialize
        accumulated_weights = torch.zeros(batch_size, 0).cuda()
        accumulated_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()
        fired_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()

        # Begin integrate and fire
        for i in range(max_length):
            # Get previous states from the recorded tensor
            prev_accumulated_weight = (
                torch.zeros([batch_size]).cuda()
                if i == 0
                else accumulated_weights[:, i - 1]
            )
            prev_accumulated_state = (
                torch.zeros([batch_size, encoder_embed_dim]).cuda()
                if i == 0
                else accumulated_states[:, i - 1, :]
            )

            # Decide whether positioning a boundary
            cur_is_fired = (
                (prev_accumulated_weight + weight[:, i]) >= self.cif_threshold
            ).unsqueeze(dim=-1)
            # cur_is_fired with shape [batch_size, 1]

            # Update the accumulated weights by considering whether positioning a boundary
            cur_weight = weight[:, i].unsqueeze(-1)
            # cur_weight has shape [batch_size, 1]
            prev_accumulated_weight = prev_accumulated_weight.unsqueeze(-1)
            # prev_accumulated_weight also has shape [batch_size ,1]
            remained_weight = (
                torch.ones_like(prev_accumulated_weight).cuda()
                - prev_accumulated_weight
            )
            # remained_weight with shape [batch_size ,1]

            # Obtain the accumulated weight of current step
            cur_accumulated_weight = torch.where(
                cur_is_fired,
                cur_weight - remained_weight,
                cur_weight + prev_accumulated_weight,
            )  # [batch_size, 1]

            # Obtain accumulated state of current step
            cur_accumulated_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                (cur_weight - remained_weight) * encoder_raw_outputs[:, i, :],
                prev_accumulated_state + cur_weight * encoder_raw_outputs[:, i, :],
            )  # [batch_size, encoder_embed_dim]

            # Obtain fired state of current step:
            # firing locations has meaningful representations, while non-firing locations is all-zero embeddings
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                prev_accumulated_state + remained_weight * encoder_raw_outputs[:, i, :],
                torch.zeros([batch_size, encoder_embed_dim]).cuda(),
            )  # shape = [batch_size, encoder_embed_dim]

            # Handling the speech tail by rounding up and down
            if (not self.training) and self.apply_tail_handling:
                # When encoder output position exceeds the max valid position,
                # if accumulated weights is greater than 0.6,
                # current state should be reserved, otherwise it is discarded.
                cur_fired_state = torch.where(
                    i
                    == padding_start_id.unsqueeze(dim=-1).repeat(
                        [1, encoder_embed_dim]
                    ),
                    # shape = [batch_size, encoder_embed_dim]
                    torch.where(
                        cur_accumulated_weight.repeat([1, encoder_embed_dim])
                        <= self.tail_handling_firing_threshold,
                        # shape = [batch_size, encoder_embed_dim]
                        torch.zeros([batch_size, encoder_embed_dim]).cuda(),
                        # less equal than 0.5, discarded.
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                        # bigger than 0.5, normalized and kept.
                    ),
                    cur_fired_state,
                )
                # shape = [batch_size, encoder_embed_dim]

            # For normal condition, including both training and evaluation
            # Mask padded locations with all-zero embeddings
            cur_fired_state = torch.where(
                torch.full([batch_size, encoder_embed_dim], i).cuda()
                > padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                torch.zeros([batch_size, encoder_embed_dim]).cuda(),
                cur_fired_state,
            )

            # Update accumulated arguments
            accumulated_weights = torch.cat(
                (accumulated_weights, cur_accumulated_weight), 1
            )  # shape = [batch_size, Len]
            accumulated_states = torch.cat(
                (accumulated_states, cur_accumulated_state.unsqueeze(1)), 1
            )  # shape = [B, L, D]
            fired_states = torch.cat(
                (fired_states, cur_fired_state.unsqueeze(1)), 1
            )  # shape = [B, L, D]

        # Extracts cif_outputs for each utterance
        fired_marks = (
            torch.abs(fired_states).sum(-1) != 0.0
        ).int()  # [batch_size, max_length]
        fired_utt_length = fired_marks.sum(-1)  # [batch_size]
        fired_max_length = (
            fired_utt_length.max().int()
        )  # The maximum of fired times in current batch
        cif_outputs = torch.zeros(
            [0, fired_max_length, encoder_embed_dim]
        ).cuda()  # Initialize cif outputs

        def dynamic_partition(
            data: torch.Tensor, partitions: torch.Tensor, num_partitions=None
        ):
            assert (
                len(partitions.shape) == 1
            ), "Only one dimensional partitions supported"
            assert (
                data.shape[0] == partitions.shape[0]
            ), "Partitions requires the same size as data"
            if num_partitions is None:
                num_partitions = max(torch.unique(partitions))
            return [data[partitions == i] for i in range(num_partitions)]

        for j in range(batch_size):
            # Get information of j-th sample
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]
            cur_utt_outputs = dynamic_partition(
                cur_utt_fired_state, cur_utt_fired_mark, 2
            )
            cur_utt_output = cur_utt_outputs[1]  # Get integrated representations
            cur_utt_length = cur_utt_output.size(0)  # The total number of firing
            pad_length = fired_max_length - cur_utt_length  # Calculate padding length
            cur_utt_output = torch.cat(
                (
                    cur_utt_output,
                    torch.full([pad_length, encoder_embed_dim], 0.0).cuda(),
                ),
                dim=0,
            )  # Pad current utterance cif outputs to fired_max_length
            cur_utt_output = cur_utt_output.unsqueeze(0)
            # Reshape to [1, fired_max_length, encoder_embed_dim]

            # Concatenate cur_utt_output and cif_outputs along batch axis
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).int()
        # cif_out_padding_mask shape = [batch_size, fired_max_length], where locations with value 0 is False.

        if self.training:
            # In training phase, use summation of original weights as quantity out for quantity loss
            quantity_out = org_weight.sum(-1)
        else:
            quantity_out = weight.sum(-1)

        if self.cif_output_dim != encoder_embed_dim:
            cif_outputs = self.cif_output_proj(cif_outputs)

        return {
            "cif_out": cif_outputs,  # shape = [batch_size, fired_max_length, encoder_embed_dim]
            "quantity_out": quantity_out,  # shape = [batch_size]
            "cif_out_padding_mask": cif_out_padding_mask,  # shape = [batch_size, fired_max_length]
        }


class NarProjAsrDecoder(nn.Module):
    def __init__(self, cfg, dictionary):
        super().__init__()

        # Load parameters and build model
        self.dictionary = dictionary
        self.pre_final_proj_dim = cfg.pre_final_proj_dim
        self.output_dim = len(self.dictionary)
        self.output_proj = Linear(self.pre_final_proj_dim, self.output_dim).cuda()

    def forward(self, prev_output_tokens=None, cif_out=None, **kwargs):
        x = cif_out["cif_out"]

        # Collect shape information
        batch_size, cif_len, cif_embed_dim = x.size()
        prev_output_tokens_len = prev_output_tokens.size(1)

        # Handle exception of No Elements in cif_outputs
        if cif_len == 0 and not self.training:
            cif_len = 1
            x = torch.zeros([batch_size, cif_len, cif_embed_dim]).cuda()

        # Regularize the length of input tokens and cif outputs
        min_len = min(prev_output_tokens_len, cif_len)
        x = x[:, :min_len, :]  # B x min_len x C

        # Forword decoder
        x = self.output_proj(x)

        return x, None


class NarTransformerAsrDecoder(NarProjAsrDecoder):
    def __init__(self, cfg, dictionary):
        super().__init__(cfg, dictionary)

        # Load decoder parameters
        self.decoder_layers = cfg.nar_decoder_layers
        self.decoder_embed_dim = cfg.nar_decoder_embed_dim
        self.decoder_ffn_embed_dim = cfg.nar_decoder_ffn_dim
        self.decoder_attention_heads = cfg.nar_decoder_num_heads
        self.decoder_normalize_before = cfg.layer_norm_first
        self.decoder_dropout = cfg.nar_decoder_dropout
        self.decoder_attention_dropout = cfg.nar_decoder_attention_dropout
        self.decoder_activation_dropout = cfg.nar_decoder_activation_dropout

        assert (
            self.decoder_embed_dim == self.pre_final_proj_dim
        ), "ensure that the dimension of decoder outputs is equal to pre_final_proj_dim"

        # Build decoder stacks
        self.decoder_stacks = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.decoder_embed_dim,
                    ffn_embedding_dim=self.decoder_ffn_embed_dim,
                    num_attention_heads=self.decoder_attention_heads,
                    dropout=self.decoder_dropout,
                    activation_dropout=self.decoder_activation_dropout,
                    attention_dropout=self.decoder_attention_dropout,
                    layer_norm_first=self.decoder_normalize_before,
                )
                for _ in range(cfg.decoder_layers)
            ]
        )

    def forward(self, prev_output_tokens=None, cif_out=None, **kwargs):
        x = cif_out["cif_out"]
        padding_mask = ~cif_out["cif_out_padding_mask"].bool()

        # Collect shape information
        _, cif_len, _ = x.size()
        prev_output_tokens_len = prev_output_tokens.size(1)

        # Handle exception of No Elements in cif_outputs
        if cif_len == 0 and not self.training:
            cif_len = 1
            x = torch.zeros([batch_size, cif_len, cif_embed_dim]).cuda()  # B x 1 x C
            padding_mask = torch.zeros([batch_size, cif_len]).cuda()  # B x 1

        # Regularize the length of input tokens and cif outputs, and padding_mask
        min_len = min(prev_output_tokens_len, cif_len)
        x = x[:, :min_len, :]  # B x min_len x C
        padding_mask = padding_mask[:, :min_len]  # B x min_len

        # Forward decoder
        x = x.transpose(0, 1)
        for layer in self.decoder_stacks:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
        x = x.transpose(0, 1)
        final_logits = self.output_proj(x)

        return final_logits, None


class TtsVocoder(nn.Module):
    def __init__(self):
        super().__init__()


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        layers = []
        for _ in range(args.encoder_layers):
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
            )
            if args.checkpoint_activations:
                layer = fsdp_wrap(layer)
                layer = checkpoint_wrapper(layer)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                if tgt_layer is not None:
                    layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        # For float16 training
        x = x.half()

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x = x.half()

            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
            )

            x = self.dropout1(x)
            x = residual + x

            # self.self_attn_layer_norm = self.self_attn_layer_norm.half()
            x = x.half()
            x = self.self_attn_layer_norm(x)
            x = x.half()

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
