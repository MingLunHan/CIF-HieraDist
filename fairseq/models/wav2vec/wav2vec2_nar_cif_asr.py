# @Time    : 2021/11/1
# @Author  : Minglun Han
# @File    : wav2vec2_nar_cif_asr.py

from argparse import Namespace
import logging
import sys
import contextlib
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Any, Optional

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import (
    MASKING_DISTRIBUTION_CHOICES,
    TransformerSentenceEncoderLayer,
)
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
)
from fairseq.models.wav2vec.wav2vec2_asr import (
    Wav2VecEncoder,
    Wav2Vec2AsrConfig,
    Embedding,
    Linear,
)


@dataclass
class Wav2Vec2NarCIFConfig(Wav2Vec2AsrConfig):
    # Decoder settings
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    autoregressive: bool = II("task.autoregressive")
    decoder_mode: str = field(
        default="proj",
        metadata={
            "help": "the mode of decoder, there are three options: ar_decoder, nar_decoder, proj"
        },
    )
    pre_final_proj_dim: int = field(default=768)

    # CIF settings
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder output embedding dimension"}
    )
    cif_embedding_dim: int = field(
        default=768, metadata={"help": "cif output embedding dimension"}
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
    cif_ctxt_layers: int = field(default=2)
    cif_ctxt_embed_dim: int = field(default=768)
    cif_ctxt_ffn_embed_dim: int = field(default=3072)
    cif_ctxt_attention_heads: int = field(default=8)
    cif_ctxt_dropout: float = field(default=0.1)
    cif_ctxt_activation_dropout: float = field(default=0.0)
    cif_ctxt_attention_dropout: float = field(default=0.1)
    cif_ctxt_normalize_before: bool = field(default=True)


class CifMiddleware(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Get configurations related to continuous integrate-and-fire
        self.cif_threshold = cfg.cif_threshold
        self.cif_output_dim = cfg.cif_embedding_dim
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.produce_weight_type = cfg.produce_weight_type
        self.apply_scaling = cfg.apply_scaling
        self.apply_tail_handling = cfg.apply_tail_handling
        self.tail_handling_firing_threshold = cfg.tail_handling_firing_threshold
        self.add_cif_ctxt_layers = cfg.add_cif_ctxt_layers

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

        # Build cif contextual layers
        if self.add_cif_ctxt_layers:
            self.cif_ctxt_embed_dim = cfg.cif_ctxt_embed_dim
            self.cif_ctxt_stacks = nn.ModuleList(
                [
                    TransformerSentenceEncoderLayer(
                        embedding_dim=cfg.cif_ctxt_embed_dim,
                        ffn_embedding_dim=cfg.cif_ctxt_ffn_embed_dim,
                        num_attention_heads=cfg.cif_ctxt_attention_heads,
                        dropout=cfg.cif_ctxt_dropout,
                        activation_dropout=cfg.cif_ctxt_activation_dropout,
                        attention_dropout=cfg.cif_ctxt_attention_dropout,
                        layer_norm_first=cfg.cif_ctxt_normalize_before,
                    )
                    for _ in range(cfg.cif_ctxt_layers)
                ]
            )

    def forward(self, encoder_outputs, target_lengths, **kwargs):
        # Collect inputs
        encoder_raw_outputs = encoder_outputs["encoder_raw_out"]  # B x T x C
        encoder_padding_mask = encoder_outputs["encoder_padding_mask"]  # B x T
        # Convert boolean value to integer type
        # encoder_raw_outputs should have shape B x T x C
        # targets_length should have shape B
        # encoder_padding_mask should have shape B x T

        # print(encoder_raw_outputs.size())
        # print(encoder_padding_mask.size())

        # Produce weights
        if self.produce_weight_type == "dense":
            proj_out = self.dense_proj(encoder_raw_outputs)
            act_proj_out = torch.relu(proj_out)
            sig_input = self.weight_proj(act_proj_out)
            weight = torch.sigmoid(sig_input)
            # weight has shape [batch_size, length, 1]
        elif self.produce_weight_type == "conv":
            conv_input = encoder_raw_outputs.permute(0, 2, 1)
            # Adjust the shape of convolution layer input [B, C_in, T]
            conv_out = self.conv(conv_input)
            # conv_out has shape [B, C_out, T]
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.conv_dropout(proj_input)
            # Adjust conv output to shape [B, T, C_cif]
            sig_input = self.weight_proj(proj_input)
            weight = torch.sigmoid(sig_input)
        else:
            sig_input = self.weight_proj(encoder_raw_outputs)
            weight = torch.sigmoid(sig_input)

        not_padding_mask = ~encoder_padding_mask

        # print(not_padding_mask.size())
        # print(torch.squeeze(weight, dim=-1).size())

        weight = (
            torch.squeeze(weight, dim=-1) * not_padding_mask.int()
        )  # weight has shape B x T
        org_weight = weight

        # Sum weights
        if self.training and self.apply_scaling and target_lengths is not None:
            # if self.apply_scaling and target_lengths is not None:   # For validation debugging
            # Conduct scaling when training
            # (target_lengths + 1 because this target_lengths does not take <eos> into consideration)
            weight_sum = weight.sum(-1)  # weight_sum has shape [batch_size]
            normalize_scalar = torch.unsqueeze(
                target_lengths / weight_sum, -1
            )  # normalize_scalar has shape [batch_size, 1]
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
            cur_weight = torch.unsqueeze(weight[:, i], -1)
            # cur_weight has shape [batch_size, 1]
            prev_accumulated_weight = torch.unsqueeze(prev_accumulated_weight, -1)
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
                # if accumulated weights is greater than tail_handling_firing_threshold,
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
                        # less equal than tail_handling_firing_threshold, discarded.
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                        # bigger than tail_handling_firing_threshold, normalized and kept.
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
                (accumulated_states, torch.unsqueeze(cur_accumulated_state, 1)), 1
            )  # shape = [B, L, D]
            fired_states = torch.cat(
                (fired_states, torch.unsqueeze(cur_fired_state, 1)), 1
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
            cur_utt_output = torch.unsqueeze(cur_utt_output, 0)
            # Reshape to [1, fired_max_length, encoder_embed_dim]

            # Concatenate cur_utt_output and cif_outputs along batch axis
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).int()
        # cif_out_padding_mask shape = [batch_size, fired_max_length], where locations with value 0 is False.

        if self.training:
            # In training phase, use the sum of original weights as quantity out for quantity loss
            quantity_out = org_weight.sum(-1)
        else:
            quantity_out = weight.sum(-1)

        if self.cif_output_dim != encoder_embed_dim:
            cif_outputs = self.cif_output_proj(cif_outputs)

        ctxt_cif_outputs = None
        if self.add_cif_ctxt_layers and self.cif_output_dim == self.cif_ctxt_embed_dim:
            x = cif_outputs.transpose(0, 1)
            padding_mask = ~cif_out_padding_mask.bool()
            for layer in self.cif_ctxt_stacks:
                x, _ = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
            ctxt_cif_outputs = x.transpose(0, 1)

        return {
            "cif_out": cif_outputs,  # shape = [batch_size, fired_max_length, cif_output_dim]
            "ctxt_cif_out": ctxt_cif_outputs,  # shape = [batch_size, fired_max_length, cif_ctxt_embed_dim]
            "quantity_out": quantity_out,  # shape = [batch_size]
            "cif_out_padding_mask": cif_out_padding_mask,  # shape = [batch_size, fired_max_length]
        }


@register_model("wav2vec_nar_cif", dataclass=Wav2Vec2NarCIFConfig)
class Wav2Vec2NarCIF(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, cif):
        # Build encoder and decoder part
        super().__init__(encoder, decoder)

        # Build continuous integrate and fire module
        self.cif = cif

    @classmethod
    def build_model(cls, cfg: Wav2Vec2NarCIFConfig, task: FairseqTask):
        """Build a new model instance."""

        assert (
            cfg.autoregressive
        ), "Please set task.autoregressive=true for seq2seq asr models"

        # Obtain dictionary
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        # Register important tokens, such as [CLS] and [SEP]
        cls.bos_id = tgt_dict.bos()  # <bos>
        cls.pad_id = tgt_dict.pad()  # <pad>
        cls.eos_id = tgt_dict.eos()  # <eos>
        cls.unk_id = tgt_dict.unk()  # <unk>

        # Build the whole model
        encoder = cls.build_encoder(cfg, vocab_size=len(tgt_dict))
        cif = cls.build_cif_middleware(cfg)
        decoder = cls.build_decoder(cfg, tgt_dict)

        return Wav2Vec2NarCIF(encoder, decoder, cif)

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2AsrConfig, vocab_size=None):
        return Wav2VecEncoder(cfg, vocab_size)

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2NarCIFConfig, tgt_dict):
        cls.decoder_mode = cfg.decoder_mode
        if cfg.decoder_mode == "nar_decoder":
            return CifNarTransformerDecoder(cfg, tgt_dict)
        elif cfg.decoder_mode == "proj":
            return CifProjDecoder(cfg, tgt_dict)
        else:
            # Default Settings: proj option
            return CifProjDecoder(cfg, tgt_dict)

    @classmethod
    def build_cif_middleware(cls, cfg: Wav2Vec2NarCIFConfig):
        return CifMiddleware(cfg)

    def forward(self, target_lengths_with_eos=None, **kwargs):
        # Forward ASR model (for speech recogntion)
        encoder_out = self.encoder(tbc=False, **kwargs)
        cif_out = self.cif(
            encoder_out, target_lengths_with_eos if self.training else None, **kwargs
        )
        # cif_out = self.cif(encoder_out, target_lengths_with_eos, **kwargs)  # For validation debugging
        decoder_out = self.decoder(cif_out=cif_out, **kwargs)

        model_outputs = {
            # Encoder outputs
            "encoder_out": encoder_out[
                "encoder_out"
            ],  # Encoder out for CTC calculation
            "encoder_raw_out": encoder_out[
                "encoder_raw_out"
            ],  # Encoder raw outputs without projection
            "encoder_padding_mask": encoder_out["padding_mask"],  # B x T
            "padding_mask": encoder_out["padding_mask"],  # B x T
            # Cif outputs
            "quantity_out": cif_out[
                "quantity_out"
            ],  # Quantity out for quantity loss calculation
            "cif_out": cif_out["cif_out"],  # CIF out for decoder prediction, B x T x C
            "ctxt_cif_out": cif_out[
                "ctxt_cif_out"
            ],  # Contextualized cif outputs, B x T x C
            "cif_out_padding_mask": cif_out["cif_out_padding_mask"],  # B x T
            # Decoder outputs
            "decoder_out": decoder_out,  # Decoder outputs
        }

        return model_outputs

    def get_ctc_output(self, **kwargs):
        encoder_outputs = self.encoder(tbc=False, **kwargs)
        ctc_outputs = encoder_outputs["encoder_out"]
        encoder_outputs_padding_mask = encoder_outputs["encoder_padding_mask"]

        return ctc_outputs, encoder_outputs_padding_mask

    def get_cif_output(self, target_lengths_with_eos=None, **kwargs):
        # Fetch the outputs of CifMiddleware
        encoder_outputs = self.encoder(tbc=False, **kwargs)
        cif_out = self.cif(
            encoder_outputs,
            target_lengths_with_eos if self.training else None,
            **kwargs
        )

        return cif_out

    def step_forward_decoder(
        self,
        prev_decoded_tokens=None,
        cif_outputs=None,
        incremental_state=None,
        **kwargs
    ):
        """
        forward decoder with one step
        """
        step_decoder_out, extra_outputs = self.decoder(
            prev_output_tokens=prev_decoded_tokens,
            cif_out=cif_outputs,
            incremental_state=incremental_state,
            **kwargs
        )

        return step_decoder_out, extra_outputs

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @staticmethod
    def get_probs_from_logits(logits, log_probs=False):
        """
        Get normalized probabilities (or log probs) from logits.
        """
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)


class CifProjDecoder(FairseqDecoder):
    def __init__(self, cfg, dictionary):
        super().__init__(dictionary)

        # Load parameters and build model
        self.pre_final_proj_dim = cfg.pre_final_proj_dim
        self.output_dim = len(self.dictionary)
        self.output_proj = Linear(self.pre_final_proj_dim, self.output_dim).cuda()

    def forward(self, prev_output_tokens=None, cif_out=None, **kwargs):
        x = (
            cif_out["ctxt_cif_out"]
            if cif_out["ctxt_cif_out"] is not None
            else cif_out["cif_out"]
        )

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
        final_logits = self.output_proj(x)

        return final_logits, None


class CifNarTransformerDecoder(CifProjDecoder):
    def __init__(self, cfg, dictionary):
        super().__init__(cfg, dictionary)

        # Load decoder parameters
        self.decoder_layers = cfg.decoder_layers
        self.decoder_embed_dim = cfg.decoder_embed_dim
        self.decoder_ffn_embed_dim = cfg.decoder_ffn_embed_dim
        self.decoder_attention_heads = cfg.decoder_attention_heads
        self.decoder_normalize_before = cfg.decoder_normalize_before
        self.decoder_dropout = cfg.decoder_dropout
        self.decoder_attention_dropout = cfg.decoder_attention_dropout
        self.decoder_activation_dropout = cfg.decoder_activation_dropout

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
        x = (
            cif_out["ctxt_cif_out"]
            if cif_out["ctxt_cif_out"] is not None
            else cif_out["cif_out"]
        )
        padding_mask = ~cif_out["cif_out_padding_mask"].bool()

        # Collect shape information
        batch_size, cif_len, cif_embed_dim = x.size()
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
        x = x.transpose(0, 1)  # T x B x C
        for layer in self.decoder_stacks:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
        x = x.transpose(0, 1)  # B x T x C

        final_logits = self.output_proj(x)

        return final_logits, None
