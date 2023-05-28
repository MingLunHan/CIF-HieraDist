# @Time    : 2022/1/20
# @Author  : Minglun Han
# @File    : cif_transformer.py

import sys
import logging
import math
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.wav2vec.wav2vec2 import (
    MASKING_DISTRIBUTION_CHOICES,
    TransformerSentenceEncoderLayer,
)
from fairseq.models.transformer import Embedding, Linear, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from fairseq.models.speech_to_text.s2t_transformer import (
    Conv1dSubsampler,
    S2TTransformerEncoder,
)
from torch import Tensor


logger = logging.getLogger(__name__)
np.set_printoptions(threshold=10000000)
torch.set_printoptions(profile="full")


@register_model("cif_transformer")
class CifTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder, cif, ctc_proj):
        # Register encoder and decoder
        super().__init__(encoder, decoder)

        # Register cif module and ctc projection
        self.cif = cif
        self.ctc_proj = ctc_proj

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-freezing-updates",
            type=int,
            metavar="N",
            help="freeze encoder for first N updates",
        )

        # Cif settings
        parser.add_argument(
            "--cif-embedding-dim",
            type=int,
            help="the dimension of the inputs of cif module",
        )
        parser.add_argument(
            "--produce-weight-type",
            type=str,
            help="choose how to produce the weight for accumulation",
        )
        parser.add_argument(
            "--cif-threshold", type=float, help="the threshold of firing"
        )
        parser.add_argument(
            "--conv-cif-layer-num",
            type=int,
            help="the number of convolutional layers for cif weight generation",
        )
        parser.add_argument(
            "--conv-cif-width",
            type=int,
            help="the width of kernel of convolutional layers",
        )
        parser.add_argument(
            "--conv-cif-output-channels-num",
            type=int,
            help="the number of output channels of cif convolutional layers",
        )
        parser.add_argument(
            "--conv-cif-dropout",
            type=float,
        )
        parser.add_argument(
            "--dense-cif-units-num",
            type=int,
        )
        parser.add_argument(
            "--apply-scaling",
            type=bool,
        )
        parser.add_argument(
            "--apply-tail-handling",
            type=bool,
        )
        parser.add_argument(
            "--tail-handling-firing-threshold",
            type=float,
        )
        parser.add_argument(
            "--add-cif-ctxt-layers",
            type=bool,
        )
        parser.add_argument(
            "--cif-ctxt-layers",
            type=int,
        )
        parser.add_argument(
            "--cif-ctxt-embed-dim",
            type=int,
        )
        parser.add_argument(
            "--cif-ctxt-ffn-embed-dim",
            type=int,
        )
        parser.add_argument(
            "--cif-ctxt-attention-heads",
            type=int,
        )
        parser.add_argument(
            "--cif-ctxt-dropout",
            type=float,
        )
        parser.add_argument(
            "--cif-ctxt-activation-dropout",
            type=float,
        )
        parser.add_argument(
            "--cif-ctxt-attention-dropout",
            type=float,
        )
        parser.add_argument(
            "--cif-ctxt-normalize-before",
            type=bool,
        )

        # Extra decoder settings
        parser.add_argument(
            "--pre-final-proj-dim",
            type=int,
        )
        parser.add_argument(
            "--decoder-type",
            type=str,
        )
        parser.add_argument(
            "--nar-decoder-type",
            type=str,
        )
        parser.add_argument(
            "--decoder-dropout",
            type=float,
        )
        parser.add_argument(
            "--decoder-attention-dropout",
            type=float,
        )
        parser.add_argument(
            "--decoder-activation-dropout",
            type=float,
        )
        parser.add_argument(
            "--no-decoder-input-dropout",
            type=bool,
            # default=True,
        )
        parser.add_argument(
            "--no-decoder-final-normalize",
            type=bool,
            # default=True,
        )

        # Other settings
        parser.add_argument(
            "--calulate-ctc-logits",
            type=bool,
        )

    @classmethod
    def build_encoder(cls, args):
        # Apply original S2T Transformer Encoder as acoustic encoder
        encoder = S2TTransformerEncoder(args)
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        if args.decoder_type == "nar":  # build non-auto regressive decoder
            if args.nar_decoder_type == "projection":
                return CifProjDecoder(args, task.target_dictionary)
            elif args.nar_decoder_type == "transformer":
                return CifNarTransformerDecoder(args, task.target_dictionary)
            else:
                raise NotImplementedError("Not implemented options.")
        else:
            raise NotImplementedError("Not implemented options")

    @classmethod
    def build_cif_middleware(cls, args):
        return CifMiddleware(args)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )

        # build model main body
        encoder = cls.build_encoder(args)
        cif = cls.build_cif_middleware(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)

        # build ctc projection
        ctc_proj = None
        if args.calulate_ctc_logits:
            ctc_proj = Linear(
                args.encoder_embed_dim, len(task.target_dictionary)
            ).cuda()

        return cls(encoder, decoder, cif, ctc_proj)

    @staticmethod
    def get_probs_from_logits(logits, log_probs=False):
        """
        Get normalized probabilities (or log probs) from logits.
        """

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens, target_lengths):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.

        args:
            src_tokens: speech features
            src_lengths: speech feture sequence lengths
        """

        # Forward acoustic encoder
        encoder_out = self.encoder(
            src_tokens=src_tokens, src_lengths=src_lengths
        )  # 2/4 times down sampling after acoustic encoder, just keep it

        # Forward ctc projection to obtain ctc logits for ctc loss calculation
        ctc_logits = None
        if self.ctc_proj is not None:
            ctc_logits = self.ctc_proj(encoder_out["encoder_out"][0])  # T x B x V

        # Forward cif module
        cif_out = self.cif(
            encoder_outputs=encoder_out,
            target_lengths=target_lengths if self.training else None,
            input_lengths=encoder_out["conv_lengths"][0],
        )

        # Forward decoder part
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, cif_out=cif_out
        )

        final_outputs = {
            # Encoder part outputs
            "encoder_padding_mask": lengths_to_padding_mask(
                encoder_out["conv_lengths"][0]
            ),  # B x T
            "ctc_logits": ctc_logits.transpose(0, 1),  # B x T x V
            # Cif module outputs
            "quantity_out": cif_out[
                "quantity_out"
            ],  # Quantity out for quantity loss calculation
            "cif_out": cif_out["cif_out"],  # CIF out for decoder prediction, B x T x C
            "cif_out_padding_mask": cif_out["cif_out_padding_mask"],  # B x T
            # Decoder part outputs
            "decoder_out": decoder_out,  # Decoder outputs (which is final logits for ce calculation)
        }

        return final_outputs

    def get_ctc_output(self, src_tokens, src_lengths, **kwargs):
        with torch.no_grad():
            # Forward acoustic encoder
            encoder_out = self.encoder(
                src_tokens=src_tokens, src_lengths=src_lengths
            )  # 2/4 times down sampling after acoustic encoder, just keep it

            # Forward ctc projection to obtain ctc logits for ctc loss calculation
            ctc_logits = None
            if self.ctc_proj is not None:
                ctc_logits = self.ctc_proj(encoder_out["encoder_out"][0])  # T x B x V

        encoder_outputs_padding_mask = lengths_to_padding_mask(
            encoder_out["conv_lengths"][0]
        )

        return ctc_logits, encoder_outputs_padding_mask

    def get_cif_output(self, src_tokens, src_lengths, target_lengths, **kwargs):
        with torch.no_grad():
            # Forward acoustic encoder
            encoder_out = self.encoder(
                src_tokens=src_tokens, src_lengths=src_lengths
            )  # 2/4 times down sampling after acoustic encoder, just keep it

            # Forward cif module
            cif_out = self.cif(
                encoder_outputs=encoder_out,
                target_lengths=target_lengths if self.training else None,
                input_lengths=encoder_out["conv_lengths"][0],
            )

        return cif_out

    def step_forward_decoder(
        self, prev_decoded_tokens, cif_outputs, incremental_state=None, **kwargs
    ):
        step_decoder_out, extra_outputs = self.decoder(
            prev_output_tokens=prev_decoded_tokens,
            cif_out=cif_outputs,
            incremental_state=incremental_state,
            **kwargs
        )

        return step_decoder_out, extra_outputs


class CifMiddleware(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Get configurations related to continuous integrate-and-fire
        self.cif_threshold = args.cif_threshold
        self.cif_output_dim = args.cif_embedding_dim
        self.encoder_embed_dim = args.encoder_embed_dim
        self.produce_weight_type = args.produce_weight_type
        self.apply_scaling = args.apply_scaling
        self.apply_tail_handling = args.apply_tail_handling
        self.tail_handling_firing_threshold = args.tail_handling_firing_threshold
        self.add_cif_ctxt_layers = args.add_cif_ctxt_layers

        # Build weight projection layer to compute weight from encoder outputs
        if self.produce_weight_type == "dense":
            self.dense_proj = Linear(
                self.encoder_embed_dim, args.dense_cif_units_num
            ).cuda()
            self.weight_proj = Linear(args.dense_cif_units_num, 1).cuda()
        elif self.produce_weight_type == "conv":
            self.cif_conv_layer_num = args.conv_cif_layer_num
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim,
                args.conv_cif_output_channels_num,
                args.conv_cif_width,
                stride=1,
                padding=int(args.conv_cif_width / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ).cuda()
            self.conv_dropout = torch.nn.Dropout(p=args.conv_cif_dropout).cuda()
            self.weight_proj = Linear(args.conv_cif_output_channels_num, 1).cuda()
        else:
            self.weight_proj = Linear(self.encoder_embed_dim, 1).cuda()

        # Build the final projection layer for cif outputs
        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = Linear(
                self.encoder_embed_dim, self.cif_output_dim, bias=False
            ).cuda()

        # Build cif contextual layers
        if self.add_cif_ctxt_layers:
            self.cif_ctxt_embed_dim = args.cif_ctxt_embed_dim
            self.cif_ctxt_stacks = nn.ModuleList(
                [
                    TransformerSentenceEncoderLayer(
                        embedding_dim=args.cif_ctxt_embed_dim,
                        ffn_embedding_dim=args.cif_ctxt_ffn_embed_dim,
                        num_attention_heads=args.cif_ctxt_attention_heads,
                        dropout=args.cif_ctxt_dropout,
                        activation_dropout=args.cif_ctxt_activation_dropout,
                        attention_dropout=args.cif_ctxt_attention_dropout,
                        layer_norm_first=args.cif_ctxt_normalize_before,
                    )
                    for _ in range(args.cif_ctxt_layers)
                ]
            )

    def forward(self, encoder_outputs, target_lengths=None, input_lengths=None):
        """
        encoder_out should have shape B x T x C
        encoder_padding_mask should have shape B x T
        targets_length should have shape B
        """

        # Prepare inputs
        encoder_out = encoder_outputs["encoder_out"][0].transpose(0, 1)  # B x T x C
        if len(encoder_outputs["encoder_padding_mask"]) != 0:
            encoder_padding_mask = encoder_outputs["encoder_padding_mask"][0]  # B x T
        else:
            assert (
                input_lengths is not None
            ), "Please ensure that input_lengths is provided."
            encoder_padding_mask = lengths_to_padding_mask(input_lengths)  # B x T

        # Forward weight generation
        if self.produce_weight_type == "dense":
            proj_out = self.dense_proj(encoder_out)
            act_proj_out = torch.relu(proj_out)
            sig_input = self.weight_proj(act_proj_out)
            weight = torch.sigmoid(sig_input)
            # weight has shape [batch_size, length, 1]
        elif self.produce_weight_type == "conv":
            conv_input = encoder_out.permute(0, 2, 1)
            # Adjust the shape of convolution layer input [B, C_in, T]
            conv_out = self.conv(conv_input)
            # conv_out has shape [B, C_out, T]
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.conv_dropout(proj_input)
            # Adjust conv output to shape [B, T, C_cif]
            sig_input = self.weight_proj(proj_input)
            weight = torch.sigmoid(sig_input)
        else:
            sig_input = self.weight_proj(encoder_out)
            weight = torch.sigmoid(sig_input)

        not_padding_mask = ~encoder_padding_mask
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
                target_lengths / (weight_sum + 1e-8), -1
            )  # normalize_scalar has shape [batch_size, 1]
            weight = weight * normalize_scalar

        # Integrate and fire
        batch_size = encoder_out.size(0)
        max_length = encoder_out.size(1)
        encoder_embed_dim = encoder_out.size(2)
        padding_start_id = not_padding_mask.sum(-1)  # shape B

        # Initialize
        accumulated_weights = torch.zeros(batch_size, 0, dtype=encoder_out.dtype).cuda()
        accumulated_states = torch.zeros(
            batch_size, 0, encoder_embed_dim, dtype=encoder_out.dtype
        ).cuda()
        fired_states = torch.zeros(
            batch_size, 0, encoder_embed_dim, dtype=encoder_out.dtype
        ).cuda()

        # Begin integrate and fire
        for i in range(max_length):
            # Get previous states from the recorded tensor
            prev_accumulated_weight = (
                torch.zeros([batch_size], dtype=encoder_out.dtype).cuda()
                if i == 0
                else accumulated_weights[:, i - 1]
            )
            prev_accumulated_state = (
                torch.zeros(
                    [batch_size, encoder_embed_dim], dtype=encoder_out.dtype
                ).cuda()
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
                torch.ones_like(prev_accumulated_weight, dtype=encoder_out.dtype).cuda()
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
                (cur_weight - remained_weight) * encoder_out[:, i, :],
                prev_accumulated_state + cur_weight * encoder_out[:, i, :],
            )  # B x C

            # Obtain fired state of current step
            # firing locations has meaningful representations, while non-firing locations is all-zero embeddings
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                prev_accumulated_state + remained_weight * encoder_out[:, i, :],
                torch.zeros(
                    [batch_size, encoder_embed_dim], dtype=encoder_out.dtype
                ).cuda(),
            )  # B x C

            # Handling the speech tail by rounding up and down
            if (not self.training) and self.apply_tail_handling:
                # When encoder output position exceeds the max valid position,
                # if accumulated weights is greater than tail_handling_firing_threshold,
                # current state should be reserved, otherwise it is discarded.
                cur_fired_state = torch.where(
                    i
                    == padding_start_id.unsqueeze(dim=-1).repeat(
                        [1, encoder_embed_dim]
                    ),  # B x C
                    torch.where(
                        cur_accumulated_weight.repeat([1, encoder_embed_dim])
                        <= self.tail_handling_firing_threshold,  # B x C
                        torch.zeros(
                            [batch_size, encoder_embed_dim], dtype=encoder_out.dtype
                        ).cuda(),
                        # less equal than tail_handling_firing_threshold, discarded.
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                        # bigger than tail_handling_firing_threshold, normalized and kept.
                        # eps = 1e-10 for preveting overflow.
                    ),
                    cur_fired_state,
                )  # B x C

            # For normal condition, including both training and evaluation
            # Mask padded locations with all-zero embeddings
            cur_fired_state = torch.where(
                torch.full(
                    [batch_size, encoder_embed_dim], i, dtype=encoder_out.dtype
                ).cuda()
                > padding_start_id.unsqueeze(dim=-1).repeat(
                    [1, encoder_embed_dim]
                ),  # B x C
                torch.zeros(
                    [batch_size, encoder_embed_dim], dtype=encoder_out.dtype
                ).cuda(),
                cur_fired_state,
            )

            # Update accumulated arguments
            accumulated_weights = torch.cat(
                (accumulated_weights, cur_accumulated_weight), 1
            )  # B x T
            accumulated_states = torch.cat(
                (accumulated_states, torch.unsqueeze(cur_accumulated_state, 1)), 1
            )  # shape = [B, L, D]
            fired_states = torch.cat(
                (fired_states, torch.unsqueeze(cur_fired_state, 1)), 1
            )  # shape = [B, L, D]

        # Extracts cif_outputs for each utterance
        fired_marks = (torch.abs(fired_states).sum(-1) != 0.0).int()  # B x T
        fired_utt_length = fired_marks.sum(-1)  # B
        fired_max_length = (
            fired_utt_length.max().int()
        )  # The maximum of fired times in current batch
        cif_outputs = torch.zeros(
            [0, fired_max_length, encoder_embed_dim], dtype=encoder_out.dtype
        ).cuda()  # Initialize cif outputs
        cif_durations = torch.zeros(
            [0, fired_max_length], dtype=torch.int32
        ).cuda()  # Initialize cif durations

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
                    torch.full(
                        [pad_length, encoder_embed_dim], 0.0, dtype=encoder_out.dtype
                    ).cuda(),
                ),
                dim=0,
            )  # Pad current utterance cif outputs to fired_max_length
            cur_utt_output = torch.unsqueeze(cur_utt_output, 0)
            # Reshape to [1, fired_max_length, encoder_embed_dim]

            # Concatenate cur_utt_output and cif_outputs along batch axis
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)

            # Collect cif durations
            cur_fired_indices = torch.nonzero(cur_utt_fired_mark)[:, -1]
            shifted_cur_fired_indices = torch.cat(
                [torch.zeros([1], dtype=torch.int32).cuda(), cur_fired_indices], dim=-1
            )[: cur_fired_indices.size(0)]
            cur_cif_durations = cur_fired_indices - shifted_cur_fired_indices
            cur_cif_durations = torch.cat(
                (
                    cur_cif_durations,
                    torch.full([pad_length], 0, dtype=torch.int32).cuda(),
                ),
                dim=0,
            ).unsqueeze(dim=0)
            cif_durations = torch.cat(
                [cif_durations, cur_cif_durations], dim=0
            )  # cancat at batch axis

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).int()
        cif_outputs_lens = cif_out_padding_mask.sum(-1)
        cif_outputs = torch.where(
            cif_outputs_lens.unsqueeze(-1).unsqueeze(-1) > 0,  # B x T x C
            cif_outputs,
            torch.zeros_like(cif_outputs).type_as(cif_outputs).cuda(),
        )

        # Check: Handle exceptions
        if cif_out_padding_mask.size(-1) == 0:
            cif_outputs = (
                torch.zeros([batch_size, 1, cif_outputs.size(-1)])
                .type_as(cif_outputs)
                .cuda()
            )
            cif_out_padding_mask = torch.ones([batch_size, 1]).bool().cuda()

        base_cif_out_padding_mask = (
            torch.cat(
                [torch.ones(1), torch.zeros(cif_out_padding_mask.size(-1) - 1)], dim=0
            )
            .unsqueeze(0)
            .cuda()
        )  # 1 x 1
        temp_cif_out_padding_mask = base_cif_out_padding_mask.repeat(
            cif_out_padding_mask.size(0), 1
        )  # B x T
        cif_out_padding_mask = torch.where(
            cif_outputs_lens.unsqueeze(-1) > 0,  # B x T
            cif_out_padding_mask,
            temp_cif_out_padding_mask.type_as(cif_out_padding_mask),
        )

        if self.training:
            # In training phase, use the sum of original weights
            # as quantity out for quantity loss.
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
            "cif_out_padding_mask": cif_out_padding_mask,  # shape = [batch_size, fired_max_length]
            "ctxt_cif_out": ctxt_cif_outputs,  # shape = [batch_size, fired_max_length, cif_ctxt_embed_dim]
            "quantity_out": quantity_out,  # shape = [batch_size]
            "cif_durations": cif_durations,  # shape = [batch_size, fired_max_length]
        }


class CifProjDecoder(FairseqDecoder):
    def __init__(self, cfg, dictionary):
        super().__init__(dictionary)

        # Load parameters and build model
        if cfg.no_decoder_input_dropout:
            self.input_dropout = None
        else:
            self.input_dropout = FairseqDropout(
                p=cfg.decoder_dropout, module_name=self.__class__.__name__
            )

        if cfg.decoder_normalize_before and not cfg.no_decoder_final_normalize:
            self.layer_norm = LayerNorm(cfg.encoder_embed_dim)
        else:
            self.layer_norm = None

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
            x = torch.zeros([batch_size, cif_len, cif_embed_dim], dtype=x.dtype).cuda()

        # Regularize the length of input tokens and cif outputs
        min_len = min(prev_output_tokens_len, cif_len)
        x = x[:, :min_len, :]  # B x min_len x C

        # Add dropout
        if self.input_dropout is not None:
            x = self.input_dropout(x)

        # Add normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)

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
        )  # B x T x C
        padding_mask = ~cif_out["cif_out_padding_mask"].bool()  # B x T

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

        # Forward input dropout
        if self.input_dropout is not None:
            x = self.input_dropout(x)

        # Forward decoder
        x = x.transpose(0, 1)  # T x B x C
        for layer in self.decoder_stacks:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
        x = x.transpose(0, 1)  # B x T x C

        # Forward normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        final_logits = self.output_proj(x)

        return final_logits, None


@register_model_architecture(model_name="cif_transformer", arch_name="cif_transformer")
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 768
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 768)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(
        args, "tail_handling_firing_threshold", 0.5
    )
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(
        args, "cif_ctxt_embed_dim", args.encoder_embed_dim
    )
    args.cif_ctxt_ffn_embed_dim = getattr(
        args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.cif_ctxt_attention_heads = getattr(
        args, "cif_ctxt_attention_heads", args.encoder_attention_heads
    )
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(
        args, "cif_ctxt_activation_dropout", args.activation_dropout
    )
    args.cif_ctxt_attention_dropout = getattr(
        args, "cif_ctxt_attention_dropout", args.attention_dropout
    )
    args.cif_ctxt_normalize_before = getattr(
        args, "cif_ctxt_normalize_before", args.encoder_normalize_before
    )

    # Cif-style Decoder settings
    args.pre_final_proj_dim = getattr(
        args, "pre_final_proj_dim", args.decoder_embed_dim
    )
    args.decoder_type = getattr(args, "decoder_type", "nar")
    args.nar_decoder_type = getattr(args, "nar_decoder_type", "projection")
    args.decoder_dropout = getattr(args, "decoder_dropout", args.dropout)
    args.decoder_attention_dropout = getattr(
        args, "decoder_attention_dropout", args.attention_dropout
    )
    args.decoder_activation_dropout = getattr(
        args, "decoder_activation_dropout", args.activation_dropout
    )

    # Other settings
    args.calulate_ctc_logits = getattr(args, "calulate_ctc_logits", True)


@register_model_architecture(
    model_name="cif_transformer", arch_name="cif_transformer_alpha"
)
def cif_transformer_alpha(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture(
    model_name="cif_transformer", arch_name="cif_transformer_exp1_1"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 512
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 512)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(
        args, "tail_handling_firing_threshold", 0.4
    )
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(
        args, "cif_ctxt_embed_dim", args.encoder_embed_dim
    )
    args.cif_ctxt_ffn_embed_dim = getattr(
        args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.cif_ctxt_attention_heads = getattr(
        args, "cif_ctxt_attention_heads", args.encoder_attention_heads
    )
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(
        args, "cif_ctxt_activation_dropout", args.activation_dropout
    )
    args.cif_ctxt_attention_dropout = getattr(
        args, "cif_ctxt_attention_dropout", args.attention_dropout
    )
    args.cif_ctxt_normalize_before = getattr(
        args, "cif_ctxt_normalize_before", args.encoder_normalize_before
    )

    # Cif-style Decoder settings
    args.pre_final_proj_dim = getattr(
        args, "pre_final_proj_dim", args.decoder_embed_dim
    )
    args.decoder_type = getattr(args, "decoder_type", "nar")
    args.nar_decoder_type = getattr(args, "nar_decoder_type", "projection")
    args.decoder_dropout = getattr(args, "decoder_dropout", args.dropout)
    args.decoder_attention_dropout = getattr(
        args, "decoder_attention_dropout", args.attention_dropout
    )
    args.decoder_activation_dropout = getattr(
        args, "decoder_activation_dropout", args.activation_dropout
    )

    # Other settings
    args.calulate_ctc_logits = getattr(args, "calulate_ctc_logits", True)


@register_model_architecture(
    model_name="cif_transformer", arch_name="cif_transformer_exp1_2"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 512
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 512)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(
        args, "tail_handling_firing_threshold", 0.4
    )
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(
        args, "cif_ctxt_embed_dim", args.encoder_embed_dim
    )
    args.cif_ctxt_ffn_embed_dim = getattr(
        args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.cif_ctxt_attention_heads = getattr(
        args, "cif_ctxt_attention_heads", args.encoder_attention_heads
    )
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(
        args, "cif_ctxt_activation_dropout", args.activation_dropout
    )
    args.cif_ctxt_attention_dropout = getattr(
        args, "cif_ctxt_attention_dropout", args.attention_dropout
    )
    args.cif_ctxt_normalize_before = getattr(
        args, "cif_ctxt_normalize_before", args.encoder_normalize_before
    )

    # Cif-style Decoder settings
    args.pre_final_proj_dim = getattr(
        args, "pre_final_proj_dim", args.decoder_embed_dim
    )
    args.decoder_type = getattr(args, "decoder_type", "nar")
    args.nar_decoder_type = getattr(args, "nar_decoder_type", "projection")
    args.decoder_dropout = getattr(args, "decoder_dropout", args.dropout)
    args.decoder_attention_dropout = getattr(
        args, "decoder_attention_dropout", args.attention_dropout
    )
    args.decoder_activation_dropout = getattr(
        args, "decoder_activation_dropout", args.activation_dropout
    )

    # Other settings
    args.calulate_ctc_logits = getattr(args, "calulate_ctc_logits", True)


@register_model_architecture(
    model_name="cif_transformer", arch_name="cif_transformer_exp1_3"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 512)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 256
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 256)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(
        args, "tail_handling_firing_threshold", 0.4
    )
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(
        args, "cif_ctxt_embed_dim", args.encoder_embed_dim
    )
    args.cif_ctxt_ffn_embed_dim = getattr(
        args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.cif_ctxt_attention_heads = getattr(
        args, "cif_ctxt_attention_heads", args.encoder_attention_heads
    )
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(
        args, "cif_ctxt_activation_dropout", args.activation_dropout
    )
    args.cif_ctxt_attention_dropout = getattr(
        args, "cif_ctxt_attention_dropout", args.attention_dropout
    )
    args.cif_ctxt_normalize_before = getattr(
        args, "cif_ctxt_normalize_before", args.encoder_normalize_before
    )

    # Cif-style Decoder settings
    args.pre_final_proj_dim = getattr(
        args, "pre_final_proj_dim", args.decoder_embed_dim
    )
    args.decoder_type = getattr(args, "decoder_type", "nar")
    args.nar_decoder_type = getattr(args, "nar_decoder_type", "projection")
    args.decoder_dropout = getattr(args, "decoder_dropout", args.dropout)
    args.decoder_attention_dropout = getattr(
        args, "decoder_attention_dropout", args.attention_dropout
    )
    args.decoder_activation_dropout = getattr(
        args, "decoder_activation_dropout", args.activation_dropout
    )

    # Other settings
    args.calulate_ctc_logits = getattr(args, "calulate_ctc_logits", True)


@register_model_architecture(
    model_name="cif_transformer", arch_name="cif_transformer_exp1_4"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 512
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 512)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(
        args, "tail_handling_firing_threshold", 0.4
    )
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(
        args, "cif_ctxt_embed_dim", args.encoder_embed_dim
    )
    args.cif_ctxt_ffn_embed_dim = getattr(
        args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.cif_ctxt_attention_heads = getattr(
        args, "cif_ctxt_attention_heads", args.encoder_attention_heads
    )
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(
        args, "cif_ctxt_activation_dropout", args.activation_dropout
    )
    args.cif_ctxt_attention_dropout = getattr(
        args, "cif_ctxt_attention_dropout", args.attention_dropout
    )
    args.cif_ctxt_normalize_before = getattr(
        args, "cif_ctxt_normalize_before", args.encoder_normalize_before
    )

    # Cif-style Decoder settings
    args.pre_final_proj_dim = getattr(
        args, "pre_final_proj_dim", args.decoder_embed_dim
    )
    args.decoder_type = getattr(args, "decoder_type", "nar")
    args.nar_decoder_type = getattr(args, "nar_decoder_type", "transformer")
    args.decoder_dropout = getattr(args, "decoder_dropout", args.dropout)
    args.decoder_attention_dropout = getattr(
        args, "decoder_attention_dropout", args.attention_dropout
    )
    args.decoder_activation_dropout = getattr(
        args, "decoder_activation_dropout", args.activation_dropout
    )
    args.no_decoder_final_normalize = getattr(args, "no_decoder_final_normalize", True)
    args.no_decoder_input_dropout = getattr(args, "no_decoder_input_dropout", True)

    # Other settings
    args.calulate_ctc_logits = getattr(args, "calulate_ctc_logits", True)


@register_model_architecture(
    model_name="cif_transformer", arch_name="cif_transformer_exp1_5"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 512
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 512)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(
        args, "tail_handling_firing_threshold", 0.4
    )
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(
        args, "cif_ctxt_embed_dim", args.encoder_embed_dim
    )
    args.cif_ctxt_ffn_embed_dim = getattr(
        args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.cif_ctxt_attention_heads = getattr(
        args, "cif_ctxt_attention_heads", args.encoder_attention_heads
    )
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(
        args, "cif_ctxt_activation_dropout", args.activation_dropout
    )
    args.cif_ctxt_attention_dropout = getattr(
        args, "cif_ctxt_attention_dropout", args.attention_dropout
    )
    args.cif_ctxt_normalize_before = getattr(
        args, "cif_ctxt_normalize_before", args.encoder_normalize_before
    )

    # Cif-style Decoder settings
    args.pre_final_proj_dim = getattr(
        args, "pre_final_proj_dim", args.decoder_embed_dim
    )
    args.decoder_type = getattr(args, "decoder_type", "nar")
    args.nar_decoder_type = getattr(args, "nar_decoder_type", "projection")
    args.decoder_dropout = getattr(args, "decoder_dropout", args.dropout)
    args.decoder_attention_dropout = getattr(
        args, "decoder_attention_dropout", args.attention_dropout
    )
    args.decoder_activation_dropout = getattr(
        args, "decoder_activation_dropout", args.activation_dropout
    )
    args.no_decoder_final_normalize = getattr(args, "no_decoder_final_normalize", False)
    args.no_decoder_input_dropout = getattr(args, "no_decoder_input_dropout", False)

    # Other settings
    args.calulate_ctc_logits = getattr(args, "calulate_ctc_logits", True)


@register_model_architecture(
    model_name="cif_transformer", arch_name="cif_transformer_exp1_6"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 512
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 512)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(
        args, "tail_handling_firing_threshold", 0.4
    )
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(
        args, "cif_ctxt_embed_dim", args.encoder_embed_dim
    )
    args.cif_ctxt_ffn_embed_dim = getattr(
        args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.cif_ctxt_attention_heads = getattr(
        args, "cif_ctxt_attention_heads", args.encoder_attention_heads
    )
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(
        args, "cif_ctxt_activation_dropout", args.activation_dropout
    )
    args.cif_ctxt_attention_dropout = getattr(
        args, "cif_ctxt_attention_dropout", args.attention_dropout
    )
    args.cif_ctxt_normalize_before = getattr(
        args, "cif_ctxt_normalize_before", args.encoder_normalize_before
    )

    # Cif-style Decoder settings
    args.pre_final_proj_dim = getattr(
        args, "pre_final_proj_dim", args.decoder_embed_dim
    )
    args.decoder_type = getattr(args, "decoder_type", "nar")
    args.nar_decoder_type = getattr(args, "nar_decoder_type", "transformer")
    args.decoder_dropout = getattr(args, "decoder_dropout", args.dropout)
    args.decoder_attention_dropout = getattr(
        args, "decoder_attention_dropout", args.attention_dropout
    )
    args.decoder_activation_dropout = getattr(
        args, "decoder_activation_dropout", args.activation_dropout
    )
    args.no_decoder_final_normalize = getattr(args, "no_decoder_final_normalize", False)
    args.no_decoder_input_dropout = getattr(args, "no_decoder_input_dropout", False)

    # Other settings
    args.calulate_ctc_logits = getattr(args, "calulate_ctc_logits", True)


@register_model_architecture(
    model_name="cif_transformer", arch_name="cif_transformer_exp1_7"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 512
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 512)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(
        args, "tail_handling_firing_threshold", 0.4
    )
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(
        args, "cif_ctxt_embed_dim", args.encoder_embed_dim
    )
    args.cif_ctxt_ffn_embed_dim = getattr(
        args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.cif_ctxt_attention_heads = getattr(
        args, "cif_ctxt_attention_heads", args.encoder_attention_heads
    )
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(
        args, "cif_ctxt_activation_dropout", args.activation_dropout
    )
    args.cif_ctxt_attention_dropout = getattr(
        args, "cif_ctxt_attention_dropout", args.attention_dropout
    )
    args.cif_ctxt_normalize_before = getattr(
        args, "cif_ctxt_normalize_before", args.encoder_normalize_before
    )

    # Cif-style Decoder settings
    args.pre_final_proj_dim = getattr(
        args, "pre_final_proj_dim", args.decoder_embed_dim
    )
    args.decoder_type = getattr(args, "decoder_type", "nar")
    args.nar_decoder_type = getattr(args, "nar_decoder_type", "transformer")
    args.decoder_dropout = getattr(args, "decoder_dropout", args.dropout)
    args.decoder_attention_dropout = getattr(
        args, "decoder_attention_dropout", args.attention_dropout
    )
    args.decoder_activation_dropout = getattr(
        args, "decoder_activation_dropout", args.activation_dropout
    )
    args.no_decoder_final_normalize = getattr(args, "no_decoder_final_normalize", True)
    args.no_decoder_input_dropout = getattr(args, "no_decoder_input_dropout", True)

    # Other settings
    args.calulate_ctc_logits = getattr(args, "calulate_ctc_logits", True)


@register_model_architecture(
    model_name="cif_transformer", arch_name="cif_transformer_exp1_8"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 512
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 512)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(
        args, "tail_handling_firing_threshold", 0.4
    )
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(
        args, "cif_ctxt_embed_dim", args.encoder_embed_dim
    )
    args.cif_ctxt_ffn_embed_dim = getattr(
        args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.cif_ctxt_attention_heads = getattr(
        args, "cif_ctxt_attention_heads", args.encoder_attention_heads
    )
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(
        args, "cif_ctxt_activation_dropout", args.activation_dropout
    )
    args.cif_ctxt_attention_dropout = getattr(
        args, "cif_ctxt_attention_dropout", args.attention_dropout
    )
    args.cif_ctxt_normalize_before = getattr(
        args, "cif_ctxt_normalize_before", args.encoder_normalize_before
    )

    # Cif-style Decoder settings
    args.pre_final_proj_dim = getattr(
        args, "pre_final_proj_dim", args.decoder_embed_dim
    )
    args.decoder_type = getattr(args, "decoder_type", "nar")
    args.nar_decoder_type = getattr(args, "nar_decoder_type", "projection")
    args.decoder_dropout = getattr(args, "decoder_dropout", args.dropout)
    args.decoder_attention_dropout = getattr(
        args, "decoder_attention_dropout", args.attention_dropout
    )
    args.decoder_activation_dropout = getattr(
        args, "decoder_activation_dropout", args.activation_dropout
    )
    args.no_decoder_final_normalize = getattr(args, "no_decoder_final_normalize", True)
    args.no_decoder_input_dropout = getattr(args, "no_decoder_input_dropout", True)

    # Other settings
    args.calulate_ctc_logits = getattr(args, "calulate_ctc_logits", True)


@register_model_architecture(
    model_name="cif_transformer", arch_name="cif_transformer_exp1_9"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 512
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 512)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(
        args, "tail_handling_firing_threshold", 0.4
    )
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(
        args, "cif_ctxt_embed_dim", args.encoder_embed_dim
    )
    args.cif_ctxt_ffn_embed_dim = getattr(
        args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.cif_ctxt_attention_heads = getattr(
        args, "cif_ctxt_attention_heads", args.encoder_attention_heads
    )
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(
        args, "cif_ctxt_activation_dropout", args.activation_dropout
    )
    args.cif_ctxt_attention_dropout = getattr(
        args, "cif_ctxt_attention_dropout", args.attention_dropout
    )
    args.cif_ctxt_normalize_before = getattr(
        args, "cif_ctxt_normalize_before", args.encoder_normalize_before
    )

    # Cif-style Decoder settings
    args.pre_final_proj_dim = getattr(
        args, "pre_final_proj_dim", args.decoder_embed_dim
    )
    args.decoder_type = getattr(args, "decoder_type", "nar")
    args.nar_decoder_type = getattr(args, "nar_decoder_type", "transformer")
    args.decoder_dropout = getattr(args, "decoder_dropout", 0.25)
    args.decoder_attention_dropout = getattr(args, "decoder_attention_dropout", 0.25)
    args.decoder_activation_dropout = getattr(args, "decoder_activation_dropout", 0.25)
    args.no_decoder_final_normalize = getattr(args, "no_decoder_final_normalize", False)
    args.no_decoder_input_dropout = getattr(args, "no_decoder_input_dropout", False)

    # Other settings
    args.calulate_ctc_logits = getattr(args, "calulate_ctc_logits", True)


@register_model_architecture(
    model_name="cif_transformer", arch_name="cif_transformer_exp6"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 512)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.15)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 256
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 256)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(
        args, "tail_handling_firing_threshold", 0.4
    )
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(
        args, "cif_ctxt_embed_dim", args.encoder_embed_dim
    )
    args.cif_ctxt_ffn_embed_dim = getattr(
        args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.cif_ctxt_attention_heads = getattr(
        args, "cif_ctxt_attention_heads", args.encoder_attention_heads
    )
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(
        args, "cif_ctxt_activation_dropout", args.activation_dropout
    )
    args.cif_ctxt_attention_dropout = getattr(
        args, "cif_ctxt_attention_dropout", args.attention_dropout
    )
    args.cif_ctxt_normalize_before = getattr(
        args, "cif_ctxt_normalize_before", args.encoder_normalize_before
    )

    # Cif-style Decoder settings
    args.pre_final_proj_dim = getattr(
        args, "pre_final_proj_dim", args.decoder_embed_dim
    )
    args.decoder_type = getattr(args, "decoder_type", "nar")
    args.nar_decoder_type = getattr(args, "nar_decoder_type", "transformer")
    args.decoder_dropout = getattr(args, "decoder_dropout", args.dropout)
    args.decoder_attention_dropout = getattr(
        args, "decoder_attention_dropout", args.attention_dropout
    )
    args.decoder_activation_dropout = getattr(
        args, "decoder_activation_dropout", args.activation_dropout
    )
    args.no_decoder_final_normalize = getattr(args, "no_decoder_final_normalize", False)
    args.no_decoder_input_dropout = getattr(args, "no_decoder_input_dropout", False)

    # Other settings
    args.calulate_ctc_logits = getattr(args, "calulate_ctc_logits", True)
