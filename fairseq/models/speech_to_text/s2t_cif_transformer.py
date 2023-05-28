#!/usr/bin/env python3

import sys
import logging
import argparse
import math
import copy
import edlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, utils, tasks
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerConfig
from fairseq.modules.conformer_layer import ConformerEncoderLayer
from fairseq.modules.positional_encoding import RelPositionalEncoding
from fairseq.modules.rotary_positional_embedding import RotaryPositionalEmbedding
from fairseq.models.speech_to_text.cif_transformer import CifMiddleware
from fairseq.modules import (
    FairseqDropout,
    TransformerEncoderLayer,
    AdaptiveSoftmax,
    BaseLayer,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer
from fairseq.modules import transformer_layer
from torch import Tensor


logger = logging.getLogger(__name__)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBase":
        return "TransformerDecoder"
    else:
        return module_name


# Expand the dimension of given tensors
def expand_tensor_dim(x, expand_size, target_dim=1, reduce=False):
    assert target_dim == 1, "only the expansion at the second dimension is available."

    rank = len(x.size())
    unsq_x = x.unsqueeze(target_dim)
    if rank == 1:
        sz1 = x.size()
        x = unsq_x.repeat(1, expand_size)
        x = x.view(sz1 * expand_size) if reduce else x
    elif rank == 2:
        sz1, sz2 = x.size()
        x = unsq_x.repeat(1, expand_size, 1)
        x = x.view((sz1 * expand_size), sz2) if reduce else x
    elif rank == 3:
        sz1, sz2, sz3 = x.size()
        x = unsq_x.repeat(1, expand_size, 1, 1)
        x = x.view((sz1 * expand_size), sz2, sz3) if reduce else x
    else:
        raise NotImplementedError("Not supported rank %d" % rank)

    return x


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
            # GLU activation will cause dimension discount 50% in default.

        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)

        return x, self.get_out_seq_lens_tensor(src_lengths)


class Conv2dSubsampler(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, conv_output_channels, kernel_sizes):
        """Construct an Conv2dSubsampling object."""
        super().__init__()

        assert len(conv_output_channels) == len(kernel_sizes)
        self._num_conv_layers = len(kernel_sizes)
        self.conv = nn.ModuleList([])
        for layer_id, (output_channel, kernel_size) in enumerate(
            zip(conv_output_channels, kernel_sizes)
        ):
            if layer_id == 0:
                self.conv.append(torch.nn.Conv2d(1, output_channel, kernel_size, 2))
            else:
                prev_output_channel = conv_output_channels[layer_id - 1]
                self.conv.append(
                    torch.nn.Conv2d(prev_output_channel, output_channel, kernel_size, 2)
                )
            self.conv.append(torch.nn.ReLU())

        conv_final_dim = conv_output_channels[-1]
        if self._num_conv_layers == 1:
            self.out = torch.nn.Sequential(
                torch.nn.Linear(conv_final_dim * ((idim - 1) // 2), odim),
            )
        elif self._num_conv_layers == 2:
            self.out = torch.nn.Sequential(
                torch.nn.Linear(conv_final_dim * (((idim - 1) // 2 - 1) // 2), odim),
            )
        else:
            raise NotImplementedError("Not supported value.")

    def forward(self, x, src_lens):
        """Subsample x."""
        x = x.unsqueeze(1)  # B x C x T x F
        for module in self.conv:
            x = module(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))  # B x T x C
        x = x.permute(1, 0, 2)  # T x B x C
        out_seq_lens = self.get_out_seq_lens_tensor(src_lens)

        return x, out_seq_lens

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self._num_conv_layers):
            out = ((out.float() - 1) / 2).floor().long()
        return out

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class CtcConstrainedCifMiddleware(nn.Module):
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
                        activation_dropout=args.cif_ctxt_dropout
                        or args.cif_ctxt_activation_dropout,
                        attention_dropout=args.cif_ctxt_dropout
                        or args.cif_ctxt_activation_dropout,
                        layer_norm_first=args.cif_ctxt_normalize_before,
                    )
                    for _ in range(args.cif_ctxt_layers)
                ]
            )

        # CTC Constrained training settings
        self.use_ctc_constraint = args.use_ctc_constraint
        if self.use_ctc_constraint:
            self.ctc_prob_threshold = args.ctc_prob_threshold

    def forward(
        self, encoder_outputs, target_lengths=None, input_lengths=None, ctc_logits=None
    ):
        """
        Args:
            encoder_out: B x T x C
            encoder_padding_mask: B x T
            targets_length: B
            ctc_logits: B x T x V (including blank_token_id)
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
            sig_input = sig_input.float()
            weight = torch.sigmoid(sig_input)
            weight = weight.type_as(encoder_out)
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
            weight = weight.float()
            weight_sum = weight.sum(-1)  # weight_sum has shape [batch_size]
            normalize_scalar = torch.unsqueeze(
                target_lengths / (weight_sum + 1e-8), -1
            )  # B x 1
            weight = weight * normalize_scalar
            weight = weight.type_as(org_weight)  # B x T

        ctc_border_marks = None
        if self.use_ctc_constraint and ctc_logits is not None:
            ctc_probs = utils.softmax(
                ctc_logits.transpose(0, 1).float(), dim=-1
            )  # B x T x V

            # TODO: remember the default blank id should be <bos> id (0)
            blank_probs = ctc_probs[:, :, 0]  # B x T
            non_blank_probs = 1.0 - blank_probs  # B x T
            ctc_border_marks = (
                non_blank_probs > self.ctc_prob_threshold
            ).int()  # B x T
            # Seems like [[0,0,0,0,1,0,1], ...]

        # Integrate and fire
        batch_size = encoder_out.size(0)
        max_length = encoder_out.size(1)
        encoder_embed_dim = encoder_out.size(2)
        padding_start_id = not_padding_mask.sum(-1)  # B

        # Initialize
        accumulated_weights = torch.zeros(batch_size, 0, dtype=encoder_out.dtype).cuda()
        accumulated_states = torch.zeros(
            batch_size, 0, encoder_embed_dim, dtype=encoder_out.dtype
        ).cuda()
        fired_states = torch.zeros(
            batch_size, 0, encoder_embed_dim, dtype=encoder_out.dtype
        ).cuda()
        ctc_accum_weights = (
            torch.zeros(batch_size, 0, dtype=encoder_out.dtype).cuda()
            if self.use_ctc_constraint
            else None
        )  # B x T

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
            )  # B x 1

            cur_ctc_accum_weight = None
            if self.use_ctc_constraint and ctc_border_marks is not None:
                if i == 0:
                    prev_ctc_accum_weight = torch.zeros(
                        [batch_size], dtype=encoder_out.dtype
                    ).cuda()  # B
                else:
                    prev_ctc_border_marks = ctc_border_marks[:, i - 1]  # B
                    prev_ctc_accum_weight = torch.where(
                        prev_ctc_border_marks.float() == 1.0,  # B
                        torch.zeros([batch_size], dtype=encoder_out.dtype).cuda(),  # B
                        ctc_accum_weights[:, i - 1],  # B
                    )  # B x 1
                cur_ctc_accum_weight = prev_ctc_accum_weight.unsqueeze(-1) + cur_weight

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
                # print("______________________")
                # print(i)
                # print("cur_accumulated_state:", cur_accumulated_state[:, :10])
                # print("cur_accumulated_weight: ", cur_accumulated_weight)
                # print(i == padding_start_id)
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
            if self.use_ctc_constraint and cur_ctc_accum_weight is not None:
                ctc_accum_weights = torch.cat(
                    [ctc_accum_weights, cur_ctc_accum_weight], -1
                )  # B x T

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
            return [data[partitions == part_id] for part_id in range(num_partitions)]

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
                [-1 * torch.ones([1], dtype=torch.int32).cuda(), cur_fired_indices],
                dim=-1,
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
        # cif_out_padding_mask shape = [batch_size, fired_max_length], where locations with value 0 is False.

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

        ctc_align_outputs = None
        if self.use_ctc_constraint and ctc_accum_weights is not None:
            org_ctc_align_outputs = ctc_accum_weights * ctc_border_marks  # B x T_a
            ctc_align_max_len = ctc_border_marks.size(1)
            ctc_align_outputs = (
                torch.zeros([0, ctc_align_max_len])
                .type_as(org_ctc_align_outputs)
                .cuda()
            )
            for k in range(batch_size):
                cur_border_marks = ctc_border_marks[k, :]  # T
                cur_borders_num = cur_border_marks.sum()  # 1
                cur_ctc_accum_weight = ctc_accum_weights[k, :]  # T_a
                compressed_ctc_weight = cur_ctc_accum_weight[
                    cur_border_marks.float() != 0.0
                ]
                pad_length = ctc_align_max_len - cur_borders_num  # get padding length
                padded_compressed_ctc_weight = torch.cat(
                    [
                        compressed_ctc_weight,
                        torch.full([pad_length], 0.0)
                        .type_as(compressed_ctc_weight)
                        .cuda(),
                    ],
                    dim=0,
                ).unsqueeze(
                    0
                )  # 1 x T
                ctc_align_outputs = torch.cat(
                    [ctc_align_outputs, padded_compressed_ctc_weight], dim=0
                )  # B x T

        return {
            "cif_out": cif_outputs,  # shape = [batch_size, fired_max_length, cif_output_dim]
            "cif_out_padding_mask": cif_out_padding_mask,  # shape = [batch_size, fired_max_length]
            "ctxt_cif_out": ctxt_cif_outputs,  # shape = [batch_size, fired_max_length, cif_ctxt_embed_dim]
            "quantity_out": quantity_out,  # shape = [batch_size]
            "cif_durations": cif_durations,  # shape = [batch_size, fired_max_length]
            "ctc_align_outputs": ctc_align_outputs,  # B x T
        }


# Confidence Estimation Module
class UncertaintyEstimationModule(nn.Module):
    def __init__(self, args, dict):
        super().__init__()
        self._uem_input_state = args.uem_input_state.strip().split(",")
        self.vsz = len(dict)
        self.expand_size = 1 + args.K_corr_samp

        # Determine the input dim for UEM
        uem_input_dim = 0
        for state in args.uem_input_state.strip().split(","):
            if state == "cif_outputs":
                uem_input_dim += args.cif_embedding_dim
            elif state == "decoder_states":
                uem_input_dim += args.decoder_embed_dim
            elif state == "logits":
                uem_input_dim += self.vsz
            elif state == "pred_embeds":
                uem_input_dim += args.decoder_embed_dim
            else:
                raise NotImplementedError("Unknown input type: %s" % state)

        # Build modules for UEM
        self.dropout_module = FairseqDropout(
            args.corr_dropout,
            module_name=module_name_fordropout(self.__class__.__name__),
        )
        self.use_uem_bn_layer = args.use_uem_bn_layer
        if self.use_uem_bn_layer:
            self.uem_bottleneck_proj = nn.Linear(uem_input_dim, args.uem_bn_proj_dim)
            self.uem_pred_proj = nn.Linear(args.uem_bn_proj_dim, 1)
        else:
            self.uem_pred_proj = nn.Linear(uem_input_dim, 1)

    def forward(
        self,
        cif_outputs,
        decoder_states,
        logits=None,
        pred_embeds=None,
        prev_output_tokens=None,
    ):
        # regulerize lengths
        x = cif_outputs  # B x T x C
        min_reg_len = min(x.size(1), prev_output_tokens.size(1))
        x = x[:, :min_reg_len, :]
        decoder_states = decoder_states[:, :min_reg_len, :]
        logits = logits[:, :min_reg_len, :] if logits is not None else None
        pred_embeds = (
            pred_embeds[:, :, :min_reg_len, :] if pred_embeds is not None else None
        )
        pred_embeds = (
            pred_embeds.view(-1, pred_embeds.size(-2), pred_embeds.size(-1))
            if pred_embeds is not None
            else None
        )  # B x T x C

        # Expand size
        x = expand_tensor_dim(x, expand_size=self.expand_size, reduce=True)
        decoder_states = (
            expand_tensor_dim(
                decoder_states,
                expand_size=self.expand_size,
                reduce=True,
            )
            if decoder_states is not None
            else None
        )
        logits = (
            expand_tensor_dim(logits, expand_size=self.expand_size, reduce=True)
            if logits is not None
            else None
        )

        input_list = [x, decoder_states]
        if logits is not None and "logits" in self._uem_input_state:
            input_list.append(logits)
        if pred_embeds is not None and "pred_embeds" in self._uem_input_state:
            input_list.append(pred_embeds)

        x = torch.cat(input_list, dim=-1)  # B x T x C
        x = self.dropout_module(x)

        # Forward bottleneck layer
        bottleneck_embeds = None
        if self.use_uem_bn_layer:
            x = self.uem_bottleneck_proj(x)
            x = torch.relu(x)
            x = self.dropout_module(x)
            bottleneck_embeds = x  # B x T x C

        x = self.uem_pred_proj(x)  # B x T x 1
        x = x.squeeze(-1)  # B x T

        return x, bottleneck_embeds


# Correction Decoder
class CorrectionDecoder(nn.Module):
    def __init__(self, args, dict):
        super().__init__()
        self.expand_size = 1 + args.K_corr_samp

        # Preparation for transformer masking
        self._future_mask = torch.empty(0)

        # Get original dict
        ori_dict_size = len(dict)
        self.padding_idx = dict.pad()
        self.dict_size = (
            ori_dict_size + 1
        )  # extra one placeholder for no correcting option <no-cor>

        # Get hyper-parameters
        self._cordec_input_state = args.cordec_input_state.strip().split(",")
        self._cordec_output_state = args.cordec_output_state.strip().split(",")

        # build input projection
        input_dim = 0
        for state in self._cordec_input_state:
            if state == "cif_outputs":
                input_dim += args.cif_embedding_dim
            elif state == "decoder_states":
                input_dim += args.decoder_embed_dim
            elif state == "pred_embeds":
                input_dim += args.decoder_embed_dim
            else:
                raise NotImplementedError("Unknown options.")
        self.input_proj = nn.Linear(input_dim, args.decoder_embed_dim)

        # build position embedding
        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                args.decoder_embed_dim,  # decoder embed dim
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        # build input dropout
        self.dropout_module = FairseqDropout(
            args.corr_dropout,
            module_name=module_name_fordropout(self.__class__.__name__),
        )

        # build transformer layers
        args_dict = vars(args)
        args_dict["dropout"] = args_dict["corr_dropout"]
        args_dict["attention_dropout"] = args_dict["corr_attention_dropout"]
        args_dict["activation_dropout"] = args_dict["corr_activation_dropout"]
        args = argparse.Namespace(**args_dict)
        self.cordec_tfm_layers = nn.ModuleList(
            [self.build_transformer_layer(args) for _ in range(args.num_cordec_layers)]
        )

        # build final layer norm
        if args.decoder_normalize_before:
            self.layer_norm = LayerNorm(args.decoder_embed_dim)
        else:
            self.layer_norm = None

        # build final projection
        output_dim = 0
        for state in self._cordec_output_state:
            if state == "cordec_state":
                output_dim += args.cif_embedding_dim
            elif state == "bn_embeds":
                output_dim += args.uem_bn_proj_dim
            else:
                raise NotImplementedError("Unknown options.")
        self.output_proj = nn.Linear(output_dim, args.decoder_embed_dim)
        self.stop_bn_grad = args.stop_bn_grad

        # build prediction projection
        self.pred_proj = nn.Linear(args.decoder_embed_dim, self.dict_size)

    def build_transformer_layer(self, cfg, no_encoder_attn=True):
        layer = transformer_layer.TransformerDecoderLayerBaseDirectArgs(
            cfg, no_encoder_attn=no_encoder_attn
        )
        return layer

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def forward(
        self,
        cif_outputs,
        decoder_states,
        pred_embeds=None,
        uem_bn_embeds=None,
        prev_output_tokens=None,
        incremental_state=None,
    ):
        if uem_bn_embeds is not None and self.stop_bn_grad:
            uem_bn_embeds = uem_bn_embeds.detach()

        x = cif_outputs

        # regularize input lengths
        if prev_output_tokens is not None:
            min_reg_len = min(x.size(1), prev_output_tokens.size(1))
        else:
            min_reg_len = x.size(1)
        x = x[:, :min_reg_len, :]
        decoder_states = decoder_states[:, :min_reg_len, :]
        prev_output_tokens = (
            prev_output_tokens[:, :min_reg_len]
            if prev_output_tokens is not None
            else None
        )
        pred_embeds = (
            pred_embeds[:, :, :min_reg_len, :] if pred_embeds is not None else None
        )
        pred_embeds = (
            pred_embeds.view(-1, pred_embeds.size(-2), pred_embeds.size(-1))
            if pred_embeds is not None
            else None
        )  # (B x (1 + K)) x T x C
        uem_bn_embeds = (
            uem_bn_embeds[:, :min_reg_len, :] if uem_bn_embeds is not None else None
        )  # (B x (1 + K)) x T x C

        # Expand size
        x = expand_tensor_dim(x, expand_size=self.expand_size, reduce=True)
        decoder_states = (
            expand_tensor_dim(
                decoder_states,
                expand_size=self.expand_size,
                reduce=True,
            )
            if decoder_states is not None
            else None
        )
        prev_output_tokens = (
            expand_tensor_dim(
                prev_output_tokens, expand_size=self.expand_size, reduce=True
            )
            if prev_output_tokens is not None
            else None
        )

        # Get inputs for correction decoder
        input_list = [x, decoder_states]
        if "pred_embeds" in self._cordec_input_state and pred_embeds is not None:
            input_list.append(pred_embeds)
        x = torch.cat(input_list, dim=-1)  # B x T x C
        x = self.input_proj(x) if self.input_proj is not None else x  # B x T x C

        # Embed positions & dropout
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
        x += positions
        x = self.dropout_module(x)
        x = x.transpose(0, 1)  # T x B x C

        # Prepare future mask
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.cordec_tfm_layers):
            self_attn_mask = self.buffered_future_mask(x)  # T x T
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)  # B x T
            x, _, _ = layer(
                x,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            inner_states.append(x)

        x = x.transpose(0, 1)  # B x T x C
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.output_proj is not None:
            if uem_bn_embeds is not None and "bn_embeds" in self._cordec_output_state:
                x = torch.cat([x, uem_bn_embeds], dim=-1)
            x = self.output_proj(x)
            x = torch.relu(x)

        if self.pred_proj is not None:
            x = self.pred_proj(x)  # logits, B x T x V

        return x, inner_states


# Preparation Module
class PrepareModule(nn.Module):
    def __init__(self, args, dict):
        super().__init__()

        # dictionary settings
        self.dict = dict
        self.padding_id = dict.pad()
        self.no_correction_id = len(dict)  # max_id + 1

        # Settings about generating labels for correction modules
        self.corr_tgt_type = args.corr_tgt_type
        self.K_corr_samp = args.K_corr_samp

    def forward(self, encoder_out, decoder_out, prev_output_tokens=None, targets=None):
        decoder_logits = decoder_out[0]  # B x T x V
        decoder_states = decoder_out[-1]  # B x T x C
        cif_outputs = encoder_out["encoder_out"][0]  # B x T x C
        cif_padding_mask = encoder_out["encoder_padding_mask"][0]  # B x T

        # Regularize length
        cif_len = cif_padding_mask.size(-1)
        tgt_len = prev_output_tokens.size(-1)
        reg_min_len = min(cif_len, tgt_len)
        decoder_logits = decoder_logits[:, :reg_min_len, :]
        decoder_states = decoder_states[:, :reg_min_len, :]
        cif_outputs = cif_outputs[:, :reg_min_len, :]
        cif_padding_mask = cif_padding_mask[:, :reg_min_len]

        # Prepare labels for UEM and Correction Decoder
        uem_labels, cordec_labels = None, None
        uem_padding_mask, sample_full_preds = None, None
        if targets is not None:
            targets = targets[:, :reg_min_len]  # B x T
            (
                uem_labels_list,
                uem_padding_mask_list,
                cordec_ce_labels_list,
                sample_pred_list,
            ) = ([], [], [], [])

            if "tf-argmax" in self.corr_tgt_type:
                argmax_preds = torch.argmax(decoder_logits, dim=-1)  # B x T
                uem_argmax_labels = (targets != argmax_preds).int()  # B x T
                cordec_ce_labels = targets * uem_argmax_labels  # B x T
                uem_padding_mask = (targets != self.padding_id).int()
                uem_argmax_labels = expand_tensor_dim(
                    uem_argmax_labels, expand_size=1
                )  # B x 1 x T
                cordec_ce_labels = expand_tensor_dim(
                    cordec_ce_labels, expand_size=1
                )  # B x 1 x T
                uem_padding_mask = expand_tensor_dim(
                    uem_padding_mask, expand_size=1
                )  # B x 1 x T
                argmax_preds = expand_tensor_dim(
                    argmax_preds, expand_size=1
                )  # B x 1 x T
                argmax_preds = torch.where(
                    expand_tensor_dim(targets, expand_size=1) != self.padding_id,
                    argmax_preds,
                    expand_tensor_dim(targets, expand_size=1),
                )
                uem_labels_list.append(uem_argmax_labels)
                uem_padding_mask_list.append(uem_padding_mask)
                cordec_ce_labels_list.append(cordec_ce_labels)
                sample_pred_list.append(argmax_preds)

            if "tf-sample" in self.corr_tgt_type:
                bsz, tsz = targets.size()
                _, _, vsz = decoder_logits.size()
                expd_targets = expand_tensor_dim(
                    targets, expand_size=self.K_corr_samp, reduce=False
                )  # B x K x T
                dec_probs = self.get_probs_from_logits(
                    decoder_logits, log_probs=False
                )  # B x T x V
                expd_dec_probs = expand_tensor_dim(
                    dec_probs, expand_size=self.K_corr_samp, reduce=False
                )  # B x K x T x V
                sampled_preds = torch.multinomial(
                    expd_dec_probs.view(-1, vsz),  # (B x K x T) x V
                    num_samples=1,
                    replacement=True,
                )  # (B x K x T) x 1
                sampled_preds = sampled_preds.view(
                    bsz, self.K_corr_samp, tsz
                )  # B x K x T
                sampled_preds = torch.where(
                    expd_targets != self.padding_id,
                    sampled_preds,
                    expd_targets,
                )
                uem_sample_labels = (sampled_preds != expd_targets).int()  # B x K x T
                cordec_ce_labels = expd_targets * uem_sample_labels  # B x K x T
                uem_labels_list.append(uem_sample_labels)
                uem_padding_mask_list.append((expd_targets != self.padding_id).int())
                cordec_ce_labels_list.append(cordec_ce_labels)
                sample_pred_list.append(sampled_preds)

            uem_labels = torch.cat(uem_labels_list, dim=1)  # B x (K + 1) x T
            cordec_labels = torch.cat(cordec_ce_labels_list, dim=1)  # B x (K + 1) x T
            uem_padding_mask = torch.cat(
                uem_padding_mask_list, dim=1
            )  # B x (K + 1) x T
            sample_full_preds = torch.cat(sample_pred_list, dim=1)  # B x (K + 1) x T
            uem_labels = (
                uem_labels * uem_padding_mask
            )  # Set padding label to confidence label

            # TODO: make the zero value in cordec_labels with no_correction_mark
            cordec_labels = torch.where(
                uem_padding_mask.bool(),
                cordec_labels,
                self.padding_id * torch.ones_like(cordec_labels),
            )  # Replace padded locations with pad token
            cordec_labels = torch.where(
                cordec_labels.float() != 0.0,
                cordec_labels,
                self.no_correction_id * torch.ones_like(cordec_labels),
            )  # Replace the zero elements woth no_correction token

        return (
            cif_outputs,
            cif_padding_mask,
            decoder_states,
            decoder_logits,
            uem_labels,
            cordec_labels,
            uem_padding_mask,
            sample_full_preds,
        )

    @staticmethod
    def get_probs_from_logits(logits, log_probs=False):
        """
        Get normalized probabilities (or log probs) from logits.
        """

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)


# Model Main Body
@register_model("s2t_cif_transformer")
class S2TCifTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as down-sample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder, uem=None, cordec=None, ppm=None, args=None):
        # Register encoder and decoder
        super().__init__(encoder, decoder)

        # Register correction module
        self.uem = uem
        self.cordec = cordec
        self.ppm = ppm

        # Register args
        if args is not None:
            self.args = args

        if args.apply_bert_distill:
            logger.info("Use BERT distillation. ")
            self.tokenwise_cif_dis_proj = nn.Linear(
                args.cif_embedding_dim, args.bert_distill_feat_dim
            )
            self.semantic_cif_dis_proj = nn.Linear(
                args.cif_embedding_dim, args.bert_distill_feat_dim
            )
            self.tokenwise_dec_state_proj = nn.Linear(
                args.decoder_embed_dim, args.bert_distill_feat_dim
            )

        # Load initial model from target path
        self.init_model_path = args.load_init_asr_model_from
        if self.training and self.init_model_path:
            logger.info("Load initial model from %s" % self.init_model_path)
            state = torch.load(self.init_model_path, map_location=torch.device("cpu"))
            params_dict = dict()
            for k, v in state["model"].items():
                logging.info(f"{k}")
                params_dict[k] = v
            self.load_state_dict(params_dict, strict=False)

        logging.info(
            "Remeber to convert batch normlization from train mode to eval mode!!!!"
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument(
            "--frontend-type",
            type=str,
            default="conv1d",
            help="the type of frontend acoustic low-level extraction",
        )
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
        parser.add_argument(
            "--conv2d-output-channels",
            type=str,
            help="# of channels in Conv2d subsampling layers",
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
            "--encoder-layers",
            type=int,
            metavar="N",
            help="num encoder layers",
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
            "--decoder-dropout",
            type=float,
            help="decoder dropout probability",
        )
        parser.add_argument(
            "--decoder-attention-dropout",
            type=float,
            help="decoder dropout probability for attention weights",
        )
        parser.add_argument(
            "--decoder-activation-dropout",
            type=float,
            help="decoder dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block,"
            " if true apply ln before each module in each block,"
            " else apply after each residual outputs",
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
        parser.add_argument(
            "--cross-self-attention",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--no-decoder-final-norm",
            type=bool,
        )
        parser.add_argument(
            "--decoder-layerdrop",
            type=float,
        )
        parser.add_argument(
            "--do-encoder-attn",
            action="store_true",
        )
        parser.add_argument(
            "--decoder-enc-attn-kv-type",
            type=str,
        )
        parser.add_argument(
            "--do-decoder-nar",
            action="store_true",  # default false
            help="whether to conduct non-auto-regressive (NAR) decoding for ASR decoder",
        )
        parser.add_argument(
            "--decoder-nar-pad-type",
            type=str,
            help="specify the type of NAR decoder input padding mask, options: triangle, full",
        )
        parser.add_argument(
            "--add-pos-to-cif",
            action="store_true",
            help="whether to add position encoding or embedding to cif inputs of the NAR decoder",
        )

        # Encoder layer down-sampling settings
        parser.add_argument(
            "--layer-downsampling",
            action="store_true",
            help="whether conduct down-sampling between layers",
        )
        parser.add_argument(
            "--pooling-layer-ids",
            type=str,
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
        parser.add_argument("--apply-scaling", type=bool, default=True)
        parser.add_argument(
            "--apply-tail-handling",
            type=bool,
            default=True,
        )
        parser.add_argument(
            "--tail-handling-firing-threshold",
            type=float,
        )
        parser.add_argument(
            "--add-cif-ctxt-layers",
            action="store_true",
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

        # Other settings
        parser.add_argument(
            "--calulate-ctc-logits",
            type=bool,
            default=True,
        )
        parser.add_argument("--use-ctc-constraint", action="store_true")
        parser.add_argument(
            "--ctc-prob-threshold",
            type=float,
            default=0.5,
        )

        # Uncertainty Estimation Module (UEM) settings
        parser.add_argument("--use-uem", action="store_true")  # args.use_uem
        parser.add_argument(
            "--uem-input-state",
            type=str,
            default="cif_outputs,decoder_states,logits",
        )
        parser.add_argument(
            "--use-uem-bn-layer",
            action="store_true",
        )
        parser.add_argument(
            "--uem-bn-proj-dim",
            type=int,
            default=512,
        )

        # Correction Decoder (Cordec) settings
        parser.add_argument("--use-cordec", action="store_true")
        parser.add_argument(
            "--num-cordec-layers",
            type=int,
            default=4,  # could be 4 or 2
        )
        parser.add_argument(
            "--uncertainty-embed-fusion-mode",
            type=str,
            default="top-concat",
        )
        parser.add_argument(
            "--cordec-input-state",
            type=str,
            default="cif_outputs,decoder_states",
        )
        parser.add_argument(
            "--cordec-output-state",
            type=str,
            default="cordec_state,bn_embeds",
        )
        parser.add_argument(
            "--corr-tgt-type",
            type=str,
            default="tf-argmax,tf-sample",
        )
        parser.add_argument(
            "--K-corr-samp",
            type=int,
            default=5,
        )
        parser.add_argument(
            "--freeze-asr-main-body",
            action="store_true",
        )
        parser.add_argument(
            "--load-init-asr-model-from",
            type=str,
            default="",
        )
        parser.add_argument(
            "--corr-dropout",
            type=float,
            metavar="D",
            help="correction module dropout probability",
        )
        parser.add_argument(
            "--corr-attention-dropout",
            type=float,
            metavar="D",
            help="correction module dropout probability for attention weights",
        )
        parser.add_argument(
            "--corr-activation-dropout",
            "--corr-relu-dropout",
            type=float,
            metavar="D",
            help="correction module dropout probability " "after activation in FFN.",
        )
        parser.add_argument("--stop-bn-grad", action="store_true")
        parser.add_argument(
            "--fetch-decoder-states-from", type=str, default="tfm_outputs"
        )
        parser.add_argument(
            "--encoder-attn-type",
            type=str,
            default="normal",
        )

        # Conformer encoder settings
        parser.add_argument("--apply-conformer-encoder", action="store_true")
        parser.add_argument(
            "--conformer-attn-type",
            type=str,
            default="espnet",
        )
        parser.add_argument(
            "--conformer-pos-enc-type",
            type=str,
            default="rel_pos",
        )
        parser.add_argument(
            "--conformer-depthwise-conv-kernel-size",
            type=int,
            default=15,
        )

        # BERT Distillation Settings
        parser.add_argument(
            "--apply-bert-distill",
            action="store_true",
        )
        parser.add_argument(
            "--use-contextualized-cif-feats-for-distill",
            action="store_true",
        )
        parser.add_argument(
            "--bert-distill-feat-dim",
            type=int,
            default=768,
        )

    @classmethod
    def build_encoder(cls, args, task):
        if args.layer_downsampling:
            if args.apply_conformer_encoder:
                encoder = S2TCifConformerLayerPoolingEncoder(args, task)
            else:
                encoder = S2TCifTransformerLayerPoolingEncoder(args, task)
        else:
            encoder = S2TCifTransformerEncoder(args, task)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return CifArTransformerDecoder(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_uncertainty_estimation_module(cls, args, task):
        return UncertaintyEstimationModule(args, task.target_dictionary)

    @classmethod
    def build_correction_decoder(cls, args, task):
        return CorrectionDecoder(args, task.target_dictionary)

    @classmethod
    def build_prepare_module(cls, args, task):
        return PrepareModule(args, task.target_dictionary)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        # Main body
        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)

        # Extra modules
        ppm, uem, cordec = None, None, None
        if args.use_uem or args.use_cordec:
            ppm = cls.build_prepare_module(args, task=task)
        if args.use_uem:
            uem = cls.build_uncertainty_estimation_module(args, task=task)
        if args.use_cordec:
            cordec = cls.build_correction_decoder(args, task=task)

        return cls(encoder, decoder, uem=uem, cordec=cordec, ppm=ppm, args=args)

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

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, target_lengths, **kwargs
    ):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """

        if self.args.freeze_asr_main_body:
            with torch.no_grad():
                encoder_out = self.encoder(
                    src_tokens=src_tokens,
                    src_lengths=src_lengths,
                    target_lengths=target_lengths,
                )
                decoder_out = self.decoder(
                    prev_output_tokens=prev_output_tokens,
                    encoder_out=encoder_out,
                )
        else:
            encoder_out = self.encoder(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                target_lengths=target_lengths,
            )
            decoder_out = self.decoder(
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )

        cordec_labels, cordec_logits = None, None
        uem_labels, uem_logits, uem_padding_mask, cordec_full_labels = (
            None,
            None,
            None,
            None,
        )
        if self.ppm is not None:
            target = (
                None if "target" not in kwargs.keys() else kwargs["target"]
            )  # B x T
            (
                cif_outputs,
                cif_padding_mask,
                decoder_states,
                decoder_logits,
                uem_labels,
                cordec_labels,
                uem_padding_mask,
                sampled_full_labels,
            ) = self.ppm(
                encoder_out=encoder_out,
                decoder_out=decoder_out,
                prev_output_tokens=prev_output_tokens,
                targets=target,
            )

            uem_bn_embeds = None
            pred_embeds = self.decoder.embed_tokens(
                sampled_full_labels
            ).detach()  # B x (1 + K) x T x C
            if self.uem is not None:
                uem_logits, uem_bn_embeds = self.uem(
                    cif_outputs=cif_outputs,
                    decoder_states=decoder_states,
                    logits=decoder_logits,
                    pred_embeds=pred_embeds,
                    prev_output_tokens=prev_output_tokens,
                )

            if self.cordec is not None:
                cordec_logits, _ = self.cordec(
                    cif_outputs=cif_outputs,
                    decoder_states=decoder_states,
                    pred_embeds=pred_embeds,
                    uem_bn_embeds=uem_bn_embeds,
                    prev_output_tokens=prev_output_tokens,
                )

        token_distill_cif_feat = None
        semantic_distill_cif_feat = None
        token_distill_decoder_states = None
        if self.args.apply_bert_distill:
            # obtain raw cif features / contextualized cif features
            if self.args.use_contextualized_cif_feats_for_distill:
                cif_outputs_for_distill = encoder_out["ctxt_cif_out"][0]
            else:
                cif_outputs_for_distill = encoder_out["encoder_out"][0]
            cif_padding_for_distill = encoder_out["encoder_padding_mask"][0]
            cif_outputs_for_distill = (
                cif_outputs_for_distill * cif_padding_for_distill.unsqueeze(-1)
            )  # B x T x C

            # process tokenwise acoustic cif feats
            token_distill_cif_feat = cif_outputs_for_distill  # B x T x C
            token_distill_cif_feat = self.tokenwise_cif_dis_proj(
                token_distill_cif_feat
            )  # B x T x C_bert

            # process semantic acoustic cif feats
            cif_lengths = (
                cif_padding_for_distill.int().sum(-1).type_as(cif_outputs_for_distill)
            )
            cif_length_scale = torch.reciprocal(cif_lengths).type_as(
                cif_outputs_for_distill
            )  # B
            semantic_distill_cif_feat = cif_outputs_for_distill.sum(1)  # B x C
            semantic_distill_cif_feat = (
                semantic_distill_cif_feat * cif_length_scale.unsqueeze(-1)
            )  # B x C
            semantic_distill_cif_feat = self.semantic_cif_dis_proj(
                semantic_distill_cif_feat
            )  # B x C_bert

            # process decoder states for bert distillation
            token_distill_decoder_states = decoder_out[-1]  # B x T x C
            token_distill_decoder_states = self.tokenwise_dec_state_proj(
                token_distill_decoder_states
            )

        # return decoder_out
        final_outputs = {
            # Encoder part outputs
            "encoder_padding_mask": encoder_out["raw_encoder_padding_mask"][0],  # B x T
            "ctc_logits": encoder_out["ctc_logits"][0].transpose(0, 1),  # B x T x V
            # Cif module outputs
            "quantity_out": encoder_out["quantity_out"][
                0
            ],  # Quantity out for quantity loss calculation
            "ctc_align_outputs": encoder_out["ctc_align_outputs"][0]
            if encoder_out["ctc_align_outputs"]
            else None,  # B x T
            "cif_out": encoder_out["encoder_out"][
                0
            ],  # CIF out for decoder prediction, B x T x C
            "cif_out_padding_mask": encoder_out["encoder_padding_mask"][0],  # B x T
            # Decoder part outputs
            "decoder_out": decoder_out,  # Decoder outputs (which is final logits for ce calculation), B x T x V
            # UEM & Cordec outputs
            "uem_logits": uem_logits,
            "uem_labels": uem_labels,
            "cordec_logits": cordec_logits,
            "cordec_labels": cordec_labels,
            "cordec_full_labels": cordec_full_labels,
            "uem_padding_mask": uem_padding_mask,  # B x T
            # BERT distillation outputs
            "token_distill_cif_feat": token_distill_cif_feat,
            "semantic_distill_cif_feat": semantic_distill_cif_feat,
            "token_distill_decoder_states": token_distill_decoder_states,
        }

        return final_outputs

    def get_cif_output(self, src_tokens, src_lengths, target_lengths=None):
        with torch.no_grad():
            encoder_out = self.encoder(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                target_lengths=target_lengths,
            )

        return {
            # Cif outputs
            "cif_out": encoder_out["encoder_out"][0],  # B x T x C
            "cif_out_padding_mask": encoder_out["encoder_padding_mask"][0],  # B x T
            "cif_durations": encoder_out["cif_durations"][0],
            # Raw encoder acoustic outputs
            "encoder_out": encoder_out["raw_encoder_out"][0],  # T x B x C
            "encoder_padding_mask": encoder_out["raw_encoder_padding_mask"][0],  # B x T
        }

    def step_forward_decoder(
        self, prev_decoded_tokens, cif_outputs, incremental_state=None
    ):
        for k, v in cif_outputs.items():
            if cif_outputs[k] is not None:
                cif_outputs[k] = [v]
            else:
                cif_outputs[k] = None

        cif_outputs["encoder_out"] = cif_outputs["cif_out"]
        cif_outputs["encoder_padding_mask"] = cif_outputs["cif_out_padding_mask"]
        cif_outputs["raw_encoder_out"] = cif_outputs["raw_encoder_out"]
        cif_outputs["raw_encoder_padding_mask"] = cif_outputs[
            "raw_encoder_padding_mask"
        ]

        with torch.no_grad():
            decoder_out = self.decoder(
                prev_output_tokens=prev_decoded_tokens,
                encoder_out=cif_outputs,
                incremental_state=incremental_state,
            )

        return decoder_out

    def forward_uem(
        self, cif_outputs, decoder_states, decoder_logits, prev_output_tokens
    ):
        return self.uem(
            cif_outputs=cif_outputs,
            decoder_states=decoder_states,
            logits=decoder_logits,
            prev_output_tokens=prev_output_tokens,
        )

    def forward_cordec(
        self, cif_outputs, decoder_states, uem_bn_embeds, prev_output_tokens
    ):
        return self.cordec(
            cif_outputs=cif_outputs,
            decoder_states=decoder_states,
            uem_bn_embeds=uem_bn_embeds,
            prev_output_tokens=prev_output_tokens,
        )


class S2TCifTransformerEncoder(FairseqEncoder):
    """
    Speech-to-text Transformer encoder that consists of
    input convolutional subsampler & Transformer-based encoder.
    """

    def __init__(self, args, task):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = task.target_dictionary.pad()

        if args.frontend_type == "conv2d":
            # Conv2d downsampling is borrowed from espnet
            conv_output_channels = [
                int(x) for x in args.conv2d_output_channels.split(",")
            ]
            kernel_sizes = [int(k) for k in args.conv_kernel_sizes.split(",")]
            self.subsample = Conv2dSubsampler(
                idim=args.input_feat_per_channel,
                odim=args.encoder_embed_dim,
                conv_output_channels=conv_output_channels,
                kernel_sizes=kernel_sizes,
            )
        else:
            # Conv1d downsampling is proposed in (https://arxiv.org/abs/1911.08460)
            self.subsample = Conv1dSubsampler(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )
            # self.downsample_rate = 1.0 / (2 ** len(args.conv_kernel_sizes.split(",")))

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        # build cif module
        self.use_ctc_constraint = args.use_ctc_constraint
        self.ctc_prob_threshold = args.ctc_prob_threshold
        self.cif = CtcConstrainedCifMiddleware(args)

        # build ctc projection
        self.ctc_proj = None
        if args.calulate_ctc_logits:
            self.ctc_proj = Linear(
                args.encoder_embed_dim, len(task.target_dictionary)
            ).cuda()

    def _forward(
        self, src_tokens, src_lengths, target_lengths=None, return_all_hiddens=False
    ):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        encoder_states = []
        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        ctc_logits = None
        if self.ctc_proj is not None:
            ctc_logits = self.ctc_proj(x)  # T x B x C

        encoder_outputs = {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "conv_lengths": [input_lengths],
            "ctc_logits": [ctc_logits] if ctc_logits is not None else [],  # T x B x C
        }

        if self.use_ctc_constraint:
            cif_out = self.cif(
                encoder_outputs=encoder_outputs,
                target_lengths=target_lengths if self.training else None,
                input_lengths=input_lengths,
                ctc_logits=ctc_logits,
            )
        else:
            cif_out = self.cif(
                encoder_outputs=encoder_outputs,
                target_lengths=target_lengths if self.training else None,
                input_lengths=input_lengths,
            )

        encoder_outputs["raw_encoder_out"] = [x]
        encoder_outputs["raw_encoder_padding_mask"] = [encoder_padding_mask]
        encoder_outputs["encoder_out"] = [cif_out["cif_out"]]  # B x T x C
        encoder_outputs["encoder_padding_mask"] = [
            cif_out["cif_out_padding_mask"].bool()
        ]  # B x T
        # encoder_outputs["encoder_padding_mask"] = [~cif_out["cif_out_padding_mask"].bool()]
        encoder_outputs["quantity_out"] = [cif_out["quantity_out"]]
        encoder_outputs["cif_durations"] = [cif_out["cif_durations"]]
        encoder_outputs["ctc_align_outputs"] = (
            [cif_out["ctc_align_outputs"]] if self.use_ctc_constraint else None
        )

        return encoder_outputs

    def forward(
        self, src_tokens, src_lengths, target_lengths=None, return_all_hiddens=False
    ):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens,
                    src_lengths,
                    target_lengths,
                    return_all_hiddens=return_all_hiddens,
                )
        else:
            x = self._forward(
                src_tokens,
                src_lengths,
                target_lengths,
                return_all_hiddens=return_all_hiddens,
            )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class S2TCifTransformerLayerPoolingEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        if args.frontend_type == "conv2d":
            # Conv2d downsampling is borrowed from espnet
            conv_output_channels = [
                int(x) for x in args.conv2d_output_channels.split(",")
            ]
            kernel_sizes = [int(k) for k in args.conv_kernel_sizes.split(",")]
            self.subsample = Conv2dSubsampler(
                idim=args.input_feat_per_channel,
                odim=args.encoder_embed_dim,
                conv_output_channels=conv_output_channels,
                kernel_sizes=kernel_sizes,
            )
        else:
            # Conv1d downsampling is proposed in (https://arxiv.org/abs/1911.08460)
            self.subsample = Conv1dSubsampler(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )
            # self.downsample_rate = 1.0 / (2 ** len(args.conv_kernel_sizes.split(",")))

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            args.encoder_embed_dim,
            self.padding_idx,
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        # build cif module
        self.use_ctc_constraint = args.use_ctc_constraint
        self.ctc_prob_threshold = args.ctc_prob_threshold
        if not self.use_ctc_constraint:
            self.cif = CifMiddleware(args)
        else:
            self.cif = CtcConstrainedCifMiddleware(args)

        # build ctc projection
        self.ctc_proj = None
        if args.calulate_ctc_logits:
            self.ctc_proj = Linear(
                args.encoder_embed_dim, len(task.target_dictionary)
            ).cuda()

        # Layer Pooling settings
        self.layer_downsampling = args.layer_downsampling
        self.pooling_layer_ids = (
            [int(num) for num in args.pooling_layer_ids.split(",")]
            if self.layer_downsampling
            else None
        )
        self.pooling_layer = (
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
            if self.layer_downsampling
            else None
        )

    def _forward(
        self, src_tokens, src_lengths, target_lengths=None, return_all_hiddens=False
    ):
        # Convolutional Subsampler
        x, input_lengths = self.subsample(src_tokens, src_lengths)

        # Transformer input preparation
        x = self.embed_scale * x
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        encoder_states = []
        for layer_id, layer in enumerate(self.transformer_layers):
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)
            if self.layer_downsampling:
                if (layer_id + 1) in self.pooling_layer_ids:
                    x = x.transpose(0, 1).unsqueeze(
                        dim=1
                    )  # N (B) x C (1) x H (T) x W (D)
                    x = self.pooling_layer(x)  # N (B) x C (1) x H(T)/2 x W (D)
                    x = x.squeeze(dim=1).transpose(0, 1)

                    encoder_padding_mask = (
                        (~encoder_padding_mask)
                        .float()
                        .unsqueeze(dim=-1)
                        .unsqueeze(dim=1)
                    )
                    encoder_padding_mask = (
                        self.pooling_layer(encoder_padding_mask)
                        .squeeze(dim=1)
                        .squeeze(dim=-1)
                    )
                    encoder_padding_mask = ~encoder_padding_mask.bool()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        ctc_logits = None
        if self.ctc_proj is not None:
            ctc_logits = self.ctc_proj(x)

        encoder_outputs = {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "conv_lengths": [input_lengths],
            "ctc_logits": [ctc_logits] if ctc_logits is not None else [],  # T x B x C
        }

        if self.use_ctc_constraint:
            cif_out = self.cif(
                encoder_outputs=encoder_outputs,
                target_lengths=target_lengths if self.training else None,
                input_lengths=input_lengths,
                ctc_logits=ctc_logits,
            )
        else:
            cif_out = self.cif(
                encoder_outputs=encoder_outputs,
                target_lengths=target_lengths if self.training else None,
                input_lengths=input_lengths,
            )

        encoder_outputs["raw_encoder_out"] = [x]
        encoder_outputs["raw_encoder_padding_mask"] = [encoder_padding_mask]
        encoder_outputs["encoder_out"] = [cif_out["cif_out"]]  # B x T x C
        encoder_outputs["encoder_padding_mask"] = [
            cif_out["cif_out_padding_mask"].bool()
        ]  # B x T
        encoder_outputs["quantity_out"] = [cif_out["quantity_out"]]
        encoder_outputs["cif_durations"] = [cif_out["cif_durations"]]
        encoder_outputs["ctc_align_outputs"] = (
            [cif_out["ctc_align_outputs"]] if self.use_ctc_constraint else None
        )

        return encoder_outputs

    def forward(
        self, src_tokens, src_lengths, target_lengths=None, return_all_hiddens=False
    ):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens,
                    src_lengths,
                    target_lengths,
                    return_all_hiddens=return_all_hiddens,
                )
        else:
            x = self._forward(
                src_tokens,
                src_lengths,
                target_lengths,
                return_all_hiddens=return_all_hiddens,
            )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class S2TCifConformerLayerPoolingEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        if args.frontend_type == "conv2d":
            # Conv2d downsampling is borrowed from espnet
            conv_output_channels = [
                int(x) for x in args.conv2d_output_channels.split(",")
            ]
            kernel_sizes = [int(k) for k in args.conv_kernel_sizes.split(",")]
            self.subsample = Conv2dSubsampler(
                idim=args.input_feat_per_channel,
                odim=args.encoder_embed_dim,
                conv_output_channels=conv_output_channels,
                kernel_sizes=kernel_sizes,
            )
        else:
            # Conv1d downsampling is proposed in (https://arxiv.org/abs/1911.08460)
            self.subsample = Conv1dSubsampler(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )
            # self.downsample_rate = 1.0 / (2 ** len(args.conv_kernel_sizes.split(",")))

        self.pos_enc_type = args.conformer_pos_enc_type
        if self.pos_enc_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(
                args.max_source_positions, args.encoder_embed_dim
            )
        else:
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, args.encoder_embed_dim, self.padding_idx
            )

        self.linear = torch.nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.dropout = torch.nn.Dropout(args.dropout)
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    embed_dim=args.encoder_embed_dim,
                    ffn_embed_dim=args.encoder_ffn_embed_dim,
                    attention_heads=args.encoder_attention_heads,
                    dropout=args.dropout,
                    depthwise_conv_kernel_size=args.conformer_depthwise_conv_kernel_size,
                    attn_type=args.conformer_attn_type,
                    pos_enc_type=args.conformer_pos_enc_type,
                    use_fp16=args.fp16,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        # build cif module
        self.use_ctc_constraint = args.use_ctc_constraint
        self.ctc_prob_threshold = args.ctc_prob_threshold
        if not self.use_ctc_constraint:
            self.cif = CifMiddleware(args)
        else:
            self.cif = CtcConstrainedCifMiddleware(args)

        # build ctc projection
        self.ctc_proj = None
        if args.calulate_ctc_logits:
            self.ctc_proj = Linear(
                args.encoder_embed_dim, len(task.target_dictionary)
            ).cuda()

        # Layer Pooling settings
        self.layer_downsampling = args.layer_downsampling
        self.pooling_layer_ids = (
            [int(num) for num in args.pooling_layer_ids.split(",")]
            if self.layer_downsampling
            else None
        )
        self.pooling_layer = (
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
            if self.layer_downsampling
            else None
        )

    def _forward(
        self, src_tokens, src_lengths, target_lengths=None, return_all_hiddens=False
    ):
        if torch.isnan(src_tokens).sum() != 0:
            print("src failure!!!")
            print(x)

        # Convolutional Subsampler
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)

        if torch.isnan(x).sum() != 0:
            print("conv failure!!!")
            print(x)

        # Prepare inputs for Conformer layers
        orig_x = x
        if torch.isnan(x).sum() != 0:
            print("TFM inputs orig_x linear failure!!!")
        x = self.embed_scale * x
        if torch.isnan(x).sum() != 0:
            print("TFM inputs after scaling linear failure!!!")
        if self.pos_enc_type == "rel_pos":
            positions = self.embed_positions(x)
        else:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
            positions = None
        x = self.linear(x)
        if torch.isnan(x).sum() != 0:
            print("TFM inputs after linear failure!!!")
        x = self.dropout_module(x)  # T x B x C

        if torch.isnan(x).sum() != 0:
            print("TFM inputs failure!!!")
            print(orig_x.size())

        # Forward Conformer layers
        encoder_states = []
        for layer_id, layer in enumerate(self.conformer_layers):
            # forward each conformer layer
            x, _ = layer(x, encoder_padding_mask, positions)
            if torch.isnan(x).sum() != 0:
                print("output failure!!! @ %d" % layer_id)
            if return_all_hiddens:
                encoder_states.append(x)

            # if apply layer downsampling
            if self.layer_downsampling:
                if (layer_id + 1) in self.pooling_layer_ids:
                    # Update flowing data x
                    x = x.transpose(0, 1).unsqueeze(
                        dim=1
                    )  # N (B) x C (1) x H (T) x W (D)
                    x = self.pooling_layer(x)  # N (B) x C (1) x H(T)/2 x W (D)
                    x = x.squeeze(dim=1).transpose(0, 1)  # T/2 x B x C

                    # Update padding mask
                    encoder_padding_mask = (
                        (~encoder_padding_mask)
                        .float()
                        .unsqueeze(dim=-1)
                        .unsqueeze(dim=1)
                    )
                    encoder_padding_mask = (
                        self.pooling_layer(encoder_padding_mask)
                        .squeeze(dim=1)
                        .squeeze(dim=-1)
                    )
                    encoder_padding_mask = ~encoder_padding_mask.bool()

                    # Update positions
                    if self.pos_enc_type == "rel_pos":
                        positions = self.embed_positions(x)

        # Forward CTC logits calculation
        ctc_logits = None
        if self.ctc_proj is not None:
            ctc_logits = self.ctc_proj(x)

        encoder_outputs = {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "conv_lengths": [input_lengths],
            "ctc_logits": [ctc_logits] if ctc_logits is not None else [],  # T x B x C
        }

        if self.use_ctc_constraint:
            cif_out = self.cif(
                encoder_outputs=encoder_outputs,
                target_lengths=target_lengths if self.training else None,
                input_lengths=input_lengths,
                ctc_logits=ctc_logits,
            )
        else:
            cif_out = self.cif(
                encoder_outputs=encoder_outputs,
                target_lengths=target_lengths if self.training else None,
                input_lengths=input_lengths,
            )

        if torch.isnan(cif_out["cif_out"]).sum() != 0:
            print("cif failure!!!")

        encoder_outputs["raw_encoder_out"] = [x]
        encoder_outputs["raw_encoder_padding_mask"] = [encoder_padding_mask]
        encoder_outputs["encoder_out"] = [cif_out["cif_out"]]  # B x T x C
        encoder_outputs["encoder_padding_mask"] = [
            cif_out["cif_out_padding_mask"].bool()
        ]  # B x T
        encoder_outputs["quantity_out"] = [cif_out["quantity_out"]]
        encoder_outputs["cif_durations"] = [cif_out["cif_durations"]]
        encoder_outputs["ctxt_cif_out"] = [cif_out["ctxt_cif_out"]]  # B x T x C
        encoder_outputs["ctc_align_outputs"] = (
            [cif_out["ctc_align_outputs"]] if self.use_ctc_constraint else None
        )

        return encoder_outputs

    def forward(
        self, src_tokens, src_lengths, target_lengths=None, return_all_hiddens=False
    ):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens,
                    src_lengths,
                    target_lengths,
                    return_all_hiddens=return_all_hiddens,
                )
        else:
            x = self._forward(
                src_tokens,
                src_lengths,
                target_lengths,
                return_all_hiddens=return_all_hiddens,
            )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class CifArTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        # NAR decoder settings
        self.do_decoder_nar = cfg.do_decoder_nar
        self.decoder_nar_pad_type = cfg.decoder_nar_pad_type
        self.add_pos_to_cif = cfg.add_pos_to_cif

        # Dropout settings
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder_layerdrop

        # Embedding settings
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        # Dimension settings
        self.cif_output_dim = cfg.cif_embedding_dim
        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions
        if self.do_decoder_nar:
            self.embed_tokens = None
        else:
            self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        # Add quantized noise and adaptive inputs
        self.quant_noise = None
        if not cfg.adaptive_input and cfg.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise_pq,
                cfg.quant_noise_pq_block_size,
            )

        # build input module
        if self.do_decoder_nar:
            self.project_in_dim = (
                Linear(self.cif_output_dim, embed_dim, bias=False)
                if embed_dim != self.cif_output_dim
                else None
            )
            self.embed_positions = (
                PositionalEmbedding(
                    self.max_target_positions,
                    embed_dim,  # decoder embed dim
                    self.padding_idx,
                    learned=cfg.decoder_learned_pos,
                )
                if not cfg.no_token_positional_embeddings and self.add_pos_to_cif
                else None
            )
        else:
            self.project_in_dim = (
                Linear((input_embed_dim + self.cif_output_dim), embed_dim, bias=False)
                if embed_dim != (input_embed_dim + self.cif_output_dim)
                else None
            )
            self.embed_positions = (
                PositionalEmbedding(
                    self.max_target_positions,
                    embed_dim,
                    self.padding_idx,
                    learned=cfg.decoder_learned_pos,
                )
                if not cfg.no_token_positional_embeddings
                else None
            )

        # Attention Settings
        self.cross_self_attention = cfg.cross_self_attention
        self.do_encoder_attn = cfg.do_encoder_attn
        self.no_encoder_attn = not self.do_encoder_attn
        self.decoder_enc_attn_kv_type = cfg.decoder_enc_attn_kv_type

        # dropout settings
        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        temp_decoder_cfg = copy.deepcopy(cfg)
        cfg_dict = vars(temp_decoder_cfg)
        cfg_dict["dropout"] = cfg_dict["decoder_dropout"]
        cfg_dict["attention_dropout"] = cfg_dict["decoder_attention_dropout"]
        cfg_dict["activation_dropout"] = cfg_dict["decoder_activation_dropout"]
        temp_decoder_cfg = argparse.Namespace(**cfg_dict)

        # build transformer layers
        self.layers.extend(
            [
                self.build_decoder_layer(temp_decoder_cfg, self.no_encoder_attn)
                for _ in range(cfg.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        # build layernorm
        if cfg.decoder_normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        # build output module
        if self.do_decoder_nar:
            self.project_out_dim = (
                Linear(embed_dim, self.output_embed_dim, bias=False)
                if self.output_embed_dim != embed_dim
                else None
            )
        else:
            self.project_out_dim = (
                Linear(
                    (embed_dim + self.cif_output_dim), self.output_embed_dim, bias=False
                )
                if self.output_embed_dim != (embed_dim + self.cif_output_dim)
                else None
            )
        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(dictionary)

        # Settings about decoder states
        self.fetch_decoder_states_from = cfg.fetch_decoder_states_from

    def build_output_projection(self, dictionary):
        if not self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )  # D x V
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
            )
        else:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )  # D x V
            self.output_projection.weight = self.embed_tokens.weight

    def build_decoder_layer(self, cfg, no_encoder_attn=True):
        layer = transformer_layer.TransformerDecoderLayerBaseDirectArgs(
            cfg, no_encoder_attn=no_encoder_attn
        )
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output logits with shape B x T x V (vocab_size)
                - a dictionary with any model-specific outputs
                - the decoder's output states with shape B x T x C
        """

        x, extra, decoder_states = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)

        return x, extra, decoder_states

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        A scriptable subclass of this class has an extract_features method and calls
        super().extract_features, but super() is not supported in torchscript. A copy of
        this function is made to be used in the subclass instead.
        """

        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # Prepare inputs for encoder-decoder attention
        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if self.do_encoder_attn:
            if self.decoder_enc_attn_kv_type == "cif":
                enc = encoder_out["encoder_out"][0].transpose(
                    0, 1
                )  # Transpose to T x B x C
                padding_mask = ~encoder_out["encoder_padding_mask"][0]  # B x T
            else:
                enc = encoder_out["raw_encoder_out"][0]  # T x B x C
                padding_mask = encoder_out["raw_encoder_padding_mask"][0]  # B x T

        # cif outputs
        cif_outs = encoder_out["encoder_out"][0]
        _, cif_max_len, cif_embed_dim = cif_outs.size()
        min_reg_len = min(cif_max_len, slen)
        shifted_cif_outs = torch.cat(
            [torch.zeros(bs, 1, cif_embed_dim, dtype=cif_outs.dtype).cuda(), cif_outs],
            dim=1,
        )[:, :cif_max_len, :]

        # regularize lengths
        cif_outs = cif_outs[:, :min_reg_len, :].cuda()
        shifted_cif_outs = shifted_cif_outs[:, :min_reg_len, :].cuda()
        prev_output_tokens = prev_output_tokens[:, :min_reg_len].cuda()

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]  # B x T x C

        if incremental_state is not None:
            shifted_cif_outs = shifted_cif_outs[:, -1:, :]
            cif_outs = cif_outs[:, -1:, :]

        # embed tokens and positions
        if self.do_decoder_nar:
            x = cif_outs  # B x T x C
        else:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)  # B x T x C

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if not self.do_decoder_nar:
            x = torch.cat([x, shifted_cif_outs], dim=-1)  # B x T x C

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            # prepare attention mask for transformer layers
            if self.do_decoder_nar:
                if self.decoder_nar_pad_type == "full":
                    self_attn_mask = None
                elif self.decoder_nar_pad_type == "triangle":
                    self_attn_mask = self.buffered_future_mask(x)
                else:
                    self_attn_mask = None
            else:
                if incremental_state is None and not full_context_alignment:
                    self_attn_mask = self.buffered_future_mask(x)
                else:
                    self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        decoder_states = None
        if self.fetch_decoder_states_from == "tfm_outputs":
            decoder_states = x  # B x T x C
        if not self.do_decoder_nar:
            x = torch.cat([x, cif_outs], dim=-1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.fetch_decoder_states_from == "pre_final_output_proj":
            decoder_states = x  # B x T x C

        return x, {"attn": [attn], "inner_states": inner_states}, decoder_states

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

        return x, None


@register_model_architecture(
    model_name="s2t_cif_transformer", arch_name="s2t_cif_transformer"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.frontend_type = getattr(args, "frontend_type", "conv1d")
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
    args.relu_dropout = getattr(args, "relu_dropout", args.dropout)
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
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0.0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)
    args.do_encoder_attn = getattr(args, "do_encoder_attn", False)
    args.decoder_enc_attn_kv_type = getattr(args, "decoder_enc_attn_kv_type", "raw")
    args.do_decoder_nar = getattr(args, "do_decoder_nar", False)
    args.decoder_nar_pad_type = getattr(args, "decoder_nar_pad_type", "full")
    args.add_pos_to_cif = getattr(args, "add_pos_to_cif", False)

    # Encoder layer downsampling settings
    args.layer_downsampling = getattr(args, "layer_downsampling", False)
    args.pooling_layer_ids = getattr(args, "pooling_layer_ids", "4,8")

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
    args.use_ctc_constraint = getattr(args, "use_ctc_constraint", False)
    args.ctc_prob_threshold = getattr(args, "ctc_prob_threshold", 0.5)

    # Correction Module Settings
    args.use_uem = getattr(args, "use_uem", False)
    args.uem_input_state = getattr(
        args, "uem_input_state", "cif_outputs,decoder_states,logits"
    )
    args.use_uem_bn_layer = getattr(args, "use_uem_bn_layer", False)
    args.uem_bn_proj_dim = getattr(args, "uem_bn_proj_dim", 512)
    args.use_cordec = getattr(args, "use_cordec", False)
    args.num_cordec_layers = getattr(args, "num_cordec_layers", 4)
    args.encoder_attn_type = getattr(args, "encoder_attn_type", "normal")
    args.uncertainty_embed_fusion_mode = getattr(
        args, "uncertainty_embed_fusion_mode", "top-concat"
    )
    args.cordec_input_state = getattr(
        args, "cordec_input_state", "cif_outputs,decoder_states"
    )
    args.cordec_output_state = getattr(
        args, "cordec_output_state", "cordec_state,bn_embeds"
    )
    args.corr_tgt_type = getattr(args, "corr_tgt_type", "tf-argmax,tf-sample")
    args.K_corr_samp = getattr(args, "K_corr_samp", 5)
    args.freeze_asr_main_body = getattr(args, "freeze_asr_main_body", False)
    args.load_init_asr_model_from = getattr(args, "load_init_asr_model_from", "")
    args.corr_dropout = getattr(args, "corr_dropout", 0.2)
    args.corr_attention_dropout = getattr(
        args, "corr_attention_dropout", args.corr_dropout
    )
    args.corr_activation_dropout = getattr(
        args, "corr_activation_dropout", args.corr_dropout
    )
    args.stop_bn_grad = getattr(args, "stop_bn_grad", False)
    args.fetch_decoder_states_from = getattr(
        args, "fetch_decoder_states_from", "tfm_outputs"
    )


@register_model_architecture(
    model_name="s2t_cif_transformer", arch_name="s2t_cif_transformer_wide"
)
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)

    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1280)
    args.conv2d_output_channels = getattr(args, "conv2d_output_channels", "128")

    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 640)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2560)
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
    args.relu_dropout = getattr(args, "relu_dropout", args.dropout)
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
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0.0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)
    args.do_encoder_attn = getattr(args, "do_encoder_attn", False)
    args.decoder_enc_attn_kv_type = getattr(args, "decoder_enc_attn_kv_type", "raw")
    args.do_decoder_nar = getattr(args, "do_decoder_nar", False)
    args.decoder_nar_pad_type = getattr(args, "decoder_nar_pad_type", "full")
    args.add_pos_to_cif = getattr(args, "add_pos_to_cif", False)

    # Encoder layer downsampling settings
    args.layer_downsampling = getattr(args, "layer_downsampling", False)
    args.pooling_layer_ids = getattr(args, "pooling_layer_ids", "4,8")

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(
        args, "conv_cif_output_channels_num", 320
    )
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 320)
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
    args.use_ctc_constraint = getattr(args, "use_ctc_constraint", False)
    args.ctc_prob_threshold = getattr(args, "ctc_prob_threshold", 0.5)

    # Correction Module Settings
    args.use_uem = getattr(args, "use_uem", False)
    args.uem_input_state = getattr(
        args, "uem_input_state", "cif_outputs,decoder_states,logits"
    )
    args.use_uem_bn_layer = getattr(args, "use_uem_bn_layer", False)
    args.uem_bn_proj_dim = getattr(args, "uem_bn_proj_dim", 512)
    args.use_cordec = getattr(args, "use_cordec", False)
    args.num_cordec_layers = getattr(args, "num_cordec_layers", 4)
    args.encoder_attn_type = getattr(args, "encoder_attn_type", "normal")
    args.uncertainty_embed_fusion_mode = getattr(
        args, "uncertainty_embed_fusion_mode", "top-concat"
    )
    args.cordec_input_state = getattr(
        args, "cordec_input_state", "cif_outputs,decoder_states"
    )
    args.cordec_output_state = getattr(
        args, "cordec_output_state", "cordec_state,bn_embeds"
    )
    args.corr_tgt_type = getattr(args, "corr_tgt_type", "tf-argmax,tf-sample")
    args.K_corr_samp = getattr(args, "K_corr_samp", 5)
    args.freeze_asr_main_body = getattr(args, "freeze_asr_main_body", False)
    args.load_init_asr_model_from = getattr(args, "load_init_asr_model_from", "")
    args.corr_dropout = getattr(args, "corr_dropout", 0.2)
    args.corr_attention_dropout = getattr(
        args, "corr_attention_dropout", args.corr_dropout
    )
    args.corr_activation_dropout = getattr(
        args, "corr_activation_dropout", args.corr_dropout
    )
    args.stop_bn_grad = getattr(args, "stop_bn_grad", False)
    args.fetch_decoder_states_from = getattr(
        args, "fetch_decoder_states_from", "tfm_outputs"
    )

    # Conformer settings
    args.apply_conformer_encoder = getattr(args, "apply_conformer_encoder", False)
    args.conformer_depthwise_conv_kernel_size = getattr(
        args, "conformer_depthwise_conv_kernel_size", 15
    )
    args.conformer_attn_type = getattr(args, "conformer_attn_type", "espnet")
    args.conformer_pos_enc_type = getattr(args, "conformer_pos_enc_type", "rel_pos")


@register_model_architecture(
    model_name="s2t_cif_transformer", arch_name="s2t_cif_transformer_s"
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
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.relu_dropout = getattr(args, "relu_dropout", args.dropout)
    args.decoder_dropout = getattr(args, "decoder_dropout", args.dropout)
    args.decoder_activation_dropout = getattr(
        args, "decoder_activation_dropout", args.activation_dropout
    )
    args.decoder_attention_dropout = getattr(
        args, "decoder_attention_dropout", args.attention_dropout
    )
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
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0.0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)
    args.do_encoder_attn = getattr(args, "do_encoder_attn", False)
    args.decoder_enc_attn_kv_type = getattr(
        args, "decoder_enc_attn_kv_type", "cif"
    )  # "cif" or "raw"
    args.do_decoder_nar = getattr(args, "do_decoder_nar", False)
    args.decoder_nar_pad_type = getattr(args, "decoder_nar_pad_type", "full")
    args.add_pos_to_cif = getattr(args, "add_pos_to_cif", False)

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

    # Correction Module Settings
    args.use_uem = getattr(args, "use_uem", False)
    args.uem_input_state = getattr(
        args, "uem_input_state", "cif_outputs,decoder_states,logits"
    )
    args.use_uem_bn_layer = getattr(args, "use_uem_bn_layer", False)
    args.uem_bn_proj_dim = getattr(args, "uem_bn_proj_dim", 512)
    args.use_cordec = getattr(args, "use_cordec", False)
    args.num_cordec_layers = getattr(args, "num_cordec_layers", 4)
    args.encoder_attn_type = getattr(args, "encoder_attn_type", "normal")
    args.uncertainty_embed_fusion_mode = getattr(
        args, "uncertainty_embed_fusion_mode", "top-concat"
    )
    args.cordec_input_state = getattr(
        args, "cordec_input_state", "cif_outputs,decoder_states"
    )
    args.cordec_output_state = getattr(
        args, "cordec_output_state", "cordec_state,bn_embeds"
    )
    args.corr_tgt_type = getattr(args, "corr_tgt_type", "tf-argmax,tf-sample")
    args.K_corr_samp = getattr(args, "K_corr_samp", 5)
    args.freeze_asr_main_body = getattr(args, "freeze_asr_main_body", False)
    args.load_init_asr_model_from = getattr(args, "load_init_asr_model_from", "")
    args.corr_dropout = getattr(args, "corr_dropout", 0.2)
    args.corr_attention_dropout = getattr(
        args, "corr_attention_dropout", args.corr_dropout
    )
    args.corr_activation_dropout = getattr(
        args, "corr_activation_dropout", args.corr_dropout
    )
    args.stop_bn_grad = getattr(args, "stop_bn_grad", False)
    args.fetch_decoder_states_from = getattr(
        args, "fetch_decoder_states_from", "tfm_outputs"
    )

    # Conformer settings
    args.apply_conformer_encoder = getattr(args, "apply_conformer_encoder", False)
    args.conformer_depthwise_conv_kernel_size = getattr(
        args, "conformer_depthwise_conv_kernel_size", 15
    )
    args.conformer_attn_type = getattr(args, "conformer_attn_type", "espnet")
    args.conformer_pos_enc_type = getattr(args, "conformer_pos_enc_type", "rel_pos")

    # BERT Distillation Settings
    args.bert_distill_feat_dim = getattr(args, "bert_distill_feat_dim", 768)
    args.apply_bert_distill = getattr(args, "apply_bert_distill", False)
    args.use_contextualized_cif_feats_for_distill = getattr(
        args, "use_contextualized_cif_feats_for_distill", False
    )
