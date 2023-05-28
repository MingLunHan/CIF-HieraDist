# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter


@with_incremental_state
class CustomizedEncDecMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=True,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        # projection layers for each kind of inputs
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        # projection layer for concatenated outputs
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        # initialize/reset parameters
        self.reset_parameters()
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        query: Time x Batch x Channel
        key: Time x N x B x Channel
        """
        if need_head_weights:
            need_head_weights = True

        # force some inputs to be None,
        # cause this is encoder decoder attention
        incremental_state = None
        attn_mask = None

        tgt_len, bsz, embed_dim = query.size()  # query with shape T x B x C
        tgt_len, src_len, key_bsz, _ = key.size()  # key with shape T x N x B x C

        # encoder-decoder attention
        if self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        # TIPS:
        # At this step, the dimension of each input tensor is:
        # query = (T x B x C), keys = values = (T x N x B x C)

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
            .unsqueeze(2)
        )  # (B x num_heads) x T x 1 x (C / num_heads)
        if k is not None:
            k = (
                k.contiguous()
                .view(tgt_len, src_len, bsz * self.num_heads, self.head_dim)
                .permute([2, 0, 1, 3])
            )  # (B x num_heads) x T x N x (C / num_heads)
        if v is not None:
            v = (
                v.contiguous()
                .view(tgt_len, src_len, bsz * self.num_heads, self.head_dim)
                .permute([2, 0, 1, 3])
            )  # (B x num_heads) x T x N x (C / num_heads)

        squ_q = q.contiguous().view(
            (bsz * self.num_heads * tgt_len), -1, self.head_dim
        )  # (B x N_h x T) x 1 x head_dim
        squ_k = k.contiguous().view(
            (bsz * self.num_heads * tgt_len), -1, self.head_dim
        )  # (B x N_h x T) x N x head_dim
        attn_weights = torch.bmm(squ_q, squ_k.transpose(1, 2))  # (B x N_h x T) x 1 x N
        attn_weights = attn_weights.view(
            (bsz * self.num_heads),
            tgt_len,
            1,
            src_len,
        )  # (B x N_h) x T x 1 x N

        attn_weights_float = utils.softmax(
            attn_weights.float(), dim=-1, onnx_trace=self.onnx_trace
        )  # (B x N_h) x T x 1 x N
        attn_weights = attn_weights_float.type_as(attn_weights)  # (B x N_h) x T x 1 x N
        attn_probs = self.dropout_module(attn_weights)  # (B x N_h) x T x 1 x N

        # attn_probs = (B x N_h) x T x 1 x N， v = (B x N_h) x T x N x (C / N_h)
        attn_probs = attn_probs.view(
            (bsz * self.num_heads * tgt_len), 1, src_len
        )  # (B x N_h x T) x 1 x N
        squ_v = v.contiguous().view(
            (bsz * self.num_heads * tgt_len), src_len, self.head_dim
        )
        attn = torch.bmm(attn_probs, squ_v)  # (B x N_h x T) x 1 x head_dim
        attn = attn.view((bsz * self.num_heads), tgt_len, 1, self.head_dim,).squeeze(
            2
        )  # (B x N_h) x T x head_dim

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)  # T x B x C

        # prepare attention heads
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
