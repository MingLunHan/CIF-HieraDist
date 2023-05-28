# @Time    : 2021/7/14
# @Author  : Minglun Han
# @File    : cif_decoder.py

"""""
    Update:
    By 2022/06/19
        1. support LM decoding with language model by shallow fusion;
""" ""

import os
import sys
import torch
import logging
import numpy as np
import itertools as it
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils

np.set_printoptions(threshold=10000000)
torch.set_printoptions(profile="full")


class CifDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest
        self.beam = args.beam

        self.tail_handling_firing_threshold = args.tail_handling_firing_threshold

        # Obtain ids of special tokens
        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        self.bos = tgt_dict.bos()
        self.eos = tgt_dict.eos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()

        self.cif_decoder_mode = args.cif_decoder_mode
        self.use_nnlm = args.use_nnlm
        self.fetch_nnlm_from = args.fetch_nnlm_from
        self.lm_weight = args.lm_weight
        self.specified_dict_path = args.specified_dict_path

        # Load language model
        self.lm_decoder = None
        if self.use_nnlm:
            logging.info("load language model from %s" % self.fetch_nnlm_from)
            state = checkpoint_utils.load_checkpoint_to_cpu(self.fetch_nnlm_from)

            # build task
            cfg = None
            if "args" in state and state["args"] is not None:
                cfg = convert_namespace_to_omegaconf(state["args"])
            elif "cfg" in state and state["cfg"] is not None:
                cfg = state["cfg"]
            assert cfg is not None, "Configuration is None"
            cfg.task.data = self.specified_dict_path
            task = tasks.setup_task(cfg.task)

            if "task_state" in state:
                task.load_state_dict(state["task_state"])

            # build model & load model parameters
            model = task.build_model(cfg.model)
            model.load_state_dict(
                state["model"],
                strict=True,
                model_cfg=cfg.model,
            )
            if args.fp16:
                model.half()
            model.cuda()
            model.eval()

            # register language model
            self.lm_decoder = model

            # # Check: inspect LM loading process and LM model
            # logging.info(" Checking language model ...... ")
            # dummy_inputs = torch.tensor(
            #     [[2,38,817,72,220,80,594,168,
            #       29,19,17,42,146,518,436]]
            # ).cuda()    # For validation
            # # dummy_inputs = torch.tensor(
            # #     [[2, 320, 1018, 1090, 553]]
            # # ).cuda()    # For training
            # dummy_lm_logits, _ = self.lm_decoder(src_tokens=dummy_inputs)
            # dummy_preds = dummy_lm_logits.max(-1).indices
            # dummy_logprobs = utils.log_softmax(
            #     dummy_lm_logits.float(), dim=-1)
            # nonmean_dummy_nll_loss = F.nll_loss(
            #     dummy_logprobs[0], dummy_inputs[0],
            #     ignore_index=self.pad, reduction="none")
            # dummy_nll_loss = F.nll_loss(
            #     dummy_logprobs[0], dummy_inputs[0],
            #     ignore_index=self.pad, reduction="mean")
            # logging.info(f"dummy_inputs: {dummy_inputs[0, 1:]}")
            # logging.info(f"dummy_preds:  {dummy_preds[0]}")
            # logging.info(f"dummy_nll_loss: {dummy_nll_loss}")
            # logging.info(f"nonmean_dummy_nll_loss: {nonmean_dummy_nll_loss}")
            # logging.info(f"Language model inspection is done.")

        if self.beam == 1:
            if self.cif_decoder_mode == "ar":
                logging.info("employ ar greedy decoder")
                self.decode = self.ar_batch_greedy_decode
            elif self.cif_decoder_mode == "fast_ar":
                logging.info("employ ar fast greedy decoder")
                self.decode = self.ar_fast_batch_greedy_decode
            else:
                logging.info("employ nar greedy decoder")
                # self.decode = self.nar_batch_greedy_decode
                self.decode = self.nar_batch_parallel_greedy_decode
                # Parallel Greedy Decoding which is better for NAR decoder
        else:
            if self.cif_decoder_mode == "ar":
                logging.info("employ ar beam decoder")
                self.decode = self.ar_batch_beam_decode
            elif self.cif_decoder_mode == "fast_ar":
                logging.info("employ ar fast beam decoder")
                self.decode = self.ar_fast_batch_beam_decode
            else:
                logging.info("employ nar beam decoder")
                self.decode = self.nar_batch_beam_decode

    def generate(self, models, sample, **kwargs):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder

        # Prepare model inputs
        model_inputs = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }  # remove prev_output_tokens

        # Forward encoder and cif
        if self.tail_handling_firing_threshold:
            models[
                0
            ].encoder.cif.tail_handling_firing_threshold = (
                self.tail_handling_firing_threshold
            )
        cif_outputs = models[0].get_cif_output(
            src_tokens=model_inputs["src_tokens"],
            src_lengths=model_inputs["src_lengths"],
            target_lengths=sample["target_lengths"],
        )

        # Decode
        beam_results, beam_scores, out_seqlens = self.decode(models[0], cif_outputs)

        # Truncate at <eos>
        tmp_beam_results = []
        bsz, beam_size, max_len = beam_results.size()
        beam_results = beam_results.view((bsz * beam_size), -1)  # (B * beam_size) x T
        for n in range(bsz):
            cur_res = beam_results[n]  # T
            eos_inds = (cur_res == 2).nonzero()
            if len(eos_inds) > 0:
                cur_max_valid_len = eos_inds[0][0]
            else:
                cur_max_valid_len = max_len
            cur_res = cur_res[:cur_max_valid_len]
            pad_len = max_len - cur_max_valid_len
            cur_res = torch.cat(
                [cur_res, torch.tensor([self.pad for _ in range(pad_len)]).cuda()],
                dim=0,
            )
            tmp_beam_results.append(cur_res.unsqueeze(0))
        beam_results = torch.cat(tmp_beam_results, dim=0).view(bsz, beam_size, -1)

        return self.generate_hypos(
            beam_results=beam_results,
            beam_scores=beam_scores,
            out_seqlens=out_seqlens,
        )

    def generate_hypos(self, beam_results, beam_scores, out_seqlens):
        hypos = []
        for beam_result, scores, lengths in zip(beam_results, beam_scores, out_seqlens):
            # beam_ids: beam x id; score: beam; length: beam
            top = []
            for result, score, length in zip(beam_result, scores, lengths):
                top.append({"tokens": self.get_tokens(result[:length]), "score": score})
            hypos.append(top)

        return hypos

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        # Remove blank id and eos id
        # idxs = (g[0] for g in it.groupby(idxs))   # remove repetition
        idxs = filter(lambda x: x != self.blank, idxs)
        idxs = filter(lambda x: x != self.eos, idxs)
        idxs = filter(lambda x: x != self.pad, idxs)

        return torch.LongTensor(list(idxs))

    def ar_batch_greedy_decode(self, model, cif_outputs):
        """
        :param model: the model in usage
        :param cif_outputs: the outputs of cif module
        :return: prev_tokens, out_seqlens, scores
        """

        # Get Cif outputs
        cif_out = cif_outputs["cif_out"]
        cif_out_padding_mask = cif_outputs["cif_out_padding_mask"]
        raw_encoder_out = cif_outputs["encoder_out"]
        raw_encoder_padding_mask = cif_outputs["encoder_padding_mask"]

        # Get the maximum length of decoding steps
        batch_size, max_decode_length, _ = cif_out.size()
        out_seqlens = cif_out_padding_mask.sum(-1)  # B

        # Initialize previous decoded tokens
        prev_tokens = torch.ones([batch_size, 1]).long().cuda() * self.eos
        # B x 1, use <eos> as the beginning of sentence (<bos>)
        scores = torch.ones([batch_size]).cuda()  # B
        for step_i in range(max_decode_length):
            # Conduct forward of current step t
            cur_step_cif_outputs = cif_out[:, : (step_i + 1), :]  # B x t x C
            cur_step_cif_out_padding_mask = cif_out_padding_mask[
                :, : (step_i + 1)
            ]  # B x t
            cur_step_cif_out = {
                "cif_out": cur_step_cif_outputs,
                "cif_out_padding_mask": cur_step_cif_out_padding_mask,
                "ctxt_cif_out": None,
                "raw_encoder_out": raw_encoder_out,
                "raw_encoder_padding_mask": raw_encoder_padding_mask,
            }

            # Get decoder outputs of current step
            decoder_output_i, extra_outputs, _ = model.step_forward_decoder(
                prev_decoded_tokens=prev_tokens, cif_outputs=cur_step_cif_out
            )

            # Update previous decoded tokens & scores
            decoder_output_i = model.get_probs_from_logits(
                decoder_output_i[:, -1, :], log_probs=False
            )
            latest_token = torch.argmax(decoder_output_i, dim=-1).unsqueeze(
                dim=-1
            )  # shape = B x 1
            prev_tokens = torch.cat([prev_tokens, latest_token], dim=-1)
            max_prob_of_last_step = decoder_output_i.max(-1)[0]  # shape = B
            scores = scores * max_prob_of_last_step

        # Reform outputs
        prev_tokens = torch.unsqueeze(prev_tokens, dim=1)[:, :, 1:]  # B x 1 x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=1)  # B x 1
        scores = torch.unsqueeze(scores, dim=-1)  # B x 1

        return prev_tokens, scores, out_seqlens

    def ar_fast_batch_greedy_decode(self, model, cif_outputs):
        """
        :param model: the model in usage
        :param cif_outputs: the outputs of cif module
        :return: prev_tokens, out_seqlens, scores
        """

        # Get Cif outputs
        cif_out = cif_outputs["cif_out"]
        cif_out_padding_mask = cif_outputs["cif_out_padding_mask"]
        raw_encoder_out = cif_outputs["encoder_out"]
        raw_encoder_padding_mask = cif_outputs["encoder_padding_mask"]

        # Get the maximum length of decoding steps
        batch_size, max_decode_length, _ = cif_out.size()
        out_seqlens = cif_out_padding_mask.sum(-1)  # B

        # Initialize incremental states for fast decoding
        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]], {}
        )
        # incremental_states is a dictionary of dictionaries of tensors

        # Initialize previous decoded tokens
        prev_tokens = torch.ones([batch_size, 1]).long().cuda() * self.eos
        # B x 1, use <eos> as the beginning of sentence (<bos>)
        scores = torch.ones([batch_size]).cuda()  # B
        for step_i in range(max_decode_length):
            # Forward decoder
            cur_step_cif_outputs = cif_out[:, : (step_i + 1), :]  # B x t x C
            cur_step_cif_out_padding_mask = cif_out_padding_mask[
                :, : (step_i + 1)
            ]  # B x t
            cur_step_cif_out = {
                "cif_out": cur_step_cif_outputs,
                "cif_out_padding_mask": cur_step_cif_out_padding_mask,
                "ctxt_cif_out": None,
                "raw_encoder_out": raw_encoder_out,
                "raw_encoder_padding_mask": raw_encoder_padding_mask,
            }

            # Get decoder outputs of current step
            decoder_output_i, _, _ = model.step_forward_decoder(
                prev_decoded_tokens=prev_tokens,
                cif_outputs=cur_step_cif_out,
                incremental_state=incremental_state,
            )
            # This is different from normal decoding process,
            # because the historical states are put into buffer

            # Update previous decoded tokens
            decoder_output_i = model.get_probs_from_logits(
                decoder_output_i[:, -1, :], log_probs=False
            )
            latest_token = torch.argmax(decoder_output_i, dim=-1).unsqueeze(
                dim=-1
            )  # B x 1
            prev_tokens = torch.cat([prev_tokens, latest_token], dim=-1)
            max_prob_of_last_step = decoder_output_i.max(-1)[0]  # B
            scores = scores * max_prob_of_last_step

        # Reform outputs
        prev_tokens = torch.unsqueeze(prev_tokens, dim=1)[:, :, 1:]  # B x 1 x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=1)  # B x 1
        scores = torch.unsqueeze(scores, dim=-1)  # B x 1

        return prev_tokens, scores, out_seqlens

    def ar_batch_beam_decode(self, model, cif_outputs):
        """
        :param model: the model in usage
        :param cif_outputs: the outputs of cif module
        :return: prev_tokens, out_seqlens, scores
        """

        cif_out = cif_outputs["cif_out"]  # B x T x C
        cif_out_padding_mask = cif_outputs["cif_out_padding_mask"]  # B x T
        raw_encoder_out = None
        raw_encoder_padding_mask = None

        # Get the maximum length of decoding steps
        batch_size, max_decode_length, cif_out_dim = cif_out.size()  # B x T x C
        out_seqlens = cif_out_padding_mask.sum(-1)  # B

        # Initialize all needed variables
        cif_out = torch.unsqueeze(cif_out, dim=1).repeat(
            1, self.beam, 1, 1
        )  # B x beam_size x T x C
        prev_tokens = (
            torch.ones([batch_size, self.beam, 1]).long().cuda() * self.eos
        )  # B x beam_size x 1
        scores = torch.zeros([batch_size, self.beam]).float().cuda()  # B x beam_size
        cif_out_padding_mask = torch.unsqueeze(cif_out_padding_mask, dim=1).repeat(
            [1, self.beam, 1]
        )
        # B x beam_size x T

        cif_out = cif_out.view(
            [batch_size * self.beam, max_decode_length, cif_out_dim]
        )  # (B * beam_size) x T x C
        prev_tokens = prev_tokens.view(
            [batch_size * self.beam, 1]
        )  # (B * beam_size) x 1
        scores = scores.view([batch_size * self.beam])  # (B * beam_size)
        cif_out_padding_mask = cif_out_padding_mask.view(
            [batch_size * self.beam, max_decode_length]
        )  # (B * beam_size) x T

        if not model.decoder.no_encoder_attn:
            raw_encoder_out = cif_outputs["encoder_out"]  # T x B x C
            raw_encoder_padding_mask = cif_outputs["encoder_padding_mask"]  # B x T
            max_raw_out_length, _, raw_out_dim = raw_encoder_out.size()
            raw_encoder_out = (
                raw_encoder_out.transpose(0, 1)
                .unsqueeze(dim=1)
                .repeat(1, self.beam, 1, 1)
                .view(batch_size * self.beam, max_raw_out_length, raw_out_dim)
                .transpose(0, 1)
            )  # T x (B x beam_size) x C
            raw_encoder_padding_mask = (
                raw_encoder_padding_mask.unsqueeze(dim=1)
                .repeat(1, self.beam, 1)
                .view(batch_size * self.beam, max_raw_out_length)
            )  # (B * beam_size) x T

        for step_i in range(1, max_decode_length + 1):
            # Get cif outputs of current step
            cur_step_cif_outputs = cif_out[:, :step_i, :]  # (B * beam_size) x t x C
            cur_step_cif_out_padding_mask = cif_out_padding_mask[
                :, :step_i
            ]  # (B * beam_size) x t
            cur_step_cif_out = {
                "cif_out": cur_step_cif_outputs,
                "cif_out_padding_mask": cur_step_cif_out_padding_mask,
                "ctxt_cif_out": None,
                "raw_encoder_out": raw_encoder_out,
                "raw_encoder_padding_mask": raw_encoder_padding_mask,
            }

            # Get decoder outputs at step_i
            decoder_output_i, extra_outputs, _ = model.step_forward_decoder(
                prev_decoded_tokens=prev_tokens,  # (B x beam_size) x t
                cif_outputs=cur_step_cif_out,
                # cif_out: (B * beam_size) x t x C, cif_out_padding_mask: (B * beam_size) x t
            )  # decoder_output_i has shape [(B * beam_size), t, V]
            cur_decoder_output = model.get_probs_from_logits(
                decoder_output_i[:, -1, :], log_probs=True
            )  # [B * beam_size, V]
            tmp_scores = scores  # Backup scores, with shape [B * beam_size]
            scores = scores.unsqueeze(dim=-1).repeat(
                [1, self.vocab_size]
            )  # [B * beam_size, V]

            cur_score = cur_decoder_output
            # cur_score, with shape [(B x beam_size) x V]

            updated_scores = (scores + cur_score).view(
                [batch_size, self.beam * self.vocab_size]
            )  # converted from shape [B * beam_size, V] to [B, beam_size * V]

            # Handle the first timestep with special operation
            if step_i == 1:
                # For the first step, due to the same input token, only consider one beam.
                topk_scores, topk_indices = torch.topk(
                    updated_scores.view([batch_size, self.beam, self.vocab_size])[
                        :, 0, :
                    ],
                    k=self.beam,
                    dim=-1,
                )
                beam_indices = (
                    torch.zeros(batch_size, self.beam).long().cuda()
                )  # [B, beam_size] with all zero elements
                fixed_topk_indices = topk_indices  # [B, beam_size]
            else:
                # For all the other beams, due to their inputs are varying, consider all beams.
                topk_scores, topk_indices = torch.topk(
                    updated_scores, k=self.beam, dim=-1
                )  # topk_scores shape [B, beam_size], topk_indices shape [B, beam_size]
                # beam_indices = \
                #    torch.div(topk_indices, self.vocab_size, rounding_mode='floor')  # [B, beam_size]
                beam_indices = topk_indices // vocab_size
                fixed_topk_indices = topk_indices % self.vocab_size  # [B, beam_size]

            # Update previous decoded tokens and scores
            prev_tokens = prev_tokens.view(
                [batch_size, self.beam, -1]
            )  # [B, beam_size, t]
            tmp_scores = tmp_scores.view(
                [batch_size, self.beam]
            )  # previous scores, with shape [B, beam_size]
            prev_token_tmp_list = []
            scores_tmp_list = []
            for n in range(batch_size):  # n ranges from 0 to (batch_size - 1)
                # Get the max length of current sample
                cur_output_maxlen = out_seqlens[n]

                # If some sample's decode length is smaller than current step id, keep its score and decoded results
                if step_i > cur_output_maxlen:
                    cur_scores = tmp_scores[n, :]  # beam_size
                    cur_prev_tokens = prev_tokens[n, :, :]  # beam_size x t
                else:
                    cur_scores = topk_scores[n, :]  # beam_size
                    cur_prev_tokens = prev_tokens[n, :, :]  # beam_size x t
                    cur_beam_indices = beam_indices[n, :]  # beam_size

                    # Get reformed previous tokens
                    cur_prev_tokens = torch.index_select(
                        cur_prev_tokens, dim=0, index=cur_beam_indices
                    )  # beam_size x t

                scores_tmp_list.append(cur_scores.unsqueeze(dim=0))
                prev_token_tmp_list.append(cur_prev_tokens.unsqueeze(dim=0))

            fixed_prev_tokens = torch.cat(prev_token_tmp_list, dim=0)
            fixed_topk_indices = torch.where(
                step_i <= out_seqlens.unsqueeze(dim=-1).repeat([1, self.beam]),
                fixed_topk_indices,  # B x beam_size
                torch.ones_like(fixed_topk_indices).cuda() * self.pad,
            )  # Mask locations that outnumber cif max length using <pad>
            fixed_topk_indices = fixed_topk_indices.unsqueeze(
                dim=-1
            )  # [B, beam_size, 1]
            prev_tokens = torch.cat(
                [fixed_prev_tokens, fixed_topk_indices], dim=-1
            ).view(
                [batch_size * self.beam, -1]
            )  # [B * beam_size, t + 1]
            scores = torch.cat(scores_tmp_list, dim=0).view(
                [batch_size * self.beam]
            )  # [B * beam_size]

        scores = scores.view([batch_size, self.beam])[:, : self.nbest]  # B x beam_size
        prev_tokens = prev_tokens.view([batch_size, self.beam, -1])[
            :, : self.nbest, 1:
        ]  # B x beam_size x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=-1).repeat(1, self.beam)[
            :, : self.nbest
        ]  # B x beam_size

        return prev_tokens, scores, out_seqlens

    def ar_fast_batch_beam_decode(self, model, cif_outputs):
        """
        :param model: the model in usage
        :param cif_outputs: the outputs of cif module
        :return: prev_tokens, out_seqlens, scores
        """

        cif_out = cif_outputs["cif_out"]  # B x T x C
        cif_out_padding_mask = cif_outputs["cif_out_padding_mask"]  # B x T
        raw_encoder_out = None
        raw_encoder_padding_mask = None

        # Get the maximum length of decoding steps
        batch_size, max_decode_length, cif_out_dim = cif_out.size()  # B x T x C
        out_seqlens = cif_out_padding_mask.sum(-1)  # B

        # Initialize all needed variables
        cif_out = torch.unsqueeze(cif_out, dim=1).repeat(
            1, self.beam, 1, 1
        )  # B x beam_size x T x C
        prev_tokens = (
            torch.ones([batch_size, self.beam, 1]).long().cuda() * self.eos
        )  # B x beam_size x 1
        scores = torch.zeros([batch_size, self.beam]).float().cuda()  # B x beam_size
        cif_out_padding_mask = torch.unsqueeze(cif_out_padding_mask, dim=1).repeat(
            [1, self.beam, 1]
        )
        # B x beam_size x T

        cif_out = cif_out.view(
            [batch_size * self.beam, max_decode_length, cif_out_dim]
        )  # (B * beam_size) x T x C
        prev_tokens = prev_tokens.view(
            [batch_size * self.beam, 1]
        )  # (B * beam_size) x 1
        scores = scores.view([batch_size * self.beam])  # (B * beam_size)
        cif_out_padding_mask = cif_out_padding_mask.view(
            [batch_size * self.beam, max_decode_length]
        )  # (B * beam_size) x T

        if not model.decoder.no_encoder_attn:
            raw_encoder_out = cif_outputs["encoder_out"]  # T x B x C
            raw_encoder_padding_mask = cif_outputs["encoder_padding_mask"]  # B x T
            max_raw_out_length, _, raw_out_dim = raw_encoder_out.size()
            raw_encoder_out = (
                raw_encoder_out.transpose(0, 1)
                .unsqueeze(dim=1)
                .repeat(1, self.beam, 1, 1)
                .view(batch_size * self.beam, max_raw_out_length, raw_out_dim)
                .transpose(0, 1)
            )  # T x (B x beam_size) x C
            raw_encoder_padding_mask = (
                raw_encoder_padding_mask.unsqueeze(dim=1)
                .repeat(1, self.beam, 1)
                .view(batch_size * self.beam, max_raw_out_length)
            )  # (B * beam_size) x T

        # Initialize incremental states for fast decoding
        reorder_state = None
        lm_reorder_state = None
        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]], {}
        )
        lm_incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]], {}
        )
        # incremental_states is a dictionary of dictionaries of tensors

        for step_i in range(1, max_decode_length + 1):
            # Reorder decoder internal states
            if reorder_state is not None:
                model.decoder.reorder_incremental_state_scripting(
                    incremental_state, reorder_state
                )
            if self.use_nnlm and lm_reorder_state is not None:
                self.lm_decoder.decoder.reorder_incremental_state_scripting(
                    lm_incremental_state, lm_reorder_state
                )

            # Get cif outputs of current step
            cur_step_cif_outputs = cif_out[:, :step_i, :]  # (B * beam_size) x t x C
            cur_step_cif_out_padding_mask = cif_out_padding_mask[
                :, :step_i
            ]  # (B * beam_size) x t
            cur_step_cif_out = {
                "cif_out": cur_step_cif_outputs,
                "cif_out_padding_mask": cur_step_cif_out_padding_mask,
                "ctxt_cif_out": None,
                "raw_encoder_out": raw_encoder_out,
                "raw_encoder_padding_mask": raw_encoder_padding_mask,
            }

            # Get decoder outputs at step_i
            decoder_output_i, extra_outputs, _ = model.step_forward_decoder(
                prev_decoded_tokens=prev_tokens,
                cif_outputs=cur_step_cif_out,
                incremental_state=incremental_state,
            )
            cur_decoder_output = model.get_probs_from_logits(
                decoder_output_i[:, -1, :], log_probs=True
            )  # [B * beam_size, V]
            tmp_scores = scores  # Backup scores, with shape [B * beam_size]
            scores = scores.unsqueeze(dim=-1).repeat(
                [1, self.vocab_size]
            )  # [B * beam_size, V]

            # Forward language model
            cur_lm_decoder_output = None
            if self.use_nnlm and self.lm_decoder is not None:
                lm_decoder_output_i, _ = self.lm_decoder(
                    src_tokens=prev_tokens,
                    incremental_state=lm_incremental_state,
                )
                cur_lm_decoder_output = model.get_probs_from_logits(
                    lm_decoder_output_i[:, -1, :],
                    log_probs=True,
                )  # [B * beam_size, V]

            # Update scores
            if self.use_nnlm:
                cur_score = cur_decoder_output + self.lm_weight * cur_lm_decoder_output
            else:
                cur_score = cur_decoder_output
            # cur_score, with shape [(B x beam_size) x V]
            updated_scores = (scores + cur_score).view(
                [batch_size, self.beam * self.vocab_size]
            )  # converted from shape [B * beam_size, V] to [B, beam_size * V]

            # Handle the first timestep with special operation
            if step_i == 1:
                # For the first step, due to the same input token, only consider one beam.
                topk_scores, topk_indices = torch.topk(
                    updated_scores.view([batch_size, self.beam, self.vocab_size])[
                        :, 0, :
                    ],
                    k=self.beam,
                    dim=-1,
                )
                beam_indices = (
                    torch.zeros(batch_size, self.beam).long().cuda()
                )  # [B, beam_size] with all zero elements
                fixed_topk_indices = topk_indices  # [B, beam_size]
            else:
                # For all the other steps, due to their inputs are varying, consider all beams.
                topk_scores, topk_indices = torch.topk(
                    updated_scores, k=self.beam, dim=-1
                )  # topk_scores shape [B, beam_size], topk_indices shape [B, beam_size]
                beam_indices = topk_indices // self.vocab_size
                fixed_topk_indices = topk_indices % self.vocab_size  # [B, beam_size]

            stage_index = torch.arange(batch_size) * self.beam
            cand_indices = beam_indices + stage_index.unsqueeze(-1).cuda()
            reorder_state = cand_indices.view(batch_size * self.beam)
            lm_reorder_state = reorder_state

            # Update previous decoded tokens and scores
            prev_tokens = prev_tokens.view(
                [batch_size, self.beam, -1]
            )  # [B, beam_size, t]
            tmp_scores = tmp_scores.view(
                [batch_size, self.beam]
            )  # previous scores, with shape [B, beam_size]
            prev_token_tmp_list = []
            scores_tmp_list = []
            for n in range(batch_size):  # n ranges from 0 to (batch_size - 1)
                # Get the max length of current sample
                cur_output_maxlen = out_seqlens[n]

                # If some sample's decode length is smaller than current step id, keep its score and decoded results
                if step_i > cur_output_maxlen:
                    cur_scores = tmp_scores[n, :]  # beam_size
                    cur_prev_tokens = prev_tokens[n, :, :]  # beam_size x t
                else:
                    cur_scores = topk_scores[n, :]  # beam_size
                    cur_prev_tokens = prev_tokens[n, :, :]  # beam_size x t
                    cur_beam_indices = beam_indices[n, :]  # beam_size

                    # Get reformed previous tokens
                    cur_prev_tokens = torch.index_select(
                        cur_prev_tokens, dim=0, index=cur_beam_indices
                    )  # beam_size x t

                scores_tmp_list.append(cur_scores.unsqueeze(dim=0))
                prev_token_tmp_list.append(cur_prev_tokens.unsqueeze(dim=0))

            fixed_prev_tokens = torch.cat(prev_token_tmp_list, dim=0)
            fixed_topk_indices = torch.where(
                step_i <= out_seqlens.unsqueeze(dim=-1).repeat([1, self.beam]),
                fixed_topk_indices,  # B x beam_size
                torch.ones_like(fixed_topk_indices).cuda() * self.pad,
            )  # Mask locations that outnumber cif max length using <pad>
            fixed_topk_indices = fixed_topk_indices.unsqueeze(
                dim=-1
            )  # [B, beam_size, 1]
            prev_tokens = torch.cat(
                [fixed_prev_tokens, fixed_topk_indices], dim=-1
            ).view(
                [batch_size * self.beam, -1]
            )  # [B * beam_size, t + 1]
            scores = torch.cat(scores_tmp_list, dim=0).view(
                [batch_size * self.beam]
            )  # [B * beam_size]

        scores = scores.view([batch_size, self.beam])[:, : self.nbest]  # B x beam_size
        prev_tokens = prev_tokens.view([batch_size, self.beam, -1])[
            :, : self.nbest, 1:
        ]  # B x beam_size x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=-1).repeat(1, self.beam)[
            :, : self.nbest
        ]  # B x beam_size

        return prev_tokens, scores, out_seqlens

    def nar_batch_parallel_greedy_decode(self, model, cif_outputs):
        """
        :param model: the model in usage
        :param cif_outputs: the outputs of cif module
        :return: prev_tokens, out_seqlens, scores
        """

        # Get cif outputs
        cif_out = cif_outputs["cif_out"]
        cif_out_padding_mask = cif_outputs["cif_out_padding_mask"]
        raw_encoder_out = cif_outputs["encoder_out"]
        raw_encoder_padding_mask = cif_outputs["encoder_padding_mask"]

        # Get the maximum length of decoding steps
        batch_size, max_decode_length, _ = cif_out.size()
        out_seqlens = cif_out_padding_mask.sum(-1)  # B

        # Initialize previous decoded tokens and cif outputs
        prev_decoded_tokens = torch.zeros(
            [batch_size, max_decode_length]
        ).long()  # B x T
        cif_outputs = {
            "cif_out": cif_out,
            "cif_out_padding_mask": cif_out_padding_mask,
            "raw_encoder_out": raw_encoder_out,
            "raw_encoder_padding_mask": raw_encoder_padding_mask,
        }

        decoder_output, _, _ = model.step_forward_decoder(
            prev_decoded_tokens=prev_decoded_tokens, cif_outputs=cif_outputs
        )  # B x T x V

        # Update previous decoded tokens
        decoder_output = model.get_probs_from_logits(
            decoder_output, log_probs=False
        )  # B x T x V
        decoded_tokens = torch.argmax(decoder_output, dim=-1)  # B x T
        scores = torch.prod(decoder_output.max(-1)[0], dim=-1)  # B

        # Reform outputs, now prev_tokens has shape B x (T + 1)
        prev_tokens = torch.unsqueeze(decoded_tokens, dim=1)  # B x 1 x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=1)  # B x 1
        scores = torch.unsqueeze(scores, dim=-1)  # B x 1

        return prev_tokens, scores, out_seqlens

    def nar_batch_beam_decode(self, model, cif_outputs):
        """
        :param model: the model in usage
        :param cif_outputs: the outputs of cif module
        :return: prev_tokens, out_seqlens, scores
        """

        cif_out = cif_outputs["cif_out"]  # B x T x C
        cif_out_padding_mask = cif_outputs["cif_out_padding_mask"]  # B x T
        raw_encoder_out = cif_outputs["encoder_out"]
        raw_encoder_padding_mask = cif_outputs["encoder_padding_mask"]

        # Get the maximum length of decoding steps
        batch_size, max_decode_length, cif_out_dim = cif_out.size()  # B x T x C
        out_seqlens = cif_out_padding_mask.sum(-1)  # B

        # Initialize all needed variables
        cif_out = torch.unsqueeze(cif_out, dim=1).repeat(
            1, self.beam, 1, 1
        )  # B x beam_size x T x C
        prev_tokens = (
            torch.ones([batch_size, self.beam, 1]).long().cuda() * self.eos
        )  # B x beam_size x 1
        scores = torch.zeros([batch_size, self.beam]).float().cuda()  # B x beam_size
        cif_out_padding_mask = torch.unsqueeze(cif_out_padding_mask, dim=1).repeat(
            [1, self.beam, 1]
        )  # B x beam_size x T

        cif_out = cif_out.view(
            [batch_size * self.beam, max_decode_length, cif_out_dim]
        )  # (B * beam_size) x T x C
        prev_tokens = prev_tokens.view(
            [batch_size * self.beam, 1]
        )  # (B * beam_size) x 1
        scores = scores.view([batch_size * self.beam])  # (B * beam_size)
        cif_out_padding_mask = cif_out_padding_mask.view(
            [batch_size * self.beam, max_decode_length]
        )  # (B * beam_size) x T

        for step_i in range(1, max_decode_length + 1):
            # Get cif outputs of current step
            cur_step_cif_outputs = cif_out[:, :step_i, :]  # (B * beam_size) x t x C
            cur_step_cif_out_padding_mask = cif_out_padding_mask[
                :, :step_i
            ]  # (B * beam_size) x t
            cur_step_cif_out = {
                "cif_out": cur_step_cif_outputs,
                "cif_out_padding_mask": cur_step_cif_out_padding_mask,
                "ctxt_cif_out": None,
                "raw_encoder_out": raw_encoder_out,
                "raw_encoder_padding_mask": raw_encoder_padding_mask,
            }

            # Get decoder outputs at step_i
            decoder_output_i, extra_outputs, _ = model.step_forward_decoder(
                prev_decoded_tokens=prev_tokens,  # (B x beam_size) x t
                cif_outputs=cur_step_cif_out,
                # cif_out: (B * beam_size) x t x C, cif_out_padding_mask: (B * beam_size) x t
            )  # decoder_output_i has shape [(B * beam_size), t, V]
            cur_decoder_output = model.get_probs_from_logits(
                decoder_output_i[:, -1, :], log_probs=True
            )  # [B * beam_size, V]
            tmp_scores = scores  # Backup scores, with shape [B * beam_size]
            scores = scores.unsqueeze(dim=-1).repeat(
                [1, self.vocab_size]
            )  # [B * beam_size, V]

            cur_score = cur_decoder_output
            # cur_score, with shape [(B x beam_size) x V]

            updated_scores = (scores + cur_score).view(
                [batch_size, self.beam * self.vocab_size]
            )  # converted from shape [B * beam_size, V] to [B, beam_size * V]

            # Handle the first timestep with special operation
            if step_i == 1:
                # For the first step, due to the same input token, only consider one beam.
                topk_scores, topk_indices = torch.topk(
                    updated_scores.view([batch_size, self.beam, self.vocab_size])[
                        :, 0, :
                    ],
                    k=self.beam,
                    dim=-1,
                )
                beam_indices = (
                    torch.zeros(batch_size, self.beam).long().cuda()
                )  # [B, beam_size] with all zero elements
                fixed_topk_indices = topk_indices  # [B, beam_size]
            else:
                # For all the other beams, due to their inputs are varying, consider all beams.
                topk_scores, topk_indices = torch.topk(
                    updated_scores, k=self.beam, dim=-1
                )  # topk_scores shape [B, beam_size], topk_indices shape [B, beam_size]
                beam_indices = torch.div(
                    topk_indices, self.vocab_size, rounding_mode="floor"
                )  # [B, beam_size]
                fixed_topk_indices = topk_indices % self.vocab_size  # [B, beam_size]

            # Update previous decoded tokens and scores
            prev_tokens = prev_tokens.view(
                [batch_size, self.beam, -1]
            )  # [B, beam_size, t]
            tmp_scores = tmp_scores.view(
                [batch_size, self.beam]
            )  # previous scores, with shape [B, beam_size]
            prev_token_tmp_list = []
            scores_tmp_list = []
            for n in range(batch_size):  # n ranges from 0 to (batch_size - 1)
                # Get the max length of current sample
                cur_output_maxlen = out_seqlens[n]

                # If some sample's decode length is smaller than current step id, keep its score and decoded results
                if step_i > cur_output_maxlen:
                    cur_scores = tmp_scores[n, :]  # beam_size
                    cur_prev_tokens = prev_tokens[n, :, :]  # beam_size x t
                else:
                    cur_scores = topk_scores[n, :]  # beam_size
                    cur_prev_tokens = prev_tokens[n, :, :]  # beam_size x t
                    cur_beam_indices = beam_indices[n, :]  # beam_size

                    # Get reformed previous tokens
                    cur_prev_tokens = torch.index_select(
                        cur_prev_tokens, dim=0, index=cur_beam_indices
                    )  # beam_size x t

                scores_tmp_list.append(cur_scores.unsqueeze(dim=0))
                prev_token_tmp_list.append(cur_prev_tokens.unsqueeze(dim=0))

            fixed_prev_tokens = torch.cat(prev_token_tmp_list, dim=0)
            fixed_topk_indices = torch.where(
                step_i <= out_seqlens.unsqueeze(dim=-1).repeat([1, self.beam]),
                fixed_topk_indices,  # B x beam_size
                torch.ones_like(fixed_topk_indices).cuda() * self.pad,
            )  # Mask locations that outnumber cif max length using <pad>
            fixed_topk_indices = fixed_topk_indices.unsqueeze(
                dim=-1
            )  # B x beam_size x 1

            prev_tokens = torch.cat(
                [fixed_prev_tokens, fixed_topk_indices], dim=-1
            ).view([batch_size * self.beam, -1])
            scores = torch.cat(scores_tmp_list, dim=0).view(
                [batch_size * self.beam]
            )  # B x beam_size

        scores = scores.view([batch_size, self.beam])[:, : self.nbest]  # B x beam_size
        prev_tokens = prev_tokens.view([batch_size, self.beam, -1])[
            :, : self.nbest, 1:
        ]  # B x beam_size x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=-1).repeat(1, self.beam)[
            :, : self.nbest
        ]  # B x beam_size

        return prev_tokens, scores, out_seqlens
