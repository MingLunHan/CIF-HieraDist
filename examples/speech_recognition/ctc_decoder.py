# @Time    : 2021/7/26
# @Author  : Minglun Han
# @File    : ctc_decoder.py

import os
import sys
import torch
import random
import logging
import torch.nn.functional as F
import numpy as np
import itertools as it

# Control print options
torch.set_printoptions(profile="full")
torch.set_printoptions(profile="default")
np.set_printoptions(threshold=sys.maxsize)


class CtcDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.beam = args.beam

        # Get the index of special tokens
        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )  # if <ctc_blank> in dictionary, use its index else use bos token's index
        self.bos = tgt_dict.bos()
        self.eos = tgt_dict.eos()
        self.pad = tgt_dict.pad()

        if self.beam == 1:
            logging.info("employ ctc greedy decoder")
            self.decode = self.batch_greedy_decode
        else:
            raise NotImplementedError("Not supported options!")

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        model_inputs = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }  # remove prev_output_tokens

        # Forward encoder
        ctc_logits, encoder_outputs_padding_mask = models[0].get_ctc_output(
            src_tokens=model_inputs["src_tokens"],
            src_lengths=model_inputs["src_lengths"],
        )

        # Obtain log-probabilities and conduct decoding
        ctc_log_probs = models[0].get_probs_from_logits(ctc_logits, log_probs=True)
        beam_results, beam_scores, out_seqlens = self.decode(
            ctc_log_probs, encoder_outputs_padding_mask
        )

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
        """
        Normalize tokens by handling CTC blank, ASG replabels, etc.
        """

        # Remove blank id and eos id
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        idxs = filter(lambda x: x != self.eos, idxs)
        return torch.LongTensor(list(idxs))

    def batch_greedy_decode(self, ctc_log_probs, encoder_outputs_padding_mask):
        """
        :param model: the model in usage
        :param ctc_log_probs: the log probabilities of ctc outputs
        :return: prev_tokens, out_seqlens, scores
        """

        # Get the maximum length of decoding steps
        batch_size, max_ctc_outputs_len, _ = ctc_log_probs.size()
        input_lengths = (~encoder_outputs_padding_mask).int().sum(-1)

        # Acquire output seqlens and scores
        out_seqlens = []
        scores = []
        for sample_id in range(batch_size):
            # Acquire current sample's ctc log probabilities
            cur_sample_encoder_out_len = input_lengths[sample_id]
            # print(cur_sample_encoder_out_len)

            cur_ctc_log_probs = ctc_log_probs[sample_id, :cur_sample_encoder_out_len, :]
            # cur_sample_encoder_out_len x V
            # print(cur_ctc_log_probs.size())

            cur_score = cur_ctc_log_probs.max(dim=-1)[0].sum().item()  # 1
            cur_toks = cur_ctc_log_probs.argmax(
                dim=-1
            ).unique_consecutive()  # cur_sample_encoder_out_len
            cur_toks = cur_toks[cur_toks != self.blank]
            cur_out_seqlen = cur_toks.size(0)

            scores.append(cur_score)
            out_seqlens.append(cur_out_seqlen)

        # Acquire output hypotheses
        scores = torch.tensor(scores)
        out_seqlens = torch.tensor(out_seqlens)
        prev_tokens = []
        max_output_seqlen = out_seqlens.max().item()
        for sample_id in range(batch_size):
            cur_sample_encoder_out_len = input_lengths[sample_id]
            cur_ctc_log_probs = ctc_log_probs[sample_id, :cur_sample_encoder_out_len, :]
            cur_toks = cur_ctc_log_probs.argmax(dim=-1)
            # print(cur_toks)
            cur_toks = cur_toks.unique_consecutive()
            # print(cur_toks)
            cur_toks = cur_toks[cur_toks != self.blank]
            # print(cur_toks)
            cur_out_seqlen = cur_toks.size(0)

            padding_tensor = (
                (torch.ones([max_output_seqlen - cur_out_seqlen]) * self.tgt_dict.pad())
                .long()
                .cuda()
            )
            sample_pred = torch.unsqueeze(
                torch.cat([cur_toks, padding_tensor], dim=0), dim=0
            )

            prev_tokens.append(sample_pred)

            sys.exit(0)

        prev_tokens = torch.cat(prev_tokens, dim=0)

        # Reform outputs
        prev_tokens = torch.unsqueeze(prev_tokens, dim=1)  # B x 1 x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=-1)  # B x 1
        scores = torch.unsqueeze(scores, dim=-1)  # B x 1

        return prev_tokens, scores, out_seqlens
