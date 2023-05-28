# @Time    : 2021/7/14
# @Author  : Minglun Han
# @File    : nar_cif.py

import sys
import math
import editdistance
import numpy as np
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round


@dataclass
class NarCifCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(default=False)
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(default="letter")

    # Cif loss settings
    ce_loss_lambda: float = field(default=1.0)
    apply_quantity_loss: bool = field(
        default=True, metadata={"help": "apply quantity loss"}
    )
    apply_ctc_loss: bool = field(default=True, metadata={"help": "apply ctc loss"})
    quantity_loss_lambda: float = field(
        default=1.0, metadata={"help": "the interpolation weight of quantity loss"}
    )
    ctc_loss_lambda: float = field(
        default=0.3, metadata={"help": "the interpolation weight of ctc loss"}
    )

    use_ctxt_cif_outputs: bool = field(default=False)


@register_criterion("nar_cif", dataclass=NarCifCriterionConfig)
class NarCifCriterion(FairseqCriterion):
    def __init__(self, cfg: NarCifCriterionConfig, task: FairseqTask):
        super().__init__(task)

        # Register default special tokens
        self.blank_idx = (
            task.target_dictionary.index("<ctc_blank>")
            if "<ctc_blank>" in task.target_dictionary.indices
            else task.target_dictionary.bos()
        )
        self.pad_idx = task.target_dictionary.pad()  # 1
        self.eos_idx = task.target_dictionary.eos()  # 2
        self.bos_idx = task.target_dictionary.bos()  # 0

        # Loss settings
        self.ce_loss_lambda = cfg.ce_loss_lambda
        self.apply_quantity_loss = cfg.apply_quantity_loss
        self.apply_ctc_loss = cfg.apply_ctc_loss
        self.quantity_loss_lambda = cfg.quantity_loss_lambda
        self.ctc_loss_lambda = cfg.ctc_loss_lambda
        self.use_ctxt_cif_outputs = cfg.use_ctxt_cif_outputs

        # other settings
        self.post_process = cfg.post_process
        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

    def get_loss(self, model, sample, net_output, reduce=True):
        # Get model outputs
        encoder_out = net_output["encoder_out"]  # B x T x C
        quantity_out = net_output["quantity_out"]  # B
        cif_out = net_output["cif_out"]  # B x T x C
        ctxt_cif_out = net_output["ctxt_cif_out"]  # B x T x C
        cif_out_padding_mask = net_output["cif_out_padding_mask"]  # B x T
        decoder_out = net_output["decoder_out"][
            0
        ]  # Get final decoder outputs (logits for cross-entropy loss)

        # Collect src_lengths for the calculation of ctc loss
        non_padding_mask = ~net_output["padding_mask"]
        input_lengths = non_padding_mask.int().sum(-1)

        # Collect targets and target_length for ctc loss and ce loss
        target_lengths = sample["target_lengths"]  # targets length w/o eos
        target_with_eos = sample["target"]  # this target has <eos> at the end
        target_with_eos_lengths = target_lengths + 1  # targets length w/ eos

        # Adjust targets: move the eos token from the last location to the end of valid location
        batch_size = target_with_eos.size(0)
        target_with_eos_non_padding_mask = (
            (target_with_eos != self.eos_idx) & (target_with_eos != self.pad_idx)
        ).int()  # B x T
        add_eos_idx = (
            ((target_with_eos * target_with_eos_non_padding_mask) != 0)
            .int()
            .sum(dim=-1)
            .unsqueeze(dim=-1)
        )  # B x 1
        add_one_hot_tensor = (
            torch.zeros(batch_size, target_with_eos_non_padding_mask.size(1))
            .int()
            .cuda()
            .scatter_(1, add_eos_idx, 1)
            * self.eos_idx
        )
        adjusted_target_with_eos = torch.where(
            (
                (target_with_eos.int() * target_with_eos_non_padding_mask)
                + add_one_hot_tensor
            )
            == 0,
            torch.ones_like(target_with_eos).int().cuda() * self.pad_idx,
            (target_with_eos.int() * target_with_eos_non_padding_mask)
            + add_one_hot_tensor,
        )
        # target_with_eos: [[20,56,7,8,1,1,1,1,2], ..., [60,6,7,349,34,1,1,1,2]]
        # adjusted_target_with_eos: [[20,56,7,8,2,1,1,1,1], ..., [60,6,7,349,34,2,1,1,1]]

        # Calculate the ctc loss on encoder outputs
        ctc_loss = torch.tensor(0.0)
        if self.apply_ctc_loss:
            pad_mask = adjusted_target_with_eos != self.pad_idx
            targets_flat = adjusted_target_with_eos.masked_select(pad_mask)
            ctc_lprobs = model.get_probs_from_logits(
                encoder_out, log_probs=True
            ).contiguous()  # (B, T, V) from the encoder
            target_lengths_for_ctc_loss = target_with_eos_lengths
            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = F.ctc_loss(
                    ctc_lprobs.transpose(0, 1),  # T x B x v
                    targets_flat,
                    input_lengths,
                    target_lengths_for_ctc_loss,
                    blank=self.blank_idx,
                    reduction="sum",
                    zero_infinity=self.zero_infinity,
                )

        # Calculate the quantity loss
        qtt_loss = torch.tensor(0.0)
        if self.apply_quantity_loss:
            target_lengths_for_qtt_loss = (
                target_with_eos_lengths  # Lengths after adding eos token, [B]
            )
            qtt_loss = torch.abs(quantity_out - target_lengths_for_qtt_loss).sum()

        # Calculate the cross-entropy loss
        cif_max_len = cif_out_padding_mask.size(1)  # Get max length of cif outputs
        target_max_length = target_with_eos_lengths.max()  # Get max length of targets
        min_len = min(
            cif_max_len, target_max_length
        )  # Obtain the minimum length of cif length and target length
        ce_logprobs = model.get_probs_from_logits(
            decoder_out, log_probs=True
        ).contiguous()  # B x T x C
        truncated_target = adjusted_target_with_eos[
            :, :min_len
        ]  # Truncate target to min_len, B x T
        truncated_ce_logprobs = ce_logprobs[
            :, :min_len, :
        ]  # Truncate ce probs to min_len,  B x T x C
        # Truncate target to the minimum length of original target and cif outputs,
        # because sometimes the firing of CIF may lost <eos>.
        truncated_ce_logprobs = truncated_ce_logprobs.view(
            -1, truncated_ce_logprobs.size(-1)
        )
        truncated_target = truncated_target.contiguous().view(
            -1
        )  # flatten targets tensor
        ce_loss = F.nll_loss(
            truncated_ce_logprobs,
            truncated_target.long(),
            ignore_index=self.pad_idx,
            reduction="sum" if reduce else "none",
        )

        # Calculate the total loss
        loss = (
            self.ce_loss_lambda * ce_loss
            + self.quantity_loss_lambda * qtt_loss
            + self.ctc_loss_lambda * ctc_loss
        )

        # Collect the number of tokens in current batch
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )
        ntokens_with_eos = target_with_eos_lengths.sum().item()

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ce_loss": utils.item(ce_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "quantity_loss": utils.item(qtt_loss.data),
            "ntokens": ntokens,
            "ntokens_with_eos": ntokens_with_eos,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        # Evaluate on valid sets
        if not model.training:
            with torch.no_grad():
                lprobs_t = ce_logprobs.float().contiguous().cpu()
                cif_lengths = cif_out_padding_mask.int().sum(dim=-1)  # B x T

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t, adjusted_target_with_eos, cif_lengths
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    # Process targets
                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()
                    # print(targ_units_arr)

                    # Handle log probabilities without elements
                    if min(lp.shape) == 0:
                        toks = targ
                    else:
                        toks = lp.argmax(dim=-1)
                    pred_units_arr = toks[
                        (toks != self.blank_idx)
                        & (toks != self.pad_idx)
                        & (toks != self.eos_idx)
                    ].tolist()
                    # print(pred_units_arr)

                    # Calculate character error
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()
                    # print("targ_words: ", targ_words)

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    # print("pred_units: ", pred_units)
                    pred_words_raw = post_process(pred_units, self.post_process).split()
                    # print("pred_words_raw: ", pred_words_raw)

                    # Calculate word error
                    dist = editdistance.eval(pred_words_raw, targ_words)
                    w_errs += dist
                    wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        net_output = model(
            target_lengths_with_eos=sample["target_lengths"] + 1, **sample["net_input"]
        )
        loss, sample_size, logging_output = self.get_loss(
            model, sample, net_output, reduce=True
        )
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ce_loss_sum = utils.item(sum(log.get("ce_loss", 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(
            sum(log.get("ctc_loss", 0) for log in logging_outputs)
        )
        quantity_loss_sum = utils.item(
            sum(log.get("quantity_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        ntokens_with_eos = utils.item(
            sum(log.get("ntokens_with_eos", 0) for log in logging_outputs)
        )
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "ce_loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "quantity_loss",
            quantity_loss_sum / sample_size / math.log(2),
            sample_size,
            round=5,
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("ntokens_with_eos", ntokens_with_eos)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
