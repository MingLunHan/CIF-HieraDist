# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import math
import torch
import logging
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # Check: inspect LM loading process and LM model
        # logging.info(" Checking language model ...... ")
        # model.eval()
        # # dummy_inputs = torch.tensor(
        # #     [[2,38,817,72,220,80,594,168,
        # #       29,19,17,42,146,518,436]]
        # # ).cuda()    # For validation
        # dummy_inputs = torch.tensor(
        #     [[2, 320, 1018, 1090, 553]]
        # ).cuda()  # For training
        # dummy_lm_logits, _ = model(src_tokens=dummy_inputs)
        # dummy_preds = dummy_lm_logits.max(-1).indices
        # dummy_logprobs = utils.log_softmax(
        #     dummy_lm_logits.float(), dim=-1)
        # dummy_nll_loss = F.nll_loss(
        #     dummy_logprobs[0], dummy_inputs[0],
        #     ignore_index=self.padding_idx, reduction="mean")
        # logging.info(f"dummy_inputs: {dummy_inputs[0, 1:]}")
        # logging.info(f"dummy_preds:  {dummy_preds[0]}")
        # logging.info(f"dummy_nll_loss: {dummy_nll_loss}")
        # logging.info(f"Language model inspection is done.")

        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        # Check: little test for single sample
        # tokens = torch.tensor(
        #     [[   2,   38,  817,   72,  220,   80,  594,  168,   29,   19,   17,   42,146,  518,  436]]).cuda()
        # model.eval()
        # output, _ = model(src_tokens=tokens)
        # print(output.max(-1))
        # sys.exit(0)

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_scalar(
                "normal_nll_loss", loss_sum / ntokens / math.log2(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl",
                lambda meters: utils.get_perplexity(
                    meters["nll_loss"].avg, base=math.e
                ),
            )
            metrics.log_derived(
                "normal_ppl",
                lambda meters: utils.get_perplexity(
                    meters["normal_nll_loss"].avg, base=math.e
                ),
            )
        else:
            metrics.log_derived(
                "ppl",
                lambda meters: utils.get_perplexity(meters["loss"].avg, base=math.e),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
