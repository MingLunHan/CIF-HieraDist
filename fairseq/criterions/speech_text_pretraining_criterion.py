# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round
from fairseq.utils import buffered_arange, index_put, is_xla_tensor


@dataclass
class SpeechTextPretrainingCriterionConfig(FairseqDataclass):
    text_masked_language_modeling_weight: float = field(
        default=1.0,
        metadata={"help": "the weight of text masked language modeling loss"},
    )
    speech_masked_language_modeling_weight: float = field(
        default=1.0,
        metadata={"help": "the weight of speech masked language modeling loss"},
    )
    speech_text_matching_weight: float = field(
        default=1.0, metadata={"help": "the weight of speech text matching loss"}
    )
    asr_ce_loss_weight: float = field(
        default=1.0, metadata={"help": "the weight of asr cross entropy loss"}
    )
    asr_quantity_loss_weight: float = field(
        default=1.0, metadata={"help": "the weight of asr quantity loss"}
    )
    asr_ctc_loss_weight: float = field(
        default=1.0, metadata={"help": "the weight of asr ctc loss"}
    )
    translation_language_modeling_weight: float = field(
        default=1.0,
        metadata={"help": "the weight of translation language modeling loss"},
    )
    tts_loss_weight: float = field(
        default=1.0, metadata={"help": "the weight of text-to-speech loss"}
    )
    infonce_weight: float = field(
        default=1.0, metadata={"help": "the weight of audio contrastive loss"}
    )
    prob_ppl_weight: float = field(
        default=1.0, metadata={"help": "the weight of probability perplexity"}
    )
    feat_pen_weight: float = field(
        default=1.0, metadata={"help": "the weight of feature penalty"}
    )
    available_losses: Optional[List[str]] = field(
        default=None, metadata={"help": "the list of all available losses"}
    )

    mode: int = field(
        default=1,
        metadata={"help": "the training mode used for different data structure"},
    )


@register_criterion(
    "speech_text_pretraining_criterion", dataclass=SpeechTextPretrainingCriterionConfig
)
class SpeechTextPretrainingCriterion(FairseqCriterion):
    def __init__(self, cfg: SpeechTextPretrainingCriterionConfig, task):
        super().__init__(task)

        ## All losses weight configuration
        # 1. Losses for unpaired samples
        self._infonce_weight = cfg.infonce_weight
        self._prob_ppl_weight = cfg.prob_ppl_weight
        self._feat_pen_weight = cfg.feat_pen_weight
        self._text_masked_language_modeling_weight = (
            cfg.text_masked_language_modeling_weight
        )
        self._speech_masked_language_modeling_weight = (
            cfg.speech_masked_language_modeling_weight
        )

        # 2. Losses for paired samples
        self._speech_text_matching_weight = cfg.speech_text_matching_weight
        self._asr_ce_loss_weight = cfg.asr_ce_loss_weight
        self._asr_ctc_loss_weight = cfg.asr_ctc_loss_weight
        self._asr_quantity_loss_weight = cfg.asr_quantity_loss_weight
        self._translation_language_modeling_weight = (
            cfg.translation_language_modeling_weight
        )
        self._tts_loss_weight = cfg.tts_loss_weight

        ## All available losses
        self._available_losses = cfg.available_losses

        # Other settings
        self.default_dict = task.default_dictionary
        self.pad_ids = self.default_dict.pad()

        # Data config
        self._mode = cfg.mode

    @staticmethod
    def get_probs_from_logits(logits, log_probs=False):
        """Get normalized probabilities (or log probs) from logits."""
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_text_masked_language_modeling_loss(self, net_output):
        text_mlm_logits = net_output["text_mlm_logits"]
        text_mlm_targets = net_output["text_mlm_targets"]
        text_mlm_num_tokens = text_mlm_targets.numel()

        text_mlm_logprobs = self.get_probs_from_logits(text_mlm_logits, log_probs=True)
        text_mlm_logprobs = text_mlm_logprobs.view(-1, text_mlm_logprobs.size(-1))
        text_mlm_targets = text_mlm_targets.contiguous().view(
            -1
        )  # flatten targets tensor B x T_mask
        text_mlm_loss = F.nll_loss(
            text_mlm_logprobs,
            text_mlm_targets.long(),
            ignore_index=self.pad_ids,
            reduction="sum",
        )  # CE loss is the summation of all tokens, without any form of averaging

        if text_mlm_targets.numel() == 0 and text_mlm_logprobs.numel() == 0:
            text_mlm_loss = torch.tensor(0.0).cuda()

        if text_mlm_loss != text_mlm_loss:  # Handling nan loss error
            print("text_mlm_logits: ", text_mlm_logits)
            print(torch.isnan(text_mlm_logits).sum())
            print("text_mlm_targets: ", text_mlm_targets)
            print("text_mlm_targets: ", text_mlm_targets.size())
            print(torch.isnan(text_mlm_logprobs).sum())
            print("text_mlm_logprobs: ", text_mlm_logprobs)
            print("text_mlm_logprobs: ", text_mlm_logprobs.size())
            raise ValueError("loss equals nan errors.")

        return text_mlm_loss, text_mlm_num_tokens

    def get_speech_masked_language_modeling_loss(self, net_output):
        num_spec_samples = net_output["num_spec_samples"]
        spec_mlm_targets = net_output["spec_mlm_targets"]
        spec_mlm_logits = net_output["spec_mlm_logits"]
        spec_mlm_num_tokens = spec_mlm_targets.numel()

        spec_mlm_logprobs = self.get_probs_from_logits(
            spec_mlm_logits, log_probs=True
        )  #
        spec_mlm_logprobs = spec_mlm_logprobs.view(
            -1, spec_mlm_logprobs.size(-1)
        )  # (B x T_mask) x V
        spec_mlm_targets = spec_mlm_targets.contiguous().view(
            -1
        )  # flatten targets tensor B x T_mask
        spec_mlm_loss = F.nll_loss(
            spec_mlm_logprobs,
            spec_mlm_targets.long(),
            reduction="sum",
        )  # CE loss is the summation of all tokens, without any form of averaging

        return spec_mlm_loss, spec_mlm_num_tokens

    def get_translation_language_modeling_loss(self, net_output):
        num_pair_samples = net_output["num_pair_samples"]
        paired_text_tlm_logits = net_output["paired_text_tlm_logits"]
        paired_text_tlm_targets = net_output["paired_text_tlm_targets"]
        paired_spec_tlm_logits = net_output["paired_spec_tlm_logits"]
        paired_spec_tlm_targets = net_output["paired_spec_tlm_targets"]

        paired_text_tlm_logprobs = self.get_probs_from_logits(
            paired_text_tlm_logits, log_probs=True
        )
        paired_spec_tlm_logprobs = self.get_probs_from_logits(
            paired_spec_tlm_logits, log_probs=True
        )

        paired_text_tlm_logprobs = paired_text_tlm_logprobs.view(
            -1, paired_text_tlm_logprobs.size(-1)
        )
        paired_text_tlm_targets = paired_text_tlm_targets.contiguous().view(
            -1
        )  # flatten targets tensor
        text_tlm_loss = F.nll_loss(
            paired_text_tlm_logprobs,
            paired_text_tlm_targets.long(),
            ignore_index=self.pad_ids,
            reduction="sum",
        )  # CE loss is the summation of all tokens, without any form of averaging

        paired_spec_tlm_logprobs = paired_spec_tlm_logprobs.view(
            -1, paired_spec_tlm_logprobs.size(-1)
        )
        paired_spec_tlm_targets = paired_spec_tlm_targets.contiguous().view(
            -1
        )  # flatten targets tensor
        spec_tlm_loss = F.nll_loss(
            paired_spec_tlm_logprobs, paired_spec_tlm_targets.long(), reduction="sum"
        )  # CE loss is the summation of all tokens, without any form of averaging

        tlm_loss = text_tlm_loss + spec_tlm_loss
        tlm_num_tokens = (
            paired_text_tlm_targets.numel() + paired_spec_tlm_targets.numel()
        )

        return tlm_loss, tlm_num_tokens

    def get_speech_text_matching_loss(self, net_output):
        num_pair_samples = net_output["num_pair_samples"]
        stm_logits = net_output["stm_logits"]
        stm_labels = net_output["stm_labels"]
        stm_loss = F.binary_cross_entropy_with_logits(
            stm_logits, stm_labels, reduction="sum"
        )

        return stm_loss

    def get_asr_losses(self, net_output):
        pass

    def get_tts_loss(self, net_output):
        pass

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample."""

        # Forward the whole model
        net_output = model(sample, mode=self._mode)
        losses = dict()

        assert (
            self._available_losses != ""
        ), "please ensure there is at least one criterion in available criterions"

        self.xla = False

        text_mlm_num_tokens = None
        spec_mlm_num_tokens = None
        tlm_num_tokens = None

        # Calculate all losses
        if "infonce_loss" in self._available_losses:
            infonce_logits = model.get_infonce_logits(net_output).float()
            infonce_targets = model.get_infonce_targets(net_output)
            infonce_loss = F.cross_entropy(
                infonce_logits, infonce_targets, reduction="sum"
            )

            # Get number of samples in speech track
            spec_mlm_num_tokens = infonce_targets.numel()

            # Other relevant losses
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            for k, value in extra_losses.items():
                if k == "prob_perplexity":
                    p = value.float() * spec_mlm_num_tokens * self._prob_ppl_weight
                    losses["prob_perplexity_loss"] = p
                elif k == "features_pen":
                    p = value.float() * spec_mlm_num_tokens * self._feat_pen_weight
                    losses["feature_pen_loss"] = p
                else:
                    raise NotImplementedError("Unsupported options.")

            losses["infonce_loss"] = self._infonce_weight * infonce_loss

            # Calculate Accuracy and Correct number
            assert infonce_logits is not None
            with torch.no_grad():
                if infonce_logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert infonce_logits.dim() > 1, infonce_logits.shape
                    max = infonce_logits.argmax(-1) == 0
                    min = infonce_logits.argmin(-1) == 0
                    if is_xla_tensor(infonce_logits):
                        max, min = max * mi, min * mi
                        both = max & min
                        corr = max.long().sum() - both.long().sum()
                        count = mi.sum()
                    else:
                        both = max & min
                        corr = max.long().sum().item() - both.long().sum().item()
                        count = float(max.numel())

        if "text_mlm_loss" in self._available_losses:
            (
                text_mlm_loss,
                text_mlm_num_tokens,
            ) = self.get_text_masked_language_modeling_loss(net_output)
            losses["text_mlm_loss"] = (
                self._text_masked_language_modeling_weight * text_mlm_loss
            )

        if "spec_mlm_loss" in self._available_losses:
            (
                spec_mlm_loss,
                spec_mlm_num_tokens,
            ) = self.get_speech_masked_language_modeling_loss(net_output)
            losses["spec_mlm_loss"] = (
                self._speech_masked_language_modeling_weight * spec_mlm_loss
            )

        if "tlm_loss" in self._available_losses:
            tlm_loss, tlm_num_tokens = self.get_translation_language_modeling_loss(
                net_output
            )
            losses["tlm_loss"] = self._translation_language_modeling_weight * tlm_loss

        if "stm_loss" in self._available_losses:
            stm_loss = self.get_speech_text_matching_loss(net_output)
            losses["stm_loss"] = self._speech_text_matching_weight * stm_loss

        if "tts_loss" in self._available_losses:
            tts_loss = self.get_tts_loss(net_output)
            losses["tts_loss"] = self._tts_loss_weight * tts_loss

        if "asr_loss" in self._available_losses:
            asr_ce_loss, asr_ctc_loss, asr_quantity_loss = None, None, None
            losses["asr_ce_loss"] = self._asr_ce_loss_weight * asr_ce_loss
            losses["asr_ctc_loss"] = self._asr_ctc_loss_weight * asr_ctc_loss
            losses["asr_quantity_loss"] = (
                self._asr_quantity_loss_weight * asr_quantity_loss
            )

        # All sample size
        sample_size = sample["data_labels"].size(0)
        nsentences = sample_size
        ntokens = sample_size

        # Total losses values
        loss = torch.tensor(0.0).cuda()
        for loss_value in losses.values():
            loss += loss_value

        logging_output = {
            "loss": loss.item(),
            "sample_size": sample_size,
            "nsentences": nsentences,
            "ntokens": ntokens,
        }

        if "infonce_loss" in self._available_losses:
            logging_output["infonce_correct"] = corr
            logging_output["infonce_count"] = count

        # Collect the number of samples for each data class
        if "num_pair_samples" in net_output.keys():
            num_pair_samples = net_output["num_pair_samples"]
            logging_output["num_pair_samples"] = num_pair_samples
        if "num_spec_samples" in net_output.keys():
            num_spec_samples = net_output["num_spec_samples"]
            logging_output["num_spec_samples"] = num_spec_samples
        if "num_text_samples" in net_output.keys():
            num_text_samples = net_output["num_text_samples"]
            logging_output["num_text_samples"] = num_text_samples

        if "tlm_loss" in self._available_losses:
            logging_output["tlm_num_tokens"] = tlm_num_tokens

        if (
            "spec_mlm_loss" in self._available_losses
            or "infonce_loss" in self._available_losses
        ):
            logging_output["spec_mlm_num_tokens"] = spec_mlm_num_tokens

        if "text_mlm_loss" in self._available_losses:
            logging_output["text_mlm_num_tokens"] = text_mlm_num_tokens

        if len(losses.keys()) >= 1:
            for i, key in enumerate(losses.keys()):
                logging_output[key] = losses[key].item()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        # Collect the total loss_summation over these many steps
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        num_spec_samples = utils.item(
            sum(log.get("num_spec_samples", 0) for log in logging_outputs)
        )
        num_text_samples = utils.item(
            sum(log.get("num_text_samples", 0) for log in logging_outputs)
        )
        num_pair_samples = utils.item(
            sum(log.get("num_pair_samples", 0) for log in logging_outputs)
        )
        spec_mlm_num_tokens = utils.item(
            sum(log.get("spec_mlm_num_tokens", 0) for log in logging_outputs)
        )
        text_mlm_num_tokens = utils.item(
            sum(log.get("text_mlm_num_tokens", 0) for log in logging_outputs)
        )
        tlm_num_tokens = utils.item(
            sum(log.get("tlm_num_tokens", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        # metrics.log_scalar(
        #     "loss", loss_sum / math.log(2) / len(logging_outputs), round=3
        # )
        metrics.log_scalar(
            "loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )

        metrics.log_scalar("spec_mlm_num_tokens", spec_mlm_num_tokens)
        metrics.log_scalar("text_mlm_num_tokens", text_mlm_num_tokens)
        metrics.log_scalar("tlm_num_tokens", tlm_num_tokens)
        metrics.log_scalar("num_spec_samples", num_spec_samples)
        metrics.log_scalar("num_text_samples", num_text_samples)
        metrics.log_scalar("num_pair_samples", num_pair_samples)
        metrics.log_scalar("sample_size", sample_size)
        metrics.log_scalar("nsentences", nsentences)
        metrics.log_scalar("ntokens", ntokens)

        builtin_keys = {
            "loss",
            "nsentences",
            "sample_size",
            "num_spec_samples",
            "num_text_samples",
            "num_pair_samples",
            "spec_mlm_num_tokens",
            "text_mlm_num_tokens",
            "tlm_num_tokens",
        }

        # infonce relevant information if necessary
        if "infonce_loss" in logging_outputs[0].keys():
            infonce_correct = sum(
                log.get("infonce_correct", 0) for log in logging_outputs
            )
            metrics.log_scalar("infonce_correct", infonce_correct)
            infonce_total = sum(log.get("infonce_count", 0) for log in logging_outputs)
            metrics.log_scalar("infonce_total", infonce_total)
            if infonce_total > 0:
                metrics.log_derived(
                    "infonce_accuracy",
                    lambda meters: safe_round(
                        meters["infonce_correct"].sum / meters["infonce_total"].sum, 5
                    )
                    if meters["infonce_total"].sum > 0
                    else float("nan"),
                )

        for key in logging_outputs[0].keys():
            if key not in builtin_keys:
                val = sum(log.get(key, 0) for log in logging_outputs)
                if "loss" in key:  # Handling loss
                    val = utils.item(val)
                    if val != val:  # Handling nan loss errors
                        for i in range(len(logging_outputs)):
                            single_value = logging_outputs[i][key]
                            print(single_value)
                        raise ValueError("nan appears at %s" % key)

                    # speech loss part
                    if key == "infonce_loss":
                        metrics.log_scalar(
                            key,
                            val / spec_mlm_num_tokens / math.log(2),
                            spec_mlm_num_tokens,
                            round=6,
                        )
                    if key == "prob_perplexity_loss":
                        metrics.log_scalar(
                            key,
                            val / spec_mlm_num_tokens / math.log(2),
                            spec_mlm_num_tokens,
                            round=6,
                        )
                    if key == "feature_pen_loss":
                        metrics.log_scalar(
                            key,
                            val / spec_mlm_num_tokens / math.log(2),
                            spec_mlm_num_tokens,
                            round=6,
                        )
                    if key == "spec_mlm_loss":
                        metrics.log_scalar(
                            key,
                            val / spec_mlm_num_tokens / math.log(2),
                            spec_mlm_num_tokens,
                            round=6,
                        )

                    # text loss part
                    if key == "text_mlm_loss":
                        if text_mlm_num_tokens == 0:
                            text_mlm_num_tokens = 1
                        metrics.log_scalar(
                            key,
                            val / text_mlm_num_tokens / math.log(2),
                            text_mlm_num_tokens,
                            round=6,
                        )

                    # pair loss part
                    if key == "tlm_loss":
                        metrics.log_scalar(
                            key,
                            val / tlm_num_tokens / math.log(2),
                            tlm_num_tokens,
                            round=6,
                        )
                    if key == "stm_loss":
                        metrics.log_scalar(
                            key,
                            val / num_pair_samples / math.log(2),
                            num_pair_samples,
                            round=6,
                        )

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        # XXX: Gather based reduction not implemented for xla yet.
        # So we fall to sum based reduction for xla.
        return self.xla
