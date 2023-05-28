# @Time    : 2021/7/14
# @Author  : Minglun Han
# @File    : cif_correction_loss.py

"""""
    Update:
    By 2022/0805
        1. Add monitoring for negative correction loss and positive correction loss (done);
        2. Freeze the batch normalization parameters for Conformer structure (done);
        3. Remove redundant codes (done);
""" ""

import sys
import math
import editdistance
import numpy as np
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round

import sklearn
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

np.set_printoptions(threshold=100000)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """
    :param lprobs: log probabilities with shape B x T x V
    :param target: targets with shape B x T
    :param epsilon: Epsilon
    :param ignore_index: padding index
    :param reduce: whether sum all positions loss
    :return: smoothed cross entropy loss
    """

    # Calculate losses
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)  # B x T x 1
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    # Reduce losses
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)

    # Get final smoothed cross-entropy loss
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


@dataclass
class CifCriterionConfig(FairseqDataclass):
    # General settings
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="char",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )

    # Settings of cif losses
    no_quantity_loss: bool = field(
        default=False, metadata={"help": "apply quantity loss"}
    )
    no_ctc_loss: bool = field(default=False, metadata={"help": "apply ctc loss"})
    apply_align_loss: bool = field(default=False, metadata={"help": "apply align loss"})
    quantity_loss_lambda: float = field(
        default=1.0, metadata={"help": "the interpolation weight of quantity loss"}
    )
    ctc_loss_lambda: float = field(
        default=0.25, metadata={"help": "the interpolation weight of ctc loss"}
    )
    align_loss_lambda: float = field(
        default=1.0,
        metadata={"help": "the interpolation weight of ctc-constrained alignment loss"},
    )
    apply_label_smoothing: bool = field(
        default=False,
        metadata={"help": "apply label smoothing over cross entropy loss"},
    )
    label_smoothing_type: str = field(
        default="uniform", metadata={"help": "specify the label smoothing type"}
    )
    label_smoothing_rate: float = field(
        default=0.1, metadata={"help": "the rate of label smoothing"}
    )
    no_eos_label: bool = field(default=False)
    apply_focal_loss: bool = field(default=False)
    focal_loss_gamma: float = field(default=2.0)
    no_comb_loss_sum: bool = field(default=False)

    # uncertainty estimation loss (UE loss) settings
    ue_loss_lambda: float = field(default=1.0)
    apply_ue_focal_loss: bool = field(default=False)
    ue_focal_scaling_weight: float = field(default=1.0)
    ue_focal_loss_gamma: float = field(default=2.0)
    apply_ue_cb_loss: bool = field(default=False)
    ue_cb_loss_beta: float = field(default=0.90)
    reverse_uem_labels: bool = field(default=False)
    use_pred_softmax_prob_as_conf: bool = field(default=False)

    # correction loss (Correction loss) settings
    corr_loss_lambda: float = field(default=1.0)
    apply_corr_focal_loss: bool = field(default=False)
    corr_focal_scaling_weight: float = field(default=1.0)
    corr_focal_loss_gamma: float = field(default=2.0)


def eval_nce(labels, probs, eps=1e-8):
    labels = torch.tensor(labels).float()
    probs = torch.tensor(probs).float()

    def stable_log(tensor):
        return torch.log(torch.clip(tensor, min=eps, max=1.0))

    total_sample_num = labels.size()[0]
    correct_sample_num = labels[labels == 1].size()[0]
    cross_entropy = (
        -1
        * (
            torch.dot(labels, stable_log(probs))
            + torch.dot((1 - labels), stable_log(1 - probs))
        )
        / total_sample_num
    )
    p_c = torch.tensor(float(correct_sample_num) / float(total_sample_num))
    entropy = (
        -1
        * (
            correct_sample_num * torch.log(p_c)
            + (total_sample_num - correct_sample_num) * torch.log(1 - p_c)
        )
        / total_sample_num
    )
    normalized_ce = (entropy - cross_entropy) / entropy

    return normalized_ce


@register_criterion("cif_correction_loss", dataclass=CifCriterionConfig)
class CifCorrectionLoss(FairseqCriterion):
    def __init__(self, cfg: CifCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index("<ctc_blank>")
            if "<ctc_blank>" in task.target_dictionary.indices
            else task.target_dictionary.bos()
        )
        self.pad_idx = task.target_dictionary.pad()  # 1
        self.eos_idx = task.target_dictionary.eos()  # 2
        self.bos_idx = task.target_dictionary.bos()  # 0
        self.no_correction_idx = len(task.target_dictionary)
        self.post_process = cfg.post_process
        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

        # Register losses settings
        self.apply_quantity_loss = not cfg.no_quantity_loss
        self.apply_ctc_loss = not cfg.no_ctc_loss
        self.apply_align_loss = cfg.apply_align_loss
        self.quantity_loss_lambda = cfg.quantity_loss_lambda
        self.ctc_loss_lambda = cfg.ctc_loss_lambda
        self.align_loss_lambda = cfg.align_loss_lambda

        # Register label smoothing settings
        self.label_smoothing_type = cfg.label_smoothing_type
        self.label_smoothing_rate = cfg.label_smoothing_rate
        self.apply_label_smoothing = cfg.apply_label_smoothing
        self.apply_focal_loss = cfg.apply_focal_loss
        self.focal_loss_gamma = cfg.focal_loss_gamma

        self.no_eos_label = cfg.no_eos_label
        self.no_comb_loss_sum = cfg.no_comb_loss_sum

        # Register correction loss settings
        self.ue_loss_lambda = cfg.ue_loss_lambda
        self.apply_ue_focal_loss = cfg.apply_ue_focal_loss
        self.ue_focal_scaling_weight = cfg.ue_focal_scaling_weight
        self.ue_focal_loss_gamma = cfg.ue_focal_loss_gamma
        self.apply_ue_cb_loss = cfg.apply_ue_cb_loss
        self.ue_cb_loss_beta = cfg.ue_cb_loss_beta
        self.reverse_uem_labels = cfg.reverse_uem_labels
        self.use_pred_softmax_prob_as_conf = cfg.use_pred_softmax_prob_as_conf

        self.corr_loss_lambda = cfg.corr_loss_lambda
        self.apply_corr_focal_loss = cfg.apply_corr_focal_loss
        self.corr_focal_scaling_weight = cfg.corr_focal_scaling_weight
        self.corr_focal_loss_gamma = cfg.corr_focal_loss_gamma

    def get_loss(self, model, sample, net_output, reduce=True):
        # Get model outputs
        cif_out_padding_mask = net_output["cif_out_padding_mask"]  # B x T

        # Correction training needed values
        # corr_cif_outputs = net_output["corr_cif_outputs"]   # B x T x C
        corr_cif_padding_mask = net_output["corr_cif_padding_mask"]  # B x T
        uem_logits = net_output["uem_logits"]  # B x T
        uem_labels = net_output["uem_labels"]  # B x (1 + K) x T
        cordec_logits = net_output[
            "cordec_logits"
        ]  # B x T x (V + 1), the extra one token is <no-cor> mark
        cordec_labels = net_output["cordec_labels"]  # B x (1 + K) x T
        uem_padding_mask = net_output["uem_padding_mask"].bool()  # B x T
        aligned_targets = net_output["aligned_targets"]  # B x T
        aligned_preds = net_output["aligned_preds"]
        decoder_logits = net_output["asr_decoder_logits"]  # B x T
        if self.reverse_uem_labels:
            uem_labels = 1 - uem_labels

        # Collect targets and target_length for ctc loss and ce loss
        target_lengths = sample["target_lengths"]  # targets length without eos
        target_with_eos = sample["target"]
        target_with_eos_lengths = target_lengths  # targets length with eos
        if self.no_eos_label:
            target_with_eos_lengths = target_with_eos_lengths - 1
            target_with_eos = torch.where(
                target_with_eos == self.eos_idx,
                self.pad_idx * torch.ones_like(target_with_eos),
                target_with_eos,
            )
        adjusted_target_with_eos = target_with_eos

        # Calculate the cross-entropy loss
        cif_max_len = cif_out_padding_mask.size(1)  # Get max length of cif outputs
        tgt_max_len = target_with_eos_lengths.max()  # Get max length of targets
        reg_min_len = min(
            cif_max_len, tgt_max_len
        )  # Obtain the minimum length of cif length and target length

        # Calculate the uncertainty estimation (UE) loss
        cif_max_len = corr_cif_padding_mask.size(-1)
        org_uem_probs, org_uem_labels = None, None
        ue_loss = torch.tensor(0.0)
        ue_pos_loss = torch.tensor(0.0)
        ue_neg_loss = torch.tensor(0.0)
        if model.args.use_uem or self.ue_loss_lambda != 0.0:
            # process labels and probs
            uem_labels = uem_labels[:, :cif_max_len]  # B x T
            uem_logits = uem_logits[:, :cif_max_len]  # B x T
            uem_probs = torch.sigmoid(uem_logits.float())  # B x T
            org_uem_probs = uem_probs  # B x T
            org_uem_labels = uem_labels  # B x T

            if self.apply_ue_focal_loss:
                scaling_weight = torch.where(
                    uem_labels == 1, 1 - uem_probs, uem_probs
                )  # B x T
                scaling_weight = self.ue_focal_scaling_weight * (
                    scaling_weight**self.ue_focal_loss_gamma
                )
                scaling_weight = scaling_weight[uem_padding_mask]  # (B x T)
                uem_probs = uem_probs[uem_padding_mask]  # (B x T)
                uem_labels = uem_labels[uem_padding_mask]  # (B x T)
            elif self.apply_ue_cb_loss:
                uem_probs = uem_probs[uem_padding_mask]  # (B x T)
                uem_labels = uem_labels[uem_padding_mask]  # (B x T)
                pos_label_num = uem_labels[uem_labels == 1].size(0)
                neg_label_num = uem_labels[uem_labels == 0].size(0)
                scaling_weight = torch.where(
                    uem_labels == 1,
                    (1 - self.ue_cb_loss_beta)
                    / (1 - self.ue_cb_loss_beta**pos_label_num),
                    (1 - self.ue_cb_loss_beta)
                    / (1 - self.ue_cb_loss_beta**neg_label_num),
                )  # (B x T)
            else:
                scaling_weight = torch.ones_like(uem_labels)[uem_padding_mask]
                uem_probs = uem_probs[uem_padding_mask]  # (B x T)
                uem_labels = uem_labels[uem_padding_mask]  # (B x T)

            # Calculate cross-entropy loss
            try:
                ue_loss = F.binary_cross_entropy(
                    uem_probs.float(), uem_labels.float(), reduction="none"
                )
            except:
                print(uem_probs)
                print(uem_labels)
                print(uem_probs[(uem_probs < 0) | (uem_probs > 1)])
                print(uem_labels[(uem_labels < 0) | (uem_labels > 1)])
                sys.exit(0)
            ue_loss = (scaling_weight * ue_loss).sum()

            # Calculate negtive loss
            uem_neg_labels = uem_labels[uem_labels == 0]
            uem_neg_probs = uem_probs[uem_labels == 0]
            ue_neg_loss = F.binary_cross_entropy(
                uem_neg_probs.float(), uem_neg_labels.float(), reduction="sum"
            )

            # Calculate positive loss
            uem_pos_labels = uem_labels[uem_labels == 1]
            uem_pos_probs = uem_probs[uem_labels == 1]
            ue_pos_loss = F.binary_cross_entropy(
                uem_pos_probs.float(),
                uem_pos_labels.float(),
                reduction="sum",
            )

        # Calculate the correction cross-entropy (Corr-CE) loss
        corr_loss = torch.tensor(0.0)
        pos_corr_loss = torch.tensor(0.0)
        neg_corr_loss = torch.tensor(0.0)
        org_corr_probs, org_corr_lprobs, org_corr_labels = None, None, None
        pos_corr_nsamples, neg_corr_nsamples = 1, 1
        if model.args.use_cordec and self.corr_loss_lambda != 0.0:
            # process labels for correction loss
            corr_ce_labels = cordec_labels[:, :cif_max_len]  # B x T
            org_corr_labels = corr_ce_labels
            corr_ce_labels = corr_ce_labels.view(-1)  # (B x T)

            # process probs and log-probs for correction loss
            corr_probs = model.get_probs_from_logits(cordec_logits, log_probs=False)[
                :, :cif_max_len, :
            ]  # B x T x V
            if model.args.use_corr_pgn:
                gate_ratio = net_output["gate_ratio"]  # B x T x 1
                copy_probs = net_output["copy_probs"][:, :cif_max_len, :]  # B x T x V
                corr_probs = gate_ratio * corr_probs + (1 - gate_ratio) * copy_probs
            corr_lprobs = torch.log(corr_probs)
            org_corr_lprobs = corr_lprobs  # B x T x V
            corr_lprobs = corr_lprobs.view(-1, corr_lprobs.size(-1))  # (B x T) x V
            org_corr_probs = corr_probs  # B x T x V
            corr_probs = corr_probs.view(-1, corr_probs.size(-1))  # (B x T) x V

            # calculate focal loss weight
            corr_scaling_weight = torch.ones_like(corr_ce_labels)  # (B x T)
            if self.apply_corr_focal_loss:
                corr_scaling_weight = F.one_hot(
                    corr_ce_labels.long(), num_classes=corr_lprobs.size(-1)
                )  # (B x T) x V
                corr_scaling_weight = (corr_scaling_weight * (1 - corr_probs)).sum(
                    -1
                )  # (B x T)
                corr_scaling_weight = self.corr_focal_scaling_weight * (
                    corr_scaling_weight**self.corr_focal_loss_gamma
                )  # (B x T)

            # calculate correction ce loss
            corr_loss = F.nll_loss(
                corr_lprobs.float(),
                corr_ce_labels.long(),
                ignore_index=self.pad_idx,
                reduction="none",
            )
            corr_loss = (corr_scaling_weight * corr_loss).sum()

            # calculate positive loss and negative loss
            pos_corr_ce_labels = corr_ce_labels[
                corr_ce_labels != self.no_correction_idx
            ]
            pos_corr_probs = corr_lprobs[corr_ce_labels != self.no_correction_idx]
            pos_corr_loss = F.nll_loss(
                pos_corr_probs.float(),
                pos_corr_ce_labels.long(),
                ignore_index=self.pad_idx,
                reduction="none",
            )
            pos_corr_loss = pos_corr_loss.sum().detach()
            pos_corr_nsamples = corr_ce_labels[
                (corr_ce_labels != self.pad_idx)
                & (corr_ce_labels != self.no_correction_idx)
            ].size(0)

            neg_corr_ce_labels = corr_ce_labels[
                corr_ce_labels == self.no_correction_idx
            ]
            neg_corr_probs = corr_lprobs[corr_ce_labels == self.no_correction_idx]
            neg_corr_loss = F.nll_loss(
                neg_corr_probs.float(),
                neg_corr_ce_labels.long(),
                ignore_index=self.pad_idx,
                reduction="none",
            )
            neg_corr_loss = neg_corr_loss.sum().detach()
            neg_corr_nsamples = max(
                corr_ce_labels[
                    (corr_ce_labels != self.pad_idx)
                    & (corr_ce_labels == self.no_correction_idx)
                ].size(0),
                1,
            )

        # Collect the number of tokens in current batch
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )
        ntokens_with_eos = target_with_eos_lengths.sum().item()
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        corr_nsamples = uem_labels.size(0)
        neg_ue_nsamples = uem_labels[uem_labels == 0].size(0)
        pos_ue_nsamples = uem_labels[uem_labels == 1].size(0)

        loss = ue_loss * self.ue_loss_lambda + corr_loss * self.corr_loss_lambda

        # CHECK: check all losses
        if torch.isnan(ue_loss).sum() != 0 or torch.isinf(ue_loss).sum() != 0:
            print("Gradient problems: ")
            print(ue_loss)
        if torch.isnan(corr_loss).sum() != 0 or torch.isinf(corr_loss).sum() != 0:
            print("Gradient problems: ")
            print(corr_loss)

        # Build final logging outputs
        logging_output = {
            "loss": utils.item(loss.data),
            "ue_loss": utils.item(ue_loss.data),
            "ue_neg_loss": utils.item(ue_neg_loss.data),
            "ue_pos_loss": utils.item(ue_pos_loss.data),
            "corr_loss": utils.item(corr_loss.data),
            "pos_corr_loss": utils.item(pos_corr_loss.data),
            "neg_corr_loss": utils.item(neg_corr_loss.data),
            "ntokens": ntokens,
            "ntokens_with_eos": ntokens_with_eos,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "corr_nsamples": corr_nsamples,
            "neg_ue_nsamples": neg_ue_nsamples,
            "pos_ue_nsamples": pos_ue_nsamples,
            "pos_corr_nsamples": pos_corr_nsamples,
            "neg_corr_nsamples": neg_corr_nsamples,
        }

        # Evaluate on valid sets
        # if not model.training:
        #     with torch.no_grad():
        #         lprobs_t = ce_logprobs.float().contiguous().cpu()
        #         cif_lengths = cif_out_padding_mask.int().sum(dim=-1)  # B x T
        #
        #         c_err = 0
        #         c_len = 0
        #         w_errs = 0
        #         w_len = 0
        #         wv_errs = 0
        #
        #         # Loop over all hypothesis
        #         for lp, t, inp_l in zip(
        #                 lprobs_t,
        #                 adjusted_target_with_eos,
        #                 cif_lengths
        #         ):
        #             lp = lp[:inp_l].unsqueeze(0)
        #
        #             # Process targets
        #             # p = (t != self.task.target_dictionary.pad()) & (t != self.task.target_dictionary.eos())
        #             p = (t != self.task.target_dictionary.pad())
        #             targ = t[p]
        #             targ_units = self.task.target_dictionary.string(targ)
        #             targ_units_arr = targ.tolist()
        #
        #             # Handle lp without elements
        #             if min(lp.shape) == 0:
        #                 toks = targ
        #             else:
        #                 toks = lp.argmax(dim=-1)
        #
        #             pred_units_arr = \
        #                 toks[(toks != self.blank_idx) & (toks != self.pad_idx)].tolist()
        #
        #             # Calculate character error
        #             c_err += editdistance.eval(pred_units_arr, targ_units_arr)
        #             c_len += len(targ_units_arr)
        #
        #             targ_words = post_process(targ_units, self.post_process).split()
        #             pred_units = self.task.target_dictionary.string(pred_units_arr)
        #             pred_words_raw = post_process(pred_units, self.post_process).split()
        #
        #             # Calculate word error
        #             dist = editdistance.eval(pred_words_raw, targ_words)
        #             w_errs += dist
        #             wv_errs += dist
        #
        #             w_len += len(targ_words)
        #
        #         logging_output["wv_errors"] = wv_errs
        #         logging_output["w_errors"] = w_errs
        #         logging_output["w_total"] = w_len
        #         logging_output["c_errors"] = c_err
        #         logging_output["c_total"] = c_len

        # Check uncertainty estimation accuracy on validation set
        if self.ue_loss_lambda != 0.0:
            with torch.no_grad():
                uem_probs = org_uem_probs.float().contiguous().cpu()  # B x T
                uem_labels = org_uem_labels.int().contiguous().cpu()  # B x T
                uem_lens = uem_padding_mask.int().sum(dim=-1)  # B

                # Get softmax confidence score / uncertainty score
                conf_ce_probs = model.get_probs_from_logits(
                    decoder_logits, log_probs=False
                )[
                    :, :cif_max_len, :
                ]  # B x T x V
                onehot_mask = F.one_hot(
                    aligned_preds.long(), num_classes=conf_ce_probs.size(-1)
                )
                conf_probs = (conf_ce_probs * onehot_mask).sum(-1)  # B x T
                # conf_probs = conf_ce_probs.max(-1).values   # B x T
                ue_sm_probs = 1 - conf_probs  # B x T
                ue_sm_probs = ue_sm_probs.float().contiguous().cpu()  # B x T

                # Loop over all hypothesis
                pred_pos_num, pred_neg_num = 0, 0
                label_pos_num, label_neg_num = 0, 0
                uem_total_num, uem_correct_num = 0, 0
                uem_recall_total_pos_num, uem_recall_correct_pos_num = 0, 0
                uem_precision_total_pos_num, uem_precision_correct_pos_num = 0, 0
                auc_labels, auc_probs, auc_sm_probs = [], [], []
                for probs, sm_probs, label, max_len in zip(
                    uem_probs, ue_sm_probs, uem_labels, uem_lens
                ):
                    probs = probs[:max_len]  # T
                    sm_probs = sm_probs[:max_len]  # T
                    label = label[:max_len]  # T
                    pred = (probs > 0.5).int()

                    pred_pos_num += (pred == 1).sum()
                    pred_neg_num += (pred != 1).sum()
                    label_pos_num += (label == 1).sum()
                    label_neg_num += (label != 1).sum()

                    label_pos_locations = (label == 1).bool()
                    pred_for_recall = pred[label_pos_locations]
                    label_for_recall = label[label_pos_locations]
                    uem_recall_correct_pos_num += (
                        pred_for_recall == label_for_recall
                    ).sum()
                    uem_recall_total_pos_num += label_for_recall.size()[0]

                    pred_pos_locations = (pred == 1).bool()
                    pred_for_precision = pred[pred_pos_locations]
                    label_for_precision = label[pred_pos_locations]
                    uem_precision_correct_pos_num += (
                        pred_for_precision == label_for_precision
                    ).sum()
                    uem_precision_total_pos_num += pred_for_precision.size()[0]

                    comp_res = (pred == label).int()
                    uem_total_num += comp_res.size()[0]
                    uem_correct_num += (comp_res == 1).sum()

                    # Collect labels and probs for the calculation of AUC
                    auc_labels.append(label)
                    auc_probs.append(probs)
                    auc_sm_probs.append(sm_probs)

                auc_labels = np.concatenate(auc_labels, axis=0)
                auc_probs = np.concatenate(auc_probs, axis=0)
                auc_sm_probs = np.concatenate(auc_sm_probs, axis=0)

                if self.reverse_uem_labels:
                    auc_sm_probs = 1 - auc_sm_probs

                sm_nce = eval_nce(auc_labels, auc_sm_probs)
                uem_nce = eval_nce(auc_labels, auc_probs)

                try:
                    # For uem probs
                    uem_auc = roc_auc_score(auc_labels, auc_probs, average=None)
                    precision, recall, _ = precision_recall_curve(auc_labels, auc_probs)
                    uem_pr_auc = auc(recall, precision)
                    neg_precision, neg_recall, _ = precision_recall_curve(
                        1 - auc_labels, 1 - auc_probs
                    )
                    uem_pr_neg_auc = auc(neg_recall, neg_precision)

                    # For softmax probs
                    uem_sm_auc = roc_auc_score(auc_labels, auc_sm_probs, average=None)
                    precision, recall, _ = precision_recall_curve(
                        auc_labels, auc_sm_probs
                    )
                    uem_sm_pr_auc = auc(recall, precision)
                    neg_precision, neg_recall, _ = precision_recall_curve(
                        1 - auc_labels, 1 - auc_sm_probs
                    )
                    uem_sm_pr_neg_auc = auc(neg_recall, neg_precision)
                except ValueError:
                    print("Encounter ValueError, ignore it.")
                    uem_auc, uem_pr_auc, uem_pr_neg_auc = 1.0, 1.0, 1.0
                    uem_sm_auc, uem_sm_pr_auc, uem_sm_pr_neg_auc = 1.0, 1.0, 1.0

                logging_output["uem_pred_pos_num"] = pred_pos_num.item()
                logging_output["uem_pred_neg_num"] = pred_neg_num.item()
                logging_output["uem_label_pos_num"] = label_pos_num.item()
                logging_output["uem_label_neg_num"] = label_neg_num.item()
                logging_output["uem_total_num"] = uem_total_num
                logging_output["uem_correct_num"] = uem_correct_num.item()
                logging_output["uem_recall_total_pos_num"] = uem_recall_total_pos_num
                logging_output[
                    "uem_recall_correct_pos_num"
                ] = uem_recall_correct_pos_num.item()
                logging_output[
                    "uem_precision_total_pos_num"
                ] = uem_precision_total_pos_num
                logging_output[
                    "uem_precision_correct_pos_num"
                ] = uem_precision_correct_pos_num.item()
                logging_output["uem_auc"] = uem_auc
                logging_output["uem_pr_auc"] = uem_pr_auc
                logging_output["uem_pr_neg_auc"] = uem_pr_neg_auc
                logging_output["uem_sm_auc"] = uem_sm_auc
                logging_output["uem_sm_pr_auc"] = uem_sm_pr_auc
                logging_output["uem_sm_pr_neg_auc"] = uem_sm_pr_neg_auc
                logging_output["sm_nce"] = sm_nce
                logging_output["uem_nce"] = uem_nce

        # Check correction outputs accuracy on validation set
        if self.corr_loss_lambda != 0.0:
            with torch.no_grad():
                corr_probs = org_corr_lprobs.float().contiguous().cpu()  # B x T x V
                corr_preds = torch.argmax(corr_probs, dim=-1)  # B x T
                cordec_labels = org_corr_labels.int().contiguous().cpu()  # B x T
                uem_lens = uem_padding_mask.int().sum(dim=-1)  # B

                # Loop over all hypothesis
                corr_total_num, corr_correct_num = 0, 0
                corr_total_pos_num, corr_correct_pos_num = 0, 0
                for pred, label, max_len in zip(corr_preds, cordec_labels, uem_lens):
                    pred = pred[:max_len]  # T
                    label = label[:max_len]  # T

                    comp_res = (pred == label).int()
                    corr_total_num += comp_res.size()[0]
                    corr_correct_num += (comp_res == 1).sum()

                    label_pos_locations = (label != self.no_correction_idx).bool()
                    label_for_recall = label[label_pos_locations]
                    pred_for_recall = pred[label_pos_locations]
                    corr_correct_pos_num += (pred_for_recall == label_for_recall).sum()
                    corr_total_pos_num += label_for_recall.size()[0]

                logging_output["corr_total_num"] = corr_total_num
                logging_output["corr_correct_num"] = corr_correct_num.item()
                logging_output["corr_total_pos_num"] = corr_total_pos_num
                logging_output["corr_correct_pos_num"] = corr_correct_pos_num.item()

        # Calculate the total loss
        if self.no_comb_loss_sum:
            loss = (
                ue_loss * self.ue_loss_lambda / corr_nsamples
                + corr_loss * self.corr_loss_lambda / corr_nsamples
            )

        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        # Attention: Freeze all batch normalization
        if model.args.freeze_asr_main_body:
            model.freeze_batch_norm(model)

        # Forward the whole model
        net_output = model(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            target_lengths=sample["target_lengths"],
            target=sample["target"],
        )

        # Calculate losses
        loss, sample_size, logging_output = self.get_loss(
            model, sample, net_output, reduce=True
        )

        return loss, sample_size, logging_output

    def expand_tensor_dim(self, x, expand_size, target_dim=1, reduce=False):
        assert (
            target_dim == 1
        ), "only the expansion at the second dimension is available."

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

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """
        Aggregate logging outputs from data parallel training.
        """

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ue_loss_sum = utils.item(sum(log.get("ue_loss", 0) for log in logging_outputs))
        ue_neg_loss_sum = utils.item(
            sum(log.get("ue_neg_loss", 0) for log in logging_outputs)
        )
        ue_pos_loss_sum = utils.item(
            sum(log.get("ue_pos_loss", 0) for log in logging_outputs)
        )
        corr_loss_sum = utils.item(
            sum(log.get("corr_loss", 0) for log in logging_outputs)
        )
        neg_corr_loss_sum = utils.item(
            sum(log.get("neg_corr_loss", 0) for log in logging_outputs)
        )
        pos_corr_loss_sum = utils.item(
            sum(log.get("pos_corr_loss", 0) for log in logging_outputs)
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
        corr_nsamples = utils.item(
            sum(log.get("corr_nsamples", 0) for log in logging_outputs)
        )
        pos_ue_nsamples = utils.item(
            sum(log.get("pos_ue_nsamples", 0) for log in logging_outputs)
        )
        neg_ue_nsamples = utils.item(
            sum(log.get("neg_ue_nsamples", 0) for log in logging_outputs)
        )
        pos_corr_nsamples = utils.item(
            sum(log.get("pos_corr_nsamples", 0) for log in logging_outputs)
        )
        neg_corr_nsamples = utils.item(
            sum(log.get("neg_corr_nsamples", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "ue_loss", ue_loss_sum / corr_nsamples / math.log(2), corr_nsamples, round=5
        )
        metrics.log_scalar(
            "ue_pos_loss",
            ue_pos_loss_sum / pos_ue_nsamples / math.log(2),
            pos_ue_nsamples,
            round=5,
        )
        metrics.log_scalar(
            "ue_neg_loss",
            ue_neg_loss_sum / neg_ue_nsamples / math.log(2),
            neg_ue_nsamples,
            round=5,
        )
        metrics.log_scalar(
            "corr_loss",
            corr_loss_sum / corr_nsamples / math.log(2),
            corr_nsamples,
            round=5,
        )
        metrics.log_scalar(
            "corr_pos_loss",
            pos_corr_loss_sum / pos_corr_nsamples / math.log(2),
            pos_corr_nsamples,
            round=5,
        )
        metrics.log_scalar(
            "corr_neg_loss",
            neg_corr_loss_sum / neg_corr_nsamples / math.log(2),
            neg_corr_nsamples,
            round=5,
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("ntokens_with_eos", ntokens_with_eos)
        metrics.log_scalar("nsentences", nsentences)
        metrics.log_scalar("corr_nsamples", corr_nsamples)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["c_errors"].sum * 100.0 / meters["c_total"].sum, 3
                )
                if meters["c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["w_errors"].sum * 100.0 / meters["w_total"].sum, 3
                )
                if meters["w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["wv_errors"].sum * 100.0 / meters["w_total"].sum, 3
                )
                if meters["w_total"].sum > 0
                else float("nan"),
            )

        uem_pred_pos_num = sum(
            log.get("uem_pred_pos_num", 0) for log in logging_outputs
        )
        metrics.log_scalar("uem_pred_pos_num", uem_pred_pos_num)
        uem_pred_neg_num = sum(
            log.get("uem_pred_neg_num", 0) for log in logging_outputs
        )
        metrics.log_scalar("uem_pred_neg_num", uem_pred_neg_num)
        uem_label_pos_num = sum(
            log.get("uem_label_pos_num", 0) for log in logging_outputs
        )
        metrics.log_scalar("uem_label_pos_num", uem_label_pos_num)
        uem_label_neg_num = sum(
            log.get("uem_label_neg_num", 0) for log in logging_outputs
        )
        metrics.log_scalar("uem_label_neg_num", uem_label_neg_num)
        uem_total_num = sum(log.get("uem_total_num", 0) for log in logging_outputs)
        metrics.log_scalar("uem_total_num", uem_total_num)
        uem_correct_num = sum(log.get("uem_correct_num", 0) for log in logging_outputs)
        metrics.log_scalar("uem_correct_num", uem_correct_num)
        uem_recall_correct_pos_num = sum(
            log.get("uem_recall_correct_pos_num", 0) for log in logging_outputs
        )
        metrics.log_scalar("uem_recall_correct_pos_num", uem_recall_correct_pos_num)
        uem_recall_total_pos_num = sum(
            log.get("uem_recall_total_pos_num", 0) for log in logging_outputs
        )
        metrics.log_scalar("uem_recall_total_pos_num", uem_recall_total_pos_num)
        uem_precision_correct_pos_num = sum(
            log.get("uem_precision_correct_pos_num", 0) for log in logging_outputs
        )
        metrics.log_scalar(
            "uem_precision_correct_pos_num", uem_precision_correct_pos_num
        )
        uem_precision_total_pos_num = sum(
            log.get("uem_precision_total_pos_num", 0) for log in logging_outputs
        )
        metrics.log_scalar("uem_precision_total_pos_num", uem_precision_total_pos_num)
        uem_auc_sum = sum(log.get("uem_auc", 0) for log in logging_outputs)
        metrics.log_scalar("uem_auc_sum", uem_auc_sum)
        uem_pr_auc_sum = sum(log.get("uem_pr_auc", 0) for log in logging_outputs)
        metrics.log_scalar("uem_pr_auc_sum", uem_pr_auc_sum)
        uem_pr_neg_auc_sum = sum(
            log.get("uem_pr_neg_auc", 0) for log in logging_outputs
        )
        metrics.log_scalar("uem_pr_neg_auc_sum", uem_pr_neg_auc_sum)
        uem_sm_auc_sum = sum(log.get("uem_sm_auc", 0) for log in logging_outputs)
        metrics.log_scalar("uem_sm_auc_sum", uem_sm_auc_sum)
        uem_sm_pr_auc_sum = sum(log.get("uem_sm_pr_auc", 0) for log in logging_outputs)
        metrics.log_scalar("uem_sm_pr_auc_sum", uem_sm_pr_auc_sum)
        uem_sm_pr_neg_auc_sum = sum(
            log.get("uem_sm_pr_neg_auc", 0) for log in logging_outputs
        )
        metrics.log_scalar("uem_sm_pr_neg_auc_sum", uem_sm_pr_neg_auc_sum)

        sm_nce_sum = sum(log.get("sm_nce", 0) for log in logging_outputs)
        metrics.log_scalar("sm_nce_sum", sm_nce_sum)
        uem_nce_sum = sum(log.get("uem_nce", 0) for log in logging_outputs)
        metrics.log_scalar("uem_nce_sum", uem_nce_sum)

        if uem_correct_num > 0:
            metrics.log_derived(
                "uem_accuracy",
                lambda meters: safe_round(
                    meters["uem_correct_num"].sum * 100.0 / meters["uem_total_num"].sum,
                    3,
                )
                if meters["uem_correct_num"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "uem_recall",
                lambda meters: safe_round(
                    meters["uem_recall_correct_pos_num"].sum
                    * 100.0
                    / meters["uem_recall_total_pos_num"].sum,
                    3,
                )
                if meters["uem_recall_correct_pos_num"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "uem_precision",
                lambda meters: safe_round(
                    meters["uem_precision_correct_pos_num"].sum
                    * 100.0
                    / meters["uem_precision_total_pos_num"].sum,
                    3,
                )
                if meters["uem_precision_correct_pos_num"].sum > 0
                else float("nan"),
            )

        corr_total_num = sum(log.get("corr_total_num", 0) for log in logging_outputs)
        metrics.log_scalar("corr_total_num", corr_total_num)
        corr_correct_num = sum(
            log.get("corr_correct_num", 0) for log in logging_outputs
        )
        metrics.log_scalar("corr_correct_num", corr_correct_num)
        corr_correct_pos_num = sum(
            log.get("corr_correct_pos_num", 0) for log in logging_outputs
        )
        metrics.log_scalar("corr_correct_pos_num", corr_correct_pos_num)
        corr_total_pos_num = sum(
            log.get("corr_total_pos_num", 0) for log in logging_outputs
        )
        metrics.log_scalar("corr_total_pos_num", corr_total_pos_num)

        if corr_correct_num > 0:
            metrics.log_derived(
                "corr_accuracy",
                lambda meters: safe_round(
                    meters["corr_correct_num"].sum
                    * 100.0
                    / meters["corr_total_num"].sum,
                    3,
                )
                if meters["corr_correct_num"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "corr_recall",
                lambda meters: safe_round(
                    meters["corr_correct_pos_num"].sum
                    * 100.0
                    / meters["corr_total_pos_num"].sum,
                    3,
                )
                if meters["corr_correct_pos_num"].sum > 0
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
