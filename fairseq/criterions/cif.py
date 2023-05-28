# @Time    : 2021/7/14
# @Author  : Minglun Han
# @File    : cif.py

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

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)  # B x T x 1
    nll_loss = -lprobs.gather(dim=-1, index=target)  # NllLoss, B x T x 1
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)  # Smoothed Loss, B x T x 1

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    eps_i = epsilon / (lprobs.size(-1) - 1)

    # Get final smoothed cross-entropy loss
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss

    if reduce:
        loss = loss.sum()
        nll_loss = nll_loss.sum()

    return loss, nll_loss


def calculate_cif_contrastive_dis_cos_loss(
    cif_feats,
    bert_feats,
    mask,
    temperature=0.1,
    num_contrastive_negative_samples=300,
    remove_overlap_in_negs=False,
    sample_std_negs=False,
    loss_type="cos",
    arccos_margin=2,
):

    bsz, tsz, _ = cif_feats.size()
    student_feats = cif_feats.view(bsz * tsz, -1)  # (bsz * tsz) * C = B x C
    teacher_feats = bert_feats.view(bsz * tsz, -1)  # (bsz * tsz) * C = B x C
    mask = mask.contiguous().view(bsz * tsz)
    student_feats = student_feats[mask]  # N x C
    teacher_feats = teacher_feats[mask]  # N x C

    num_samples = teacher_feats.size(0)
    if num_samples <= num_contrastive_negative_samples:
        num_contrastive_negative_samples = int(num_samples * 0.8)
    assert (
        num_samples > num_contrastive_negative_samples
    ), "number of negative samples must smaller than that of total samples"

    # 1. Create negative keys
    sampling_weights = (
        torch.ones([num_samples, num_samples]).type_as(teacher_feats).cuda()
    )  # N x N
    neg_ids = torch.multinomial(
        sampling_weights,
        num_samples=num_contrastive_negative_samples,
        replacement=False,
    ).long()  # N x N_negs

    # sample negative samples from students
    if sample_std_negs:
        mix_probs = torch.rand([num_samples, 1]).cuda()
        negative_samples = torch.where(
            (mix_probs.repeat(1, cif_feats.size(-1)) > 0.5).bool(),
            teacher_feats,
            student_feats,
        )[
            neg_ids
        ]  # N x N_negs x C
    else:
        negative_samples = teacher_feats[neg_ids]  # N x N_negs x C

    # 2. Normalize queries and all keys
    def normalize(*xs):
        return [None if x is None else F.normalize(x, dim=-1, eps=1e-6) for x in xs]

    student_feats, teacher_feats, negative_samples = normalize(
        student_feats, teacher_feats, negative_samples
    )

    # 3. Calculate logits
    if loss_type == "arccos":
        positive_logit = torch.cos(
            torch.arccos(torch.sum(student_feats * teacher_feats, dim=-1, keepdim=True))
            + arccos_margin
        )
    else:
        positive_logit = torch.sum(
            student_feats * teacher_feats, dim=-1, keepdim=True
        )  # N x 1

    negative_logits = torch.matmul(
        student_feats.unsqueeze(1), negative_samples.transpose(-2, -1)
    ).squeeze(
        1
    )  # N x N_negs
    if remove_overlap_in_negs:
        triangle_ids = (
            torch.ones_like(neg_ids).cuda()
            * torch.tensor([i for i in range(num_samples)]).unsqueeze(-1).cuda()
        )  # N x N_negs
        negative_logits = torch.where(
            (neg_ids == triangle_ids).bool(),  # N x N_negs
            torch.ones_like(negative_logits) * -1e9,
            negative_logits,
        )  # N x N_negs x C

    logits = torch.cat([positive_logit, negative_logits], dim=-1)  # N x (N_negs + 1)
    labels = torch.zeros(len(logits)).long().cuda()

    # 4. Caluclate loss
    loss = F.cross_entropy(logits.float() / temperature, labels, reduction="none")

    # 5. Calculate contrastive accuracy
    max_indices = torch.max(logits, -1).indices  # N
    contrastive_accuracy = torch.tensor(
        float((max_indices == labels).sum()) / float(len(logits))
    )

    return loss, contrastive_accuracy


def calculate_cif_contrastive_semantic_dis_cos_loss(
    student_feats,
    teacher_feats,
    temperature=0.1,
    num_contrastive_negative_samples=50,
    remove_overlap_in_negs=False,
    sample_std_negs=False,
):
    """
    :param student_feats: tensor with shape B x C,
    :param teacher_feats: tensor with shape B x C,
    :param temperature: optional in [0, 1], recommend [0.1, 0.5],
    :param num_contrastive_negative_samples: default = 50,
    :return: cif_contrastive_semantic_dis_cos_loss,
    """

    num_samples = teacher_feats.size(0)
    if num_samples <= num_contrastive_negative_samples:
        num_contrastive_negative_samples = int(num_samples * 0.8)
    assert (
        num_samples > num_contrastive_negative_samples
    ), "number of negative samples must smaller than that of total samples"

    # 1. Create negative keys
    sampling_weights = (
        torch.ones([num_samples, num_samples]).type_as(teacher_feats).cuda()
    )  # N x N
    neg_ids = torch.multinomial(
        sampling_weights,
        num_samples=num_contrastive_negative_samples,
        replacement=False,
    ).long()  # N x N_negs

    # sample negative samples from students
    if sample_std_negs:
        mix_probs = torch.rand([num_samples, 1]).cuda()
        negative_samples = torch.where(
            (mix_probs.repeat(1, cif_feats.size(-1)) > 0.5).bool(),
            teacher_feats,
            student_feats,
        )[
            neg_ids
        ]  # N x N_negs x C
    else:
        negative_samples = teacher_feats[neg_ids]  # N x N_negs x C

    if remove_overlap_in_negs:
        triangle_ids = (
            torch.ones_like(neg_ids).cuda()
            * torch.tensor([i for i in range(num_samples)]).unsqueeze(-1).cuda()
        )  # N x N_negs
        negative_samples = torch.where(
            (neg_ids == triangle_ids)
            .unsqueeze(-1)
            .repeat(1, 1, negative_samples.size(-1))
            .bool(),
            torch.randn_like(negative_samples),
            negative_samples,
        )  # N x N_negs x C

    # 2. Normalize queries and all keys
    def normalize(*xs):
        return [None if x is None else F.normalize(x, dim=-1, eps=1e-6) for x in xs]

    student_feats, teacher_feats, negative_samples = normalize(
        student_feats, teacher_feats, negative_samples
    )

    # 3. Calculate logits
    positive_logit = torch.sum(
        student_feats * teacher_feats, dim=-1, keepdim=True
    )  # N x 1
    negative_logits = torch.matmul(
        student_feats.unsqueeze(1), negative_samples.transpose(-2, -1)
    ).squeeze(
        1
    )  # N x N_negs
    logits = torch.cat([positive_logit, negative_logits], dim=-1)  # N x (N_negs + 1)
    labels = torch.zeros(len(logits)).long().cuda()

    # 4. Caluclate loss
    loss = F.cross_entropy(logits.float() / temperature, labels, reduction="none")

    # 5. Calculate contrastive accuracy
    max_indices = torch.max(logits, -1).indices  # N
    contrastive_accuracy = torch.tensor(
        float((max_indices == labels).sum()) / float(len(logits))
    )

    return loss, contrastive_accuracy


def calculate_constrastive_distillation_loss(
    student_feats,
    teacher_feats,
    temperature=0.1,
    num_contrastive_negative_samples=50,
    remove_overlap_in_negs=False,
    sample_std_negs=False,
):
    """
    :param student_feats: tensor with shape B x C,
    :param teacher_feats: tensor with shape B x C,
    :param temperature: optional in [0, 1], recommend [0.1, 0.5],
    :param num_contrastive_negative_samples: default = 50,
    :return: cif_contrastive_semantic_dis_cos_loss,
    """

    num_samples = teacher_feats.size(0)
    if num_samples <= num_contrastive_negative_samples:
        num_contrastive_negative_samples = int(num_samples * 0.8)
    assert (
        num_samples > num_contrastive_negative_samples
    ), "number of negative samples must smaller than that of total samples"

    # 1. Create negative keys
    sampling_weights = (
        torch.ones([num_samples, num_samples]).type_as(teacher_feats).cuda()
    )  # N x N
    neg_ids = torch.multinomial(
        sampling_weights,
        num_samples=num_contrastive_negative_samples,
        replacement=False,
    ).long()  # N x N_negs

    # sample negative samples from students
    if sample_std_negs:
        mix_probs = torch.rand([num_samples, 1]).cuda()
        negative_samples = torch.where(
            (mix_probs.repeat(1, cif_feats.size(-1)) > 0.5).bool(),
            teacher_feats,
            student_feats,
        )[
            neg_ids
        ]  # N x N_negs x C
    else:
        negative_samples = teacher_feats[neg_ids]  # N x N_negs x C

    # 2. Normalize queries and all keys
    def normalize(*xs):
        return [None if x is None else F.normalize(x, dim=-1, eps=1e-6) for x in xs]

    student_feats, teacher_feats, negative_samples = normalize(
        student_feats, teacher_feats, negative_samples
    )

    # 3. Calculate logits
    positive_logit = torch.sum(
        student_feats * teacher_feats, dim=-1, keepdim=True
    )  # N x 1
    negative_logits = torch.matmul(
        student_feats.unsqueeze(1), negative_samples.transpose(-2, -1)
    ).squeeze(
        1
    )  # N x N_negs
    if remove_overlap_in_negs:
        triangle_ids = (
            torch.ones_like(neg_ids).cuda()
            * torch.tensor([i for i in range(num_samples)]).unsqueeze(-1).cuda()
        )  # N x N_negs
        negative_logits = torch.where(
            (neg_ids == triangle_ids).bool(),  # N x N_negs
            torch.ones_like(negative_logits) * -1e9,
            negative_logits,
        )  # N x N_negs x C

    logits = torch.cat([positive_logit, negative_logits], dim=-1)  # N x (N_negs + 1)
    labels = torch.zeros(len(logits)).long().cuda()

    # 4. Caluclate loss
    loss = F.cross_entropy(logits.float() / temperature, labels, reduction="none")

    # 5. Calculate contrastive accuracy
    max_indices = torch.max(logits, -1).indices  # N
    contrastive_accuracy = torch.tensor(
        float((max_indices == labels).sum()) / float(len(logits))
    )

    return loss, contrastive_accuracy


@dataclass
class CifCriterionConfig(FairseqDataclass):
    # Settings of basic Cif losses
    no_quantity_loss: bool = field(
        default=False, metadata={"help": "apply quantity loss"}
    )
    no_ctc_loss: bool = field(default=False, metadata={"help": "apply ctc loss"})
    apply_align_loss: bool = field(default=False, metadata={"help": "apply align loss"})
    quantity_loss_lambda: float = field(
        default=1.0, metadata={"help": "the interpolation weight of quantity loss"}
    )
    boost_qtt_loss: bool = field(
        default=False,
        metadata={"help": "whether to boost the contribution of quantity loss"},
    )
    ctc_loss_lambda: float = field(
        default=0.25, metadata={"help": "the interpolation weight of ctc loss"}
    )
    align_loss_lambda: float = field(
        default=1.0,
        metadata={"help": "the interpolation weight of ctc-constrained alignment loss"},
    )

    # Settings for distillation losses
    apply_cif_dis_loss: bool = field(default=False)
    cif_dis_loss_lambda: float = field(default=0.000)

    apply_cif_dis_cos_loss: bool = field(default=False)
    cif_dis_cos_loss_lambda: float = field(default=0.000)
    cif_dis_cos_loss_boost: float = field(default=1.0)

    apply_cif_contrastive_dis_cos_loss: bool = field(default=False)
    cif_contrastive_dis_cos_loss_lambda: float = field(default=0.000)
    cif_contrastive_dis_cos_loss_type: str = field(default="cos")
    arccos_margin: float = field(default=2.0)
    num_contrastive_negative_samples: int = field(default=300)
    contrastive_temperature: float = field(default=0.1)

    apply_cif_semantic_dis_loss: bool = field(default=False)
    cif_semantic_dis_loss_lambda: float = field(default=0.000)

    apply_cif_cont_semantic_dis_loss: bool = field(default=False)
    cif_cont_semantic_dis_loss_lambda: float = field(default=0.0)
    num_contrastive_semantic_negative_samples: int = field(default=50)

    apply_dec_state_dis_loss: bool = field(default=False)
    dec_state_dis_loss_lambda: float = field(default=0.000)

    apply_dec_state_cont_dis_loss: bool = field(default=False)
    dec_state_cont_dis_loss_lambda: float = field(default=0.00)
    dec_state_cont_dis_lambda_scheduler: str = field(default="")

    # General settings for distillation losses
    no_dim_scaling_for_mse_loss: bool = field(default=False)
    remove_overlap_in_negs: bool = field(default=False)
    sample_std_negs: bool = field(default=False)
    mix_std_neg_ratio: float = field(default=0.0)

    # Label smoothing settings
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

    # Focal loss settings
    apply_focal_loss: bool = field(default=False)
    focal_loss_lambda: float = field(default=0.0)
    focal_loss_gamma: float = field(default=1.0)

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
    no_eos_label: bool = field(default=False)
    no_comb_loss_sum: bool = field(default=False)


@register_criterion("cif", dataclass=CifCriterionConfig)
class CifCriterion(FairseqCriterion):
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
        self.post_process = cfg.post_process
        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

        # Register losses settings
        self.apply_quantity_loss = not cfg.no_quantity_loss
        self.apply_ctc_loss = not cfg.no_ctc_loss
        self.apply_cif_dis_loss = cfg.apply_cif_dis_loss
        self.apply_cif_semantic_dis_loss = cfg.apply_cif_semantic_dis_loss
        self.apply_dec_state_dis_loss = cfg.apply_dec_state_dis_loss
        self.apply_dec_state_cont_dis_loss = cfg.apply_dec_state_cont_dis_loss
        self.dec_state_cont_dis_loss_lambda = cfg.dec_state_cont_dis_loss_lambda
        self.dec_state_cont_dis_lambda_scheduler = (
            cfg.dec_state_cont_dis_lambda_scheduler
        )
        self.apply_align_loss = cfg.apply_align_loss
        self.quantity_loss_lambda = cfg.quantity_loss_lambda
        self.boost_qtt_loss = cfg.boost_qtt_loss
        self.ctc_loss_lambda = cfg.ctc_loss_lambda
        self.align_loss_lambda = cfg.align_loss_lambda
        self.cif_dis_loss_lambda = cfg.cif_dis_loss_lambda
        self.cif_semantic_dis_loss_lambda = cfg.cif_semantic_dis_loss_lambda
        self.dec_state_dis_loss_lambda = cfg.dec_state_dis_loss_lambda
        self.no_dim_scaling_for_mse_loss = cfg.no_dim_scaling_for_mse_loss

        # Contrastive loss settings
        self.apply_cif_dis_cos_loss = cfg.apply_cif_dis_cos_loss
        self.cif_dis_cos_loss_lambda = cfg.cif_dis_cos_loss_lambda
        self.cif_dis_cos_loss_boost = cfg.cif_dis_cos_loss_boost
        self.apply_cif_contrastive_dis_cos_loss = cfg.apply_cif_contrastive_dis_cos_loss
        self.cif_contrastive_dis_cos_loss_lambda = (
            cfg.cif_contrastive_dis_cos_loss_lambda
        )
        self.cif_contrastive_dis_cos_loss_type = cfg.cif_contrastive_dis_cos_loss_type
        self.arccos_margin = cfg.arccos_margin
        self.num_contrastive_negative_samples = cfg.num_contrastive_negative_samples
        self.contrastive_temperature = cfg.contrastive_temperature
        self.apply_cif_cont_semantic_dis_loss = cfg.apply_cif_cont_semantic_dis_loss
        self.cif_cont_semantic_dis_loss_lambda = cfg.cif_cont_semantic_dis_loss_lambda
        self.num_contrastive_semantic_negative_samples = (
            cfg.num_contrastive_semantic_negative_samples
        )

        self.remove_overlap_in_negs = cfg.remove_overlap_in_negs
        self.sample_std_negs = cfg.sample_std_negs
        self.mix_std_neg_ratio = cfg.mix_std_neg_ratio

        # Register label smoothing settings
        self.label_smoothing_type = cfg.label_smoothing_type
        self.label_smoothing_rate = cfg.label_smoothing_rate
        self.apply_label_smoothing = cfg.apply_label_smoothing
        self.apply_focal_loss = cfg.apply_focal_loss
        self.focal_loss_lambda = cfg.focal_loss_lambda
        self.focal_loss_gamma = cfg.focal_loss_gamma

        self.no_eos_label = cfg.no_eos_label
        self.no_comb_loss_sum = cfg.no_comb_loss_sum

    def get_loss(self, model, sample, net_output, reduce=True):
        num_updates = model.num_updates

        # Get model outputs
        ctc_logits = net_output["ctc_logits"]  # B x T x V
        quantity_out = net_output["quantity_out"]  # 1
        decoder_out = net_output["decoder_out"][
            0
        ]  # Get final decoder outputs (logits for cross-entropy loss)
        # decoder_states = net_output["decoder_out"][-1]  # Get the decoder states (states, not logits for CE calculation)
        cif_out_padding_mask = net_output["cif_out_padding_mask"]  # B x T

        # Collect src_lengths for the calculation of ctc loss
        non_padding_mask = ~net_output["encoder_padding_mask"]
        input_lengths = non_padding_mask.int().sum(-1)

        # CTC alignment outputs
        ctc_align_outputs = net_output["ctc_align_outputs"]

        # Collect targets and target_length for ctc loss and ce loss
        target_lengths = sample["target_lengths"]  # targets length without eos
        target_with_eos = sample["target"]
        target_with_eos_lengths = target_lengths  # targets length with eos
        if self.no_eos_label:
            target_with_eos_lengths = target_with_eos_lengths - 1
            target_with_eos = torch.where(
                target_with_eos != self.eos_idx,
                target_with_eos,
                self.pad_idx * torch.ones_like(target_with_eos),
            )
        adjusted_target_with_eos = target_with_eos

        # Calculate the ctc loss on encoder outputs
        ctc_loss = torch.tensor(0.0)
        if self.apply_ctc_loss:
            pad_mask = adjusted_target_with_eos != self.pad_idx
            targets_flat = adjusted_target_with_eos.masked_select(pad_mask)
            ctc_lprobs = model.get_probs_from_logits(
                ctc_logits, log_probs=True
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

        # Calculate CTC alignment loss
        align_loss = torch.tensor(0.0)
        if self.apply_align_loss and ctc_align_outputs is not None:
            align_self_padding_mask = ~(
                ctc_align_outputs.eq(0.0)
            )  # B x T, where padding locations are False
            align_loss = torch.abs(ctc_align_outputs - 1.0) * align_self_padding_mask
            align_loss = align_loss.sum()

        # Calculate the quantity loss
        qtt_loss = torch.tensor(0.0)
        if self.apply_quantity_loss:
            target_lengths_for_qtt_loss = (
                target_with_eos_lengths  # Lengths after adding eos token, [B]
            )
            qtt_loss = torch.abs(quantity_out - target_lengths_for_qtt_loss).sum()

        # Process bert feats
        bert_distill_feats = None
        token_distill_cif_feat = None
        processed_bert_distill_feats = None
        if (
            self.apply_cif_semantic_dis_loss
            or self.apply_cif_dis_loss
            or self.apply_cif_dis_cos_loss
            or self.apply_dec_state_dis_loss
            or self.apply_cif_contrastive_dis_cos_loss
            or self.apply_cif_cont_semantic_dis_loss
            or self.apply_dec_state_cont_dis_loss
        ):

            bert_distill_feats = sample["bert_distill_feats"]  # B x T x C
            processed_bert_distill_feats = bert_distill_feats[:, 1:, :]  # B x T x C
            token_distill_cif_feat = net_output[
                "token_distill_cif_feat"
            ]  # B x T x C_bert
            reg_len = min(
                processed_bert_distill_feats.size(1), token_distill_cif_feat.size(1)
            )
            cif_out_padding_mask = cif_out_padding_mask[:, :reg_len]
            processed_bert_distill_feats = processed_bert_distill_feats[
                :, :reg_len, :
            ] * cif_out_padding_mask.unsqueeze(
                -1
            )  # B x T x C
            token_distill_cif_feat = token_distill_cif_feat[
                :, :reg_len, :
            ] * cif_out_padding_mask.unsqueeze(
                -1
            )  # B x T x C

        # Calculate cif distillation loss
        cif_dis_loss = torch.tensor(0.0)
        if self.apply_cif_dis_loss:
            if self.no_dim_scaling_for_mse_loss:
                cif_dis_loss = F.mse_loss(
                    token_distill_cif_feat.float(),
                    processed_bert_distill_feats.float(),
                    reduction="none",
                ).sum()
            else:
                cif_dis_loss = (
                    F.mse_loss(
                        token_distill_cif_feat.float(),
                        processed_bert_distill_feats.float(),
                        reduction="none",
                    )
                    .mean(-1)
                    .sum()
                )

        # Calculate cif distillation loss with cosine similiarity
        cif_dis_cos_loss = torch.tensor(0.0)
        if self.apply_cif_dis_cos_loss:
            bsz, tsz, dsz = token_distill_cif_feat.size()
            cif_feat_for_cos = token_distill_cif_feat.view(bsz * tsz, -1)
            bert_feat_for_cos = processed_bert_distill_feats.view(bsz * tsz, -1)
            cif_mask = cif_out_padding_mask.contiguous().view(bsz * tsz)
            cif_feat_for_cos = cif_feat_for_cos[cif_mask]
            bert_feat_for_cos = bert_feat_for_cos[cif_mask]

            cif_dis_cos_loss = F.cosine_embedding_loss(
                cif_feat_for_cos.float(),
                bert_feat_for_cos.float(),
                target=torch.ones(cif_feat_for_cos.size(0)).long().cuda(),
                reduction="none",
            )
            cif_dis_cos_loss = self.cif_dis_cos_loss_boost * cif_dis_cos_loss.sum()

        # Calculate cif contrastive distillation loss
        cont_acc = torch.tensor(0.0)
        cif_contrastive_dis_cos_loss = torch.tensor(0.0)
        if self.apply_cif_contrastive_dis_cos_loss:
            (
                cif_contrastive_dis_cos_loss,
                cont_acc,
            ) = calculate_cif_contrastive_dis_cos_loss(
                cif_feats=token_distill_cif_feat,
                bert_feats=processed_bert_distill_feats,
                mask=cif_out_padding_mask,
                temperature=self.contrastive_temperature,
                num_contrastive_negative_samples=self.num_contrastive_negative_samples,
                remove_overlap_in_negs=self.remove_overlap_in_negs,
                sample_std_negs=self.sample_std_negs,
                loss_type=self.cif_contrastive_dis_cos_loss_type,
                arccos_margin=self.arccos_margin,
            )
            cif_contrastive_dis_cos_loss = cif_contrastive_dis_cos_loss.sum()

        # Calculate semantic cif distillation loss
        cif_semantic_dis_loss = torch.tensor(0.0)
        if self.apply_cif_semantic_dis_loss:
            processed_bert_semantic_distill_feats = bert_distill_feats[:, 0, :]  # B x C
            semantic_distill_cif_feat = net_output[
                "semantic_distill_cif_feat"
            ]  # B x C_bert

            if self.no_dim_scaling_for_mse_loss:
                cif_semantic_dis_loss = F.mse_loss(
                    semantic_distill_cif_feat.float(),
                    processed_bert_semantic_distill_feats.float(),
                    reduction="none",
                ).sum()
            else:
                cif_semantic_dis_loss = (
                    F.mse_loss(
                        semantic_distill_cif_feat.float(),
                        processed_bert_semantic_distill_feats.float(),
                        reduction="none",
                    )
                    .mean(-1)
                    .sum()
                )

        # Calculate contrastive semantic cif distillation loss
        cont_semantic_acc = torch.tensor(0.0)
        cif_cont_semantic_dis_loss = torch.tensor(0.0)
        if self.apply_cif_cont_semantic_dis_loss:
            processed_bert_semantic_distill_feats = bert_distill_feats[:, 0, :]  # B x C
            semantic_distill_cif_feat = net_output[
                "semantic_distill_cif_feat"
            ]  # B x C_bert

            (
                cif_cont_semantic_dis_loss,
                cont_semantic_acc,
            ) = calculate_cif_contrastive_semantic_dis_cos_loss(
                student_feats=semantic_distill_cif_feat,
                teacher_feats=processed_bert_semantic_distill_feats,
                temperature=self.contrastive_temperature,
                num_contrastive_negative_samples=self.num_contrastive_semantic_negative_samples,
                remove_overlap_in_negs=self.remove_overlap_in_negs,
                sample_std_negs=self.sample_std_negs,
            )
            cif_cont_semantic_dis_loss = cif_cont_semantic_dis_loss.sum()

        # Calculate decoder state distillation loss
        dec_state_dis_loss = torch.tensor(0.0)
        dec_state_cont_dis_loss = torch.tensor(0.0)
        dec_state_cont_dis_acc = torch.tensor(0.0)
        dec_state_cont_dis_lambda_scale = 1.0
        if self.apply_dec_state_dis_loss:
            processed_bert_distill_feats = bert_distill_feats[:, 1:, :]  # B x T x C
            token_distill_decoder_states = net_output[
                "token_distill_decoder_states"
            ]  # B x T x C

            # regularize length
            reg_len = min(
                processed_bert_distill_feats.size(1),
                token_distill_decoder_states.size(1),
            )
            cif_out_padding_mask = cif_out_padding_mask[:, :reg_len]
            processed_bert_distill_feats = processed_bert_distill_feats[
                :, :reg_len, :
            ] * cif_out_padding_mask.unsqueeze(-1)
            token_distill_decoder_states = token_distill_decoder_states[
                :, :reg_len, :
            ] * cif_out_padding_mask.unsqueeze(
                -1
            )  # B x T x C

            # calculate scaling factor
            if self.no_dim_scaling_for_mse_loss:
                dec_state_dis_loss = F.mse_loss(
                    token_distill_decoder_states.float(),
                    processed_bert_distill_feats.float(),
                    reduction="none",
                ).sum()
            else:
                dec_state_dis_loss = (
                    F.mse_loss(
                        token_distill_decoder_states.float(),
                        processed_bert_distill_feats.float(),
                        reduction="none",
                    )
                    .mean(-1)
                    .sum()
                )

            # calculate the contrastive distillation loss for the decoder state
            if self.apply_dec_state_cont_dis_loss:
                bsz, tsz, _ = token_distill_decoder_states.size()
                student_feats = token_distill_decoder_states.view(
                    bsz * tsz, -1
                )  # (bsz * tsz) * C = B x C
                teacher_feats = processed_bert_distill_feats.view(
                    bsz * tsz, -1
                )  # (bsz * tsz) * C = B x C
                mask = cif_out_padding_mask.contiguous().view(bsz * tsz)
                student_feats = student_feats[mask]  # N x C
                teacher_feats = teacher_feats[mask]  # N x C

                (
                    dec_state_cont_dis_loss,
                    dec_state_cont_dis_acc,
                ) = calculate_constrastive_distillation_loss(
                    student_feats=student_feats,
                    teacher_feats=teacher_feats,
                    temperature=self.contrastive_temperature,
                    num_contrastive_negative_samples=self.num_contrastive_negative_samples,
                    remove_overlap_in_negs=self.remove_overlap_in_negs,
                    sample_std_negs=self.sample_std_negs,
                )
                dec_state_cont_dis_loss = dec_state_cont_dis_loss.type_as(
                    token_distill_decoder_states
                ).sum()

                if self.dec_state_cont_dis_lambda_scheduler == "exp_decay":
                    k = 1.918820910828371e-05
                    dec_state_cont_dis_lambda_scale = math.exp(-1 * k * num_updates)

        # Calculate the cross-entropy loss
        cif_max_len = cif_out_padding_mask.size(1)  # Get max length of cif outputs
        tgt_max_len = target_with_eos_lengths.max()  # Get max length of targets
        reg_min_len = min(
            cif_max_len, tgt_max_len
        )  # Obtain the minimum length of cif length and target length
        ce_logprobs = model.get_probs_from_logits(
            decoder_out, log_probs=True
        )  # B x T x V
        truncated_target = adjusted_target_with_eos[
            :, :reg_min_len
        ]  # Truncate target to reg_min_len, B x T
        truncated_ce_logprobs = ce_logprobs[
            :, :reg_min_len, :
        ]  # Truncate ce probs to reg_min_len,  B x T x V
        # Truncate target to the minimum length of original target and cif outputs,
        # because sometimes the firing number of CIF may drop <eos>.

        # Apply focal loss
        if self.apply_focal_loss:
            # truncated_ce_logprobs B x T x V
            ce_probs = model.get_probs_from_logits(
                decoder_out, log_probs=False
            ).contiguous()[
                :, :reg_min_len, :
            ]  # B x T x V
            uncertainty = (1 - ce_probs) ** self.focal_loss_gamma  # B x T x V
            truncated_ce_logprobs = (
                uncertainty * truncated_ce_logprobs
            )  # (1 - p_k) * log(p_k)

        # Calculate cross-entropy loss
        if not self.apply_label_smoothing:
            truncated_ce_logprobs = truncated_ce_logprobs.contiguous().view(
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
            )  # CE loss is the summation of all tokens, without any form of averaging
            nll_loss = ce_loss
        else:
            if self.label_smoothing_type == "uniform":
                ce_loss, nll_loss = label_smoothed_nll_loss(
                    truncated_ce_logprobs,
                    truncated_target.long(),
                    self.label_smoothing_rate,
                    self.pad_idx,
                    reduce=True if reduce else False,
                )
            else:
                raise NotImplementedError(
                    "Invalid option: %s" % self.label_smoothing_type
                )

        # Collect the number of tokens in current batch
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )
        ntokens_with_eos = target_with_eos_lengths.sum().item()
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens

        if self.boost_qtt_loss:
            qtt_loss = qtt_loss * (ntokens / sample["id"].numel())
            # boosted by averge length of each reference

        loss = (
            ce_loss
            + self.quantity_loss_lambda * qtt_loss
            + self.ctc_loss_lambda * ctc_loss
            + self.align_loss_lambda * align_loss
            + self.cif_dis_loss_lambda * cif_dis_loss
            + self.cif_semantic_dis_loss_lambda * cif_semantic_dis_loss
            + self.dec_state_dis_loss_lambda * dec_state_dis_loss
            + self.cif_dis_cos_loss_lambda * cif_dis_cos_loss
            + self.cif_contrastive_dis_cos_loss_lambda * cif_contrastive_dis_cos_loss
            + self.cif_cont_semantic_dis_loss_lambda * cif_cont_semantic_dis_loss
            + self.dec_state_cont_dis_loss_lambda
            * dec_state_cont_dis_lambda_scale
            * dec_state_cont_dis_loss
        )

        # Build final logging outputs
        logging_output = {
            "loss": utils.item(loss.data),
            "ce_loss": utils.item(nll_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "align_loss": utils.item(align_loss.data),
            "quantity_loss": utils.item(qtt_loss.data),
            "cif_dis_loss": utils.item(cif_dis_loss.data),
            "cif_semantic_dis_loss": utils.item(cif_semantic_dis_loss.data),
            "dec_state_dis_loss": utils.item(dec_state_dis_loss.data),
            "cif_dis_cos_loss": utils.item(cif_dis_cos_loss.data),
            "cif_contrastive_dis_cos_loss": utils.item(
                cif_contrastive_dis_cos_loss.data
            ),
            "cif_cont_semantic_dis_loss": utils.item(cif_cont_semantic_dis_loss.data),
            "dec_state_cont_dis_loss": utils.item(dec_state_cont_dis_loss.data),
            "cont_acc": cont_acc.item(),
            "cont_semantic_acc": cont_semantic_acc.item(),
            "dec_state_cont_dis_acc": dec_state_cont_dis_acc.item(),
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

                # Loop over all hypothesis
                for lp, t, inp_l in zip(
                    lprobs_t, adjusted_target_with_eos, cif_lengths
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    # Process targets
                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    # p = (t != self.task.target_dictionary.pad())
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    # print("______________________________________")
                    # print(targ_units_arr)

                    # Process hypothesis
                    # If decoded is None, conduct greedy search decoding
                    # toks = lp.argmax(dim=-1).unique_consecutive()
                    # For ctc decoding, remove blank indices and repetitive consecutive ids

                    # Handle lp without elements
                    if min(lp.shape) == 0:
                        toks = targ
                    else:
                        toks = lp.argmax(dim=-1)

                    # toks = lp.argmax(dim=-1)    # For ce decoding
                    pred_units_arr = toks[
                        (toks != self.blank_idx)
                        & (toks != self.pad_idx)
                        & (toks != self.eos_idx)
                    ].tolist()
                    # pred_units_arr = \
                    #     toks[(toks != self.blank_idx) & (toks != self.pad_idx)].tolist()

                    # Calculate character error
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()
                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

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

        # Calculate the total loss
        if self.no_comb_loss_sum:
            loss = (
                ce_loss / ntokens
                + self.quantity_loss_lambda * qtt_loss / ntokens
                + self.ctc_loss_lambda * ctc_loss / ntokens
                + self.align_loss_lambda * align_loss / ntokens
            )

        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        # forward the whole model
        net_output = model(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            target_lengths=sample["target_lengths"],
            target=sample["target"],
            bert_doc_feats=sample["net_input"]["bert_doc_feats"],
            vit_image_feats=sample["net_input"]["vit_image_feats"],
        )
        loss, sample_size, logging_output = self.get_loss(
            model, sample, net_output, reduce=True
        )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """
        Aggregate logging outputs from data parallel training.
        """

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ce_loss_sum = utils.item(sum(log.get("ce_loss", 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(
            sum(log.get("ctc_loss", 0) for log in logging_outputs)
        )
        align_loss_sum = utils.item(
            sum(log.get("align_loss", 0) for log in logging_outputs)
        )
        quantity_loss_sum = utils.item(
            sum(log.get("quantity_loss", 0) for log in logging_outputs)
        )
        cif_dis_loss_sum = utils.item(
            sum(log.get("cif_dis_loss", 0) for log in logging_outputs)
        )
        cif_dis_semantic_loss_sum = utils.item(
            sum(log.get("cif_semantic_dis_loss", 0) for log in logging_outputs)
        )
        dec_state_dis_loss_sum = utils.item(
            sum(log.get("dec_state_dis_loss", 0) for log in logging_outputs)
        )
        dec_state_cont_dis_loss_sum = utils.item(
            sum(log.get("dec_state_cont_dis_loss", 0) for log in logging_outputs)
        )
        cif_dis_cos_loss_sum = utils.item(
            sum(log.get("cif_dis_cos_loss", 0) for log in logging_outputs)
        )
        cif_contrastive_dis_cos_loss_sum = utils.item(
            sum(log.get("cif_contrastive_dis_cos_loss", 0) for log in logging_outputs)
        )
        cif_cont_semantic_dis_loss_sum = utils.item(
            sum(log.get("cif_cont_semantic_dis_loss", 0) for log in logging_outputs)
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
        cont_acc_sum = utils.item(
            sum(log.get("cont_acc", 0) for log in logging_outputs)
        )
        cont_semantic_acc_sum = utils.item(
            sum(log.get("cont_semantic_acc", 0) for log in logging_outputs)
        )
        dec_state_cont_dis_acc_sum = utils.item(
            sum(log.get("dec_state_cont_dis_acc", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "ce_loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "normal_nll_loss",
            ce_loss_sum / sample_size / math.log2(2),
            sample_size,
            round=5,
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "align_loss",
            align_loss_sum / sample_size / math.log(2),
            sample_size,
            round=5,
        )
        metrics.log_scalar(
            "quantity_loss",
            quantity_loss_sum / sample_size / math.log(2),
            sample_size,
            round=5,
        )
        metrics.log_scalar(
            "cif_dis_loss",
            cif_dis_loss_sum / sample_size / math.log(2),
            sample_size,
            round=5,
        )
        metrics.log_scalar(
            "cif_dis_semantic_loss",
            cif_dis_semantic_loss_sum / sample_size / math.log(2),
            sample_size,
            round=5,
        )
        metrics.log_scalar(
            "dec_state_dis_loss",
            dec_state_dis_loss_sum / sample_size / math.log(2),
            sample_size,
            round=5,
        )
        metrics.log_scalar(
            "dec_state_cont_dis_loss",
            dec_state_cont_dis_loss_sum / sample_size / math.log(2),
            sample_size,
            round=5,
        )
        metrics.log_scalar(
            "cif_dis_cos_loss",
            cif_dis_cos_loss_sum / sample_size / math.log(2),
            sample_size,
            round=5,
        )
        metrics.log_scalar(
            "cif_contrastive_dis_cos_loss",
            cif_contrastive_dis_cos_loss_sum / sample_size / math.log(2),
            sample_size,
            round=5,
        )
        metrics.log_scalar(
            "cif_cont_semantic_dis_loss",
            cif_cont_semantic_dis_loss_sum / sample_size / math.log(2),
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
        metrics.log_scalar("contrastive_accuracy", cont_acc_sum / len(logging_outputs))
        metrics.log_scalar(
            "semantic_contrastive_accuracy",
            cont_semantic_acc_sum / len(logging_outputs),
        )
        metrics.log_scalar(
            "dec_state_contrastive_accuracy",
            dec_state_cont_dis_acc_sum / len(logging_outputs),
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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """

        return True
