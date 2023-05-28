# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import random
import os
import sys
import numpy as np

from argparse import Namespace
from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import (
    FairseqDataset,
    BinarizedAudioDataset,
    FileAudioDataset,
    MixedModalAudioDataset,
    FileMixedModalAudioDataset,
)
from fairseq.data import data_utils
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.data.text_compressor import TextCompressionLevel
from fairseq.data import Dictionary, FairseqDataset, data_utils, encoders, iterators

from . import FairseqTask, register_task
from fairseq.data.dictionary import Dictionary


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@dataclass
class InferredW2vConfig:
    # The following are needed to precompute mask and mask channel indices
    #   before model's forward.
    mask_length: Optional[int] = II("model.mask_length")
    mask_prob: Optional[float] = II("model.mask_prob")
    mask_selection: Optional[str] = II("model.mask_selection")
    mask_other: Optional[float] = II("model.mask_other")
    no_mask_overlap: Optional[bool] = II("model.no_mask_overlap")
    mask_min_space: Optional[int] = II("model.mask_min_space")
    mask_channel_length: Optional[int] = II("model.mask_channel_length")
    mask_channel_prob: Optional[float] = II("model.mask_channel_prob")
    mask_channel_selection: Optional[str] = II("model.mask_channel_selection")
    mask_channel_other: Optional[float] = II("model.mask_channel_other")
    no_mask_channel_overlap: Optional[bool] = II("model.no_mask_channel_overlap")
    mask_channel_min_space: Optional[int] = II("model.mask_channel_min_space")

    conv_feature_layers: Optional[str] = II("model.conv_feature_layers")
    encoder_embed_dim: Optional[int] = II("model.encoder_embed_dim")


@dataclass
class SpeechTextPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    binarized_dataset: bool = field(
        default=False,
        metadata={
            "help": "if true, loads binarized dataset (useful for very large datasets). "
            "See examples/wav2vec/scripts/binarize_manifest.sh"
        },
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )
    min_text_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small text examples"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    data_ratio: str = field(
        default="{'text': 1000000, 'speech': 1, 'pair': 1}",
        metadata={"help": "The ratio of different data modes."},
    )
    default_dom_cls: str = field(
        default="text", metadata={"help": "the dominant modality for data sampling"}
    )
    precompute_mask_indices: bool = field(
        default=False,
        metadata={
            "help": "flag to compute mask indices in data preparation.",
        },
    )

    inferred_w2v_config: Optional[InferredW2vConfig] = field(
        default=None,
        metadata={
            "help": "wav2vec 2.0 masking arguments used to pre-compute masks (required for TPU)",
        },
    )

    tpu: bool = II("common.tpu")
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )


@register_task("speech_text_pretraining", dataclass=SpeechTextPretrainingConfig)
class SpeechTextPretrainingTask(FairseqTask):
    """ """

    cfg: SpeechTextPretrainingConfig

    def __init__(self, cfg: SpeechTextPretrainingConfig):
        super().__init__(cfg)
        self.blank_symbol = "<s>"
        self.state.add_factory("target_dictionary", self.load_target_dictionary)
        self.state.add_factory("default_dictionary", self.load_target_dictionary)

        self.default_dom_cls = cfg.default_dom_cls

    @classmethod
    def setup_task(cls, cfg: SpeechTextPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (SpeechTextPretrainingConfig): configuration of this task
        """
        return cls(cfg)

    def _get_mask_precompute_kwargs(self, cfg):
        if self.cfg.precompute_mask_indices or self.cfg.tpu:
            assert (
                cfg.inferred_w2v_config is not None
            ), "inferred_w2v_config must be set"
            return OmegaConf.to_container(
                cfg.inferred_w2v_config, resolve=True, enum_to_str=True
            )
        else:
            return {}

    def load_target_dictionary(self):
        if self.cfg.labels:
            dict_path = os.path.join(self.cfg.data, f"dict.{self.cfg.labels}.txt")
            return Dictionary.load(dict_path)
        return None

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg
        self.data_ratio = task_cfg.data_ratio
        self.min_text_size = task_cfg.min_text_size

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )

        # Obtain text processer function
        process_label = LabelEncoder(self.target_dictionary)

        text_config = dict()
        text_config["batch_targets"] = True
        text_config["pad"] = self.target_dictionary.pad()
        text_config["eos"] = self.target_dictionary.eos()
        text_config["process_label"] = process_label
        text_config["add_to_input"] = False
        text_config["min_text_size"] = self.min_text_size

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))
        label_path = os.path.join(data_path, "{}.{}".format(split, self.cfg.labels))

        self.datasets[split] = FileMixedModalAudioDataset(
            manifest_path=manifest_path,
            label_path=label_path,
            sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            shuffle=True,
            pad=task_cfg.labels is not None or task_cfg.enable_padding,
            normalize=task_cfg.normalize,
            num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
            compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
            text_compression_level=text_compression_level,
            text_config=text_config,
            **self._get_mask_precompute_kwargs(task_cfg),
        )

        if self.cfg.tpu and task_cfg.inferred_w2v_config.mask_channel_prob == 0.0:
            logger.info(
                "Pretraining on TPUs may suffer convergence "
                "issues when training with `mask_channel_prob` value of "
                "0. You may want to set this to a low value close to 0."
            )

    @property
    def source_dictionary(self):
        return None

    @property
    def default_dictionary(self):
        return self.state.default_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)
        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            # if "w2v_args" in actualized_cfg:
            if hasattr(actualized_cfg, "w2v_args"):
                model_cfg.w2v_args = actualized_cfg.w2v_args
        return model

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = (
            not disable_iterator_cache
            and not update_epoch_batch_itr
            and self.can_reuse_epoch_itr(dataset)
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get all indices ordered by example size for each data mode
        data_cls_batch_ratio = eval(
            self.data_ratio
        )  # Obtain ratios for different data class
        total_indices = dict()
        with data_utils.numpy_seed(seed):
            if "text" in data_cls_batch_ratio.keys():
                text_indices = dataset.ordered_indices("text")
                total_indices["text"] = text_indices

            if "speech" in data_cls_batch_ratio.keys():
                spec_indices = dataset.ordered_indices("speech")
                total_indices["speech"] = spec_indices

            if "pair" in data_cls_batch_ratio.keys():
                pair_indices = dataset.ordered_indices("pair")
                total_indices["pair"] = pair_indices

        # filter examples that are too large
        if max_positions is not None:
            for data_cls in total_indices.keys():
                total_indices[data_cls] = self.filter_indices_by_size(
                    total_indices[data_cls],
                    dataset,
                    max_positions,
                    ignore_invalid_inputs,
                )

        # create mini-batches with given size constraints
        total_batch_sampler = dict()
        for data_cls in total_indices.keys():
            if data_cls == "text":
                batch_sampler = dataset.batch_by_size(
                    total_indices[data_cls],
                    max_tokens=None,
                    max_sentences=max_sentences,
                    required_batch_size_multiple=required_batch_size_multiple,
                )
            else:
                batch_sampler = dataset.batch_by_size(
                    total_indices[data_cls],
                    max_tokens=max_tokens,
                    max_sentences=None,
                    required_batch_size_multiple=required_batch_size_multiple,
                )
            total_batch_sampler[data_cls] = (batch_sampler, len(batch_sampler))

        # Create accompanying batches of other data modes, and finally combine them all
        dominant_data_cls = (
            self.default_dom_cls if self.default_dom_cls is not None else None
        )  # Obtain dominant data class
        if dominant_data_cls is not None:
            num_max_batches = max(
                [
                    len(total_batch_sampler[data_cls][0])
                    for data_cls in total_batch_sampler.keys()
                ]
            )
            num_dom_batches = len(total_batch_sampler[dominant_data_cls][0])  #
        else:
            # TODO: try this branch may cause errors.
            num_max_batches = max(
                [
                    len(total_batch_sampler[data_cls][0])
                    for data_cls in total_batch_sampler.keys()
                ]
            )
            num_dom_batches = num_max_batches
            for data_cls in total_batch_sampler.keys():
                if len(total_batch_sampler[data_cls][0]) == num_max_batches:
                    dominant_data_cls = data_cls
                    continue

        for data_cls in total_batch_sampler.keys():
            if data_cls != dominant_data_cls:
                cur_basic_batch_sampler = total_batch_sampler[data_cls][0]
                cur_data_cls_num_batches = total_batch_sampler[data_cls][1]
                itr_times = int(num_dom_batches / cur_data_cls_num_batches) + 1
                total_batch_sampler[data_cls] = (
                    (cur_basic_batch_sampler * itr_times)[:num_dom_batches],
                    num_dom_batches,
                )

        final_batch_sampler = []
        for key, value in data_cls_batch_ratio.items():
            data_cls_batch_ratio[key] = float(value)
        sum_value = sum(data_cls_batch_ratio.values())
        for key, value in data_cls_batch_ratio.items():
            data_cls_batch_ratio[key] = float(value / sum_value)

        ## Obtain the ultra final batch sampler
        dom_rate = data_cls_batch_ratio[dominant_data_cls]
        for batch_id, cur_batch in enumerate(total_batch_sampler[dominant_data_cls][0]):
            cur_dom_bsz = cur_batch.shape[0]
            for data_cls in total_batch_sampler.keys():
                if data_cls != dominant_data_cls:
                    cur_cls_batch = total_batch_sampler[data_cls][0][batch_id]
                    cur_cls_bsz = cur_cls_batch.shape[0]
                    cur_data_rate = data_cls_batch_ratio[data_cls]
                    max_bsz = max(int(cur_data_rate / dom_rate), 1) * cur_dom_bsz
                    if cur_cls_bsz >= max_bsz:
                        np.random.shuffle(cur_cls_batch)
                        cur_cls_batch = cur_cls_batch[:max_bsz]
                    cur_batch = np.concatenate([cur_batch, cur_cls_batch], axis=0)
            final_batch_sampler.append(cur_batch)

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=final_batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            grouped_shuffling=grouped_shuffling,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter
