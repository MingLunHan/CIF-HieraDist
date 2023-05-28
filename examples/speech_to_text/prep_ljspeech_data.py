#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import pandas as pd
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from tqdm import tqdm


log = logging.getLogger(__name__)

SPLITS = ["train", "dev", "test"]
MANIFEST_COLUMNS = [
    "id",
    "audio",
    "n_frames",
    "tgt_text",
    "speaker",
    "src_text",
    "duration",
    "pitch",
    "energy",
]


def process(args):
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)

    # Read features from zipped files
    feats_zip_path = out_root / "logmelspec80.zip"
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(feats_zip_path)

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in SPLITS:
        # Prepare dataset
        print(f"Fetching split {split}...")
        dataset = []  # dataset is list of lists
        org_tsv_path = str(out_root / split) + ".tsv"
        print(org_tsv_path)
        f_org_tsv = open(org_tsv_path, "r")
        org_headers_in_org_tsv = f_org_tsv.readline().strip().split("\t")
        for line in f_org_tsv:
            items = line.strip().split("\t")
            cur_sample = []
            for item in items:
                cur_sample.append(item)
            dataset.append(cur_sample)

        # Check consistency
        assert (
            org_headers_in_org_tsv == MANIFEST_COLUMNS
        ), "Please ensure that manifest column headers are consistent."

        # Generate manifest files
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        for cur_sample in dataset:
            (
                sample_id,
                audio,
                n_frames,
                tgt_text,
                spk_id,
                src_text,
                duration,
                pitch,
                energy,
            ) = cur_sample
            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_paths[sample_id])
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["tgt_text"].append(tgt_text)
            manifest["speaker"].append(spk_id)
            manifest["src_text"].append(src_text)
            manifest["duration"].append(duration)
            manifest["pitch"].append(pitch)
            manifest["energy"].append(energy)
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}_final.tsv"
        )
        if split.startswith("train"):
            train_text.extend(manifest["tgt_text"])

    # Generate config YAML
    gen_config_yaml(
        out_root,
        vocab_name="phoneme_vocab.txt",
        specaugment_policy=None,
        cmvn_type="global",
        gcmvn_path=out_root / "gcmvn_stats.npz",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
