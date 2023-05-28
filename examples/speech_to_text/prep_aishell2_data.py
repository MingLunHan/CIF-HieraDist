#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from torch.utils.data import Dataset
from typing import Tuple, Union
import torchaudio

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

# Define data splits
SPLITS = [
    "train-error"
    # "train"
    # "test-ios", "test-android",
    # "test-mic", "dev-ios",
    # "dev-android", "dev-mic"
]

# Define the headers of columns
MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

# Define special tokens
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3


def load_aishell2_item(file_id, id2audios, id2trans, id2spk):
    speaker_id = id2spk[file_id]
    file_audio = id2audios[file_id]
    waveform, sample_rate = torchaudio.load(file_audio)
    assert sample_rate == 16000, "sample rate is not correct."
    if file_id in id2trans.keys():
        transcript = id2trans[file_id]
        transcript = transcript.strip().replace(" ", "")
    else:
        # Translation not found
        print("Translation not found for " + fileid)
        transcript = None

    return (
        waveform,
        sample_rate,
        transcript,
        str(speaker_id),
        str(file_id),
    )


class AISHELL2(Dataset):
    """Create a Dataset for AISHELL2."""

    txt_filename = "trans.txt"
    audio_scp_filename = "wav.scp"
    speaker_filename = "spk_info.txt"

    def __init__(self, root, split):

        assert split in [
            "train",
            "test-ios",
            "test-android",
            "test-mic",
            "dev-ios",
            "dev-android",
            "dev-mic",
            "train-error",
        ], "data split is invalid."

        root = os.fspath(root)
        print(root)

        if split == "train":
            data_root_dir = os.path.join(root, "iOS", "data")
        elif split == "train-error":
            data_root_dir = os.path.join(root, "iOS", "data_error")
        elif "dev" in split or "test" in split:
            if "dev" in split:
                if "ios" in split:
                    data_root_dir = os.path.join(
                        root, "AISHELL-DEV-TEST-SET", "iOS", "dev"
                    )
                elif "android" in split:
                    data_root_dir = os.path.join(
                        root, "AISHELL-DEV-TEST-SET", "Android", "dev"
                    )
                elif "mic" in split:
                    data_root_dir = os.path.join(
                        root, "AISHELL-DEV-TEST-SET", "Mic", "dev"
                    )
                else:
                    raise ValueError("Invalid options %s" % split)
            else:
                if "ios" in split:
                    data_root_dir = os.path.join(
                        root, "AISHELL-DEV-TEST-SET", "iOS", "test"
                    )
                elif "android" in split:
                    data_root_dir = os.path.join(
                        root, "AISHELL-DEV-TEST-SET", "Android", "test"
                    )
                elif "mic" in split:
                    data_root_dir = os.path.join(
                        root, "AISHELL-DEV-TEST-SET", "Mic", "test"
                    )
                else:
                    raise ValueError("Invalid options %s" % split)
        else:
            raise ValueError("Invalid options %s" % split)

        self.trans_filename = os.path.join(data_root_dir, self.txt_filename)
        self.wav_scp_filename = os.path.join(data_root_dir, self.audio_scp_filename)

        self.id2txt_dict = dict()
        with open(self.trans_filename, "r") as f_trans:
            for line in f_trans:
                uttid, text = line.strip().split("\t", 1)
                self.id2txt_dict[uttid] = text

        self.id2audios_dict = dict()
        self.id2spk_dict = dict()
        with open(self.wav_scp_filename, "r") as f_audios:
            for line in f_audios:
                uttid, audio_path = line.strip().split("\t", 1)
                spk_id = audio_path.split("/")[1]
                abs_audio_path = os.path.join(data_root_dir, audio_path)
                self.id2audios_dict[uttid] = abs_audio_path
                self.id2spk_dict[uttid] = spk_id

        self._walker = list(self.id2txt_dict.keys())
        self._walker.sort()

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, int, int, int):
            ``(waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)``
        """

        fileid = self._walker[n]
        return load_aishell2_item(
            fileid, self.id2audios_dict, self.id2txt_dict, self.id2spk_dict
        )

    def __len__(self) -> int:
        return len(self._walker)


def process(args):
    print("Begin process...")

    input_root = Path(args.input_root).absolute()
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)

    # Extract features
    feature_root = out_root / "fbank80"
    feature_root.mkdir(exist_ok=True)
    for split in SPLITS:
        print(f"Fetching split {split}...")
        dataset = AISHELL2(input_root.as_posix(), split=split)
        print("Extracting log mel filter bank features...")
        for wav, sample_rate, _, spk_id, utt_id in tqdm(dataset):
            sample_id = utt_id
            try:
                extract_fbank_features(
                    wav, sample_rate, feature_root / f"{sample_id}.npy"
                )
            except Exception as e:
                print(e)
                print("Encounter error for %s" % utt_id)
            else:
                continue

    # Pack features into ZIP
    zip_path = out_root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in SPLITS:  # conduct for each data split
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = AISHELL2(input_root.as_posix(), split=split)
        for _, _, trans, spk_id, utt_id in tqdm(dataset):
            if trans is not None and utt_id.strip() in audio_paths.keys():
                # Add items one-by-one
                sample_id = utt_id
                manifest["id"].append(sample_id)
                manifest["audio"].append(audio_paths[sample_id])
                manifest["n_frames"].append(audio_lengths[sample_id])
                manifest["tgt_text"].append(" ".join(list(trans.lower())))
                manifest["speaker"].append(spk_id)
        save_df_to_tsv(pd.DataFrame.from_dict(manifest), out_root / f"{split}.tsv")
        if split.startswith("train"):
            train_text.extend(manifest["tgt_text"])

    # Generate vocab
    vocab_file_path = os.path.join(str(out_root), "vocab.txt")
    if len(train_text) != 0:
        vocab_dict = dict()
        for line in train_text:
            tokens_list = line.strip().split(" ")
            for tok in tokens_list:
                if tok not in vocab_dict:
                    vocab_dict[tok] = 1
                else:
                    vocab_dict[tok] += 1
        sorted_vocab_dict = {
            sort_k: sort_v
            for sort_k, sort_v in sorted(
                vocab_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True
            )
        }
        f_vocab = open(vocab_file_path, "w")
        f_vocab.write("\t".join([BOS_TOKEN, str(0)]) + "\n")
        f_vocab.write("\t".join([PAD_TOKEN, str(0)]) + "\n")
        f_vocab.write("\t".join([EOS_TOKEN, str(0)]) + "\n")
        f_vocab.write("\t".join([UNK_TOKEN, str(0)]) + "\n")
        for idx, (tok, freq) in enumerate(sorted_vocab_dict.items()):
            f_vocab.write("\t".join([tok, str(freq)]) + "\n")
        f_vocab.close()

    # Generate config YAML
    gen_config_yaml(out_root, vocab_name=vocab_file_path, specaugment_policy="ld")

    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        "-i",
        default="/data/LibriSpeech/mlhan_extra_files/AISHELL2",
        required=False,
        type=str,
    )  # assign the data output root directory
    parser.add_argument(
        "--output-root",
        "-o",
        default="/workspace/fairseq-uni/examples/speech_to_text/egs/aishell2/data/train_error",
        required=False,
        type=str,
    )  # assign the data output root directory
    parser.add_argument(
        "--vocab-type",
        default="char",
        required=False,
        type=str,
        choices=["bpe", "unigram", "char"],
    )  # assign the vocabulary type
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
