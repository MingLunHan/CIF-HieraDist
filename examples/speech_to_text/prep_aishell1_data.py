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
SPLITS = ["dev", "test", "train_sp"]
# SPLITS = ["dev"]

# Define the headers of columns
MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

# Define special tokens
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3


def load_aishell1_item(fileid: str, path: str, ext_audio: str, id2txt_dict):

    # get speaker id
    speaker_id = "".join(list(fileid)[6:11])

    # Specify the path to audio
    file_audio = fileid + ext_audio
    file_audio = os.path.join(path, speaker_id, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    # Load text
    if fileid in id2txt_dict.keys():
        transcript = id2txt_dict[fileid]
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
        str(fileid),
    )


def load_speed_perturbated_aishell1_item(fileid: str, id2txt_dict, id2filedir_dict):

    if fileid.startswith("sp"):
        # For speed perturbated audio
        temp_fileid = fileid.split("-", 1)[-1]
        speaker_id = "".join(list(temp_fileid)[6:11])
    else:
        # For original audio
        speaker_id = "".join(list(fileid)[6:11])

    # Load audio
    file_path = id2filedir_dict[fileid]
    waveform, sample_rate = torchaudio.load(file_path)

    # Load text
    if fileid in id2txt_dict.keys():
        transcript = id2txt_dict[fileid]
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
        str(fileid),
    )


class AISHELL1(Dataset):
    """Create a Dataset for AISHELL1.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _txt_file_name = "aishell_transcript_v0.8.txt"
    _ext_audio = ".wav"

    FOLDER_IN_ARCHIVE = "AISHELL1/data_aishell/wav"
    TRANSCROPT_IN_ARCHIVE = "AISHELL1/data_aishell/transcript"

    def __init__(self, root, split):

        if split in [
            "train",
            "dev",
            "test",
        ]:
            print("Valid data split detected.")

        basename = split
        root = os.fspath(root)
        folder_in_archive = os.path.join(self.FOLDER_IN_ARCHIVE, basename)

        self._path = os.path.join(root, folder_in_archive)  # Obtain target wav path
        self._walker = sorted(
            str(p.stem) for p in Path(self._path).glob("*/*" + self._ext_audio)
        )  # Traverse all samples
        self._txt_file_path = os.path.join(
            root, self.TRANSCROPT_IN_ARCHIVE, self._txt_file_name
        )

        self._id2txt_dict = dict()
        with open(self._txt_file_path, "r") as ft:
            for line in ft:
                uttid, text = line.strip().split(" ", 1)
                self._id2txt_dict[uttid] = text

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, int, int, int):
            ``(waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)``
        """

        fileid = self._walker[n]
        return load_aishell1_item(
            fileid, self._path, self._ext_audio, self._id2txt_dict
        )

    def __len__(self) -> int:
        return len(self._walker)


class SpeedPerturbatedAISHELL1(Dataset):
    """Create a Dataset for AISHELL1.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _txt_file_name = "text"
    _scp_file_name = "wav.scp"
    _ext_audio = ".wav"

    FOLDER_IN_ARCHIVE = "dump/raw"

    def __init__(self, root, split):

        if split in ["dev", "test", "train_sp"]:
            print("Valid data split detected.")

        basename = split
        root = os.fspath(root)
        folder_in_archive = os.path.join(self.FOLDER_IN_ARCHIVE, basename)

        self._path = os.path.join(root, folder_in_archive)

        # Register path
        self._walker = []
        self.id2filedir_dict = dict()
        self._scp_file_path = os.path.join(root, folder_in_archive, self._scp_file_name)
        with open(self._scp_file_path) as fp:
            for line in fp:
                uttid, utt_dir = line.strip().split(" ", 1)
                if uttid.startswith("sp"):
                    self.id2filedir_dict[uttid] = os.path.join(root, utt_dir.strip())
                else:
                    self.id2filedir_dict[uttid] = utt_dir.strip()
                self._walker.append(uttid.strip())
        self._walker = sorted(self._walker)
        logging.info("Utterance path registration done")

        # Register text
        self._id2txt_dict = dict()
        self._txt_file_path = os.path.join(root, folder_in_archive, self._txt_file_name)
        with open(self._txt_file_path, "r") as ft:
            line_cnt = 0
            for line in ft:
                if uttid in self._walker:
                    uttid, text = line.strip().split(" ", 1)
                    self._id2txt_dict[uttid] = text.strip().replace(" ", "")
                line_cnt += 1
                if line_cnt % 10000 == 0:
                    logging.info("have processed %d lines" % line_cnt)
        logging.info("Vocabulary collection done")

        logging.info("Dataset initialization done")

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            ``(waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)``
        """

        fileid = self._walker[n]
        return load_speed_perturbated_aishell1_item(
            fileid, self._id2txt_dict, self.id2filedir_dict
        )

    def __len__(self) -> int:
        return len(self._walker)


def process(args):
    input_root = Path(args.input_root).absolute()
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)

    # Extract features
    feature_root = out_root / "fbank80"
    feature_root.mkdir(exist_ok=True)
    for split in SPLITS:
        print(f"Fetching split {split}...")
        dataset = AISHELL1(input_root.as_posix(), split=split)
        # dataset = SpeedPerturbatedAISHELL1(input_root.as_posix(), split=split)
        print("Extracting log mel filter bank features...")
        for wav, sample_rate, _, spk_id, utt_id in tqdm(dataset):
            sample_id = utt_id
            try:
                extract_fbank_features(
                    wav, sample_rate, feature_root / f"{sample_id}.npy"
                )
            except:
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
        # dataset = AISHELL1(input_root.as_posix(), split=split)
        dataset = SpeedPerturbatedAISHELL1(input_root.as_posix(), split=split)
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
    vocab_file_path = os.path.join(str(out_root), "vocab.txt")
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
        default="/data/LibriSpeech/mlhan_extra_files/",
        required=False,
        type=str,
    )  # assign the data output root directory
    parser.add_argument(
        "--output-root",
        "-o",
        default="/workspace/fairseq-uni/examples/speech_to_text/egs/aishell1/data/",
        required=False,
        type=str,
    )  # assign the data output root directory
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
