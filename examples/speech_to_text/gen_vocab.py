# @Time    : 2022/3/2
# @Author  : Minglun Han
# @File    : gen_vocab.py

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
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm

# Define data splits
SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

output_root = sys.argv[0]
vocab_size = sys.argv[1]
vocab_type = sys.argv[2]

out_root = Path(output_root).absolute()
out_root.mkdir(exist_ok=True)

# Load text
train_text = []
for split in SPLITS:  # conduct for each data split
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    dataset = LIBRISPEECH(out_root.as_posix(), url=split)
    for _, _, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
        # Add items one-by-one
        sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
        manifest["tgt_text"].append(utt.lower())
    if split.startswith("train"):
        train_text.extend(manifest["tgt_text"])

# Generate vocabulary
vocab_size = "" if vocab_type == "char" else str(vocab_size)
spm_filename_prefix = f"spm_{vocab_type}{vocab_size}"
with NamedTemporaryFile(mode="w") as f:
    for t in train_text:
        f.write(t + "\n")
    gen_vocab(
        Path(f.name),
        out_root / spm_filename_prefix,
        vocab_type,
        vocab_size,
    )
