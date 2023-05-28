# @Time    : 2022/8/17
# @Author  : Minglun Han
# @File    : visual_feats_reader.py

import os
import sys
import random
import h5py
import cn2an
import string
import argparse
import logging
import numpy as np
import torch


class ImageFeaturesHdfReader(object):
    """
    A reader for H5 files containing pre-extracted image features. A typical
    H5 file is expected to have a column named "image_id", and another column
    named "features".

    Example of an H5 file:
    ```
    faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    Parameters
    ----------
    features_h5path : str
        Path to an H5 file containing COCO train / val image features.
    in_memory : bool
        Whether to load the whole H5 file in memory. Beware, these files are
        sometimes tens of GBs in size. Set this to true if you have sufficient
        RAM - trade-off between speed and memory.
    """

    def __init__(self, features_path: str, in_memory: bool = False):
        self.features_hdfpath = features_path
        self._in_memory = in_memory
        with h5py.File(self.features_hdfpath, "r") as features_hdf:
            self.image_id = list(features_hdf.keys())

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, image_id):
        with h5py.File(self.features_hdfpath, "r") as features_hdf:
            features = features_hdf[image_id][:]

        return features

    def keys(self):
        return self.image_id


def parse_args():
    parse = argparse.ArgumentParser(description="preprocess text")
    parse.add_argument("--h5_file", type=str, help="h5 file name")
    args = parse.parse_args()
    return args


def main(args):
    feat_reader = ImageFeaturesHdfReader(features_path=args.h5_file)
    for i in range(len(feat_reader)):
        print(f"image id: {feat_reader.image_id[i]}")
        vit_feat = feat_reader[feat_reader.image_id[i]]  # 197 x C (768)
        # print(vit_feat.shape)


if __name__ == "__main__":
    args = parse_args()
    main(args)
