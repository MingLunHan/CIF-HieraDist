# @Time    : 2022/8/15
# @Author  : Minglun Han
# @File    : extract_bert_feats.py

import os
import sys
import argparse
import random
import string
import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel


"""
Description:
    This program is used to extract the features from pretrained models.
    You can specify the pretrained model and its vocabulary. The input should be a file of text_id and text pairs.
    The outputs will be npy files with text_id as prefix, and a hash table with text_id to feature.npy mapping.

Outputs: 
    1. ${utterance_id}.npy files;
    2. features hash json;
    
Chinese Pretraining models:
MODEL_NAME	MODEL_KEY
Bert-base-chinese   bert-base-chinese

"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parse = argparse.ArgumentParser(description="generate data tables")
    parse.add_argument(
        "--input_text_file_dir",
        type=str,
        default="/data1/student/mlhan/myprojects/fairseq-uni/examples/speech_to_text/egs/aishell2/data/aishell2.map",
        help="directory to texts, format '${utterance_id}\t${text}'",
    )
    parse.add_argument(
        "--split_name", type=str, default="aishell2", help="the split name"
    )
    parse.add_argument(
        "--output_dir",
        type=str,
        default="/data1/student/mlhan/myprojects/fairseq-uni/examples/speech_to_text/egs/aishell2/bert_feats/bert-base-chinese/",
        help="directory used to save outputs",
    )
    parse.add_argument(
        "--pretrained_model",
        type=str,
        default="bert-base-chinese",
        help="determine which pretrained model to be used to extract features",
    )
    parse.add_argument(
        "--pretrained_model_vocab",
        type=str,
        default="bert-base-chinese",
        help="the vocabulary of the pretrained model",
    )
    parse.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="the batch size for feature extraction",
    )
    parse.add_argument(
        "--lang", "-l", type=str, default="cn", help="the language of text"
    )
    parse.add_argument("--gpu", action="store_true")

    args = parse.parse_args()
    return args


def split_and_save(
    final_output_dir, utt_id_list, input_ids, attention_mask, last_hidden_states
):
    output_list = []
    for utt_id, ids, padding_mask, feat in zip(
        utt_id_list, input_ids, attention_mask, last_hidden_states
    ):
        cur_dict = dict()
        cur_dict["utt_id"] = utt_id
        cur_dict["input_ids"] = ids.cpu().detach().numpy().tolist()
        cur_dict["padding_mask"] = padding_mask.cpu().detach().numpy().tolist()
        cur_dict["length"] = int(padding_mask.sum().cpu().detach().numpy())

        cur_output_filename = os.path.join(final_output_dir, utt_id + ".npy")
        if not os.path.exists(cur_output_filename):
            np.save(cur_output_filename, feat.cpu().detach().numpy())
        cur_dict["feat_path"] = cur_output_filename

        output_list.append(cur_dict)

    return output_list


def main(args):
    input_text_file_dir = args.input_text_file_dir
    split_name = args.split_name
    output_dir = args.output_dir
    pretrained_model = args.pretrained_model
    pretrained_model_vocab = args.pretrained_model_vocab
    lang = args.lang

    # Load tokenizer and model
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert = BertModel.from_pretrained('bert-base-uncased')
    print("1. Load pretrained models and vocabulary")
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_vocab)
    bert = BertModel.from_pretrained(pretrained_model)
    if args.gpu:
        bert = bert.cuda()

    # Prepare output directory
    print("2. Create working directory")
    final_output_dir = os.path.join(output_dir, split_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if os.path.exists(final_output_dir):
        print("features are existing in %s" % final_output_dir)
    else:
        os.mkdir(final_output_dir)
    hash_table_dir = os.path.join(final_output_dir, split_name + "_text_feat" + ".json")
    f_hash = open(hash_table_dir, "w")

    # Extract features from pretrained models
    print("3. Extracting features")
    utt_id_list = []
    batch_inputs = []
    data_list = []
    batch_counter = 0
    with open(input_text_file_dir, "r") as f:
        batch_size_counter = 0
        for line in f:
            utt_id, text = line.strip().split("\t", 1)
            if lang == "cn":
                text = text.strip().replace(" ", "")  # For Chinese temporarily
            else:
                text = text.strip()
            batch_inputs.append(text)
            utt_id_list.append(utt_id)
            batch_size_counter += 1

            if batch_size_counter % args.batch_size == 0:
                # Forward pretrained models
                inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True)

                if args.gpu:
                    for k in inputs.keys():
                        inputs[k] = inputs[k].cuda()
                outputs = bert(**inputs)

                # Split and save samples
                input_ids = inputs["input_ids"]  # B x T
                attention_mask = inputs["attention_mask"]  # B x T
                last_hidden_states = outputs.last_hidden_state  # B x T x C
                output_list = split_and_save(
                    final_output_dir,
                    utt_id_list,
                    input_ids,
                    attention_mask,
                    last_hidden_states,
                )
                data_list.extend(output_list)

                # Empty buffers
                batch_counter += 1
                batch_size_counter = 0
                batch_inputs = []
                utt_id_list = []

                print("have processed %d batches. " % batch_counter)

        # Process samples in the last batch
        print("4. Process residual batch")
        inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True)
        if args.gpu:
            for k in inputs.keys():
                inputs[k] = inputs[k].cuda()
        outputs = bert(**inputs)
        input_ids = inputs["input_ids"]  # B x T
        attention_mask = inputs["attention_mask"]  # B x T
        last_hidden_states = outputs.last_hidden_state  # B x T x C
        output_list = split_and_save(
            final_output_dir, utt_id_list, input_ids, attention_mask, last_hidden_states
        )
        data_list.extend(output_list)

    data_dict = {"data": data_list}
    json_data_dict = json.dumps(data_dict, indent=4)
    f_hash.write(json_data_dict)
    f_hash.close()
    print("Feature extraction from pretrained language model is Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
