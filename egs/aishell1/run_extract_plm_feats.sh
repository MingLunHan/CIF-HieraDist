#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

MODEL_NAME=bert-base-chinese
MODEL_KEY=bert-base-chinese

python -u ../../examples/speech_to_text/bert_feat_extract/extract_bert_feats.py \
  --output_dir ./bert_feats/${MODEL_NAME}/ \
  --pretrained_model ${MODEL_KEY} \
  --pretrained_model_vocab ${MODEL_KEY} \
  --gpu --batch_size 256
