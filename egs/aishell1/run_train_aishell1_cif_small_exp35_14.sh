#!/bin/bash

FAIRSEQ_ROOT=/data1/student/mlhan/myprojects/CIF-HieraDist
export PYTHONPATH="${FAIRSEQ_ROOT}:${PYTHONPATH}"

export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
export CUDA_VISIBLE_DEVICES=4,5

EXP_NAME=aishell1_cif_small_exp35_14
LS_ROOT=./data/
SAVE_DIR=./exp/${EXP_NAME}

fairseq-train ${LS_ROOT} \
  --save-dir ${SAVE_DIR} --tensorboard-logdir ${SAVE_DIR} \
  --config-yaml config_Tmask50.yaml --train-subset train \
  --valid-subset dev --num-workers 4 --max-tokens 30000 \
  --max-update 240000 --task speech_to_text --criterion cif \
  --keep-last-epochs 20 --arch s2t_cif_transformer \
  --encoder-embed-dim 256 --encoder-ffn-embed-dim 2048 --encoder-attention-heads 4 \
  --decoder-embed-dim 256 --decoder-ffn-embed-dim 2048 --decoder-attention-heads 4 \
  --cif-embedding-dim 256 --conv-cif-output-channels-num 256 --dense-cif-units-num 256 \
  --optimizer adam --lr 3e-4 --lr-scheduler tri_stage \
  --warmup-steps 30000 --hold-steps 90000 --decay-steps 120000 \
  --decoder-layers 2 --encoder-layers 15 --conv-kernel-sizes 5 \
  --clip-norm 1.0 --seed 1 --update-freq 1 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --decoder-dropout 0.2 --decoder-attention-dropout 0.2 --decoder-activation-dropout 0.2 \
  --log-format simple --log-interval 50 --validate-after-updates 100 \
  --conv-cif-width 3 --ctc-loss-lambda 0.5 --patience 40 \
  --best-checkpoint-metric wer --fp16 --weight-decay 1e-2 \
  --post-process char --conv-cif-dropout 0.2 --apply-label-smoothing \
  --layer-downsampling --pooling-layer-ids 5,10 \
  --apply-conformer-encoder --conformer-pos-enc-type rel_pos \
  --conformer-depthwise-conv-kernel-size 15 --conformer-attn-type espnet \
  --frontend-type conv2d --conv-kernel-sizes 3 --conv2d-output-channels 128

cp ./$EXP_NAME.log ./$SAVE_DIR/

echo "done"

