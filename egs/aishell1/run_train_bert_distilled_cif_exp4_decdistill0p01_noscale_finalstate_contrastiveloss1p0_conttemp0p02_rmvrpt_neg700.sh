#!/bin/bash

FAIRSEQ_ROOT=/data1/student/mlhan/myprojects/CIF-HieraDist
export PYTHONPATH="${FAIRSEQ_ROOT}:${PYTHONPATH}"

export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
export CUDA_VISIBLE_DEVICES=6,7

EXP_NAME=aishell1_bert_distilled_cif_exp4_decdistill0p01_noscale_finalstate_contrastiveloss1p0_conttemp0p02_rmvrpt_neg700
LS_ROOT=./data/
SAVE_DIR=./exp/${EXP_NAME}

fairseq-train ${LS_ROOT} --ddp-backend no_c10d \
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
  --frontend-type conv2d --conv-kernel-sizes 3 --conv2d-output-channels 128 \
  --apply-bert-distill --apply-dec-state-dis-loss --dec-state-dis-loss-lambda 0.01 \
  --no-dim-scaling-for-mse-loss --fetch-decoder-states-from pre_final_output_proj \
  --apply-cif-contrastive-dis-cos-loss --cif-contrastive-dis-cos-loss-lambda 1.0 \
  --contrastive-temperature 0.02 --num-contrastive-negative-samples 700 \
  --remove-overlap-in-negs  

cp ./$EXP_NAME.log ./$SAVE_DIR/

echo "done"

