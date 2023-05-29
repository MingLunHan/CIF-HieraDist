#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

# General configs
FAIRSEQ_ROOT=/data1/student/mlhan/myprojects/CIF-HieraDist/
export PYTHONPATH="${FAIRSEQ_ROOT}:${PYTHONPATH}"
#EXP_NAME=aishell1_bert_distilled_cif_exp4_decdistill0p01_noscale_finalstate_contrastiveloss1p0_conttemp0p02_rmvrpt_neg700
EXP_NAME=aishell1_cif_small_exp35_14
DATA_ROOT=./data/
SAVE_DIR=./exp/${EXP_NAME}
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt

# Decoding configs
TAIL_THRESHOLD=0.4
BATCH_SIZE=8
BEAM_SIZE=10
DECO_MODE=fast_ar

# Stage control
stage=1

# Average checkpoints
if [ ${stage} -le 0 ]; then
  python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
    --num-epoch-checkpoints 10 \
    --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
fi

# Conduct decoding
if [ ${stage} -le 1 ]; then
  for SUBSET in test; do
    RES_DIR=./res/${EXP_NAME}
    if [ ! -d ${RES_DIR} ]; then
      mkdir -p $RES_DIR
    fi
    python -u $FAIRSEQ_ROOT/examples/speech_recognition/infer.py ${DATA_ROOT} \
      --task speech_to_text --batch-size ${BATCH_SIZE} --nbest 1 --beam ${BEAM_SIZE} \
      --cif-decoder-mode ${DECO_MODE} --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
      --gen-subset ${SUBSET} --results-path ${RES_DIR} --cif-decoder cif \
      --tail-handling-firing-threshold ${TAIL_THRESHOLD} \
      --criterion cif --post-process char --config-yaml config_Tmask50.yaml
  done
fi

echo "done"
