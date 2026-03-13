#!/bin/bash

CKPT=${1}
TEST_SET=${2:-"./all_data/RAbD/test.json"}
SAVE_DIR=${3}
USE_AF2AG=${4:-"True"}


CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python generate.py \
    --use_af2ag "$USE_AF2AG" \
    --ckpt "$CKPT" \
    --test_set "$TEST_SET" \
    --save_dir "$SAVE_DIR"
