#!/bin/bash

# 设置环境变量
export PYTHONHASHSEED=777
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# 设置 Python 解释器路径
PYTHON="/home/lzj/anaconda3/envs/GIM/bin/python"

# 运行参数
BASE_PATH="data"
VIDEO_LIST="100h.txt"
VERSION="gim_lightglue"
OUTPUT_DIR="reconstruction_out"
SEED=777
TIMEOUT=3600
PREFIX="100h"

# 运行命令
$PYTHON video_cut.py \
    --base_path "$BASE_PATH" \
    --video_list "$VIDEO_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --version "$VERSION" \
    --seed "$SEED" \
    --durations 60 90 120\
    --timeout "$TIMEOUT" \
    --prefix "$PREFIX"