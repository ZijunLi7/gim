#!/bin/bash

# 设置环境变量
export PYTHONHASHSEED=777
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# 设置 Python 解释器路径
PYTHON="/home/lzj/anaconda3/envs/GIM/bin/python"

# 运行参数
BASE_PATH="/home/lzj/lzj/matching_codes/cursor/balanced_dust3r/data/walk"
VIDEO_LIST="2.txt"
VERSION="gim_lightglue"
OUTPUT_DIR="reconstruction_out"
SEED=777

# 运行命令
$PYTHON video_cut.py \
    --base_path "$BASE_PATH" \
    --video_list "$VIDEO_LIST" \
    --version "$VERSION" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"