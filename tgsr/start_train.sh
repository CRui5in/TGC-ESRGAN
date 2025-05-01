#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/experiments/"
mkdir -p "$PROJECT_ROOT/tb_logger/"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/logs/tgc_esrgan_train_log_${TIMESTAMP}.log"

CONFIG="$PROJECT_ROOT/options/train_tgcesrgan_x4plus.yml"

echo "开始TGC-ESRGAN训练..." | tee -a "$LOG_FILE"
cd $SCRIPT_DIR
nohup python -u train.py -opt $CONFIG --launcher none --auto_resume > "$LOG_FILE" 2>&1 & 

PID=$!
echo $PID > "$SCRIPT_DIR/train_pid.txt"
echo "训练进程PID: $PID 已保存到 $SCRIPT_DIR/train_pid.txt"

echo "训练日志已保存到: $LOG_FILE"