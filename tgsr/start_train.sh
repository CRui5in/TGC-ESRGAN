#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 设置环境变量
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

mkdir "$PROJECT_ROOT/logs"
mkdir "$PROJECT_ROOT/experiments/"
mkdir "$PROJECT_ROOT/tb_logger/"

# 设置日志文件名（使用时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/logs/train_log_${TIMESTAMP}.log"

# 配置文件路径
CONFIG="$PROJECT_ROOT/options/train_tgsr_x4plus.yml"

# 执行训练 (使用项目内的训练脚本)，同时保存日志
echo "开始训练..." | tee -a "$LOG_FILE"
cd $SCRIPT_DIR
nohup python -u train.py -opt $CONFIG --launcher none --auto_resume > "$LOG_FILE" 2>&1 & 

echo "训练日志已保存到: $LOG_FILE"