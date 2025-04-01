#!/bin/bash

# 停止训练相关进程的脚本

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 获取项目根目录
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

echo "正在停止训练相关进程..."

# 创建日志目录（如果不存在）
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p $LOG_DIR

# 停止训练进程
if [ -f "${LOG_DIR}/train_pid.txt" ]; then
    TRAIN_PID=$(cat "${LOG_DIR}/train_pid.txt")
    echo "停止训练进程: $TRAIN_PID"
    kill -9 $TRAIN_PID 2>/dev/null || true
    rm "${LOG_DIR}/train_pid.txt"
else
    echo "没有找到训练进程PID文件，尝试查找并终止所有训练进程..."
    ps -ef | grep "tgsr/train.py" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
fi

# 停止TensorBoard进程
if [ -f "${LOG_DIR}/tensorboard_pid.txt" ]; then
    TB_PID=$(cat "${LOG_DIR}/tensorboard_pid.txt")
    echo "停止TensorBoard进程: $TB_PID"
    kill -9 $TB_PID 2>/dev/null || true
    rm "${LOG_DIR}/tensorboard_pid.txt"
else
    echo "没有找到TensorBoard PID文件，尝试查找并终止所有TensorBoard进程..."
    ps -ef | grep tensorboard | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
fi

# 清理CUDA缓存和GPU内存
echo "清理GPU内存..."
# 1. 清理PyTorch的CUDA缓存
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize()" 2>/dev/null || true

# 2. 清理系统级GPU内存
if command -v nvidia-smi &> /dev/null; then
    # 获取所有GPU进程
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true
    
    # 等待GPU内存释放
    sleep 2
    
    # 显示清理后的GPU状态
    echo "GPU内存状态:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
else
    echo "未检测到nvidia-smi，跳过系统级GPU内存清理"
fi

# 3. 清理系统缓存
echo "清理系统缓存..."
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

echo "所有训练相关进程已停止，GPU内存已清理" 