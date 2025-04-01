#!/bin/bash

# 从检查点恢复训练的脚本

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 获取项目根目录
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# 设置PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 进入项目根目录
cd "$PROJECT_ROOT"

# 默认配置文件
CONFIG="${PROJECT_ROOT}/options/train_tgsr_x4.yml"
CHECKPOINT=""
DEBUG=""
MEMORY_OPT="--memory_opt"
NUM_GPU=0  # 0表示自动检测

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --opt)
            # 如果输入的是相对路径，则转换为绝对路径
            if [[ "$2" == /* ]]; then
                CONFIG="$2"
            else
                CONFIG="${PROJECT_ROOT}/$2"
            fi
            shift
            shift
            ;;
        --checkpoint)
            # 如果输入的是相对路径，则转换为绝对路径
            if [[ "$2" == /* ]]; then
                CHECKPOINT="$2"
            else
                CHECKPOINT="${PROJECT_ROOT}/$2"
            fi
            shift
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --gpu)
            NUM_GPU="$2"
            shift
            shift
            ;;
        --no-memory-opt)
            MEMORY_OPT=""
            shift
            ;;
        --help)
            echo "用法: ./resume_train.sh --checkpoint CHECKPOINT_PATH [--opt CONFIG_FILE] [--debug] [--gpu NUM_GPU] [--no-memory-opt]"
            echo "  --checkpoint PATH   指定恢复训练的检查点路径（必需）"
            echo "  --opt CONFIG_FILE   指定配置文件路径 (默认: options/train_tgsr_x4.yml)"
            echo "  --debug             启用调试模式"
            echo "  --gpu NUM_GPU       指定使用的GPU数量 (0: 自动检测)"
            echo "  --no-memory-opt     禁用内存优化"
            echo "  --help              显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [ -z "$CHECKPOINT" ]; then
    echo "错误: 必须指定检查点路径 (--checkpoint)"
    exit 1
fi

# 检查检查点文件是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 检查点文件 '$CHECKPOINT' 不存在"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件 '$CONFIG' 不存在"
    exit 1
fi

# 创建日志目录
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p $LOG_DIR

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/resume_${TIMESTAMP}.log"

# 自动检测GPU数量
if [ $NUM_GPU -eq 0 ]; then
    NUM_GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    echo "检测到 ${NUM_GPU} 个GPU"
fi

# 检查是否使用分布式训练
if [ $NUM_GPU -gt 1 ]; then
    LAUNCHER="pytorch"
    echo "使用分布式训练模式 (${NUM_GPU} GPUs)"
else
    LAUNCHER="none"
    echo "使用单GPU训练模式"
fi

# 清理CUDA缓存
echo "清理CUDA缓存..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# 启动TensorBoard
echo "启动TensorBoard..."
# 从配置文件中获取实验名称
EXP_NAME=$(grep "name:" "$CONFIG" | head -n 1 | cut -d' ' -f2-)
TB_DIR="${PROJECT_ROOT}/tb_logger/${EXP_NAME}"
mkdir -p $TB_DIR

# 后台启动TensorBoard
tensorboard --port 6007 --logdir $TB_DIR > "${LOG_DIR}/tensorboard_${TIMESTAMP}.log" 2>&1 &
TB_PID=$!
echo "TensorBoard已启动，PID: $TB_PID，访问地址: http://localhost:6007"
echo $TB_PID > "${LOG_DIR}/tensorboard_pid.txt"

# 修改配置文件中的恢复路径
TMP_CONFIG="${CONFIG}.tmp"
cp "$CONFIG" "$TMP_CONFIG"
# 使用sed替换resume_state行
sed -i "s|resume_state:.*|resume_state: $CHECKPOINT|g" "$TMP_CONFIG"
echo "已设置恢复检查点: $CHECKPOINT"

# 启动训练
echo "开始恢复训练..."
if [ "$LAUNCHER" = "pytorch" ]; then
    # 使用torchrun启动分布式训练
    nohup python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPU \
        --master_port=$(( ( RANDOM % 50000 )  + 10000 )) \
        "${PROJECT_ROOT}/tgsr/train.py" \
        --launcher pytorch \
        --opt $TMP_CONFIG \
        $DEBUG $MEMORY_OPT > $LOG_FILE 2>&1 &
else
    # 使用单GPU训练
    nohup python "${PROJECT_ROOT}/tgsr/train.py" \
        --launcher none \
        --opt $TMP_CONFIG \
        $DEBUG $MEMORY_OPT > $LOG_FILE 2>&1 &
fi

TRAIN_PID=$!
echo "训练进程已在后台启动，PID: $TRAIN_PID"
echo $TRAIN_PID > "${LOG_DIR}/train_pid.txt"

echo "训练日志将保存到: $LOG_FILE"
echo "使用 'tail -f $LOG_FILE' 查看训练日志" 