#!/bin/bash

# 文本引导超分辨率模型数据集生成与训练脚本

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
DEBUG=""
MEMORY_OPT="--memory_opt"
NUM_GPU=0  # 0表示自动检测
COCO_DIR="/root/autodl-tmp/COCO2017"  # COCO数据集路径
DATASET_DIR="/root/autodl-tmp/tgsr_dataset_hr_only"  # 新数据集输出路径
GENERATE_DATASET=true  # 是否生成数据集
REMOVE_EXISTING=false  # 是否删除已存在的数据集
REUSE_EXP_DIR=false  # 是否复用已有实验目录

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
        --coco-dir)
            COCO_DIR="$2"
            shift
            shift
            ;;
        --dataset-dir)
            DATASET_DIR="$2"
            shift
            shift
            ;;
        --no-dataset-gen)
            GENERATE_DATASET=false
            shift
            ;;
        --remove-existing)
            REMOVE_EXISTING=true
            shift
            ;;
        --reuse-exp-dir)
            REUSE_EXP_DIR=true
            shift
            ;;
        --help)
            echo "用法: ./start_training.sh [选项]"
            echo "选项:"
            echo "  --opt CONFIG_FILE       指定配置文件路径 (默认: options/train_tgsr_x4.yml)"
            echo "  --debug                 启用调试模式"
            echo "  --gpu NUM_GPU           指定使用的GPU数量 (0: 自动检测)"
            echo "  --no-memory-opt         禁用内存优化"
            echo "  --coco-dir DIR          指定COCO数据集路径 (默认: /root/autodl-tmp/COCO2017)"
            echo "  --dataset-dir DIR       指定数据集输出路径 (默认: /root/autodl-tmp/tgsr_dataset_hr_only)"
            echo "  --no-dataset-gen        跳过数据集生成步骤"
            echo "  --remove-existing       如果数据集目录已存在则删除重新生成"
            echo "  --reuse-exp-dir         继续使用已存在的实验目录，避免创建新目录"
            echo "  --help                  显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

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
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
DATASET_LOG_FILE="${LOG_DIR}/dataset_gen_${TIMESTAMP}.log"

# 检查COCO数据集路径是否存在
if [ "$GENERATE_DATASET" = true ] && [ ! -d "$COCO_DIR" ]; then
    echo "错误: COCO数据集路径 '$COCO_DIR' 不存在"
    exit 1
fi

# 数据集生成
if [ "$GENERATE_DATASET" = true ]; then
    # 检查数据集目录是否已存在
    if [ -d "$DATASET_DIR" ]; then
        if [ "$REMOVE_EXISTING" = true ]; then
            echo "警告: 数据集目录 '$DATASET_DIR' 已存在，将被删除重新生成"
            rm -rf "$DATASET_DIR"
        else
            echo "数据集目录 '$DATASET_DIR' 已存在，跳过生成步骤"
            echo "如需重新生成，请使用 --remove-existing 参数"
            GENERATE_DATASET=false
        fi
    fi

    # 生成数据集
    if [ "$GENERATE_DATASET" = true ]; then
        echo "开始从COCO2017生成新的数据集..."
        echo "数据集生成日志将保存到: $DATASET_LOG_FILE"
        
        python datasets_coco.py \
            --coco_dir "$COCO_DIR" \
            --output_dir "$DATASET_DIR" \
            --jpeg_quality 100 \
            --max_workers 16 > "$DATASET_LOG_FILE" 2>&1
        
        if [ $? -ne 0 ]; then
            echo "错误: 数据集生成失败，请查看日志: $DATASET_LOG_FILE"
            exit 1
        fi
        
        echo "数据集生成完成，保存在: $DATASET_DIR"
    fi
fi

# 更新配置文件中的数据集路径
echo "更新配置文件中的数据集路径..."
sed -i "s|dataroot_gt:.*train/hr|dataroot_gt: ${DATASET_DIR}/train/hr|g" "$CONFIG"
sed -i "s|text_file:.*train_captions.json|text_file: ${DATASET_DIR}/train_captions.json|g" "$CONFIG"
sed -i "s|dataroot_gt:.*val/hr|dataroot_gt: ${DATASET_DIR}/val/hr|g" "$CONFIG"
sed -i "s|text_file:.*val_captions.json|text_file: ${DATASET_DIR}/val_captions.json|g" "$CONFIG"
sed -i "s|dataroot_gt:.*test/hr|dataroot_gt: ${DATASET_DIR}/test/hr|g" "$CONFIG"
sed -i "s|text_file:.*test_captions.json|text_file: ${DATASET_DIR}/test_captions.json|g" "$CONFIG"

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
# 注意：我们不再创建TB_DIR，而是使用训练脚本会自动创建的tb_logger目录
TB_DIR="${PROJECT_ROOT}/tb_logger"
mkdir -p $TB_DIR

# 检查是否有正在运行的TensorBoard进程
TB_PID_FILE="${LOG_DIR}/tensorboard_pid.txt"
if [ -f "$TB_PID_FILE" ]; then
    OLD_TB_PID=$(cat "$TB_PID_FILE")
    if kill -0 $OLD_TB_PID 2>/dev/null; then
        echo "检测到正在运行的TensorBoard进程 (PID: $OLD_TB_PID)，将终止它"
        kill $OLD_TB_PID 2>/dev/null || true
    fi
fi

# 后台启动TensorBoard，只指向tb_logger目录，让train.py创建具体的实验日志
tensorboard --port 6007 --logdir $TB_DIR > "${LOG_DIR}/tensorboard_${TIMESTAMP}.log" 2>&1 &
TB_PID=$!
echo "TensorBoard已启动，PID: $TB_PID，访问地址: http://localhost:6007"
echo $TB_PID > "$TB_PID_FILE"

# 构建训练命令中添加参数
REUSE_EXP_OPT=""
if [ "$REUSE_EXP_DIR" = true ]; then
    REUSE_EXP_OPT="--reuse_exp_dir"
fi

# 启动训练
echo "开始训练..."
if [ "$LAUNCHER" = "pytorch" ]; then
    # 使用torchrun启动分布式训练
    echo "使用分布式训练模式 (torchrun)..."
    nohup python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPU \
        --master_port=$(( ( RANDOM % 50000 )  + 10000 )) \
        "${PROJECT_ROOT}/tgsr/train.py" \
        --launcher pytorch \
        --opt $CONFIG \
        $DEBUG $MEMORY_OPT $REUSE_EXP_OPT > $LOG_FILE 2>&1 &
else
    # 使用单GPU训练
    echo "使用单GPU训练模式..."
    nohup python "${PROJECT_ROOT}/tgsr/train.py" \
        --launcher none \
        --opt $CONFIG \
        $DEBUG $MEMORY_OPT $REUSE_EXP_OPT > $LOG_FILE 2>&1 &
fi

TRAIN_PID=$!
echo "训练进程已在后台启动，PID: $TRAIN_PID"
echo $TRAIN_PID > "${LOG_DIR}/train_pid.txt"

echo "训练日志将保存到: $LOG_FILE"
echo "使用 'tail -f $LOG_FILE' 查看训练日志"
echo "训练可以在TensorBoard中监控: http://localhost:6007" 