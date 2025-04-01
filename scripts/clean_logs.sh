#!/bin/bash

# 清理日志文件的脚本

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 获取项目根目录
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

echo "准备清理日志文件..."

# 确保没有训练进程在运行
if [ -f "${PROJECT_ROOT}/logs/train_pid.txt" ] || [ -f "${PROJECT_ROOT}/logs/tensorboard_pid.txt" ]; then
    echo "警告: 检测到训练进程可能仍在运行。请先运行 ${SCRIPT_DIR}/stop_train.sh 停止所有进程。"
    read -p "是否仍要继续? (y/n): " CONTINUE
    if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
        echo "取消操作。"
        exit 0
    fi
fi

# 创建日志目录如果不存在
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p $LOG_DIR

# 创建备份文件夹
BACKUP_DIR="${LOG_DIR}/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

echo "将所有日志备份到 $BACKUP_DIR..."

# 复制所有日志到备份文件夹
cp -a ${LOG_DIR}/*.log ${BACKUP_DIR}/ 2>/dev/null || echo "没有找到日志文件"
cp -a ${LOG_DIR}/*.txt ${BACKUP_DIR}/ 2>/dev/null || echo "没有找到PID文件"

# 删除原始日志
echo "删除原始日志文件..."
rm -f ${LOG_DIR}/*.log
rm -f ${LOG_DIR}/*.txt

echo "日志清理完成。所有日志已备份到 $BACKUP_DIR"

# 询问是否清理TensorBoard日志
read -p "是否清理TensorBoard日志? (y/n): " CLEAN_TB
if [[ "$CLEAN_TB" == "y" || "$CLEAN_TB" == "Y" ]]; then
    TB_DIR="${PROJECT_ROOT}/tb_logger"
    if [ -d "$TB_DIR" ]; then
        echo "备份TensorBoard日志..."
        TB_BACKUP_DIR="${BACKUP_DIR}/tensorboard"
        mkdir -p $TB_BACKUP_DIR
        cp -a ${TB_DIR}/* ${TB_BACKUP_DIR}/ 2>/dev/null || echo "TensorBoard目录为空"
        
        echo "删除TensorBoard日志..."
        rm -rf ${TB_DIR}/*
        echo "TensorBoard日志已清理并备份"
    else
        echo "TensorBoard目录不存在，跳过"
    fi
else
    echo "保留TensorBoard日志"
fi

echo "清理完成!" 