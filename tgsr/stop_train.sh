#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 从PID文件中读取训练进程PID
PID_FILE="$SCRIPT_DIR/train_pid.txt"
echo "正在停止训练进程..."

if [ -f "$PID_FILE" ]; then
    TRAIN_PID=$(cat "$PID_FILE")
    if [ -n "$TRAIN_PID" ] && ps -p $TRAIN_PID > /dev/null; then
        echo "找到训练进程 PID: $TRAIN_PID，正在终止..."
        kill -9 $TRAIN_PID
        echo "训练进程已终止"
        # 删除PID文件
        rm "$PID_FILE"
        echo "已删除PID文件: $PID_FILE"
    else
        echo "PID文件中的进程($TRAIN_PID)不存在或已终止"
        rm -f "$PID_FILE"
    fi
else
    echo "PID文件不存在: $PID_FILE"
fi

# 清除实验结果
echo "是否清除所有实验结果? [y/N]"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "清除实验结果..."
    
    # 清理experiments目录
    if [ -d "$PROJECT_ROOT/experiments" ]; then
        rm -rf $PROJECT_ROOT/experiments
        echo "已清除 $PROJECT_ROOT/experiments"
    else
        echo "$PROJECT_ROOT/experiments 不存在，跳过"
    fi
    
    # 清理tensorboard日志
    if [ -d "$PROJECT_ROOT/tb_logger" ]; then
        rm -rf $PROJECT_ROOT/tb_logger
        echo "已清除 $PROJECT_ROOT/tb_logger"
    else
        echo "$PROJECT_ROOT/tb_logger 不存在，跳过"
    fi
    
    # 清理训练日志
    if [ -d "$PROJECT_ROOT/logs" ]; then
        rm -rf $PROJECT_ROOT/logs
        echo "已清除 $PROJECT_ROOT/logs"
    else
        echo "$PROJECT_ROOT/logs 不存在，跳过"
    fi
    
    echo "清除完成"
else
    echo "保留实验结果，操作已取消"
fi

echo "操作完成" 