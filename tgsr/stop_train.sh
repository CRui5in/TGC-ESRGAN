#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 找到并终止训练进程
echo "正在停止训练进程..."
TRAIN_PID=$(ps -ef | grep "[p]ython -u train.py" | awk '{print $2}')

if [ -n "$TRAIN_PID" ]; then
    echo "找到训练进程 PID: $TRAIN_PID，正在终止..."
    kill -9 $TRAIN_PID
    echo "训练进程已终止"
else
    echo "未找到训练进程"
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