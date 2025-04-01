#!/bin/bash

# 设置工作目录
WORK_DIR="/root/autodl-tmp/TGSR"
cd $WORK_DIR

# 环境变量
export PYTHONPATH=.:$PYTHONPATH

# 设置路径
COCO_DIR="/root/autodl-tmp/COCO2017"  # COCO数据集路径
DATASET_NEW="/root/autodl-tmp/tgsr_dataset_hr_only"  # 新数据集输出路径

# 确认是否需要生成数据集
if [ ! -d "$DATASET_NEW" ]; then
    echo "开始从COCO2017生成新的数据集..."
    python datasets_coco.py \
        --coco_dir $COCO_DIR \
        --output_dir $DATASET_NEW \
        --jpeg_quality 100 \
        --max_workers 8
else
    echo "数据集已存在，跳过生成步骤"
    echo "如果需要重新生成数据集，请先删除目录: $DATASET_NEW"
fi

echo "数据集准备完成!" 