#!/bin/bash

# 设置路径
COCO_PATH="/root/autodl-tmp/COCO2017"
MODEL_PATH="/root/autodl-tmp/experiments/pretrained_models/net_g_latest.pth"
OUTPUT_DIR="/root/autodl-tmp/TGSR/test/results"
DETECTION_MODEL_PATH="/path/to/groundingdino_swinb_cogcoor.pth"
DETECTION_CONFIG_PATH="/path/to/GroundingDINO_SwinB.cfg.py"

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 运行类别提示词测试
echo "运行类别提示词测试..."
python test_tgsr.py \
    --coco_path $COCO_PATH \
    --model_path $MODEL_PATH \
    --output_dir "${OUTPUT_DIR}/category" \
    --prompt_type category \
    --use_ema \
    --detection_model_path $DETECTION_MODEL_PATH \
    --detection_config_path $DETECTION_CONFIG_PATH

# 运行描述提示词测试
echo "运行描述提示词测试..."
python test_tgsr.py \
    --coco_path $COCO_PATH \
    --model_path $MODEL_PATH \
    --output_dir "${OUTPUT_DIR}/caption" \
    --prompt_type caption \
    --use_ema \
    --detection_model_path $DETECTION_MODEL_PATH \
    --detection_config_path $DETECTION_CONFIG_PATH

# 运行应用退化的测试
echo "运行应用退化的测试..."
python test_tgsr.py \
    --coco_path $COCO_PATH \
    --model_path $MODEL_PATH \
    --output_dir "${OUTPUT_DIR}/degradation" \
    --prompt_type category \
    --apply_degradation \
    --use_ema \
    --detection_model_path $DETECTION_MODEL_PATH \
    --detection_config_path $DETECTION_CONFIG_PATH

echo "测试完成！" 