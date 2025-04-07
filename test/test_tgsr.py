"""
TGSR测试主脚本
用于测试TGSR模型在提升开放词汇目标检测能力上的效果
"""
import os
import sys
import time
import json
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# 添加TGSR路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TestConfig
from dataset import build_dataloader
from evaluator import SuperResolutionEvaluator, DetectionEvaluator

# 导入TGSR模型
from tgsr.models.tgsr_model import TGSRModel
from basicsr.archs import build_network
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str


def load_tgsr_model(config):
    """
    加载TGSR模型
    
    Args:
        config: 测试配置
        
    Returns:
        TGSRModel: 加载好的模型
    """
    # 创建日志记录器
    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join(config.output_dir, f'test_{get_time_str()}.log')
    logger = get_root_logger(logging.INFO, log_file)
    logger.info(f'测试配置:\n{dict2str(vars(config))}')
    
    # 构建模型选项字典
    opt = {
        'name': 'TGSR',
        'type': 'TGSRModel',
        'path': {'experiments_root': config.output_dir, 'log': config.output_dir},
        'network_g': {
            'type': 'TGSRNet',
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'text_dim': 512,
            'use_text_features': True,
            'num_heads': 8,
            'scale': config.scale
        },
        'scale': config.scale,
        'text_encoder': {
            'name': config.clip_model_path,  # 使用本地CLIP路径
            'text_dim': 512,
            'freeze': True,
            'save_with_model': False
        },
        'use_ema': config.use_ema
    }
    
    # 创建模型实例
    model = TGSRModel(opt)
    
    # 加载模型权重
    logger.info(f'加载模型权重: {config.model_path}')
    if os.path.exists(config.model_path):
        model.load_network(model.net_g, 'net_g', config.model_path)
    else:
        logger.error(f"模型权重文件不存在: {config.model_path}")
        raise FileNotFoundError(f"模型权重文件不存在: {config.model_path}")
    
    # 移动到指定设备
    model.net_g = model.net_g.to(config.device)
    
    # 如果使用EMA模型，需要切换
    if config.use_ema and hasattr(model, 'net_g_ema'):
        logger.info('使用EMA模型进行测试')
        model.net_g = model.net_g_ema
    
    # 设置为评估模式
    model.net_g.eval()
    
    return model


def test_tgsr(config):
    """
    测试TGSR模型
    
    Args:
        config: 测试配置
        
    Returns:
        Dict: 评估指标
    """
    # 加载模型
    model = load_tgsr_model(config)
    
    # 创建数据加载器
    dataloader = build_dataloader(config)
    
    # 创建评估器
    sr_evaluator = SuperResolutionEvaluator(config)
    detection_evaluator = DetectionEvaluator(config)
    
    # 创建输出目录
    os.makedirs(os.path.join(config.output_dir, 'sr_results'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'detection_results'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'attention_maps'), exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 开始测试
    logger = get_root_logger()
    logger.info(f'开始测试 {len(dataloader)} 张图像...')
    
    # 遍历数据集
    for idx, data in enumerate(tqdm(dataloader, desc='测试进度')):
        # 获取数据
        lq = data['lq'].to(config.device)
        gt = data['gt'].to(config.device)
        text_prompt = data['text_prompt'][0]
        img_id = data['img_id'].item()
        img_path = data['img_path'][0]
        objects = data['objects']
        img_info = data['img_info'][0]
        
        # 打印当前图像信息
        logger.debug(f'处理图像: {img_path}, ID: {img_id}, 提示词: {text_prompt}')
        
        # 处理GT边界框
        detection_evaluator.process_gt_boxes(objects, img_id, img_info)
        
        # 使用TGSR模型进行推理
        with torch.no_grad():
            # 如果有提示词，进行文本编码
            if text_prompt and model.use_text_features:
                text_hidden, text_pooled = model.extract_text_features(text_prompt)
                # 使用前向传播获取注意力图
                output = model.forward_with_attention(lq, text_pooled, return_attention=True)
            else:
                output = model.forward_with_attention(lq, return_attention=False)
        
        # 评估超分辨率质量
        sr_save_path = os.path.join(config.output_dir, 'sr_results', f'{img_id:012d}.png')
        sr_metrics = sr_evaluator.evaluate_image(output, gt, img_id, sr_save_path)
        
        # 获取注意力图并可视化
        if config.attention_maps and hasattr(model.net_g, 'attention_maps'):
            attention_maps = model.get_current_attention_maps()
            if attention_maps:
                # 处理注意力图
                for name, attn_map in attention_maps.items():
                    # 可视化注意力图
                    attn_save_path = os.path.join(
                        config.output_dir, 
                        'attention_maps', 
                        f'{img_id:012d}_{name}.png'
                    )
                    # 从 tgsr.utils.visualization_utils 导入 tensor2img
                    from tgsr.utils.visualization_utils import improved_tensor2img as tensor2img
                    sr_img_np = tensor2img(output, rgb2bgr=False)
                    sr_evaluator.visualize_attention(
                        attn_map, 
                        sr_img_np, 
                        attn_save_path, 
                        config.colormap
                    )
        
        # 在超分辨率图像上进行目标检测
        from tgsr.utils.visualization_utils import improved_tensor2img as tensor2img
        sr_img_np = tensor2img(output, rgb2bgr=False)
        detection_boxes = detection_evaluator.detect_objects(sr_img_np, text_prompt, img_id)
        
        # 可视化检测结果
        if detection_boxes:
            det_save_path = os.path.join(
                config.output_dir, 
                'detection_results', 
                f'{img_id:012d}.png'
            )
            detection_evaluator.visualize_detection(sr_img_np, detection_boxes, det_save_path)
    
    # 计算评估指标
    sr_metrics = sr_evaluator.get_mean_metrics()
    map_metrics = detection_evaluator.calculate_map()
    
    # 合并所有指标
    all_metrics = {**sr_metrics, **map_metrics}
    
    # 记录结束时间
    end_time = time.time()
    
    # 输出结果
    logger.info('-' * 50)
    logger.info('测试完成!')
    logger.info(f'总耗时: {end_time - start_time:.2f} 秒')
    logger.info('评估指标:')
    for name, value in all_metrics.items():
        logger.info(f'  {name}: {value:.4f}')
    
    # 保存指标
    metrics_path = os.path.join(config.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f'指标已保存到: {metrics_path}')
    
    return all_metrics


def run_tgsr_test(prompt_type='category', apply_degradation=False):
    """
    运行TGSR测试
    
    Args:
        prompt_type: 提示词类型 ('category' 或 'caption')
        apply_degradation: 是否应用图像退化
    """
    # 创建配置
    config = TestConfig(
        prompt_type=prompt_type,
        apply_degradation=apply_degradation,
        output_dir=os.path.join(os.getcwd(), "results", prompt_type)
    )
    
    # 运行测试
    return test_tgsr(config)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TGSR测试脚本')
    parser.add_argument('--coco_path', type=str, help='COCO数据集路径')
    parser.add_argument('--model_path', type=str, help='模型路径')
    parser.add_argument('--clip_model_path', type=str, help='CLIP模型路径')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--prompt_type', type=str, choices=['category', 'caption'],
                        default='category', help='提示词类型')
    parser.add_argument('--apply_degradation', action='store_true',
                        help='是否应用图像退化')
    parser.add_argument('--use_ema', action='store_true',
                        help='是否使用EMA模型')
    parser.add_argument('--detection_model_name', type=str, 
                        default="IDEA-Research/grounding-dino-base",
                        help='GroundingDINO模型名称')
    parser.add_argument('--box_threshold', type=float, default=0.4,
                        help='检测框置信度阈值')
    parser.add_argument('--text_threshold', type=float, default=0.3,
                        help='文本匹配阈值')
    
    args = parser.parse_args()
    return args


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置
    config = TestConfig()
    
    # 更新配置
    if args.coco_path:
        config.coco_path = args.coco_path
    if args.model_path:
        config.model_path = args.model_path
    if args.clip_model_path:
        config.clip_model_path = args.clip_model_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.prompt_type:
        config.prompt_type = args.prompt_type
    if args.apply_degradation:
        config.apply_degradation = args.apply_degradation
    if args.use_ema:
        config.use_ema = args.use_ema
    if args.detection_model_name:
        config.detection_model_name = args.detection_model_name
    if args.box_threshold:
        config.box_threshold = args.box_threshold
    if args.text_threshold:
        config.text_threshold = args.text_threshold
    
    # 运行测试
    test_tgsr(config)


if __name__ == '__main__':
    main() 