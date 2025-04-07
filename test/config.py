"""
测试配置文件
包含TGSR模型测试的配置选项
"""
import os
import platform
from typing import Optional, List, Dict, Union, Any
from dataclasses import dataclass, field
import torch
import logging


@dataclass
class TestConfig:
    """测试配置类"""
    # 基本路径配置
    coco_path: str = os.path.join(os.getcwd(), "COCO2017")  # COCO数据集路径
    model_path: str = os.path.join(os.getcwd(), "TGSR", "experiments", "models", "net_g_250000.pth")  # 模型路径
    output_dir: str = os.path.join(os.getcwd(), "results")  # 输出目录
    
    # 数据集配置
    annotation_file: str = "annotations/instances_val2017.json" if platform.system() != "Windows" else "annotations\\instances_val2017.json"  # 标注文件
    caption_file: str = "annotations/captions_val2017.json" if platform.system() != "Windows" else "annotations\\captions_val2017.json"  # 图像描述文件
    val_set: str = "val2017"  # 验证集目录
    
    # 模型配置
    use_ema: bool = True  # 是否使用EMA模型
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 使用的设备
    scale: int = 4  # 超分辨率比例
    
    # CLIP配置
    clip_model_path: str = os.path.join(os.getcwd(), "clip-vit-base-patch32")  # CLIP模型路径
    
    # 测试配置
    batch_size: int = 1  # 批处理大小
    num_workers: int = 0 if platform.system() == "Windows" else 4  # 数据加载线程数，Windows下默认为0避免多进程问题
    save_images: bool = True  # 是否保存图像
    apply_degradation: bool = False  # 是否应用图像退化
    
    # 提示词配置
    prompt_type: str = "category"  # 'category' 或 'caption'
    prompt_separator: str = " . "  # 类别连接符
    
    # 目标检测配置
    detection_model_name: str = "IDEA-Research/grounding-dino-base"  # GroundingDINO模型名称
    box_threshold: float = 0.4  # 检测框置信度阈值
    text_threshold: float = 0.3  # 文本匹配阈值
    
    # 评估配置
    eval_metrics: List[str] = field(default_factory=lambda: ["psnr", "ssim", "map"])
    
    # 可视化配置
    visualize: bool = True  # 是否可视化结果
    attention_maps: bool = True  # 是否生成注意力热力图
    colormap: str = "jet"  # 热力图颜色映射
    
    def __post_init__(self):
        """初始化后处理"""
        # 设置日志
        logger = logging.getLogger()
        
        # 输出系统信息
        logger.info(f"运行环境: {platform.system()} {platform.release()}")
        logger.info(f"Python版本: {platform.python_version()}")
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        
        # 确保路径存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 修复路径分隔符
        self.coco_path = os.path.normpath(self.coco_path)
        self.model_path = os.path.normpath(self.model_path)
        self.output_dir = os.path.normpath(self.output_dir)
        self.clip_model_path = os.path.normpath(self.clip_model_path)
        
        # 构建完整的数据集路径
        self.annotation_path = os.path.normpath(os.path.join(self.coco_path, self.annotation_file))
        self.caption_path = os.path.normpath(os.path.join(self.coco_path, self.caption_file))
        self.val_images_path = os.path.normpath(os.path.join(self.coco_path, self.val_set))
        
        # 输出重要路径信息
        logger.info(f"COCO数据集路径: {self.coco_path}")
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"CLIP模型路径: {self.clip_model_path}")
        logger.info(f"输出目录: {self.output_dir}")
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            logger.warning(f"模型文件不存在: {self.model_path}")
        
        # 检查CLIP模型是否存在
        if not os.path.exists(self.clip_model_path):
            logger.warning(f"CLIP模型不存在: {self.clip_model_path}")
        
        # 仅当需要验证时检查数据集文件
        if os.path.exists(self.coco_path):
            # 判断必要文件是否存在
            if not os.path.exists(self.annotation_path):
                logger.warning(f"找不到标注文件: {self.annotation_path}")
            
            if not os.path.exists(self.caption_path):
                logger.warning(f"找不到描述文件: {self.caption_path}")
            
            if not os.path.exists(self.val_images_path):
                logger.warning(f"找不到验证集目录: {self.val_images_path}")
        else:
            logger.warning(f"COCO数据集路径不存在: {self.coco_path}")
    
    def get_prompt_config(self) -> Dict[str, Any]:
        """获取提示词配置"""
        return {
            "type": self.prompt_type,
            "separator": self.prompt_separator
        }
    
    def get_detection_config(self) -> Dict[str, Any]:
        """获取目标检测配置"""
        return {
            "model_name": self.detection_model_name,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "path": self.model_path,
            "use_ema": self.use_ema,
            "device": self.device,
            "scale": self.scale,
            "clip_model_path": self.clip_model_path
        } 