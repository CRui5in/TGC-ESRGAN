"""
COCO数据集加载与处理
为TGSR测试提供数据集加载和提示词生成功能
"""
import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import logging


class COCOPromptGenerator:
    """COCO数据集提示词生成器"""
    def __init__(self, annotation_path: str, caption_path: str, 
                 prompt_type: str = "category", separator: str = " . "):
        """
        初始化提示词生成器
        
        Args:
            annotation_path: COCO实例标注文件路径
            caption_path: COCO描述文件路径
            prompt_type: 提示词类型，'category'或'caption'
            separator: 类别提示词之间的分隔符
        """
        self.annotation_path = annotation_path
        self.caption_path = caption_path
        self.prompt_type = prompt_type
        self.separator = separator
        
        # 检查文件是否存在
        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"标注文件不存在: {self.annotation_path}")
        
        if not os.path.exists(self.caption_path):
            raise FileNotFoundError(f"描述文件不存在: {self.caption_path}")
        
        # 加载标注文件
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 加载描述文件
        with open(self.caption_path, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)
        
        # 构建类别ID到名称的映射
        self.categories = {cat['id']: cat['name'] for cat in self.annotations['categories']}
        
        # 构建图像ID到标注的映射
        self.image_to_anns = self._build_image_to_anns()
        
        # 构建图像ID到描述的映射
        self.image_to_caps = self._build_image_to_caps()
        
        # 输出类别信息
        logger = logging.getLogger()
        logger.info(f"已加载 {len(self.categories)} 个类别")
        logger.info(f"已加载 {len(self.annotations['images'])} 张图像")
    
    def _build_image_to_anns(self) -> Dict[int, List[Dict]]:
        """构建图像ID到标注的映射"""
        image_to_anns = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in image_to_anns:
                image_to_anns[image_id] = []
            image_to_anns[image_id].append(ann)
        return image_to_anns
    
    def _build_image_to_caps(self) -> Dict[int, List[str]]:
        """构建图像ID到描述的映射"""
        image_to_caps = {}
        for cap in self.captions['annotations']:
            image_id = cap['image_id']
            if image_id not in image_to_caps:
                image_to_caps[image_id] = []
            image_to_caps[image_id].append(cap['caption'])
        return image_to_caps
    
    def get_category_prompt(self, image_id: int) -> str:
        """获取基于类别的提示词
        
        Args:
            image_id: 图像ID
            
        Returns:
            str: 连接后的类别提示词
        """
        if image_id not in self.image_to_anns:
            return ""
        
        # 获取图像中所有物体的类别
        categories = set()
        for ann in self.image_to_anns[image_id]:
            category_id = ann['category_id']
            if category_id in self.categories:
                categories.add(self.categories[category_id])
        
        # 连接所有类别名称
        return self.separator.join(categories)
    
    def get_caption_prompt(self, image_id: int) -> str:
        """获取基于描述的提示词
        
        Args:
            image_id: 图像ID
            
        Returns:
            str: 随机选择的图像描述
        """
        if image_id not in self.image_to_caps:
            return ""
        
        # 随机选择一条描述
        captions = self.image_to_caps[image_id]
        return random.choice(captions) if captions else ""
    
    def get_prompt(self, image_id: int) -> str:
        """获取图像的提示词
        
        Args:
            image_id: 图像ID
            
        Returns:
            str: 提示词
        """
        if self.prompt_type == "category":
            return self.get_category_prompt(image_id)
        elif self.prompt_type == "caption":
            return self.get_caption_prompt(image_id)
        else:
            raise ValueError(f"不支持的提示词类型: {self.prompt_type}")
    
    def get_image_objects(self, image_id: int) -> List[Dict]:
        """获取图像中的所有物体标注
        
        Args:
            image_id: 图像ID
            
        Returns:
            List[Dict]: 物体标注列表
        """
        if image_id not in self.image_to_anns:
            return []
        
        return self.image_to_anns[image_id]
    
    def get_image_info(self, image_id: int) -> Optional[Dict]:
        """获取图像信息
        
        Args:
            image_id: 图像ID
            
        Returns:
            Optional[Dict]: 图像信息字典
        """
        for img in self.annotations['images']:
            if img['id'] == image_id:
                return img
        return None


class COCOTestDataset(Dataset):
    """COCO测试数据集"""
    def __init__(self, config, apply_degradation: bool = False):
        """
        初始化COCO测试数据集
        
        Args:
            config: 测试配置
            apply_degradation: 是否应用退化处理
        """
        self.config = config
        self.apply_degradation = apply_degradation
        self.images_dir = config.val_images_path
        
        # 检查图像目录是否存在
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.images_dir}")
        
        # 加载标注文件
        with open(config.annotation_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 获取图像列表
        self.images = self.annotations['images']
        
        # 创建提示词生成器
        self.prompt_generator = COCOPromptGenerator(
            config.annotation_path, 
            config.caption_path, 
            config.prompt_type, 
            config.prompt_separator
        )
        
        # 图像转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # 如果需要应用退化，加载退化模块
        if self.apply_degradation:
            from tgsr.models.degradation_layers import DegradationModule
            self.degradation = DegradationModule({})
        
        # 记录信息
        logger = logging.getLogger()
        logger.info(f"数据集包含 {len(self.images)} 张图像")
        logger.info(f"图像目录: {self.images_dir}")
        logger.info(f"提示词类型: {config.prompt_type}")
        logger.info(f"退化处理: {'开启' if self.apply_degradation else '关闭'}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取数据集项
        
        Args:
            idx: 索引
            
        Returns:
            Dict: 包含图像和其他信息的字典
        """
        # 获取图像信息
        img_info = self.images[idx]
        img_id = img_info['id']
        img_file = img_info['file_name']
        img_path = os.path.join(self.images_dir, img_file)
        
        # 检查文件是否存在
        if not os.path.exists(img_path):
            logger = logging.getLogger()
            logger.warning(f"图像文件不存在: {img_path}")
            # 返回一个空图像作为替代
            h, w = 384, 384  # 默认尺寸
            lq_tensor = torch.zeros(3, h // self.config.scale, w // self.config.scale)
            gt_tensor = torch.zeros(3, h, w)
            return {
                'lq': lq_tensor,
                'gt': gt_tensor,
                'text_prompt': "",
                'img_id': img_id,
                'img_path': img_path,
                'objects': [],
                'img_info': img_info
            }
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
            gt_tensor = self.transform(img)
        except Exception as e:
            logger = logging.getLogger()
            logger.error(f"加载图像失败: {img_path}, 错误: {str(e)}")
            # 返回一个空图像作为替代
            h, w = 384, 384  # 默认尺寸
            lq_tensor = torch.zeros(3, h // self.config.scale, w // self.config.scale)
            gt_tensor = torch.zeros(3, h, w)
            return {
                'lq': lq_tensor,
                'gt': gt_tensor,
                'text_prompt': "",
                'img_id': img_id,
                'img_path': img_path,
                'objects': [],
                'img_info': img_info
            }
        
        # 获取提示词
        prompt = self.prompt_generator.get_prompt(img_id)
        
        # 获取物体标注
        objects = self.prompt_generator.get_image_objects(img_id)
        
        # 准备低分辨率输入
        try:
            if self.apply_degradation:
                # 使用退化模块生成低质量图像
                with torch.no_grad():
                    lq_tensor = self.degradation(gt_tensor.unsqueeze(0))[0]
            else:
                # 直接下采样
                h, w = gt_tensor.shape[1], gt_tensor.shape[2]
                lq_tensor = transforms.functional.resize(
                    gt_tensor, 
                    (h // self.config.scale, w // self.config.scale),
                    interpolation=transforms.InterpolationMode.BICUBIC
                )
        except Exception as e:
            logger = logging.getLogger()
            logger.error(f"生成低分辨率图像失败: {img_path}, 错误: {str(e)}")
            # 使用简单的下采样替代
            h, w = gt_tensor.shape[1], gt_tensor.shape[2]
            lq_tensor = transforms.functional.resize(
                gt_tensor, 
                (h // self.config.scale, w // self.config.scale),
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        
        return {
            'lq': lq_tensor,         # 低分辨率图像
            'gt': gt_tensor,         # 高分辨率图像（GT）
            'text_prompt': prompt,   # 提示词
            'img_id': img_id,        # 图像ID
            'img_path': img_path,    # 图像路径
            'objects': objects,      # 物体标注
            'img_info': img_info     # 图像信息
        }


def build_dataloader(config) -> DataLoader:
    """构建数据加载器
    
    Args:
        config: 测试配置
        
    Returns:
        DataLoader: 数据加载器
    """
    try:
        dataset = COCOTestDataset(config, config.apply_degradation)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=min(config.num_workers, 4),  # Windows环境中限制工作线程数
            pin_memory=True
        )
        return dataloader
    except Exception as e:
        logger = logging.getLogger()
        logger.error(f"构建数据加载器失败: {str(e)}")
        raise 