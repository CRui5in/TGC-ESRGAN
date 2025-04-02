import os
import random
import torch
import numpy as np
import json
import math
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from collections import OrderedDict
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import (
    FileClient, 
    get_root_logger, 
    imfrombytes, 
    img2tensor, 
    USMSharp,
    DiffJPEG
)
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.data_util import paths_from_lmdb
from basicsr.data.degradations import (
    random_add_gaussian_noise, 
    random_add_poisson_noise,
    circular_lowpass_kernel, 
    random_mixed_kernels
)
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image


@DATASET_REGISTRY.register()
class TGSRDataset(Dataset):
    """文本引导超分辨率数据集 (RealESRGAN风格)
    
    只需要HR图像，在训练时使用RealESRGAN风格退化生成LR图像。
    对于文本引导区域，使用较轻的退化以保留文本特征。
    
    支持COCO-TGSR格式的JSON文件。
    """

    def __init__(self, opt):
        super(TGSRDataset, self).__init__()
        self.opt = opt
        
        # 基本设置
        self.phase = opt.get('phase', 'train')
        self.use_original_size = opt.get('use_original_size', True)
        self.use_hflip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)

        self.logger = get_root_logger()
        
        # 数据加载设置
        self.io_backend_opt = opt.get('io_backend', {})
        self.file_client = FileClient(self.io_backend_opt.get('type', 'disk'))
        
        # 获取数据集配置
        self.gt_folder = opt['dataroot_gt']
        self.text_file = opt.get('text_file')
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        
        # 数据增强配置
        self.is_train = self.phase == 'train'
        self.gt_size = opt.get('gt_size', 256)
        self.scale = opt.get('scale', 4)
        self.out_size = self.gt_size // self.scale  # LQ尺寸
        
        # 加载图像路径和文本描述
        captions_data, img_paths = self._load_text_prompts(opt['text_file'])
        self.gt_paths = img_paths
        # 将文本提示转换为字典格式
        self.text_prompts = {item['hr_path']: item['caption'] for item in captions_data}
        
        # 数据增强设置
        self.transform = self._get_transform()
        
        # 初始化basicsr工具，不在初始化时使用cuda
        self.usm_sharpen = USMSharp()
        self.jpeg = DiffJPEG(differentiable=False)
        
        # RealESRGAN风格退化参数
        self.blur_kernel_size = opt.get('blur_kernel_size', 21)
        self.kernel_list = opt.get('kernel_list', ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'])
        self.kernel_prob = opt.get('kernel_prob', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.sinc_prob = opt.get('sinc_prob', 0.1)
        self.blur_sigma = opt.get('blur_sigma', [0.2, 3.0])
        self.betag_range = opt.get('betag_range', [0.5, 4.0])
        self.betap_range = opt.get('betap_range', [1.0, 2.0])
        
        # 第二次退化参数
        self.blur_kernel_size2 = opt.get('blur_kernel_size2', 21)
        self.kernel_list2 = opt.get('kernel_list2', ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'])
        self.kernel_prob2 = opt.get('kernel_prob2', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.sinc_prob2 = opt.get('sinc_prob2', 0.1)
        self.blur_sigma2 = opt.get('blur_sigma2', [0.2, 1.5])
        self.betag_range2 = opt.get('betag_range2', [0.5, 4.0])
        self.betap_range2 = opt.get('betap_range2', [1.0, 2.0])
        
        # 最终sinc滤波器配置
        self.final_sinc_prob = opt.get('final_sinc_prob', 0.8)
        
        # 噪声配置
        self.gaussian_noise_prob = opt.get('gaussian_noise_prob', 0.5)
        self.noise_range = opt.get('noise_range', [1, 30])
        self.poisson_scale_range = opt.get('poisson_scale_range', [0.05, 3])
        self.gray_noise_prob = opt.get('gray_noise_prob', 0.4)
        self.gaussian_noise_prob2 = opt.get('gaussian_noise_prob2', 0.5)
        self.noise_range2 = opt.get('noise_range2', [1, 25])
        self.poisson_scale_range2 = opt.get('poisson_scale_range2', [0.05, 2.5])
        self.gray_noise_prob2 = opt.get('gray_noise_prob2', 0.4)
        
        # JPEG压缩配置
        self.jpeg_range = opt.get('jpeg_range', [30, 95])
        self.jpeg_range2 = opt.get('jpeg_range2', [30, 95])
        
        # 调整大小配置
        self.resize_prob = opt.get('resize_prob', [0.2, 0.7, 0.1])
        self.resize_range = opt.get('resize_range', [0.15, 1.5])
        self.resize_prob2 = opt.get('resize_prob2', [0.3, 0.4, 0.3])
        self.resize_range2 = opt.get('resize_range2', [0.3, 1.2])
        self.second_blur_prob = opt.get('second_blur_prob', 0.8)
        
        # 用于GT锐化的USM设置
        self.l1_gt_usm = opt.get('l1_gt_usm', True)
        self.percep_gt_usm = opt.get('percep_gt_usm', True)
        self.gan_gt_usm = opt.get('gan_gt_usm', False)
    
    def _get_transform(self):
        """获取数据增强转换"""
        transforms_list = []
        
        # 水平翻转
        if self.use_hflip:
            transforms_list.append(
                transforms.RandomHorizontalFlip(p=0.5)
            )
            
        # 旋转
        if self.use_rot:
            transforms_list.append(
                transforms.RandomRotation(degrees=90)
            )
            
        # 调整大小
        transforms_list.append(
            transforms.Resize((self.gt_size, self.gt_size), antialias=True)
        )
            
        # 添加ToTensor转换
        transforms_list.append(transforms.ToTensor())
            
        return transforms.Compose(transforms_list)
    
    def __getitem__(self, index):
        """获取数据项"""
        # 加载图像
        gt_path = self.gt_paths[index]
        img_gt = self._load_image(gt_path)
        
        # 数据增强
        if self.transform is not None:
            # 将numpy数组转换为PIL图像，确保RGB顺序
            img_gt = (img_gt * 255).astype(np.uint8)
            if img_gt.shape[2] == 3:  # RGB图像
                # 确保RGB顺序
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
                img_gt = Image.fromarray(img_gt)
            else:  # 灰度图像
                img_gt = Image.fromarray(img_gt[:, :, 0], mode='L')
            
            # 应用变换（包括调整大小和ToTensor）
            img_gt = self.transform(img_gt)
        
        # 获取文本描述
        text_prompt = self.text_prompts.get(gt_path, '')
        
        return {
            'lq': img_gt,  # 这里lq和gt是同一个图像，退化操作会在模型中完成
            'gt': img_gt,
            'text_prompt': text_prompt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def _load_text_prompts(self, text_file):
        """加载文本提示文件，支持COCO-TGSR格式的JSON文件"""
        default_prompt = "a high-resolution image"
        captions_data = []
        img_paths = []
        
        # 如果没有指定文本文件，使用默认提示
        if text_file is None or not os.path.exists(text_file):
            self.logger.info(f'文本提示文件未找到，使用默认提示: "{default_prompt}"')
            # 获取图像列表
            if self.io_backend_opt['type'] == 'lmdb':
                img_paths = paths_from_lmdb(self.gt_folder)
            else:
                img_paths = sorted(list(Path(self.gt_folder).glob('*')))
                img_paths = [str(path) for path in img_paths]
                
            # 为每个图像创建默认文本项
            for path in img_paths:
                img_name = os.path.basename(path)
                captions_data.append({
                    'image_id': os.path.splitext(img_name)[0],
                    'caption': default_prompt,
                    'hr_path': path
                })
            return captions_data, img_paths
            
        try:
            # 加载JSON格式文件
            with open(text_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # COCO-TGSR格式JSON处理
            if isinstance(json_data, list):
                # 遍历每个数据项
                for item in json_data:
                    if isinstance(item, dict) and 'image_id' in item and 'caption' in item:
                        if 'hr_path' in item:
                            hr_path = item['hr_path']
                        else:
                            # 根据image_id构建路径
                            img_name = f"{item['image_id']}.jpg"  # 假设为jpg格式
                            hr_path = os.path.join(self.gt_folder, img_name)
                        
                        # 检查文件是否存在
                        if os.path.exists(hr_path):
                            # 更新路径信息
                            item['hr_path'] = hr_path
                            captions_data.append(item)
                            img_paths.append(hr_path)
            else:
                self.logger.error(f'不支持的JSON格式: {text_file}')
                
        except Exception as e:
            self.logger.error(f'加载文本提示时出错: {e}')
            # 获取图像列表并使用默认提示
            if self.io_backend_opt['type'] == 'lmdb':
                img_paths = paths_from_lmdb(self.gt_folder)
            else:
                img_paths = sorted(list(Path(self.gt_folder).glob('*')))
                img_paths = [str(path) for path in img_paths]
                
            # 为每个图像创建默认文本项
            for path in img_paths:
                img_name = os.path.basename(path)
                captions_data.append({
                    'image_id': os.path.splitext(img_name)[0],
                    'caption': default_prompt,
                    'hr_path': path
                })
            
        # 若文本提示为空，使用默认配置
        if len(captions_data) == 0:
            self.logger.warning(f'在{text_file}中未找到有效的文本提示，使用默认提示')
            if self.io_backend_opt['type'] == 'lmdb':
                img_paths = paths_from_lmdb(self.gt_folder)
            else:
                img_paths = sorted(list(Path(self.gt_folder).glob('*')))
                img_paths = [str(path) for path in img_paths]
                
            # 为每个图像创建默认文本项
            for path in img_paths:
                img_name = os.path.basename(path)
                captions_data.append({
                    'image_id': os.path.splitext(img_name)[0],
                    'caption': default_prompt,
                    'hr_path': path
                })
                
        self.logger.info(f'已加载{len(captions_data)}个带有文本提示的数据项')
        return captions_data, img_paths
        
    def __len__(self):
        return len(self.gt_paths)
    
    def generate_kernel(self, kernel_size, kernel_list, kernel_prob, blur_sigma, betag_range, betap_range, sinc_prob):
        """使用basicsr的random_mixed_kernels生成模糊内核"""
        kernel_type = random.choices(kernel_list, kernel_prob)[0]
        
        if kernel_type == 'sinc' or np.random.uniform() < sinc_prob:
            # 使用basicsr的circular_lowpass_kernel
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            # 使用basicsr的random_mixed_kernels
            kernel = random_mixed_kernels(
                kernel_list,
                kernel_prob,
                kernel_size,
                blur_sigma,
                blur_sigma,
                [-math.pi, math.pi],
                betag_range,
                betap_range,
                noise_range=None
            )
        
        return kernel
    
    def jpeg_compression(self, img, quality):
        """使用basicsr的DiffJPEG进行JPEG压缩"""
        # 将numpy数组转换为tensor，不使用cuda
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        # 应用JPEG压缩
        img_tensor = self.jpeg(img_tensor, quality)
        # 转回numpy数组，先分离梯度
        img = img_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
        return img 

    def _get_paths_from_lmdb(self, dataroot):
        """从LMDB或文件夹中获取图像路径"""
        if self.io_backend_opt['type'] == 'lmdb':
            paths = paths_from_lmdb(dataroot)
        else:
            paths = sorted(list(Path(dataroot).glob('*')))
            paths = [str(path) for path in paths]
        return paths

    def _load_image(self, path):
        """加载图像"""
        img_bytes = self.file_client.get(path)
        img = imfrombytes(img_bytes, float32=True)
        return img 