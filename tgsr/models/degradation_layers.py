import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import List, Tuple, Optional
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    random_mixed_kernels,
    circular_lowpass_kernel
)
import cv2

class DegradationModule(nn.Module):
    """退化模块，包含所有退化操作"""
    def __init__(self, opt):
        super(DegradationModule, self).__init__()
        self.opt = opt
        
        # 初始化basicsr工具
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpen = USMSharp().cuda()
        
        # 退化参数
        self.blur_kernel_size = opt.get('blur_kernel_size', 19)
        self.kernel_list = opt.get('kernel_list', ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'])
        self.kernel_prob = opt.get('kernel_prob', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.sinc_prob = opt.get('sinc_prob', 0.08)
        self.blur_sigma = opt.get('blur_sigma', [0.2, 2.8])
        self.betag_range = opt.get('betag_range', [0.5, 3.8])
        self.betap_range = opt.get('betap_range', [1, 1.8])
        
        # 第二次退化参数
        self.blur_kernel_size2 = opt.get('blur_kernel_size2', 19)
        self.kernel_list2 = opt.get('kernel_list2', ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'])
        self.kernel_prob2 = opt.get('kernel_prob2', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.sinc_prob2 = opt.get('sinc_prob2', 0.08)
        self.blur_sigma2 = opt.get('blur_sigma2', [0.2, 1.3])
        self.betag_range2 = opt.get('betag_range2', [0.5, 3.8])
        self.betap_range2 = opt.get('betap_range2', [1, 1.8])
        
        # 最终sinc滤波器
        self.final_sinc_prob = opt.get('final_sinc_prob', 0.7)
        
        # 噪声参数
        self.gaussian_noise_prob = opt.get('gaussian_noise_prob', 0.4)
        self.noise_range = opt.get('noise_range', [1, 25])
        self.poisson_scale_range = opt.get('poisson_scale_range', [0.05, 2.5])
        self.gray_noise_prob = opt.get('gray_noise_prob', 0.3)
        self.gaussian_noise_prob2 = opt.get('gaussian_noise_prob2', 0.4)
        self.noise_range2 = opt.get('noise_range2', [1, 20])
        self.poisson_scale_range2 = opt.get('poisson_scale_range2', [0.05, 2.0])
        self.gray_noise_prob2 = opt.get('gray_noise_prob2', 0.3)
        
        # JPEG压缩参数
        self.jpeg_range = opt.get('jpeg_range', [40, 95])
        self.jpeg_range2 = opt.get('jpeg_range2', [40, 95])
        
        # 调整大小参数
        self.resize_prob = opt.get('resize_prob', [0.2, 0.7, 0.1])
        self.resize_range = opt.get('resize_range', [0.15, 1.5])
        self.resize_prob2 = opt.get('resize_prob2', [0.3, 0.4, 0.3])
        self.resize_range2 = opt.get('resize_range2', [0.3, 1.2])
        self.second_blur_prob = opt.get('second_blur_prob', 0.8)
        
        # 预计算退化核
        self.register_buffer('kernel', self._generate_kernel())
        self.register_buffer('kernel2', self._generate_kernel2())
        self.register_buffer('sinc_kernel', self._generate_sinc_kernel(self.blur_kernel_size))
        
    def _generate_kernel(self) -> torch.Tensor:
        """生成第一次退化核"""
        kernel = random_mixed_kernels(
            kernel_list=self.kernel_list,
            kernel_prob=self.kernel_prob,
            kernel_size=self.blur_kernel_size,
            sigma_x_range=self.blur_sigma,
            sigma_y_range=self.blur_sigma,
            rotation_range=[-np.pi, np.pi],
            betag_range=self.betag_range,
            betap_range=self.betap_range,
            noise_range=None
        )
        # 确保kernel维度正确 [3, 1, kernel_size, kernel_size]
        kernel = torch.from_numpy(kernel).float()
        kernel = kernel.unsqueeze(0).repeat(3, 1, 1, 1)
        return kernel
    
    def _generate_kernel2(self) -> torch.Tensor:
        """生成第二次退化核"""
        kernel = random_mixed_kernels(
            kernel_list=self.kernel_list2,
            kernel_prob=self.kernel_prob2,
            kernel_size=self.blur_kernel_size2,
            sigma_x_range=self.blur_sigma2,
            sigma_y_range=self.blur_sigma2,
            rotation_range=[-np.pi, np.pi],
            betag_range=self.betag_range2,
            betap_range=self.betap_range2,
            noise_range=None
        )
        # 确保kernel维度正确 [3, 1, kernel_size, kernel_size]
        kernel = torch.from_numpy(kernel).float()
        kernel = kernel.unsqueeze(0).repeat(3, 1, 1, 1)
        return kernel
    
    def _generate_sinc_kernel(self, kernel_size):
        """生成sinc滤波器"""
        if kernel_size < 13:
            omega_c = float(torch.rand(1) * (np.pi - np.pi / 3) + np.pi / 3)
        else:
            omega_c = float(torch.rand(1) * (np.pi / 5 - np.pi / 3) + np.pi / 3)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=0)
        # 确保kernel维度正确 [3, 1, kernel_size, kernel_size]
        kernel = torch.from_numpy(kernel).float()
        kernel = kernel.unsqueeze(0).repeat(3, 1, 1, 1)
        return kernel
    
    def forward(self, img):
        """前向传播
        
        Args:
            img: 输入图像 [B, C, H, W]
            
        Returns:
            退化后的图像 [B, C, H, W]
        """
        # 保存原始尺寸
        ori_h, ori_w = img.shape[2:]
        
        # 1. 第一次退化
        with torch.no_grad():
            # 生成第一次退化的kernel
            kernel = self._generate_kernel()
            kernel = kernel.to(img.device)
            
            # 应用第一次退化
            img = F.conv2d(img, kernel, padding=self.blur_kernel_size//2, groups=3)
            
            # 清理显存
            del kernel
            torch.cuda.empty_cache()
            
            # 应用第一次噪声
            if torch.rand(1) < self.gaussian_noise_prob:
                img = self._add_gaussian_noise(img)
            
            # 应用第一次JPEG压缩
            if torch.rand(1) < self.sinc_prob:
                img = self._add_jpeg_compression(img)
            
            # 应用第一次调整大小
            if torch.rand(1) < self.resize_prob[0]:
                img = self._resize(img, self.resize_range[0], self.resize_range[1])
        
        # 2. 第二次退化
        with torch.no_grad():
            # 生成第二次退化的kernel
            kernel2 = self._generate_kernel2()
            kernel2 = kernel2.to(img.device)
            
            # 应用第二次退化
            img = F.conv2d(img, kernel2, padding=self.blur_kernel_size2//2, groups=3)
            
            # 清理显存
            del kernel2
            torch.cuda.empty_cache()
            
            # 应用第二次噪声
            if torch.rand(1) < self.gaussian_noise_prob2:
                img = self._add_gaussian_noise2(img)
            
            # 应用第二次JPEG压缩
            if torch.rand(1) < self.sinc_prob2:
                img = self._add_jpeg_compression2(img)
            
            # 应用第二次调整大小
            if torch.rand(1) < self.resize_prob2[0]:
                img = self._resize(img, self.resize_range2[0], self.resize_range2[1])
        
        # 3. 最终sinc滤波器
        with torch.no_grad():
            if torch.rand(1) < self.final_sinc_prob:
                # 生成sinc kernel
                sinc_kernel = self._generate_sinc_kernel(self.blur_kernel_size)
                sinc_kernel = sinc_kernel.to(img.device)
                
                # 应用sinc滤波器
                img = F.conv2d(img, sinc_kernel, padding=self.blur_kernel_size//2, groups=3)
                
                # 清理显存
                del sinc_kernel
                torch.cuda.empty_cache()
        
        # 4. 确保输出尺寸与输入一致
        with torch.no_grad():
            if img.shape[2:] != (ori_h, ori_w):
                img = F.interpolate(
                    img,
                    size=(ori_h, ori_w),
                    mode='bilinear',
                    align_corners=False
                )
        
        return img

    def _add_gaussian_noise(self, img):
        """添加高斯噪声"""
        with torch.no_grad():
            noise = torch.randn_like(img) * (torch.rand(1, device=img.device) * (self.noise_range[1] - self.noise_range[0]) + self.noise_range[0])
            if torch.rand(1, device=img.device) < self.gray_noise_prob:
                noise = noise.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            return img + noise

    def _add_gaussian_noise2(self, img):
        """添加第二次高斯噪声"""
        with torch.no_grad():
            noise = torch.randn_like(img) * (torch.rand(1, device=img.device) * (self.noise_range2[1] - self.noise_range2[0]) + self.noise_range2[0])
            if torch.rand(1, device=img.device) < self.gray_noise_prob2:
                noise = noise.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            return img + noise

    def _add_jpeg_compression(self, img):
        """添加JPEG压缩"""
        with torch.no_grad():
            quality = torch.rand(1, device=img.device) * (self.jpeg_range[1] - self.jpeg_range[0]) + self.jpeg_range[0]
            # 使用GPU上的JPEG压缩
            img = torch.clamp(img, 0, 1)
            img = self.jpeger(img, quality=quality)
            return img

    def _add_jpeg_compression2(self, img):
        """添加第二次JPEG压缩"""
        with torch.no_grad():
            quality = torch.rand(1, device=img.device) * (self.jpeg_range2[1] - self.jpeg_range2[0]) + self.jpeg_range2[0]
            # 使用GPU上的JPEG压缩
            img = torch.clamp(img, 0, 1)
            img = self.jpeger(img, quality=quality)
            return img

    def _resize(self, img, min_scale, max_scale):
        """调整图像大小"""
        with torch.no_grad():
            scale = torch.rand(1, device=img.device) * (max_scale - min_scale) + min_scale
            h, w = img.shape[2:]
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
            return img 