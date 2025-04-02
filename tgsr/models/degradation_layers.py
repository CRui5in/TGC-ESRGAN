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
        
        # 初始化basicsr工具，但不立即移动到CUDA
        # 工具将在使用时移动到相应设备
        self.jpeger = DiffJPEG(differentiable=False)
        self.usm_sharpen = USMSharp()
        
        # 退化参数
        self.blur_kernel_size = opt.get('blur_kernel_size', 21)
        self.kernel_list = opt.get('kernel_list', ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'])
        self.kernel_prob = opt.get('kernel_prob', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.sinc_prob = opt.get('sinc_prob', 0.1)
        self.blur_sigma = opt.get('blur_sigma', [0.2, 3.0])
        self.betag_range = opt.get('betag_range', [0.5, 4.0])
        self.betap_range = opt.get('betap_range', [1, 2.0])
        
        # 第二次退化参数
        self.blur_kernel_size2 = opt.get('blur_kernel_size2', 21)
        self.kernel_list2 = opt.get('kernel_list2', ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'])
        self.kernel_prob2 = opt.get('kernel_prob2', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.sinc_prob2 = opt.get('sinc_prob2', 0.1)
        self.blur_sigma2 = opt.get('blur_sigma2', [0.2, 1.5])
        self.betag_range2 = opt.get('betag_range2', [0.5, 4.0])
        self.betap_range2 = opt.get('betap_range2', [1, 2.0])
        
        # 最终sinc滤波器
        self.final_sinc_prob = opt.get('final_sinc_prob', 0.8)
        
        # 噪声参数
        self.gaussian_noise_prob = opt.get('gaussian_noise_prob', 0.5)
        self.noise_range = opt.get('noise_range', [1, 30])
        self.poisson_scale_range = opt.get('poisson_scale_range', [0.05, 3.0])
        self.gray_noise_prob = opt.get('gray_noise_prob', 0.4)
        self.gaussian_noise_prob2 = opt.get('gaussian_noise_prob2', 0.5)
        self.noise_range2 = opt.get('noise_range2', [1, 25])
        self.poisson_scale_range2 = opt.get('poisson_scale_range2', [0.05, 2.5])
        self.gray_noise_prob2 = opt.get('gray_noise_prob2', 0.4)
        
        # JPEG压缩参数
        self.jpeg_range = opt.get('jpeg_range', [30, 95])
        self.jpeg_range2 = opt.get('jpeg_range2', [30, 95])
        
        # 调整大小参数
        self.resize_prob = opt.get('resize_prob', [0.2, 0.7, 0.1])
        self.resize_range = opt.get('resize_range', [0.15, 1.5])
        self.resize_prob2 = opt.get('resize_prob2', [0.3, 0.4, 0.3])
        self.resize_range2 = opt.get('resize_range2', [0.3, 1.2])
        self.second_blur_prob = opt.get('second_blur_prob', 0.8)
        
    def _ensure_on_device(self, device):
        """确保所有组件都在相同的设备上"""
        # 将工具移至对应设备
        if not hasattr(self, '_initialized_device') or self._initialized_device != device:
            self.jpeger = self.jpeger.to(device)
            self.usm_sharpen = self.usm_sharpen.to(device)
            self._initialized_device = device
        
    def _generate_kernel(self, device=None) -> torch.Tensor:
        """生成第一次退化核，确保在正确的设备上"""
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
        if device is not None:
            kernel = kernel.to(device)
        return kernel
    
    def _generate_kernel2(self, device=None) -> torch.Tensor:
        """生成第二次退化核，确保在正确的设备上"""
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
        if device is not None:
            kernel = kernel.to(device)
        return kernel
    
    def _generate_sinc_kernel(self, kernel_size, device=None):
        """生成sinc滤波器，确保在正确的设备上"""
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 3, np.pi / 2)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=0)
        # 确保kernel维度正确 [3, 1, kernel_size, kernel_size]
        kernel = torch.from_numpy(kernel).float()
        kernel = kernel.unsqueeze(0).repeat(3, 1, 1, 1)
        if device is not None:
            kernel = kernel.to(device)
        return kernel
    
    def forward(self, img):
        """前向传播 - 应用完整退化过程
        
        Args:
            img: 输入图像 [B, C, H, W]
            
        Returns:
            退化后的图像 [B, C, H, W]
        """
        # 确保所有组件都在相同的设备上
        device = img.device
        self._ensure_on_device(device)
        
        # 保存原始尺寸
        ori_h, ori_w = img.shape[2:]
        
        # 1. 第一次退化
        # 生成第一次退化的kernel
        kernel = self._generate_kernel(device)
            
        # 应用第一次退化
        img = F.conv2d(img, kernel, padding=self.blur_kernel_size//2, groups=3)
            
        # 应用第一次噪声
        if torch.rand(1, device=device) < self.gaussian_noise_prob:
            noise_level = torch.rand(1, device=device) * (self.noise_range[1] - self.noise_range[0]) + self.noise_range[0]
            gray_noise = torch.rand(1, device=device) < self.gray_noise_prob
            img = random_add_gaussian_noise_pt(
                img, sigma_range=(noise_level.item(), noise_level.item()), clip=True, 
                rounds=False, gray_prob=gray_noise.item()
            )
            
        # 应用第一次JPEG压缩
        if torch.rand(1, device=device) < self.sinc_prob:
            quality = torch.rand(1, device=device) * (self.jpeg_range[1] - self.jpeg_range[0]) + self.jpeg_range[0]
            img = torch.clamp(img, 0, 1)
            img = self.jpeger(img, quality=quality.item())
            
        # 应用第一次调整大小
        resize_prob = torch.rand(1, device=device)
        if resize_prob < self.resize_prob[0]:  # 放大
            scale = torch.rand(1, device=device) * (self.resize_range[1] - self.resize_range[0]) + self.resize_range[0]
            img = self._resize_on_gpu(img, scale)
        elif resize_prob < self.resize_prob[0] + self.resize_prob[1]:  # 缩小
            scale = 1 / (torch.rand(1, device=device) * (self.resize_range[1] - self.resize_range[0]) + self.resize_range[0])
            img = self._resize_on_gpu(img, scale)
        
        # 2. 第二次退化
        # 应用第二次模糊
        if torch.rand(1, device=device) < self.second_blur_prob:
            kernel2 = self._generate_kernel2(device)
            img = F.conv2d(img, kernel2, padding=self.blur_kernel_size2//2, groups=3)
            
        # 应用第二次噪声
        if torch.rand(1, device=device) < self.gaussian_noise_prob2:
            noise_level = torch.rand(1, device=device) * (self.noise_range2[1] - self.noise_range2[0]) + self.noise_range2[0]
            gray_noise = torch.rand(1, device=device) < self.gray_noise_prob2
            img = random_add_gaussian_noise_pt(
                img, sigma_range=(noise_level.item(), noise_level.item()), clip=True, 
                rounds=False, gray_prob=gray_noise.item()
            )
            
        # 应用第二次JPEG压缩
        if torch.rand(1, device=device) < self.sinc_prob2:
            quality = torch.rand(1, device=device) * (self.jpeg_range2[1] - self.jpeg_range2[0]) + self.jpeg_range2[0]
            img = torch.clamp(img, 0, 1)
            img = self.jpeger(img, quality=quality.item())
            
        # 应用第二次调整大小
        resize_prob = torch.rand(1, device=device)
        if resize_prob < self.resize_prob2[0]:  # 放大
            scale = torch.rand(1, device=device) * (self.resize_range2[1] - self.resize_range2[0]) + self.resize_range2[0]
            img = self._resize_on_gpu(img, scale)
        elif resize_prob < self.resize_prob2[0] + self.resize_prob2[1]:  # 缩小
            scale = 1 / (torch.rand(1, device=device) * (self.resize_range2[1] - self.resize_range2[0]) + self.resize_range2[0])
            img = self._resize_on_gpu(img, scale)
        
        # 3. 最终sinc滤波器
        if torch.rand(1, device=device) < self.final_sinc_prob:
            sinc_kernel = self._generate_sinc_kernel(self.blur_kernel_size, device)
            img = F.conv2d(img, sinc_kernel, padding=self.blur_kernel_size//2, groups=3)
        
        # 4. 确保输出尺寸与输入一致
        if img.shape[2:] != (ori_h, ori_w):
            img = F.interpolate(
                img,
                size=(ori_h, ori_w),
                mode='bilinear',
                align_corners=False
            )
        
        return img
    
    def _resize_on_gpu(self, img, scale):
        """在GPU上调整图像大小"""
        img_shape = img.shape
        h, w = img_shape[2:]
        new_h, new_w = int(h * scale), int(w * scale)
        return F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False) 