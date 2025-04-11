import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import math
import random
import json
import sys
from collections import OrderedDict
from transformers import CLIPTokenizer, CLIPTextModel
from pathlib import Path

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from basicsr.archs.rrdbnet_arch import RRDBNet
from tgsr.archs.tgsr_arch import TextGuidanceNet
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
import torch.nn as nn
from basicsr.utils import DiffJPEG, USMSharp

# RealESRGAN式图像处理函数
def pad_reflect(image, pad_size):
    """反射填充图像边缘"""
    imsize = image.shape
    height, width = imsize[:2]
    new_img = np.zeros([height+pad_size*2, width+pad_size*2, imsize[2]]).astype(np.uint8)
    new_img[pad_size:-pad_size, pad_size:-pad_size, :] = image
    new_img[0:pad_size, pad_size:-pad_size, :] = np.flip(image[0:pad_size, :, :], axis=0) # top
    new_img[-pad_size:, pad_size:-pad_size, :] = np.flip(image[-pad_size:, :, :], axis=0) # bottom
    new_img[:, 0:pad_size, :] = np.flip(new_img[:, pad_size:pad_size*2, :], axis=1) # left
    new_img[:, -pad_size:, :] = np.flip(new_img[:, -pad_size*2:-pad_size, :], axis=1) # right
    return new_img

def unpad_image(image, pad_size):
    """移除图像填充"""
    return image[pad_size:-pad_size, pad_size:-pad_size, :]

def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    """将图像分割为部分重叠的patch
    
    Args:
        image_array: 输入图像的numpy数组
        patch_size: 原始图像中patch的大小（不含填充）
        padding_size: 重叠区域的大小
    
    Returns:
        patches: 分割后的patch数组
        padded_image_shape: 填充后图像的形状
    """
    xmax, ymax, _ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size
    
    # 确保图像可以被patch_size整除
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size
    
    # 确保图像可以被规则patch整除
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')
    
    # 在图像周围添加填充简化计算
    padded_image = np.pad(
        extended_image,
        ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
        'edge',
    )
    
    xmax, ymax, _ = padded_image.shape
    patches = []
    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)
    
    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)
    
    return np.array(patches), padded_image.shape

def stich_together(patches, padded_image_shape, target_shape, padding_size=4):
    """从重叠的patch重构图像
    
    Args:
        patches: 通过split_image_into_overlapping_patches获得的patch
        padded_image_shape: split_image_into_overlapping_patches中构建的填充图像形状
        target_shape: 最终图像的形状
        padding_size: 重叠区域的大小
    
    Returns:
        完整的重构图像
    """
    xmax, ymax, _ = padded_image_shape
    
    # 移除patch中的填充部分
    patches_without_padding = patches[:, padding_size:-padding_size, padding_size:-padding_size, :]
    patch_size = patches_without_padding.shape[1]
    
    n_patches_per_row = ymax // patch_size
    complete_image = np.zeros((xmax, ymax, 3))
    
    row = -1
    col = 0
    for i in range(len(patches_without_padding)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
            
        complete_image[
            row * patch_size: (row + 1) * patch_size,
            col * patch_size: (col + 1) * patch_size,
            :
        ] = patches_without_padding[i]
        
        col += 1
    
    return complete_image[0: target_shape[0], 0: target_shape[1], :]

# 安全的filter2D函数，直接从tgsr_model.py复制
def safe_filter2D(img, kernel):
    """一个安全的filter2D版本，处理任意尺寸的内核"""
    # 确保kernel是2D形式 [k,k]
    if kernel.dim() > 2:
        # 如果是多维的，只保留最后两维
        kernel = kernel.reshape(-1, *kernel.shape[-2:])[-1]  # 取最后一个kernel
    
    # 确保kernel是2D
    assert kernel.dim() == 2, f"处理后的kernel应该是2D，但得到的是{kernel.dim()}D"
    
    # 获取尺寸
    k = kernel.size(0)  # 假设是正方形
    b, c, h, w = img.size()
    
    # 只支持奇数尺寸的核
    if k % 2 == 0:
        print(f"警告: 核大小{k}是偶数，裁剪为{k-1}")
        k = k - 1
        kernel = kernel[:k, :k]
    
    # 创建输出
    output = torch.zeros_like(img)
    
    # 逐样本、逐通道处理
    for batch_idx in range(b):
        sample = img[batch_idx:batch_idx+1]  # 保持4D: [1, c, h, w]
        
        for channel_idx in range(c):
            # 提取单通道图像
            channel_img = sample[:, channel_idx:channel_idx+1, :, :]  # [1, 1, h, w]
            
            # 准备核 - 改为标准的depthwise conv格式
            conv_kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
            
            # 填充
            padding = k // 2
            padded = F.pad(channel_img, (padding, padding, padding, padding), mode='reflect')
            
            # 应用卷积 - 使用 groups=1 确保通道数匹配
            filtered = F.conv2d(padded, conv_kernel)
            
            # 存储结果
            output[batch_idx:batch_idx+1, channel_idx:channel_idx+1, :, :] = filtered
    
    return output

class TGSRTester:
    """TGSR测试类，实现与训练时相同的退化流程"""
    def __init__(self, sr_model_path, text_guidance_path, text_encoder_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化退化工具
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        
        # 加载SR网络
        self.net_g = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32).to(self.device)
        sr_checkpoint = torch.load(sr_model_path, map_location=self.device)
        
        # 直接加载模型权重 - 不需要属性映射
        if 'params' in sr_checkpoint:
            self.net_g.load_state_dict(sr_checkpoint['params'], strict=True)
        elif 'params_ema' in sr_checkpoint:
            self.net_g.load_state_dict(sr_checkpoint['params_ema'], strict=True)
        
        self.net_g.eval()
        
        # 加载文本引导网络
        self.net_t = TextGuidanceNet(num_feat=64, text_dim=512, num_blocks=3, num_heads=8).to(self.device)
        text_checkpoint = torch.load(text_guidance_path, map_location=self.device)
        self.net_t.load_state_dict(text_checkpoint['params'], strict=True)
        self.net_t.eval()
        
        # 加载CLIP
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_path)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to(self.device)
        self.text_encoder.eval()
        
        # 文本特征设置
        self.use_text_features = True
        self.text_dim = 512
        self.freeze_text_encoder = True
        
        # 创建激活函数
        if not hasattr(self.net_g, 'lrelu'):
            self.net_g.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        print(f"所有模型已加载到{self.device}设备")
        
        # 退化参数设置 - 直接从tgsr_model.py中提取
        self.opt = {
            'resize_prob': [0.2, 0.7, 0.1],  # up, down, keep
            'resize_range': [0.15, 1.5],
            'gaussian_noise_prob': 0.5,
            'noise_range': [1, 30],
            'poisson_scale_range': [0.05, 3],
            'gray_noise_prob': 0.4,
            'jpeg_range': [30, 95],
            
            # 第二阶段退化
            'second_blur_prob': 0.8,
            'resize_prob2': [0.3, 0.4, 0.3],  # up, down, keep
            'resize_range2': [0.3, 1.2],
            'gaussian_noise_prob2': 0.5,
            'noise_range2': [1, 25],
            'poisson_scale_range2': [0.05, 2.5],
            'gray_noise_prob2': 0.4,
            'jpeg_range2': [30, 95],
            
            # 模型参数
            'scale': 4,
            
            # 文本区域特殊处理参数
            'text_blur': {'kernel_size': 11, 'surround_weight': 0.01, 'center_weight': 1.0},
            'text_noise': {'strength_factor': 0.5},
            'text_jpeg': {'quality_min_factor': 1.5, 'quality_max_factor': 1.2, 'quality_max': 90}
        }
        
        print("退化参数已设置")
    
    def apply_text_guidance(self, features, text_hidden=None, text_pooled=None, block_idx=None):
        """应用文本引导到特征图"""
        if not self.use_text_features:
            return features, None
        
        # 创建位置编码（如果需要）
        position_info = None
        if block_idx is not None and hasattr(self.net_t, 'with_position') and self.net_t.with_position:
            # 如果文本引导网络支持位置信息，可以创建位置编码
            num_blocks = len(self.net_g.body)
            position_info = torch.zeros(features.size(0), num_blocks, device=features.device)
            position_info[:, block_idx] = 1.0
        
        # 应用文本引导
        if position_info is not None:
            enhanced_features, attention_logits = self.net_t(features, text_hidden, text_pooled, position_info)
        else:
            enhanced_features, attention_logits = self.net_t(features, text_hidden, text_pooled)
        
        return enhanced_features, attention_logits
    
    def forward_sr_network(self, x, apply_guidance=True):
        """SR网络的前向传播，与修改后的tgsr_model.py保持一致"""
        # 编码文本（如果需要）
        if self.use_text_features and apply_guidance:
            with torch.no_grad():
                text_hidden, text_pooled = self.encode_text(self.text_prompts)
        else:
            text_hidden, text_pooled = None, None
        
        # 浅层特征提取
        fea = self.net_g.conv_first(x)
        
        # 主干处理
        trunk = fea
        attention_maps = []
        
        # 确定在哪些位置应用文本引导
        if self.use_text_features and apply_guidance and text_hidden is not None and text_pooled is not None:
            num_blocks = len(self.net_g.body)
            
            # 设置引导位置 - 与修改后的tgsr_model.py保持一致，使用5个引导点
            guidance_positions = [
                num_blocks // 8,             # 浅层 - 捕获边缘和纹理
                num_blocks // 4,             # 浅中层 - 捕获局部特征
                num_blocks // 2,             # 中层 - 捕获中等规模特征
                num_blocks * 3 // 4,         # 中深层 - 捕获对象部件
                num_blocks - 2               # 深层 - 捕获整体语义
            ]
            
            # 为不同层级设置引导强度
            guidance_weights = {
                num_blocks // 8: 0.6,        # 浅层权重
                num_blocks // 4: 0.8,        # 浅中层
                num_blocks // 2: 1.0,        # 中层使用最高权重
                num_blocks * 3 // 4: 0.8,    # 中深层
                num_blocks - 2: 0.6          # 深层使用较低权重
            }
            
            for i, block in enumerate(self.net_g.body):
                trunk = block(trunk)
                
                # 在关键位置应用文本引导
                if i in guidance_positions:
                    # 文本引导
                    enhanced_features, attn_maps = self.apply_text_guidance(trunk, text_hidden, text_pooled, block_idx=i)
                    
                    # 应用引导强度权重
                    weight = guidance_weights.get(i, 1.0)
                    # 加权融合：原始特征 + 权重 * 增强特征
                    trunk = trunk + weight * (enhanced_features - trunk)
                    
                    # 保存注意力图
                    if attn_maps is not None:
                        if isinstance(attn_maps, list):
                            processed_attn_maps = [torch.sigmoid(logits) for logits in attn_maps if logits is not None]
                            attention_maps.extend(processed_attn_maps)
                        else:
                            processed_attn = torch.sigmoid(attn_maps)
                            attention_maps.append(processed_attn)
        else:
            # 不使用文本引导
            for block in self.net_g.body:
                trunk = block(trunk)
        
        # 残差连接
        trunk = self.net_g.conv_body(trunk)
        fea = fea + trunk
        
        # 上采样
        fea = self.net_g.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.net_g.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest'))
        
        # 最终输出
        out = self.net_g.conv_hr(fea)
        out = self.net_g.conv_last(out)
        
        # 保存注意力图
        if len(attention_maps) > 0:
            self.attention_maps = attention_maps
        else:
            self.attention_maps = None
        
        return out
    
    def encode_text(self, text_prompts):
        """编码文本提示"""
        if not self.use_text_features:
            batch_size = 1
            text_hidden = torch.zeros(batch_size, 77, self.text_dim).to(self.device)
            text_pooled = torch.zeros(batch_size, self.text_dim).to(self.device)
            return text_hidden, text_pooled
        
        # 使用CLIP编码文本
        with torch.no_grad():
            text_inputs = self.tokenizer(
                text_prompts, 
                padding="max_length", 
                max_length=77, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            text_outputs = self.text_encoder(**text_inputs)
            text_hidden = text_outputs.last_hidden_state
            text_pooled = text_outputs.pooler_output
        
        return text_hidden, text_pooled
    
    def apply_degradation(self, img, text_mask=None):
        """应用与训练时相同的退化流程
        
        Args:
            img: [B, C, H, W] tensor，范围[0, 1]
            text_mask: [B, 1, H, W] tensor，0表示文本区域(轻度退化)，1表示其他区域(标准退化)
        
        Returns:
            degraded_img: 退化后的LQ图像
        """
        use_targeted_degradation = text_mask is not None
        
        # 锐化处理
        gt_usm = self.usm_sharpener(img)
        
        # 第一阶段退化 - 模糊
        if use_targeted_degradation:
            # 按照RealESRGAN流程进行模糊处理，对文本区域使用较轻的模糊
            batch_size = gt_usm.size(0)
            out = torch.zeros_like(gt_usm)
            
            # 针对每个样本单独处理
            for i in range(batch_size):
                sample_mask = text_mask[i:i+1]  # [1, 1, H, W]
                sample_gt = gt_usm[i:i+1]  # [1, C, H, W]
                
                # 创建两个掩码：文本区域和非文本区域
                text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                
                # 为文本区域创建轻度的模糊核
                text_kernel_size = self.opt['text_blur']['kernel_size']
                text_surround_weight = self.opt['text_blur']['surround_weight']
                text_center_weight = self.opt['text_blur']['center_weight']
                
                text_kernel = torch.ones((1, 1, text_kernel_size, text_kernel_size), 
                                      device=self.device) * text_surround_weight
                center = text_kernel_size // 2
                text_kernel[0, 0, center, center] = text_center_weight  # 中心点权重更高，减少模糊效果
                
                # 添加归一化处理，确保卷积核权重和为1.0，防止图像变白
                text_kernel = text_kernel / text_kernel.sum()
                
                # 文本区域应用轻度模糊
                text_blurred = safe_filter2D(sample_gt, text_kernel)
                
                # 其他区域应用标准模糊
                # 创建简单的高斯模糊核
                kernel_size = 21
                sigma = 3.0
                # 创建高斯核
                x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.).to(self.device)
                x_grid, y_grid = torch.meshgrid(x, x)
                kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
                kernel = kernel / kernel.sum()
                
                other_blurred = safe_filter2D(sample_gt, kernel)
                
                # 使用掩码加权混合
                out[i:i+1] = text_blurred * text_region_mask + other_blurred * other_region_mask
        else:
            # 标准模糊 - 使用简单的高斯模糊
            kernel_size = 21
            sigma = 3.0
            # 创建高斯核
            x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.).to(self.device)
            x_grid, y_grid = torch.meshgrid(x, x)
            kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            
            out = safe_filter2D(gt_usm, kernel)
        
        # 随机调整大小
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        
        # 如果有退化掩码，也需要调整其大小
        if use_targeted_degradation:
            text_mask = F.interpolate(text_mask, scale_factor=scale, mode='nearest')
        
        # 添加噪声
        gray_noise_prob = self.opt['gray_noise_prob']
        if use_targeted_degradation:
            # 对不同区域分别应用不同强度的噪声，然后混合
            batch_size = out.size(0)
            out_with_noise = torch.zeros_like(out)
            
            for i in range(batch_size):
                sample_mask = text_mask[i:i+1]  # [1, 1, H, W]
                sample_out = out[i:i+1]  # [1, C, H, W]
                
                # 创建掩码
                text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                
                # 文本区域：使用较轻的噪声参数
                noise_strength_factor = self.opt['text_noise']['strength_factor']
                text_noise_sigma = np.array(self.opt['noise_range']) * noise_strength_factor
                
                # 对两个区域分别应用高斯噪声
                text_region_noisy = random_add_gaussian_noise_pt(
                    sample_out, sigma_range=text_noise_sigma, clip=True, 
                    rounds=False, gray_prob=gray_noise_prob)
                
                other_region_noisy = random_add_gaussian_noise_pt(
                    sample_out, sigma_range=self.opt['noise_range'], clip=True, 
                    rounds=False, gray_prob=gray_noise_prob)
                
                # 合并结果
                out_with_noise[i:i+1] = text_region_noisy * text_region_mask + other_region_noisy * other_region_mask
                
            out = out_with_noise
        else:
            # 标准噪声
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range'], clip=True, 
                rounds=False, gray_prob=gray_noise_prob)
        
        # JPEG压缩
        if use_targeted_degradation:
            batch_size = out.size(0)
            out_compressed = torch.zeros_like(out)
            
            for i in range(batch_size):
                sample_mask = text_mask[i:i+1]
                sample_out = out[i:i+1]
                
                # 创建掩码
                text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                
                # 对两个区域分别完整应用不同质量的JPEG压缩
                quality_min_factor = self.opt['text_jpeg']['quality_min_factor']
                quality_max_factor = self.opt['text_jpeg']['quality_max_factor']
                quality_max = self.opt['text_jpeg']['quality_max']
                
                text_jpeg_quality = torch.FloatTensor(1).uniform_(
                    min(self.opt['jpeg_range'][0] * quality_min_factor, quality_max),  # 提高最低质量
                    min(100, self.opt['jpeg_range'][1] * quality_max_factor)  # 最高不超过100
                ).to(self.device)
                
                other_jpeg_quality = torch.FloatTensor(1).uniform_(
                    *self.opt['jpeg_range']
                ).to(self.device)
                
                # 进行压缩
                sample_out = torch.clamp(sample_out, 0, 1)
                
                # 对两个区域分别完整应用JPEG压缩
                text_compressed = self.jpeger(sample_out, quality=text_jpeg_quality) 
                other_compressed = self.jpeger(sample_out, quality=other_jpeg_quality)
                
                # 合并结果 - 使用掩码
                out_compressed[i:i+1] = text_compressed * text_region_mask + other_compressed * other_region_mask
            
            out = out_compressed
        else:
            # 标准JPEG压缩
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # 裁剪到[0, 1]，否则JPEGer会产生不良伪影
            out = self.jpeger(out, quality=jpeg_p)
        
        # 根据比例调整大小到目标尺寸 (1/scale)
        gt_size = img.shape[2]
        scale = self.opt['scale']
        lq_size = gt_size // scale
        out = F.interpolate(out, size=(lq_size, lq_size), mode='bicubic', align_corners=True)
        
        # 同样调整掩码大小
        if use_targeted_degradation:
            text_mask = F.interpolate(text_mask, size=(lq_size, lq_size), mode='nearest')
        
        return out

    def predict_realsr_style(self, img, apply_guidance=True, batch_size=4, patch_size=192, padding=24, pad_size=15):
        """使用RealESRGAN风格的分块处理进行预测"""
        device = self.device
        scale = self.opt['scale']
        
        # 首先，将张量转换为numpy数组
        img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # 应用边缘填充
        padded_img = pad_reflect(img_np, pad_size)
        
        # 将图像分成重叠的patch
        patches, p_shape = split_image_into_overlapping_patches(
            padded_img, patch_size=patch_size, padding_size=padding
        )
        
        # 将patch转换回张量
        patches_tensor = torch.FloatTensor(patches/255).permute((0, 3, 1, 2)).to(device)
        
        # 保存原始文本提示
        original_text_prompts = self.text_prompts.copy() if hasattr(self, 'text_prompts') else [""]
        
        # 批量处理patch - 但每次只处理一个patch！
        with torch.no_grad():
            results = []
            
            for i in range(0, patches_tensor.size(0)):
                # 处理单个patch
                current_patch = patches_tensor[i:i+1]  # 确保维度为 [1, C, H, W]
                
                # 每个patch使用相同的文本提示
                self.text_prompts = original_text_prompts
                
                # 对单个patch进行超分处理
                res = self.forward_sr_network(current_patch, apply_guidance=apply_guidance)
                results.append(res)
            
            # 合并所有patch的结果
            sr_patches = torch.cat(results, dim=0)
        
        # 恢复原始文本提示
        self.text_prompts = original_text_prompts
        
        # 将结果转换回numpy
        sr_patches_np = sr_patches.permute((0, 2, 3, 1)).clamp_(0, 1).cpu().numpy()
        
        # 计算填充尺寸和目标尺寸
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(padded_img.shape[0:2], scale)) + (3,)
        
        # 重构完整图像
        sr_img_np = stich_together(
            sr_patches_np, 
            padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape,
            padding_size=padding * scale
        )
        
        # 去除填充
        sr_img_np = unpad_image(sr_img_np, pad_size * scale)
        
        # 转换为[0, 255]的uint8
        sr_img_np = (sr_img_np * 255).astype(np.uint8)
        
        # 转回张量格式
        sr_img_tensor = torch.from_numpy(sr_img_np).float() / 255.0
        sr_img_tensor = sr_img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        return sr_img_tensor
    
    def process_image(self, img_path, text, objects_info, output_dir=None):
        """处理单张图像，包括应用退化和生成SR结果
        
        Args:
            img_path: 输入图像路径
            text: 文本描述
            objects_info: 对象信息列表
            output_dir: 输出目录
        
        Returns:
            results: 包含各种处理结果的字典
        """
        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        
        # 设置文本提示
        self.text_prompts = [text if text else ""]
        
        # 调整图像大小到256x256（与训练一致）
        img = F.interpolate(img, size=(256, 256), mode='bicubic', align_corners=True)
        
        # 创建文本区域掩码
        text_mask = torch.ones((1, 1, img.shape[2], img.shape[3]), dtype=torch.float32).to(self.device)
        
        # 如果有对象信息，创建文本区域掩码
        if objects_info:
            # 处理对象掩码
            h, w = img.shape[2], img.shape[3]
            for obj in objects_info:
                if 'mask_encoded' in obj and 'category' in obj:
                    # 只处理在文本中提到的类别
                    if obj['category'].lower() in text.lower():
                        try:
                            # 解码掩码
                            from pycocotools import mask as mask_util
                            if isinstance(obj['mask_encoded']['counts'], str):
                                obj['mask_encoded']['counts'] = obj['mask_encoded']['counts'].encode('utf-8')
                            
                            mask = mask_util.decode(obj['mask_encoded'])
                            # 调整掩码大小
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            # 将文本区域标记为0
                            text_mask[0, 0][mask > 0] = 0.0
                        except Exception as e:
                            print(f"掩码处理失败: {e}")
        
        # 准备测试不同情况
        results = {}
        
        # 1. 原始高质量图像直接超分（当前问题情况）
        with torch.no_grad():
            self.output = self.forward_sr_network(img, apply_guidance=True)
            results['sr_no_degradation'] = self.output.clone()
        
        # 2. 应用退化后再超分（验证退化是否解决问题）
        with torch.no_grad():
            # 应用退化
            degraded_img = self.apply_degradation(img, text_mask)
            # 保存退化后的图像
            results['degraded'] = degraded_img.clone()
            # 对退化后的图像进行超分
            self.output = self.forward_sr_network(degraded_img, apply_guidance=True)
            results['sr_with_degradation'] = self.output.clone()
            
            # 对比：没有文本引导的超分
            self.output = self.forward_sr_network(degraded_img, apply_guidance=False)
            results['sr_with_degradation_no_guidance'] = self.output.clone()
        
        # 3. 使用RealESRGAN风格的分块处理直接超分（无退化）
        with torch.no_grad():
            self.output = self.predict_realsr_style(img, apply_guidance=True)
            results['sr_realsr_style'] = self.output.clone()
        
        # 将结果转换为RGB图像并保存
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            imgname = os.path.splitext(os.path.basename(img_path))[0]
            
            # 保存原始图像
            original_img = img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            original_img = np.transpose(original_img[[2, 1, 0], :, :], (1, 2, 0))
            original_img = (original_img * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f'{imgname}_original.png'), original_img)
            
            # 保存退化后的图像
            degraded_img_np = results['degraded'].data.squeeze().float().cpu().clamp_(0, 1).numpy()
            degraded_img_np = np.transpose(degraded_img_np[[2, 1, 0], :, :], (1, 2, 0))
            degraded_img_np = (degraded_img_np * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f'{imgname}_degraded.png'), degraded_img_np)
            
            # 保存不同超分结果
            for result_name, tensor in results.items():
                if result_name.startswith('sr_'):
                    result_img = tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    result_img = np.transpose(result_img[[2, 1, 0], :, :], (1, 2, 0))
                    result_img = (result_img * 255.0).round().astype(np.uint8)
                    cv2.imwrite(os.path.join(output_dir, f'{imgname}_{result_name}.png'), result_img)
            
            # 保存文本区域掩码
            mask_np = text_mask.data.squeeze().float().cpu().numpy()
            mask_np = (mask_np * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f'{imgname}_text_mask.png'), mask_np)
            
            # 创建并排比较图 - 现在包含4列图像
            h, w = original_img.shape[:2]
            
            # 获取SR图像
            sr_no_degradation = cv2.resize(
                (results['sr_no_degradation'].data.squeeze().float().cpu().clamp_(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1],
                (w, h), interpolation=cv2.INTER_LANCZOS4
            )
            
            sr_with_degradation = cv2.resize(
                (results['sr_with_degradation'].data.squeeze().float().cpu().clamp_(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1],
                (w, h), interpolation=cv2.INTER_LANCZOS4
            )
            
            sr_realsr_style = cv2.resize(
                (results['sr_realsr_style'].data.squeeze().float().cpu().clamp_(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1],
                (w, h), interpolation=cv2.INTER_LANCZOS4
            )
            
            # 创建4列比较图
            comparison = np.zeros((h, w*4, 3), dtype=np.uint8)
            comparison[:, :w] = original_img  # 原始GT
            comparison[:, w:2*w] = sr_no_degradation  # 无退化直接超分
            comparison[:, 2*w:3*w] = sr_with_degradation  # 退化后超分
            comparison[:, 3*w:] = sr_realsr_style  # RealESRGAN风格处理
            
            # 添加分割线
            comparison[:, w-1:w+1] = [0, 0, 255]  # 红色分割线
            comparison[:, 2*w-1:2*w+1] = [0, 0, 255]  # 红色分割线
            comparison[:, 3*w-1:3*w+1] = [0, 0, 255]  # 红色分割线
            
            cv2.imwrite(os.path.join(output_dir, f'{imgname}_comparison.png'), comparison)
            
            # 保存文本
            if text:
                with open(os.path.join(output_dir, f'{imgname}_text.txt'), 'w') as f:
                    f.write(text)
        
        return results

def main():
    """主函数：加载测试样本并应用不同处理"""
    # 模型路径
    sr_model_path = "/root/autodl-tmp/TGSR_copy/experiments/train_TGSRx4plus_400k_B12G4/models/net_g_10000.pth"
    text_guidance_path = "/root/autodl-tmp/TGSR_copy/experiments/train_TGSRx4plus_400k_B12G4/models/net_t_10000.pth"
    text_encoder_path = "/root/autodl-tmp/clip-vit-base-patch32"
    
    # 输出目录
    output_dir = "/root/autodl-tmp/tgsr_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载元数据
    with open("/root/autodl-tmp/tgsr_dataset_hr_only/val_captions.json", 'r') as f:
        metadata = json.load(f)
    
    # 使用第一个样本进行测试
    sample = metadata[0]
    img_path = sample['hr_path']
    text = sample['caption']
    objects_info = sample['objects']
    
    print(f"处理图像: {img_path}")
    print(f"文本描述: {text}")
    print(f"识别到 {len(objects_info)} 个对象")
    
    # 创建测试器并处理图像
    tester = TGSRTester(sr_model_path, text_guidance_path, text_encoder_path)
    results = tester.process_image(img_path, text, objects_info, output_dir)
    
    print(f"结果已保存到: {output_dir}")
    print("测试完成，请比较以下结果:")
    print("1. sr_no_degradation.png - 无退化直接超分（问题情况）")
    print("2. sr_with_degradation.png - 退化后超分（期望解决）")
    print("3. sr_realsr_style.png - RealESRGAN风格分块处理（无退化）")
    print("4. degraded.png - 退化后的LQ图像")
    print("5. comparison.png - 各种结果的并排比较")

if __name__ == "__main__":
    main() 