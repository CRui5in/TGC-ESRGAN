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
from pathlib import Path

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
import torch.nn as nn
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D

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
    """将图像分割为部分重叠的patch"""
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
    """从重叠的patch重构图像"""
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

class RealESRGANTester:
    """RealESRGAN测试类，实现标准的RealESRGAN退化和超分流程"""
    def __init__(self, sr_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化退化工具
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        
        # 加载SR网络
        self.net_g = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32).to(self.device)
        sr_checkpoint = torch.load(sr_model_path, map_location=self.device)
        
        # 直接加载模型权重
        if 'params' in sr_checkpoint:
            self.net_g.load_state_dict(sr_checkpoint['params'], strict=True)
        elif 'params_ema' in sr_checkpoint:
            self.net_g.load_state_dict(sr_checkpoint['params_ema'], strict=True)
        
        self.net_g.eval()
        
        # 创建激活函数
        if not hasattr(self.net_g, 'lrelu'):
            self.net_g.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        print(f"模型已加载到{self.device}设备")
        
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
        }
        
        print("退化参数已设置")
    
    def forward_sr_network(self, x):
        """SR网络的前向传播，无文本引导"""
        # 浅层特征提取
        fea = self.net_g.conv_first(x)
        
        # 主干处理
        trunk = fea
        
        # 处理所有RRDB模块
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
        
        return out
    
    def apply_degradation(self, img):
        """应用标准RealESRGAN退化流程"""
        # 锐化处理
        gt_usm = self.usm_sharpener(img)
        
        # 第一阶段退化 - 模糊
        # 标准模糊 - 使用简单的高斯模糊
        kernel_size = 21
        sigma = 3.0
        # 创建高斯核
        x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.).to(self.device)
        x_grid, y_grid = torch.meshgrid(x, x)
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        out = filter2D(gt_usm, kernel)
        
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
        
        # 添加噪声
        gray_noise_prob = self.opt['gray_noise_prob']
        # 标准噪声
        out = random_add_gaussian_noise_pt(
            out, sigma_range=self.opt['noise_range'], clip=True, 
            rounds=False, gray_prob=gray_noise_prob)
        
        # JPEG压缩
        # 标准JPEG压缩
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # 裁剪到[0, 1]，否则JPEGer会产生不良伪影
        out = self.jpeger(out, quality=jpeg_p)
        
        # 第二阶段退化 - 模糊
        if np.random.uniform() < self.opt['second_blur_prob']:
            # 创建新的高斯核
            kernel_size = 21
            sigma = 1.5
            x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.).to(self.device)
            x_grid, y_grid = torch.meshgrid(x, x)
            kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            
            out = filter2D(out, kernel)
                
        # 随机调整大小
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        
        # 获取原始尺寸
        ori_h, ori_w = img.shape[2:4]
        out = F.interpolate(
            out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
        
        # 添加噪声
        gray_noise_prob = self.opt['gray_noise_prob2']
        # 标准噪声
        out = random_add_gaussian_noise_pt(
            out, sigma_range=self.opt['noise_range2'], clip=True, 
            rounds=False, gray_prob=gray_noise_prob)
        
        # JPEG压缩 + 最终sinc滤波器 - 随机顺序
        if np.random.uniform() < 0.5:
            # 调整回原尺寸
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            
            # JPEG压缩
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG压缩
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
                
            # 调整回原尺寸
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
        
        # clamp and round
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        
        return out

    def predict_realsr_style(self, img, batch_size=4, patch_size=192, padding=24, pad_size=15):
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
        
        # 批量处理patch
        with torch.no_grad():
            results = []
            
            for i in range(0, patches_tensor.size(0), batch_size):
                # 获取当前批次
                batch = patches_tensor[i:i+batch_size]
                # 应用SR网络
                res = self.forward_sr_network(batch)
                results.append(res)
            
            # 合并所有batch的结果
            sr_patches = torch.cat(results, dim=0)
        
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
    
    def process_image(self, img_path, output_dir=None):
        """处理单张图像，包括应用退化和生成SR结果"""
        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        
        # 调整图像大小到256x256（与训练一致）
        img = F.interpolate(img, size=(256, 256), mode='bicubic', align_corners=True)
        
        # 准备测试不同情况
        results = {}
        
        # 1. 原始高质量图像直接超分
        with torch.no_grad():
            self.output = self.forward_sr_network(img)
            results['sr_no_degradation'] = self.output.clone()
        
        # 2. 应用退化后再超分
        with torch.no_grad():
            # 应用退化
            degraded_img = self.apply_degradation(img)
            # 保存退化后的图像
            results['degraded'] = degraded_img.clone()
            # 对退化后的图像进行超分
            self.output = self.forward_sr_network(degraded_img)
            results['sr_with_degradation'] = self.output.clone()
        
        # 3. 使用RealESRGAN风格的分块处理直接超分
        with torch.no_grad():
            self.output = self.predict_realsr_style(img)
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
        
        return results

def main():
    """主函数：加载测试样本并应用不同处理"""
    # 模型路径
    sr_model_path = "/root/autodl-tmp/TGSR_copy/experiments/train_TGSRx4plus_400k_B12G4/models/net_g_10000.pth"
    
    # 输出目录
    output_dir = "/root/autodl-tmp/realesrgan_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取测试图像
    img_path = "/root/autodl-tmp/test_images/0001.png"  # 替换为实际测试图像路径
    
    print(f"处理图像: {img_path}")
    
    # 创建测试器并处理图像
    tester = RealESRGANTester(sr_model_path)
    results = tester.process_image(img_path, output_dir)
    
    print(f"结果已保存到: {output_dir}")
    print("测试完成，请比较以下结果:")
    print("1. sr_no_degradation.png - 无退化直接超分")
    print("2. sr_with_degradation.png - 退化后超分")
    print("3. sr_realsr_style.png - RealESRGAN风格分块处理")
    print("4. degraded.png - 退化后的LQ图像")
    print("5. comparison.png - 各种结果的并排比较")

if __name__ == "__main__":
    main() 