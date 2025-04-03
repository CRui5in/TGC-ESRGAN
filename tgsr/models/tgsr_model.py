import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPTokenizer, CLIPTextModel
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple
from contextlib import contextmanager
import os
import os.path as osp
from basicsr.utils import get_root_logger, USMSharp
from basicsr.utils.img_util import imwrite
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

from tgsr.utils.visualization_utils import improved_tensor2img as tensor2img
from tgsr.utils.visualization_utils import rgb_imwrite

# 从tgsr_arch导入需要的类
from tgsr.archs.tgsr_arch import (
    PositionalEncoding2D,
    BidirectionalCrossAttention,
    TextConditionedNorm,
    RegionAwareTextFusion
)

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

from tgsr.models.degradation_layers import DegradationModule

# 实现UNet判别器
class UNetDiscriminatorWithSpectralNorm(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64):
        super(UNetDiscriminatorWithSpectralNorm, self).__init__()
        
        # 初始卷积层
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)
        self.conv_first = nn.utils.spectral_norm(self.conv_first)
        
        # 下采样路径
        self.down1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 2, kernel_size=4, stride=2, padding=1),
            nn.utils.spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat * 4, kernel_size=4, stride=2, padding=1),
            nn.utils.spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(num_feat * 4, num_feat * 8, kernel_size=4, stride=2, padding=1),
            nn.utils.spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 8, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        
        self.down4 = nn.Sequential(
            nn.Conv2d(num_feat * 8, num_feat * 8, kernel_size=4, stride=2, padding=1),
            nn.utils.spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 8, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(num_feat * 8, num_feat * 8, kernel_size=4, stride=2, padding=1),
            nn.utils.spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 8, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(num_feat * 8, num_feat * 4, kernel_size=4, stride=2, padding=1),
            nn.utils.spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(num_feat * 4, num_feat * 2, kernel_size=4, stride=2, padding=1),
            nn.utils.spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(num_feat * 2, num_feat, kernel_size=4, stride=2, padding=1),
            nn.utils.spectral_norm(nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        
        # 最后的卷积层
        self.conv_last = nn.Conv2d(num_feat, 1, kernel_size=3, padding=1)
        self.conv_last = nn.utils.spectral_norm(self.conv_last)
        
    def forward(self, x):
        # 下采样路径
        feat = self.conv_first(x)
        down1 = self.down1(feat)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        
        # 上采样路径
        up1 = self.up1(down4)
        up2 = self.up2(up1 + down3)
        up3 = self.up3(up2 + down2)
        up4 = self.up4(up3 + down1)
        
        # 最后的卷积层
        out = self.conv_last(up4 + feat)
        
        return out

# 实现GAN损失函数
class GANLoss(nn.Module):
    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight
        
        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f'GAN类型 {gan_type} 未实现')
    
    def forward(self, pred, target_is_real, is_disc=False):
        """
        Args:
            pred (Tensor): 判别器的预测结果
            target_is_real (bool): 目标是否为真实图像
            is_disc (bool): 是否在判别器训练阶段
        """
        if self.gan_type == 'vanilla':
            if target_is_real:
                target = torch.ones_like(pred) * self.real_label_val
            else:
                target = torch.zeros_like(pred) * self.fake_label_val
            loss = self.loss(pred, target)
        elif self.gan_type == 'lsgan':
            if target_is_real:
                target = torch.ones_like(pred) * self.real_label_val
            else:
                target = torch.zeros_like(pred) * self.fake_label_val
            loss = self.loss(pred, target)
        
        return loss * self.loss_weight

# CLIP文本编码器
class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        super(TextEncoder, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
    def forward(self, **inputs):
        outputs = self.text_encoder(**inputs)
        return outputs
    
    def encode_text(self, text_prompts, device='cuda'):
        # 将文本转换为token
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        # 确保设备参数是字符串
        if not isinstance(device, str):
            device = str(device)
        
        inputs = self.tokenizer(
            text_prompts, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            text_features = self.forward(**inputs)
            
        return text_features.last_hidden_state, text_features.pooler_output

# 文本引导超分辨率模型
@MODEL_REGISTRY.register()
class TGSRModel(SRModel):
    """文本引导超分辨率模型类
    
    基于BasicSR框架，扩展了SRModel，添加了文本条件处理能力
    """
    def __init__(self, opt):
        super(TGSRModel, self).__init__(opt)
        
        # 添加退化模块
        self.degradation = DegradationModule(opt['datasets']['train'])
        
        # 添加USM锐化工具
        self.usm_sharpener = USMSharp().cuda()
        
        # 加载配置
        self.text_encoder_name = self.opt.get('text_encoder', {}).get('name', "/root/autodl-tmp/clip-vit-base-patch16")
        self.text_dim = self.opt.get('text_encoder', {}).get('text_dim', 512)
        self.use_text_features = self.opt.get('network_g', {}).get('use_text_features', True)
        self.save_text_encoder = self.opt.get('text_encoder', {}).get('save_with_model', False)
        
        # 记录当前的步骤，在训练过程中会更新
        self.curr_iter = 0
        
        # 初始化文本编码器
        if self.use_text_features:
            self.text_encoder = TextEncoder(self.text_encoder_name)
            self.text_encoder = self.text_encoder.to(self.device)
            
            if opt.get('text_encoder', {}).get('freeze', True):
                self.freeze_text_encoder()
        
        # 注册额外的ema模型，如果需要
        if self.is_train and self.opt.get('use_ema', False):
            self.net_g_ema = self.model_to_device(self.net_g)
            # 深度复制net_g参数到net_g_ema
            self.model_ema(0)
            self.net_g_ema.eval()
            
        # 初始化GAN相关组件
        if self.is_train:
            # 初始化判别器
            if 'network_d' in self.opt:
                self.net_d = self.model_to_device(self.get_network_d())
                self.net_d.train()
            
            # 初始化GAN损失函数
            if 'gan_opt' in self.opt['train']:
                self.cri_gan = GANLoss(
                    gan_type=self.opt['train']['gan_opt'].get('gan_type', 'vanilla'),
                    real_label_val=self.opt['train']['gan_opt'].get('real_label_val', 1.0),
                    fake_label_val=self.opt['train']['gan_opt'].get('fake_label_val', 0.0),
                    loss_weight=self.opt['train']['gan_opt'].get('loss_weight', 1.0)
                )
            
            # 初始化判别器优化器
            if hasattr(self, 'net_d') and self.net_d is not None:
                self.optimizer_d = torch.optim.Adam(
                    self.net_d.parameters(),
                    lr=self.opt['train']['optim_d']['lr'],
                    betas=self.opt['train']['optim_d'].get('betas', [0.9, 0.99]),
                    weight_decay=self.opt['train']['optim_d'].get('weight_decay', 0)
                )
    
    def freeze_text_encoder(self):
        """冻结文本编码器参数"""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_text_encoder(self):
        """解冻文本编码器参数"""
        for param in self.text_encoder.parameters():
            param.requires_grad = True
    
    def extract_text_features(self, text_prompts):
        """提取文本特征"""
        if self.use_text_features and text_prompts is not None:
            with torch.no_grad():
                text_hidden_states, text_pooled = self.text_encoder.encode_text(text_prompts, self.device)
            return text_hidden_states, text_pooled
        return None, None
    
    def forward_with_attention(self, x, text_features=None, return_attention=False):
        """带注意力的前向传播，用于收集注意力图
        
        Args:
            x: 输入图像
            text_features: 文本特征
            return_attention: 是否返回注意力图
            
        Returns:
            输出图像和注意力图
        """
        # 存储注意力图的字典
        attention_maps = {}
        
        # 获取模型各层，用于提取中间激活
        if hasattr(self.net_g, 'body'):
            body = self.net_g.body
            # 如果body是RRDB模块的序列
            if hasattr(body, 'blocks'):
                # 获取中间层特征
                features = []
                x_input = x
                
                # 头部处理
                if hasattr(self.net_g, 'conv_first'):
                    x = self.net_g.conv_first(x)
                
                # 主体模块处理
                for idx, block in enumerate(body.blocks):
                    x = block(x)
                    # 保存一些关键层的特征
                    if idx % 5 == 0 or idx == len(body.blocks) - 1:
                        features.append(x)
                
                # 如果有文本特征注入，记录注意力图
                if text_features is not None and hasattr(body, 'text_attention'):
                    for idx, attn in enumerate(body.text_attention):
                        if hasattr(attn, 'attention_weights'):
                            attention_maps[f'block_{idx}'] = attn.attention_weights
                
                # 如果有上采样模块
                if hasattr(self.net_g, 'upsampler'):
                    x = self.net_g.upsampler(x)
                
                # 最后的卷积层
                if hasattr(self.net_g, 'conv_last'):
                    x = self.net_g.conv_last(x)
        
        # 保存注意力图
        if return_attention:
            self.net_g.attention_maps = attention_maps
            
        return x

    def feed_data(self, data):
        """接收并处理输入数据
        
        Args:
            data: 包含lq (低分辨率), gt (高分辨率) 和 text_prompt (文本描述) 的字典
        """
        if self.is_train:
            # 训练数据合成
            self.gt = data['gt'].to(self.device)
            # 保存原始图像用于可视化
            self.original_gt = data['gt'].clone().detach()
            # 对GT进行锐化
            self.gt_usm = self.usm_sharpener(self.gt)
            
            # 应用退化操作
            self.lq = self.degradation(self.gt_usm)
            
            # 抽取文本特征
            if self.use_text_features:
                text_prompts = data.get('text_prompt')
                if text_prompts is not None:
                    # 使用CLIP编码文本
                    with torch.no_grad():
                        text_hidden, text_pooled = self.text_encoder.encode_text(text_prompts, device=self.device)
                        
                    # 确保维度匹配 - 如果text_dim不匹配，调整维度
                    if self.opt.get('network_g', {}).get('text_dim') != text_pooled.shape[-1]:
                        text_dim = self.opt.get('network_g', {}).get('text_dim', 512)
                        if text_pooled.shape[-1] > text_dim:
                            # 降维情况
                            if not hasattr(self, 'text_projector'):
                                self.text_projector = nn.Linear(text_pooled.shape[-1], text_dim).to(self.device)
                            text_pooled = self.text_projector(text_pooled)
                            text_hidden = self.text_projector(text_hidden)
                        else:
                            # 升维情况
                            text_pooled = F.pad(text_pooled, (0, text_dim - text_pooled.shape[-1]))
                            text_hidden = F.pad(text_hidden, (0, text_dim - text_hidden.shape[-1]))
                    
                    # 保存文本特征以供模型使用
                    self.text_hidden = text_hidden
                    self.text_pooled = text_pooled
            else:
                # 如果不使用文本特征或无文本提示，使用零向量
                batch_size = self.lq.shape[0]
                text_dim = self.opt.get('network_g', {}).get('text_dim', 512)
                self.text_hidden = torch.zeros((batch_size, 77, text_dim), device=self.device)  # 默认CLIP序列长度
                self.text_pooled = torch.zeros((batch_size, text_dim), device=self.device)
        else:
            # 验证或测试阶段
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                # 保存原始图像用于可视化
                self.original_gt = data['gt'].clone().detach()
                # 对GT进行锐化
                self.gt_usm = self.usm_sharpener(self.gt)
            
            # 抽取文本特征
            if self.use_text_features:
                text_prompts = data.get('text_prompt')
                if text_prompts is not None:
                    with torch.no_grad():
                        text_hidden, text_pooled = self.text_encoder.encode_text(text_prompts, device=self.device)
                    self.text_hidden = text_hidden
                    self.text_pooled = text_pooled
            else:
                batch_size = self.lq.shape[0]
                text_dim = self.opt.get('network_g', {}).get('text_dim', 512)
                self.text_hidden = torch.zeros((batch_size, 77, text_dim), device=self.device)
                self.text_pooled = torch.zeros((batch_size, text_dim), device=self.device)
    
    def optimize_parameters(self, current_iter):
        """优化模型参数，执行反向传播和参数更新
        
        Args:
            current_iter: 当前迭代次数
        """
        # 清除之前的log_dict
        self.log_dict = OrderedDict()
        
        # 清空优化器梯度
        self.optimizer_g.zero_grad()
        
        # 前向传播
        if self.use_text_features and hasattr(self, 'text_pooled'):
            self.output = self.net_g(self.lq, self.text_pooled)
        else:
            self.output = self.net_g(self.lq)
        
        # 确保输出和GT的尺寸一致
        if self.output.shape != self.gt.shape:
            self.output = F.interpolate(
                self.output,
                size=(self.gt.shape[2], self.gt.shape[3]),
                mode='bilinear',
                align_corners=False
            )
        
        # 计算像素损失(L1损失)
        if hasattr(self, 'cri_pix'):
            l_pix = self.cri_pix(self.output, self.gt_usm if self.is_train and self.opt['train'].get('use_sharp_gt', False) else self.gt)
            self.log_dict['l_pix'] = l_pix
        else:
            l_pix = torch.tensor(0.0, device=self.device)
            self.log_dict['l_pix'] = l_pix
            
        # 计算感知损失(VGG特征损失)
        l_percep = torch.tensor(0.0, device=self.device)
        l_style = torch.tensor(0.0, device=self.device)
        
        if hasattr(self, 'cri_perceptual'):
            perceptual_results = self.cri_perceptual(self.output, self.gt_usm if self.is_train and self.opt['train'].get('use_sharp_gt', False) else self.gt)
            if isinstance(perceptual_results, tuple) and len(perceptual_results) == 2:
                l_percep_tmp, l_style_tmp = perceptual_results
                if l_percep_tmp is not None:
                    l_percep = l_percep_tmp
                if l_style_tmp is not None:
                    l_style = l_style_tmp
        
        self.log_dict['l_percep'] = l_percep
        self.log_dict['l_style'] = l_style
        
        # 计算总损失
        l_g_total = l_pix + l_percep + l_style
            
        # 检查是否到达开始GAN训练的迭代次数
        gan_start_iter = self.opt.get('train', {}).get('gan_start_iter', 10000)
        
        # 默认GAN损失为0
        l_g_gan = torch.tensor(0.0, device=self.device)
        
        # 如果到了开始GAN训练的迭代次数或有GAN损失记录
        if current_iter > gan_start_iter or (hasattr(self, 'log_dict') and 'l_g_gan' in self.log_dict):
            # 确保每次都更新判别器
            if not hasattr(self, 'net_d') or self.net_d is None:
                print('判别器未初始化，跳过GAN训练')
                return
            if not hasattr(self, 'cri_gan') or self.cri_gan is None:
                print('GAN损失函数未初始化，跳过GAN训练')
                return
            
            # 生成器对抗损失
            if hasattr(self, 'net_d') and self.net_d is not None:
                fake_g_pred = self.net_d(self.output)
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                
                # 添加GAN训练开始的日志
                if current_iter == gan_start_iter + 1 or (hasattr(self, 'log_dict') and 'l_g_gan' in self.log_dict):
                    print(f'开始GAN训练 (当前迭代次数: {current_iter})')
                    self.log_dict['gan_start'] = 1.0  # 添加一个标记到日志中
                else:
                    self.log_dict['gan_start'] = 0.0
        
        self.log_dict['l_g_gan'] = l_g_gan
        self.log_dict['l_total'] = l_g_total
        
        # 执行反向传播
        l_g_total.backward()
        self.optimizer_g.step()
        
        # 判别器默认损失为0
        l_d_real = torch.tensor(0.0, device=self.device)
        l_d_fake = torch.tensor(0.0, device=self.device)
        
        # 判别器优化 - 仅在开始GAN训练后执行
        if current_iter > gan_start_iter or (hasattr(self, 'log_dict') and 'l_g_gan' in self.log_dict):
            # 确保每次都更新判别器
            for p in self.net_d.parameters():
                p.requires_grad = True
                
            # 清空判别器梯度
            self.optimizer_d.zero_grad()
            
            # 真实图像的判别器损失
            real_d_pred = self.net_d(self.gt_usm if self.is_train and self.opt['train'].get('use_sharp_gt', False) else self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            
            # 生成图像的判别器损失
            fake_d_pred = self.net_d(self.output.detach())  # 分离计算图
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            
            # 判别器总损失
            l_d_total = (l_d_real + l_d_fake) * 0.5
            l_d_total.backward()
            self.optimizer_d.step()
        
        self.log_dict['l_d_real'] = l_d_real
        self.log_dict['l_d_fake'] = l_d_fake
    
    def test(self):
        """测试过程，执行前向传播并保存注意力图"""
        self.net_g.eval()
        with torch.no_grad():
            if self.use_text_features:
                # 使用保存的文本特征，需要在feed_data中已经生成
                self.output = self.forward_with_attention(self.lq, self.text_pooled, return_attention=True)
            else:
                self.output = self.forward_with_attention(self.lq, return_attention=True)
        self.net_g.train()
    
    def get_visualization(self):
        """获取可视化结果"""
        out_dict = OrderedDict()
        # 原始低分辨率图像
        out_dict['LQ'] = self.lq.detach().cpu()
        # 生成的超分辨率图像
        out_dict['SR'] = self.output.detach().cpu()
        # 真实高分辨率图像（如果有）
        if hasattr(self, 'gt'):
            out_dict['GT'] = self.gt.detach().cpu()
        return out_dict
    
    def get_current_visuals(self):
        """获取当前可视化结果，供BasicSR框架使用"""
        return self.get_visualization()
    
    def get_grad_cam(self, target_layer=None):
        """获取GradCAM可视化热力图，使用注意力图代替传统GradCAM
        
        Args:
            target_layer: 要可视化的目标层，默认为None表示使用注意力图
            
        Returns:
            dict: 包含不同层的热力图
        """
        # 首先尝试获取模型中保存的注意力图
        if hasattr(self.net_g, 'attention_maps') and self.net_g.attention_maps:
            # 将注意力图转换为热力图格式
            cam_maps = {}
            for name, attn_map in self.net_g.attention_maps.items():
                # 对于多头注意力，取平均
                if len(attn_map.shape) > 2:
                    # [batch, heads, seq_len, seq_len] -> [batch, seq_len, seq_len]
                    attn_map = attn_map.mean(dim=1)
                
                # 取第一个样本的注意力图
                cam = attn_map[0].mean(dim=0)  # 平均所有注意力头
                
                # 重塑为2D图像格式
                if len(cam.shape) == 1:
                    h = w = int(math.sqrt(cam.shape[0] + 0.5))
                    cam = cam[:h*w].reshape(h, w)
                
                # 归一化
                if torch.max(cam) > 0:
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                
                # 调整大小以匹配输出
                cam = F.interpolate(
                    cam.unsqueeze(0).unsqueeze(0), 
                    size=(self.output.shape[2], self.output.shape[3]),
                    mode='bilinear', 
                    align_corners=False
                )
                
                # 转换为numpy
                cam = cam[0, 0].cpu().numpy()
                cam_maps[f"attention_{name}"] = cam
            
            # 如果找到了注意力图，直接返回
            if cam_maps:
                return cam_maps
        
        # 如果没有注意力图，使用传统GradCAM计算
        if not hasattr(self, 'net_g') or not hasattr(self, 'output'):
            return None
            
        # 如果没有指定目标层，自动选择合适的层
        if target_layer is None:
            # 查找合适的目标层
            target_layers = {}
            if hasattr(self.net_g, 'body'):
                if hasattr(self.net_g.body, 'blocks'):
                    for i, block in enumerate(self.net_g.body.blocks):
                        # 每5个块取一个，减少层数
                        if i % 5 == 0:
                            target_layers[f'body_block_{i}'] = block
            else:
                # 如果找不到合适的层，返回空
                return None
        else:
            # 使用指定的目标层
            if hasattr(self.net_g, target_layer):
                target_layers = {target_layer: getattr(self.net_g, target_layer)}
            else:
                return None
            
        if not target_layers:
            return None
            
        # 使用输出SR图像的梯度
        self.net_g.zero_grad()
        
        # 收集注册的钩子
        hooks = []
        activations = {}
        gradients = {}
        
        # 定义钩子函数
        def forward_hook(module, input, output):
            activations[module] = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            gradients[module] = grad_output[0].detach()
        
        # 注册钩子
        for name, layer in target_layers.items():
            if layer is not None:
                hooks.append(layer.register_forward_hook(forward_hook))
                hooks.append(layer.register_backward_hook(backward_hook))
        
        # 前向传播
        if hasattr(self, 'lq'):
            # 重新计算输出
            if self.use_text_features and hasattr(self, 'text_pooled'):
                sr = self.net_g(self.lq, self.text_pooled)
            else:
                sr = self.net_g(self.lq)
            
            # 计算输出的模值的均值作为伪标签
            pseudo_target = sr.abs().mean()
            pseudo_target.backward()
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 生成热力图
        cam_maps = {}
        for layer_name, layer in target_layers.items():
            if layer in activations and layer in gradients:
                # 获取激活和梯度
                activation = activations[layer]
                gradient = gradients[layer]
                
                # 梯度的全局平均池化
                weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
                
                # 加权的激活图
                cam = torch.sum(weights * activation, dim=1, keepdim=True)
                cam = F.relu(cam)  # ReLU应用于热力图
                
                # 归一化
                if torch.max(cam) > 0:
                    cam = cam / torch.max(cam)
                
                # 调整热力图大小以匹配输出
                cam = F.interpolate(
                    cam, 
                    size=(self.output.shape[2], self.output.shape[3]),
                    mode='bilinear', 
                    align_corners=False
                )
                
                # 转换为numpy数组
                cam = cam[0, 0].cpu().detach().numpy()
                cam_maps[f"gradcam_{layer_name}"] = cam
        
        return cam_maps
    
    def get_current_attention_maps(self):
        """获取当前的注意力图，用于可视化模型的关注区域
        
        Returns:
            dict: 包含不同注意力头的注意力图
        """
        if not hasattr(self, 'net_g') or not hasattr(self, 'output'):
            return None
            
        attention_maps = {}
        
        # 提取文本-图像交叉注意力
        try:
            # 检查网络是否有注意力模块
            if hasattr(self.net_g, 'attention_maps') and self.net_g.attention_maps:
                for name, attn_map in self.net_g.attention_maps.items():
                    # 对多头注意力图取平均
                    if len(attn_map.shape) > 3:  # [B, H, Q, K]
                        attn_map = attn_map.mean(dim=1)  # 平均所有注意力头
                    
                    # 获取最后一个token对所有像素的注意力
                    # 通常代表[CLS]或者整体文本的注意力
                    if attn_map.shape[1] > 1:  # 有多个查询token
                        text_to_image_attn = attn_map[0, -1]  # 最后一个token的注意力
                    else:
                        text_to_image_attn = attn_map[0, 0]  # 只有一个token时
                    
                    # 将注意力图重塑为2D
                    h = w = int(math.sqrt(text_to_image_attn.shape[0]))
                    attention_2d = text_to_image_attn.reshape(h, w)
                    
                    # 归一化
                    attention_2d = (attention_2d - attention_2d.min()) / (attention_2d.max() - attention_2d.min() + 1e-8)
                    
                    # 转换为numpy
                    attention_2d = attention_2d.cpu().detach().numpy()
                    
                    # 调整大小以匹配输出尺寸
                    attention_2d = cv2.resize(
                        attention_2d, 
                        (self.output.shape[3], self.output.shape[2]),
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    attention_maps[name] = attention_2d
        except Exception as e:
            print(f"获取注意力图失败: {e}")
        
        return attention_maps
    
    def get_target_layers(self):
        """获取适合GradCAM的目标层
        
        Returns:
            dict: 目标层的字典，键为层名，值为层对象
        """
        target_layers = {}
        
        # 检查模型结构，找到合适的层
        if hasattr(self.net_g, 'body'):
            if hasattr(self.net_g.body, 'blocks'):
                for i, block in enumerate(self.net_g.body.blocks):
                    # 每5个块取一个，减少层数
                    if i % 5 == 0 or i == len(self.net_g.body.blocks) - 1:
                        target_layers[f'body_block_{i}'] = block
        
        # 如果有上采样层，也添加到目标
        if hasattr(self.net_g, 'upsampler'):
            target_layers['upsampler'] = self.net_g.upsampler
            
        # 最后一个卷积层
        if hasattr(self.net_g, 'conv_last'):
            target_layers['conv_last'] = self.net_g.conv_last
            
        return target_layers
    
    def save(self, epoch, current_iter):
        """保存模型检查点"""
        # 保存网络G
        self.save_network(self.net_g, 'net_g', current_iter)
        
        # 保存EMA网络
        if self.opt.get('use_ema', False) and hasattr(self, 'net_g_ema'):
            self.save_network(self.net_g_ema, 'net_g_ema', current_iter)
            
        # 保存判别器网络
        if hasattr(self, 'net_d'):
            self.save_network(self.net_d, 'net_d', current_iter)
            
        # 保存文本编码器
        if self.use_text_features and self.save_text_encoder:
            self.save_network(self.text_encoder, 'text_encoder', current_iter)
            
        # 保存优化器
        self.save_training_state(epoch, current_iter)
    
    @contextmanager
    def nullcontext(self):
        """空上下文管理器，用于条件场景"""
        yield 

    def validation(self, dataloader, current_iter, tb_logger, save_img=False, is_test=False):
        """验证过程
        
        Args:
            dataloader: 数据加载器
            current_iter: 当前迭代次数
            tb_logger: TensorBoard日志记录器
            save_img: 是否保存图像
            is_test: 是否是测试过程
        """
        if is_test:
            self.net_g.eval()
            if hasattr(self, 'net_d') and self.net_d is not None:
                self.net_d.eval()
            max_samples = self.opt['test'].get('max_test_samples', 1)
            log_prefix = 'Test'
        else:
            self.net_g.eval()
            if hasattr(self, 'net_d') and self.net_d is not None:
                self.net_d.eval()
            max_samples = self.opt['val'].get('max_val_samples', 1)
            log_prefix = 'Validation'
        
        # 创建保存图像的目录
        if save_img:
            if is_test:
                save_dir = osp.join(self.opt['path']['experiments_root'], 'test_results')
            else:
                save_dir = osp.join(self.opt['path']['experiments_root'], 'validation_results')
            os.makedirs(save_dir, exist_ok=True)
        
        # 设置随机种子以确保可重复性
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 随机选择固定数量的样本进行验证
        total_samples = len(dataloader.dataset)
        num_samples = min(max_samples, total_samples)
        vis_indices = np.random.choice(total_samples, num_samples, replace=False)
        
        # 初始化指标结果
        self.metric_results = dict()
        for metric, opt_ in self.opt['val']['metrics'].items():
            self.metric_results[metric] = []
        
        # 验证过程
        for idx, val_data in enumerate(dataloader):
            if idx >= num_samples:
                break
                
            self.feed_data(val_data)
            self.test()
            
            # 计算指标
            for metric, opt_ in self.opt['val']['metrics'].items():
                # 确保输出和GT的尺寸一致
                if self.output.shape != self.gt.shape:
                    # 使用双线性插值调整输出尺寸以匹配GT
                    self.output = F.interpolate(
                        self.output,
                        size=(self.gt.shape[2], self.gt.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # 将张量转换为numpy数组，确保是RGB格式
                output_np = tensor2img(self.output, rgb2bgr=False, min_max=(0, 1))
                gt_np = tensor2img(self.gt, rgb2bgr=False, min_max=(0, 1))
                
                # 确保图像是3通道RGB格式
                if len(output_np.shape) == 2:
                    output_np = np.stack([output_np] * 3, axis=-1)
                elif output_np.shape[-1] == 1:
                    output_np = np.concatenate([output_np] * 3, axis=-1)
                
                if len(gt_np.shape) == 2:
                    gt_np = np.stack([gt_np] * 3, axis=-1)
                elif gt_np.shape[-1] == 1:
                    gt_np = np.concatenate([gt_np] * 3, axis=-1)
                
                if metric == 'psnr':
                    self.metric_results[metric].append(calculate_psnr(
                        output_np, gt_np, opt_['crop_border'],
                        test_y_channel=opt_['test_y_channel']))
                elif metric == 'ssim':
                    self.metric_results[metric].append(calculate_ssim(
                        output_np, gt_np, opt_['crop_border'],
                        test_y_channel=opt_['test_y_channel']))
            
            # 记录图像到TensorBoard
            if tb_logger is not None and idx < 3:  # 记录前3个样本
                # 使用规范的命名格式
                # 取第一个样本并确保是RGB格式 (C,H,W)
                lq_tensor = self.lq[0].detach().cpu().float().clamp_(0, 1)
                output_tensor = self.output[0].detach().cpu().float().clamp_(0, 1)
                gt_tensor = self.gt[0].detach().cpu().float().clamp_(0, 1)
                
                # 使用规范的命名 Test/Images/xxx 或 Validation/Images/xxx
                tb_logger.add_image(f'{log_prefix}/Images/LQ_{idx+1}', lq_tensor, current_iter)
                tb_logger.add_image(f'{log_prefix}/Images/SR_{idx+1}', output_tensor, current_iter)
                tb_logger.add_image(f'{log_prefix}/Images/GT_{idx+1}', gt_tensor, current_iter)
            
            # 生成热力图
            diff_map = np.abs(output_np.astype(np.float32) - gt_np.astype(np.float32))
            diff_map = diff_map.mean(axis=2)  # 转为单通道
            diff_map = diff_map / diff_map.max() * 255  # 归一化到0-255
            
            # 应用热力图颜色映射
            heatmap = cv2.applyColorMap(diff_map.astype(np.uint8), cv2.COLORMAP_JET)
            
            # 将热力图与SR图像混合
            overlay = cv2.addWeighted(output_np, 0.7, heatmap, 0.3, 0)
            
            # 转换为CHW格式并添加到TensorBoard
            overlay_chw = overlay.transpose(2, 0, 1)  # [C, H, W]
            heatmap_chw = heatmap.transpose(2, 0, 1)  # [C, H, W]
            tb_logger.add_image(f'{log_prefix}/Images/heatmap_{idx+1}', overlay_chw, current_iter)
            tb_logger.add_image(f'{log_prefix}/Images/diff_map_{idx+1}', heatmap_chw, current_iter)
            
            # 保存图像
            if save_img:
                img_name = osp.basename(val_data['lq_path'][0])
                save_img_path = osp.join(save_dir, f'{img_name}')
                output_img = tensor2img(self.output, rgb2bgr=False, min_max=(0, 1))  # 确保不转换为BGR
                rgb_imwrite(save_img_path, output_img)  # 使用rgb_imwrite确保保存为RGB格式
        
        # 计算平均指标
        for metric, opt_ in self.opt['val']['metrics'].items():
            self.metric_results[metric] = np.mean(self.metric_results[metric])
        
        # 记录到TensorBoard
        if tb_logger is not None:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'{log_prefix}/Metrics/{metric}', value, current_iter)
        
        # 恢复训练模式
        self.net_g.train()
        if hasattr(self, 'net_d') and self.net_d is not None:
            self.net_d.train()

    def _collect_attention_maps(self, vis_images, img_name, lq_img):
        """收集注意力图并转换为热力图
        
        Args:
            vis_images: 可视化图像字典
            img_name: 图像名称
            lq_img: 低分辨率输入图像
        """
        # 此方法不再需要
        pass

    def _attention_to_heatmap(self, attention_map):
        """将注意力图转换为热力图可视化
        
        Args:
            attention_map: 形状为 [B, 1, H, W] 或 [1, H, W] 的注意力图
            
        Returns:
            形状为 [H, W, 3] 的RGB热力图
        """
        # 此方法不再需要
        pass
        
    def _overlay_heatmap_on_image(self, heatmap, image, alpha=0.5):
        """将热力图叠加到原始图像上
        
        Args:
            heatmap: 热力图 [H, W, 3]
            image: 原始图像 [H, W, 3]
            alpha: 透明度
            
        Returns:
            叠加后的图像
        """
        # 此方法不再需要
        pass
    
    def _add_text_marker(self, img, text):
        """在图像上添加文本标记
        
        Args:
            img (numpy.ndarray): 输入图像，uint8类型，形状为(H, W, C)
            text (str): 要添加的文本
            
        Returns:
            numpy.ndarray: 添加了文本的图像
        """
        # 确保图像是uint8类型
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
            
        # 使用PIL添加文本，处理更加稳定
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # 创建合适的字体和大小
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except IOError:
            # 使用默认字体
            font = ImageFont.load_default()
        
        # 计算文本位置 - 左上角
        # 使用textbbox替代textsize (适用于Pillow >= 8.0.0)
        try:
            # 新版Pillow使用textbbox
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # 旧版Pillow使用textsize
            text_width, text_height = draw.textsize(text, font=font)
        
        position = (5, 5)
        
        # 绘制黑色背景提高可见性
        draw.rectangle(
            [position[0], position[1], position[0] + text_width + 10, position[1] + text_height + 10],
            fill=(0, 0, 0)
        )
        
        # 绘制白色文本
        draw.text((position[0] + 5, position[1] + 5), text, font=font, fill=(255, 255, 255))
        
        return np.array(pil_img)
        
    def _make_comparison_image(self, lq, output, gt):
        """创建用于比较的拼接图像
        
        Args:
            lq (Tensor): 低清输入图像
            output (Tensor): 模型输出的超分辨率图像
            gt (Tensor): 高清真值图像
            
        Returns:
            numpy.ndarray: 拼接后的比较图像，包含三张图像和标记
        """
        # 转换为numpy数组，指定不转换为BGR
        lq_img = tensor2img(lq, rgb2bgr=False, min_max=(0, 1))  # [0, 255], uint8
        sr_img = tensor2img(output, rgb2bgr=False, min_max=(0, 1))  # [0, 255], uint8
        gt_img = tensor2img(gt, rgb2bgr=False, min_max=(0, 1))  # [0, 255], uint8
        
        # 放大LQ图像以匹配SR尺寸而不是裁剪
        if lq_img.shape[0] != sr_img.shape[0] or lq_img.shape[1] != sr_img.shape[1]:
            lq_img = cv2.resize(lq_img, (sr_img.shape[1], sr_img.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        
        # 添加文本标记
        lq_img = self._add_text_marker(lq_img, 'LQ')
        sr_img = self._add_text_marker(sr_img, 'SR')
        sr_img = self._add_text_marker(sr_img, 'SR')
        
        # 边框宽度用于分隔图像
        border_width = 5
        white_border = np.ones((sr_img.shape[0], border_width, 3), dtype=np.uint8) * 255
        
        # 水平拼接图像，并添加白色边框进行分隔
        comparison = np.concatenate([lq_img, white_border, sr_img, white_border, gt_img], axis=1)
        
        return comparison
        
    def _make_full_comparison_image(self, lq, output, gt):
        """创建用于比较的完整图像拼接视图
        
        这个方法与_make_comparison_image类似，但会处理完整图像，
        并根据需要调整图像大小以适应合理的视图尺寸。
        
        Args:
            lq (Tensor): 低清输入图像
            output (Tensor): 模型输出的超分辨率图像
            gt (Tensor): 高清真值图像
            
        Returns:
            numpy.ndarray: 拼接后的比较图像，包含三张调整大小的完整图像和标记
        """
        # 转换为numpy数组，指定不转换为BGR
        lq_img = tensor2img(lq, rgb2bgr=False, min_max=(0, 1))  # [0, 255], uint8
        sr_img = tensor2img(output, rgb2bgr=False, min_max=(0, 1))  # [0, 255], uint8
        gt_img = tensor2img(gt, rgb2bgr=False, min_max=(0, 1))  # [0, 255], uint8
        
        # 对于大图像，将其调整到合理的大小以便在TensorBoard中显示
        max_height = 1024  # TensorBoard友好的最大高度
        
        # 计算调整后的尺寸，保持纵横比
        scale = min(max_height / sr_img.shape[0], 1.0)  # 如果图像已经足够小，则不缩小
        
        if scale < 1.0:
            # 调整所有图像大小
            new_h, new_w = int(sr_img.shape[0] * scale), int(sr_img.shape[1] * scale)
            
            sr_img = cv2.resize(sr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            gt_img = cv2.resize(gt_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 对LQ图像，考虑尺寸差异（可能是超分因子）
            lq_scale = scale * (sr_img.shape[1] / lq_img.shape[1])  # 调整缩放比例
            lq_new_h, lq_new_w = int(lq_img.shape[0] * lq_scale), int(lq_img.shape[1] * lq_scale)
            lq_img = cv2.resize(lq_img, (lq_new_w, lq_new_h), interpolation=cv2.INTER_AREA)
        else:
            # 放大LQ图像以匹配SR尺寸，便于比较
            lq_img = cv2.resize(lq_img, (lq_img.shape[1]*self.opt['scale'], lq_img.shape[0]*self.opt['scale']), 
                              interpolation=cv2.INTER_NEAREST)
        
        # 添加文本标记
        lq_img = self._add_text_marker(lq_img, 'LQ')
        sr_img = self._add_text_marker(sr_img, 'SR')
        gt_img = self._add_text_marker(gt_img, 'GT')
        
        # 制作垂直比较图像
        # 空白间隔
        border_height = 10
        white_border = np.ones((border_height, sr_img.shape[1], 3), dtype=np.uint8) * 255
        
        # 垂直拼接图像
        comparison = np.concatenate([
            lq_img, 
            white_border, 
            sr_img, 
            white_border, 
            gt_img
        ], axis=0)
        
        return comparison
        
    def get_current_log(self):
        """获取当前日志记录的损失值
        
        这个方法将确保所有类型的损失都被记录，即使它们的值为0
        这样可以保证TensorBoard图表连续显示
        
        Returns:
            OrderedDict: 损失日志字典
        """
        log_dict = OrderedDict()
        
        # 确保所有类型的损失都被记录
        all_loss_types = [
            'l_pix',      # 像素损失 
            'l_percep',   # 感知损失
            'l_style',    # 风格损失
            'l_g_gan',    # 生成器对抗损失
            'l_d_real',   # 判别器真实样本损失
            'l_d_fake',   # 判别器生成样本损失
            'l_total'     # 总损失
        ]
        
        # 获取当前已计算的损失
        current_loss = {k: v.item() for k, v in self.log_dict.items() if isinstance(v, torch.Tensor)}
        
        # 确保所有类型的损失都有记录，缺失的设为0
        for loss_type in all_loss_types:
            # 使用规范的命名方式
            log_dict[loss_type] = current_loss.get(loss_type, 0.0)
        
        # 其他可能的非损失日志项
        for k, v in self.log_dict.items():
            if k not in all_loss_types and isinstance(v, torch.Tensor):
                # 保持其他日志项的原始名称
                log_dict[k] = v.item()
        
        return log_dict

    def log_current(self, current_iter, tb_logger, wandb_logger=None):
        """记录当前训练状态"""
        # 记录损失
        for k, v in self.get_current_log().items():
            tb_logger.add_scalar(f'Train/{k}', v, current_iter)
            
        # 记录学习率
        for k, v in self.get_current_learning_rate().items():
            tb_logger.add_scalar(f'Train/learning_rate/{k}', v, current_iter)
            
        # 每隔一定迭代次数记录当前训练图像
        if current_iter % self.opt['logger']['save_training_image_freq'] == 0:
            try:
                # 获取当前训练批次的图像
                lq = self.lq[:1]  # 只取第一张图片
                gt = self.gt[:1]
                sr = self.output[:1]
                
                # 转换为numpy图像，保持RGB格式
                lq_img = tensor2img(lq, rgb2bgr=False, min_max=(0, 1))  # [H, W, C] RGB
                sr_img = tensor2img(sr, rgb2bgr=False, min_max=(0, 1))  # [H, W, C] RGB
                gt_img = tensor2img(gt, rgb2bgr=False, min_max=(0, 1))  # [H, W, C] RGB
                
                # 将LQ图像放大到与SR图像相同的尺寸
                if lq_img.shape[0] != sr_img.shape[0] or lq_img.shape[1] != sr_img.shape[1]:
                    lq_img = cv2.resize(lq_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                
                # 确保GT图像与SR尺寸相同
                if gt_img.shape[0] != sr_img.shape[0] or gt_img.shape[1] != sr_img.shape[1]:
                    gt_img = cv2.resize(gt_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                
                # 添加文本标记
                lq_img = self._add_text_marker(lq_img, 'LQ')
                sr_img = self._add_text_marker(sr_img, 'SR')
                gt_img = self._add_text_marker(gt_img, 'GT')
                
                # 横向拼接三个图像并添加水平空白间隔
                h, w = sr_img.shape[:2]
                spacer = np.ones((h, 10, 3), dtype=np.uint8) * 255  # 白色水平空白间隔
                comparison = np.concatenate(
                    [lq_img, spacer, sr_img, spacer, gt_img], 
                    axis=1  # 使用axis=1进行水平拼接
                )
                
                # 转换为CHW格式用于TensorBoard
                comparison_chw = comparison.transpose(2, 0, 1)  # [C, H, W]
                lq_chw = lq_img.transpose(2, 0, 1)  # [C, H, W]
                sr_chw = sr_img.transpose(2, 0, 1)  # [C, H, W]
                gt_chw = gt_img.transpose(2, 0, 1)  # [C, H, W]
                
                # 添加到TensorBoard，使用CHW格式
                tb_logger.add_image('Train/images/comparison', comparison_chw, current_iter)
                tb_logger.add_image('Train/images/LQ', lq_chw, current_iter)
                tb_logger.add_image('Train/images/SR', sr_chw, current_iter)
                tb_logger.add_image('Train/images/GT', gt_chw, current_iter)
                
                # 生成简单的热力图
                diff_map = np.abs(sr_img.astype(np.float32) - gt_img.astype(np.float32))
                diff_map = diff_map.mean(axis=2)  # 转为单通道
                diff_map = diff_map / diff_map.max() * 255  # 归一化到0-255
                
                # 应用热力图颜色映射
                heatmap = cv2.applyColorMap(diff_map.astype(np.uint8), cv2.COLORMAP_JET)
                
                # 将热力图与SR图像混合
                overlay = cv2.addWeighted(sr_img, 0.7, heatmap, 0.3, 0)
                
                # 转换为CHW格式并添加到TensorBoard
                overlay_chw = overlay.transpose(2, 0, 1)  # [C, H, W]
                heatmap_chw = heatmap.transpose(2, 0, 1)  # [C, H, W]
                tb_logger.add_image('Train/attention/diff_map', overlay_chw, current_iter)
                tb_logger.add_image('Train/heatmap/diff_map', heatmap_chw, current_iter)
                
            except Exception as e:
                self.logger.error(f"记录训练图像到TensorBoard时出错: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
    
    def get_current_learning_rate(self):
        """获取当前的学习率
        
        Returns:
            list: 所有优化器的学习率列表
                - lr_0: 生成器的学习率
                - lr_1: 判别器的学习率（如果存在）
        """
        if hasattr(self, 'optimizer_g'):
            lrs = [param_group['lr'] for param_group in self.optimizer_g.param_groups]
            
            # 如果有判别器，添加其学习率
            if hasattr(self, 'optimizer_d'):
                lrs.extend([param_group['lr'] for param_group in self.optimizer_d.param_groups])
                
            return {f'lr_{i}': lr for i, lr in enumerate(lrs)}
        return None 

    def get_network_d(self):
        """获取判别器网络"""
        net_d = UNetDiscriminatorWithSpectralNorm(
            num_in_ch=self.opt['network_d'].get('num_in_ch', 3),
            num_feat=self.opt['network_d'].get('num_feat', 64)
        )
        return net_d
    
    def get_criterion(self, opt):
        """获取损失函数"""
        return GANLoss(
            gan_type=opt.get('gan_type', 'vanilla'),
            real_label_val=opt.get('real_label_val', 1.0),
            fake_label_val=opt.get('fake_label_val', 0.0),
            loss_weight=opt.get('loss_weight', 1.0)
        )
    
    def get_optimizer(self, optim_type, params, **kwargs):
        """获取优化器
        
        Args:
            optim_type: 优化器类型
            params: 需要优化的参数
            **kwargs: 其他优化器参数
            
        Returns:
            torch.optim.Optimizer: 优化器实例
        """
        if optim_type == 'Adam':
            return torch.optim.Adam(params, **kwargs)
        elif optim_type == 'SGD':
            return torch.optim.SGD(params, **kwargs)
        else:
            raise ValueError(f'不支持的优化器类型: {optim_type}')
    
    def model_ema(self, current_iter=None):
        """更新指数移动平均（EMA）模型
        
        Args:
            current_iter: 当前迭代次数，用于调整衰减率
        """
        # 检查是否应该使用EMA
        if not hasattr(self, 'net_g_ema'):
            return
            
        # 根据当前迭代次数确定衰减率
        decay = 0.999  # 默认衰减率
        if current_iter is not None:
            # 调整衰减率，可以根据迭代次数动态调整
            ema_start = self.opt.get('train', {}).get('ema_start', 5000)
            if current_iter < ema_start and not hasattr(self, 'net_g_ema'):
                return
                
        # 对所有参数应用EMA更新
        if self.net_g_ema is not None:
            self.accumulate(self.net_g_ema, self.net_g, decay)
                    
    def accumulate(self, model_ema, model, decay=0.999):
        """累积参数的EMA
        
        Args:
            model_ema: EMA模型
            model: 当前模型
            decay: 衰减率
        """
        with torch.no_grad():
            for p_ema, p in zip(model_ema.parameters(), model.parameters()):
                p_ema.copy_(p_ema * decay + p * (1 - decay))
            for b_ema, b in zip(model_ema.buffers(), model.buffers()):
                b_ema.copy_(b)
    
    def get_scheduler(self, optimizer, scheduler_type, **kwargs):
        """获取学习率调度器
        
        Args:
            optimizer: 优化器实例
            scheduler_type: 调度器类型
            **kwargs: 其他调度器参数
            
        Returns:
            torch.optim.lr_scheduler._LRScheduler: 学习率调度器实例
        """
        if scheduler_type == 'CosineAnnealingRestartLR':
            from torch.optim.lr_scheduler import CosineAnnealingRestartLR
            return CosineAnnealingRestartLR(optimizer, **kwargs)
        elif scheduler_type == 'MultiStepLR':
            from torch.optim.lr_scheduler import MultiStepLR
            return MultiStepLR(optimizer, **kwargs)
        else:
            raise ValueError(f'不支持的调度器类型: {scheduler_type}') 

def dict2str(opt, indent_level=0):
    """将字典转换为格式化的字符串
    
    Args:
        opt: 要转换的字典
        indent_level: 缩进级别
        
    Returns:
        str: 格式化后的字符串
    """
    msg = []
    indent = '  ' * indent_level
    
    for k, v in opt.items():
        if isinstance(v, dict):
            msg.append(f'{indent}{k}:')
            msg.append(dict2str(v, indent_level + 1))
        else:
            msg.append(f'{indent}{k}: {v}')
            
    return '\n'.join(msg) 