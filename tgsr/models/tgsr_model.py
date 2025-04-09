import time
import numpy as np
import random
import torch
import torch.nn.functional as F
import cv2
import math
from collections import OrderedDict
from os import path as osp
import os

from basicsr import tensor2img, imwrite
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp, get_root_logger
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from transformers import CLIPTokenizer, CLIPTextModel
from basicsr.losses import build_loss
import torch.nn as nn

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

@MODEL_REGISTRY.register()
class TGSRModel(SRGANModel):
    """文本引导的超分辨率模型

    基于SRGANModel，但增加了独立的文本引导网络，可以处理文本描述，
    使用CLIP作为文本编码器，并在SR过程中的关键位置引入文本引导。
    """

    def __init__(self, opt):
        self.use_text_features = True
        
        # 调用基类初始化
        super(TGSRModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # 模拟JPEG压缩
        self.usm_sharpener = USMSharp().cuda()  # USM锐化
        self.queue_size = opt.get('queue_size', 180)
        
        # 初始化文本相关组件
        if self.use_text_features:
            self.init_text_encoder()
            self.init_text_guidance_losses()
        
        # 存储注意力图
        self.attention_maps = None
        
        # 梯度累积设置
        self.accumulate_grad_batches = opt.get('accumulate_grad_batches', 1)
        self.grad_count = 0
    
    def init_training_settings(self):
        """重写init_training_settings，确保文本引导网络在setup_optimizers之前被初始化"""
        # 先初始化文本引导网络（如果使用）
        if self.is_train and self.use_text_features:
            self.init_text_guidance_net()
        
        # 调用父类的初始化训练设置
        super(TGSRModel, self).init_training_settings()
    
    def init_text_encoder(self):
        """初始化文本编码器（CLIP）"""
        logger = get_root_logger()
        
        text_encoder_opt = self.opt.get('text_encoder', {})
        self.text_encoder_name = text_encoder_opt.get('name', None)
        self.text_dim = text_encoder_opt.get('text_dim', 512)
        self.freeze_text_encoder = text_encoder_opt.get('freeze', True)
        
        # 加载CLIP文本编码器
        if self.text_encoder_name:
            try:
                logger.info(f'加载文本编码器: {self.text_encoder_name}')
                self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder_name)
                self.clip_text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name)
                
                # 冻结文本编码器参数（如果需要）
                if self.freeze_text_encoder:
                    logger.info('冻结文本编码器参数')
                    for param in self.clip_text_encoder.parameters():
                        param.requires_grad = False
                    self.clip_text_encoder.eval()
                
                # 移动到GPU
                self.clip_text_encoder = self.clip_text_encoder.to(self.device)
                logger.info(f'成功加载文本编码器: {self.text_encoder_name}')
            except Exception as e:
                logger.error(f'加载文本编码器失败: {e}')
                self.clip_tokenizer = None
                self.clip_text_encoder = None
                self.use_text_features = False
    
    def init_text_guidance_net(self):
        """初始化文本引导网络"""
        if not self.use_text_features:
            return
            
        logger = get_root_logger()
        
        try:
            # 检查配置是否存在
            if 'network_t' not in self.opt:
                logger.error("配置中缺少network_t参数")
                raise ValueError("配置中缺少network_t参数")
            
            # 构建文本引导网络
            self.net_t = build_network(self.opt['network_t'])
            
            # 设置参数为可训练
            for param in self.net_t.parameters():
                param.requires_grad = True
            
            # 移动到设备
            self.net_t = self.model_to_device(self.net_t)
            
            # 打印网络结构
            self.print_network(self.net_t)
            
            # 加载预训练权重（如果有）
            load_path = self.opt['path'].get('pretrain_network_t', None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_t', 'params')
                self.load_network(self.net_t, load_path, self.opt['path'].get('strict_load_t', True), param_key)
            
            self.net_t.train()
            return True
            
        except Exception as e:
            logger.error(f"初始化文本引导网络失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def init_text_guidance_losses(self):
        """初始化文本引导网络的特定损失函数"""
        logger = get_root_logger()
        train_opt = self.opt['train']
        
        # 1. 语义一致性损失
        if 'cri_semantic' in train_opt:
            cri_semantic_opt = train_opt['cri_semantic']
            self.cri_semantic = build_loss(cri_semantic_opt).to(self.device)
            logger.info(f'初始化语义一致性损失: {cri_semantic_opt["type"]}')
        
        # 2. 文本区域监督注意力损失
        if 'cri_attention' in train_opt:
            cri_attention_opt = train_opt['cri_attention']
            self.cri_attention = build_loss(cri_attention_opt).to(self.device)
            logger.info(f'初始化文本区域监督注意力损失: {cri_attention_opt["type"]}')
        
        # 3. 特征细化损失
        if 'cri_refinement' in train_opt:
            cri_refinement_opt = train_opt['cri_refinement']
            self.cri_refinement = build_loss(cri_refinement_opt).to(self.device)
            logger.info(f'初始化特征细化损失: {cri_refinement_opt["type"]}')
        
        # 特征投影层，用于投影特征到文本空间维度
        if hasattr(self, 'text_dim'):
            feat_dim = train_opt.get('feat_dim', 64)  # 默认特征维度
            if feat_dim != self.text_dim:
                self.feat_proj = nn.Linear(feat_dim, self.text_dim).to(self.device)
                logger.info(f'创建特征投影层: {feat_dim} -> {self.text_dim}')

    def setup_optimizers(self):
        """设置优化器，包括SR网络和文本引导网络"""
        train_opt = self.opt['train']
        
        # 优化器G - 超分辨率网络
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
        
        # 注意：使用copy避免修改原始配置
        optim_g_config = train_opt['optim_g'].copy()
        optim_type = optim_g_config.pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **optim_g_config)
        self.optimizers.append(self.optimizer_g)
        
        # 优化器D - 判别器
        # 注意：使用copy避免修改原始配置
        optim_d_config = train_opt['optim_d'].copy()
        optim_type = optim_d_config.pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **optim_d_config)
        self.optimizers.append(self.optimizer_d)
        
        # 优化器T - 文本引导网络（如果使用）
        if self.use_text_features and hasattr(self, 'net_t'):
            if 'optim_t' in train_opt:
                # 收集可训练参数
                optim_params_t = []
                for k, v in self.net_t.named_parameters():
                    if v.requires_grad:
                        optim_params_t.append(v)
                
                # 文本编码器参数（如果未冻结）
                if hasattr(self, 'clip_text_encoder') and not self.freeze_text_encoder:
                    for k, v in self.clip_text_encoder.named_parameters():
                        if v.requires_grad:
                            optim_params_t.append(v)
                
                # 注意：使用copy避免修改原始配置
                optim_t_config = train_opt['optim_t'].copy()
                optim_type = optim_t_config.pop('type')
                
                # 保存配置用于可能的重新创建
                self._optim_t_config = {
                    'type': optim_type,
                    **optim_t_config
                }
                
                if optim_params_t:
                    self.optimizer_t = self.get_optimizer(optim_type, optim_params_t, **optim_t_config)
                    self.optimizers.append(self.optimizer_t)
    
    def encode_text(self, text_prompts):
        """编码文本提示"""
        if not self.use_text_features or not hasattr(self, 'clip_tokenizer') or not hasattr(self, 'clip_text_encoder'):
            # 如果不使用文本特征，返回空张量
            batch_size = self.lq.size(0) if hasattr(self, 'lq') else 1
            text_hidden = torch.zeros(batch_size, 77, self.text_dim).to(self.device)
            text_pooled = torch.zeros(batch_size, self.text_dim).to(self.device)
            return text_hidden, text_pooled
        
        # 使用CLIP编码文本
        with torch.no_grad() if self.freeze_text_encoder else torch.enable_grad():
            text_inputs = self.clip_tokenizer(
                text_prompts, 
                padding="max_length", 
                max_length=77, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            text_outputs = self.clip_text_encoder(**text_inputs)
            text_hidden = text_outputs.last_hidden_state
            text_pooled = text_outputs.pooler_output
        
        return text_hidden, text_pooled

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """训练数据队列，用于增加批次中合成退化的多样性"""
        # 初始化
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'队列大小 {self.queue_size} 必须能被批次大小 {b} 整除'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            # 存储文本提示
            if self.use_text_features:
                self.queue_text_prompts = [''] * self.queue_size
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # 队列已满
            # 出队和入队
            # 打乱
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # 打乱文本提示
            if self.use_text_features:
                self.queue_text_prompts = [self.queue_text_prompts[i] for i in idx.tolist()]
            # 获取前b个样本
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # 获取文本提示
            if self.use_text_features:
                text_prompts_dequeue = self.queue_text_prompts[0:b]
            # 更新队列
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()
            # 更新文本提示
            if self.use_text_features:
                self.queue_text_prompts[0:b] = self.text_prompts
                self.text_prompts = text_prompts_dequeue
            
            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # 只进行入队
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            # 入队文本提示
            if self.use_text_features:
                self.queue_text_prompts[self.queue_ptr:self.queue_ptr + b] = self.text_prompts
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """接收数据并添加二阶退化以获得低质量图像，对文本相关区域进行特殊处理"""
        # 存储文本提示
        if self.use_text_features and 'text_prompt' in data:
            self.text_prompts = data['text_prompt']
        else:
            self.text_prompts = [''] * len(data['gt'] if 'gt' in data else data['lq'])
            
        # 获取文本区域掩码（如果有）
        if 'text_regions_mask' in data:
            self.degradation_mask = data['text_regions_mask'].to(self.device)
            self.use_targeted_degradation = True
            # 验证掩码值范围
            mask_min = self.degradation_mask.min().item()
            mask_max = self.degradation_mask.max().item()
            text_region_ratio = (self.degradation_mask < 0.5).float().mean().item() * 100
        else:
            self.degradation_mask = None
            self.use_targeted_degradation = False
            print("无掩码数据，使用标准退化")
        
        # 解码对象信息（如果有）
        self.objects_info = None
        if 'objects_info_str' in data:
            try:
                import json
                # 将批次中的每个JSON字符串解码为Python对象
                batch_objects = []
                for obj_str in data['objects_info_str']:
                    batch_objects.append(json.loads(obj_str))
                self.objects_info = batch_objects
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f"解析对象信息失败: {e}")
                self.objects_info = None
        
        if self.is_train and self.opt.get('high_order_degradation', True):
            # 训练数据合成
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # 模糊 - 使用差异化退化（如果有掩码）
            if self.use_targeted_degradation and self.degradation_mask is not None:
                # 按照RealESRGAN流程进行模糊处理，对文本区域使用较轻的模糊
                # 对不同区域分别使用不同参数的完整RealESRGAN退化
                
                # 创建文本区域和非文本区域的掩码
                # 文本区域掩码：退化掩码值小于1.0的区域
                batch_size = self.gt_usm.size(0)
                out = torch.zeros_like(self.gt_usm)

                # 针对每个样本单独处理
                for i in range(batch_size):
                    sample_mask = self.degradation_mask[i:i+1]  # [1, 1, H, W]
                    sample_gt = self.gt_usm[i:i+1]  # [1, C, H, W]
                    
                    # 创建两个掩码：文本区域和非文本区域
                    text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                    other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                    
                    # 为文本区域创建轻度的模糊核
                    text_kernel_size = self.opt.get('text_blur', {}).get('kernel_size', 11)  # 从配置读取，默认11
                    text_surround_weight = self.opt.get('text_blur', {}).get('surround_weight', 0.01)  # 从配置读取，默认0.01
                    text_center_weight = self.opt.get('text_blur', {}).get('center_weight', 1.0)  # 从配置读取，默认1.0
                    
                    text_kernel = torch.ones((1, 1, text_kernel_size, text_kernel_size), 
                                          device=self.device) * text_surround_weight
                    center = text_kernel_size // 2
                    text_kernel[0, 0, center, center] = text_center_weight  # 中心点权重更高，减少模糊效果
                    
                    # 添加归一化处理，确保卷积核权重和为1.0，防止图像变白
                    text_kernel = text_kernel / text_kernel.sum()
                    
                    # 文本区域应用轻度模糊
                    text_blurred = safe_filter2D(sample_gt, text_kernel)
                    
                    # 其他区域应用标准模糊 - 使用第一个kernel1避免批次问题
                    kernel1_single = self.kernel1[0:1] if self.kernel1.dim() == 4 and self.kernel1.size(0) > 1 else self.kernel1
                    other_blurred = safe_filter2D(sample_gt, kernel1_single)
                    
                    # 使用掩码加权混合
                    out[i:i+1] = text_blurred * text_region_mask + other_blurred * other_region_mask
                    
            else:
                # 标准模糊
                out = safe_filter2D(self.gt_usm, self.kernel1)
            
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
            if self.use_targeted_degradation and self.degradation_mask is not None:
                self.degradation_mask = F.interpolate(
                    self.degradation_mask, scale_factor=scale, mode='nearest'
                )
            
            # 添加噪声
            gray_noise_prob = self.opt['gray_noise_prob']
            if self.use_targeted_degradation and self.degradation_mask is not None:
                # 对不同区域分别应用不同强度的噪声，然后混合
                batch_size = out.size(0)
                out_with_noise = torch.zeros_like(out)
                
                for i in range(batch_size):
                    sample_mask = self.degradation_mask[i:i+1]  # [1, 1, H, W]
                    sample_out = out[i:i+1]  # [1, C, H, W]
                    
                    # 创建掩码
                    text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                    other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                    
                    # 文本区域：使用较轻的噪声参数
                    noise_strength_factor = self.opt.get('text_noise', {}).get('strength_factor', 0.5)  # 从配置读取，默认0.5
                    text_noise_sigma = np.array(self.opt['noise_range']) * noise_strength_factor
                    
                    # 对两个区域分别完整应用RealESRGAN退化
                    if np.random.uniform() < self.opt['gaussian_noise_prob']:
                        # 对两个区域分别应用高斯噪声
                        text_region_noisy = random_add_gaussian_noise_pt(
                            sample_out, sigma_range=text_noise_sigma, clip=True, 
                            rounds=False, gray_prob=gray_noise_prob)
                        
                        other_region_noisy = random_add_gaussian_noise_pt(
                            sample_out, sigma_range=self.opt['noise_range'], clip=True, 
                            rounds=False, gray_prob=gray_noise_prob)
                        
                        # 合并结果
                        out_with_noise[i:i+1] = text_region_noisy * text_region_mask + other_region_noisy * other_region_mask
                    else:
                        # 对两个区域分别应用泊松噪声
                        noise_strength_factor = self.opt.get('text_noise', {}).get('strength_factor', 0.5)  # 从配置读取，默认0.5
                        text_poisson_scale = np.array(self.opt['poisson_scale_range']) * noise_strength_factor
                        
                        text_region_noisy = random_add_poisson_noise_pt(
                            sample_out, scale_range=text_poisson_scale, gray_prob=gray_noise_prob, 
                            clip=True, rounds=False)
                        
                        other_region_noisy = random_add_poisson_noise_pt(
                            sample_out, scale_range=self.opt['poisson_scale_range'], 
                            gray_prob=gray_noise_prob, clip=True, rounds=False)
                        
                        # 合并结果
                        out_with_noise[i:i+1] = text_region_noisy * text_region_mask + other_region_noisy * other_region_mask
                
                out = out_with_noise
            else:
                # 标准噪声
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range'], clip=True, 
                        rounds=False, gray_prob=gray_noise_prob)
            
            # JPEG压缩 - 为文本区域使用较高的质量
            if self.use_targeted_degradation and self.degradation_mask is not None:
                batch_size = out.size(0)
                out_compressed = torch.zeros_like(out)
                
                for i in range(batch_size):
                    sample_mask = self.degradation_mask[i:i+1]
                    sample_out = out[i:i+1]
                    
                    # 创建掩码
                    text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                    other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                    
                    # 对两个区域分别完整应用不同质量的JPEG压缩
                    quality_min_factor = self.opt.get('text_jpeg', {}).get('quality_min_factor', 1.5)  # 从配置读取，默认1.5
                    quality_max_factor = self.opt.get('text_jpeg', {}).get('quality_max_factor', 1.2)  # 从配置读取，默认1.2
                    quality_max = self.opt.get('text_jpeg', {}).get('quality_max', 90)  # 从配置读取，默认90
                    
                    text_jpeg_quality = torch.FloatTensor(1).uniform_(
                        min(self.opt['jpeg_range2'][0] * quality_min_factor, quality_max),  # 提高最低质量
                        min(100, self.opt['jpeg_range2'][1] * quality_max_factor)  # 最高不超过100
                    ).to(self.device)
                    
                    other_jpeg_quality = torch.FloatTensor(1).uniform_(
                        *self.opt['jpeg_range2']
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
            
            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # ----------------------- 第二阶段退化过程 ----------------------- #
            # 模糊 - 类似地对文本区域使用较轻的模糊
            if np.random.uniform() < self.opt['second_blur_prob']:
                if self.use_targeted_degradation and self.degradation_mask is not None:
                    # 对不同区域分别使用不同参数的完整RealESRGAN退化
                    batch_size = out.size(0)
                    out_blurred = torch.zeros_like(out)
                    
                    for i in range(batch_size):
                        sample_mask = self.degradation_mask[i:i+1]  # [1, 1, H, W]
                        sample_out = out[i:i+1]  # [1, C, H, W]
                        
                        # 创建文本区域和非文本区域的掩码
                        text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                        other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                        
                        # 为文本区域创建轻度的模糊核
                        text_kernel_size = self.opt.get('text_blur', {}).get('kernel_size', 11)  # 从配置读取，默认11
                        text_surround_weight = self.opt.get('text_blur', {}).get('surround_weight', 0.01)  # 从配置读取，默认0.01
                        text_center_weight = self.opt.get('text_blur', {}).get('center_weight', 1.0)  # 从配置读取，默认1.0
                        
                        text_kernel = torch.ones((1, 1, text_kernel_size, text_kernel_size), 
                                            device=self.device) * text_surround_weight
                        center = text_kernel_size // 2
                        text_kernel[0, 0, center, center] = text_center_weight  # 中心点权重更高，减少模糊效果
                        
                        # 添加归一化处理，确保卷积核权重和为1.0，防止图像变白
                        text_kernel = text_kernel / text_kernel.sum()
                        
                        # 文本区域应用轻度模糊
                        text_blurred = safe_filter2D(sample_out, text_kernel)
                        
                        # 其他区域应用标准模糊 - 使用第一个kernel2避免批次问题
                        kernel2_single = self.kernel2[0:1] if self.kernel2.dim() == 4 and self.kernel2.size(0) > 1 else self.kernel2
                        other_blurred = safe_filter2D(sample_out, kernel2_single)
                        
                        # 使用掩码加权混合
                        out_blurred[i:i+1] = text_blurred * text_region_mask + other_blurred * other_region_mask
                    
                    out = out_blurred
                else:
                    out = safe_filter2D(out, self.kernel2)
                    
            # 随机调整大小
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
                
            # 同样调整掩码大小
            if self.use_targeted_degradation and self.degradation_mask is not None:
                self.degradation_mask = F.interpolate(
                    self.degradation_mask, 
                    size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), 
                    mode='nearest'
                )
                
            # 添加噪声 - 与前一阶段类似，但使用第二阶段的参数
            gray_noise_prob = self.opt['gray_noise_prob2']
            if self.use_targeted_degradation and self.degradation_mask is not None:
                # 对不同区域分别应用不同强度的噪声，然后混合
                batch_size = out.size(0)
                out_with_noise = torch.zeros_like(out)
                
                for i in range(batch_size):
                    sample_mask = self.degradation_mask[i:i+1]  # [1, 1, H, W]
                    sample_out = out[i:i+1]  # [1, C, H, W]
                    
                    # 创建掩码
                    text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                    other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                    
                    # 对两个区域分别完整应用RealESRGAN退化
                    if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                        # 文本区域：使用较轻的噪声参数
                        noise_strength_factor = self.opt.get('text_noise', {}).get('strength_factor', 0.5)  # 从配置读取，默认0.5
                        text_noise_sigma = np.array(self.opt['noise_range2']) * noise_strength_factor
                        
                        # 对两个区域分别应用高斯噪声
                        text_region_noisy = random_add_gaussian_noise_pt(
                            sample_out, sigma_range=text_noise_sigma, clip=True, 
                            rounds=False, gray_prob=gray_noise_prob)
                        
                        other_region_noisy = random_add_gaussian_noise_pt(
                            sample_out, sigma_range=self.opt['noise_range2'], clip=True, 
                            rounds=False, gray_prob=gray_noise_prob)
                        
                        # 合并结果
                        out_with_noise[i:i+1] = text_region_noisy * text_region_mask + other_region_noisy * other_region_mask
                    else:
                        # 文本区域：使用较轻的泊松噪声参数
                        noise_strength_factor = self.opt.get('text_noise', {}).get('strength_factor', 0.5)  # 从配置读取，默认0.5
                        text_poisson_scale = np.array(self.opt['poisson_scale_range2']) * noise_strength_factor
                        
                        # 对两个区域分别应用泊松噪声
                        text_region_noisy = random_add_poisson_noise_pt(
                            sample_out, scale_range=text_poisson_scale, gray_prob=gray_noise_prob, 
                            clip=True, rounds=False)
                        
                        other_region_noisy = random_add_poisson_noise_pt(
                            sample_out, scale_range=self.opt['poisson_scale_range2'], 
                            gray_prob=gray_noise_prob, clip=True, rounds=False)
                        
                        # 合并结果
                        out_with_noise[i:i+1] = text_region_noisy * text_region_mask + other_region_noisy * other_region_mask
                    
                out = out_with_noise
            else:
                # 标准噪声
                if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range2'], clip=True, 
                        rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out, scale_range=self.opt['poisson_scale_range2'], 
                        gray_prob=gray_noise_prob, clip=True, rounds=False)
  
            # JPEG压缩 + 最终sinc滤波器 - 两种顺序
            if np.random.uniform() < 0.5:
                # 1. [调整回原尺寸 + sinc滤波] + JPEG压缩
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                
                # 同样调整掩码大小
                if self.use_targeted_degradation and self.degradation_mask is not None:
                    self.degradation_mask = F.interpolate(
                        self.degradation_mask, 
                        size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), 
                        mode='nearest'
                    )
                
                # 差异化sinc滤波
                if self.use_targeted_degradation and self.degradation_mask is not None:
                    # 对不同区域分别应用不同sinc滤波
                    batch_size = out.size(0)
                    out_sinc = torch.zeros_like(out)
                    
                    for i in range(batch_size):
                        sample_mask = self.degradation_mask[i:i+1]
                        sample_out = out[i:i+1]
                        
                        # 创建掩码
                        text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                        other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                        
                        # 对文本区域，几乎不应用sinc滤波
                        text_sinc = sample_out  # 不应用滤波，保持原样
                        
                        # 对其他区域应用sinc滤波 - 使用单个kernel避免批次问题
                        sinc_kernel_single = self.sinc_kernel if self.sinc_kernel.dim() == 2 else self.sinc_kernel[0:1]
                        other_sinc = safe_filter2D(sample_out, sinc_kernel_single)
                        
                        # 合并结果
                        out_sinc[i:i+1] = text_sinc * text_region_mask + other_sinc * other_region_mask
                    
                    out = out_sinc
                else:
                    out = safe_filter2D(out, self.sinc_kernel)
                
                # JPEG压缩，文本区域质量较高
                if self.use_targeted_degradation and self.degradation_mask is not None:
                    batch_size = out.size(0)
                    out_compressed = torch.zeros_like(out)
                    
                    for i in range(batch_size):
                        sample_mask = self.degradation_mask[i:i+1]
                        sample_out = out[i:i+1]
                        
                        # 创建掩码
                        text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                        other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                        
                        # 对两个区域分别完整应用不同质量的JPEG压缩
                        quality_min_factor = self.opt.get('text_jpeg', {}).get('quality_min_factor', 1.5)  # 从配置读取，默认1.5
                        quality_max_factor = self.opt.get('text_jpeg', {}).get('quality_max_factor', 1.2)  # 从配置读取，默认1.2
                        quality_max = self.opt.get('text_jpeg', {}).get('quality_max', 90)  # 从配置读取，默认90
                        
                        text_jpeg_quality = torch.FloatTensor(1).uniform_(
                            min(self.opt['jpeg_range2'][0] * quality_min_factor, quality_max),  # 提高最低质量
                            min(100, self.opt['jpeg_range2'][1] * quality_max_factor)  # 最高不超过100
                        ).to(self.device)
                        
                        other_jpeg_quality = torch.FloatTensor(1).uniform_(
                            *self.opt['jpeg_range2']
                        ).to(self.device)
                        
                        # 进行压缩
                        sample_out = torch.clamp(sample_out, 0, 1)
                        
                        # 对两个区域分别完整应用JPEG压缩
                        text_compressed = self.jpeger(sample_out, quality=text_jpeg_quality)
                        other_compressed = self.jpeger(sample_out, quality=other_jpeg_quality)
                        
                        # 合并结果
                        out_compressed[i:i+1] = text_compressed * text_region_mask + other_compressed * other_region_mask
                    
                    out = out_compressed
                else:
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
            else:
                # 2. JPEG压缩 + [调整回原尺寸 + sinc滤波]
                # JPEG压缩，文本区域质量较高
                if self.use_targeted_degradation and self.degradation_mask is not None:
                    batch_size = out.size(0)
                    out_compressed = torch.zeros_like(out)
                    
                    for i in range(batch_size):
                        sample_mask = self.degradation_mask[i:i+1]
                        sample_out = out[i:i+1]
                        
                        # 创建掩码
                        text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                        other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                        
                        # 对两个区域分别完整应用不同质量的JPEG压缩
                        quality_min_factor = self.opt.get('text_jpeg', {}).get('quality_min_factor', 1.5)  # 从配置读取，默认1.5
                        quality_max_factor = self.opt.get('text_jpeg', {}).get('quality_max_factor', 1.2)  # 从配置读取，默认1.2
                        quality_max = self.opt.get('text_jpeg', {}).get('quality_max', 90)  # 从配置读取，默认90
                        
                        text_jpeg_quality = torch.FloatTensor(1).uniform_(
                            min(self.opt['jpeg_range2'][0] * quality_min_factor, quality_max),  # 提高最低质量
                            min(100, self.opt['jpeg_range2'][1] * quality_max_factor)  # 最高不超过100
                        ).to(self.device)
                        
                        other_jpeg_quality = torch.FloatTensor(1).uniform_(
                            *self.opt['jpeg_range2']
                        ).to(self.device)
                        
                        # 进行压缩
                        sample_out = torch.clamp(sample_out, 0, 1)
                        
                        # 对两个区域分别完整应用JPEG压缩
                        text_compressed = self.jpeger(sample_out, quality=text_jpeg_quality)
                        other_compressed = self.jpeger(sample_out, quality=other_jpeg_quality)
                        
                        # 合并结果
                        out_compressed[i:i+1] = text_compressed * text_region_mask + other_compressed * other_region_mask
                    
                    out = out_compressed
                else:
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                    
                # 调整回原尺寸
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                
                # 同样调整掩码大小
                if self.use_targeted_degradation and self.degradation_mask is not None:
                    self.degradation_mask = F.interpolate(
                        self.degradation_mask, 
                        size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), 
                        mode='nearest'
                    )
                
                # 差异化sinc滤波
                if self.use_targeted_degradation and self.degradation_mask is not None:
                    # 对不同区域分别应用不同sinc滤波
                    batch_size = out.size(0)
                    out_sinc = torch.zeros_like(out)
                    
                    for i in range(batch_size):
                        sample_mask = self.degradation_mask[i:i+1]
                        sample_out = out[i:i+1]
                        
                        # 创建掩码
                        text_region_mask = (sample_mask < 0.5).float()  # 文本区域掩码 (0值区域)
                        other_region_mask = (sample_mask >= 0.5).float()  # 其他区域掩码 (1值区域)
                        
                        # 对文本区域，几乎不应用sinc滤波
                        text_sinc = sample_out  # 不应用滤波，保持原样
                        
                        # 对其他区域应用sinc滤波 - 使用单个kernel避免批次问题
                        sinc_kernel_single = self.sinc_kernel if self.sinc_kernel.dim() == 2 else self.sinc_kernel[0:1]
                        other_sinc = safe_filter2D(sample_out, sinc_kernel_single)
                        
                        # 合并结果
                        out_sinc[i:i+1] = text_sinc * text_region_mask + other_sinc * other_region_mask
                    
                    out = out_sinc
                else:
                    out = safe_filter2D(out, self.sinc_kernel)
            
            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # 直接调整图像大小到目标尺寸 (1/scale)，不使用随机裁剪
            gt_size = self.opt['gt_size']
            scale = self.opt['scale']
            
            # 确保GT图像为目标大小
            if self.gt.shape[2:] != (gt_size, gt_size):
                self.gt = F.interpolate(self.gt, size=(gt_size, gt_size), mode='bicubic', align_corners=True)
                self.gt_usm = F.interpolate(self.gt_usm, size=(gt_size, gt_size), mode='bicubic', align_corners=True)
            
            # 确保LQ图像为 GT/scale 大小
            lq_size = gt_size // scale
            self.lq = F.interpolate(self.lq, size=(lq_size, lq_size), mode='bicubic', align_corners=True)
            
            # 同样调整掩码大小
            if self.use_targeted_degradation and self.degradation_mask is not None:
                self.degradation_mask = F.interpolate(
                    self.degradation_mask, size=(lq_size, lq_size), mode='nearest'
                )
            
            # 训练数据队列
            self._dequeue_and_enqueue()
            # 再次锐化gt，因为我们使用_dequeue_and_enqueue改变了gt
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # 防止警告：grad和param不遵循梯度布局契约
            
            # 释放不再需要的变量
            del out, self.kernel1, self.kernel2, self.sinc_kernel
            torch.cuda.empty_cache()
        else:
            # 用于配对训练或验证
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def apply_text_guidance(self, features, text_hidden=None, text_pooled=None, block_idx=None):
        """应用文本引导到特征图
        
        Args:
            features: 要增强的特征图
            text_hidden: 文本隐藏状态
            text_pooled: 文本池化表示
            block_idx: RRDB块的索引，用于损失计算
        """
        if not self.use_text_features or not hasattr(self, 'net_t'):
            return features, None
        
        # 如果未提供文本特征，则编码文本提示
        if text_hidden is None or text_pooled is None:
            text_hidden, text_pooled = self.encode_text(self.text_prompts)
        
        # 创建位置编码（如果需要）
        position_info = None
        if block_idx is not None and hasattr(self.net_t, 'with_position') and self.net_t.with_position:
            # 如果文本引导网络支持位置信息，可以创建位置编码
            # 例如，可以使用one-hot编码表示位置
            num_blocks = len(self.net_g.RRDB_trunk)
            position_info = torch.zeros(features.size(0), num_blocks, device=features.device)
            position_info[:, block_idx] = 1.0
        
        # 应用文本引导
        if position_info is not None:
            enhanced_features, attention_maps = self.net_t(features, text_hidden, text_pooled, position_info)
        else:
            enhanced_features, attention_maps = self.net_t(features, text_hidden, text_pooled)
        
        # 存储块位置信息和注意力图，供损失函数使用
        if self.is_train and attention_maps is not None:
            if not hasattr(self, 'current_attention_maps') or self.current_attention_maps is None:
                self.current_attention_maps = {}
            self.current_attention_maps[block_idx] = attention_maps
        
        return enhanced_features, attention_maps
    
    def forward_sr_network(self, x, apply_guidance=True):
        """SR网络的前向传播，在关键位置应用文本引导
        
        Args:
            x: 输入的低质量图像
            apply_guidance: 是否应用文本引导
            
        Returns:
            sr: 超分辨率输出
        """
        # 属性名适配 - 解决body与RRDB_trunk的命名差异
        if hasattr(self.net_g, 'body') and not hasattr(self.net_g, 'RRDB_trunk'):
            # 仅首次调用时打印信息
            if not hasattr(self, '_attribute_adapted'):
                print("适配RRDBNet属性名：将body映射为RRDB_trunk，conv_body映射为trunk_conv")
                self._attribute_adapted = True
                
            # 创建属性别名，不修改原始对象
            self.net_g.RRDB_trunk = self.net_g.body
            self.net_g.trunk_conv = self.net_g.conv_body
        
        # 适配上采样层名称
        if (not hasattr(self.net_g, 'upconv1') or not hasattr(self.net_g, 'upconv2')) and not hasattr(self, '_upsampler_adapted'):
            print("适配RRDBNet上采样层命名")
            self._upsampler_adapted = True
            
            # 可能的上采样层命名方式
            possible_names = [
                ('upblock1', 'upblock2'),
                ('upsample1', 'upsample2'),
                ('upsampler1', 'upsampler2'),
                ('upconv_1', 'upconv_2'),
                ('upsample_block1', 'upsample_block2'),
                ('conv_up1', 'conv_up2')  # 添加这个匹配RRDBNet实际使用的名称
            ]
            
            # 检查各种可能的命名
            for name1, name2 in possible_names:
                if hasattr(self.net_g, name1) and hasattr(self.net_g, name2):
                    self.net_g.upconv1 = getattr(self.net_g, name1)
                    self.net_g.upconv2 = getattr(self.net_g, name2)
                    print(f"上采样层已映射: {name1} -> upconv1, {name2} -> upconv2")
                    break
            
            # 如果没有找到预定义的名称对，尝试推断命名模式
            if not hasattr(self.net_g, 'upconv1'):
                # 查找所有可能的上采样相关属性
                upsampler_attrs = [attr for attr in dir(self.net_g) if 'up' in attr.lower() and not attr.startswith('_')]
                if len(upsampler_attrs) >= 2:
                    print(f"找到可能的上采样层: {upsampler_attrs}")
                    # 假设前两个是我们需要的
                    self.net_g.upconv1 = getattr(self.net_g, upsampler_attrs[0])
                    self.net_g.upconv2 = getattr(self.net_g, upsampler_attrs[1])
                    print(f"上采样层已映射: {upsampler_attrs[0]} -> upconv1, {upsampler_attrs[1]} -> upconv2")
            
            # 检查lrelu
            if not hasattr(self.net_g, 'lrelu'):
                # 查找可能的激活函数
                for attr in dir(self.net_g):
                    if 'relu' in attr.lower() or 'activ' in attr.lower():
                        self.net_g.lrelu = getattr(self.net_g, attr)
                        print(f"激活函数已映射: {attr} -> lrelu")
                        break
                # 如果未找到，创建一个默认的lrelu
                if not hasattr(self.net_g, 'lrelu'):
                    import torch.nn as nn
                    self.net_g.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
                    print("创建默认的LeakyReLU激活函数")
                    
            # 检查最后一个卷积层
            if not hasattr(self.net_g, 'conv_last'):
                for attr in dir(self.net_g):
                    if 'last' in attr.lower() or 'final' in attr.lower() or 'out' in attr.lower():
                        if 'conv' in attr.lower():
                            self.net_g.conv_last = getattr(self.net_g, attr)
                            print(f"最后一个卷积层已映射: {attr} -> conv_last")
                            break
                            
            # 检查HRconv
            if not hasattr(self.net_g, 'HRconv'):
                for attr in dir(self.net_g):
                    if 'hr' in attr.lower() and 'conv' in attr.lower():
                        self.net_g.HRconv = getattr(self.net_g, attr)
                        print(f"HR卷积层已映射: {attr} -> HRconv")
                        break
                # 如果未找到，可能是直接使用conv_last
                if not hasattr(self.net_g, 'HRconv'):
                    if hasattr(self.net_g, 'conv_last'):
                        # 使用恒等映射，保持与原始结构兼容
                        self.net_g.HRconv = nn.Identity()
                        print("创建默认的HR卷积层(Identity)")
        
        # 安全检查 - 如果仍然没有所需属性，回退到简单方式
        if not hasattr(self.net_g, 'RRDB_trunk') or not hasattr(self.net_g, 'trunk_conv'):
            print("警告：找不到RRDB_trunk或trunk_conv属性，使用简单前向传播")
            return self.net_g(x)
        
        # 编码文本（如果需要）
        if self.use_text_features and apply_guidance and hasattr(self, 'net_t'):
            with torch.no_grad() if self.freeze_text_encoder else torch.enable_grad():
                text_hidden, text_pooled = self.encode_text(self.text_prompts)
        else:
            text_hidden, text_pooled = None, None
        
        # 浅层特征提取
        fea = self.net_g.conv_first(x)
        
        # RRDB主干处理
        trunk = fea
        attention_maps = []
        
        # 确定在哪些位置应用文本引导
        if self.use_text_features and apply_guidance and text_hidden is not None and text_pooled is not None:
            num_blocks = len(self.net_g.RRDB_trunk)
            # 选择更合理的引导位置: 浅层、中层和深层
            guidance_positions = [num_blocks // 6, num_blocks // 2, num_blocks * 5 // 6]
            
            # 缓存特征用于损失计算
            self.original_features_cache = {}
            self.enhanced_features_cache = {}
            
            for i, block in enumerate(self.net_g.RRDB_trunk):
                trunk = block(trunk)
                
                # 在关键位置应用文本引导
                if i in guidance_positions:
                    # 缓存原始特征用于损失计算
                    if self.is_train:
                        self.original_features_cache[i] = trunk.clone().detach()
                    
                    # 文本引导
                    trunk, attn_maps = self.apply_text_guidance(trunk, text_hidden, text_pooled, block_idx=i)
                    
                    # 缓存增强后的特征
                    if self.is_train:
                        self.enhanced_features_cache[i] = trunk.clone().detach()
                    
                    # 保存注意力图
                    if attn_maps is not None and (not self.is_train or getattr(self, 'save_attention', False)):
                        if isinstance(attn_maps, list):
                            attention_maps.extend(attn_maps)
                        else:
                            attention_maps.append(attn_maps)
        else:
            # 不使用文本引导
            for block in self.net_g.RRDB_trunk:
                trunk = block(trunk)
        
        # 残差连接
        trunk = self.net_g.trunk_conv(trunk)
        fea = fea + trunk
        
        # 上采样
        try:
            # 尝试使用标准上采样路径
            fea = self.net_g.lrelu(self.net_g.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.net_g.lrelu(self.net_g.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
            
            # 最终输出
            out = self.net_g.conv_last(self.net_g.lrelu(self.net_g.HRconv(fea)))
        except (AttributeError, RuntimeError) as e:
            # 出错时，尝试调用完整的前向传播
            print(f"上采样出错: {e}，使用模型自带的前向传播")
            # 直接使用模型自带的前向方法
            out = self.net_g(x)
        
        # 保存注意力图
        if len(attention_maps) > 0:
            self.attention_maps = attention_maps
        
        return out
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """验证过程，修改为直接使用高分辨率图像验证SR结果，并增加文本引导可视化"""
        # 禁用数据合成过程
        self.is_train = False
        
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        
        # 初始化最佳指标结果
        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            # 重置指标结果
            self.metric_results = {metric: 0 for metric in self.metric_results}
        
        # 创建结果目录
        save_path_root = self.opt['path']['visualization']
        if save_img and not osp.exists(save_path_root):
            os.makedirs(save_path_root, exist_ok=True)
            if self.opt.get('val', {}).get('save_attention_maps', False):
                os.makedirs(osp.join(save_path_root, 'attention'), exist_ok=True)
        
        for idx, val_data in enumerate(dataloader):
            # 获取文本提示（如果有）
            if 'text_prompt' in val_data:
                self.text_prompts = val_data['text_prompt']
            else:
                self.text_prompts = [''] * len(val_data['gt'])
            
            # 设置图像数据
            if 'lq' in val_data:
                # 使用提供的低质量图像
                self.lq = val_data['lq'].to(self.device)
            elif 'gt' in val_data:
                # 直接从GT下采样生成LQ（简单方式）
                self.gt = val_data['gt'].to(self.device)
                self.lq = F.interpolate(self.gt, scale_factor=1/self.opt['scale'], mode='bicubic')
            
            # 获取图像名称
            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            
            # 进行两种测试：有文本引导和无文本引导
            results = {}
            
            # 1. 测试带文本引导的结果
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.forward_sr_network(self.lq, apply_guidance=True)
                    results['guided'] = self.output.clone()
                self.net_g.train()  # 恢复训练模式
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.forward_sr_network(self.lq, apply_guidance=True)
                    results['guided'] = self.output.clone()
                
            # 只保存前20张图片的结果
            save_this_img = save_img and idx < 20
                
            # 保存注意力图（如果有）
            if save_this_img and self.opt.get('val', {}).get('save_attention_maps', False) and hasattr(self, 'attention_maps') and self.attention_maps:
                attention_maps = self.get_current_attention_maps()
                if attention_maps:
                    for attn_name, attn_map in attention_maps.items():
                        attn_colored = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        save_path = osp.join(save_path_root, 'attention', f'{img_name}_{attn_name}_{current_iter}.png')
                        cv2.imwrite(save_path, attn_colored)
            
            # 2. 可选: 测试无文本引导的结果进行对比
            if self.opt.get('val', {}).get('compare_with_unguided', False):
                if hasattr(self, 'net_g_ema'):
                    with torch.no_grad():
                        self.output = self.forward_sr_network(self.lq, apply_guidance=False)
                        results['unguided'] = self.output.clone()
                else:
                    with torch.no_grad():
                        self.output = self.forward_sr_network(self.lq, apply_guidance=False)
                        results['unguided'] = self.output.clone()
            
            # 设置回引导结果用于指标计算
            self.output = results['guided']
            
            # 计算指标和保存图像
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            
            if save_this_img:
                
                # 保存文本提示信息
                if 'text_prompt' in val_data:
                    prompt_path = osp.join(save_path_root, f'{img_name}_prompt_{current_iter}.txt')
                    with open(prompt_path, 'w') as f:
                        f.write(val_data['text_prompt'][0])
                
                # 保存无引导结果（如果有）
                if 'unguided' in results:
                    unguided_output = results['unguided']
                    unguided_img = tensor2img([unguided_output.detach().cpu()])
                    # 创建对比图
                    if self.opt.get('val', {}).get('save_comparison', False):
                        # 创建并排对比图
                        guided_img = sr_img
                        h, w = guided_img.shape[:2]
                        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
                        comparison[:, :w] = unguided_img
                        comparison[:, w:] = guided_img
                        # 添加分割线
                        comparison[:, w-1:w+1] = [0, 0, 255]  # 红色分割线
                        save_comp_path = osp.join(save_path_root, f'{img_name}_comparison_{current_iter}.png')
                        imwrite(comparison, save_comp_path)
            
            # 计算指标
            if with_metrics and 'gt' in val_data:
                gt_img = tensor2img([visuals['gt']])
                metric_data = {'img': sr_img, 'img2': gt_img}
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                
                # 对比引导与非引导结果（如果有）
                if 'unguided' in results and self.opt.get('val', {}).get('compare_metrics', False):
                    unguided_img = tensor2img([results['unguided'].detach().cpu()])
                    unguided_metric_data = {'img': unguided_img, 'img2': gt_img}
                    for name, opt_ in self.opt['val']['metrics'].items():
                        unguided_value = calculate_metric(unguided_metric_data, opt_)
                        if not hasattr(self, 'unguided_metric_results'):
                            self.unguided_metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
                        self.unguided_metric_results[name] += unguided_value
                        
                        # 计算提升
                        if not hasattr(self, 'improvement_metric_results'):
                            self.improvement_metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
                        
                        guided_value = calculate_metric(metric_data, opt_)
                        improvement = guided_value - unguided_value
                        self.improvement_metric_results[name] += improvement
            
            # 清理内存
            torch.cuda.empty_cache()
        
        # 计算平均指标
        if with_metrics and idx > 0:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # 更新最佳指标
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
                
                # 计算无引导结果和改进的平均值（如果有）
                if hasattr(self, 'unguided_metric_results'):
                    self.unguided_metric_results[metric] /= (idx + 1)
                if hasattr(self, 'improvement_metric_results'):
                    self.improvement_metric_results[metric] /= (idx + 1)
            
            # 记录验证指标
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
            # 记录对比指标（如果有）
            if hasattr(self, 'unguided_metric_results') and hasattr(self, 'improvement_metric_results'):
                log_str = f'引导与非引导比较 {dataset_name}\n'
                for metric in self.metric_results.keys():
                    guided_value = self.metric_results[metric]
                    unguided_value = self.unguided_metric_results[metric]
                    improvement = self.improvement_metric_results[metric]
                    log_str += f'\t# {metric}: 引导={guided_value:.4f}, 无引导={unguided_value:.4f}, 提升={improvement:.4f} ({improvement/unguided_value*100:.2f}%)\n'
                
                logger = get_root_logger()
                logger.info(log_str)
                
                if tb_logger:
                    for metric in self.metric_results.keys():
                        tb_logger.add_scalar(f'metrics/{dataset_name}/unguided_{metric}', 
                                           self.unguided_metric_results[metric], current_iter)
                        tb_logger.add_scalar(f'metrics/{dataset_name}/improvement_{metric}', 
                                           self.improvement_metric_results[metric], current_iter)
        
        # 恢复训练模式
        self.is_train = True

    def get_current_visuals(self):
        """获取当前可视化结果"""
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        
        # 可选：添加注意力图
        if hasattr(self, 'attention_maps') and self.attention_maps:
            # 转换注意力图为可视化格式
            for i, attn_map in enumerate(self.attention_maps):
                if attn_map is not None:
                    # 使用第一个样本的注意力图
                    attn_vis = attn_map[0, 0].detach().cpu().unsqueeze(0)
                    out_dict[f'attn_{i}'] = attn_vis
        
        return out_dict

    def forward(self, apply_guidance=True):
        """模型前向传播"""
        self.output = self.forward_sr_network(self.lq, apply_guidance)

    def optimize_parameters(self, current_iter):
        """优化参数，整合SR网络和文本引导网络，支持梯度累积"""
        # 定期清理CUDA缓存以减少内存碎片
        if current_iter % 50 == 0:
            torch.cuda.empty_cache()
            
        # 设置GT图像（带锐化或不带）
        l1_gt = self.gt_usm if self.opt.get('l1_gt_usm', True) else self.gt
        percep_gt = self.gt_usm if self.opt.get('percep_gt_usm', True) else self.gt
        gan_gt = self.gt_usm if self.opt.get('gan_gt_usm', False) else self.gt
        
        # 优化生成器（SR网络）
        for p in self.net_d.parameters():
            p.requires_grad = False

        # 只在累积的第一个批次时清零梯度
        if self.grad_count == 0:
            self.optimizer_g.zero_grad()
            if self.use_text_features and hasattr(self, 'net_t') and hasattr(self, 'optimizer_t'):
                self.optimizer_t.zero_grad()
        
        # 前向传播，应用文本引导
        self.output = self.forward_sr_network(self.lq, apply_guidance=True)

        l_g_total = 0
        loss_dict = OrderedDict()
        
        # 像素损失
        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output, l1_gt)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix
        # 感知损失
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style
        # GAN损失
        fake_g_pred = self.net_d(self.output)
        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
        l_g_total += l_g_gan
        loss_dict['l_g_gan'] = l_g_gan

        # 使用累积批次大小进行归一化
        l_g_total = l_g_total / self.accumulate_grad_batches
        # 如果有文本引导网络，保留计算图以便后续的反向传播
        retain_graph = self.use_text_features and hasattr(self, 'net_t') and hasattr(self, 'optimizer_t')
        l_g_total.backward(retain_graph=retain_graph)
        
        # 文本引导网络损失
        l_t_total = 0
        if self.use_text_features and hasattr(self, 'net_t') and hasattr(self, 'optimizer_t'):
            # 获取文本特征和对象信息
            text_hidden, text_pooled = self.encode_text(self.text_prompts)
            objects_info = None
            if hasattr(self, 'objects_info'):
                objects_info = self.objects_info
            
            # 1. 语义一致性损失
            if hasattr(self, 'cri_semantic') and hasattr(self, 'enhanced_features_cache') and self.enhanced_features_cache:
                # 对每个位置的增强特征应用损失
                for pos, enhanced_feat in self.enhanced_features_cache.items():
                    l_t_semantic = self.cri_semantic(enhanced_feat, text_pooled, 
                                                   getattr(self, 'feat_proj', None))
                    l_t_semantic = l_t_semantic / self.accumulate_grad_batches  # 归一化
                    l_t_total += l_t_semantic
                    loss_dict[f'l_t_semantic_{pos}'] = l_t_semantic
            
            # 2. 文本区域监督注意力损失
            if (hasattr(self, 'cri_attention') and hasattr(self, 'current_attention_maps') 
                and self.current_attention_maps and objects_info is not None):
                # 对每个位置的注意力图应用损失
                for pos, attn_maps in self.current_attention_maps.items():
                    if isinstance(attn_maps, list) and len(attn_maps) > 0:
                        # 使用最后一个注意力图（通常最精细）
                        attn_map = attn_maps[-1]
                        
                        # 使用逐样本处理方式处理objects_info
                        batch_size = attn_map.size(0)
                        l_t_attn_batch = 0
                        valid_samples = 0
                        
                        for i in range(batch_size):
                            # 获取当前样本的文本提示和对象信息
                            sample_text = [self.text_prompts[i]]
                            sample_objects = [objects_info[i]] if i < len(objects_info) else [[]]
                            sample_attn_map = attn_map[i:i+1]
                            
                            # 对单个样本进行处理
                            try:
                                sample_loss = self.cri_attention(sample_attn_map, sample_text, sample_objects, self.device)
                                
                                # 只有当样本损失有效时才累加
                                if sample_loss.item() > 0:
                                    l_t_attn_batch += sample_loss
                                    valid_samples += 1
                            except Exception:
                                continue
                        
                        # 计算批次平均损失
                        if valid_samples > 0:
                            l_t_attn = l_t_attn_batch / valid_samples
                            l_t_attn = l_t_attn / self.accumulate_grad_batches  # 归一化
                            l_t_total += l_t_attn
                            loss_dict[f'l_t_attn_{pos}'] = l_t_attn
            
            # 3. 特征细化损失
            if (hasattr(self, 'cri_refinement') and hasattr(self, 'original_features_cache') 
                and hasattr(self, 'enhanced_features_cache')):
                # 确保两个缓存包含相同的位置
                common_positions = set(self.original_features_cache.keys()) & set(self.enhanced_features_cache.keys())
                for pos in common_positions:
                    orig_feat = self.original_features_cache[pos]
                    enhanced_feat = self.enhanced_features_cache[pos]
                    l_t_refine = self.cri_refinement(orig_feat, enhanced_feat, text_pooled, 
                                                  getattr(self, 'feat_proj', None))
                    l_t_refine = l_t_refine / self.accumulate_grad_batches  # 归一化
                    l_t_total += l_t_refine
                    loss_dict[f'l_t_refine_{pos}'] = l_t_refine
            
            # 记录文本引导损失总和并反向传播
            if l_t_total > 0:
                loss_dict['l_t_total'] = l_t_total
                l_t_total.backward()
        
        # 增加梯度计数
        self.grad_count += 1
        
        # 当达到累积批次数时执行优化器步骤
        if self.grad_count >= self.accumulate_grad_batches:
            self.optimizer_g.step()
            if self.use_text_features and hasattr(self, 'net_t') and hasattr(self, 'optimizer_t'):
                self.optimizer_t.step()
            # 执行完optimizer.step()后立即清零梯度
            self.optimizer_g.zero_grad()
            if self.use_text_features and hasattr(self, 'net_t') and hasattr(self, 'optimizer_t'):
                self.optimizer_t.zero_grad()
            self.grad_count = 0
        
        # 清理GPU内存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # 优化判别器
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # 真实样本
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        
        # 清理GPU内存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
        # 虚假样本
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()
        # 判别器更新后也立即清零梯度
        self.optimizer_d.zero_grad()

        self.log_dict = loss_dict
        
        # 更新EMA模型
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        
        # 清理缓存的特征和注意力图
        if hasattr(self, 'current_attention_maps'):
            self.current_attention_maps = {}
        if hasattr(self, 'original_features_cache'):
            self.original_features_cache = {}
        if hasattr(self, 'enhanced_features_cache'):
            self.enhanced_features_cache = {}
    
    def get_current_attention_maps(self):
        """获取当前注意力图，用于可视化"""
        if self.attention_maps is None or len(self.attention_maps) == 0:
            return None
        
        attention_dict = {}
        for i, attn_map in enumerate(self.attention_maps):
            # 提取第一个样本的注意力图
            attention_2d = attn_map[0, 0].cpu().detach()
            
            # 归一化
            min_val = attention_2d.min()
            max_val = attention_2d.max()
            normalized_attn = (attention_2d - min_val) / (max_val - min_val + 1e-8)
            
            # 转换为numpy
            attention_np = normalized_attn.numpy()
            # 调整大小以匹配输出
            if hasattr(self, 'output'):
                h, w = self.output.shape[2], self.output.shape[3]
                attention_np = cv2.resize(attention_np, (w, h), interpolation=cv2.INTER_LINEAR)
            
            attention_dict[f'attention_{i}'] = attention_np
        
        return attention_dict
    
    def save(self, epoch, current_iter):
        """保存模型"""
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_network(self.net_t, 'net_t', current_iter)
        # 保存判别器
        self.save_network(self.net_d, 'net_d', current_iter)
        # 保存训练状态
        self.save_training_state(epoch, current_iter) 