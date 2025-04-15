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

@MODEL_REGISTRY.register()
class TGSRModel(SRGANModel):
    """文本引导的超分辨率模型

    基于SRGANModel，但增加了独立的文本引导网络，可以处理文本描述，
    使用CLIP作为文本编码器，并在SR过程中的关键位置引入文本引导。
    """

    def __init__(self, opt):
        self.use_text_features = True
        super(TGSRModel, self).__init__(opt)
        
        # 初始化日志记录器
        self.logger = get_root_logger()
        
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # 模拟JPEG压缩
        self.usm_sharpener = USMSharp().cuda()  # USM锐化
        self.queue_size = opt.get('queue_size', 180)
        
        # 初始化文本相关组件
        if self.use_text_features:
            self.init_text_encoder()
            self.init_text_guidance_losses()
        
        # 存储注意力图
        self.attention_maps = None
    
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
        
        if 'cri_control' in train_opt:
            cri_control_opt = train_opt['cri_control']
            self.cri_control = build_loss(cri_control_opt).to(self.device)
            logger.info(f'初始化控制特征损失: {cri_control_opt["type"]}')
        
        if 'cri_attention' in train_opt:
            cri_attention_opt = train_opt['cri_attention']
            self.cri_attention = build_loss(cri_attention_opt).to(self.device)
            logger.info(f'初始化文本区域监督注意力损失: {cri_attention_opt["type"]}')
        
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
            # 存储对象信息
            if hasattr(self, 'objects_info') and self.objects_info is not None:
                self.queue_objects_info = [None] * self.queue_size
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # 队列已满
            # 出队和入队
            # 打乱
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # 打乱文本提示和对象信息
            if self.use_text_features:
                self.queue_text_prompts = [self.queue_text_prompts[i] for i in idx.tolist()]
            # 打乱对象信息
            if hasattr(self, 'objects_info') and self.objects_info is not None and hasattr(self, 'queue_objects_info'):
                self.queue_objects_info = [self.queue_objects_info[i] for i in idx.tolist()]
            # 获取前b个样本
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # 获取文本提示
            if self.use_text_features:
                text_prompts_dequeue = self.queue_text_prompts[0:b]
            # 获取对象信息
            objects_info_dequeue = None
            if hasattr(self, 'objects_info') and self.objects_info is not None and hasattr(self, 'queue_objects_info'):
                objects_info_dequeue = self.queue_objects_info[0:b]
            # 更新队列
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()
            # 更新文本提示
            if self.use_text_features:
                self.queue_text_prompts[0:b] = self.text_prompts
                self.text_prompts = text_prompts_dequeue
            # 更新对象信息
            if hasattr(self, 'objects_info') and self.objects_info is not None and hasattr(self, 'queue_objects_info'):
                self.queue_objects_info[0:b] = self.objects_info
                self.objects_info = objects_info_dequeue
            
            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # 只进行入队
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            # 入队文本提示
            if self.use_text_features:
                self.queue_text_prompts[self.queue_ptr:self.queue_ptr + b] = self.text_prompts
            # 入队对象信息
            if hasattr(self, 'objects_info') and self.objects_info is not None:
                if not hasattr(self, 'queue_objects_info'):
                    self.queue_objects_info = [None] * self.queue_size
                self.queue_objects_info[self.queue_ptr:self.queue_ptr + b] = self.objects_info
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """接收数据并添加二阶退化以获得低质量图像"""
        # 存储文本提示
        if self.use_text_features and 'text_prompt' in data:
            self.text_prompts = data['text_prompt']
        else:
            self.text_prompts = [''] * len(data['gt'] if 'gt' in data else data['lq'])
        
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
            # 训练数据合成 - 严格遵循RealESRGAN方式
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- 第一阶段退化过程 ----------------------- #
            # 模糊
            out = filter2D(self.gt_usm, self.kernel1)
            
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
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, 
                    rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out, scale_range=self.opt['poisson_scale_range'], 
                    gray_prob=gray_noise_prob, clip=True, rounds=False)
            
            # JPEG压缩
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # 裁剪到[0, 1]，否则JPEGer会产生不良伪影
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- 第二阶段退化过程 ----------------------- #
            # 模糊
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
                
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
            
            # 添加噪声
            gray_noise_prob = self.opt['gray_noise_prob2']
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
                out = filter2D(out, self.sinc_kernel)
                
                # JPEG压缩
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # 2. JPEG压缩 + [调整回原尺寸 + sinc滤波]
                # JPEG压缩
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                    
                # 调整回原尺寸
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
            
            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # 去除随机裁剪，仅确保lq尺寸为gt的1/scale
            # 确保gt和lq尺寸匹配 (gt尺寸为scale倍的lq)
            target_h, target_w = self.lq.size()[2:4]  # lq尺寸
            gt_h, gt_w = target_h * self.opt['scale'], target_w * self.opt['scale']  # 期望的gt尺寸
            
            # 如果gt尺寸不匹配，则调整gt和gt_usm的尺寸
            if self.gt.size(2) != gt_h or self.gt.size(3) != gt_w:
                self.gt = F.interpolate(self.gt, size=(gt_h, gt_w), mode='bicubic')
                self.gt_usm = F.interpolate(self.gt_usm, size=(gt_h, gt_w), mode='bicubic')
            
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
                
                # 确保验证/测试时gt和lq尺寸匹配
                if self.gt.size(2) != self.lq.size(2) * self.opt['scale'] or self.gt.size(3) != self.lq.size(3) * self.opt['scale']:
                    target_h, target_w = self.lq.size()[2:4]
                    gt_h, gt_w = target_h * self.opt['scale'], target_w * self.opt['scale']
                    self.gt = F.interpolate(self.gt, size=(gt_h, gt_w), mode='bicubic')
                    self.gt_usm = F.interpolate(self.gt_usm, size=(gt_h, gt_w), mode='bicubic')

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
            num_blocks = len(self.net_g.body)
            position_info = torch.zeros(features.size(0), num_blocks, device=features.device)
            position_info[:, block_idx] = 1.0
        
        # 应用文本引导
        if position_info is not None:
            enhanced_features, attention_logits = self.net_t(features, text_hidden, text_pooled, position_info)
        else:
            enhanced_features, attention_logits = self.net_t(features, text_hidden, text_pooled)
        
        # 存储块位置信息和注意力图，供损失函数使用
        if self.is_train and attention_logits is not None:
            if not hasattr(self, 'current_attention_maps') or self.current_attention_maps is None:
                self.current_attention_maps = {}
            self.current_attention_maps[block_idx] = attention_logits
        
        return enhanced_features, attention_logits
    
    def forward_sr_network(self, x, apply_guidance=True):
        """SR网络的前向传播，使用ControlNet风格的文本引导
        
        Args:
            x: 输入的低质量图像
            apply_guidance: 是否应用文本引导
            
        Returns:
            sr: 超分辨率输出
        """
        # 获取文本特征（如果需要）
        if self.use_text_features and apply_guidance and hasattr(self, 'net_t'):
            with torch.no_grad() if self.freeze_text_encoder else torch.enable_grad():
                text_hidden, text_pooled = self.encode_text(self.text_prompts)
        else:
            text_hidden, text_pooled = None, None
            
        # 如果不使用文本引导，直接使用原始SR网络
        if not self.use_text_features or not apply_guidance or text_hidden is None:
            return self.net_g(x)
        
        # ====================== ControlNet风格的前向传播 ======================
        # 1. 锁定的前向传播（ControlNet的Locked Copy概念）
        with torch.no_grad():
            # 浅层特征提取
            locked_fea = self.net_g.conv_first(x)
            
            # 分组处理RRDB块，便于更粗粒度的引导
            locked_features = []
            locked_trunk = locked_fea
            
            # RRDB块分组（每4个为一组）
            block_groups = []
            total_blocks = len(self.net_g.body)
            group_size = 4  # 可根据总块数调整
            
            for i in range(0, total_blocks, group_size):
                end = min(i + group_size, total_blocks)
                block_groups.append(self.net_g.body[i:end])
            
            # 处理每个块组
            for group in block_groups:
                for block in group:
                    locked_trunk = block(locked_trunk)
                locked_features.append(locked_trunk.clone())
            
            # 主干结束的特征
            locked_body_out = self.net_g.conv_body(locked_trunk)
            locked_features.append(locked_body_out)
            
            # 上采样和最终输出的特征
            locked_up1 = self.net_g.conv_up1(F.interpolate(locked_body_out + locked_fea, scale_factor=2, mode='nearest'))
            locked_up2 = self.net_g.conv_up2(F.interpolate(locked_up1, scale_factor=2, mode='nearest'))
            
        # 2. 使用文本引导网络生成控制信号 - ControlNet的核心思想
        # 准备空间上的文本控制
        guided_features, attention_maps = self.net_t(locked_features[0], text_hidden, text_pooled)
        
        # 3. 应用控制信号和特征融合 
        # 注：与ControlNet类似，这里我们从锁定特征开始，逐步应用控制
        fea = self.net_g.conv_first(x)
        trunk = fea
        
        # 处理主干
        num_groups = len(block_groups)
        for i, group in enumerate(block_groups):
            # 处理每个块
            for block in group:
                trunk = block(trunk)
            
            # 在每个组结束后应用控制 - 如果有足够的控制特征
            if i < len(attention_maps):
                # 获取当前注意力图和控制强度
                attn = torch.sigmoid(attention_maps[i])
                # 自适应强度控制 - 训练开始时较低，随着训练进行逐渐增强
                if self.is_train:
                    # 计算训练进度调整强度
                    if hasattr(self, 'opt') and 'train' in self.opt:
                        total_iter = self.opt['train'].get('total_iter', 400000)
                        if hasattr(self, 'total_iter'):
                            total_iter = self.total_iter
                        progress = min(1.0, self.iter / total_iter) if hasattr(self, 'iter') else 0.5
                        control_strength = 0.1 + 0.9 * progress  # 从0.1逐渐增加到1.0
                    else:
                        control_strength = 0.5
                else:
                    control_strength = 1.0  # 测试时使用全强度
                
                # 应用控制 - trunk是变化的特征，locked_features[i]是锁定的参考特征
                # guided_features提供从锁定特征到修改后特征的映射
                control_signal = guided_features - locked_features[i]
                trunk = trunk + control_strength * control_signal * attn
        
        # 处理主干结束
        trunk = self.net_g.conv_body(trunk)
        fea = fea + trunk
        
        # 上采样和最终输出
        fea = self.net_g.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.net_g.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest'))
        out = self.net_g.conv_hr(fea)
        out = self.net_g.conv_last(out)
        
        # 保存注意力图用于可视化
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
        # if self.opt.get('val', {}).get('save_attention_maps', False):
        #     os.makedirs(osp.join(save_path_root, 'attention'), exist_ok=True)
        
        # 添加计数器，限制只测试前1000张图片
        test_count = 0
        max_test_samples = 1000
        
        for idx, val_data in enumerate(dataloader):
            # 检查是否已经测试了1000张图片
            if test_count >= max_test_samples:
                break
            
            test_count += 1
            
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
            
            # 设置回引导结果用于指标计算
            self.output = results['guided']
            
            # 计算指标和保存图像
            visuals = self.get_current_visuals()
            sr_img_original = tensor2img([visuals['result']])  # 原始超分辨率结果
            
            # 只保存前20张图片的结果
            save_this_img = save_img and idx < 20
                
            # 保存注意力图（如果有）
            if save_this_img and self.attention_maps:
                attention_maps = self.get_current_attention_maps()
                if attention_maps:
                    # 创建sr_img的独立副本用于热力图处理
                    sr_img_heatmap = sr_img_original.copy()
                    
                    # 1. 使用GradCAM风格保存合并后的注意力图
                    if self.opt.get('val', {}).get('save_attention_maps', False):
                        attn_save_path = self.save_gradcam_attention(sr_img_heatmap, attention_maps, img_name, save_path_root, current_iter)
                    
                    # 2. 新增：直接在输出图像上叠加注意力图
                    # 合并所有注意力图
                    all_maps = np.stack([attn for attn in attention_maps.values()])
                    combined_map = np.mean(all_maps, axis=0)
                    
                    # 确保尺寸匹配
                    combined_map = cv2.resize(combined_map, (sr_img_heatmap.shape[1], sr_img_heatmap.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
                    # 应用CLAHE自适应直方图均衡化增强对比度
                    combined_map_uint8 = (combined_map * 255).astype(np.uint8)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    combined_map_enhanced = clahe.apply(combined_map_uint8) / 255.0
                    
                    # 创建热力图
                    heatmap = cv2.applyColorMap((combined_map_enhanced * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    
                    # 使用另一个副本叠加热力图，避免修改sr_img_heatmap
                    overlay_img = cv2.addWeighted(sr_img_heatmap.copy(), 0.7, heatmap, 0.3, 0)
                    
                    # 保存叠加图
                    overlay_save_path = osp.join(save_path_root, f'{img_name}_overlay_{current_iter}.png')
                    imwrite(overlay_img, overlay_save_path)
            
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
            
            # 保存文本提示和比较图
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
                        # 创建四列对比图（LQ、无引导、有引导、GT+掩码）
                        # 使用原始的SR结果，而不是可能被修改的sr_img
                        guided_img = sr_img_original.copy()  # 确保创建新的副本
                        lq_img = tensor2img([self.lq])
                        gt_img = tensor2img([visuals['gt']])
                        
                        # 准备注意力图和掩码图像
                        overlay_ratio = 0.0  # 默认重叠率
                        gt_with_mask = gt_img.copy()  # 初始化为原始GT图像
                        
                        # 获取注意力图
                        attention_map = None
                        if hasattr(self, 'attention_maps') and self.attention_maps is not None and len(self.attention_maps) > 0:
                            # 获取最后一个注意力图（通常是最终的高分辨率注意力）
                            attn_logits = self.attention_maps[-1]
                            # 取第一个样本的注意力图
                            attention_map = torch.sigmoid(attn_logits[0, 0]).detach().cpu().numpy()
                            # 调整大小到GT尺寸
                            attention_map = cv2.resize(attention_map, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                        
                        # 处理掩码 - 使用验证数据中的掩码
                        if 'objects_info_str' in val_data:
                            try:
                                import json
                                objects_info = json.loads(val_data['objects_info_str'][0])
                                
                                if objects_info:
                                    h, w = gt_img.shape[:2]
                                    mask = np.zeros((h, w), dtype=np.uint8)
                                    
                                    # 处理所有对象掩码
                                    for obj in objects_info:
                                        if 'mask_encoded' in obj:
                                            try:
                                                from tgsr.losses.tgsr_loss import decode_mask
                                                obj_mask = decode_mask(obj['mask_encoded'])
                                                if obj_mask is not None and obj_mask.sum() > 0:
                                                    # 调整掩码大小
                                                    obj_mask_resized = cv2.resize(obj_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                                                    # 合并掩码
                                                    mask = np.maximum(mask, obj_mask_resized)
                                            except Exception as e:
                                                print(f"掩码处理错误: {e}")
                                                continue
                                    
                                    # 生成掩码覆盖的GT图像
                                    if mask.sum() > 0:
                                        # 创建带边界线的掩码覆盖
                                        mask_overlay = gt_img.copy()
                                        
                                        # 为掩码区域添加半透明覆盖
                                        mask_colored = np.zeros_like(gt_img)
                                        mask_colored[mask > 0] = [0, 255, 0]  # 绿色
                                        gt_with_mask = cv2.addWeighted(gt_img, 0.85, mask_colored, 0.15, 0)
                                        
                                        # 添加边界线以增强可见性
                                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        cv2.drawContours(gt_with_mask, contours, -1, (0, 255, 0), 1)  # 绿色边界线
                                        
                                        # 计算掩码和注意力图的重叠率
                                        if attention_map is not None:
                                            # 正确计算重叠率
                                            # 1. 注意力图中高于均值的点被认为是"注意"区域
                                            attention_thresh = np.mean(attention_map) + 0.5 * np.std(attention_map)
                                            attention_binary = (attention_map > attention_thresh).astype(np.float32)
                                            
                                            # 2. 计算交集和并集
                                            mask_binary = (mask > 0).astype(np.float32)
                                            intersection = np.sum(attention_binary * mask_binary)
                                            mask_area = np.sum(mask_binary)
                                            
                                            # 3. 计算重叠率（交集除以掩码面积）
                                            if mask_area > 0:
                                                overlay_ratio = intersection / mask_area
                                                
                                                # 可视化注意力图和掩码重叠区域（仅调试使用）
                                                if self.opt.get('val', {}).get('save_debug_overlap', False):
                                                    overlap_debug = np.zeros((h, w, 3), dtype=np.uint8)
                                                    # 蓝色表示掩码
                                                    overlap_debug[mask_binary > 0] = [255, 0, 0]
                                                    # 绿色表示注意力高区域
                                                    overlap_debug[attention_binary > 0] = [0, 255, 0]
                                                    # 红色表示重叠区域
                                                    overlap_debug[(mask_binary > 0) & (attention_binary > 0)] = [0, 0, 255]
                                                    
                                                    # 保存可视化图像
                                                    debug_path = osp.join(save_path_root, f'{img_name}_overlap_debug_{current_iter}.png')
                                                    imwrite(overlap_debug, debug_path)
                            except Exception as e:
                                print(f"对象信息处理错误: {e}")
                        
                        h, w = guided_img.shape[:2]
                        # 将所有图像调整到相同大小以便对比显示
                        lq_img = cv2.resize(lq_img, (w, h), interpolation=cv2.INTER_NEAREST)
                        gt_with_mask = cv2.resize(gt_with_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        # 创建四列对比图（LQ、无引导、有引导、GT+掩码）
                        comparison = np.zeros((h, w*4, 3), dtype=np.uint8)
                        comparison[:, :w] = lq_img        # 第一列显示LQ
                        comparison[:, w:2*w] = unguided_img  # 第二列显示无引导结果
                        comparison[:, 2*w:3*w] = guided_img    # 第三列显示引导结果
                        comparison[:, 3*w:] = gt_with_mask      # 第四列显示GT+掩码
                        
                        # 添加分割线
                        comparison[:, w-1:w+1] = [0, 0, 255]  # 红色分割线
                        comparison[:, 2*w-1:2*w+1] = [0, 0, 255]  # 红色分割线
                        comparison[:, 3*w-1:3*w+1] = [0, 0, 255]  # 红色分割线
                        
                        # 将重叠率添加到文件名中
                        overlap_text = f"_overlap{overlay_ratio:.2f}"
                        save_comp_path = osp.join(save_path_root, f'{img_name}_comparison{overlap_text}_{current_iter}.png')
                        imwrite(comparison, save_comp_path)
            
            # 计算指标
            if with_metrics and 'gt' in val_data:
                gt_img = tensor2img([visuals['gt']])
                metric_data = {'img': sr_img_original, 'img2': gt_img}
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
        """优化参数，整合SR网络和文本引导网络"""
        # 保存当前迭代次数
        self.iter = current_iter
        
        # 检查是否处于阶段转换点，如果是，重置判别器
        if self.opt['train'].get('stage_train', False) and hasattr(self.opt['train'], 'stages') and len(self.opt['train']['stages']) > 1:
            stage_iters = [0]
            for i, stage in enumerate(self.opt['train']['stages']):
                if i > 0:
                    stage_iters.append(stage_iters[i-1] + stage['iters'])
            
            # 如果当前迭代次数是某个阶段的开始(允许有1次误差)
            if current_iter in stage_iters or current_iter - 1 in stage_iters:
                # 获取当前所处阶段
                current_stage_idx = 0
                for i, iter_point in enumerate(stage_iters[1:], 1):
                    if current_iter >= iter_point - 1:
                        current_stage_idx = i
                
                # 只在进入联合训练阶段时重置判别器
                if current_stage_idx > 0 and "warmup" in self.opt['train']['stages'][current_stage_idx-1]['name']:
                    self.logger.info(f'在阶段 {current_stage_idx} 开始时重置判别器参数')
                    # 使用Kaiming初始化重置判别器
                    for m in self.net_d.modules():
                        if isinstance(m, nn.Conv2d):
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.BatchNorm2d):
                            nn.init.constant_(m.weight, 1)
                            nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.Linear):
                            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0)
                    
                    # 重新创建判别器优化器
                    if hasattr(self, 'optimizer_d'):
                        optim_d_config = self.opt['train']['optim_d'].copy()
                        optim_type = optim_d_config.pop('type')
                        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **optim_d_config)
                        # 更新优化器列表
                        if len(self.optimizers) > 1:
                            self.optimizers[1] = self.optimizer_d
        
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

        # 清零梯度
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

        # 文本控制网络损失
        if hasattr(self, 'cri_control') and self.use_text_features and hasattr(self, 'attention_maps'):
            # 获取文本特征
            text_hidden, text_pooled = self.encode_text(self.text_prompts)
            
            # 获取控制特征和注意力图 - 在前向传播中已生成
            attention_maps = self.attention_maps
            
            # 获取控制特征 - 这里简化为使用第一个特征，在实际应用中可能需要更复杂的逻辑
            if hasattr(self, 'net_g') and hasattr(self.net_g, 'conv_first'):
                with torch.no_grad():
                    control_features = self.net_g.conv_first(self.lq)
            
            # 计算控制特征损失
            objects_info = self.objects_info if hasattr(self, 'objects_info') else None
            l_control = self.cri_control(control_features, attention_maps, text_pooled, objects_info, self.device)
            l_g_total += l_control
            loss_dict['l_control'] = l_control

        # 如果有文本引导网络，保留计算图以便后续的反向传播
        retain_graph = self.use_text_features and hasattr(self, 'net_t') and hasattr(self, 'optimizer_t')
        l_g_total.backward(retain_graph=retain_graph)
        
        # 文本引导网络损失 - 原有的语义一致性损失等
        l_t_total = 0
        if self.use_text_features and hasattr(self, 'net_t') and hasattr(self, 'optimizer_t'):
            # 获取文本特征和对象信息
            text_hidden, text_pooled = self.encode_text(self.text_prompts)
            objects_info = None
            if hasattr(self, 'objects_info'):
                objects_info = self.objects_info
            
            # ControlNet架构不再使用以下缓存
            # 1. 注意力损失
            if hasattr(self, 'cri_attention') and hasattr(self, 'attention_maps') and objects_info is not None:
                # 计算注意力损失
                attn_logits = self.attention_maps[-1]
                
                # 从对象信息中创建掩码
                batch_size = attn_logits.shape[0]
                h, w = attn_logits.shape[-2:]
                text_masks = torch.zeros((batch_size, 1, h, w), device=self.device)
                
                # 为每个样本创建掩码
                for i in range(batch_size):
                    if i >= len(objects_info) or not objects_info[i]:
                        continue
                        
                    objects = objects_info[i]
                    # 合并对象掩码为单个文本掩码
                    for obj in objects:
                        if 'mask_encoded' in obj:
                            try:
                                # 解码掩码并调整大小
                                from tgsr.losses.tgsr_loss import decode_mask
                                obj_mask = decode_mask(obj['mask_encoded'])
                                if obj_mask is not None and obj_mask.sum() > 0:
                                    # 调整掩码大小并合并
                                    obj_mask_tensor = torch.from_numpy(obj_mask).float().to(self.device)
                                    obj_mask_tensor = F.interpolate(
                                        obj_mask_tensor.unsqueeze(0).unsqueeze(0), 
                                        size=(h, w), 
                                        mode='bilinear', 
                                        align_corners=False
                                    )
                                    text_masks[i] = torch.max(text_masks[i], obj_mask_tensor[0])
                            except Exception:
                                continue
                
                # 使用正确的参数调用attention损失函数
                l_t_attn = self.cri_attention(attn_logits, text_masks)
                l_t_total += l_t_attn
                loss_dict['l_t_attn'] = l_t_attn
            
            # 特征平滑性损失 - 防止特征过于剧烈变化
            if hasattr(self, 'attention_maps') and len(self.attention_maps) > 0:
                # 计算空间平滑性
                attention = torch.sigmoid(self.attention_maps[-1])
                
                # 水平和垂直梯度
                h_grad = torch.abs(attention[:, :, :, :-1] - attention[:, :, :, 1:]).mean()
                v_grad = torch.abs(attention[:, :, :-1, :] - attention[:, :, 1:, :]).mean()
                
                # 总梯度惩罚
                smoothness = (h_grad + v_grad) * 0.5
                smoothness_weight = 0.1  # 可调
                
                l_t_smooth = smoothness * smoothness_weight
                l_t_total += l_t_smooth
                loss_dict['l_t_smooth'] = l_t_smooth
            
            # 记录文本引导损失总和并反向传播
            if l_t_total > 0:
                loss_dict['l_t_total'] = l_t_total
                l_t_total.backward()
                
                # 添加梯度裁剪，放在optimizer_t.step()前
                torch.nn.utils.clip_grad_norm_(self.net_t.parameters(), max_norm=1.0)
        
        # 执行优化器步骤
        self.optimizer_g.step()
        if self.use_text_features and hasattr(self, 'net_t') and hasattr(self, 'optimizer_t'):
            self.optimizer_t.step()
        
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
    
    def get_current_attention_maps(self):
        """获取当前注意力图，用于可视化"""
        if self.attention_maps is None or len(self.attention_maps) == 0:
            return None
        
        attention_dict = {}
        for i, attn_logits in enumerate(self.attention_maps):
            # 提取第一个样本的注意力图并应用sigmoid将logits转换为概率
            attention_2d = torch.sigmoid(attn_logits[0, 0]).cpu().detach()
            
            # 归一化 - 改进的对比度增强方法
            min_val = attention_2d.min()
            max_val = attention_2d.max()
            
            # 如果差距太小，强制拉开差距
            if max_val - min_val < 0.3:
                # 使用更强的对比度增强
                mean_val = attention_2d.mean()
                std_val = attention_2d.std()
                # 提高标准差以增加对比度
                normalized_attn = torch.clamp((attention_2d - mean_val) / (std_val * 2 + 1e-8) + 0.5, 0, 1)
            else:
                # 标准归一化
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
        # 保存训练状态
        self.save_training_state(epoch, current_iter) 
    
    def save_gradcam_attention(self, img, attention_maps, img_name, save_path, iter_num):
        """将所有注意力图以GradCAM风格叠加在原图上，并添加颜色图例
        
        Args:
            img: 原始图像，BGR格式的numpy数组，shape (H, W, 3)
            attention_maps: 注意力图字典，每个元素是一个shape为(H, W)的numpy数组
            img_name: 图像名称，用于保存文件
            save_path: 保存路径
            iter_num: 当前迭代次数
        """
        if len(attention_maps) == 0:
            return
        
        # 创建输入图像的副本，避免修改原始图像
        img_copy = img.copy()
        
        # 1. 合并所有注意力图（使用平均值）
        all_maps = np.stack([attn for attn in attention_maps.values()])
        combined_map = np.mean(all_maps, axis=0)
        
        # 确保尺寸匹配
        combined_map = cv2.resize(combined_map, (img_copy.shape[1], img_copy.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 应用非线性变换增强对比度 - 使用自适应阈值化
        # 1. 计算均值和标准差
        mean_val = np.mean(combined_map)
        std_val = np.std(combined_map)
        
        # 2. 根据均值和标准差设置阈值，强化高注意力区域，抑制低注意力区域
        enhanced_map = np.zeros_like(combined_map)
        high_threshold = mean_val + 0.5 * std_val
        low_threshold = mean_val - 0.5 * std_val
        
        # 高于高阈值的区域
        high_mask = combined_map > high_threshold
        # 低于低阈值的区域
        low_mask = combined_map < low_threshold
        # 中间区域
        mid_mask = ~(high_mask | low_mask)
        
        # 高注意力区域增强
        enhanced_map[high_mask] = 0.75 + 0.25 * (combined_map[high_mask] - high_threshold) / (1 - high_threshold + 1e-8)
        # 低注意力区域降低
        enhanced_map[low_mask] = 0.25 * combined_map[low_mask] / (low_threshold + 1e-8)
        # 中间区域线性映射
        if np.any(mid_mask):
            enhanced_map[mid_mask] = 0.25 + 0.5 * (combined_map[mid_mask] - low_threshold) / (high_threshold - low_threshold + 1e-8)
        
        # 确保值范围在[0,1]内
        enhanced_map = np.clip(enhanced_map, 0, 1)
        
        # 2. 转换为热力图
        heatmap = cv2.applyColorMap((enhanced_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 3. 叠加到原图（透明度0.4）- 使用副本避免修改原始图像
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB) if img_copy.shape[2] == 3 else img_copy.copy()
        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
        
        # 4. 添加颜色图例 - 创建新的图例图像，不修改原始输入
        h, w = overlay.shape[:2]
        # 图例高度和宽度
        legend_h, legend_w = 30, w
        legend = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
        
        # 创建从蓝到红的渐变色条
        for i in range(legend_w):
            ratio = i / (legend_w - 1)
            # 颜色映射 - 使用相同的JET色彩映射
            if ratio < 0.25:  # 蓝到青
                b = 255
                g = int(255 * ratio * 4)
                r = 0
            elif ratio < 0.5:  # 青到绿
                b = int(255 * (0.5 - ratio) * 4)
                g = 255
                r = 0
            elif ratio < 0.75:  # 绿到黄
                b = 0
                g = 255
                r = int(255 * (ratio - 0.5) * 4)
            else:  # 黄到红
                b = 0
                g = int(255 * (1.0 - ratio) * 4)
                r = 255
                
            legend[:, i] = [r, g, b]
        
        # 添加文字标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        cv2.putText(legend, 'Low', (5, 20), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(legend, 'High', (legend_w - 45, 20), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 5. 将图例添加到叠加图像下方
        result = np.vstack([overlay, legend])
        
        # 6. 添加文本描述（如果有）
        if hasattr(self, 'text_prompts') and len(self.text_prompts) > 0:
            text = self.text_prompts[0]
            if text:
                # 创建文本区域
                text_h = 50  # 略微增加高度以容纳更多文本
                text_area = np.ones((text_h, w, 3), dtype=np.uint8) * 240  # 淡灰色背景
                
                # 修改：减小字体大小
                font_scale_text = 0.4  # 从0.5减小到0.4
                line_height = 16  # 行高从20减小到16
                
                # 将长文本分成多行
                max_chars = w // 8  # 估计每个字符的宽度 (从10改为8，每行可容纳更多字符)
                if len(text) > max_chars:
                    words = text.split()
                    lines = []
                    current_line = words[0]
                    for word in words[1:]:
                        if len(current_line) + len(word) + 1 <= max_chars:
                            current_line += " " + word
                        else:
                            lines.append(current_line)
                            current_line = word
                    lines.append(current_line)
                    
                    # 如果有多行，调整文本区域的高度
                    if len(lines) > 1:
                        text_h = min(len(lines) * line_height + 10, 80)  # 限制最高80像素
                        text_area = np.ones((text_h, w, 3), dtype=np.uint8) * 240
                    
                    # 添加每行文本
                    for i, line in enumerate(lines):
                        y_pos = line_height + i * line_height
                        if y_pos < text_h - 5:
                            cv2.putText(text_area, line, (5, y_pos), font, font_scale_text, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    # 单行文本
                    cv2.putText(text_area, text, (5, line_height), font, font_scale_text, (0, 0, 0), 1, cv2.LINE_AA)
                
                # 添加文本区域到结果图像
                result = np.vstack([text_area, result])
        
        # 7. 保存结果
        save_path = osp.join(save_path, f'{img_name}_gradcam_{iter_num}.png')
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        return save_path