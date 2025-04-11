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

            # 随机裁剪
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                self.opt['scale'])
            
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
        """SR网络的前向传播，在关键位置应用文本引导
        
        Args:
            x: 输入的低质量图像
            apply_guidance: 是否应用文本引导
            
        Returns:
            sr: 超分辨率输出
        """
        # 编码文本（如果需要）
        if self.use_text_features and apply_guidance and hasattr(self, 'net_t'):
            with torch.no_grad() if self.freeze_text_encoder else torch.enable_grad():
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
            
            # 改进：自适应分层引导策略 - 更多引导位置及动态引导强度
            if num_blocks >= 20:  # 模型较大时使用5个引导点
                # 设置引导位置 - 在不同层次均匀分布
                guidance_positions = [
                    num_blocks // 8,             # 浅层 - 捕获边缘和纹理
                    num_blocks // 4,             # 浅中层 - 捕获局部特征
                    num_blocks // 2,             # 中层 - 捕获中等规模特征
                    num_blocks * 3 // 4,         # 中深层 - 捕获对象部件
                    num_blocks - 2               # 深层 - 捕获整体语义
                ]
                
                # 为不同层级设置引导强度 - 中间层级给予更高的权重
                guidance_weights = {
                    num_blocks // 8: 0.6,        # 浅层权重
                    num_blocks // 4: 0.8,        # 浅中层
                    num_blocks // 2: 1.0,        # 中层使用最高权重
                    num_blocks * 3 // 4: 0.8,    # 中深层
                    num_blocks - 2: 0.6          # 深层使用较低权重
                }
            
            # 验证时使用所有引导位置，训练时可选随机丢弃部分引导位置
            if self.is_train and random.random() < 0.3:  # 30%概率随机丢弃部分引导位置
                # 至少保留3个引导位置
                keep_count = max(3, len(guidance_positions) - random.randint(1, 2))
                selected_positions = sorted(random.sample(guidance_positions, keep_count))
                guidance_positions = selected_positions
            
            # 缓存特征用于损失计算
            self.original_features_cache = {}
            self.enhanced_features_cache = {}
            
            # 防止过拟合和注意力崩塌的随机屏蔽策略
            apply_masking = self.is_train and random.random() < 0.3  # 30%概率在训练时应用
            
            for i, block in enumerate(self.net_g.body):
                trunk = block(trunk)
                
                # 在指定位置应用文本引导
                if i in guidance_positions:
                    # 缓存原始特征用于损失计算
                    if self.is_train:
                        self.original_features_cache[i] = trunk.clone()
                    
                    # 随机特征掩码 - 防止模型过度依赖特定区域
                    if apply_masking:
                        b, c, h, w = trunk.shape
                        mask = torch.rand(b, 1, h, w, device=trunk.device) > 0.2
                        mask = mask.float().expand_as(trunk)
                        
                        # 应用掩码，保留80%的特征，其余置为均值
                        mean_val = trunk.mean(dim=[2, 3], keepdim=True)
                        trunk = trunk * mask + mean_val * (1 - mask)
                    
                    # 文本引导
                    enhanced_features, attn_maps = self.apply_text_guidance(trunk, text_hidden, text_pooled, block_idx=i)
                    
                    # 新增：应用引导强度权重
                    weight = guidance_weights.get(i, 1.0)
                    # 加权融合：原始特征 + 权重 * 增强特征
                    trunk = trunk + weight * (enhanced_features - trunk)
                    
                    # 缓存增强后的特征
                    if self.is_train:
                        self.enhanced_features_cache[i] = trunk.clone()
                    
                    # 保存注意力图
                    if attn_maps is not None and (not self.is_train or getattr(self, 'save_attention', False)):
                        if isinstance(attn_maps, list):
                            processed_attn_maps = [torch.sigmoid(logits) for logits in attn_maps if logits is not None]
                            attention_maps.extend(processed_attn_maps)
                        else:
                            processed_attn = torch.sigmoid(attn_maps)
                            attention_maps.append(processed_attn)
                    
                    # 随机重置特征 - 防止过度依赖注意力
                    if self.is_train and random.random() < 0.1:  # 10%概率在训练时应用
                        b, c, h, w = trunk.shape
                        reset_mask = torch.rand(b, 1, h, w, device=trunk.device) < 0.2
                        reset_mask = reset_mask.float().expand_as(trunk)
                        
                        # 保存原始特征用于重置
                        orig_feat = self.original_features_cache[i]
                        
                        # 部分重置为原始特征
                        trunk = trunk * (1 - reset_mask) + orig_feat * reset_mask
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
            
            # 设置回引导结果用于指标计算
            self.output = results['guided']
            
            # 计算指标和保存图像
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            
            # 只保存前20张图片的结果
            save_this_img = save_img and idx < 20
                
            # 保存注意力图（如果有）
            if save_this_img and self.opt.get('val', {}).get('save_attention_maps', False) and self.attention_maps:
                attention_maps = self.get_current_attention_maps()
                if attention_maps:
                    # 使用GradCAM风格保存合并后的注意力图
                    self.save_gradcam_attention(sr_img, attention_maps, img_name, save_path_root, current_iter)
            
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
                        # 创建三列对比图（LQ、无引导、有引导）
                        guided_img = sr_img
                        lq_img = tensor2img([self.lq])
                        
                        h, w = guided_img.shape[:2]
                        # 将lq调整到相同大小以便对比显示
                        lq_img = cv2.resize(lq_img, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
                        comparison[:, :w] = lq_img        # 第一列显示LQ
                        comparison[:, w:2*w] = unguided_img  # 第二列显示无引导结果
                        comparison[:, 2*w:] = guided_img    # 第三列显示引导结果
                        
                        # 添加分割线
                        comparison[:, w-1:w+1] = [0, 0, 255]  # 红色分割线
                        comparison[:, 2*w-1:2*w+1] = [0, 0, 255]  # 红色分割线
                        
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
                
        # 新增：动态调整注意力温度 - 从高温到低温
        # 高温使得注意力分布更均匀，低温使得注意力更加集中
        if hasattr(self, 'net_t'):
            total_iters = self.opt.get('train', {}).get('total_iter', 400000)
            # 温度从1.5逐渐降低到0.5
            temp = max(0.5, 1.5 - current_iter / total_iters)
            
            # 为每个带有temperature参数的模块设置温度
            for module in self.net_t.modules():
                if hasattr(module, 'temperature'):
                    module.temperature = temp
        
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
                        # 使用最后一个注意力图（通常最精细）- 现在是logits形式
                        attn_logits = attn_maps[-1]
                        
                        # 使用逐样本处理方式处理objects_info
                        batch_size = attn_logits.size(0)
                        l_t_attn_batch = 0
                        valid_samples = 0
                        
                        for i in range(batch_size):
                            # 获取当前样本的文本提示和对象信息
                            sample_text = [self.text_prompts[i]]
                            sample_objects = [objects_info[i]] if i < len(objects_info) else [[]]
                            sample_attn_logits = attn_logits[i:i+1]
                            
                            # 对单个样本进行处理 - 传入logits而非sigmoid后的结果
                            try:
                                # 注意：cri_attention现在接收logits
                                sample_loss = self.cri_attention(sample_attn_logits, sample_text, sample_objects, self.device)
                                
                                # 只有当样本损失有效时才累加
                                if sample_loss.item() > 0:
                                    l_t_attn_batch += sample_loss
                                    valid_samples += 1
                            except Exception as e:
                                self.logger.warning(f"注意力损失计算失败: {e}")
                                continue
                        
                        # 计算批次平均损失
                        if valid_samples > 0:
                            l_t_attn = l_t_attn_batch / valid_samples
                            l_t_attn = l_t_attn / self.accumulate_grad_batches  # 归一化
                            l_t_total += l_t_attn
                            loss_dict[f'l_t_attn_{pos}'] = l_t_attn
                            
                            # 新增：如果注意力损失太大，可能表明已经崩塌，增加熵正则化权重
                            if l_t_attn.item() > 5.0 and hasattr(self.cri_attention, 'entropy_weight'):
                                # 动态增加熵权重，鼓励更均匀的注意力
                                self.cri_attention.entropy_weight = min(0.3, self.cri_attention.entropy_weight * 1.2)
                                self.logger.info(f"检测到高注意力损失，增加熵权重至:{self.cri_attention.entropy_weight:.4f}")
            
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
                
                # 新增：梯度裁剪，防止梯度爆炸
                if self.is_train:
                    torch.nn.utils.clip_grad_norm_(self.net_t.parameters(), max_norm=1.0)
        
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
        # 保存判别器
        self.save_network(self.net_d, 'net_d', current_iter)
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
        
        # 1. 合并所有注意力图（使用平均值）
        all_maps = np.stack([attn for attn in attention_maps.values()])
        combined_map = np.mean(all_maps, axis=0)
        
        # 确保尺寸匹配
        combined_map = cv2.resize(combined_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
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
        
        # 3. 叠加到原图（透明度0.4）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img.copy()
        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
        
        # 4. 添加颜色图例
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