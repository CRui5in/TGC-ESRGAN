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
class TGCESRModel(SRGANModel):
    """文本引导的超分辨率模型

    基于SRGANModel，但增加了独立的文本引导网络，可以处理文本描述，
    使用CLIP作为文本编码器，并在SR过程中的关键位置引入文本引导。
    """

    def __init__(self, opt):
        self.use_text_features = True
        super(TGCESRModel, self).__init__(opt)
        
        # 初始化日志记录器
        self.logger = get_root_logger()
        
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        self.queue_size = opt.get('queue_size', 180)
        
        if self.use_text_features:
            self.init_text_encoder()
            self.init_text_guidance_losses()
        
        self.attention_maps = None
        
        self.control_map = None
    
    def init_training_settings(self):
        """重写init_training_settings，确保文本引导网络在setup_optimizers之前被初始化"""
        # 先初始化文本引导网络（如果使用）
        if self.is_train and self.use_text_features:
            self.init_text_guidance_net()
        
        # 调用父类的初始化训练设置
        super(TGCESRModel, self).init_training_settings()
        
        logger = get_root_logger()
        
        # 初始化CLIP语义损失
        train_opt = self.opt['train']
        if self.is_train and 'clip_opt' in train_opt:
            self.cri_clip = build_loss(train_opt['clip_opt']).to(self.device)
            logger.info(f'初始化CLIP语义损失: {train_opt["clip_opt"]["type"]}')
    
    def init_text_encoder(self):
        """初始化文本编码器（CLIP）"""
        logger = get_root_logger()
        
        text_encoder_opt = self.opt.get('text_encoder', {})
        self.text_encoder_name = text_encoder_opt.get('name')
        self.text_dim = text_encoder_opt.get('text_dim', 512)
        self.freeze_text_encoder = text_encoder_opt.get('freeze')
        
        # 加载CLIP文本编码器
        if self.text_encoder_name:
            try:
                logger.info(f'加载文本编码器: {self.text_encoder_name}')
                self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder_name)
                self.clip_text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name)
                
                if self.freeze_text_encoder:
                    logger.info('冻结文本编码器参数')
                    for param in self.clip_text_encoder.parameters():
                        param.requires_grad = False
                    self.clip_text_encoder.eval()
                
                self.clip_text_encoder = self.clip_text_encoder.to(self.device)
                logger.info(f'成功加载文本编码器: {self.text_encoder_name}')
            except Exception as e:
                logger.error(f'加载文本编码器失败: {e}')
                self.clip_tokenizer = None
                self.clip_text_encoder = None
                self.use_text_features = False
    
    def init_text_guidance_net(self):
        """初始化ControlNet"""
        if not self.use_text_features:
            return
            
        logger = get_root_logger()
        
        try:
            if 'network_control' not in self.opt:
                logger.error("配置中缺少network_control参数")
                raise ValueError("配置中缺少network_control参数")
            
            # 初始化ControlNet
            logger.info("使用ControlNet架构进行训练")
            # 创建ControlNet，传入SR网络的副本
            self.opt['network_control']['orig_net_g'] = self.net_g  # 传递原始网络给ControlNet
            self.net_control = build_network(self.opt['network_control'])
            
            # 设置参数为可训练
            for param in self.net_control.parameters():
                param.requires_grad = True
            
            # 将SR网络设置为评估模式并冻结参数（标准ControlNet要求）
            if self.opt.get('freeze_original'):
                logger.info("已冻结原始SR网络参数")
                self.net_g.eval()
                for param in self.net_g.parameters():
                    param.requires_grad = False
            else:
                logger.warning("原始SR网络未冻结，这不是标准ControlNet实现")
            
            self.net_control = self.model_to_device(self.net_control)
            
            self.print_network(self.net_control)
            
            if hasattr(self.net_control, 'load_pretrained_controlnet'):
                canny_path = self.opt.get('pretrain_canny_model')
                depth_path = self.opt.get('pretrain_depth_model')
                
                if canny_path or depth_path:
                    logger.info(f"加载预训练ControlNet模型 - Canny: {canny_path}, Depth: {depth_path}")
                    self.net_control.load_pretrained_controlnet(canny_path, depth_path)
            
            self.net_control.train()
            logger.info("ControlNet初始化成功！")
            
            return hasattr(self, 'net_control')
            
        except Exception as e:
            logger.error(f"初始化ControlNet失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def init_text_guidance_losses(self):
        """初始化ControlNet的特定损失函数"""
        logger = get_root_logger()
        train_opt = self.opt['train']
        
        # 特征投影层，用于投影特征到文本空间维度
        if hasattr(self, 'text_dim'):
            feat_dim = train_opt.get('feat_dim', 64)
            if feat_dim != self.text_dim:
                self.feat_proj = nn.Linear(feat_dim, self.text_dim).to(self.device)
                logger.info(f'创建特征投影层: {feat_dim} -> {self.text_dim}')

    def setup_optimizers(self):
        """设置优化器，支持ControlNet训练模式"""
        logger = get_root_logger()
        train_opt = self.opt['train']
        
        # 获取初始学习率（从第一个阶段或默认值）
        init_lr_g = train_opt['stages'][0]['lr_g'] if train_opt.get('stage_train') and 'stages' in train_opt else 1e-4
        init_lr_d = train_opt['stages'][0]['lr_d'] if train_opt.get('stage_train') and 'stages' in train_opt else 1e-4
        init_lr_control = train_opt['stages'][0].get('lr_control', 1e-4) if train_opt.get('stage_train') and 'stages' in train_opt else 1e-4
        
        # 检查是否使用ControlNet模式
        use_controlnet = hasattr(self, 'net_control')
        
        # 清空优化器列表，避免重复添加
        self.optimizers = []
        
        # 根据配置选择是否冻结SR网络
        freeze_original = self.opt.get('freeze_original', True)
        if freeze_original:
            logger.info("根据配置冻结原始SR网络")
            self.net_g.eval()
            for param in self.net_g.parameters():
                param.requires_grad = False
        
        # ==================== 优化器D - 判别器 ====================
        logger.info("设置判别器优化器")
        optim_d_config = train_opt['optim_d'].copy()
        optim_type = optim_d_config.pop('type')
        # 从配置中移除lr字段，使用阶段学习率
        if 'lr' in optim_d_config:
            optim_d_config.pop('lr')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), lr=init_lr_d, **optim_d_config)
        
        # 添加这一行，确保每个参数组都有initial_lr参数
        for param_group in self.optimizer_d.param_groups:
            param_group['initial_lr'] = param_group['lr']
        
        # 更新优化器列表
        self.optimizers.append(self.optimizer_d)
        
        # ==================== 优化器Control - ControlNet ====================
        # ControlNet优化器
        if use_controlnet:
            logger.info("设置ControlNet优化器")
            # 为ControlNet创建独立优化器
            if 'optim_control' in train_opt:
                optim_config = train_opt['optim_control'].copy()
                optim_type = optim_config.pop('type')
                
                # 从配置中移除lr字段，使用阶段学习率
                if 'lr' in optim_config:
                    optim_config.pop('lr')
                
                # 收集ControlNet参数
                optim_params_control = []
                for k, v in self.net_control.named_parameters():
                    if v.requires_grad:
                        optim_params_control.append(v)
                
                # 文本编码器参数（如果未冻结）
                if hasattr(self, 'clip_text_encoder') and not self.freeze_text_encoder:
                    for k, v in self.clip_text_encoder.named_parameters():
                        if v.requires_grad:
                            optim_params_control.append(v)
                
                # 保存配置
                self._optim_control_config = {
                    'type': optim_type,
                    **optim_config
                }
                
                if optim_params_control:
                    self.optimizer_control = self.get_optimizer(optim_type, optim_params_control, lr=init_lr_control, **optim_config)
                    # 确保每个参数组都有initial_lr参数
                    for param_group in self.optimizer_control.param_groups:
                        param_group['initial_lr'] = param_group['lr']
                    self.optimizers.append(self.optimizer_control)
                    
        # 记录优化器设置完成
        logger.info(f"优化器设置完成，共有 {len(self.optimizers)} 个优化器")

    def optimize_parameters(self, current_iter):
        """优化参数，整合SR网络和ControlNet"""
        logger = get_root_logger()
        # 保存当前迭代次数
        self.iter = current_iter
        
        # 检查使用的架构类型
        use_controlnet = hasattr(self, 'net_control') and hasattr(self, 'optimizer_control')
        
        # 检查是否处于阶段转换点
        if self.opt['train'].get('stage_train'):
            self.current_stage_idx = 0
            
            # 计算每个阶段的起始点
            accumulated_iters = 0
            for idx, stage in enumerate(self.opt['train']['stages']):
                start_iter = accumulated_iters
                accumulated_iters += stage['iters']
                end_iter = accumulated_iters
                
                # 如果当前迭代在这个阶段的范围内，设置当前阶段索引
                if start_iter <= current_iter < end_iter:
                    self.current_stage_idx = idx
                    
                # 检查是否精确处于阶段切换点（前一阶段的结束=当前阶段的开始）
                if idx > 0 and current_iter == start_iter:
                    logger.info(f'进入阶段 {stage["name"]} 在迭代 {current_iter}')
                    
                    # 设置学习率 - 只设置判别器和ControlNet的学习率
                    for param_group in self.optimizer_d.param_groups:
                        param_group['lr'] = stage['lr_d']
                        param_group['initial_lr'] = stage['lr_d']
                    
                    if use_controlnet:
                        lr_control = stage.get('lr_control')
                        for param_group in self.optimizer_control.param_groups:
                            param_group['lr'] = lr_control
                            param_group['initial_lr'] = lr_control
                    
                    # 对于特定阶段的特殊处理
                    if idx == 1:  # 从预训练到联合轻度训练阶段
                        # 重新初始化判别器的某些层参数，避免梯度爆炸
                        for name, m in self.net_d.named_modules():
                            if isinstance(m, nn.Conv2d) and 'final' not in name:
                                nn.init.normal_(m.weight, 0, 0.02)
        
        # 定期清理CUDA缓存以减少内存碎片
        if current_iter % 50 == 0:
            torch.cuda.empty_cache()
            
        # 设置GT图像（带锐化或不带）
        l1_gt = self.gt_usm if self.opt.get('l1_gt_usm') else self.gt
        percep_gt = self.gt_usm if self.opt.get('percep_gt_usm') else self.gt
        gan_gt = self.gt_usm if self.opt.get('gan_gt_usm') else self.gt
        
        # ====================== 第1部分：优化生成器 ======================
        # 关闭判别器梯度
        for p in self.net_d.parameters():
            p.requires_grad = False

        # 清零梯度 - 只清除ControlNet的梯度
        if use_controlnet:
            self.optimizer_control.zero_grad()
        
        # 前向传播，应用文本引导 - 确保output依赖于control_signals以维持梯度流
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
        
        # 调整GAN损失权重
        gan_weight = self.opt.get('train', {}).get('gan_weight', 0.1)
        l_g_total += l_g_gan * gan_weight
        loss_dict['l_g_gan'] = l_g_gan
        loss_dict['l_g_gan_weighted'] = l_g_gan * gan_weight  # 记录加权后的损失值

        # 新增：CLIP语义损失 - 确保生成图像与文本描述的语义一致性
        if hasattr(self, 'cri_clip') and self.use_text_features:
            l_clip = self.cri_clip(self.output, self.text_prompts)
            
            # 调整CLIP损失权重
            clip_weight = self.opt.get('train', {}).get('clip_weight', 0.3)
            l_g_total += l_clip * clip_weight
            loss_dict['l_clip'] = l_clip
            loss_dict['l_clip_weighted'] = l_clip * clip_weight  # 记录加权后的损失值
            
            # 动态调整CLIP权重 - 防止在训练早期占比过大
            if hasattr(self, 'iter'):
                current_iter = self.iter
                if current_iter < 20000:  # 前20k次迭代降低CLIP损失权重
                    clip_scale = current_iter / 20000.0  # 从0逐渐增加到1.0
                    l_g_total -= l_clip * clip_weight * (1.0 - clip_scale)  # 减少一部分权重
                    loss_dict['clip_scale'] = clip_scale
        
        # 反向传播
        if l_g_total.requires_grad:
            l_g_total.backward()
            
            # 检查net_control参数是否有梯度
            if use_controlnet:
                has_grad = False
                for name, param in self.net_control.named_parameters():
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        has_grad = True
                        break
            
            # 梯度裁剪，防止梯度爆炸
            if use_controlnet:
                torch.nn.utils.clip_grad_norm_(self.net_control.parameters(), max_norm=10.0)
                
            # 更新ControlNet参数
            if use_controlnet:
                self.optimizer_control.step()
        else:
            logger.error("生成器总损失没有梯度，跳过反向传播")
        
        # 清理GPU内存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # ====================== 第3部分：判别器更新 ======================
        # 判别器跳过条件 - 可以根据阶段自定义
        skip_d_update = False
        current_stage_idx = 0
        
        # 获取当前所处阶段
        if self.opt['train'].get('stage_train'):
            stages = self.opt['train']['stages']
            for idx, stage in enumerate(stages):
                if current_iter < stage['iters'] or idx == len(stages) - 1 and current_iter >= stage['iters']:
                    current_stage_idx = idx
                    break
        
        # 在预训练阶段可能需要跳过判别器更新
        if current_stage_idx == 0 and self.opt['train'].get('stage_train'):
            first_stage = self.opt['train']['stages'][0]
            if first_stage.get('lr_d', 0) == 0:
                skip_d_update = True
        
        # 优化判别器（除非需要跳过）
        if not skip_d_update:
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # 真实样本
            real_d_pred = self.net_d(gan_gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            
            # 虚假样本
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            
            # 可选：添加判别器更新跳过逻辑，当判别器过强时
            fake_score = torch.mean(fake_d_pred.detach()).item()
            real_score = torch.mean(real_d_pred.detach()).item()
            discriminator_too_strong = fake_score < 0.4 and real_score > 0.7
            
            # 如果判别器不是过强，则更新参数
            if not discriminator_too_strong or current_iter % 5 == 0:  # 每5次迭代至少更新一次
                self.optimizer_d.step()
        
        # 记录损失
        self.log_dict = loss_dict
        
        # 更新EMA模型
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
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
            
            # 添加梯度缩放，防止CLIP编码器梯度过大
            if self.is_train and not self.freeze_text_encoder:
                # STE (Straight-Through Estimator) 风格的梯度缩放
                text_hidden = text_hidden.detach() * 0.9 + text_hidden * 0.1
                text_pooled = text_pooled.detach() * 0.9 + text_pooled * 0.1
        
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
        # TODO: 现在是使用的先退化再缩放，和RealESRGAN的先缩放再退化不一样
        # 存储文本提示
        if self.use_text_features and 'text_prompt' in data:
            self.text_prompts = data['text_prompt']
        else:
            self.text_prompts = [''] * len(data['gt'] if 'gt' in data else data['lq'])
        
        # 存储控制映射
        if 'control_map' in data:
            # 临时存储原始控制映射
            raw_control_map = data['control_map'].to(self.device)
            
            # 稍后在处理lq后再调整控制映射尺寸
            self.raw_control_map = raw_control_map
        
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
                logger = get_root_logger() if hasattr(self, 'logger') else None
                if logger:
                    logger.warning(f"解析对象信息失败: {e}")
                self.objects_info = None
        
        if self.is_train:
            # 训练数据合成 - 修改为先退化再缩放
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- 第一阶段退化过程 ----------------------- #
            # 模糊
            out = filter2D(self.gt_usm, self.kernel1)
            
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
  
            # JPEG压缩
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            
            # 最终的sinc滤波
            out = filter2D(out, self.sinc_kernel)
            
            # 在完成所有退化后再进行缩放 - 先退化再缩放
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            
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
            
            # ----------------------- 处理控制映射 ----------------------- #
            # 调整控制映射大小，确保与lq大小一致
            if hasattr(self, 'raw_control_map') and self.raw_control_map is not None:
                lq_h, lq_w = self.lq.shape[2:4]
                raw_h, raw_w = self.raw_control_map.shape[2:4]
                
                if raw_h != lq_h or raw_w != lq_w:
                    # 判断是单通道还是多通道控制映射
                    is_multi_channel = self.raw_control_map.shape[1] > 1
                    
                    # 使用不同的插值模式
                    interp_mode = 'bilinear' if is_multi_channel else 'nearest'
                    align_corners = False if is_multi_channel else None
                    
                    self.control_map = F.interpolate(
                        self.raw_control_map, 
                        size=(lq_h, lq_w), 
                        mode=interp_mode,
                        align_corners=align_corners if interp_mode == 'bilinear' else None
                    )
                else:
                    self.control_map = self.raw_control_map
                
                # 清理临时变量
                del self.raw_control_map
            
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
            
            # ----------------------- 处理控制映射 ----------------------- #
            # 验证/测试模式下调整控制映射大小
            if hasattr(self, 'raw_control_map') and self.raw_control_map is not None:
                lq_h, lq_w = self.lq.shape[2:4]
                raw_h, raw_w = self.raw_control_map.shape[2:4]
                
                if raw_h != lq_h or raw_w != lq_w:
                    # 判断是单通道还是多通道控制映射
                    is_multi_channel = self.raw_control_map.shape[1] > 1
                    
                    # 使用不同的插值模式
                    interp_mode = 'bilinear' if is_multi_channel else 'nearest'
                    align_corners = False if is_multi_channel else None
                    
                    self.control_map = F.interpolate(
                        self.raw_control_map, 
                        size=(lq_h, lq_w), 
                        mode=interp_mode,
                        align_corners=align_corners if interp_mode == 'bilinear' else None
                    )
                else:
                    self.control_map = self.raw_control_map
                
                # 清理临时变量
                del self.raw_control_map

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
        """SR网络的前向传播，使用ControlNet实现文本引导
        
        Args:
            x: 输入的低质量图像 [B, C, H, W]
            apply_guidance: 是否应用文本引导
            
        Returns:
            sr: 超分辨率输出 [B, C, H*4, W*4]
        """
        # 获取文本特征（如果需要）
        text_hidden, text_pooled = None, None
        if self.use_text_features and apply_guidance:
            with torch.no_grad() if self.freeze_text_encoder else torch.enable_grad():
                text_hidden, text_pooled = self.encode_text(self.text_prompts)
 
        # 获取控制映射
        control_map = getattr(self, 'control_map', None)
        if control_map is None and apply_guidance:
            b, c, h, w = x.shape
            control_map = torch.zeros(b, 3, h, w, device=x.device)

        # 确保控制映射与输入图像尺寸一致
        if control_map is not None and (control_map.shape[2] != x.shape[2] or control_map.shape[3] != x.shape[3]):
            # 根据通道数选择合适的插值模式
            interp_mode = 'bilinear' if control_map.shape[1] > 1 else 'nearest'
            align_corners = False if interp_mode == 'bilinear' else None
            
            control_map = F.interpolate(
                control_map,
                size=(x.shape[2], x.shape[3]),
                mode=interp_mode,
                align_corners=align_corners
            )

        use_controlnet = hasattr(self, 'net_control') and apply_guidance and text_hidden is not None
        if not use_controlnet:
            return self.net_g(x)

        # 生成位置编码
        position_info = None
        if hasattr(self.net_control, 'with_position') and self.net_control.with_position:
            num_blocks = len(self.net_g.body)
            position_info = torch.zeros(x.size(0), num_blocks, device=x.device)
            for i in range(num_blocks):
                position_info[:, i] = 1.0 if i % 4 == 0 else 0.0  # 每组RRDB块标记

        # 生成控制信号
        control_signals, attention_maps = self.net_control(x, control_map, text_hidden, text_pooled, position_info)

        # 校验信号数量
        block_groups_count = len(self.net_g.body) // 4 + (1 if len(self.net_g.body) % 4 > 0 else 0)
        expected_signals = block_groups_count + 4  # conv_first, block_groups, conv_body, conv_up1, conv_up2
        
        if len(control_signals) != expected_signals:
            error_msg = f"控制信号数量不正确: 预期{expected_signals}个，实际{len(control_signals)}个"
            raise ValueError(error_msg)

        # 计算自适应控制强度
        if hasattr(self, 'cri_clip') and self.use_text_features and self.is_train:
            # 根据训练阶段自适应调整控制强度
            base_strength = 0.05
            if hasattr(self, 'iter') and hasattr(self, 'opt') and self.opt.get('train', {}).get('stage_train', False):
                current_iter = getattr(self, 'iter', 0)
                # 在早期阶段使用较小的控制强度
                if current_iter < 20000:
                    control_strength = base_strength * 0.5
                # 在中期阶段逐渐增加控制强度
                elif current_iter < 100000:
                    progress = (current_iter - 20000) / 80000
                    control_strength = base_strength * (0.5 + 0.5 * progress)
                # 在后期阶段使用较大的控制强度
                else:
                    control_strength = base_strength * 1.2
            else:
                control_strength = base_strength
        else:
            # 测试/推理模式使用固定控制强度
            control_strength = 0.05

        # 应用控制信号到RRDBNet
        if self.is_train:
            # 训练模式 - 保持梯度流
            # 浅层特征提取 - 锁定参数不需要梯度
            fea = self.net_g.conv_first(x)
            
            signal_idx = 0
            control_signal = control_signals[signal_idx]
            if fea.shape[2:] != control_signal.shape[2:]:
                control_signal = F.interpolate(
                    control_signal,
                    size=(fea.shape[2], fea.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            fea_guided = fea + control_signal
            
            # 分组处理RRDB块
            trunk = fea_guided
            block_groups = []
            total_blocks = len(self.net_g.body)
            group_size = 4
            
            # 处理RRDB块并应用控制信号
            for i in range(0, total_blocks, group_size):
                end = min(i + group_size, total_blocks)
                block_group = self.net_g.body[i:end]
                
                # 使用冻结参数处理RRDB块
                for block in block_group:
                    trunk = block(trunk)
                
                signal_idx += 1
                attn = torch.sigmoid(attention_maps[signal_idx])
                
                control_signal = control_signals[signal_idx]
                if trunk.shape[2:] != control_signal.shape[2:]:
                    control_signal = F.interpolate(
                        control_signal,
                        size=(trunk.shape[2], trunk.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                
                if trunk.shape[2:] != attn.shape[2:]:
                    b, c, _, _ = attn.shape
                    attn = F.interpolate(
                        attn,
                        size=(trunk.shape[2], trunk.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                    if attn.shape[1] != c:
                        attn = attn[:, :c]
                
                trunk = trunk + control_strength * control_signal * attn
            
            # 处理主体输出
            trunk = self.net_g.conv_body(trunk)
            
            # 应用主体输出的控制信号
            signal_idx += 1
            attn = torch.sigmoid(attention_maps[signal_idx])
            
            control_signal = control_signals[signal_idx]
            if trunk.shape[2:] != control_signal.shape[2:]:
                control_signal = F.interpolate(
                    control_signal,
                    size=(trunk.shape[2], trunk.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            
            if trunk.shape[2:] != attn.shape[2:]:
                b, c, _, _ = attn.shape
                attn = F.interpolate(
                    attn,
                    size=(trunk.shape[2], trunk.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
                if attn.shape[1] != c:
                    attn = attn[:, :c]
            
            trunk = trunk + control_strength * control_signal * attn
            
            # 残差连接
            fea = fea_guided + trunk
            
            # 上采样阶段1
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
            fea = self.net_g.conv_up1(fea)
            
            # 应用第一个上采样层的控制信号
            signal_idx += 1
            attn = torch.sigmoid(attention_maps[signal_idx])
            
            control_signal = control_signals[signal_idx]
            if fea.shape[2:] != control_signal.shape[2:]:
                control_signal = F.interpolate(
                    control_signal,
                    size=(fea.shape[2], fea.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            
            # 检查注意力图尺寸是否匹配
            if fea.shape[2:] != attn.shape[2:]:
                b, c, _, _ = attn.shape
                attn = F.interpolate(
                    attn,
                    size=(fea.shape[2], fea.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
                if attn.shape[1] != c:
                    attn = attn[:, :c]
            
            fea = fea + control_strength * 1.5 * control_signal * attn  # 增加上采样层控制强度
            
            # 上采样阶段2
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
            fea = self.net_g.conv_up2(fea)
            
            # 应用第二个上采样层的控制信号
            signal_idx += 1
            attn = torch.sigmoid(attention_maps[signal_idx])
            
            control_signal = control_signals[signal_idx]
            if fea.shape[2:] != control_signal.shape[2:]:
                control_signal = F.interpolate(
                    control_signal,
                    size=(fea.shape[2], fea.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            
            if fea.shape[2:] != attn.shape[2:]:
                b, c, _, _ = attn.shape
                attn = F.interpolate(
                    attn,
                    size=(fea.shape[2], fea.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
                if attn.shape[1] != c:
                    attn = attn[:, :c]
            
            fea = fea + control_strength * 1.5 * control_signal * attn  # 增加上采样层控制强度
            
            # 最终输出层
            out = self.net_g.conv_hr(fea)
            out = self.net_g.conv_last(out)
            
            # 添加小的控制信号连接以确保梯度流
            out = out + 0.0001 * (control_signals[0].sum() + control_signals[-1].sum())
        else:
            # 验证/测试模式 - 不需要保持梯度流
            # 浅层特征提取
            fea = self.net_g.conv_first(x)
            
            # 添加第一个控制信号
            signal_idx = 0
            control_signal = control_signals[signal_idx]
            if fea.shape[2:] != control_signal.shape[2:]:
                control_signal = F.interpolate(
                    control_signal,
                    size=(fea.shape[2], fea.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            fea = fea + control_signal
            
            # 分组处理RRDB块
            trunk = fea
            block_groups = []
            total_blocks = len(self.net_g.body)
            group_size = 4
            
            # 处理RRDB块并应用控制信号
            for i in range(0, total_blocks, group_size):
                end = min(i + group_size, total_blocks)
                block_group = self.net_g.body[i:end]
                
                # 处理RRDB块
                for block in block_group:
                    trunk = block(trunk)
                
                # 应用控制信号
                signal_idx += 1
                attn = torch.sigmoid(attention_maps[signal_idx])
                
                control_signal = control_signals[signal_idx]
                if trunk.shape[2:] != control_signal.shape[2:]:
                    control_signal = F.interpolate(
                        control_signal,
                        size=(trunk.shape[2], trunk.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                
                if trunk.shape[2:] != attn.shape[2:]:
                    b, c, _, _ = attn.shape
                    attn = F.interpolate(
                        attn,
                        size=(trunk.shape[2], trunk.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                    if attn.shape[1] != c:
                        attn = attn[:, :c]
                
                trunk = trunk + control_strength * control_signal * attn
            
            # 处理主体输出
            trunk = self.net_g.conv_body(trunk)
            
            # 应用主体输出的控制信号
            signal_idx += 1
            attn = torch.sigmoid(attention_maps[signal_idx])
            
            control_signal = control_signals[signal_idx]
            if trunk.shape[2:] != control_signal.shape[2:]:
                control_signal = F.interpolate(
                    control_signal,
                    size=(trunk.shape[2], trunk.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            
            if trunk.shape[2:] != attn.shape[2:]:
                b, c, _, _ = attn.shape
                attn = F.interpolate(
                    attn,
                    size=(trunk.shape[2], trunk.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
                if attn.shape[1] != c:
                    attn = attn[:, :c]
            
            trunk = trunk + control_strength * control_signal * attn
            
            # 残差连接
            fea = fea + trunk
            
            # 上采样阶段1
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
            fea = self.net_g.conv_up1(fea)
            
            # 应用第一个上采样层的控制信号
            signal_idx += 1
            attn = torch.sigmoid(attention_maps[signal_idx])
            
            control_signal = control_signals[signal_idx]
            if fea.shape[2:] != control_signal.shape[2:]:
                control_signal = F.interpolate(
                    control_signal,
                    size=(fea.shape[2], fea.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            
            if fea.shape[2:] != attn.shape[2:]:
                b, c, _, _ = attn.shape
                attn = F.interpolate(
                    attn,
                    size=(fea.shape[2], fea.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
                if attn.shape[1] != c:
                    attn = attn[:, :c]
            
            fea = fea + control_strength * 1.5 * control_signal * attn
            
            # 上采样阶段2
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
            fea = self.net_g.conv_up2(fea)
            
            # 应用第二个上采样层的控制信号
            signal_idx += 1
            attn = torch.sigmoid(attention_maps[signal_idx])
            
            control_signal = control_signals[signal_idx]
            if fea.shape[2:] != control_signal.shape[2:]:
                control_signal = F.interpolate(
                    control_signal,
                    size=(fea.shape[2], fea.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            
            if fea.shape[2:] != attn.shape[2:]:
                b, c, _, _ = attn.shape
                attn = F.interpolate(
                    attn,
                    size=(fea.shape[2], fea.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
                if attn.shape[1] != c:
                    attn = attn[:, :c]
            
            fea = fea + control_strength * 1.5 * control_signal * attn
            
            # 最终输出层
            out = self.net_g.conv_hr(fea)
            out = self.net_g.conv_last(out)
        
        # 保存注意力图
        self.attention_maps = attention_maps
        
        return out
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """验证过程，修改为和训练模式一样退化流程进行测试，有对比图，热力图，ControlMap可视化"""
        self.is_train = False
        
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        
        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}
        
        # 创建结果目录
        save_path_root = self.opt['path']['visualization']
        if save_img and not osp.exists(save_path_root):
            os.makedirs(save_path_root, exist_ok=True)
        # 创建注意力图和控制映射目录
        if self.opt.get('val', {}).get('save_control_maps', False):
            os.makedirs(osp.join(save_path_root, 'control_maps'), exist_ok=True)
        
        # 添加计数器，限制只测试前50张图片
        test_count = 0
        max_test_samples = 50
        
        for idx, val_data in enumerate(dataloader):
            # 检查是否已经测试了设定数量的图片
            if test_count >= max_test_samples:
                break
            
            test_count += 1
            
            if 'text_prompt' in val_data:
                self.text_prompts = val_data['text_prompt']
            else:
                raise ValueError("没有提供文本提示")
            
            if 'lq' in val_data:
                # TODO: 这里应该不会出现这种情况，因为没配置LQ数据集，可以删掉
                self.lq = val_data['lq'].to(self.device)
            elif 'gt' in val_data:
                # 如果没有提供LQ，则从GT生成LQ图像（与训练过程一致）
                # TODO: 现在训练是先缩放再退化，如果训练改了这里也要改
                if not hasattr(self, 'jpeger'):
                    self.jpeger = DiffJPEG(differentiable=False).cuda()
                if not hasattr(self, 'usm_sharpener'):
                    self.usm_sharpener = USMSharp().cuda()

                self.gt = val_data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

                self.kernel1 = val_data['kernel1'].to(self.device)
                self.kernel2 = val_data['kernel2'].to(self.device)
                self.sinc_kernel = val_data['sinc_kernel'].to(self.device)

                ori_h, ori_w = self.gt.size()[2:4]

                # ----------------------- 第一阶段退化过程 ----------------------- #
                # 模糊
                out = filter2D(self.gt_usm, self.kernel1)
                
                # 添加噪声
                gray_noise_prob = self.opt['gray_noise_prob'] if 'gray_noise_prob' in self.opt else 0.4
                gaussian_noise_prob = self.opt['gaussian_noise_prob'] if 'gaussian_noise_prob' in self.opt else 0.5
                noise_range = self.opt['noise_range'] if 'noise_range' in self.opt else [1, 15]
                poisson_scale_range = self.opt['poisson_scale_range'] if 'poisson_scale_range' in self.opt else [0.05, 1.5]
                
                if np.random.uniform() < gaussian_noise_prob:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=noise_range, clip=True, 
                        rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out, scale_range=poisson_scale_range, 
                        gray_prob=gray_noise_prob, clip=True, rounds=False)
                
                # JPEG压缩
                jpeg_range = self.opt['jpeg_range'] if 'jpeg_range' in self.opt else [40, 95]
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
                out = torch.clamp(out, 0, 1)  # 裁剪到[0, 1]，否则JPEGer会产生不良伪影
                out = self.jpeger(out, quality=jpeg_p)

                # ----------------------- 第二阶段退化过程 ----------------------- #
                # 模糊
                second_blur_prob = self.opt['second_blur_prob'] if 'second_blur_prob' in self.opt else 0.8
                if np.random.uniform() < second_blur_prob:
                    out = filter2D(out, self.kernel2)
                    
                # 添加噪声
                gray_noise_prob2 = self.opt['gray_noise_prob2'] if 'gray_noise_prob2' in self.opt else 0.4
                gaussian_noise_prob2 = self.opt['gaussian_noise_prob2'] if 'gaussian_noise_prob2' in self.opt else 0.5
                noise_range2 = self.opt['noise_range2'] if 'noise_range2' in self.opt else [1, 12]
                poisson_scale_range2 = self.opt['poisson_scale_range2'] if 'poisson_scale_range2' in self.opt else [0.05, 1.25]
                
                if np.random.uniform() < gaussian_noise_prob2:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=noise_range2, clip=True, 
                        rounds=False, gray_prob=gray_noise_prob2)
                else:
                    out = random_add_poisson_noise_pt(
                        out, scale_range=poisson_scale_range2, 
                        gray_prob=gray_noise_prob2, clip=True, rounds=False)
      
                # JPEG压缩
                jpeg_range2 = self.opt['jpeg_range2'] if 'jpeg_range2' in self.opt else [40, 95]
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                
                # 最终的sinc滤波
                out = filter2D(out, self.sinc_kernel)
                
                # 在完成所有退化后再进行缩放
                scale = self.opt['scale'] if 'scale' in self.opt else 4
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // scale, ori_w // scale), mode=mode)
                
                # clamp and round
                self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

                # 确保GT和lq尺寸匹配
                target_h, target_w = self.lq.size()[2:4]  # lq尺寸
                gt_h, gt_w = target_h * scale, target_w * scale  # 期望的gt尺寸
                
                # 如果GT尺寸不匹配，则调整GT和gt_usm的尺寸
                if self.gt.size(2) != gt_h or self.gt.size(3) != gt_w:
                    self.gt = F.interpolate(self.gt, size=(gt_h, gt_w), mode='bicubic')
                    self.gt_usm = F.interpolate(self.gt_usm, size=(gt_h, gt_w), mode='bicubic')
            else:
                raise ValueError("没有提供低质量图像或高分辨率图像")
                
            if 'control_map' in val_data:
                self.control_map = val_data['control_map'].to(self.device)
            else:
                logger = get_root_logger()
                logger.warning("没有提供控制映射，尝试从其他数据创建")
                raise ValueError("没有提供控制映射")

            # 获取图像名称
            if 'gt_path' in val_data:
                img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            elif 'lq_path' in val_data:
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            else:
                img_name = f"val_img_{idx}"
            
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
            sr_img_original = tensor2img([visuals['result']])
            
            # 只保存前20张图片的结果
            save_this_img = save_img and idx < 20
                
            # 保存注意力图（如果有）
            if save_this_img and self.attention_maps and self.opt.get('val').get('save_attention_maps'):
                attention_maps = self.get_current_attention_maps()
                if attention_maps:
                    # 创建sr_img的独立副本用于热力图处理
                    sr_img_heatmap = sr_img_original.copy()
                    
                    # 使用GradCAM风格保存合并后的注意力图
                    attn_save_path = self.save_gradcam_attention(sr_img_heatmap, attention_maps, img_name, save_path_root, current_iter)
                    
                    # 直接在输出图像上叠加注意力图
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
                    
            # 保存控制映射可视化（如果有）
            if save_this_img and hasattr(self, 'control_map') and self.opt.get('val', {}).get('save_control_maps'):
                # 获取控制映射
                control_map = self.control_map[0].detach().cpu().numpy()
                
                # 转换为可视化格式 [C, H, W] -> [H, W, C]
                control_map = np.transpose(control_map, (1, 2, 0))
                
                # 将每个通道分别可视化
                control_channels = []
                titles = ["Canny Edge", "Depth Map", "Object Mask"]
                
                for i in range(control_map.shape[2]):
                    channel = control_map[:, :, i]
                    
                    # 归一化到0-255
                    if np.max(channel) > np.min(channel):
                        normalized = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255
                    else:
                        normalized = np.zeros_like(channel)
                    
                    # 转换为彩色可视化
                    if i == 0:  # Canny边缘
                        colored = cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    elif i == 1:  # 深度图 - 使用热力图
                        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
                    else:
                        colored = np.zeros((normalized.shape[0], normalized.shape[1], 3), dtype=np.uint8)
                        colored[normalized > 128, 1] = 255

                    title_height = 30
                    titled_img = np.ones((colored.shape[0] + title_height, colored.shape[1], 3), dtype=np.uint8) * 255
                    titled_img[title_height:, :, :] = colored
                    
                    cv2.putText(
                        titled_img, 
                        titles[i], 
                        (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 0), 
                        1, 
                        cv2.LINE_AA
                    )
                    
                    control_channels.append(titled_img)
            
            # 2. 可选: 测试无文本引导的结果进行对比
            if self.opt.get('val', {}).get('compare_with_unguided'):
                if hasattr(self, 'net_g_ema'):
                    with torch.no_grad():
                        self.output = self.forward_sr_network(self.lq, apply_guidance=False)
                        results['unguided'] = self.output.clone()
                else:
                    with torch.no_grad():
                        self.output = self.forward_sr_network(self.lq, apply_guidance=False)
                        results['unguided'] = self.output.clone()
            
            self.output = results['guided']
            
            # 保存文本提示和比较图
            if save_this_img:
                
                if 'text_prompt' in val_data:
                    prompt_path = osp.join(save_path_root, f'{img_name}_prompt_{current_iter}.txt')
                    with open(prompt_path, 'w') as f:
                        f.write(val_data['text_prompt'][0])
                
                # 保存无引导结果（如果有）
                if 'unguided' in results:
                    unguided_output = results['unguided']
                    unguided_img = tensor2img([unguided_output.detach().cpu()])
                    # 创建对比图
                    if self.opt.get('val', {}).get('save_comparison'):
                        # 创建四列对比图（LQ、无引导、有引导、GT+掩码）
                        guided_img = sr_img_original.copy()
                        lq_img = tensor2img([self.lq])
                        gt_img = tensor2img([visuals['gt']])
                        
                        # 准备注意力图和掩码图像
                        overlay_ratio = 0.0  # 默认重叠率
                        gt_with_mask = gt_img.copy()
                        
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
                                                    obj_mask_resized = cv2.resize(obj_mask, (w, h), interpolation=cv2.INTER_LINEAR)
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
                                            # 1. 注意力图中高于均值的点被认为是注意区域
                                            attention_thresh = np.mean(attention_map) + 0.5 * np.std(attention_map)
                                            attention_binary = (attention_map > attention_thresh).astype(np.float32)
                                            
                                            # 2. 计算交集和并集
                                            mask_binary = (mask > 0).astype(np.float32)
                                            intersection = np.sum(attention_binary * mask_binary)
                                            mask_area = np.sum(mask_binary)
                                            
                                            # 3. 计算重叠率（交集除以掩码面积）
                                            if mask_area > 0:
                                                overlay_ratio = intersection / mask_area
                                                
                            except Exception as e:
                                print(f"对象信息处理错误: {e}")
                        
                        h, w = guided_img.shape[:2]
                        # 将所有图像调整到相同大小以便对比显示
                        lq_img = cv2.resize(lq_img, (w, h), interpolation=cv2.INTER_NEAREST)
                        gt_with_mask = cv2.resize(gt_with_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        # 创建四列对比图（LQ、无引导、有引导、GT+掩码）
                        comparison = np.zeros((h, w*4, 3), dtype=np.uint8)
                        comparison[:, :w] = lq_img
                        comparison[:, w:2*w] = unguided_img
                        comparison[:, 2*w:3*w] = guided_img
                        comparison[:, 3*w:] = gt_with_mask
                        
                        # 添加分割线
                        comparison[:, w-1:w+1] = [0, 0, 255]
                        comparison[:, 2*w-1:2*w+1] = [0, 0, 255]
                        comparison[:, 3*w-1:3*w+1] = [0, 0, 255]
                        
                        # 将重叠率添加到文件名中
                        # TODO: 感觉重叠率逻辑有点问题，没考虑注意力发散的问题
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
                if 'unguided' in results and self.opt.get('val', {}).get('compare_metrics'):
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
        """保存模型 - 只保存ControlNet（MultiControlNet）网络"""
        # TODO: 只保存ControlNet（包括Canny、Depth、Mask），不保存SR网络，因为冻结了RRDBNet和Clip，然后如果需要支持恢复训练，需要保存DiscriminatorNet，存储空间不够了
        if hasattr(self, 'net_control'):
            self.save_network(self.net_control, 'net_control', current_iter)

        # TODO: 感觉暂时可以不用保存  
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