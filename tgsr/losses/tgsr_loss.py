import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY

from pycocotools import mask as mask_util
import numpy as np


def decode_mask(mask_encoded):
    """将RLE编码的掩码解码为二维数组"""
    if isinstance(mask_encoded['counts'], str):
        mask_encoded['counts'] = mask_encoded['counts'].encode('utf-8')
    mask = mask_util.decode(mask_encoded)
    return mask


@LOSS_REGISTRY.register()
class TextSemanticConsistencyLoss(nn.Module):
    """
    语义一致性损失：确保增强特征与文本语义一致
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(TextSemanticConsistencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        
    def forward(self, enhanced_features, text_pooled, feat_proj=None):
        """
        Args:
            enhanced_features (Tensor): 增强后的特征, shape (B, C, H, W)
            text_pooled (Tensor): 文本池化特征, shape (B, D)
            feat_proj (nn.Module, optional): 特征投影层
        """
        # 全局池化特征
        b, c, h, w = enhanced_features.size()
        pooled_features = F.adaptive_avg_pool2d(enhanced_features, (1, 1)).view(b, c)
        
        # 如果需要，投影到相同维度
        if feat_proj is not None:
            pooled_features = feat_proj(pooled_features)
        
        # 计算余弦相似度
        sim = F.cosine_similarity(pooled_features, text_pooled, dim=1)
        
        # 计算损失：最大化相似度，所以使用 1-sim
        loss = 1.0 - sim
        
        # 应用reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class TextRegionAttentionLoss(nn.Module):
    """
    文本区域监督注意力损失：指导注意力集中在与文本相关的对象区域
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(TextRegionAttentionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def resize_mask(self, mask, target_size):
        """调整掩码大小"""
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
            
        return F.interpolate(mask, size=target_size, mode='nearest')
        
    def forward(self, attention_maps, text_prompts, objects_info, device=None):
        """
        Args:
            attention_maps (Tensor): 注意力图，shape (B, 1, H, W)
            text_prompts (list): 文本提示列表，长度为B
            objects_info (list): 对象信息列表，每个元素包含一个样本的对象信息
            device (torch.device): 设备
        """
        batch_size = len(text_prompts)
        device = attention_maps.device if device is None else device
        loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        for i in range(batch_size):
            # 获取当前样本的文本和对象信息
            text = text_prompts[i].lower()
            if i >= len(objects_info) or not objects_info[i]:
                continue
                
            objects = objects_info[i]
            h, w = attention_maps[i].shape[-2:]
            text_region_mask = torch.zeros((1, h, w), device=device)
            found_match = False
            
            # 为文本中提到的对象创建掩码
            for obj in objects:
                if 'category' in obj and 'mask_encoded' in obj:
                    category = obj['category'].lower()
                    
                    # 检查类别是否在文本中提到
                    if category in text:
                        found_match = True
                        # 解码掩码并调整大小
                        obj_mask = decode_mask(obj['mask_encoded'])
                        obj_mask = self.resize_mask(obj_mask, (h, w)).to(device)
                        
                        # 合并掩码
                        text_region_mask = torch.max(text_region_mask, obj_mask[0])
            
            # 如果找到匹配的对象，计算损失
            if found_match:
                attention_map = attention_maps[i:i+1]
                sample_loss = self.bce_loss(attention_map, text_region_mask.unsqueeze(0))
                if self.reduction == 'mean':
                    sample_loss = sample_loss.mean()
                elif self.reduction == 'sum':
                    sample_loss = sample_loss.sum()
                    
                loss += sample_loss
                valid_samples += 1
        
        # 避免除以零
        if valid_samples > 0:
            loss = loss / valid_samples
            
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class FeatureRefinementLoss(nn.Module):
    """
    特征细化损失：确保增强过程是有效的，即增强后的特征比原始特征更接近文本语义
    """
    def __init__(self, loss_weight=1.0, margin=0.1, stability_weight=0.1, reduction='mean'):
        super(FeatureRefinementLoss, self).__init__()
        self.loss_weight = loss_weight
        self.margin = margin
        self.stability_weight = stability_weight
        self.reduction = reduction
        
    def forward(self, original_features, enhanced_features, text_pooled, feat_proj=None):
        """
        Args:
            original_features (Tensor): 原始特征，shape (B, C, H, W)
            enhanced_features (Tensor): 增强后的特征，shape (B, C, H, W)
            text_pooled (Tensor): 文本池化特征，shape (B, D)
            feat_proj (nn.Module, optional): 特征投影层
        """
        batch_size = original_features.size(0)
        
        # 全局池化特征
        orig_pooled = F.adaptive_avg_pool2d(original_features, (1, 1)).view(batch_size, -1)
        enhanced_pooled = F.adaptive_avg_pool2d(enhanced_features, (1, 1)).view(batch_size, -1)
        
        # 如果需要，投影到相同维度
        if feat_proj is not None:
            orig_pooled = feat_proj(orig_pooled)
            enhanced_pooled = feat_proj(enhanced_pooled)
        
        # 计算与文本特征的相似度
        orig_sim = F.cosine_similarity(orig_pooled, text_pooled, dim=1)
        enhanced_sim = F.cosine_similarity(enhanced_pooled, text_pooled, dim=1)
        
        # 计算提升损失：确保增强后的相似度更高
        # 使用ReLU确保仅在原始相似度高于增强相似度时有损失
        improvement_loss = F.relu(orig_sim - enhanced_sim + self.margin)
        
        # 稳定性损失：确保增强前后的特征不会相差太大
        stability_loss = F.mse_loss(enhanced_features, original_features, reduction='none')
        
        # 应用reduction
        if self.reduction == 'mean':
            improvement_loss = improvement_loss.mean()
            stability_loss = stability_loss.mean()
        elif self.reduction == 'sum':
            improvement_loss = improvement_loss.sum()
            stability_loss = stability_loss.sum()
            
        # 组合损失
        total_loss = improvement_loss + self.stability_weight * stability_loss
        
        return self.loss_weight * total_loss 