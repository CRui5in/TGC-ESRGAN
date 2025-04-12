import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from pycocotools import mask as mask_util
import numpy as np


def decode_mask(mask_encoded):
    """将RLE编码的掩码解码为二维数组
    
    处理json序列化再反序列化的情况
    - mask_encoded可能是带有'counts'和'size'的dict
    - 'counts'可能是str或bytes
    """
    try:
        # 确保mask_encoded是字典格式
        if not isinstance(mask_encoded, dict):
            return None
        
        # 确保包含必要的键
        if 'counts' not in mask_encoded or 'size' not in mask_encoded:
            return None
        
        # 处理counts字段 - 可能是str或bytes
        counts = mask_encoded['counts']
        if isinstance(counts, str):
            # 如果是字符串，转换为bytes
            counts = counts.encode('utf-8')
            mask_encoded['counts'] = counts
        
        # 解码掩码
        mask = mask_util.decode(mask_encoded)
        return mask
    except Exception:
        # 返回空掩码
        return np.zeros((1, 1), dtype=np.uint8)


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
    简化版文本区域监督注意力损失：直接使用整合后的注意力图和掩码
    """
    def __init__(self, loss_weight=1.0, reduction='mean', entropy_weight=0.01, diversity_weight=0.02):
        super(TextRegionAttentionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # 熵正则化和多样性权重
        self.entropy_weight = entropy_weight
        self.initial_entropy_weight = entropy_weight
        self.target_entropy_weight = 0.05
        
    def resize_mask(self, mask, target_size):
        """调整掩码大小，使用bilinear插值获得更平滑的边缘"""
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
            
        # 使用双线性插值获得更平滑的结果
        return F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
    
    def entropy_regularization(self, attention_map):
        """计算注意力图的熵，鼓励注意力分布更均匀"""
        attention_prob = torch.sigmoid(attention_map)
        eps = 1e-7
        attention_prob = attention_prob.clamp(min=eps, max=1.0-eps)
        entropy = -attention_prob * torch.log(attention_prob) - (1-attention_prob) * torch.log(1-attention_prob)
        entropy_loss = -entropy.mean()
        return entropy_loss
    
    def update_entropy_weight(self, current_iter, total_iter):
        """动态更新熵正则化权重"""
        progress = min(1.0, current_iter / total_iter)
        
        if progress < 0.25:
            factor = progress * 4 * 0.2
        else:
            factor = 0.2 + 0.8 * ((progress - 0.25) / 0.75) ** 2
        
        self.entropy_weight = self.initial_entropy_weight + factor * (self.target_entropy_weight - self.initial_entropy_weight)
        return self.entropy_weight
    
    def forward(self, attention_maps, text_prompts, objects_info, device=None):
        """
        计算整合注意力图与整合掩码之间的损失
        
        Args:
            attention_maps (Tensor): 注意力图，shape (B, 1, H, W)
            text_prompts (list): 文本提示列表，长度为B
            objects_info (list): 对象信息列表
            device (torch.device): 设备
        """
        batch_size = len(text_prompts)
        device = attention_maps.device if device is None else device
        loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        # 计算正则化损失
        entropy_loss = self.entropy_regularization(attention_maps)
        
        for i in range(batch_size):
            if i >= len(objects_info) or not objects_info[i]:
                continue
                
            objects = objects_info[i]
            h, w = attention_maps[i].shape[-2:]
            
            # 创建整合掩码 - 包含所有对象，不再检查类别与文本的匹配
            combined_mask = torch.zeros((1, h, w), device=device)
            has_objects = False
            
            # 整合所有对象掩码
            for obj in objects:
                if 'mask_encoded' in obj:
                    try:
                        # 解码掩码并调整大小
                        obj_mask = decode_mask(obj['mask_encoded'])
                        if obj_mask is not None and obj_mask.sum() > 0:
                            has_objects = True
                            # 调整掩码大小并转换为张量
                            obj_mask = self.resize_mask(obj_mask, (h, w)).to(device)
                            # 合并掩码 - 使用最大值
                            combined_mask = torch.max(combined_mask, obj_mask[0])
                    except Exception:
                        continue
            
            # 只有当图像中有有效对象时才计算损失
            if has_objects:
                # 获取当前样本的注意力图
                attention_map = attention_maps[i:i+1]
                
                # 计算BCE损失
                sample_loss = self.bce_loss(attention_map, combined_mask.unsqueeze(0))
                
                # 应用reduction
                if self.reduction == 'mean':
                    sample_loss = sample_loss.mean()
                elif self.reduction == 'sum':
                    sample_loss = sample_loss.sum()
                
                loss += sample_loss
                valid_samples += 1
        
        # 避免除以零
        if valid_samples > 0:
            loss = loss / valid_samples
        
        # 添加正则化损失
        final_loss = loss + self.entropy_weight * entropy_loss
        
        return self.loss_weight * final_loss


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