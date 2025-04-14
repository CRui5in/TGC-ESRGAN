import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from pycocotools import mask as mask_util
import numpy as np
import os
import cv2


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
class TextRegionAttentionLoss(nn.Module):
    """
    简化版文本区域监督注意力损失：直接使用注意力图和文本掩码
    """
    def __init__(self, loss_weight=1.0, reduction='mean', entropy_weight=0.05):
        super(TextRegionAttentionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # 使用固定熵权重
        self.entropy_weight = entropy_weight
        
    def forward(self, attention_logits, text_masks):
        """改进版TextRegionAttentionLoss，增加数值稳定性"""
        if attention_logits is None or text_masks is None:
            return None
            
        # 确保输入形状匹配
        if attention_logits.shape[-2:] != text_masks.shape[-2:]:
            text_masks = F.interpolate(text_masks, size=attention_logits.shape[-2:], 
                                      mode='bilinear', align_corners=False)
        
        # 保护性检查 - 如果掩码全0则返回安全值
        if text_masks.sum() == 0:
            return torch.tensor(0.1, device=attention_logits.device, requires_grad=True)
        
        # 计算基础BCE损失
        bce_loss = self.bce_loss(attention_logits, text_masks)
        
        # 应用reduction
        if self.reduction == 'mean':
            bce_loss = bce_loss.mean()
        elif self.reduction == 'sum':
            bce_loss = bce_loss.sum()
        
        # 计算并应用sigmoid - 使用更大的eps值
        attn_probs = torch.sigmoid(attention_logits)
        eps = 1e-7
        attn_probs = torch.clamp(attn_probs, min=eps, max=1.0-eps)
        
        # 计算熵 - 添加保护措施
        entropy = -attn_probs * torch.log(attn_probs) - (1-attn_probs) * torch.log(1-attn_probs)
        
        # 1. 强化Focal Loss - 增加gamma值增强对难样本的关注
        gamma = 3.0  # 从2.0增加到3.0
        pt = torch.where(text_masks > 0.5, attn_probs, 1-attn_probs)
        focal_weight = (1 - pt) ** gamma
        focal_bce_loss = (focal_weight * bce_loss).mean()
        
        # 3. 精确率-召回率平衡损失
        # 掩码区域的平均注意力 (召回率相关)
        mask_attn = (attn_probs * text_masks).sum() / (text_masks.sum() + eps)
        # 非掩码区域的平均注意力 (精确率相关) - 应该尽可能低
        non_mask_attn = (attn_probs * (1-text_masks)).sum() / ((1-text_masks).sum() + eps)
        # 对比损失 - 提高掩码区域与非掩码区域的注意力差距
        contrast_loss = non_mask_attn / (mask_attn + eps)
        
        # 4. 掩码边界关注损失 - 特别关注边界区域
        # 使用形态学操作找到边界 (简化版本)
        # 先模拟边界提取
        kernel_size = 3
        padding = kernel_size // 2
        pool_masks = F.max_pool2d(text_masks, kernel_size=kernel_size, 
                                 stride=1, padding=padding)
        boundary_masks = pool_masks - text_masks
        # 边界区域特别关注
        boundary_loss = F.mse_loss(attn_probs * boundary_masks, boundary_masks)
        
        # 5. 组合损失 - 调整权重分配
        final_loss = focal_bce_loss + 1.0 * contrast_loss + 0.5 * boundary_loss
        
        # 最后添加损失裁剪（在返回前）
        final_loss = torch.clamp(final_loss, max=10.0)  # 限制最大损失值为10
        
        return self.loss_weight * final_loss


@LOSS_REGISTRY.register()
class ControlFeatureLoss(nn.Module):
    """控制特征损失：确保控制特征与文本描述和对象掩码对齐
    适用于ControlNet风格的模型架构
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ControlFeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, control_features, attention_maps, text_pooled, objects_info=None, device=None):
        """
        计算控制特征损失
        
        Args:
            control_features (Tensor): 控制特征
            attention_maps (list): 注意力图列表
            text_pooled (Tensor): 文本池化特征
            objects_info (list): 对象信息列表
            device (torch.device): 设备
        """
        batch_size = text_pooled.size(0)
        device = control_features.device if device is None else device
        
        # 1. 语义一致性损失 - 确保控制特征与文本语义一致
        pooled_features = F.adaptive_avg_pool2d(control_features, (1, 1)).view(batch_size, -1)
        
        # 统一维度
        if pooled_features.size(1) != text_pooled.size(1):
            # 简单线性投影
            projection = nn.Linear(pooled_features.size(1), text_pooled.size(1), device=device)
            pooled_features = projection(pooled_features)
        
        # 计算余弦相似度损失
        similarity = F.cosine_similarity(pooled_features, text_pooled, dim=1)
        semantic_loss = 1.0 - similarity.mean()
        
        # 2. 注意力一致性损失 - 确保注意力图与对象掩码一致
        attention_loss = torch.tensor(0.0, device=device)
        if objects_info is not None and len(attention_maps) > 0:
            valid_samples = 0
            
            # 使用最后一个注意力图（通常最精细）
            attn_logits = attention_maps[-1]
            
            # 逐样本处理
            for i in range(batch_size):
                if i >= len(objects_info) or not objects_info[i]:
                    continue
                    
                objects = objects_info[i]
                h, w = attn_logits[i].shape[-2:]
                
                # 创建目标掩码
                target_mask = torch.zeros((1, h, w), device=device)
                has_objects = False
                
                # 整合对象掩码
                for obj in objects:
                    if 'mask_encoded' in obj:
                        try:
                            # 解码掩码并调整大小
                            obj_mask = decode_mask(obj['mask_encoded'])
                            if obj_mask is not None and obj_mask.sum() > 0:
                                has_objects = True
                                # 调整掩码大小并合并
                                obj_mask_tensor = torch.from_numpy(obj_mask).float().to(device)
                                obj_mask_tensor = F.interpolate(
                                    obj_mask_tensor.unsqueeze(0).unsqueeze(0), 
                                    size=(h, w), 
                                    mode='bilinear', 
                                    align_corners=False
                                )
                                target_mask = torch.max(target_mask, obj_mask_tensor[0])
                        except Exception:
                            continue
                
                if has_objects:
                    # 计算BCE损失
                    sample_attn = attn_logits[i:i+1]
                    sample_loss = self.bce_loss(sample_attn, target_mask.unsqueeze(0))
                    attention_loss += sample_loss.mean()
                    valid_samples += 1
            
            if valid_samples > 0:
                attention_loss = attention_loss / valid_samples
        
        # 3. 边界平滑损失 - 鼓励注意力图平滑变化
        smoothness_loss = torch.tensor(0.0, device=device)
        if len(attention_maps) > 0:
            attn = torch.sigmoid(attention_maps[-1])
            # 计算水平和垂直梯度
            h_gradient = torch.abs(attn[:, :, :, :-1] - attn[:, :, :, 1:])
            v_gradient = torch.abs(attn[:, :, :-1, :] - attn[:, :, 1:, :])
            
            # 对大梯度区域进行惩罚
            gradient_threshold = 0.1
            h_penalty = F.relu(h_gradient - gradient_threshold).mean()
            v_penalty = F.relu(v_gradient - gradient_threshold).mean()
            
            smoothness_loss = (h_penalty + v_penalty) * 0.5
        
        # 4. 熵正则化损失 - 防止注意力崩塌或过度扩散
        entropy_loss = torch.tensor(0.0, device=device)
        if len(attention_maps) > 0:
            attn = torch.sigmoid(attention_maps[-1])
            eps = 1e-7
            attn_clipped = torch.clamp(attn, min=eps, max=1.0-eps)
            
            # 二元熵: -p*log(p) - (1-p)*log(1-p)
            entropy = -attn_clipped * torch.log(attn_clipped) - (1-attn_clipped) * torch.log(1-attn_clipped)
            
            # 熵过高或过低都不理想
            target_entropy = 0.3  # 适中的熵值
            entropy_loss = F.mse_loss(entropy.mean(), torch.tensor(target_entropy, device=device))
        
        # 组合所有损失，权重可调
        total_loss = semantic_loss + attention_loss * 0.5 + smoothness_loss * 0.2 + entropy_loss * 0.3
        
        return self.loss_weight * total_loss 