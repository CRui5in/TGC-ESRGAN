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
class TextRegionAttentionLoss(nn.Module):
    """
    简化版文本区域监督注意力损失：直接使用整合后的注意力图和掩码
    """
    def __init__(self, loss_weight=1.0, reduction='mean', entropy_weight=0.05):
        super(TextRegionAttentionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # 熵正则化和多样性权重
        self.entropy_weight = entropy_weight
        self.initial_entropy_weight = entropy_weight
        self.target_entropy_weight = 0.1
        
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