import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger
from pycocotools import mask as mask_util
import numpy as np
import os
import cv2
from transformers import CLIPProcessor, CLIPModel, CLIPConfig


def decode_mask(mask_encoded):
    """解码RLE格式的掩码为二维数组
    
    处理json序列化后的掩码数据，支持str或bytes格式的counts
    """
    try:
        if not isinstance(mask_encoded, dict):
            return None
        
        if 'counts' not in mask_encoded or 'size' not in mask_encoded:
            return None
        
        counts = mask_encoded['counts']
        if isinstance(counts, str):
            counts = counts.encode('utf-8')
            mask_encoded['counts'] = counts
        
        mask = mask_util.decode(mask_encoded)
        return mask
    except Exception:
        return np.zeros((1, 1), dtype=np.uint8)


@LOSS_REGISTRY.register()
class CLIPSemanticLoss(nn.Module):
    """CLIP语义一致性损失
    
    计算生成图像与文本描述之间的语义相似度
    Lclip = 1 - cos(CLIP_img(G(I_LR)), CLIP_text(T))
    """
    def __init__(self, loss_weight=1.0, clip_model="/root/autodl-tmp/clip-vit-large-patch14"):
        super(CLIPSemanticLoss, self).__init__()
        self.loss_weight = loss_weight
        self.logger = get_root_logger()
        
        try:
            self.processor = CLIPProcessor.from_pretrained(clip_model)
            self.model = CLIPModel.from_pretrained(clip_model)
            
            # 冻结CLIP参数
            for param in self.model.parameters():
                param.requires_grad = False
                
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
            self.clip_available = True
            self.logger.info(f"成功加载CLIP模型: {clip_model}")
        except Exception as e:
            self.logger.error(f"加载CLIP模型失败: {e}")
            self.clip_available = False
    
    def forward(self, sr_images, text_prompts):
        """计算SR图像与文本提示的语义一致性损失
        
        Args:
            sr_images: 超分辨率图像 [B,3,H,W]，值范围[0,1]
            text_prompts: 文本提示列表
            
        Returns:
            语义一致性损失
        """
        if not self.clip_available:
            return torch.tensor(0.0, device=sr_images.device, requires_grad=True)
        
        if not text_prompts or len(text_prompts) == 0:
            return torch.tensor(0.0, device=sr_images.device, requires_grad=True)
        
        try:
            batch_size = sr_images.size(0)
            device = sr_images.device
            
            # 匹配提示数量与图像数量
            if len(text_prompts) < batch_size:
                text_prompts = text_prompts * batch_size
            text_prompts = text_prompts[:batch_size]
            
            # 预处理图像
            images_np = []
            for i in range(batch_size):
                img = sr_images[i].detach().cpu().float().clamp(0, 1).numpy()
                img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
                img = (img * 255).astype(np.uint8)
                images_np.append(img)
            
            # CLIP处理
            inputs = self.processor(
                text=text_prompts,
                images=images_np,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 计算特征
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # 归一化特征
                image_embeds = F.normalize(image_embeds, dim=-1)
                text_embeds = F.normalize(text_embeds, dim=-1)
            
            # 计算余弦相似度
            similarities = torch.sum(image_embeds * text_embeds, dim=-1)
            
            # 损失: 1 - 相似度
            loss = 1.0 - similarities.mean()
            
            # 防止梯度爆炸
            loss = torch.clamp(loss, min=0.0, max=1.0)
            
            return self.loss_weight * loss
            
        except Exception as e:
            self.logger.error(f"CLIP语义损失计算失败: {e}")
            return torch.tensor(0.0, device=sr_images.device, requires_grad=True) 

# TODO: 暂时没用到，先测试不添加的模型效果，好像有用，但是如果用这个训练太久会导致注意力莫名其妙发散，5000步感觉就够了        
@LOSS_REGISTRY.register()
class TextRegionAttentionLoss(nn.Module):
    """文本区域注意力监督损失
    
    引导模型关注文本描述中提到的图像区域，优化注意力分布以集中于目标掩码区域
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(TextRegionAttentionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.logger = get_root_logger()
        
    def compute_single_loss(self, attention_logits, text_masks):
        """计算单张注意力图的损失"""
        # 确保尺寸匹配
        if attention_logits.shape[-2:] != text_masks.shape[-2:]:
            text_masks = F.interpolate(text_masks, size=attention_logits.shape[-2:], 
                                      mode='nearest')
        
        # 应用 sigmoid
        attn_probs = torch.sigmoid(attention_logits)
        
        # BCE Loss
        bce_loss = self.bce_loss(attention_logits, text_masks)
        if self.reduction == 'mean':
            bce_loss = bce_loss.mean()
        elif self.reduction == 'sum':
            bce_loss = bce_loss.sum()
        
        # Dice Loss
        intersection = (attn_probs * text_masks).sum()
        dice_loss = 1 - (2 * intersection) / (attn_probs.sum() + text_masks.sum() + 1e-7)
        
        # 组合损失
        return bce_loss + dice_loss

    def forward(self, attention_logits, text_masks):
        """计算注意力监督损失
        
        支持多张注意力图的损失计算
        
        Args:
            attention_logits: 注意力图 logits，列表或单个张量
            text_masks: 文本区域掩码，形状为 (batch_size, 1, height, width)
        """
        # 初始化设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 首先检查 attention_logits 是否为 None 或空列表
        if attention_logits is None:
            self.logger.warning("attention_logits 是 None，返回零损失")
            return torch.tensor(0.0, requires_grad=True, device=device)
            
        # 2. 如果是单个张量，转换为列表以统一处理
        if not isinstance(attention_logits, list):
            attention_logits = [attention_logits]
            
        # 3. 检查列表是否为空
        if len(attention_logits) == 0:
            self.logger.warning("attention_logits 是空列表，返回零损失")
            return torch.tensor(0.0, requires_grad=True, device=device)
            
        # 4. 找出第一个有效的张量，用于确定设备
        valid_tensor = None
        for tensor in attention_logits:
            if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                valid_tensor = tensor
                device = tensor.device
                break
                
        # 5. 检查是否找到了有效张量
        if valid_tensor is None:
            self.logger.warning("attention_logits 中没有有效的张量，返回零损失")
            return torch.tensor(0.0, requires_grad=True, device=device)
            
        # 6. 检查 text_masks 是否为 None
        if text_masks is None:
            self.logger.warning("text_masks 是 None，返回零损失")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 7. 过滤掉 None 值和无效张量
        valid_attention_logits = []
        for i, attn_map in enumerate(attention_logits):
            if attn_map is None:
                self.logger.warning(f"第 {i} 个 attention_map 是 None，跳过")
                continue
            if not isinstance(attn_map, torch.Tensor):
                self.logger.warning(f"第 {i} 个 attention_map 不是 Tensor，跳过")
                continue
            if attn_map.numel() == 0:
                self.logger.warning(f"第 {i} 个 attention_map 是空 Tensor，跳过")
                continue
            if torch.isnan(attn_map).any():
                self.logger.warning(f"第 {i} 个 attention_map 包含 NaN 值，尝试修复")
                attn_map = torch.where(torch.isnan(attn_map), torch.zeros_like(attn_map), attn_map)
            
            valid_attention_logits.append(attn_map)
        
        # 8. 检查有效张量数量
        num_maps = len(valid_attention_logits)
        if num_maps == 0:
            self.logger.warning("过滤后没有有效的 attention_map，返回零损失")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 9. 动态生成权重 - 确保权重数量与有效张量数量匹配
        if num_maps >= 3:
            # 如果有足够数量的 attention_map，给予最后两层更高的权重
            weights = [0.1] * (num_maps - 2) + [0.3, 0.3]
        else:
            # 否则均匀分配权重
            weights = [1.0 / num_maps] * num_maps
            
        # 归一化权重
        weights = [w / sum(weights) for w in weights]
        
        # 10. 计算加权损失
        total_loss = 0
        for i, attn_map in enumerate(valid_attention_logits):
            try:
                loss = self.compute_single_loss(attn_map, text_masks)
                total_loss += weights[i] * loss
            except Exception as e:
                self.logger.error(f"计算第 {i} 个 attention_map 的损失时出错: {e}")
                continue
        
        # 11. 返回加权总损失
        return self.loss_weight * total_loss