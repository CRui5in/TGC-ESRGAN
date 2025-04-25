import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
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
    def __init__(self, loss_weight=1.0, clip_model="openai/clip-vit-large-patch14"):
        super(CLIPSemanticLoss, self).__init__()
        self.loss_weight = loss_weight
        
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
            print(f"成功加载CLIP模型: {clip_model}")
        except Exception as e:
            print(f"加载CLIP模型失败: {e}")
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
            print(f"CLIP语义损失计算失败: {e}")
            return torch.tensor(0.0, device=sr_images.device, requires_grad=True) 

# TODO: 暂时没用到，先测试不添加的模型效果，好像有用，但是如果用这个训练太久会导致注意力莫名其妙发散，5000步感觉就够了        
@LOSS_REGISTRY.register()
class TextRegionAttentionLoss(nn.Module):
    """文本区域注意力监督损失
    
    引导模型关注文本描述中提到的图像区域
    """
    def __init__(self, loss_weight=1.0, reduction='mean', entropy_weight=0.05):
        super(TextRegionAttentionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.entropy_weight = entropy_weight
        
    def forward(self, attention_logits, text_masks):
        """计算注意力监督损失
        
        同时考虑BCE损失、Focal损失、对比损失和边界损失
        """
        if attention_logits is None or text_masks is None:
            return None
            
        # 确保尺寸匹配
        if attention_logits.shape[-2:] != text_masks.shape[-2:]:
            text_masks = F.interpolate(text_masks, size=attention_logits.shape[-2:], 
                                      mode='bilinear', align_corners=False)
        
        # 空掩码处理
        if text_masks.sum() == 0:
            return torch.tensor(0.1, device=attention_logits.device, requires_grad=True)
        
        # BCE损失
        bce_loss = self.bce_loss(attention_logits, text_masks)
        
        # 应用reduction
        if self.reduction == 'mean':
            bce_loss = bce_loss.mean()
        elif self.reduction == 'sum':
            bce_loss = bce_loss.sum()
        
        # 应用sigmoid并保持数值稳定性
        eps = 1e-7
        attn_probs = torch.sigmoid(attention_logits)
        attn_probs = torch.clamp(attn_probs, min=eps, max=1.0-eps)
        
        # 熵计算
        entropy = -attn_probs * torch.log(attn_probs) - (1-attn_probs) * torch.log(1-attn_probs)
        
        # Focal Loss (关注困难样本)
        # TODO: 感觉有用，但是感知不大，可以做消融
        gamma = 3.0
        pt = torch.where(text_masks > 0.5, attn_probs, 1-attn_probs)
        focal_weight = (1 - pt) ** gamma
        focal_bce_loss = (focal_weight * bce_loss).mean()
        
        # 精确率-召回率平衡
        mask_attn = (attn_probs * text_masks).sum() / (text_masks.sum() + eps)
        non_mask_attn = (attn_probs * (1-text_masks)).sum() / ((1-text_masks).sum() + eps)
        
        # 对比损失 - 确保掩码区域注意力高于非掩码区域
        contrast_loss = torch.clamp(non_mask_attn - mask_attn + 0.5, min=0.0)
        
        # 边界区域注意力损失
        kernel_size = 3
        padding = kernel_size // 2
        pool_masks = F.max_pool2d(text_masks, kernel_size=kernel_size, 
                                stride=1, padding=padding)
        boundary_masks = pool_masks - text_masks
        
        boundary_weight = 1.5 
        boundary_loss = F.mse_loss(attn_probs * boundary_masks, boundary_masks) * boundary_weight
        
        # 组合损失
        final_loss = focal_bce_loss + 1.5 * contrast_loss + boundary_loss
        
        # 限制损失最大值
        final_loss = torch.clamp(final_loss, max=10.0)
        
        return self.loss_weight * final_loss