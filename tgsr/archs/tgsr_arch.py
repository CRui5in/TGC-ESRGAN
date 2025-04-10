import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


# 基础位置编码
class PositionalEncoding2D(nn.Module):
    """二维位置编码，为特征图中的每个位置添加位置信息"""
    def __init__(self, d_model, max_h=128, max_w=128):  # 调整为128x128，适应256x256输入的特征图
        super().__init__()
        try:
            self.d_model = d_model
            
            # 创建位置编码
            pe = torch.zeros(d_model, max_h, max_w)
            
            # 计算位置编码
            div_term = torch.exp(torch.arange(0, d_model // 2, dtype=torch.float) * (-math.log(10000.0) / (d_model // 2)))
            
            # 为每个高度和宽度位置创建位置编码 - 向量化实现
            # 创建位置索引
            h_idx = torch.arange(max_h).unsqueeze(1).expand(-1, max_w).reshape(-1)
            w_idx = torch.arange(max_w).repeat(max_h)
            
            # 偶数位置和奇数位置的指数
            even_idx = torch.arange(0, d_model, 2)
            odd_idx = torch.arange(1, d_model, 2)
            
            # 确保索引不超过维度
            even_idx = even_idx[even_idx < d_model]
            odd_idx = odd_idx[odd_idx < d_model]
            
            # 为所有位置同时计算
            for pos in range(0, h_idx.size(0)):
                h = h_idx[pos].item()
                w = w_idx[pos].item()
                
                # 高度位置编码
                pe[even_idx, h, w] = torch.sin(torch.tensor(h) * div_term[:len(even_idx)])
                pe[odd_idx, h, w] = torch.cos(torch.tensor(h) * div_term[:len(odd_idx)])
                
                # 宽度位置编码
                if d_model > 1:
                    w_term = div_term.clone()
                    if len(w_term) * 2 > d_model:
                        w_term = w_term[:d_model//2]
                    
                    w_even_idx = torch.arange(0, min(d_model, 2*len(w_term)), 2)
                    w_odd_idx = torch.arange(1, min(d_model, 2*len(w_term)), 2)
                    
                    pe[w_even_idx, h, w] = torch.sin(torch.tensor(w) * w_term)
                    pe[w_odd_idx, h, w] = torch.cos(torch.tensor(w) * w_term)
            
            self.register_buffer('pe', pe)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            位置编码: [B, C, H, W]
        """
        try:
            B, C, H, W = x.shape
            
            # 如果需要更大的位置编码，动态生成
            if H > self.pe.size(1) or W > self.pe.size(2):
                with torch.no_grad():
                    return self._generate_encoding_on_the_fly(x)
            
            # 使用预计算的位置编码
            pos_encoding = self.pe[:, :H, :W].unsqueeze(0).repeat(B, 1, 1, 1)
            
            # 确保输出维度与输入一致
            if C > self.d_model:
                # 如果需要更多通道，重复位置编码
                repeat_factor = C // self.d_model + 1
                pos_encoding = pos_encoding.repeat(1, repeat_factor, 1, 1)
                pos_encoding = pos_encoding[:, :C, :, :]
            elif C < self.d_model:
                # 如果需要更少通道，截断位置编码
                pos_encoding = pos_encoding[:, :C, :, :]
            
            return pos_encoding
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_encoding_on_the_fly(self, x):
        """动态生成位置编码"""
        B, C, H, W = x.shape
        temp_encoding = torch.zeros(C, H, W, device=x.device)
        
        # 仅对当前需要的大小进行计算
        div_term = torch.exp(torch.arange(0, C // 2, dtype=torch.float, device=x.device) * (-math.log(10000.0) / (C // 2)))
        
        # 为每个位置创建编码（简化版本）
        h_pos = torch.arange(H, device=x.device).unsqueeze(1).repeat(1, W)
        w_pos = torch.arange(W, device=x.device).repeat(H, 1)
        
        # 仅计算必要的通道
        even_idx = torch.arange(0, C, 2, device=x.device)
        odd_idx = torch.arange(1, C, 2, device=x.device)
        
        # 截断确保索引有效
        even_idx = even_idx[even_idx < C]
        odd_idx = odd_idx[odd_idx < C]
        
        # 计算偶数和奇数通道
        for i in even_idx:
            temp_encoding[i] = torch.sin(h_pos * div_term[i//2])
        
        for i in odd_idx:
            temp_encoding[i] = torch.cos(h_pos * div_term[i//2])
        
        # 添加批次维度并返回
        return temp_encoding.unsqueeze(0).repeat(B, 1, 1, 1)


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        try:
            assert d_model % num_heads == 0, f"d_model({d_model})必须能被num_heads({num_heads})整除"
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            
            # 线性投影层
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
    def forward(self, query, key, value, attn_mask=None):
        try:
            batch_size = query.shape[0]
            
            # 检查输入数据是否包含NaN
            if torch.isnan(query).any() or torch.isnan(key).any() or torch.isnan(value).any():
                query = torch.nan_to_num(query)
                key = torch.nan_to_num(key)
                value = torch.nan_to_num(value)
            
            # 线性投影并分割多头
            q = self.q_proj(query)  # [batch_size, seq_len_q, d_model]
            k = self.k_proj(key)    # [batch_size, seq_len_k, d_model]
            v = self.v_proj(value)  # [batch_size, seq_len_v, d_model]
            
            # 重塑为多头形式
            q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_q, head_dim]
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_k, head_dim]
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_v, head_dim]
            
            # 缩放点积注意力
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len_q, seq_len_k]
            
            # 应用注意力掩码（如果提供）
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e9)
            
            # 应用softmax并获取注意力权重
            attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
            attn_weights = self.dropout(attn_weights)
            
            # 应用注意力权重并合并多头
            output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len_q, head_dim]
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len_q, d_model]
            
            # 最终线性投影
            output = self.out_proj(output)
            
            # 检查输出是否包含NaN
            if torch.isnan(output).any():
                output = torch.nan_to_num(output)
            
            return output, attn_weights
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise


# 双向交叉注意力模块
class BidirectionalCrossAttention(nn.Module):
    """双向交叉注意力：图像→文本和文本→图像"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        try:
            # 图像→文本和文本→图像的注意力机制
            self.img_to_text_attn = MultiHeadAttention(embed_dim, num_heads)
            self.text_to_img_attn = MultiHeadAttention(embed_dim, num_heads)
            
            # 归一化层
            self.img_norm = nn.LayerNorm(embed_dim)
            self.text_norm = nn.LayerNorm(embed_dim)
            
            # 前馈网络
            self.img_ff = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            self.text_ff = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            
            # 前馈网络的归一化层
            self.img_ff_norm = nn.LayerNorm(embed_dim)
            self.text_ff_norm = nn.LayerNorm(embed_dim)
            
            # 维度投影（如果需要）
            self.text_projection = nn.Linear(embed_dim, embed_dim)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
    def forward(self, img_feat, text_feat):
        try:
            # 检查并修复NaN输入
            if torch.isnan(img_feat).any():
                img_feat = torch.nan_to_num(img_feat)
            if torch.isnan(text_feat).any():
                text_feat = torch.nan_to_num(text_feat)
            
            # 检查文本特征维度是否需要投影
            if text_feat.shape[-1] != img_feat.shape[-1]:
                # 投影文本特征到正确的维度
                text_feat = self.text_projection(text_feat)
                
            # 图像 -> 文本 注意力
            img_feat_norm = self.img_norm(img_feat)
            text_feat_norm = self.text_norm(text_feat)
            
            # 文本引导的图像特征增强
            enhanced_img, _ = self.text_to_img_attn(img_feat_norm, text_feat_norm, text_feat_norm)
            enhanced_img = img_feat_norm + enhanced_img
            enhanced_img = enhanced_img + self.img_ff(self.img_ff_norm(enhanced_img))
            
            # 图像增强的文本特征
            enhanced_text, _ = self.img_to_text_attn(text_feat_norm, img_feat_norm, img_feat_norm)
            enhanced_text = text_feat_norm + enhanced_text
            enhanced_text = enhanced_text + self.text_ff(self.text_ff_norm(enhanced_text))
            
            return enhanced_img, enhanced_text
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise


# 文本条件归一化
class TextConditionedNorm(nn.Module):
    """文本条件归一化层"""
    def __init__(self, channels, text_dim):
        super().__init__()
        self.channels = channels
        self.text_dim = text_dim
        
        # 从文本特征生成缩放和偏移参数
        self.scale_shift = nn.Sequential(
            nn.Linear(text_dim, channels * 2),
            nn.Sigmoid()
        )
        
    def forward(self, x, text_embedding):
        """
        Args:
            x: [B, C, H, W] - 图像特征
            text_embedding: [B, text_dim] - 文本嵌入
        """
        # 生成缩放和偏移参数
        params = self.scale_shift(text_embedding)
        scale, shift = torch.chunk(params, 2, dim=1)
        
        # 调整为适当的形状
        scale = scale.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        shift = shift.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        
        # 应用缩放和偏移
        return x * (scale + 1.0) + shift


# 区域感知文本融合
class RegionAwareTextFusion(nn.Module):
    """区域感知的文本融合模块"""
    def __init__(self, channels, text_dim):
        super().__init__()
        self.channels = channels
        self.text_dim = text_dim
        
        # 生成区域注意力权重 - 移除最后的Sigmoid层，输出为logits
        self.attention_gen = nn.Sequential(
            nn.Conv2d(channels + text_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1)
            # 移除Sigmoid层，让输出保持为logits形式，配合BCEWithLogitsLoss使用
        )
        
        # 生成文本调制特征
        self.text_modulation = nn.Linear(text_dim, channels)
        
    def forward(self, features, text_embedding):
        """
        Args:
            features: [B, C, H, W] - 图像特征
            text_embedding: [B, text_dim] - 文本嵌入
        """
        # 生成区域注意力图
        b, c, h, w = features.shape
        text_feat_expanded = text_embedding.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
        combined_feat = torch.cat([features, text_feat_expanded], dim=1)
        attention_logits = self.attention_gen(combined_feat)  # [B, 1, H, W] - 现在是logits
        
        # 在前向传播时应用sigmoid，以便于可视化和其他用途
        attention_map = torch.sigmoid(attention_logits)
        
        # 文本特征调制
        text_feat_mod = self.text_modulation(text_embedding)  # [B, C]
        text_feat_mod = text_feat_mod.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        
        # 应用区域感知的文本融合 - 使用sigmoid后的attention_map
        enhanced_features = features + features * attention_map * text_feat_mod
        
        return enhanced_features, attention_logits  # 返回logits用于损失计算


class TextAttentionBlock(nn.Module):
    """文本注意力块，包含区域感知融合、交叉注意力和条件归一化"""
    def __init__(self, nf, text_dim, num_heads=8):
        super().__init__()
        try:
            self.text_fusion = RegionAwareTextFusion(nf, text_dim)
            self.cross_attn = BidirectionalCrossAttention(nf, num_heads)
            self.text_norm = TextConditionedNorm(nf, text_dim)
            
            # 文本特征转换层，用于将text_hidden从text_dim投影到nf
            self.text_hidden_projection = nn.Linear(text_dim, nf)
            
            # 特征转换
            self.img_to_emb = nn.Conv2d(nf, nf, kernel_size=1)
            self.emb_to_img = nn.Conv2d(nf, nf, kernel_size=1)
            
            # 位置编码
            self.pos_encoder = PositionalEncoding2D(nf)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
    def forward(self, features, text_hidden, text_pooled):
        """
        Args:
            features: [B, C, H, W] - 图像特征
            text_hidden: [B, L, D] - 文本隐藏状态
            text_pooled: [B, D] - 池化的文本特征
        """
        try:
            # 添加位置编码
            b, c, h, w = features.shape
            pos_encoding = self.pos_encoder(features)
            # 确保位置编码维度与特征维度匹配
            if pos_encoding.shape[1] != c:
                pos_encoding = pos_encoding[:, :c, :, :]
            
            features = features + pos_encoding
            
            # 区域感知文本融合
            enhanced_features, attention_logits = self.text_fusion(features, text_pooled)
            
            # 将特征展平为序列形式，用于交叉注意力
            b, c, h, w = enhanced_features.shape
            img_seq = self.img_to_emb(enhanced_features).reshape(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
            
            # 处理文本特征，确保维度匹配
            if text_hidden.shape[-1] != c:
                text_hidden_projected = self.text_hidden_projection(text_hidden)
            else:
                text_hidden_projected = text_hidden
            
            # 计算文本引导的交叉注意力
            enhanced_img_feat, enhanced_text_feat = self.cross_attn(img_seq, text_hidden_projected)
            
            # 将增强的图像特征重塑回空间维度
            enhanced_img_feat = enhanced_img_feat.permute(0, 2, 1).reshape(b, c, h, w)
            enhanced_img_feat = self.emb_to_img(enhanced_img_feat)
            
            # 应用文本条件归一化
            normalized_features = self.text_norm(enhanced_img_feat, text_pooled)
            
            # 残差连接
            output_features = features + normalized_features
            
            return output_features, attention_logits
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise


@ARCH_REGISTRY.register()
class TextGuidanceNet(nn.Module):
    """文本引导网络，独立于SR网络"""
    def __init__(self, 
                num_feat=64, 
                text_dim=512, 
                num_blocks=3, 
                num_heads=8):
        super().__init__()
        
        try:
            # 文本注意力模块
            self.text_blocks = nn.ModuleList()
            for i in range(num_blocks):
                block = TextAttentionBlock(num_feat, text_dim, num_heads)
                self.text_blocks.append(block)
            
            # 初始化权重
            self._initialize_weights()
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
    def _initialize_weights(self):
        """初始化网络权重"""
        try:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def forward(self, features, text_hidden, text_pooled):
        """
        对SR特征应用文本引导
        
        Args:
            features: [B, C, H, W] - 需要增强的SR特征图
            text_hidden: [B, L, D] - 文本隐藏状态
            text_pooled: [B, D] - 池化的文本特征
        
        Returns:
            enhanced_features: [B, C, H, W] - 文本增强的特征图
            attention_maps: list - 注意力图列表，用于可视化
        """
        attention_maps = []
        enhanced_features = features
        
        # 应用文本注意力模块
        for i, block in enumerate(self.text_blocks):
            enhanced_features, attn_logits = block(enhanced_features, text_hidden, text_pooled)
            attention_maps.append(attn_logits)
        
        return enhanced_features, attention_maps 