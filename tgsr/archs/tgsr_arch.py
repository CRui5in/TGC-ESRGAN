import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import random


# 增强版位置编码 - 增加条件机制
class ConditionalPositionalEncoding2D(nn.Module):
    """增强的二维位置编码，支持条件性文本融合"""
    def __init__(self, d_model, max_h=128, max_w=128):
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
            
            # 修改：条件调制层 - 允许文本特征调整位置编码
            # 首先将512维的文本特征投影到d_model维度，再应用Sigmoid
            self.text_modulation = nn.Sequential(
                nn.Linear(512, 256),  # 先降维
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, d_model),  # 再投影到d_model维度
                nn.LayerNorm(d_model),
                nn.Sigmoid()
            )
            
            # 新增：位置编码投影 - 增强表达能力
            self.pos_proj = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
    def forward(self, x, text_feature=None):
        """
        Args:
            x: [B, C, H, W]
            text_feature: [B, D] 用于调制位置编码的文本特征
        Returns:
            位置编码: [B, C, H, W]
        """
        try:
            B, C, H, W = x.shape
            
            # 如果需要更大的位置编码，动态生成
            if H > self.pe.size(1) or W > self.pe.size(2):
                with torch.no_grad():
                    return self._generate_encoding_on_the_fly(x, text_feature)
            
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
            
            # 新增：应用条件调制 - 根据文本特征调整位置编码
            if text_feature is not None:
                # 生成调制系数
                mod_factors = self.text_modulation(text_feature)  # [B, D]
                mod_factors = mod_factors.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
                
                # 确保维度匹配
                if mod_factors.size(1) != pos_encoding.size(1):
                    if mod_factors.size(1) < pos_encoding.size(1):
                        # 重复调制系数
                        repeat_factor = pos_encoding.size(1) // mod_factors.size(1) + 1
                        mod_factors = mod_factors.repeat(1, repeat_factor, 1, 1)
                        mod_factors = mod_factors[:, :pos_encoding.size(1), :, :]
                    else:
                        # 截断调制系数
                        mod_factors = mod_factors[:, :pos_encoding.size(1), :, :]
                
                # 应用调制
                pos_encoding = pos_encoding * mod_factors
            
            # 新增：通过投影增强表达能力
            pos_encoding = self.pos_proj(pos_encoding)
            
            return pos_encoding
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_encoding_on_the_fly(self, x, text_feature=None):
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
        
        # 添加批次维度
        temp_encoding = temp_encoding.unsqueeze(0).repeat(B, 1, 1, 1)
        
        # 应用调制和投影（如果可用）
        if hasattr(self, 'text_modulation') and text_feature is not None:
            mod_factors = self.text_modulation(text_feature).unsqueeze(-1).unsqueeze(-1)
            if mod_factors.size(1) != temp_encoding.size(1):
                if mod_factors.size(1) < temp_encoding.size(1):
                    repeat_factor = temp_encoding.size(1) // mod_factors.size(1) + 1
                    mod_factors = mod_factors.repeat(1, repeat_factor, 1, 1)
                    mod_factors = mod_factors[:, :temp_encoding.size(1), :, :]
                else:
                    mod_factors = mod_factors[:, :temp_encoding.size(1), :, :]
            temp_encoding = temp_encoding * mod_factors
        
        if hasattr(self, 'pos_proj'):
            temp_encoding = self.pos_proj(temp_encoding)
        
        return temp_encoding


# 增强版多头注意力 - 添加门控机制和缩放点积
class EnhancedMultiHeadAttention(nn.Module):
    """增强的多头注意力机制，添加门控机制和注意力正则化"""
    def __init__(self, d_model, num_heads, dropout=0.1, attn_dropout=0.1):
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
            
            # 新增：注意力分数缩放因子
            self.scale = self.head_dim ** -0.5
            
            # 新增：门控机制 - 控制注意力强度
            self.gate = nn.Sequential(
                nn.Linear(d_model, 1, bias=False),
                nn.Sigmoid()
            )
            
            # 新增：注意力正则化 - 防止注意力崩塌
            self.attn_dropout = nn.Dropout(attn_dropout)
            self.dropout = nn.Dropout(dropout)
            
            # 新增：层归一化 - 增强训练稳定性
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)
            self.norm_out = nn.LayerNorm(d_model)
            
            # 新增：可调节的温度参数 - 控制注意力分布的锐度
            self.temperature = 1.0  # 默认值，训练过程中可动态调整
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        try:
            batch_size = query.shape[0]
            
            # 检查输入数据是否包含NaN
            if torch.isnan(query).any() or torch.isnan(key).any() or torch.isnan(value).any():
                query = torch.nan_to_num(query)
                key = torch.nan_to_num(key)
                value = torch.nan_to_num(value)
            
            # 预归一化
            query = self.norm_q(query)
            key = self.norm_kv(key)
            value = self.norm_kv(value)
            
            # 线性投影并分割多头
            q = self.q_proj(query)  # [batch_size, seq_len_q, d_model]
            k = self.k_proj(key)    # [batch_size, seq_len_k, d_model]
            v = self.v_proj(value)  # [batch_size, seq_len_v, d_model]
            
            # 重塑为多头形式
            q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_q, head_dim]
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_k, head_dim]
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_v, head_dim]
            
            # 应用缩放因子的点积注意力
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len_q, seq_len_k]
            
            # 应用注意力掩码（如果提供）
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e9)
            
            # 计算注意力权重 - 使用可调节的温度参数
            # 温度越高，分布越平缓；温度越低，分布越集中
            current_temp = getattr(self, 'temperature', 1.0)  # 获取当前温度，默认1.0
            attn_weights = F.softmax(scores / current_temp, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
            
            # 如果在训练中，随机应用注意力丢弃
            if self.training and random.random() < 0.3:  # 30%概率
                # 注意力丢弃：随机将部分注意力权重置为0，并重新归一化
                # 这可以防止注意力崩塌，促进更多样化的学习
                drop_prob = 0.1  # 10%的注意力权重被丢弃
                attn_size = attn_weights.size()
                mask = torch.bernoulli(torch.full(attn_size, 1.0 - drop_prob, device=attn_weights.device))
                attn_weights = attn_weights * mask
                # 重新归一化，确保每行的权重和为1
                attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            # 新增：注意力dropout - 防止过拟合
            attn_weights = self.attn_dropout(attn_weights)
            
            # 应用注意力权重并合并多头
            output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len_q, head_dim]
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len_q, d_model]
            
            # 最终线性投影
            output = self.out_proj(output)
            
            # 新增：门控机制 - 自适应控制注意力影响
            gate_val = self.gate(query)  # [batch_size, seq_len_q, 1]
            output = gate_val * output
            
            # 新增：最终归一化和dropout
            output = self.norm_out(output)
            output = self.dropout(output)
            
            # 检查输出是否包含NaN
            if torch.isnan(output).any():
                output = torch.nan_to_num(output)
            
            # 返回注意力权重（如需要）
            if need_weights:
                return output, attn_weights
            else:
                return output, None
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise


# 增强版双向交叉注意力 - 加强语义对齐和特征整合
class EnhancedBidirectionalCrossAttention(nn.Module):
    """增强的双向交叉注意力：图像→文本和文本→图像，添加更多细节层次"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        try:
            # 图像→文本和文本→图像的增强注意力机制
            self.img_to_text_attn = EnhancedMultiHeadAttention(embed_dim, num_heads, dropout)
            self.text_to_img_attn = EnhancedMultiHeadAttention(embed_dim, num_heads, dropout)
            
            # 归一化层
            self.img_norm1 = nn.LayerNorm(embed_dim)
            self.img_norm2 = nn.LayerNorm(embed_dim)
            self.text_norm1 = nn.LayerNorm(embed_dim)
            self.text_norm2 = nn.LayerNorm(embed_dim)
            
            # 新增：增强版特征融合前馈网络
            self.img_ff = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),  # 使用GELU激活函数增强非线性
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )
            
            self.text_ff = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )
            
            # 新增：门控特征整合 - 允许更灵活的文本-图像特征混合
            self.fusion_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, 1),
                nn.Sigmoid()
            )
            
            # 新增：特征增强 - 精细调整文本-图像对齐
            self.img_enhance = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
            
            self.text_enhance = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
            
            # 维度投影（如果需要）
            self.text_projection = nn.Linear(embed_dim, embed_dim)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
    def forward(self, img_feat, text_feat, need_weights=False):
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
                
            # 新增：特征增强 - 提升表达能力
            img_feat_enhanced = self.img_enhance(img_feat)
            text_feat_enhanced = self.text_enhance(text_feat)
            
            # 图像 -> 文本注意力
            identity_img = img_feat_enhanced
            identity_text = text_feat_enhanced
            
            # 文本引导的图像特征增强
            enhanced_img, img_weights = self.text_to_img_attn(
                img_feat_enhanced, text_feat_enhanced, text_feat_enhanced, need_weights=need_weights
            )
            enhanced_img = identity_img + enhanced_img
            
            # 图像引导的文本特征
            enhanced_text, text_weights = self.img_to_text_attn(
                text_feat_enhanced, img_feat_enhanced, img_feat_enhanced, need_weights=need_weights
            )
            enhanced_text = identity_text + enhanced_text
            
            # 前馈网络处理
            img_ff_out = self.img_ff(self.img_norm1(enhanced_img))
            enhanced_img = enhanced_img + img_ff_out
            enhanced_img = self.img_norm2(enhanced_img)
            
            text_ff_out = self.text_ff(self.text_norm1(enhanced_text))
            enhanced_text = enhanced_text + text_ff_out
            enhanced_text = self.text_norm2(enhanced_text)
            
            # 新增：门控特征整合
            if need_weights:
                img_weights_avg = img_weights.mean(dim=1) if img_weights is not None else None
                text_weights_avg = text_weights.mean(dim=1) if text_weights is not None else None
                return enhanced_img, enhanced_text, (img_weights_avg, text_weights_avg)
            else:
                return enhanced_img, enhanced_text
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise


# 增强版文本条件归一化 - 更精细的文本调制
class EnhancedTextConditionedNorm(nn.Module):
    """增强的文本条件归一化层"""
    def __init__(self, channels, text_dim):
        super().__init__()
        self.channels = channels
        self.text_dim = text_dim
        
        # 从文本特征生成缩放和偏移参数 - 更复杂的映射
        self.text_mapping = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, channels * 2),
            nn.Tanh()  # 限制在[-1,1]范围内，避免过度缩放
        )
        
        # 新增：自适应系数 - 控制文本影响的强度
        self.gate = nn.Sequential(
            nn.Linear(text_dim, 1),
            nn.Sigmoid()
        )
        
        # 新增：空间注意力 - 允许更精细的空间调制
        self.spatial_modulation = nn.Sequential(
            nn.Linear(text_dim, channels),
            nn.Sigmoid()
        )
        
        # 新增：通道注意力 - 增强文本对特定通道的选择性
        self.channel_attn = nn.Sequential(
            nn.Linear(text_dim, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x, text_embedding):
        """
        Args:
            x: [B, C, H, W] - 图像特征
            text_embedding: [B, text_dim] - 文本嵌入
        """
        # 生成缩放和偏移参数
        params = self.text_mapping(text_embedding)  # [B, C*2]
        scale, shift = torch.chunk(params, 2, dim=1)  # 各 [B, C]
        
        # 调整为适当的形状
        scale = scale.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        shift = shift.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        
        # 生成自适应系数
        gate_val = self.gate(text_embedding)  # [B, 1]
        gate_val = gate_val.unsqueeze(2).unsqueeze(3)  # [B, 1, 1, 1]
        
        # 新增：通道注意力
        channel_weights = self.channel_attn(text_embedding)  # [B, C]
        channel_weights = channel_weights.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        
        # 新增：空间调制 - 生成空间注意力图
        B, C, H, W = x.shape
        
        # 应用归一化 - 文本条件的AdaIN
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x - x_mean) / x_std
        
        # 将所有调制因素结合
        modulated = x_norm * (1.0 + scale * gate_val) + shift * gate_val
        modulated = modulated * channel_weights
        
        return modulated


# 增强版区域感知文本融合 - 多层次注意力生成和特征调制
class EnhancedRegionAwareTextFusion(nn.Module):
    """增强的区域感知文本融合模块"""
    def __init__(self, channels, text_dim):
        super().__init__()
        self.channels = channels
        self.text_dim = text_dim
        
        # 新增：文本特征转换 - 多层次语义转换
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim)
        )
        
        # 新增：更复杂的注意力生成网络 - 多尺度特征融合
        self.attention_gen = nn.Sequential(
            # 第一层：特征转换
            nn.Conv2d(channels + text_dim, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            
            # 第二层：中间表示
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            
            # 第三层：注意力输出
            nn.Conv2d(channels // 2, 1, kernel_size=1)
            # 移除Sigmoid层，让输出保持为logits形式，配合BCEWithLogitsLoss使用
        )
        
        # 新增：多种文本调制特征
        self.text_modulation = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, channels),
            nn.Tanh()  # 限制在[-1,1]范围内
        )
        
        # 新增：空间感知的文本信息注入
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # 新增：注意力正则化 - 用于防止注意力崩塌
        self.attn_dropout = nn.Dropout(0.1)
        
        # 新增：残差连接门控 - 控制多少原始特征保留
        self.residual_gate = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, features, text_embedding):
        """
        Args:
            features: [B, C, H, W] - 图像特征
            text_embedding: [B, text_dim] - 文本嵌入
        """
        # 转换文本特征
        transformed_text = self.text_transform(text_embedding)  # [B, text_dim]
        
        # 生成空间感知的调制特征
        b, c, h, w = features.shape
        
        # 扩展文本特征到空间维度
        text_feat_expanded = transformed_text.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
        
        # 生成注意力图
        combined_feat = torch.cat([features, text_feat_expanded], dim=1)
        attention_logits = self.attention_gen(combined_feat)  # [B, 1, H, W] - logits
        
        # 应用注意力dropout - 增加随机性，防止崩塌
        attention_logits = self.attn_dropout(attention_logits)
        
        # 在前向传播时应用sigmoid，以便于可视化和其他用途
        attention_map = torch.sigmoid(attention_logits)  # [B, 1, H, W]
        
        # 生成文本调制特征
        text_feat_mod = self.text_modulation(text_embedding)  # [B, C]
        text_feat_mod = text_feat_mod.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        
        # 应用空间门控
        spatial_importance = self.spatial_gate(features)  # [B, C, H, W]
        
        # 将所有调制因素整合
        # 1. 注意力引导的特征增强
        attention_enhanced = features * attention_map * text_feat_mod
        
        # 2. 空间重要性加权
        spatially_modulated = features * spatial_importance
        
        # 3. 残差连接 - 自适应混合原始和增强特征
        gate = torch.sigmoid(self.residual_gate)
        enhanced_features = features + gate * attention_enhanced + (1-gate) * spatially_modulated
        
        return enhanced_features, attention_logits  # 返回logits用于损失计算


# 增强版文本注意力块 - 更强大的文本引导和特征融合
class EnhancedTextAttentionBlock(nn.Module):
    """增强的文本注意力块"""
    def __init__(self, nf, text_dim, num_heads=8):
        super().__init__()
        try:
            # 使用增强版模块替换原始模块
            self.text_fusion = EnhancedRegionAwareTextFusion(nf, text_dim)
            self.cross_attn = EnhancedBidirectionalCrossAttention(nf, num_heads)
            self.text_norm = EnhancedTextConditionedNorm(nf, text_dim)
            
            # 文本特征转换层
            self.text_hidden_projection = nn.Sequential(
                nn.Linear(text_dim, text_dim),
                nn.LayerNorm(text_dim),
                nn.ReLU(),
                nn.Linear(text_dim, nf)
            )
            
            # 特征转换
            self.img_to_emb = nn.Sequential(
                nn.Conv2d(nf, nf, kernel_size=1),
                nn.BatchNorm2d(nf),
                nn.ReLU(inplace=True)
            )
            
            self.emb_to_img = nn.Sequential(
                nn.Conv2d(nf, nf, kernel_size=1),
                nn.BatchNorm2d(nf),
                nn.ReLU(inplace=True)
            )
            
            # 增强版位置编码
            self.pos_encoder = ConditionalPositionalEncoding2D(nf)
            
            # 新增：特征增强残差模块
            self.feature_enhance = nn.Sequential(
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.BatchNorm2d(nf),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.BatchNorm2d(nf)
            )
            
            # 新增：自注意力 - 增强空间上下文建模
            self.self_attn = EnhancedMultiHeadAttention(nf, num_heads//2)
            
            # 新增：注意力融合门控
            self.attn_fusion_gate = nn.Parameter(torch.ones(1) * 0.5)
            
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
            # 添加条件位置编码
            b, c, h, w = features.shape
            pos_encoding = self.pos_encoder(features, text_pooled)
            
            # 确保位置编码维度与特征维度匹配
            if pos_encoding.shape[1] != c:
                pos_encoding = pos_encoding[:, :c, :, :]
            
            # 应用位置编码
            features = features + pos_encoding
            
            # 1. 区域感知文本融合
            enhanced_features, attention_logits = self.text_fusion(features, text_pooled)
            
            # 2. 自注意力增强 - 提升空间上下文建模
            b, c, h, w = enhanced_features.shape
            features_flat = enhanced_features.reshape(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
            
            # 应用自注意力
            self_attended, _ = self.self_attn(features_flat, features_flat, features_flat)
            self_attended = self_attended.permute(0, 2, 1).reshape(b, c, h, w)
            
            # 门控融合
            gate = torch.sigmoid(self.attn_fusion_gate)
            enhanced_features = enhanced_features + gate * self_attended
            
            # 3. 转换为序列形式，用于交叉注意力
            img_seq = self.img_to_emb(enhanced_features).reshape(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
            
            # 4. 处理文本特征
            if text_hidden.shape[-1] != c:
                text_hidden_projected = self.text_hidden_projection(text_hidden)
            else:
                text_hidden_projected = text_hidden
            
            # 5. 交叉注意力处理
            enhanced_img_feat, enhanced_text_feat = self.cross_attn(img_seq, text_hidden_projected)
            
            # 6. 重塑增强图像特征回空间维度
            enhanced_img_feat = enhanced_img_feat.permute(0, 2, 1).reshape(b, c, h, w)
            enhanced_img_feat = self.emb_to_img(enhanced_img_feat)
            
            # 7. 应用特征增强残差模块
            residual_features = self.feature_enhance(enhanced_img_feat)
            enhanced_img_feat = enhanced_img_feat + residual_features
            
            # 8. 应用文本条件归一化
            normalized_features = self.text_norm(enhanced_img_feat, text_pooled)
            
            # 9. 最终残差连接
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
                num_heads=8,
                with_position=False):
        super().__init__()
        
        try:
            # 支持位置信息
            self.with_position = with_position
            
            # 新增：文本特征增强 - 增强CLIP特征的表达能力
            self.text_enhancer = nn.Sequential(
                nn.Linear(text_dim, text_dim),
                nn.LayerNorm(text_dim),
                nn.GELU(),
                nn.Linear(text_dim, text_dim),
                nn.LayerNorm(text_dim)
            )
            
            # 新增：文本池化特征增强
            self.pooled_enhancer = nn.Sequential(
                nn.Linear(text_dim, text_dim),
                nn.LayerNorm(text_dim),
                nn.GELU(),
                nn.Linear(text_dim, text_dim),
                nn.LayerNorm(text_dim)
            )
            
            # 文本注意力模块 - 使用增强版
            self.text_blocks = nn.ModuleList()
            for i in range(num_blocks):
                block = EnhancedTextAttentionBlock(num_feat, text_dim, num_heads)
                self.text_blocks.append(block)
            
            # 新增：位置感知处理（如果启用）
            if with_position:
                self.position_encoder = nn.Sequential(
                    nn.Linear(num_blocks, num_feat),
                    nn.LayerNorm(num_feat),
                    nn.ReLU(),
                    nn.Linear(num_feat, num_feat)
                )
            
            # 新增：多尺度特征融合
            self.fusion = nn.ModuleList()
            for i in range(num_blocks - 1):
                self.fusion.append(nn.Sequential(
                    nn.Conv2d(num_feat * 2, num_feat, kernel_size=1),
                    nn.BatchNorm2d(num_feat),
                    nn.ReLU(inplace=True)
                ))
            
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
    
    def forward(self, features, text_hidden, text_pooled, position_info=None):
        """
        对SR特征应用文本引导
        
        Args:
            features: [B, C, H, W] - 需要增强的SR特征图
            text_hidden: [B, L, D] - 文本隐藏状态
            text_pooled: [B, D] - 池化的文本特征
            position_info: [B, num_blocks] - 位置信息（如果启用）
        
        Returns:
            enhanced_features: [B, C, H, W] - 文本增强的特征图
            attention_maps: list - 注意力图列表，用于可视化
        """
        # 增强文本特征
        enhanced_text_hidden = self.text_enhancer(text_hidden)
        enhanced_text_pooled = self.pooled_enhancer(text_pooled)
        
        attention_maps = []
        intermediate_features = []
        
        # 处理位置信息（如果启用并提供）
        position_embedding = None
        if self.with_position and position_info is not None:
            position_embedding = self.position_encoder(position_info)
            position_embedding = position_embedding.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
        
        enhanced_features = features
        
        # 应用文本注意力模块
        for i, block in enumerate(self.text_blocks):
            # 应用位置信息（如果有）
            if position_embedding is not None:
                enhanced_features = enhanced_features + position_embedding
                
            # 应用文本注意力
            enhanced_features, attn_logits = block(enhanced_features, enhanced_text_hidden, enhanced_text_pooled)
            attention_maps.append(attn_logits)
            
            # 保存中间特征用于多尺度融合
            if i < len(self.text_blocks) - 1:
                intermediate_features.append(enhanced_features)
            
            # 多尺度特征融合
            if i > 0 and i < len(self.text_blocks) - 1:
                # 融合当前特征和早期特征
                early_feat = intermediate_features[i-1]
                enhanced_features = self.fusion[i-1](torch.cat([enhanced_features, early_feat], dim=1))
        
        return enhanced_features, attention_maps 