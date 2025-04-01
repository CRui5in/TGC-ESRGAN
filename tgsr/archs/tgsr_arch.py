import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

# 导入循环引用的类定义
# 基础位置编码
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        _, _, h, w = x.shape
        pos_x = torch.arange(w, device=x.device).float()
        pos_y = torch.arange(h, device=x.device).float()
        
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(0)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        
        emb_x = emb_x.expand(h, -1, -1)
        emb_y = emb_y.expand(-1, w, -1)
        
        pos_emb = torch.cat((emb_y, emb_x), dim=-1)
        pos_emb = pos_emb.permute(2, 0, 1)
        pos_emb = pos_emb.unsqueeze(0)
        pos_emb = pos_emb.expand(x.shape[0], -1, -1, -1)
        
        return pos_emb

# 标准多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.shape[0]
        
        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑成多头形式
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L_q, D_h]
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L_k, D_h]
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L_v, D_h]
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L_q, L_k]
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力
        context = torch.matmul(attention, v)  # [B, H, L_q, D_h]
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)
        
        # 输出投影
        output = self.out_proj(context)
        
        return output

# 双向交叉注意力模块
class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(BidirectionalCrossAttention, self).__init__()
        self.img_to_text_attn = MultiHeadAttention(embed_dim, num_heads)
        self.text_to_img_attn = MultiHeadAttention(embed_dim, num_heads)
        self.img_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        
        # 添加特征投影层，处理文本特征维度不匹配问题
        self.text_projection = nn.Linear(512, embed_dim)  # 假设CLIP特征为512维
        
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
        self.img_ff_norm = nn.LayerNorm(embed_dim)
        self.text_ff_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, img_feat, text_feat):
        # 检查并修复NaN输入
        if torch.isnan(img_feat).any() or torch.isnan(text_feat).any():
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
        enhanced_img = img_feat_norm + self.text_to_img_attn(img_feat_norm, text_feat_norm, text_feat_norm)
        enhanced_img = enhanced_img + self.img_ff(self.img_ff_norm(enhanced_img))
        
        # 图像增强的文本特征
        enhanced_text = text_feat_norm + self.img_to_text_attn(text_feat_norm, img_feat_norm, img_feat_norm)
        enhanced_text = enhanced_text + self.text_ff(self.text_ff_norm(enhanced_text))
        
        return enhanced_img, enhanced_text

# 文本条件归一化
class TextConditionedNorm(nn.Module):
    def __init__(self, channels, text_dim):
        super(TextConditionedNorm, self).__init__()
        self.channels = channels
        self.text_dim = text_dim
        
        # 从文本特征生成缩放和偏移参数
        self.scale_shift = nn.Sequential(
            nn.Linear(text_dim, channels * 2),
            nn.Sigmoid()
        )
        
    def forward(self, x, text_embedding):
        # x: [B, C, H, W], text_embedding: [B, text_dim]
        params = self.scale_shift(text_embedding)
        scale, shift = torch.chunk(params, 2, dim=1)
        
        # 调整为适当的形状
        scale = scale.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        shift = shift.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        
        # 应用缩放和偏移
        return x * (scale + 1.0) + shift

# 区域感知文本融合
class RegionAwareTextFusion(nn.Module):
    def __init__(self, channels, text_dim):
        super(RegionAwareTextFusion, self).__init__()
        self.channels = channels
        self.text_dim = text_dim
        
        # 生成区域注意力权重
        self.attention_gen = nn.Sequential(
            nn.Conv2d(channels + text_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 生成文本调制特征
        self.text_modulation = nn.Linear(text_dim, channels)
        
    def forward(self, features, text_embedding):
        # 生成区域注意力图
        b, c, h, w = features.shape
        text_feat_expanded = text_embedding.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
        combined_feat = torch.cat([features, text_feat_expanded], dim=1)
        attention_map = self.attention_gen(combined_feat)  # [B, 1, H, W]
        
        # 文本特征调制
        text_feat_mod = self.text_modulation(text_embedding)  # [B, C]
        text_feat_mod = text_feat_mod.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        
        # 应用区域感知的文本融合
        enhanced_features = features + features * attention_map * text_feat_mod
        
        return enhanced_features, attention_map

class ResidualDenseBlock(nn.Module):
    """残差密集块"""
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """残差密集块组 (RRDB)"""
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class TextAttentionBlock(nn.Module):
    """文本注意力块"""
    def __init__(self, nf, text_dim, num_heads=8):
        super(TextAttentionBlock, self).__init__()
        self.text_fusion = RegionAwareTextFusion(nf, text_dim)
        self.cross_attn = BidirectionalCrossAttention(nf, num_heads)
        self.text_norm = TextConditionedNorm(nf, text_dim)
        
        # 文本特征转换层，用于将text_hidden从text_dim投影到nf
        self.text_hidden_projection = nn.Linear(text_dim, nf)
        
        # 特征转换
        self.img_to_emb = nn.Conv2d(nf, nf, kernel_size=1)
        self.emb_to_img = nn.Conv2d(nf, nf, kernel_size=1)
        
        # 位置编码 - 确保位置编码维度与输入特征维度一致
        self.pos_encoder = PositionalEncoding2D(nf)
        
    def forward(self, features, text_hidden, text_pooled):
        # 添加位置编码
        b, c, h, w = features.shape
        pos_encoding = self.pos_encoder(features)
        # 确保位置编码维度与特征维度匹配
        if pos_encoding.shape[1] != c:
            # 如果维度不匹配，调整位置编码的维度
            pos_encoding = pos_encoding[:, :c, :, :]
        
        features = features + pos_encoding
        
        # 区域感知文本融合
        enhanced_features, attention_map = self.text_fusion(features, text_pooled)
        
        # 将特征展平为序列形式，用于交叉注意力
        b, c, h, w = enhanced_features.shape
        img_seq = self.img_to_emb(enhanced_features).reshape(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # 处理文本特征，确保维度匹配
        # 检查text_hidden的维度，进行必要的投影
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
        
        return output_features, attention_map


@ARCH_REGISTRY.register()
class TGSRNet(nn.Module):
    """文本引导的超分辨率网络模型"""
    def __init__(self, 
                 num_in_ch=3, 
                 num_out_ch=3, 
                 scale=4, 
                 num_feat=64, 
                 num_block=23, 
                 text_dim=512,
                 use_text_features=True,
                 num_heads=8):
        super(TGSRNet, self).__init__()
        
        self.use_text_features = use_text_features
        self.text_dim = text_dim
        
        # 浅层特征提取
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        
        # 主体网络：RRDB块
        self.body = nn.ModuleList()
        for i in range(num_block):
            self.body.append(RRDB(num_feat))
        
        # 主体网络之后的卷积
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        
        # 文本条件处理
        if self.use_text_features:
            # 在不同层级添加文本注意力
            self.text_blocks = nn.ModuleList([
                TextAttentionBlock(num_feat, text_dim, num_heads),
                TextAttentionBlock(num_feat, text_dim, num_heads),
                TextAttentionBlock(num_feat, text_dim, num_heads)
            ])
        
        # 上采样
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # 重建
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        # 激活
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, text_hidden=None, text_pooled=None):
        """前向传播
        
        Args:
            x: 输入的低分辨率图像, shape: [B, 3, H, W]
            text_hidden: 文本隐藏状态特征, shape: [B, seq_len, text_dim]
            text_pooled: 文本全局特征, shape: [B, text_dim]
            
        Returns:
            超分辨率图像, shape: [B, 3, H*scale, W*scale]
        """
        # 检查输入
        if torch.isnan(x).any():
            x = torch.nan_to_num(x)
            
        # 浅层特征
        feat = self.conv_first(x)
        
        # 主体网络处理
        body_feat = feat
        attention_maps = []
        
        # 特征提取并在特定位置应用文本注意力
        if self.use_text_features and text_hidden is not None and text_pooled is not None:
            # 使用body长度作为块数量
            num_body_blocks = len(self.body)
            text_block_positions = [num_body_blocks // 4, num_body_blocks // 2, num_body_blocks * 3 // 4]
            text_block_idx = 0
            
            for i, block in enumerate(self.body):
                body_feat = block(body_feat)
                
                # 在特定位置应用文本注意力
                if i in text_block_positions and text_block_idx < len(self.text_blocks):
                    body_feat, attn_map = self.text_blocks[text_block_idx](body_feat, text_hidden, text_pooled)
                    attention_maps.append(attn_map)
                    text_block_idx += 1
        else:
            for block in self.body:
                body_feat = block(body_feat)
        
        # 残差连接
        body_feat = self.conv_body(body_feat)
        body_feat = body_feat + feat
        
        # 上采样
        sr_feat = self.upsampler(body_feat)
        
        # 重建
        sr_feat = self.lrelu(self.conv_hr(sr_feat))
        sr = self.conv_last(sr_feat)
        
        # 保存注意力图供可视化
        self.attention_maps = attention_maps if attention_maps else None
        
        return sr 