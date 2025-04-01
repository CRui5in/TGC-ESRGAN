import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class TGSR(nn.Module):
    """文本引导超分辨率网络（Text-Guided Super-Resolution Network）"""
    
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32, 
                 use_attention=True, text_dim=512, num_heads=8):
        super(TGSR, self).__init__()
        self.scale = scale
        self.use_attention = use_attention
        
        # 特征提取模块
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList()
        
        # 主体残差块
        for i in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch))
        
        # 卷积后的主体输出
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # 上采样模块
        self.upsample = Upsample(scale, num_feat)
        
        # 最终输出层
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        # 激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # 文本注意力模块
        if use_attention:
            self.text_blocks = nn.ModuleList()
            for i in range(min(4, num_block)):
                self.text_blocks.append(TextAttentionBlock(num_feat, text_dim, num_heads))
        
        # 用于存储注意力图的列表
        self.attention_maps = []
        
    def forward(self, x, text_features=None):
        """前向传播函数
        
        Args:
            x (Tensor): 输入图像
            text_features (Tensor, 可选): 文本特征，如果use_attention为True则必需
            
        Returns:
            Tensor: 超分辨率输出
        """
        # 清空之前的注意力图
        self.attention_maps = []
        
        # 初始特征提取
        feat = self.conv_first(x)
        
        # 主体模块
        block_feats = []
        for idx, block in enumerate(self.body):
            feat = block(feat)
            
            # 在特定的块之后添加文本注意力
            if self.use_attention and text_features is not None and idx % 5 == 4 and idx < 20:
                block_idx = idx // 5
                if block_idx < len(self.text_blocks):
                    # 启用注意力图收集
                    self.text_blocks[block_idx].enable_attention_maps(True)
                    # 应用文本注意力
                    feat = self.text_blocks[block_idx](feat, text_features)
                    # 获取生成的注意力图
                    attn_maps = self.text_blocks[block_idx].get_attention_maps()
                    if attn_maps:
                        self.attention_maps.extend(attn_maps)
                    # 清空块中的注意力图以避免内存泄漏
                    self.text_blocks[block_idx].clear_attention_maps()
            
            # 存储用于密集连接的特征
            if (idx + 1) % 4 == 0 and len(block_feats) < 3:
                block_feats.append(feat)
        
        # 残差连接
        feat = self.conv_body(feat)
        feat = feat + self.conv_first(x)
        
        # 上采样
        feat = self.upsample(feat)
        
        # 最终处理
        feat = self.lrelu(self.conv_hr(feat))
        feat = self.conv_last(feat)
        
        # 为可能的GAN训练添加最后一层激活
        out = torch.clamp(feat, 0, 1)
        
        return out


class EnhancedMultiHeadAttention(nn.Module):
    """增强的多头注意力机制"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(EnhancedMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 头间交互
        self.head_mixing = nn.Parameter(torch.ones(num_heads, num_heads) / num_heads)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        q_seq_len = query.size(1)
        k_seq_len = key.size(1)

        # 投影
        q = self.q_proj(query).view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, k_seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, k_seq_len, self.num_heads, self.head_dim)

        # 转置以便进行注意力计算
        q = q.transpose(1, 2)  # (batch_size, num_heads, q_seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用头间交互
        mixed_weights = torch.einsum('bhij,gh->bgij', attn_weights, F.softmax(self.head_mixing, dim=-1))

        # 应用注意力
        context = torch.matmul(mixed_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.embed_dim)

        # 输出投影
        output = self.out_proj(context)
        return output


class PositionalEncoding2D(nn.Module):
    """2D位置编码，为特征图添加位置信息"""
    
    def __init__(self, channels):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels

    def forward(self, x):
        batch_size, c, h, w = x.shape

        # 创建位置编码矩阵 - 使用更稳定的实现
        pos_x = torch.linspace(0, 1, w, device=x.device).unsqueeze(0).repeat(h, 1)
        pos_y = torch.linspace(0, 1, h, device=x.device).unsqueeze(1).repeat(1, w)

        # 扩展为4D张量
        pos_x = pos_x.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.channels // 2, 1, 1)
        pos_y = pos_y.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.channels // 2, 1, 1)

        # 简化的位置编码 - 直接连接x和y位置
        pe = torch.cat([pos_x, pos_y], dim=1)

        # 如果通道数为奇数，填充最后一个通道
        if self.channels % 2 == 1:
            pe = F.pad(pe, (0, 0, 0, 0, 0, 1))

        # 使用非常弱化的位置编码，避免干扰原始特征
        return x + pe * 0.05  # 大幅降低位置编码的影响


class TextAttentionBlock(nn.Module):
    """文本引导的注意力块，从train_example.py复制实现"""
    
    def __init__(self, in_channels, text_dim, num_heads=8):
        super(TextAttentionBlock, self).__init__()
        
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.num_heads = num_heads
        
        # 图像特征投影层
        self.query_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 文本特征投影层
        self.key_proj = nn.Linear(text_dim, in_channels)
        self.value_proj = nn.Linear(text_dim, in_channels)
        
        # 输出投影层
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        
        # 残差连接标志
        self.use_residual = True
        
        # 添加位置编码
        self.pos_encoder = PositionalEncoding2D(in_channels)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)
        
        # 为可视化添加属性
        self._store_attn = False
        self._attn_maps = []
        
    def clear_attention_maps(self):
        """清除保存的注意力图"""
        self._attn_maps = []
    
    def enable_attention_maps(self, enable=True):
        """启用或禁用注意力图收集"""
        self._store_attn = enable
    
    def get_attention_maps(self):
        """获取收集的注意力图"""
        return self._attn_maps
    
    def forward(self, x, text_features):
        """前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
            text_features: 文本特征 [B, D]
            
        Returns:
            处理后的特征
        """
        # 检查并修复NaN输入
        if torch.isnan(x).any():
            print("注意力模块: 输入x包含NaN")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        if torch.isnan(text_features).any():
            print("注意力模块: 文本特征包含NaN")
            text_features = torch.where(torch.isnan(text_features), torch.zeros_like(text_features), text_features)
        
        batch_size, c, h, w = x.shape
        
        # 批次大小检查和处理
        if text_features.size(0) != batch_size:
            if text_features.size(0) > batch_size:
                text_features = text_features[:batch_size]
            else:
                text_features = text_features.repeat(batch_size // text_features.size(0) + 1, 1)[:batch_size]
        
        try:
            # 投影查询 Q
            q = self.query_proj(x)  # [B, C, H, W]
            q = self.pos_encoder(q)  # 添加位置编码
            
            # 重排为注意力计算格式
            q_flat = q.reshape(batch_size, c, -1).permute(0, 2, 1)  # [B, H*W, C]
            
            # 应用层归一化
            q_norm = self.norm1(q_flat)  # [B, H*W, C]
            
            # 重塑为多头格式
            head_dim = c // self.num_heads
            q_heads = q_norm.view(batch_size, h*w, self.num_heads, head_dim).permute(0, 2, 1, 3)
            # q_heads: [B, num_heads, H*W, head_dim]
            
            # 投影键 K 和值 V
            k = self.key_proj(text_features).unsqueeze(1)  # [B, 1, C]
            v = self.value_proj(text_features).unsqueeze(1)  # [B, 1, C]
            
            # 层归一化
            k_norm = self.norm2(k)  # [B, 1, C]
            
            # 重塑为多头格式
            k_heads = k_norm.view(batch_size, 1, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v_heads = v.view(batch_size, 1, self.num_heads, head_dim).permute(0, 2, 1, 3)
            # k_heads, v_heads: [B, num_heads, 1, head_dim]
            
            # 计算注意力分数
            scale = head_dim ** -0.5
            attn = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale  # [B, num_heads, H*W, 1]
            
            # 应用softmax
            attn_weights = F.softmax(attn, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # 收集注意力图，如果启用
            if self._store_attn:
                # 计算平均注意力权重
                avg_attn = attn_weights.mean(dim=1)  # [B, H*W, 1]
                avg_attn = avg_attn.squeeze(-1)  # [B, H*W]
                
                # 重塑为空间尺寸
                attn_map = avg_attn.view(batch_size, h, w).unsqueeze(1)  # [B, 1, H, W]
                self._attn_maps.append(attn_map.detach())
            
            # 应用注意力权重
            out = torch.matmul(attn_weights, v_heads)  # [B, num_heads, H*W, head_dim]
            
            # 重组为原始形状
            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, h*w, c)
            
            # 重塑为特征图
            out = out.permute(0, 2, 1).view(batch_size, c, h, w)
            
            # 输出投影
            out = self.output_proj(out)
            out = self.proj_dropout(out)
            
            # 残差连接
            if self.use_residual:
                out = out + x
            
            return out
            
        except Exception as e:
            print(f"注意力模块计算出错: {e}")
            # 出错时返回输入特征作为备用策略
            return x 