import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn.modules.utils import _pair

@ARCH_REGISTRY.register()
class TextGuidedSRNet(nn.Module):
    """基于文本引导的超分辨率网络"""
    
    def __init__(self, 
                num_in_ch=3, 
                num_out_ch=3, 
                scale=4,
                num_feat=64, 
                num_blocks=16, 
                text_dim=512,
                hidden_dim=64,
                use_text_proj=True):
        super(TextGuidedSRNet, self).__init__()
        self.scale = scale
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # 文本特征处理
        if use_text_proj:
            self.text_projection = nn.Linear(text_dim, hidden_dim)
        else:
            assert text_dim == hidden_dim, f"text_dim ({text_dim}) must equal hidden_dim ({hidden_dim}) when use_text_proj is False"
            self.text_projection = nn.Identity()
        
        # 用于存储注意力图
        self.store_attention_maps = False
        self.attention_maps = []
        
        # 特征提取
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # 主体网络
        main_blocks = []
        for _ in range(num_blocks):
            main_blocks.append(TextAttentionBlock(num_feat, self.hidden_dim, num_heads=8))
        self.main_blocks = nn.Sequential(*main_blocks)
        
        # 重建部分
        upscale_blocks = []
        for _ in range(int(math.log(scale, 2))):
            upscale_blocks.extend([
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(True)
            ])
        self.upscale = nn.Sequential(*upscale_blocks)
        
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
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
    
    def forward(self, x, text_hidden=None, text_pooled=None):
        """前向传播
        
        Args:
            x (Tensor): 输入的低分辨率图像
            text_hidden (Tensor): 文本的隐藏状态表示
            text_pooled (Tensor): 文本的池化表示
            
        Returns:
            Tensor: 生成的高分辨率图像
        """
        # 清空之前存储的注意力图
        self.attention_maps = []
        
        # 投影文本特征
        if text_hidden is not None:
            if len(text_hidden.shape) == 3:  # [B, L, C]
                text_features = self.text_projection(text_hidden)
            else:  # [B, C]
                text_features = self.text_projection(text_hidden.unsqueeze(1)).squeeze(1)
        else:
            text_features = None
        
        # 初始特征提取
        feat = self.conv_first(x)
        
        # 主体处理，传递文本特征
        for i, block in enumerate(self.main_blocks):
            # 对TextAttentionBlock类型的块，传递文本特征并收集注意力图
            if isinstance(block, TextAttentionBlock) and text_features is not None:
                # 启用或禁用注意力图收集
                if hasattr(block, 'enable_attention_maps'):
                    block.enable_attention_maps(self.store_attention_maps)
                
                # 特征传递    
                feat = block(feat, text_features)
                
                # 收集注意力图
                if self.store_attention_maps and hasattr(block, 'get_attention_maps'):
                    attn_maps = block.get_attention_maps()
                    if attn_maps:
                        self.attention_maps.extend(attn_maps)
                        # 清空块中的注意力图以避免内存泄漏
                        block.clear_attention_maps()
            else:
                # 普通块直接处理特征
                feat = block(feat)
        
        # 上采样
        feat = self.upscale(feat)
        
        # 最终卷积
        out = self.conv_last(feat)
        
        return out
    
    def get_attention_maps(self):
        """获取收集的注意力图
        
        Returns:
            注意力图列表
        """
        return self.attention_maps if hasattr(self, 'attention_maps') else []
    
    def enable_attention_maps(self, enable=True):
        """启用或禁用注意力图收集
        
        Args:
            enable: 是否启用注意力图收集
        """
        self.store_attention_maps = enable
        if enable:
            # 清空之前存储的注意力图
            self.attention_maps = []
        
        # 对所有TextAttentionBlock启用或禁用注意力图收集
        for module in self.modules():
            if isinstance(module, TextAttentionBlock) and hasattr(module, 'enable_attention_maps'):
                module.enable_attention_maps(enable)


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
            text_features: 文本特征 [B, D]或[B, L, D]
            
        Returns:
            处理后的特征
        """
        # 如果没有文本特征，则直接返回输入
        if text_features is None:
            return x
            
        # 检查并修复NaN输入
        if torch.isnan(x).any():
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        if torch.isnan(text_features).any():
            text_features = torch.where(torch.isnan(text_features), torch.zeros_like(text_features), text_features)
        
        batch_size, c, h, w = x.shape
        
        # 确保文本特征是3D张量 [B, L, D]
        if len(text_features.shape) == 2:  # [B, D]
            text_features = text_features.unsqueeze(1)  # [B, 1, D]
        
        # 批次大小检查和处理
        if text_features.size(0) != batch_size:
            if text_features.size(0) > batch_size:
                text_features = text_features[:batch_size]
            else:
                text_features = text_features.repeat(batch_size // text_features.size(0) + 1, 1, 1)[:batch_size]
        
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
            k = self.key_proj(text_features)  # [B, L, C]
            v = self.value_proj(text_features)  # [B, L, C]
            
            # 对键K应用层归一化
            k_norm = self.norm2(k)  # [B, L, C]
            
            # 重塑为多头格式
            text_len = k.size(1)
            k_heads = k_norm.view(batch_size, text_len, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v_heads = v.view(batch_size, text_len, self.num_heads, head_dim).permute(0, 2, 1, 3)
            # k_heads, v_heads: [B, num_heads, L, head_dim]
            
            # 计算注意力分数
            scale = head_dim ** -0.5
            attn = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale  # [B, num_heads, H*W, L]
            
            # 应用softmax
            attn_weights = F.softmax(attn, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # 收集注意力图，如果启用
            if self._store_attn:
                # 计算平均注意力权重
                avg_attn = attn_weights.mean(dim=1)  # [B, H*W, L]
                avg_attn = avg_attn.mean(dim=-1)  # [B, H*W]
                
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