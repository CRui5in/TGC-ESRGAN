import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class TextGuidanceNet(nn.Module):
    """文本引导网络，使用ControlNet架构思想"""
    def __init__(self, 
                num_feat=64, 
                text_dim=512, 
                num_blocks=3, 
                num_heads=8,
                with_position=False):
        super().__init__()
        
        # 支持位置信息
        self.with_position = with_position
        
        # 文本特征编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim)
        )
        
        # 池化特征编码器
        self.pooled_encoder = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim)
        )
        
        # 文本投影到特征空间
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, num_feat),
            nn.LayerNorm(num_feat),
            nn.SiLU()
        )
        
        # 零初始化控制块 - ControlNet核心思想
        self.zero_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.zero_blocks.append(ZeroConvBlock(num_feat, num_feat))
            
        # 位置编码器（如果启用）
        if with_position:
            self.position_encoder = nn.Sequential(
                nn.Linear(num_blocks, num_feat),
                nn.LayerNorm(num_feat),
                nn.SiLU(),
                nn.Linear(num_feat, num_feat)
            )
            
        # 注意力生成器
        self.attention_generators = nn.ModuleList()
        for i in range(num_blocks):
            self.attention_generators.append(
                nn.Sequential(
                    nn.Conv2d(num_feat, num_feat//2, kernel_size=3, padding=1),
                    nn.GroupNorm(4, num_feat//2),
                    nn.SiLU(),
                    nn.Conv2d(num_feat//2, 1, kernel_size=1)
                )
            )
            
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 确保零初始化块的最后一层确实是零初始化
        for block in self.zero_blocks:
            if hasattr(block, 'ensure_zero_init'):
                block.ensure_zero_init()
    
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
        # 处理文本特征
        text_hidden = self.text_encoder(text_hidden)
        text_pooled = self.pooled_encoder(text_pooled)
        
        # 投影文本特征到特征空间
        text_feat = self.text_projector(text_pooled)
        
        # 增加文本特征与图像特征的交互 - 计算文本特征的通道注意力
        b, c = text_feat.size()
        # 生成通道注意力
        text_channel_attn = torch.sigmoid(text_feat).view(b, c, 1, 1)
        
        # 准备位置编码（如果启用）
        position_embedding = None
        if self.with_position and position_info is not None:
            position_embedding = self.position_encoder(position_info)
            position_embedding = position_embedding.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 应用控制块
        attention_maps = []
        # 使用通道注意力增强初始特征
        enhanced = features.clone() * (1.0 + text_channel_attn * 0.2)
        
        # 在不同层级应用不同程度的文本引导
        attention_weights = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]  # 从早期层到后期层逐渐减弱
        
        for i, zero_block in enumerate(self.zero_blocks):
            # 处理当前块
            if position_embedding is not None:
                block_input = enhanced + position_embedding
            else:
                block_input = enhanced
            
            # 应用零初始化控制块 - ControlNet方式
            control_signal = zero_block(block_input, text_feat)
            
            # 通道平衡处理 - 在生成注意力图之前确保控制信号的通道平衡
            b, c, h, w = control_signal.shape
            if c % 3 == 0:
                # 分组处理RGB通道组
                channel_groups = c // 3
                control_rgb = control_signal.view(b, 3, channel_groups, h, w)
                
                # 计算每个RGB通道的均值
                rgb_means = control_rgb.mean(dim=[2, 3, 4], keepdim=True)  # [b, 3, 1, 1, 1]
                global_mean = rgb_means.mean(dim=1, keepdim=True)  # [b, 1, 1, 1, 1]
                
                # 对不平衡的通道进行归一化
                normalized_rgb = control_rgb * (global_mean / (rgb_means + 1e-8))
                
                # 重塑回原始尺寸
                control_signal = normalized_rgb.view(b, c, h, w)
            
            # 生成注意力图
            attention_logits = self.attention_generators[i](control_signal)
            attention_maps.append(attention_logits)
            
            # 应用控制信号 - 使用层级权重
            layer_weight = attention_weights[i] if i < len(attention_weights) else 0.1
            attention = torch.sigmoid(attention_logits)
            enhanced = enhanced + control_signal * attention * layer_weight
            
        return enhanced, attention_maps


class ZeroConvBlock(nn.Module):
    """零初始化卷积块 - ControlNet核心组件"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 使用非零但很小的初始化，而不是完全的零初始化
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        nn.init.constant_(self.conv2.bias, 0)
        
        # 文本调制
        self.text_modulation = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels * 2),
            nn.Tanh()
        )
    
    def ensure_zero_init(self):
        """使用小值初始化而非纯零值"""
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        nn.init.constant_(self.conv2.bias, 0)
    
    def forward(self, x, text_embedding):
        """
        前向传播
        Args:
            x: [B, C, H, W] - 输入特征
            text_embedding: [B, C] - 文本特征
        """
        # 第一个卷积
        h = self.conv1(x)
        h = self.norm1(h)
        
        # 文本调制 - 计算缩放和偏移
        text_params = self.text_modulation(text_embedding) # [B, C*2]
        scale, shift = torch.chunk(text_params, 2, dim=1)  # 各 [B, C]
        
        # 改进通道均衡 - 使用通道归一化而非硬编码权重
        c = scale.size(1)
        if c % 3 == 0:  # 确保通道数是3的倍数，对应RGB通道
            channel_groups = c // 3
            
            # 将scale重塑为RGB通道组
            scale_view = scale.view(scale.size(0), 3, channel_groups)
            
            # 计算每个样本中RGB通道的均值
            scale_mean = scale_view.mean(dim=2, keepdim=True)  # [B, 3, 1]
            
            # 计算RGB通道均值间的统计
            rgb_mean = scale_mean.mean(dim=1, keepdim=True)  # [B, 1, 1]
            
            # 归一化RGB通道组间的缩放，使它们均值一致
            # 这确保红色通道不会获得过高增益
            scale_norm = scale_view * (rgb_mean / (scale_mean + 1e-8))
            
            # 重塑回原始维度
            scale = scale_norm.view(scale.size(0), -1)
            
            # 同样处理shift - 防止颜色通道偏移
            shift_view = shift.view(shift.size(0), 3, channel_groups)
            
            # 计算偏移均值
            shift_mean = shift_view.mean(dim=2, keepdim=True)  # [B, 3, 1]
            
            # 计算通道均值
            shift_rgb_mean = shift_mean.mean(dim=1, keepdim=True)  # [B, 1, 1]
            
            # 归一化RGB通道偏移量
            shift_norm = shift_view - (shift_mean - shift_rgb_mean)
            
            # 重塑回原始维度
            shift = shift_norm.view(shift.size(0), -1)
        
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 应用缩放和偏移
        h = h * (scale + 1.0) + shift
        
        # 激活和第二个卷积
        h = self.act1(h)
        h = self.conv2(h)  # 零初始化权重，初始时不影响原始特征
        
        return h 