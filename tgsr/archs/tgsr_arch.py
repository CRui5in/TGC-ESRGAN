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
        
        # 准备位置编码（如果启用）
        position_embedding = None
        if self.with_position and position_info is not None:
            position_embedding = self.position_encoder(position_info)
            position_embedding = position_embedding.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 应用控制块
        attention_maps = []
        enhanced = features.clone()
        
        for i, zero_block in enumerate(self.zero_blocks):
            # 处理当前块
            if position_embedding is not None:
                block_input = enhanced + position_embedding
            else:
                block_input = enhanced
            
            # 应用零初始化控制块 - ControlNet方式
            control_signal = zero_block(block_input, text_feat)
            
            # 生成注意力图
            attention_logits = self.attention_generators[i](control_signal)
            attention_maps.append(attention_logits)
            
            # 应用控制信号
            attention = torch.sigmoid(attention_logits)
            enhanced = enhanced + control_signal * attention
            
        return enhanced, attention_maps


class ZeroConvBlock(nn.Module):
    """零初始化卷积块 - ControlNet核心组件"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 零初始化最后一层
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        # 文本调制
        self.text_modulation = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels * 2),
            nn.Tanh()
        )
    
    def ensure_zero_init(self):
        """确保最后一层确实是零初始化的"""
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
    
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
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 应用缩放和偏移
        h = h * (scale + 1.0) + shift
        
        # 激活和第二个卷积
        h = self.act1(h)
        h = self.conv2(h)  # 零初始化权重，初始时不影响原始特征
        
        return h 