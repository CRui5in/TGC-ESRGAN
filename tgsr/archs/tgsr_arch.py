import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.rrdbnet_arch import RRDB
from copy import deepcopy
from basicsr.utils import get_root_logger

def copy_module(module):
    """深度复制模块及其参数"""
    return deepcopy(module)


@ARCH_REGISTRY.register()
class MultiControlNet(nn.Module):
    """多控制映射的ControlNet实现
    
    使用二通道控制信号(Canny、深度图)和文本引导，
    复制浅层网络（conv_first和前八个RRDB块）作为可训练部分，
    最后一个RRDB块的输出会送给原始RRDBNet主干网络的第九个块
    """
    def __init__(self, 
                orig_net_g=None,
                num_feat=64, 
                text_dim=512, 
                num_heads=8,
                with_position=False,
                control_in_channels=2):  # Canny + 深度图
        super().__init__()
        
        self.with_position = with_position
        
        # 控制输入编码器
        self.control_encoder = nn.Sequential(
            nn.Conv2d(control_in_channels, num_feat, kernel_size=3, padding=1),
            nn.GroupNorm(8, num_feat),
            nn.SiLU(),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
            nn.GroupNorm(8, num_feat),
            nn.SiLU()
        )
        
        # 特征自注意力
        self.control_self_attention = MultiHeadSelfAttention(
            num_feat, num_heads=num_heads, dropout=0.1
        )
        
        # 文本特征投影 - 简化文本处理
        # 直接使用CLIP特征，不需要额外编码
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, num_feat),
            nn.LayerNorm(num_feat),
            nn.SiLU(),
            nn.Dropout(0.05)
        )
        
        # 复制浅层网络作为可训练部分
        if orig_net_g is not None:
            # 复制conv_first为可训练副本
            self.conv_first_copy = copy_module(orig_net_g.conv_first)
            
            # 固定前八个RRDB块
            trainable_blocks = 8
            if len(orig_net_g.body) < trainable_blocks:
                trainable_blocks = len(orig_net_g.body)
                print(f"警告：原始网络RRDB块少于8个，只复制 {trainable_blocks} 个块")
            
            # 复制前八个RRDB块为可训练副本
            self.body_copy = nn.ModuleList()
            for i in range(trainable_blocks):
                self.body_copy.append(copy_module(orig_net_g.body[i]))
            
            # 存储原始网络的参数
            self.orig_net_g = orig_net_g
            self.trainable_blocks = trainable_blocks
            
            # 确保所有其他模块都是可以访问的
            self.orig_total_blocks = len(orig_net_g.body)
        
        # 零初始化控制块
        self.zero_conv_first = ZeroConvBlock(num_feat, num_feat, text_dim)
        
        # RRDB块的控制块 - 仅为可训练副本的块创建
        self.zero_conv_body = nn.ModuleList()
        if orig_net_g is not None:
            for _ in range(trainable_blocks):
                self.zero_conv_body.append(ZeroConvBlock(num_feat, num_feat, text_dim))
        
        # 交叉注意力模块
        self.cross_attention = nn.ModuleList()
        self.cross_attention.append(CrossAttention(num_feat, text_dim, num_heads))
        
        # 为可训练RRDB块创建交叉注意力
        if orig_net_g is not None:
            for _ in range(trainable_blocks):
                self.cross_attention.append(CrossAttention(num_feat, text_dim, num_heads))
        
        # 注意力生成器
        self.attention_generators = nn.ModuleList()
        if orig_net_g is not None:
            for _ in range(trainable_blocks + 1):  # +1 是为了conv_first
                self.attention_generators.append(
                    nn.Sequential(
                        nn.Conv2d(num_feat, num_feat//2, kernel_size=3, padding=1),
                        nn.GroupNorm(4, num_feat//2),
                        nn.SiLU(),
                        nn.Conv2d(num_feat//2, num_feat//4, kernel_size=3, padding=1),
                        nn.GroupNorm(2, num_feat//4),
                        nn.SiLU(),
                        nn.Conv2d(num_feat//4, 1, kernel_size=1)
                    )
                )
        
        # 控制信号处理分支
        self.canny_branch = nn.Sequential(
            nn.Conv2d(1, num_feat//4, kernel_size=3, padding=1),
            nn.GroupNorm(2, num_feat//4),
            nn.SiLU(),
            nn.Conv2d(num_feat//4, num_feat//2, kernel_size=3, padding=1),
            nn.GroupNorm(4, num_feat//2),
            nn.SiLU(),
            nn.Conv2d(num_feat//2, num_feat, kernel_size=3, padding=1),
            nn.GroupNorm(8, num_feat),
            nn.SiLU()
        )
        
        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, num_feat//4, kernel_size=3, padding=1),
            nn.GroupNorm(2, num_feat//4),
            nn.SiLU(),
            nn.Conv2d(num_feat//4, num_feat//2, kernel_size=3, padding=1),
            nn.GroupNorm(4, num_feat//2),
            nn.SiLU(),
            nn.Conv2d(num_feat//2, num_feat, kernel_size=3, padding=1),
            nn.GroupNorm(8, num_feat),
            nn.SiLU()
        )
        
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
        
        # 零初始化
        for m in self.modules():
            if isinstance(m, ZeroConvBlock) and hasattr(m, 'ensure_zero_init'):
                m.ensure_zero_init()
    
    def forward(self, x, control_map, text_hidden, text_pooled, position_info=None):
        """前向传播，生成控制信号并处理副本网络
        
        Args:
            x: 输入的LQ图像 [B, C, H, W]
            control_map: 控制映射 [B, 2, H, W]
            text_hidden: 文本隐藏状态 [B, L, D]
            text_pooled: 池化文本特征 [B, D]
            position_info: 位置信息 [B, num_blocks]
            
        Returns:
            out_features: 处理后的特征
            attention_maps: 注意力图列表
        """
        # 调整控制映射尺寸
        if control_map.shape[2:] != x.shape[2:]:
            control_map = F.interpolate(
                control_map, 
                size=(x.shape[2], x.shape[3]), 
                mode='bilinear' if control_map.shape[1] > 1 else 'nearest',
                align_corners=False if control_map.shape[1] > 1 else None
            )
        
        # 分离控制通道
        canny_map = control_map[:, 0:1]
        depth_map = control_map[:, 1:2]
        
        # 处理控制信号
        canny_feat = self.canny_branch(canny_map)  # 边缘特征
        depth_feat = self.depth_branch(depth_map)  # 深度特征
        
        # 处理文本特征 - 直接使用CLIP输出的池化特征
        text_feat = self.text_projector(text_pooled)
        
        # 生成组合控制特征
        control_feat = self.control_encoder(control_map)
        control_feat = control_feat + canny_feat * 0.5 + depth_feat * 0.5
        control_feat = self.control_self_attention(control_feat)
        
        # 通过可训练副本网络传播
        # 1. 处理conv_first
        feat = self.conv_first_copy(x)
        
        # 应用第一个控制信号
        control_signal, attn_map = self._apply_control_signal(
            feat, text_hidden, text_feat, control_feat, 0
        )
        
        # 添加控制信号 - 简单的加法融合
        feat = feat + control_signal
        
        attention_maps = [attn_map]
        
        # 2. 处理可训练的RRDB块 - 前8个（或配置的数量）
        trunk = feat
        for i in range(len(self.body_copy)):
            # 通过RRDB块
            trunk = self.body_copy[i](trunk)
            
            # 应用控制信号
            control_signal, attn_map = self._apply_control_signal(
                trunk, text_hidden, text_feat, control_feat, i+1
            )
            
            # 添加控制信号 - 简单的加法融合
            trunk = trunk + control_signal
            attention_maps.append(attn_map)
        
        return trunk, attention_maps
    
    def _apply_control_signal(self, feat, text_hidden, text_feat, control_feat, idx):
        """应用控制信号
        
        Args:
            feat: 输入特征
            text_hidden: 文本隐藏特征
            text_feat: 文本特征向量
            control_feat: 控制特征
            idx: 模块索引
            
        Returns:
            control_signal: 控制信号
            attn_map: 注意力图
        """
        # 调整控制特征尺寸
        if control_feat.shape[2:] != feat.shape[2:]:
            adjusted_control_feat = F.interpolate(
                control_feat,
                size=(feat.shape[2], feat.shape[3]),
                mode="bilinear",
                align_corners=False
            )
        else:
            adjusted_control_feat = control_feat
        
        # 应用交叉注意力 - 促进文本与图像特征的对齐
        cross_feat, _ = self.cross_attention[idx](feat, text_hidden)
        
        # 融合控制特征
        cross_feat = cross_feat + adjusted_control_feat
        
        # 应用零卷积生成控制信号
        if idx == 0:
            control_signal = self.zero_conv_first(cross_feat, text_feat, text_hidden)
            attn_map = self.attention_generators[idx](control_signal)
        else:
            control_signal = self.zero_conv_body[idx-1](cross_feat, text_feat, text_hidden)
            attn_map = self.attention_generators[idx](control_signal)
        
        return control_signal, attn_map


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块"""
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 定义投影层
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """应用多头自注意力
        
        Args:
            x (Tensor): 输入特征 [B, C, H, W]
        
        Returns:
            Tensor: 增强特征 [B, C, H, W]
        """
        batch_size, c, h, w = x.shape
        
        # 投影查询、键、值
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头形式
        q = q.reshape(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # [B, h, HW, d]
        k = k.reshape(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 2, 3)  # [B, h, d, HW]
        v = v.reshape(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # [B, h, HW, d]
        
        # 计算注意力权重
        attn = torch.matmul(q, k) * self.scale  # [B, h, HW, HW]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重
        out = torch.matmul(attn, v)  # [B, h, HW, d]
        out = out.permute(0, 1, 3, 2).reshape(batch_size, c, h, w)
        
        # 输出投影
        out = self.out_proj(out)
        
        # 残差连接
        return out + x


class ZeroConvBlock(nn.Module):
    """零初始化卷积块 - ControlNet核心组件"""
    def __init__(self, in_channels, out_channels, text_dim=512):
        super().__init__()
        # 第一个卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        
        # 第二个卷积 - 真正的零初始化
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 完全零初始化权重
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        # 文本调制
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, out_channels),
            nn.SiLU()
        )
    
    def ensure_zero_init(self):
        """确保完全零初始化"""
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x, text_embedding, text_hidden=None):
        """
        前向传播
        Args:
            x: [B, C, H, W] - 输入特征
            text_embedding: [B, C] - 文本特征
            text_hidden: [B, L, D] - 文本隐藏状态（可选）
        """
        # 第一个卷积层
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # 第二个卷积层 - 零初始化
        h = self.conv2(h)
        
        return h


class CrossAttention(nn.Module):
    """交叉注意力模块 - 促进文本与图像特征的精确对齐"""
    def __init__(self, feat_dim, text_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 图像特征投影为查询
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        # 文本特征投影为键和值
        self.k_proj = nn.Linear(text_dim, feat_dim)
        self.v_proj = nn.Linear(text_dim, feat_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        
        # 层归一化
        self.norm_img = nn.LayerNorm(feat_dim)
        self.norm_text = nn.LayerNorm(text_dim)
        
        # 注意力dropout
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)
        
        # 初始化权重
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Xavier初始化
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(self, img_feat, text_feat):
        """
        应用交叉注意力
        Args:
            img_feat: [B, C, H, W] - 图像特征
            text_feat: [B, L, D] - 文本特征
        Returns:
            enhanced_img: [B, C, H, W] - 增强的图像特征
            attn_weights: [B, HW, L] - 注意力权重，用于可视化
        """
        # 空间维度展平
        b, c, h, w = img_feat.shape
        img_flat = img_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        
        # 特征归一化
        img_norm = self.norm_img(img_flat)
        text_norm = self.norm_text(text_feat)
        
        # 投影为查询、键、值
        q = self.q_proj(img_norm)  # [B, HW, C]
        k = self.k_proj(text_norm)  # [B, L, C]
        v = self.v_proj(text_norm)  # [B, L, C]
        
        # 重塑为多头形式
        q = q.reshape(b, h*w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, HW, d]
        k = k.reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, L, d]
        v = v.reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, L, d]
        
        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, HW, L]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # 应用注意力权重
        out = torch.matmul(attn, v)  # [B, h, HW, d]
        out = out.permute(0, 2, 1, 3).reshape(b, h*w, c)  # [B, HW, C]
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        
        # 残差连接
        out = out + img_flat
        
        # 恢复空间维度
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        
        # 用于注意力图可视化（取平均注意力权重）
        attn_weights = attn.mean(1).permute(0, 2, 1)  # [B, L, HW]
        
        return out, attn_weights