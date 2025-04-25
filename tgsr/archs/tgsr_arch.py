import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.rrdbnet_arch import RRDB
from copy import deepcopy

def copy_module(module):
    """深度复制模块及其参数"""
    return deepcopy(module)


@ARCH_REGISTRY.register()
class MultiControlNet(nn.Module):
    """多控制映射的ControlNet实现
    
    使用三通道控制信号(Canny、深度图、掩码)和文本引导
    """
    def __init__(self, 
                orig_net_g=None,
                num_feat=64, 
                text_dim=512, 
                num_blocks=6, 
                num_heads=8,
                with_position=False,
                control_in_channels=3):  # Canny + 深度图 + 掩码
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
        
        # 文本特征处理
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim)
        )
        
        # 池化特征处理
        self.pooled_encoder = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim)
        )
        
        # 文本特征投影
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, num_feat),
            nn.LayerNorm(num_feat),
            nn.SiLU(),
            nn.Dropout(0.05)
        )
        
        # 复制原始SR网络
        self.locked = True
        
        if orig_net_g is not None:
            self.conv_first = copy_module(orig_net_g.conv_first)
            
            self.body = nn.ModuleList()
            for block in orig_net_g.body:
                self.body.append(copy_module(block))
            
            self.conv_body = copy_module(orig_net_g.conv_body)
            self.conv_up1 = copy_module(orig_net_g.conv_up1)
            self.conv_up2 = copy_module(orig_net_g.conv_up2)
            
            self._lock_copied_params()
        
        # 零初始化控制块
        self.zero_conv_first = ZeroConvBlock(num_feat, num_feat, text_dim)
        
        # RRDB块的控制块(按组处理)
        self.zero_conv_body = nn.ModuleList()
        total_blocks = num_blocks
        group_size = 4
        
        for i in range(0, total_blocks, group_size):
            self.zero_conv_body.append(ZeroConvBlock(num_feat, num_feat, text_dim))
        
        # 其他控制块
        self.zero_conv_body_out = ZeroConvBlock(num_feat, num_feat, text_dim)
        self.zero_conv_up1 = ZeroConvBlock(num_feat, num_feat, text_dim)
        self.zero_conv_up2 = ZeroConvBlock(num_feat, num_feat, text_dim)
        
        # 交叉注意力模块
        self.cross_attention = nn.ModuleList()
        self.cross_attention.append(CrossAttention(num_feat, text_dim, num_heads))
        
        # 为RRDB块组创建交叉注意力
        for i in range(len(self.zero_conv_body)):
            self.cross_attention.append(CrossAttention(num_feat, text_dim, num_heads))
        
        # 为主体输出和上采样创建交叉注意力
        self.cross_attention.append(CrossAttention(num_feat, text_dim, num_heads))
        self.cross_attention.append(CrossAttention(num_feat, text_dim, num_heads))
        self.cross_attention.append(CrossAttention(num_feat, text_dim, num_heads))
        
        # 注意力生成器
        self.attention_generators = nn.ModuleList()
        for _ in range(len(self.cross_attention)):
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
        
        # 特征融合模块
        self.feature_fusion_8 = FeatureFusionModule(num_feat, num_feat)
        self.feature_fusion_16 = FeatureFusionModule(num_feat, num_feat)
        
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
    
    def _lock_copied_params(self):
        """锁定复制的网络参数"""
        if hasattr(self, 'conv_first'):
            for param in self.conv_first.parameters():
                param.requires_grad = False
        
        if hasattr(self, 'body'):
            for block in self.body:
                for param in block.parameters():
                    param.requires_grad = False
        
        if hasattr(self, 'conv_body'):
            for param in self.conv_body.parameters():
                param.requires_grad = False
        
        if hasattr(self, 'conv_up1'):
            for param in self.conv_up1.parameters():
                param.requires_grad = False
        if hasattr(self, 'conv_up2'):
            for param in self.conv_up2.parameters():
                param.requires_grad = False
    
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
    
    def load_pretrained_controlnet(self, canny_model_path=None, depth_model_path=None):
        """加载预训练ControlNet模型"""
        if canny_model_path is not None:
            try:
                canny_state_dict = torch.load(canny_model_path, map_location='cpu')
                if 'state_dict' in canny_state_dict:
                    canny_state_dict = canny_state_dict['state_dict']
                
                self._adapt_pretrained_params(canny_state_dict, self.canny_branch)
                print(f"已加载Canny预训练模型: {canny_model_path}")
            except Exception as e:
                print(f"加载Canny模型失败: {e}")
        
        if depth_model_path is not None:
            try:
                depth_state_dict = torch.load(depth_model_path, map_location='cpu')
                if 'state_dict' in depth_state_dict:
                    depth_state_dict = depth_state_dict['state_dict']
                
                self._adapt_pretrained_params(depth_state_dict, self.depth_branch)
                print(f"已加载深度预训练模型: {depth_model_path}")
            except Exception as e:
                print(f"加载深度模型失败: {e}")
    
    def _adapt_pretrained_params(self, state_dict, target_module):
        """适配预训练参数到目标模块"""
        mapped_state_dict = {}
        target_state_dict = target_module.state_dict()
        
        target_layers = list(target_state_dict.keys())
        
        # 提取编码器参数
        encoder_params = {k: v for k, v in state_dict.items() if 'input_blocks' in k and 'in_layers' in k}
        encoder_layers = sorted(encoder_params.keys())
        
        # 映射参数
        for i, (target_key, source_key) in enumerate(zip(target_layers, encoder_layers)):
            if i >= len(target_layers) or i >= len(encoder_layers):
                break
                
            target_shape = target_state_dict[target_key].shape
            source_shape = encoder_params[source_key].shape
            
            if target_shape == source_shape:
                mapped_state_dict[target_key] = encoder_params[source_key]
        
        # 加载参数
        if mapped_state_dict:
            missing, unexpected = target_module.load_state_dict(mapped_state_dict, strict=False)
            print(f"参数加载 - 缺失: {len(missing)}, 多余: {len(unexpected)}")
    
    def forward(self, x, control_map, text_hidden, text_pooled, position_info=None):
        """前向传播，生成控制信号
        
        Args:
            x: 输入的LQ图像 [B, C, H, W]
            control_map: 控制映射 [B, 3, H, W]
            text_hidden: 文本隐藏状态 [B, L, D]
            text_pooled: 池化文本特征 [B, D]
            position_info: 位置信息 [B, num_blocks]
            
        Returns:
            控制信号列表和注意力图
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
        mask_map = control_map[:, 2:3]
        
        # 处理控制信号
        canny_feat = self.canny_branch(canny_map)
        depth_feat = self.depth_branch(depth_map)
        
        # 编码文本特征
        text_hidden = self.text_encoder(text_hidden)
        text_pooled = self.pooled_encoder(text_pooled)
        text_feat = self.text_projector(text_pooled)
        
        # 处理控制特征
        control_feat = self.control_encoder(control_map)
        control_feat = control_feat + canny_feat * 0.5 + depth_feat * 0.5
        control_feat = self.control_self_attention(control_feat)
        
        # 前向传播SR网络
        locked_fea = self.conv_first(x)
        
        locked_features = []
        locked_trunk = locked_fea
        
        # 分组处理RRDB块
        block_groups = []
        if hasattr(self, 'body'):
            total_blocks = len(self.body)
            group_size = 4
            
            for i in range(0, total_blocks, group_size):
                end = min(i + group_size, total_blocks)
                block_groups.append(self.body[i:end])
            
            # 处理每个块组
            for i, group in enumerate(block_groups):
                for block in group:
                    locked_trunk = block(locked_trunk)
                
                # 存储特定层特征
                if i == 1:  # 第8层
                    locked_feature_8 = locked_trunk.clone()
                elif i == 3:  # 第16层
                    locked_feature_16 = locked_trunk.clone()
                
                locked_features.append(locked_trunk.clone())
        
        # 主体输出特征
        locked_body_out = self.conv_body(locked_trunk)
        locked_features.append(locked_body_out)
        
        # 上采样特征
        fea_up1 = F.interpolate(locked_body_out, scale_factor=2, mode='nearest')
        locked_fea_up1 = self.conv_up1(fea_up1)
        
        fea_up2 = F.interpolate(locked_fea_up1, scale_factor=2, mode='nearest')
        locked_fea_up2 = self.conv_up2(fea_up2)
        
        # 生成控制信号
        control_signals = []
        attention_maps = []
        
        # 确保control_feat与特征图尺寸一致 - 为不同阶段创建调整版本
        # 检查并调整控制特征尺寸，使其与locked_fea匹配
        if control_feat.shape[2:] != locked_fea.shape[2:]:
            adjusted_control_feat = F.interpolate(
                control_feat,
                size=(locked_fea.shape[2], locked_fea.shape[3]),
                mode="bilinear",
                align_corners=False
            )
        else:
            adjusted_control_feat = control_feat
        
        # a. 第一个卷积层的控制信号
        first_feat, first_attn = self.cross_attention[0](locked_fea, text_hidden)
        # 融合控制特征
        first_feat = first_feat + adjusted_control_feat
        first_control = self.zero_conv_first(first_feat, text_feat, text_hidden)
        first_attn_map = self.attention_generators[0](first_control)
        control_signals.append(first_control)
        attention_maps.append(first_attn_map)
        
        # b. RRDB块组的控制信号
        for i, feat in enumerate(locked_features):
            if i < len(self.zero_conv_body):
                # 确保控制特征与当前特征尺寸匹配
                if control_feat.shape[2:] != feat.shape[2:]:
                    adjusted_control_feat = F.interpolate(
                        control_feat,
                        size=(feat.shape[2], feat.shape[3]),
                        mode="bilinear",
                        align_corners=False
                    )
                else:
                    adjusted_control_feat = control_feat
                
                cross_feat, cross_attn = self.cross_attention[i+1](feat, text_hidden)
                
                # 特殊层的特征融合 - 第8和第16层
                if i == 1:  # 第8层
                    cross_feat = self.feature_fusion_8(cross_feat, adjusted_control_feat)
                elif i == 3:  # 第16层 
                    cross_feat = self.feature_fusion_16(cross_feat, adjusted_control_feat)
                
                control = self.zero_conv_body[i](cross_feat, text_feat, text_hidden)
                attn_map = self.attention_generators[i+1](control)
                
                control_signals.append(control)
                attention_maps.append(attn_map)
        
        # c. 主体输出的控制信号
        # 确保控制特征与主体输出特征尺寸匹配
        if control_feat.shape[2:] != locked_body_out.shape[2:]:
            adjusted_control_feat = F.interpolate(
                control_feat,
                size=(locked_body_out.shape[2], locked_body_out.shape[3]),
                mode="bilinear",
                align_corners=False
            )
        else:
            adjusted_control_feat = control_feat
            
        body_feat, body_attn = self.cross_attention[-3](locked_body_out, text_hidden)
        # 可选：融合调整后的控制特征
        body_feat = body_feat + adjusted_control_feat * 0.3
        body_control = self.zero_conv_body_out(body_feat, text_feat, text_hidden)
        body_attn_map = self.attention_generators[-3](body_control)
        control_signals.append(body_control)
        attention_maps.append(body_attn_map)
        
        # d. 上采样层的控制信号
        # 第一个上采样层
        # 确保控制特征与上采样特征尺寸匹配
        if control_feat.shape[2:] != locked_fea_up1.shape[2:]:
            adjusted_control_feat_up1 = F.interpolate(
                control_feat,
                size=(locked_fea_up1.shape[2], locked_fea_up1.shape[3]),
                mode="bilinear",
                align_corners=False
            )
        else:
            adjusted_control_feat_up1 = control_feat
            
        up1_feat, up1_attn = self.cross_attention[-2](locked_fea_up1, text_hidden)
        # 可选：融合调整后的控制特征
        up1_feat = up1_feat + adjusted_control_feat_up1 * 0.3
        up1_control = self.zero_conv_up1(up1_feat, text_feat, text_hidden)
        up1_attn_map = self.attention_generators[-2](up1_control)
        
        control_signals.append(up1_control)
        attention_maps.append(up1_attn_map)
        
        # 第二个上采样层
        if control_feat.shape[2:] != locked_fea_up2.shape[2:]:
            adjusted_control_feat_up2 = F.interpolate(
                control_feat,
                size=(locked_fea_up2.shape[2], locked_fea_up2.shape[3]),
                mode="bilinear",
                align_corners=False
            )
        else:
            adjusted_control_feat_up2 = control_feat
            
        up2_feat, up2_attn = self.cross_attention[-1](locked_fea_up2, text_hidden)
        # 可选：融合调整后的控制特征
        up2_feat = up2_feat + adjusted_control_feat_up2 * 0.3
        up2_control = self.zero_conv_up2(up2_feat, text_feat, text_hidden)
        up2_attn_map = self.attention_generators[-1](up2_control)
        
        control_signals.append(up2_control)
        attention_maps.append(up2_attn_map)
        
        # e. 确保生成足够的控制信号
        # 计算需要的信号数量
        expected_groups = len(self.body) // 4 + (1 if len(self.body) % 4 > 0 else 0)
        expected_signals = expected_groups + 4  # 第一个卷积 + 块组 + 主体输出 + 两个上采样层
        
        # 检查当前生成的信号数量
        current_signals = len(control_signals)
        
        # 如果信号数量不足，创建更多信号
        if current_signals < expected_signals:
            # 创建额外的控制信号
            for i in range(current_signals, expected_signals):
                # 复制最后一个控制信号并添加一些小的随机扰动，确保梯度流
                extra_control = control_signals[-1].clone()
                extra_control = extra_control * (0.95 + 0.1 * torch.rand_like(extra_control))
                control_signals.append(extra_control)
                
                # 同样为额外控制信号创建注意力图
                extra_attn = attention_maps[-1].clone()
                extra_attn = extra_attn * (0.95 + 0.1 * torch.rand_like(extra_attn))
                attention_maps.append(extra_attn)
        
        return control_signals, attention_maps


class FeatureFusionModule(nn.Module):
    """特征融合模块，用于融合SR特征和控制特征"""
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        
        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        
        # 自注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, sr_feat, control_feat):
        """融合SR特征和控制特征
        
        Args:
            sr_feat (Tensor): SR特征 [B, C, H, W]
            control_feat (Tensor): 控制特征 [B, C, H, W]
        
        Returns:
            Tensor: 融合后的特征 [B, C, H, W]
        """
        # 确保尺寸一致
        if sr_feat.shape[2:] != control_feat.shape[2:]:
            control_feat = F.interpolate(
                control_feat, size=sr_feat.shape[2:], 
                mode="bilinear", align_corners=False
            )
        
        # 拼接特征
        concat_feat = torch.cat([sr_feat, control_feat], dim=1)
        
        # 融合
        fused_feat = self.fusion_conv(concat_feat)
        
        # 应用注意力
        attention = self.attention(fused_feat)
        output = fused_feat * attention
        
        # 残差连接
        output = output + sr_feat
        
        return output


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
        
        # 应用注意力权重 - 修复这里的矩阵乘法
        out = torch.matmul(attn, v)  # [B, h, HW, d]
        out = out.permute(0, 1, 3, 2).reshape(batch_size, c, h, w)
        
        # 输出投影
        out = self.out_proj(out)
        
        # 残差连接
        return out + x


class ZeroConvBlock(nn.Module):
    """零初始化卷积块 - ControlNet核心组件，增强版"""
    def __init__(self, in_channels, out_channels, text_dim=512):
        super().__init__()
        # 增强卷积模块，使用残差结构
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        
        # 残差层
        self.res_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels)
        )
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 使用非零但很小的初始化，而不是完全的零初始化
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        nn.init.constant_(self.conv2.bias, 0)
        
        # 文本调制 - 增强版
        self.text_modulation = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(out_channels, out_channels * 2),
            nn.Tanh()
        )
        
        # 文本自适应尺度 - 增强版
        self.text_adaptive_scale = nn.Sequential(
            nn.Linear(text_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.Sigmoid()
        )
        
        # 自注意力模块 - 增强版
        self.self_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//8, kernel_size=1),
            nn.GroupNorm(4, out_channels//8),
            nn.SiLU(),
            nn.Conv2d(out_channels//8, 1, kernel_size=1)
        )
        
        # 通道注意力 - 新增
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def ensure_zero_init(self):
        """使用小值初始化而非纯零值"""
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        nn.init.constant_(self.conv2.bias, 0)
    
    def forward(self, x, text_embedding, text_hidden=None):
        """
        前向传播
        Args:
            x: [B, C, H, W] - 输入特征
            text_embedding: [B, C] - 文本特征
            text_hidden: [B, L, D] - 文本隐藏状态
        """
        # 第一个卷积
        h = self.conv1(x)
        h = self.norm1(h)
        
        # 文本调制 - 计算缩放和偏移
        text_params = self.text_modulation(text_embedding) # [B, C*2]
        scale, shift = torch.chunk(text_params, 2, dim=1)  # 各 [B, C]
        
        # 改进通道均衡 - RGB通道平衡
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
            scale_norm = scale_view * (rgb_mean / (scale_mean + 1e-10))
            
            # 对归一化后的scale再次限制最大值，防止通道差异过大
            max_scale = 1.5
            scale_norm = torch.clamp(scale_norm, 0.5, max_scale)
            
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
            
            # 对shift也应用类似的限制
            shift_norm = torch.clamp(shift_norm, -0.5, 0.5)
            
            # 重塑回原始维度
            shift = shift_norm.view(shift.size(0), -1)
        
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 应用缩放和偏移
        h = h * (scale + 1.0) + shift
        
        # 残差连接
        res = self.res_conv(h)
        h = h + res
        
        # 激活和第二个卷积
        h = self.act1(h)
        h = self.conv2(h)  # 零初始化权重，初始时不影响原始特征
        
        # 添加自注意力和自适应缩放
        if text_hidden is not None:
            text_mean = text_hidden.mean(dim=1)
            adaptive_scale = self.text_adaptive_scale(text_mean).unsqueeze(-1).unsqueeze(-1)
            
            # 计算自注意力
            attn_map = self.self_attention(h)
            attn_weight = torch.sigmoid(attn_map)
            
            # 应用自注意力
            h = h * (1.0 + attn_weight * adaptive_scale)
            
            # 应用通道注意力 - 新增
            channel_weights = self.channel_attention(h)
            h = h * channel_weights
        
        return h 


class CrossAttention(nn.Module):
    """交叉注意力模块 - 促进文本与图像特征的精确对齐，增强版"""
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