import argparse
import cv2
import glob
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import json

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from basicsr.archs.rrdbnet_arch import RRDBNet
from tgsr.archs.tgsr_arch import TextGuidanceNet
import torch.nn as nn


def preprocess_text(text, tokenizer, text_encoder, device):
    """预处理文本，获取文本特征"""
    tokens = tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    with torch.no_grad():
        text_outputs = text_encoder(**tokens)
        text_embeddings = text_outputs.last_hidden_state
        text_pooled = text_outputs.pooler_output
    
    return text_embeddings, text_pooled


class TGSRPredictor:
    """TGSR预测器类"""
    def __init__(self, sr_model_path, text_guidance_path, text_encoder_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载SR网络
        self.net_g = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32).to(self.device)
        sr_checkpoint = torch.load(sr_model_path, map_location=self.device)
        
        # 修复模型命名问题 - 在加载前重命名参数键
        if 'params' in sr_checkpoint:
            checkpoint_params = sr_checkpoint['params']
            # 创建新的state_dict，处理RRDB_trunk -> body的映射
            new_state_dict = {}
            for k, v in checkpoint_params.items():
                # 将RRDB_trunk替换为body
                new_k = k.replace('RRDB_trunk', 'body')
                # 将trunk_conv替换为conv_body
                new_k = new_k.replace('trunk_conv', 'conv_body')
                # 将upconv替换为conv_up
                new_k = new_k.replace('upconv1', 'conv_up1')
                new_k = new_k.replace('upconv2', 'conv_up2')
                # 将HRconv替换为conv_hr
                new_k = new_k.replace('HRconv', 'conv_hr')
                
                new_state_dict[new_k] = v
                
            # 加载修改后的权重
            self.net_g.load_state_dict(new_state_dict, strict=True)
        else:
            print("警告: 检查点中找不到'params'键")
            
        self.net_g.eval()
        
        # 加载文本引导网络
        self.net_t = TextGuidanceNet(num_feat=64, text_dim=512, num_blocks=3, num_heads=8).to(self.device)
        text_checkpoint = torch.load(text_guidance_path, map_location=self.device)
        self.net_t.load_state_dict(text_checkpoint['params'], strict=True)
        self.net_t.eval()
        
        # 加载CLIP
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_path)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to(self.device)
        self.text_encoder.eval()
        
        # 文本特征设置
        self.use_text_features = True
        self.text_dim = 512
        self.freeze_text_encoder = True
        
        # 属性映射，解决前向传播时的命名问题（这是必要的，不能修改）
        self.net_g.RRDB_trunk = self.net_g.body
        self.net_g.trunk_conv = self.net_g.conv_body
        self.net_g.upconv1 = self.net_g.conv_up1
        self.net_g.upconv2 = self.net_g.conv_up2
        self.net_g.HRconv = self.net_g.conv_hr
        
        # 创建激活函数
        if not hasattr(self.net_g, 'lrelu'):
            self.net_g.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        print(f"所有模型已加载到{self.device}设备")
    
    def apply_text_guidance(self, features, text_hidden=None, text_pooled=None, block_idx=None):
        """应用文本引导到特征图，与tgsr_model.py保持一致"""
        if not self.use_text_features:
            return features, None
        
        # 应用文本引导
        enhanced_features, attention_logits = self.net_t(features, text_hidden, text_pooled)
        
        return enhanced_features, attention_logits
    
    def forward_sr_network(self, x, apply_guidance=True):
        """SR网络的前向传播，与tgsr_model.py保持一致"""
        # 编码文本（如果需要）
        if self.use_text_features and apply_guidance:
            with torch.no_grad():
                text_hidden, text_pooled = self.encode_text(self.text_prompts)
        else:
            text_hidden, text_pooled = None, None
        
        # 浅层特征提取
        fea = self.net_g.conv_first(x)
        
        # RRDB主干处理
        trunk = fea
        attention_maps = []
        
        # 确定在哪些位置应用文本引导
        if self.use_text_features and apply_guidance and text_hidden is not None and text_pooled is not None:
            num_blocks = len(self.net_g.RRDB_trunk)
            # 选择合理的引导位置: 浅层、中层和深层
            guidance_positions = [num_blocks // 6, num_blocks // 2, num_blocks * 5 // 6]
            
            for i, block in enumerate(self.net_g.RRDB_trunk):
                trunk = block(trunk)
                
                # 在关键位置应用文本引导
                if i in guidance_positions:
                    # 文本引导
                    trunk, attn_maps = self.apply_text_guidance(trunk, text_hidden, text_pooled, block_idx=i)
                    
                    # 保存注意力图
                    if attn_maps is not None:
                        if isinstance(attn_maps, list):
                            attention_maps.extend(attn_maps)
                        else:
                            attention_maps.append(attn_maps)
        else:
            # 不使用文本引导
            for block in self.net_g.RRDB_trunk:
                trunk = block(trunk)
        
        # 残差连接
        trunk = self.net_g.trunk_conv(trunk)
        fea = fea + trunk
        
        # 上采样
        try:
            # 使用标准上采样路径
            fea = self.net_g.lrelu(self.net_g.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.net_g.lrelu(self.net_g.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
            
            # 最终输出
            out = self.net_g.conv_last(self.net_g.lrelu(self.net_g.HRconv(fea)))
        except (AttributeError, RuntimeError) as e:
            # 出错时，使用模型自带的前向传播
            print(f"上采样出错: {e}，使用模型自带的前向传播")
            out = self.net_g(x)
        
        # 保存注意力图
        if len(attention_maps) > 0:
            self.attention_maps = attention_maps
        else:
            self.attention_maps = None
        
        return out
    
    def encode_text(self, text_prompts):
        """编码文本提示，与tgsr_model.py保持一致"""
        if not self.use_text_features:
            batch_size = 1
            text_hidden = torch.zeros(batch_size, 77, self.text_dim).to(self.device)
            text_pooled = torch.zeros(batch_size, self.text_dim).to(self.device)
            return text_hidden, text_pooled
        
        # 使用CLIP编码文本
        with torch.no_grad():
            text_inputs = self.tokenizer(
                text_prompts, 
                padding="max_length", 
                max_length=77, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            text_outputs = self.text_encoder(**text_inputs)
            text_hidden = text_outputs.last_hidden_state
            text_pooled = text_outputs.pooler_output
        
        return text_hidden, text_pooled
    
    def predict(self, img_path, text=None, output_dir=None, save_attention=False):
        """预测单张图像的超分辨率结果"""
        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        
        # 设置文本提示
        self.text_prompts = [text if text else ""]
        
        # 推理
        with torch.no_grad():
            # 使用文本引导
            self.output = self.forward_sr_network(img, apply_guidance=True)
            guided_output = self.output.clone()
            
            # 不使用文本引导进行对比
            self.output = self.forward_sr_network(img, apply_guidance=False)
            unguided_output = self.output.clone()
            
            # 恢复为引导结果
            self.output = guided_output
        
        # 处理输出结果
        guided_output = guided_output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        guided_output = np.transpose(guided_output[[2, 1, 0], :, :], (1, 2, 0))
        guided_output = (guided_output * 255.0).round().astype(np.uint8)
        
        unguided_output = unguided_output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        unguided_output = np.transpose(unguided_output[[2, 1, 0], :, :], (1, 2, 0))
        unguided_output = (unguided_output * 255.0).round().astype(np.uint8)
        
        # 保存结果
        results = {
            'guided': guided_output,
            'unguided': unguided_output,
            'attention_maps': self.attention_maps if save_attention else None
        }
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            imgname = os.path.splitext(os.path.basename(img_path))[0]
            
            # 保存引导和非引导结果
            cv2.imwrite(os.path.join(output_dir, f'{imgname}_guided.png'), guided_output)
            cv2.imwrite(os.path.join(output_dir, f'{imgname}_unguided.png'), unguided_output)
            
            # 保存文本
            if text:
                with open(os.path.join(output_dir, f'{imgname}_text.txt'), 'w') as f:
                    f.write(text)
            
            # 创建并排比较图
            h, w = guided_output.shape[:2]
            comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
            
            # 调整输入图像大小以匹配
            input_img = cv2.imread(img_path)
            input_img = cv2.resize(input_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            comparison[:, :w] = input_img  # 左侧：输入
            comparison[:, w:2*w] = unguided_output  # 中间：非引导结果
            comparison[:, 2*w:] = guided_output  # 右侧：引导结果
            
            # 添加分割线
            comparison[:, w-1:w+1] = [0, 0, 255]  # 红色分割线
            comparison[:, 2*w-1:2*w+1] = [0, 0, 255]  # 红色分割线
            
            cv2.imwrite(os.path.join(output_dir, f'{imgname}_comparison.png'), comparison)
            
            # 如果需要，保存注意力图
            if save_attention and self.attention_maps:
                self.save_attention_maps(img_path, guided_output, text, output_dir)
        
        return results
    
    def save_attention_maps(self, img_path, sr_img, text, output_dir):
        """保存注意力图 - 处理方式与tgsr_model.py中的save_gradcam_attention类似"""
        if not hasattr(self, 'attention_maps') or not self.attention_maps:
            return
            
        imgname = os.path.splitext(os.path.basename(img_path))[0]
        
        # 转换所有注意力图为numpy
        attention_numpy_maps = []
        for attn_logits in self.attention_maps:
            # 提取第一个样本的注意力图并应用sigmoid
            attention_2d = torch.sigmoid(attn_logits[0, 0]).cpu().detach()
            
            # 归一化
            min_val = attention_2d.min()
            max_val = attention_2d.max()
            
            # 如果差距太小，强制拉开差距
            if max_val - min_val < 0.3:
                # 使用更强的对比度增强
                mean_val = attention_2d.mean()
                std_val = attention_2d.std()
                # 提高标准差以增加对比度
                normalized_attn = torch.clamp((attention_2d - mean_val) / (std_val * 2 + 1e-8) + 0.5, 0, 1)
            else:
                # 标准归一化
                normalized_attn = (attention_2d - min_val) / (max_val - min_val + 1e-8)
                
            # 调整大小以匹配输出
            h, w = sr_img.shape[:2]
            attn_numpy = normalized_attn.numpy()
            attn_numpy = cv2.resize(attn_numpy, (w, h), interpolation=cv2.INTER_LINEAR)
            attention_numpy_maps.append(attn_numpy)
        
        # 合并注意力图
        combined_map = np.mean(np.stack(attention_numpy_maps), axis=0)
        
        # 应用自适应阈值增强
        mean_val = np.mean(combined_map)
        std_val = np.std(combined_map)
        
        # 根据均值和标准差设置阈值
        enhanced_map = np.zeros_like(combined_map)
        high_threshold = mean_val + 0.5 * std_val
        low_threshold = mean_val - 0.5 * std_val
        
        # 高阈值区域
        high_mask = combined_map > high_threshold
        # 低阈值区域
        low_mask = combined_map < low_threshold
        # 中间区域
        mid_mask = ~(high_mask | low_mask)
        
        # 高注意力区域增强
        enhanced_map[high_mask] = 0.75 + 0.25 * (combined_map[high_mask] - high_threshold) / (1 - high_threshold + 1e-8)
        # 低注意力区域降低
        enhanced_map[low_mask] = 0.25 * combined_map[low_mask] / (low_threshold + 1e-8)
        # 中间区域线性映射
        if np.any(mid_mask):
            enhanced_map[mid_mask] = 0.25 + 0.5 * (combined_map[mid_mask] - low_threshold) / (high_threshold - low_threshold + 1e-8)
        
        # 确保值范围在[0,1]内
        enhanced_map = np.clip(enhanced_map, 0, 1)
        
        # 生成热力图
        heatmap = cv2.applyColorMap((enhanced_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 叠加到超分辨率图像上
        sr_img_rgb = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB) if sr_img.shape[2] == 3 else sr_img.copy()
        overlay = cv2.addWeighted(sr_img_rgb, 0.6, heatmap, 0.4, 0)
        
        # 添加图例
        h, w = overlay.shape[:2]
        legend_h = 30
        legend = np.zeros((legend_h, w, 3), dtype=np.uint8)
        
        # 创建从蓝到红的渐变色条
        for i in range(w):
            ratio = i / (w - 1)
            if ratio < 0.25:  # 蓝到青
                b, g, r = 255, int(255 * ratio * 4), 0
            elif ratio < 0.5:  # 青到绿
                b, g, r = int(255 * (0.5 - ratio) * 4), 255, 0
            elif ratio < 0.75:  # 绿到黄
                b, g, r = 0, 255, int(255 * (ratio - 0.5) * 4)
            else:  # 黄到红
                b, g, r = 0, int(255 * (1.0 - ratio) * 4), 255
            legend[:, i] = [r, g, b]
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(legend, 'Low', (5, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(legend, 'High', (w - 45, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 合并图像和图例
        result = np.vstack([overlay, legend])
        
        # 添加文本标签（如果有）
        if text:
            text_h = 50  # 增加高度以容纳更多文本
            text_area = np.ones((text_h, w, 3), dtype=np.uint8) * 240  # 淡灰色背景
            
            # 减小字体大小
            font_scale_text = 0.4
            line_height = 16
            
            # 将长文本分成多行
            max_chars = w // 8
            if len(text) > max_chars:
                words = text.split()
                lines = []
                current_line = words[0]
                for word in words[1:]:
                    if len(current_line) + len(word) + 1 <= max_chars:
                        current_line += " " + word
                    else:
                        lines.append(current_line)
                        current_line = word
                lines.append(current_line)
                
                # 如果有多行，调整文本区域的高度
                if len(lines) > 1:
                    text_h = min(len(lines) * line_height + 10, 80)  # 限制最高80像素
                    text_area = np.ones((text_h, w, 3), dtype=np.uint8) * 240
                
                # 添加每行文本
                for i, line in enumerate(lines):
                    y_pos = line_height + i * line_height
                    if y_pos < text_h - 5:
                        cv2.putText(text_area, line, (5, y_pos), font, font_scale_text, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                # 单行文本
                cv2.putText(text_area, text, (5, line_height), font, font_scale_text, (0, 0, 0), 1, cv2.LINE_AA)
            
            # 添加文本区域到结果图像
            result = np.vstack([text_area, result])
        
        # 保存结果
        cv2.imwrite(os.path.join(output_dir, f'{imgname}_attention.png'), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sr_model_path', type=str, required=True, help='SR网络模型路径')
    parser.add_argument('--text_guidance_path', type=str, required=True, help='文本引导网络模型路径')
    parser.add_argument('--text_encoder_path', type=str, required=True, help='CLIP文本编码器路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像文件夹或单个图像')
    parser.add_argument('--output', type=str, default='results/TGSR', help='输出文件夹')
    parser.add_argument('--text', type=str, default=None, help='文本描述文件(JSON格式)，每个图像对应一个描述')
    parser.add_argument('--save_attention', action='store_true', help='是否保存注意力图')
    args = parser.parse_args()
    
    # 初始化预测器
    predictor = TGSRPredictor(
        sr_model_path=args.sr_model_path,
        text_guidance_path=args.text_guidance_path,
        text_encoder_path=args.text_encoder_path
    )
    
    # 加载文本描述（如果有）
    text_dict = {}
    if args.text and os.path.exists(args.text):
        with open(args.text, 'r') as f:
            text_dict = json.load(f)
    
    # 处理输入路径
    if os.path.isdir(args.input):
        image_paths = sorted(glob.glob(os.path.join(args.input, '*.*')))
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    else:
        image_paths = [args.input]
    
    # 处理每张图像
    for idx, path in enumerate(image_paths):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print(f'处理第{idx+1}/{len(image_paths)}张图像: {imgname}')
        
        # 获取文本描述
        if imgname in text_dict:
            text = text_dict[imgname]
        else:
            text = None
            print(f'  警告: 图像{imgname}没有对应的文本描述')
        
        # 预测并保存结果
        try:
            predictor.predict(path, text, args.output, args.save_attention)
        except Exception as error:
            print(f'  错误: {error}')
        else:
            print(f'  已保存结果到: {args.output}/{imgname}_*.png')


if __name__ == '__main__':
    main() 