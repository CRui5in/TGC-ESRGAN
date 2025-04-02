import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('Agg')  # 避免需要GUI
from matplotlib import cm

def tensor_to_numpy(tensor, denormalize=True):
    """将tensor转换为numpy图像
    
    Args:
        tensor: 输入tensor，形状为[C, H, W]
        denormalize: 是否反归一化 (从[-1,1]到[0,1])
        
    Returns:
        numpy数组，形状为[H, W, C]，值范围[0,1]
    """
    if tensor.dim() == 4:  # [B, C, H, W]
        tensor = tensor[0]  # 取第一个样本
        
    img = tensor.detach().float().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
    
    if denormalize:
        img = (img + 1) / 2.0  # [-1, 1] -> [0, 1]
    img = np.clip(img, 0, 1)
    
    return img

def create_comparison_grid(lq, sr, gt=None, labels=None):
    """创建图像对比网格
    
    Args:
        lq: 低分辨率图像，形状为[B, C, H, W]
        sr: 超分辨率图像，形状为[B, C, H, W]
        gt: 高分辨率真值图像，形状为[B, C, H, W]，可选
        labels: 图像标签列表，可选
    
    Returns:
        对比网格图像，pytorch tensor
    """
    # 确保所有图像具有相同的大小
    if lq.shape[-2:] != sr.shape[-2:]:
        lq = F.interpolate(lq, size=sr.shape[-2:], mode='bilinear', align_corners=False)
    
    if gt is not None and gt.shape[-2:] != sr.shape[-2:]:
        gt = F.interpolate(gt, size=sr.shape[-2:], mode='bilinear', align_corners=False)
    
    # 转为可视化范围 [0, 1]
    lq_vis = (lq.clamp(-1, 1) + 1) / 2
    sr_vis = (sr.clamp(-1, 1) + 1) / 2
    
    # 添加标签到图像
    if labels is None:
        labels = ["LQ", "SR"]
        if gt is not None:
            labels.append("GT")
    
    # 创建带标签的图像
    lq_labeled = add_label_to_image(lq_vis, labels[0])
    sr_labeled = add_label_to_image(sr_vis, labels[1])
    
    # 组合图像
    images = [lq_labeled, sr_labeled]
    if gt is not None:
        gt_vis = (gt.clamp(-1, 1) + 1) / 2
        gt_labeled = add_label_to_image(gt_vis, labels[2])
        images.append(gt_labeled)
    
    # 创建网格
    grid = make_grid(torch.cat(images, dim=0), nrow=len(images))
    
    return grid

def add_label_to_image(img_tensor, label):
    """在图像左下角添加文本标签
    
    Args:
        img_tensor: 图像tensor，形状为[B, C, H, W]
        label: 要添加的标签
        
    Returns:
        带标签的图像tensor
    """
    batch_size = img_tensor.shape[0]
    labeled_tensors = []
    
    for i in range(batch_size):
        img = tensor_to_numpy(img_tensor[i], denormalize=False)
        img = (img * 255).astype(np.uint8)
        
        # 转换为PIL图像
        pil_img = Image.fromarray(img)
        
        # 创建画布
        draw = ImageDraw.Draw(pil_img)
        
        # 在左下角添加标签
        text_size = max(int(min(img.shape[0], img.shape[1]) * 0.05), 10)
        try:
            # 尝试加载字体
            font = ImageFont.truetype("DejaVuSans.ttf", text_size)
        except:
            # 找不到字体时使用默认字体
            font = ImageFont.load_default()
        
        # 绘制文本背景框
        text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:4]
        margin = text_size // 3
        draw.rectangle(
            [(margin, pil_img.height - text_height - margin * 2), 
             (margin + text_width + margin, pil_img.height - margin)],
            fill=(0, 0, 0, 180)
        )
        
        # 绘制文本
        draw.text(
            (margin * 2, pil_img.height - text_height - margin),
            label,
            font=font,
            fill=(255, 255, 255)
        )
        
        # 转回tensor
        img_np = np.array(pil_img) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        labeled_tensors.append(img_tensor)
    
    return torch.stack(labeled_tensors)

def generate_heatmap(attention_map):
    """从注意力图生成热力图
    
    Args:
        attention_map: 注意力图，形状为[B, 1, H, W]
        
    Returns:
        彩色热力图，形状为[B, 3, H, W]
    """
    batch_size = attention_map.shape[0]
    heatmaps = []
    
    for i in range(batch_size):
        # 转为numpy并标准化
        attn = attention_map[i, 0].detach().cpu().numpy()
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        
        # 应用colormap
        heatmap = cv2.applyColorMap((attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # 确保热力图是RGB格式
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # 转回tensor
        heatmap_tensor = torch.from_numpy(heatmap).permute(2, 0, 1).float()
        heatmaps.append(heatmap_tensor)
    
    return torch.stack(heatmaps)

def overlay_heatmap(image, heatmap, alpha=0.5):
    """将热力图覆盖到图像上
    
    Args:
        image: 原始图像，形状为[B, C, H, W]
        heatmap: 热力图，形状为[B, 3, H, W]
        alpha: 热力图不透明度
        
    Returns:
        覆盖了热力图的图像，形状为[B, 3, H, W]
    """
    batch_size = image.shape[0]
    overlays = []
    
    # 确保输入格式正确
    if image.shape[-2:] != heatmap.shape[-2:]:
        heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False)
    
    # 标准化图像到 [0, 1]
    image_vis = (image.clamp(-1, 1) + 1) / 2
    
    for i in range(batch_size):
        img = image_vis[i].detach().cpu()
        heat = heatmap[i].detach().cpu()
        
        # 创建覆盖图
        overlay = img * (1 - alpha) + heat * alpha
        overlays.append(overlay)
    
    return torch.stack(overlays)

def visualize_attention(model, images, text_prompt, device='cuda'):
    """可视化模型注意力热力图   
    
    Args:
        model: 模型实例
        images: 输入图像，形状为[B, C, H, W]
        text_prompt: 文本提示
        device: 设备
        
    Returns:
        热力图覆盖的图像，形状为[B, 3, H, W]
    """
    model.eval()
    batch_size = images.shape[0]
    
    # 前向传播
    with torch.no_grad():
        # 提取文本特征
        if hasattr(model, 'extract_text_features'):
            text_hidden, text_pooled = model.extract_text_features(text_prompt)
        else:
            # 假设是TGSRModel
            model.feed_data({'lq': images, 'text_prompt': text_prompt})
            text_hidden, text_pooled = model.text_hidden, model.text_pooled
        
        # 获取GradCAM
        cam = model.get_grad_cam()
        
        if cam is None:
            # 如果GradCAM不可用，生成一个默认热力图
            cam = torch.ones((batch_size, 1, images.shape[2], images.shape[3]), device=device)
    
    # 生成热力图
    heatmap = generate_heatmap(cam)
    
    # 覆盖到原图
    overlay = overlay_heatmap(images, heatmap)
    
    return overlay

def log_image_with_attention(writer, tag, image_tensors, attention_map, global_step, max_images=4):
    """将图像及其注意力图记录到TensorBoard
    
    Args:
        writer: TensorBoard SummaryWriter实例
        tag: 图像标签
        image_tensors: 图像tensor，形状为[B, C, H, W]
        attention_map: 注意力图，形状为[B, 1, H, W]
        global_step: 全局步数
        max_images: 最大显示的图像数量
    """
    # 限制图像数量
    batch_size = min(image_tensors.shape[0], max_images)
    images = image_tensors[:batch_size]
    
    # 生成热力图
    heatmap = generate_heatmap(attention_map[:batch_size])
    
    # 创建覆盖图
    overlay = overlay_heatmap(images, heatmap)
    
    # 将原始图像和覆盖图拼接
    combined = []
    for i in range(batch_size):
        img = (images[i].clamp(-1, 1) + 1) / 2
        over = overlay[i]
        combined.extend([img, over])
    
    # 创建网格
    nrow = 2  # 原图和热力图并排
    grid = make_grid(torch.stack(combined), nrow=nrow)
    
    # 记录到TensorBoard
    writer.add_image(tag, grid, global_step)

def log_metrics_to_tensorboard(writer, metrics, global_step, prefix='Validation/'):
    """记录评估指标到TensorBoard
    
    Args:
        writer: TensorBoard SummaryWriter实例
        metrics: 指标字典
        global_step: 全局步数
        prefix: 指标前缀，默认为'Validation/'
    """
    if writer is None or metrics is None:
        return
    
    # 确保前缀以/结尾
    if not prefix.endswith('/'):
        prefix = prefix + '/'
    
    # 确保使用标准规范: Train/Validation/Test
    if prefix.lower().startswith('val/'):
        prefix = 'Validation/Metrics/'
    elif prefix.lower().startswith('test/'):
        prefix = 'Test/Metrics/'
    elif prefix.lower().startswith('train/'):
        prefix = 'Train/Metrics/'
    else:
        # 如果没有标准前缀，添加Metrics子目录
        if not prefix.endswith('Metrics/'):
            prefix = prefix + 'Metrics/'
    
    # 记录表格数据
    if len(metrics) > 0:
        # 创建表格数据
        table_data = []
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                table_data.append([metric_name, formatted_value])
        
        # 添加表格，使用标准前缀
        if table_data:
            # 确定表格前缀
            table_prefix = prefix.split('/')[0] if '/' in prefix else prefix.rstrip('/')
            writer.add_text(
                f'{table_prefix}/Metrics_Table', 
                '| Metric | Value |\n|---|---|\n' + '\n'.join([f'| {row[0]} | {row[1]} |' for row in table_data]), 
                global_step
            )

def improved_tensor2img(tensor, rgb2bgr=False, out_type=np.uint8, min_max=(0, 1)):
    """将tensor转换为图像numpy数组
    
    Args:
        tensor (Tensor | numpy.ndarray): 输入tensor或numpy数组
        rgb2bgr (bool): 是否将RGB转换为BGR，默认为False保持RGB格式
        out_type (numpy类型): 输出的numpy类型
        min_max (tuple[float]): 输入tensor的最小和最大值范围
            
    Returns:
        (numpy.ndarray | list[numpy.ndarray]): 3D图像numpy数组 (H, W, C)
    """
    import numpy as np
    import cv2
    import torch
    
    # 如果输入已经是numpy数组，直接处理
    if isinstance(tensor, np.ndarray):
        img = tensor.copy()
        
        # 如果是灰度图，扩展为三通道
        if img.ndim == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        # 如果是3D但通道在第一维，转置为(H,W,C)
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
            
        # 确保值范围正确
        if min_max != (0, 255):
            img = np.clip(img, min_max[0], min_max[1])
            img = (img - min_max[0]) / (min_max[1] - min_max[0])
        
        # 确保图像数据类型是float32，避免OpenCV不支持的CV_64F
        if img.dtype == np.float64:
            img = img.astype(np.float32)
            
        # 应用RGB转BGR如果需要
        if rgb2bgr and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        # 转换为指定类型
        if out_type == np.uint8:
            img = (img * 255.0).round()
            
        return img.astype(out_type)
    
    # 处理torch.Tensor类型
    elif isinstance(tensor, torch.Tensor):
        tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
        
        n_dim = tensor.dim()
        if n_dim == 4:
            # NCHW -> CHW
            tensor = tensor[0]
            
        if n_dim == 2:
            # HW -> CHW (3, H, W)
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
            
        if n_dim == 3:
            img = tensor.numpy()
            if img.shape[0] == 3 or img.shape[0] == 1:
                # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
                if img.shape[2] == 1:
                    # 单通道 -> 三通道
                    img = np.repeat(img, 3, axis=2)
                
        elif tensor.dim() == 2:  # [H,W]
            img = tensor.numpy()
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)  # 扩展为3通道
        else:
            raise TypeError(f'不支持的张量维度: {tensor.dim()}')
        
        # 确保值范围在[0, 1]内
        img = np.clip(img, 0, 1)
        
        # 确保图像数据类型是float32，避免OpenCV不支持的CV_64F
        if img.dtype == np.float64:
            img = img.astype(np.float32)
            
        # 如果需要转BGR
        if rgb2bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 转换为输出类型
        if out_type == np.uint8:
            img = (img * 255.0).round()
        
        return img.astype(out_type)
        
    # 处理张量列表的情况
    elif isinstance(tensor, list):
        return [improved_tensor2img(t, rgb2bgr, out_type, min_max) for t in tensor]
        
    else:
        raise TypeError(f'输入类型必须是tensor、numpy数组或tensor列表, 但得到{type(tensor)}') 

def rgb_imwrite(img_path, img, params=None):
    """以RGB格式写入图像，而不是默认的BGR格式
    
    Args:
        img_path (str): 图像保存路径
        img (numpy.ndarray): 需要保存的图像，RGB格式，[0, 255]，uint8或float32
        params (list): OpenCV的imwrite参数
    """
    import os
    import cv2
    import numpy as np
    
    dir_name = os.path.abspath(os.path.dirname(img_path))
    os.makedirs(dir_name, exist_ok=True)
    
    # 检查是否是RGB格式
    if img.ndim == 3 and img.shape[2] == 3:
        # 转换为BGR格式，因为OpenCV保存图像时使用BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    
    # 确保uint8格式
    if img.dtype == np.float32:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    # 保存图像
    ok = cv2.imwrite(img_path, img, params if params is not None else [])
    if not ok:
        raise IOError(f"无法保存图像: {img_path}") 