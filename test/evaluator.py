"""
评估器模块
包含超分辨率和目标检测的评估功能
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib as mpl
import platform
import logging

from torchvision.ops import box_iou, box_convert
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

from tgsr.utils.visualization_utils import improved_tensor2img as tensor2img


class GroundingDINOWrapper:
    """GroundingDINO目标检测模型包装器 (Transformers版本)"""
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        初始化GroundingDINO模型 (使用Transformers)
        
        Args:
            model_name: 模型名称或本地路径
            device: 设备
        """
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger()
        
        # 动态导入Transformers
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            # 加载模型和处理器
            self.logger.info(f"正在加载GroundingDINO模型: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
            
            self.logger.info(f"成功加载GroundingDINO模型: {model_name}")
        except Exception as e:
            self.logger.warning(f"警告: 加载GroundingDINO模型失败: {e}")
            self.processor = None
            self.model = None
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.model is not None and self.processor is not None
    
    def detect(self, image: np.ndarray, prompt: str, 
               box_threshold: float = 0.4, text_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用GroundingDINO进行目标检测
        
        Args:
            image: 输入图像
            prompt: 文本提示词
            box_threshold: 框置信度阈值
            text_threshold: 文本匹配阈值
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 框坐标、得分和类别
        """
        if not self.is_available():
            return np.array([]), np.array([]), np.array([])
        
        # 将图像转换为PIL格式
        try:
            if isinstance(image, torch.Tensor):
                # 如果是tensor,转换为numpy
                image_np = tensor2img(image, rgb2bgr=False)
                image_pil = Image.fromarray(image_np)
            elif isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                image_pil = image
            else:
                raise ValueError(f"不支持的图像类型: {type(image)}")
        except Exception as e:
            self.logger.error(f"转换图像失败: {e}")
            return np.array([]), np.array([]), np.array([])
        
        # 确保文本提示词格式正确（小写并以句点结尾）
        text_prompts = []
        for p in prompt.split(". "):
            p = p.strip().lower()
            if p and not p.endswith("."):
                p += "."
            if p:
                text_prompts.append(p)
        
        if not text_prompts:
            # 如果没有有效的提示词，返回空结果
            return np.array([]), np.array([]), np.array([])
            
        text = " ".join(text_prompts)
        
        try:
            # 模型输入
            inputs = self.processor(images=image_pil, text=text, return_tensors="pt").to(self.device)
            
            # 进行预测
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 后处理
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image_pil.size[::-1]]  # (H, W)
            )[0]
            
            # 提取结果
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = [results["labels"][i].item() for i in range(len(scores))]
            
            # 获取类别名称
            class_names = []
            for i, label in enumerate(labels):
                text_id = inputs.input_ids[0, label].item()
                text_label = self.processor.tokenizer.decode([text_id]).strip()
                class_names.append(text_label.replace("<s>", "").replace("</s>", "").replace(".", "").strip())
            
            # 将边界框转换为相对坐标 [x1, y1, x2, y2] (0-1范围)
            h, w = image_pil.size[::-1]  # PIL.size 是 (W, H)
            normalized_boxes = boxes.copy()
            normalized_boxes[:, 0] /= w
            normalized_boxes[:, 1] /= h
            normalized_boxes[:, 2] /= w
            normalized_boxes[:, 3] /= h
            
            return normalized_boxes, scores, class_names
        except Exception as e:
            self.logger.error(f"目标检测失败: {e}")
            return np.array([]), np.array([]), np.array([])


class SuperResolutionEvaluator:
    """超分辨率评估器"""
    def __init__(self, config):
        """
        初始化超分辨率评估器
        
        Args:
            config: 测试配置
        """
        self.config = config
        self.metrics = {}
        self.results = defaultdict(list)
        self.visualize_dir = os.path.join(config.output_dir, "visualize")
        self.logger = logging.getLogger()
        os.makedirs(self.visualize_dir, exist_ok=True)
    
    def evaluate_image(self, sr_img: torch.Tensor, gt_img: torch.Tensor, 
                       img_id: int = None, save_path: str = None) -> Dict[str, float]:
        """
        评估单张图像的超分辨率质量
        
        Args:
            sr_img: 超分辨率图像
            gt_img: 原始高分辨率图像
            img_id: 图像ID
            save_path: 保存路径
            
        Returns:
            Dict[str, float]: 评估指标
        """
        try:
            # 转换为numpy
            sr_img_np = tensor2img(sr_img, rgb2bgr=False)
            gt_img_np = tensor2img(gt_img, rgb2bgr=False)
            
            # 确保图像尺寸一致
            if sr_img_np.shape != gt_img_np.shape:
                sr_img_np = cv2.resize(sr_img_np, (gt_img_np.shape[1], gt_img_np.shape[0]))
            
            # 计算PSNR
            psnr = calculate_psnr(sr_img_np, gt_img_np, crop_border=0)
            
            # 计算SSIM
            ssim = calculate_ssim(sr_img_np, gt_img_np, crop_border=0)
            
            # 保存结果
            result = {'psnr': psnr, 'ssim': ssim}
            
            # 保存到总结果
            for k, v in result.items():
                self.results[k].append(v)
            
            # 如果需要保存图像
            if save_path is not None and self.config.save_images:
                try:
                    # 创建目录
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    # 保存SR图像
                    save_path = os.path.normpath(save_path)  # 规范化路径
                    cv2.imwrite(save_path, cv2.cvtColor(sr_img_np, cv2.COLOR_RGB2BGR))
                    self.logger.debug(f"已保存超分辨率图像: {save_path}")
                except Exception as e:
                    self.logger.error(f"保存图像失败: {save_path}, 错误: {e}")
            
            return result
        except Exception as e:
            self.logger.error(f"评估图像失败: {e}")
            return {'psnr': 0.0, 'ssim': 0.0}
    
    def get_mean_metrics(self) -> Dict[str, float]:
        """获取平均指标"""
        mean_metrics = {}
        for k, v in self.results.items():
            if len(v) > 0:
                mean_metrics[k] = float(np.mean(v))
            else:
                mean_metrics[k] = 0.0
        return mean_metrics
    
    def visualize_attention(self, attention_map: np.ndarray, image: np.ndarray, 
                            save_path: str = None, 
                            colormap: str = 'jet', 
                            alpha: float = 0.5,
                            show_colorbar: bool = True) -> np.ndarray:
        """
        将注意力图可视化并叠加到图像上
        
        Args:
            attention_map: 注意力图
            image: 原始图像
            save_path: 保存路径
            colormap: 颜色映射
            alpha: 透明度
            show_colorbar: 是否显示颜色条
            
        Returns:
            np.ndarray: 可视化结果
        """
        try:
            # 确保注意力图为2D
            if len(attention_map.shape) > 2:
                attention_map = attention_map.mean(axis=0)
            
            # 确保图像为RGB
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            # 调整注意力图的大小以匹配原始图像
            if attention_map.shape[:2] != image.shape[:2]:
                attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
            
            # 归一化注意力图
            norm = Normalize(vmin=0, vmax=1)
            normalized_attention = norm(attention_map)
            
            # 应用颜色映射
            cmap = plt.get_cmap(colormap)
            colored_attention = cmap(normalized_attention)
            colored_attention = (colored_attention[:, :, :3] * 255).astype(np.uint8)
            
            # 叠加到原始图像
            overlay = cv2.addWeighted(
                image.astype(np.uint8), 
                1 - alpha, 
                colored_attention, 
                alpha, 
                0
            )
            
            # 如果需要显示颜色条
            if show_colorbar and save_path is not None:
                try:
                    # 创建目录
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    # 设置matplotlib后端，避免GUI相关问题
                    if platform.system() == "Windows":
                        plt.switch_backend('Agg')
                        
                    # 创建一个带有颜色条的图
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(attention_map, cmap=colormap)
                    plt.colorbar(im, ax=ax, label='注意力强度')
                    ax.axis('off')
                    
                    # 保存颜色条图像
                    color_bar_path = save_path.replace('.png', '_colorbar.png')
                    color_bar_path = os.path.normpath(color_bar_path)  # 规范化路径
                    plt.savefig(color_bar_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    self.logger.debug(f"已保存颜色条图像: {color_bar_path}")
                except Exception as e:
                    self.logger.error(f"保存颜色条图像失败: {e}")
            
            # 保存结果
            if save_path is not None:
                try:
                    # 创建目录
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    save_path = os.path.normpath(save_path)  # 规范化路径
                    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    self.logger.debug(f"已保存注意力热力图: {save_path}")
                except Exception as e:
                    self.logger.error(f"保存注意力热力图失败: {save_path}, 错误: {e}")
            
            return overlay
        except Exception as e:
            self.logger.error(f"可视化注意力图失败: {e}")
            # 返回原始图像
            return image


class DetectionEvaluator:
    """目标检测评估器"""
    def __init__(self, config):
        """
        初始化目标检测评估器
        
        Args:
            config: 测试配置
        """
        self.config = config
        self.results = []
        self.gt_boxes = defaultdict(list)
        self.pred_boxes = defaultdict(list)
        self.categories = {}
        self.logger = logging.getLogger()
        
        # 加载类别信息
        try:
            with open(config.annotation_path, 'r', encoding='utf-8') as f:
                coco = json.load(f)
                self.categories = {cat['id']: cat['name'] for cat in coco['categories']}
                self.logger.info(f"已加载 {len(self.categories)} 个类别")
        except Exception as e:
            self.logger.error(f"加载类别信息失败: {e}")
        
        # 初始化目标检测模型
        self.detector = GroundingDINOWrapper(
            config.detection_model_name,
            config.device
        )
    
    def is_detector_available(self) -> bool:
        """检查检测器是否可用"""
        return self.detector.is_available()
    
    def process_gt_boxes(self, objects: List[Dict], img_id: int, img_info: Dict) -> None:
        """
        处理真实边界框
        
        Args:
            objects: 标注对象列表
            img_id: 图像ID
            img_info: 图像信息
        """
        try:
            img_width = img_info.get('width', 0)
            img_height = img_info.get('height', 0)
            
            for obj in objects:
                category_id = obj['category_id']
                bbox = obj['bbox']  # COCO格式: [x, y, width, height]
                
                # 转换为绝对坐标 [x1, y1, x2, y2]
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = x1 + bbox[2], y1 + bbox[3]
                
                # 归一化坐标到[0, 1]
                if img_width > 0 and img_height > 0:
                    x1, x2 = x1 / img_width, x2 / img_width
                    y1, y2 = y1 / img_height, y2 / img_height
                
                # 保存到类别映射
                category_name = self.categories.get(category_id, "unknown")
                self.gt_boxes[category_name].append({
                    'image_id': img_id,
                    'bbox': [x1, y1, x2, y2],
                    'score': 1.0,  # GT框的置信度设为1
                    'category_id': category_id,
                    'category_name': category_name
                })
        except Exception as e:
            self.logger.error(f"处理真实边界框失败: {e}")
    
    def detect_objects(self, image: np.ndarray, prompt: str, img_id: int) -> List[Dict]:
        """
        检测图像中的对象
        
        Args:
            image: 图像
            prompt: 文本提示词
            img_id: 图像ID
            
        Returns:
            List[Dict]: 检测结果
        """
        if not self.is_detector_available():
            self.logger.warning("目标检测器不可用")
            return []
        
        try:
            # 执行目标检测
            boxes, scores, labels = self.detector.detect(
                image, 
                prompt, 
                self.config.box_threshold, 
                self.config.text_threshold
            )
            
            results = []
            
            # 处理检测结果
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                
                # 保存到类别映射
                self.pred_boxes[label].append({
                    'image_id': img_id,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score),
                    'category_name': label
                })
                
                # 构造结果
                results.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score),
                    'category': label
                })
            
            self.logger.debug(f"检测到 {len(results)} 个目标, 提示词: {prompt}")
            return results
        except Exception as e:
            self.logger.error(f"检测目标失败: {e}")
            return []
    
    def visualize_detection(self, image: np.ndarray, boxes: List[Dict], 
                          save_path: str = None) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 图像
            boxes: 检测框列表
            save_path: 保存路径
            
        Returns:
            np.ndarray: 可视化结果
        """
        try:
            # 复制图像以避免修改原始图像
            vis_img = image.copy()
            
            # 为不同类别生成不同颜色
            categories = set(box['category'] for box in boxes)
            colors = {}
            for i, cat in enumerate(categories):
                # 生成不同的颜色
                color = plt.cm.rainbow(i / max(1, len(categories) - 1))
                colors[cat] = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            
            # 绘制检测框
            for box in boxes:
                x1, y1, x2, y2 = [int(coord * image.shape[i % 2]) for i, coord in enumerate(box['bbox'])]
                category = box['category']
                score = box['score']
                color = colors.get(category, (0, 255, 0))
                
                # 绘制框
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{category}: {score:.2f}"
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 保存图像
            if save_path is not None:
                try:
                    # 创建目录
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    save_path = os.path.normpath(save_path)  # 规范化路径
                    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                    self.logger.debug(f"已保存检测结果: {save_path}")
                except Exception as e:
                    self.logger.error(f"保存检测结果失败: {save_path}, 错误: {e}")
            
            return vis_img
        except Exception as e:
            self.logger.error(f"可视化检测结果失败: {e}")
            # 返回原始图像
            return image
    
    def calculate_map(self, iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        计算目标检测的mAP
        
        Args:
            iou_threshold: IoU阈值
            
        Returns:
            Dict[str, float]: mAP结果
        """
        if not self.is_detector_available():
            return {"map": 0.0}
        
        try:
            # 获取所有类别
            categories = set(self.gt_boxes.keys()) | set(self.pred_boxes.keys())
            
            # 计算每个类别的AP
            aps = []
            for category in categories:
                # 获取该类别的GT和预测框
                gt_boxes_cat = self.gt_boxes.get(category, [])
                pred_boxes_cat = self.pred_boxes.get(category, [])
                
                # 如果没有GT框，跳过
                if len(gt_boxes_cat) == 0:
                    continue
                
                # 按图像ID分组
                gt_by_img = defaultdict(list)
                for box in gt_boxes_cat:
                    gt_by_img[box['image_id']].append(box)
                
                pred_by_img = defaultdict(list)
                for box in pred_boxes_cat:
                    pred_by_img[box['image_id']].append(box)
                
                # 计算各图像的TP/FP
                all_scores = []
                all_tp = []
                all_fp = []
                
                for img_id in set(gt_by_img.keys()) | set(pred_by_img.keys()):
                    gt_img = gt_by_img.get(img_id, [])
                    pred_img = pred_by_img.get(img_id, [])
                    
                    # 排序预测框
                    pred_img = sorted(pred_img, key=lambda x: x['score'], reverse=True)
                    
                    # GT框是否被匹配
                    gt_matched = [False] * len(gt_img)
                    
                    # 遍历预测框
                    for pred in pred_img:
                        # 提取预测框
                        pred_box = np.array(pred['bbox']).reshape(1, 4)
                        
                        # 最大IoU和对应的GT框索引
                        max_iou = -np.inf
                        max_idx = -1
                        
                        # 遍历GT框
                        for i, gt in enumerate(gt_img):
                            if gt_matched[i]:
                                continue
                            
                            # 计算IoU
                            gt_box = np.array(gt['bbox']).reshape(1, 4)
                            iou = self._calculate_iou(pred_box, gt_box)
                            
                            if iou > max_iou:
                                max_iou = iou
                                max_idx = i
                        
                        # 添加结果
                        all_scores.append(pred['score'])
                        
                        # 判断是否为TP
                        if max_idx >= 0 and max_iou >= iou_threshold:
                            all_tp.append(1)
                            all_fp.append(0)
                            gt_matched[max_idx] = True
                        else:
                            all_tp.append(0)
                            all_fp.append(1)
                
                # 如果没有预测框，AP为0
                if len(all_scores) == 0:
                    aps.append(0.0)
                    continue
                
                # 按得分排序
                indices = np.argsort(all_scores)[::-1]
                all_tp = np.array(all_tp)[indices]
                all_fp = np.array(all_fp)[indices]
                
                # 计算累积TP和FP
                cum_tp = np.cumsum(all_tp)
                cum_fp = np.cumsum(all_fp)
                
                # 计算精度和召回率
                precision = cum_tp / (cum_tp + cum_fp)
                recall = cum_tp / len(gt_boxes_cat)
                
                # 计算AP
                ap = self._calculate_ap(precision, recall)
                aps.append(ap)
            
            # 计算mAP
            map_score = np.mean(aps) if len(aps) > 0 else 0.0
            
            return {"map": map_score}
        except Exception as e:
            self.logger.error(f"计算mAP失败: {e}")
            return {"map": 0.0}
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        计算两个框的IoU
        
        Args:
            box1: 框1 [x1, y1, x2, y2]
            box2: 框2 [x1, y1, x2, y2]
            
        Returns:
            float: IoU值
        """
        # 计算交集区域
        x_left = max(box1[0, 0], box2[0, 0])
        y_top = max(box1[0, 1], box2[0, 1])
        x_right = min(box1[0, 2], box2[0, 2])
        y_bottom = min(box1[0, 3], box2[0, 3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # 计算交集面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集面积
        box1_area = (box1[0, 2] - box1[0, 0]) * (box1[0, 3] - box1[0, 1])
        box2_area = (box2[0, 2] - box2[0, 0]) * (box2[0, 3] - box2[0, 1])
        
        union_area = box1_area + box2_area - intersection_area
        
        # 计算IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def _calculate_ap(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """
        计算AP (Average Precision)
        
        Args:
            precision: 精度数组
            recall: 召回率数组
            
        Returns:
            float: AP值
        """
        # 确保精度和召回率以0开始
        precision = np.concatenate(([0.], precision, [0.]))
        recall = np.concatenate(([0.], recall, [1.]))
        
        # 平滑精度曲线
        for i in range(precision.size - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        
        # 查找召回率变化的点
        i = np.where(recall[1:] != recall[:-1])[0]
        
        # 计算AP
        ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
        
        return ap 