import traceback
import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
import json
from pathlib import Path
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torch.utils.data.dataloader import default_collate
from pycocotools import mask as mask_util
from transformers import DPTImageProcessor, DPTForDepthEstimation
from torch.utils.data._utils.collate import default_collate
from PIL import Image
import io
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)


def bytes_to_string_json_safe(obj):
    """递归转换嵌套数据结构中的bytes为字符串，使其可被JSON序列化"""
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif isinstance(obj, list):
        return [bytes_to_string_json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: bytes_to_string_json_safe(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return bytes_to_string_json_safe(obj.__dict__)
    else:
        return obj


def decode_mask(rle):
    """解码RLE格式的掩码为二维数组"""
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')
    mask = mask_util.decode(rle)
    return mask


def encode_mask(mask):
    """编码二维掩码为RLE格式"""
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(mask)
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def augment_mask(mask, flip_h=False, rot=False):
    """对掩码应用与图像相同的数据增强变换"""
    assert len(mask.shape) == 2, f"掩码应该是二维数组, 但是形状是 {mask.shape}"
    
    mask = mask.copy()
    
    if flip_h:
        mask = mask[:, ::-1]
    
    if rot:
        mask = mask.transpose(1, 0)
        mask = mask[:, ::-1]
    
    return mask


@DATASET_REGISTRY.register()
class TGSRDataset(data.Dataset):
    """文本引导的超分辨率数据集
    
    加载GT图像和文本描述，生成低质量图像和控制映射
    """

    def __init__(self, opt):
        super(TGSRDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_root, self.text_file = Path(opt['dataroot_gt']), opt['text_file']
        self.logger = get_root_logger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化控制映射生成组件
        self.use_depth_model = opt.get('use_depth_model')
        self.use_canny = opt.get('use_canny')
        self.canny_low = opt.get('canny_low')
        self.canny_high = opt.get('canny_high')
        
        # 加载深度估计模型
        if self.use_depth_model:
            try:
                self.logger.info("加载MiDaS深度模型...")
                depth_model_path = opt.get('depth_model_path')
                self.depth_processor = DPTImageProcessor.from_pretrained(depth_model_path, local_files_only=True)
                self.depth_model = DPTForDepthEstimation.from_pretrained(depth_model_path, local_files_only=True)
                self.depth_model = self.depth_model.to(self.device)
                self.depth_model.eval()
                self.logger.info(f"MiDaS模型加载成功: {depth_model_path}")
            except Exception as e:
                self.logger.error(f"深度模型加载失败: {e}")
                self.use_depth_model = False
        
        # 加载文本描述和对象信息
        try:
            with open(self.text_file, 'r') as f:
                self.captions = json.load(f)
            
            # 检查是否包含对象信息
            self.has_object_info = False
            if len(self.captions) > 0:
                first_item = self.captions[0] if isinstance(self.captions, list) else list(self.captions.values())[0]
                if isinstance(first_item, dict) and 'objects' in first_item:
                    if isinstance(first_item['objects'], list) and len(first_item['objects']) > 0:
                        if isinstance(first_item['objects'][0], dict) and 'mask_encoded' in first_item['objects'][0]:
                            self.has_object_info = True
                            self.logger.info("检测到对象掩码信息")
            
            self.logger.info(f"加载了{len(self.captions)}个文本描述")
        except Exception as e:
            self.logger.error(f"文本数据加载失败: {e}")
            self.captions = []
        
        # 构建图像路径列表
        self.paths = []
        for item in self.captions:
            if 'hr_path' in item:
                self.paths.append(item['hr_path'])
            elif 'image_id' in item:
                split = item.get('split', 'train')
                self.paths.append(osp.join(self.gt_root, split, 'hr', f"{item['image_id']}.jpg"))
            
            # 设置caption和objects获取方法
            self.get_caption = self._get_caption_list
            self.get_objects = self._get_objects_list
        
        # 数据增强设置
        self.use_hflip = opt.get('use_hflip')
        self.use_rot = opt.get('use_rot')
        self.gt_size = opt.get('gt_size')
        
        # 设置IO后端
        if 'io_backend' in self.opt and self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_root]
            self.io_backend_opt['client_keys'] = ['gt']
            if not str(self.gt_root).endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_root}")
            with open(osp.join(self.gt_root, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # 如果路径为空，从文件夹获取
            if len(self.paths) == 0:
                self.paths = self._get_paths_from_folder(self.gt_root)
                
        # 初始化退化参数
        # 第一阶段退化
        self.blur_kernel_size = opt.get('blur_kernel_size')
        self.kernel_list = opt.get('kernel_list')
        self.kernel_prob = opt.get('kernel_prob')
        self.blur_sigma = opt.get('blur_sigma')
        self.betag_range = opt.get('betag_range')
        self.betap_range = opt.get('betap_range')
        self.sinc_prob = opt.get('sinc_prob')

        # 第二阶段退化
        self.blur_kernel_size2 = opt.get('blur_kernel_size2')
        self.kernel_list2 = opt.get('kernel_list2')
        self.kernel_prob2 = opt.get('kernel_prob2')
        self.blur_sigma2 = opt.get('blur_sigma2')
        self.betag_range2 = opt.get('betag_range2')
        self.betap_range2 = opt.get('betap_range2')
        self.sinc_prob2 = opt.get('sinc_prob2')

        # 最终sinc滤波
        self.final_sinc_prob = opt.get('final_sinc_prob')

        # 核大小范围和脉冲张量
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # 7到21的核大小
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1
        
        # 控制映射权重
        self.canny_weight = opt.get('canny_weight', 0.4)
        self.depth_weight = opt.get('depth_weight', 0.3)
        self.mask_weight = opt.get('mask_weight', 0.3)

    def _get_paths_from_folder(self, folder):
        """获取文件夹中的图像路径"""
        img_paths = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.JPEG', '.tif')):
                    img_paths.append(os.path.join(root, file))
        return sorted(img_paths)
    
    def _generate_canny_map(self, img_gt):
        """生成Canny边缘图"""
        gray = cv2.cvtColor((img_gt * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        return edges.astype(np.float32) / 255.0
    
    def _generate_depth_map(self, img_gt):
        """使用MiDaS生成深度图"""
        if not self.use_depth_model:
            h, w = img_gt.shape[:2]
            return np.zeros((h, w), dtype=np.float32)

        try:
            # 转换为PIL并进行推理
            img_pil = Image.fromarray((img_gt * 255).astype(np.uint8))
            inputs = self.depth_processor(images=img_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth

            # 调整到原始尺寸
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(img_gt.shape[0], img_gt.shape[1]),
                mode="bicubic",
                align_corners=False,
            )

            # 归一化深度图
            depth_map = prediction.squeeze().cpu().numpy()
            depth_min, depth_max = np.min(depth_map), np.max(depth_map)
            if depth_max > depth_min:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_map = np.zeros_like(depth_map)

            return depth_map

        except Exception as e:
            self.logger.warning(f"深度图生成失败: {e}")
            traceback.print_exc()
            h, w = img_gt.shape[:2]
            return np.zeros((h, w), dtype=np.float32)
    
    def _generate_mask_map(self, objects_info, h, w):
        """从对象信息生成掩码图"""
        mask = np.zeros((h, w), dtype=np.float32)
        
        if not objects_info:
            return mask
        
        for obj in objects_info:
            if 'mask_encoded' in obj:
                try:
                    obj_mask = decode_mask(obj['mask_encoded'])
                    if obj_mask is not None and obj_mask.sum() > 0:
                        # 调整掩码大小
                        if obj_mask.shape != (h, w):
                            obj_mask_resized = cv2.resize(obj_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        else:
                            obj_mask_resized = obj_mask
                        
                        # 合并掩码
                        mask = np.maximum(mask, obj_mask_resized.astype(np.float32))
                except Exception as e:
                    self.logger.warning(f"掩码处理错误: {e}")
                    continue
        
        return mask
    
    def _create_control_map(self, img_gt, objects_info):
        """创建三通道控制映射：Canny边缘、深度图和对象掩码"""
        h, w = img_gt.shape[:2]
        
        # 生成各通道
        canny_map = self._generate_canny_map(img_gt) if self.use_canny else np.zeros((h, w), dtype=np.float32)
        depth_map = self._generate_depth_map(img_gt) if self.use_depth_model else np.zeros((h, w), dtype=np.float32)
        mask_map = self._generate_mask_map(objects_info, h, w)
        
        # 创建三通道控制映射并应用权重
        control_map = np.stack([
            canny_map * self.canny_weight,
            depth_map * self.depth_weight,
            mask_map * self.mask_weight
        ], axis=2)
        
        return control_map

    def __getitem__(self, index):
        # 初始化文件客户端
        if self.file_client is None:
            backend_opt = self.io_backend_opt.copy() if 'io_backend' in self.opt else {'type': 'disk'}
            self.file_client = FileClient(backend_opt.pop('type'), **backend_opt)
            
        # 获取正确的索引和图像路径
        index = index % len(self.paths)
        img_path = self.paths[index]
        
        # 处理图像路径
        if self.io_backend_opt.get('type') == 'lmdb':
            img_gt_path = img_path
        else:
            if osp.isabs(img_path) and osp.exists(img_path):
                img_gt_path = img_path
            else:
                img_filename = osp.basename(img_path)
                
                if not img_filename.endswith(('.jpg', '.png', '.jpeg', '.JPEG')):
                    img_gt_path = osp.join(self.gt_root, f'{img_filename}.jpg')
                    if not osp.exists(img_gt_path):
                        img_gt_path = osp.join(self.gt_root, f'{img_filename}.png')
                else:
                    img_gt_path = osp.join(self.gt_root, img_filename)
        
        # 获取文本描述和对象信息
        text_prompt = self.get_caption(index)
        objects_info = self.get_objects(index)
        
        # 加载图像（带重试机制）
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(img_gt_path)
                img_gt = imfrombytes(img_bytes, float32=True)
                break
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'文件读取错误: {e}, 剩余重试: {retry - 1}')
                # 随机选择另一个索引
                index = random.randint(0, self.__len__() - 1)
                img_path = self.paths[index]
                text_prompt = self.get_caption(index)
                objects_info = self.get_objects(index)
                
                # 更新图像路径
                if self.io_backend_opt.get('type') != 'lmdb':
                    if osp.isabs(img_path) and osp.exists(img_path):
                        img_gt_path = img_path
                    else:
                        img_filename = osp.basename(img_path)
                        if not img_filename.endswith(('.jpg', '.png', '.jpeg', '.JPEG')):
                            img_gt_path = osp.join(self.gt_root, f'{img_filename}.jpg')
                            if not osp.exists(img_gt_path):
                                img_gt_path = osp.join(self.gt_root, f'{img_filename}.png')
                        else:
                            img_gt_path = osp.join(self.gt_root, img_filename)
                
                time.sleep(1)  # 暂停1秒
            finally:
                retry -= 1
        
        if retry == 0:
            raise Exception(f"图像加载失败: {img_gt_path}")
            
        # 数据增强
        use_hflip = self.use_hflip and random.random() < 0.5
        use_rot = self.use_rot and random.random() < 0.5
        img_gt = augment(img_gt, use_hflip, use_rot)

        # 对掩码应用相同的增强
        if objects_info:
            for obj in objects_info:
                if 'mask_encoded' in obj:
                    try:
                        mask = decode_mask(obj['mask_encoded'])
                        mask = augment_mask(mask, use_hflip, use_rot)
                        obj['mask_encoded'] = encode_mask(mask)
                    except Exception as e:
                        logger = get_root_logger() if hasattr(self, 'logger') else None
                        if logger:
                            logger.warning(f"掩码增强错误: {e}")

        # 调整为固定尺寸
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.gt_size
        
        if h != crop_pad_size or w != crop_pad_size:
            # 根据情况选择合适的插值方法
            interpolation = cv2.INTER_AREA if h > crop_pad_size or w > crop_pad_size else cv2.INTER_LINEAR
            img_gt = cv2.resize(img_gt, (crop_pad_size, crop_pad_size), interpolation=interpolation)
            
            # 同步调整掩码尺寸
            if objects_info:
                for obj in objects_info:
                    if 'mask_encoded' in obj:
                        try:
                            mask = decode_mask(obj['mask_encoded'])
                            mask_resized = cv2.resize(mask, (crop_pad_size, crop_pad_size), 
                                                    interpolation=cv2.INTER_NEAREST)
                            obj['mask_encoded'] = encode_mask(mask_resized)
                        except Exception as e:
                            logger = get_root_logger() if hasattr(self, 'logger') else None
                            if logger:
                                logger.warning(f"掩码调整错误: {e}")
        
        # 确保尺寸正确
        assert img_gt.shape[0] == crop_pad_size and img_gt.shape[1] == crop_pad_size, \
            f"图像尺寸错误: {img_gt.shape}, 应为: {crop_pad_size}x{crop_pad_size}"
            
        # 创建控制映射
        control_map = self._create_control_map(img_gt, objects_info)

        # 生成模糊核 (用于第一阶段退化)
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # sinc滤波器设置
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # 填充核
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # 生成第二阶段退化的模糊核
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # 填充核
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # 最终的sinc核
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # 转换为tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        control_map = img2tensor([control_map], bgr2rgb=False, float32=True)[0]
        
        # 返回数据
        return_d = {
            'gt': img_gt, 
            'kernel1': kernel, 
            'kernel2': kernel2, 
            'sinc_kernel': sinc_kernel, 
            'gt_path': img_gt_path,
            'text_prompt': text_prompt,
            'control_map': control_map,
            'objects_info_str': json.dumps(bytes_to_string_json_safe(objects_info))
        }
        
        return return_d

    def __len__(self):
        return len(self.paths)

    def _get_caption_dict(self, idx):
        """从字典格式获取图像描述"""
        return self.captions[self.paths[idx]]
    
    def _get_caption_list(self, idx):
        """从列表格式获取图像描述"""
        return self.captions[idx].get('caption', '')
    
    def _get_objects_list(self, idx):
        """获取对象信息列表"""
        return self.captions[idx].get('objects', [])
