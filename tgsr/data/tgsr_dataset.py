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


def bytes_to_string_json_safe(obj):
    """递归地将嵌套数据结构中的bytes对象转换为字符串，使其可以被JSON序列化
    
    Args:
        obj: 要转换的对象，可以是任何Python数据类型
        
    Returns:
        转换后的对象，其中所有bytes已转换为字符串
    """
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif isinstance(obj, list):
        return [bytes_to_string_json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: bytes_to_string_json_safe(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):  # 对象类型
        return bytes_to_string_json_safe(obj.__dict__)
    else:
        return obj


def decode_mask(rle):
    """将RLE编码的掩码解码为二维数组"""
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')
    mask = mask_util.decode(rle)
    return mask


def encode_mask(mask):
    """将二维掩码编码为Run-Length编码格式"""
    # 使用pycocotools的RLE编码
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(mask)
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle


# 增强掩码与图像同步
def augment_mask(mask, flip_h=False, rot=False):
    """对掩码应用与图像相同的增强变换，确保与BasicSR中的augment函数完全一致
    
    Args:
        mask (np.ndarray): 二维掩码数组
        flip_h (bool): 是否水平翻转
        rot (bool): 是否旋转90度
        
    Returns:
        np.ndarray: 增强后的掩码
    """
    # 确保掩码是二维数组
    assert len(mask.shape) == 2, f"掩码应该是二维数组, 但是形状是 {mask.shape}"
    
    # 创建副本以避免修改原始数据
    mask = mask.copy()
    
    # 1. 水平翻转 (如需要) - 与basicsr完全一致
    if flip_h:
        mask = mask[:, ::-1]
    
    # 2. 旋转 (如需要) - 与basicsr完全一致：先转置后水平翻转
    if rot:
        mask = mask.transpose(1, 0)  # HW -> WH
        mask = mask[:, ::-1]  # 水平翻转
    
    return mask


@DATASET_REGISTRY.register()
class TGSRDataset(data.Dataset):
    """Dataset用于文本引导的超分辨率模型。
    
    它继承RealESRGAN数据集并支持文本提示。
    它加载GT图像并增强它们，同时生成用于低质量图像生成的模糊核和sinc核。
    低质量图像在GPU上以张量形式处理以实现更快的处理。

    Args:
        opt (dict): 训练数据集的配置。它包含以下键：
            dataroot_gt (str): gt的数据根路径。
            text_file (str): 文本提示文件的路径。
            io_backend (dict): IO后端类型和其他kwarg。
            use_hflip (bool): 使用水平翻转。
            use_rot (bool): 使用旋转（使用垂直翻转和转置h和w进行实现）。
            请参阅代码中的更多选项。
    """

    def __init__(self, opt):
        super(TGSRDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_root, self.text_file = Path(opt['dataroot_gt']), opt['text_file']
        self.logger = get_root_logger()
        
        # 读取文本描述和对象信息
        try:
            with open(self.text_file, 'r') as f:
                self.captions = json.load(f)
            
            # 检查是否为新格式 (带有对象信息)
            self.has_object_info = False
            if len(self.captions) > 0:
                first_item = self.captions[0] if isinstance(self.captions, list) else list(self.captions.values())[0]
                if isinstance(first_item, dict) and 'objects' in first_item:
                    if isinstance(first_item['objects'], list) and len(first_item['objects']) > 0:
                        if isinstance(first_item['objects'][0], dict) and 'mask_encoded' in first_item['objects'][0]:
                            self.has_object_info = True
                            self.logger.info("检测到包含对象掩码信息的数据集")
            
            self.logger.info(f"加载了 {len(self.captions)} 个文本描述和对象信息")
        except Exception as e:
            self.logger.error(f"加载文本描述和对象信息失败: {e}")
            self.captions = []
        
        # 判断是否为旧格式（直接映射）还是新格式（列表格式）
        if isinstance(self.captions, dict):
            # 旧格式：字典映射 image_id -> caption
            self.paths = sorted(list(self.captions.keys()))
            self.get_caption = lambda idx: self.captions[self.paths[idx]]
            self.get_objects = lambda idx: []  # 旧格式没有对象信息
        else:
            # 新格式：列表格式，每项包含image_id, caption, objects等
            self.paths = []
            for item in self.captions:
                if 'hr_path' in item:
                    self.paths.append(item['hr_path'])
                elif 'image_id' in item:
                    # 基于图像ID构建路径
                    split = item.get('split', 'train')
                    self.paths.append(osp.join(self.gt_root, split, 'hr', f"{item['image_id']}.jpg"))
            
            self.get_caption = lambda idx: self.captions[idx].get('caption', '')
            self.get_objects = lambda idx: self.captions[idx].get('objects', [])
        
        # 初始化数据增强选项
        self.use_hflip = opt.get('use_hflip', False)
        self.use_rot = opt.get('use_rot', False)  # 不推荐使用旋转，会影响文本与图像的一致性
        self.gt_size = opt.get('gt_size', 256)
        
        # file client (io backend)
        if 'io_backend' in self.opt and self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_root]
            self.io_backend_opt['client_keys'] = ['gt']
            if not str(self.gt_root).endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_root}")
            with open(osp.join(self.gt_root, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend with explicit paths
            # We search for all files in the gt_folder
            if len(self.paths) == 0:
                self.paths = self._get_paths_from_folder(self.gt_root)

        # 初始化退化参数 (从opt获取或使用默认值)
        # blur settings for the first degradation
        self.blur_kernel_size = opt.get('blur_kernel_size', 21)
        self.kernel_list = opt.get('kernel_list', ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'])
        self.kernel_prob = opt.get('kernel_prob', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.blur_sigma = opt.get('blur_sigma', [0.2, 3.0])
        self.betag_range = opt.get('betag_range', [0.5, 4.0])
        self.betap_range = opt.get('betap_range', [1, 2.0])
        self.sinc_prob = opt.get('sinc_prob', 0.1)

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt.get('blur_kernel_size2', 21)
        self.kernel_list2 = opt.get('kernel_list2', ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'])
        self.kernel_prob2 = opt.get('kernel_prob2', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.blur_sigma2 = opt.get('blur_sigma2', [0.2, 1.5])
        self.betag_range2 = opt.get('betag_range2', [0.5, 4.0])
        self.betap_range2 = opt.get('betap_range2', [1, 2.0])
        self.sinc_prob2 = opt.get('sinc_prob2', 0.1)

        # a final sinc filter
        self.final_sinc_prob = opt.get('final_sinc_prob', 0.8)

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def _get_paths_from_folder(self, folder):
        """Get image paths from folder."""
        img_paths = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.JPEG', '.tif')):
                    img_paths.append(os.path.join(root, file))
        return sorted(img_paths)

    def __getitem__(self, index):
        # 初始化文件客户端（如果尚未初始化）
        if self.file_client is None:
            # 复制io_backend_opt以避免修改原始字典
            backend_opt = self.io_backend_opt.copy() if 'io_backend' in self.opt else {'type': 'disk'}
            self.file_client = FileClient(backend_opt.pop('type'), **backend_opt)
            
        # 获取正确的索引
        index = index % len(self.paths)
        img_path = self.paths[index]
        
        # 获取图像路径 - 简化路径处理逻辑
        if self.io_backend_opt.get('type') == 'lmdb':
            # LMDB数据库使用原始路径
            img_gt_path = img_path
        else:
            # 1. 如果是绝对路径且存在，直接使用
            if osp.isabs(img_path) and osp.exists(img_path):
                img_gt_path = img_path
            else:
                # 2. 获取文件名（去除路径部分）
                img_filename = osp.basename(img_path)
                
                # 3. 处理可能的没有扩展名的情况
                if not img_filename.endswith(('.jpg', '.png', '.jpeg', '.JPEG')):
                    # 尝试不同的扩展名
                    img_gt_path = osp.join(self.gt_root, f'{img_filename}.jpg')
                    if not osp.exists(img_gt_path):
                        img_gt_path = osp.join(self.gt_root, f'{img_filename}.png')
                else:
                    # 直接拼接路径
                    img_gt_path = osp.join(self.gt_root, img_filename)
        
        # 获取文本描述和对象信息
        text_prompt = self.get_caption(index)
        objects_info = self.get_objects(index)
        
        # 加载图像 - 添加重试逻辑
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(img_gt_path)
                img_gt = imfrombytes(img_bytes, float32=True)
                break
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'文件客户端错误: {e}, 剩余重试次数: {retry - 1}')
                # 随机选择另一个索引
                index = random.randint(0, self.__len__() - 1)
                img_path = self.paths[index]
                text_prompt = self.get_caption(index)
                objects_info = self.get_objects(index)
                
                # 重新获取图像路径
                if self.io_backend_opt.get('type') != 'lmdb':
                    # 同样简化路径处理逻辑
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
                
                time.sleep(1)  # 服务器拥塞时暂停1秒
            finally:
                retry -= 1
        
        if retry == 0:
            raise Exception(f"加载图像失败，已重试3次: {img_gt_path}")
            
        # 记录是否使用翻转和旋转
        use_hflip = self.use_hflip and random.random() < 0.5
        use_rot = self.use_rot and random.random() < 0.5
            
        # 图像增强处理 - 使用与RealESRGAN相同的方法
        img_gt = augment(img_gt, use_hflip, use_rot)

        # 同步对对象的掩码进行增强
        if objects_info:
            for obj in objects_info:
                if 'mask_encoded' in obj:
                    try:
                        # 解码掩码
                        mask = decode_mask(obj['mask_encoded'])
                        
                        # 对掩码应用相同的增强
                        mask = augment_mask(mask, use_hflip, use_rot)
                        
                        # 重新编码掩码
                        obj['mask_encoded'] = encode_mask(mask)
                    except Exception as e:
                        logger = get_root_logger() if hasattr(self, 'logger') else None
                        if logger:
                            logger.warning(f"处理掩码时出错: {e}")
                        # 如果处理失败，保留原始掩码

        # 调整到固定尺寸 - 确保输出始终是crop_pad_size x crop_pad_size
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.gt_size
        
        # 如果图像尺寸与目标尺寸不同，使用缩放并确保尺寸精确匹配
        if h != crop_pad_size or w != crop_pad_size:
            # 使用INTER_AREA进行下采样（缩小），使用INTER_LINEAR进行上采样（放大）
            interpolation = cv2.INTER_AREA if h > crop_pad_size or w > crop_pad_size else cv2.INTER_LINEAR
            img_gt = cv2.resize(img_gt, (crop_pad_size, crop_pad_size), interpolation=interpolation)
            
            # 同样调整所有对象的掩码大小
            if objects_info:
                for obj in objects_info:
                    if 'mask_encoded' in obj:
                        try:
                            # 解码掩码
                            mask = decode_mask(obj['mask_encoded'])
                            
                            # 调整掩码大小 - 使用INTER_NEAREST保留二值性质
                            mask_resized = cv2.resize(mask, (crop_pad_size, crop_pad_size), 
                                                     interpolation=cv2.INTER_NEAREST)
                            
                            # 重新编码掩码
                            obj['mask_encoded'] = encode_mask(mask_resized)
                        except Exception as e:
                            logger = get_root_logger() if hasattr(self, 'logger') else None
                            if logger:
                                logger.warning(f"调整掩码大小时出错: {e}")
        
        # 确保大小正确
        assert img_gt.shape[0] == crop_pad_size and img_gt.shape[1] == crop_pad_size, \
            f"图像尺寸错误: {img_gt.shape}, 应为: {crop_pad_size}x{crop_pad_size}"

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
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
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
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

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        
        # 返回必要的训练数据
        return_d = {
            'gt': img_gt, 
            'kernel1': kernel, 
            'kernel2': kernel2, 
            'sinc_kernel': sinc_kernel, 
            'gt_path': img_gt_path,
            'text_prompt': text_prompt,
            'objects_info_str': json.dumps(bytes_to_string_json_safe(objects_info))  # 先处理bytes再序列化
        }
        
        return return_d

    def __len__(self):
        return len(self.paths)
