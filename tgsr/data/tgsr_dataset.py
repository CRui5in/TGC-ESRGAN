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
from pycocotools import mask as mask_util


def decode_mask(rle):
    """将RLE编码的掩码解码为二维数组"""
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')
    mask = mask_util.decode(rle)
    return mask


def apply_targeted_degradation(img, objects_info, text_prompt, degradation_level=0.0):
    """
    对图像应用有针对性的退化，直接使用所有objects标注区域进行轻度退化
    
    参数:
    - img: 输入图像 [H, W, C]
    - objects_info: 对象信息列表，包含掩码和类别
    - text_prompt: 文本描述 (不再使用匹配)
    - degradation_level: 现在仅用作占位符参数，实际上使用二值掩码 (0表示文本区域，1表示其他区域)
    
    返回:
    - degradation_mask: 退化区域掩码 [H, W]，二值掩码
      值为0的区域表示文本区域，将获得轻度退化处理
      值为1的区域表示其他区域，将获得标准退化处理
    """
    h, w = img.shape[0], img.shape[1]
    
    # 初始化退化掩码，默认为1.0（标准退化）
    degradation_mask = np.ones((h, w), dtype=np.float32)
    
    # 遍历所有对象 - 不再检查文本匹配
    for obj in objects_info:
        # 解码对象掩码
        if 'mask_encoded' in obj:
            try:
                obj_mask = decode_mask(obj['mask_encoded'])
                # 在掩码区域应用较轻的退化 - 将退化掩码值设为0.0（表示文本区域）
                degradation_mask[obj_mask > 0] = 0.0
                category = obj.get('category', 'unknown')
            except Exception as e:
                print(f"解码掩码失败: {e}")
                continue
    
    return degradation_mask


@DATASET_REGISTRY.register()
class TGSRDataset_basicsr(data.Dataset):
    """Dataset used for Text-Guided Super-Resolution model.
    It extends RealESRGAN dataset to support text prompts.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            text_file (str): Path for text prompts file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(TGSRDataset_basicsr, self).__init__()
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
                            self.logger.info("检测到包含对象掩码信息的数据集，将启用有针对性的退化")
            
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
        self.use_hflip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', False)  # 不推荐使用旋转，会影响文本与图像的一致性
        self.gt_size = opt.get('gt_size', 256)
        
        # 有针对性退化的设置 - 不再使用text_region_factor，而是直接用二值掩码
        self.use_targeted_degradation = opt.get('use_targeted_degradation', True) and self.has_object_info
        
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
            
        # 图像处理 - 使用水平翻转但避免随机裁剪
        # 水平翻转不会影响文本与图像内容的一致性
        if self.use_hflip and random.random() < 0.5:
            img_gt = cv2.flip(img_gt, 1)  # 水平翻转
            
            # 如果水平翻转了图像，也需要翻转对象掩码
            if self.has_object_info and objects_info:
                # 获取图像宽度
                img_width = img_gt.shape[1]
                
                # 翻转每个对象的边界框和掩码
                for obj in objects_info:
                    if 'bbox' in obj:
                        # 边界框格式: [x, y, width, height]
                        x, y, w, h = obj['bbox']
                        # 翻转x坐标
                        obj['bbox'][0] = img_width - x - w
                    
                    # 翻转掩码（如果有）
                    if 'mask_encoded' in obj:
                        try:
                            mask = decode_mask(obj['mask_encoded'])
                            # 翻转掩码
                            flipped_mask = np.fliplr(mask)
                            # 重新编码
                            obj['mask_encoded'] = encode_mask(flipped_mask)
                        except Exception as e:
                            print(f"翻转掩码失败: {e}")
        
        # 处理图像尺寸 - 直接调整到固定尺寸以确保批处理兼容性
        target_size = self.gt_size
        h, w = img_gt.shape[:2]
        
        # 直接调整到目标尺寸，不保持长宽比
        img_gt = cv2.resize(img_gt, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        
        # 更新对象掩码和边界框的尺寸
        if self.has_object_info and objects_info:
            scale_h = target_size / h
            scale_w = target_size / w
            for obj in objects_info:
                if 'bbox' in obj:
                    # 根据缩放比例调整边界框
                    obj['bbox'][0] = int(obj['bbox'][0] * scale_w)
                    obj['bbox'][1] = int(obj['bbox'][1] * scale_h)
                    obj['bbox'][2] = int(obj['bbox'][2] * scale_w)
                    obj['bbox'][3] = int(obj['bbox'][3] * scale_h)
                
                # 更新掩码尺寸
                if 'mask_encoded' in obj:
                    try:
                        mask = decode_mask(obj['mask_encoded'])
                        resized_mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                        obj['mask_encoded'] = encode_mask(resized_mask)
                    except Exception as e:
                        print(f"调整掩码尺寸失败: {e}")

        # 创建有针对性的退化掩码 (对文本提及的对象区域应用较轻的退化)
        degradation_mask = None
        if self.use_targeted_degradation and self.has_object_info:
            # 调用apply_targeted_degradation函数，为文本中提到的对象创建轻度退化掩码
            degradation_mask = apply_targeted_degradation(
                img_gt, objects_info, text_prompt, 0.0
            )
            
        # 即使没有对象或没有启用有针对性退化，也创建一个标准掩码（所有值为1.0）
        if degradation_mask is None:
            h, w = img_gt.shape[0], img_gt.shape[1]
            degradation_mask = np.ones((h, w), dtype=np.float32)
                
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
        
        # 转换退化掩码为张量（如果有）
        if degradation_mask is not None:
            degradation_mask = torch.FloatTensor(degradation_mask).unsqueeze(0)  # [1, H, W]
            
        # 简化返回数据，只保留必要的信息
        # 创建综合掩码 - 将所有文本相关区域合并为一个掩码
        h, w = self.gt_size, self.gt_size
        text_regions_mask = torch.ones((1, h, w), dtype=torch.float32)  # 默认值为1（标准退化）
        
        if self.has_object_info and objects_info and text_prompt:
            # 只有在有对象信息、对象列表非空且有文本描述时才处理
            text_lower = text_prompt.lower()
            for obj in objects_info:
                if 'category' in obj and obj['category'].lower() in text_lower:
                    if 'mask_encoded' in obj:
                        try:
                            obj_mask = decode_mask(obj['mask_encoded'])
                            # 对文本提及的对象区域设置为0.0（表示文本区域，将获得轻度退化）
                            text_regions_mask[0][obj_mask > 0] = 0.0
                        except Exception as e:
                            print(f"解码掩码失败: {e}")
                            continue
        
        # 返回必要的训练数据，避免复杂的数据结构
        return_d = {
            'gt': img_gt, 
            'kernel1': kernel, 
            'kernel2': kernel2, 
            'sinc_kernel': sinc_kernel, 
            'gt_path': img_gt_path,
            'text_prompt': text_prompt,
            'text_regions_mask': text_regions_mask,  # 包含所有文本相关区域的单一掩码
        }
        
        return return_d

    def __len__(self):
        return len(self.paths)
        
        
def encode_mask(mask):
    """将二维掩码编码为Run-Length编码格式"""
    # 使用pycocotools的RLE编码
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(mask)
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle 