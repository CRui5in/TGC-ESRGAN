import os
import json
import random
import zipfile
import traceback
from pathlib import Path
import math
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from concurrent.futures import ThreadPoolExecutor, as_completed

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 设置随机种子以确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    np.random.seed(seed)
    random.seed(seed)


# 解压缩COCO数据集
def extract_zip(zip_path, extract_to):
    print(f"正在解压 {zip_path} 到 {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"解压完成: {zip_path}")


# 生成从类别ID到类别名称的映射
def get_category_mapping(coco):
    """从COCO对象获取类别ID到类别名称的映射"""
    categories = coco.loadCats(coco.getCatIds())
    return {cat['id']: cat['name'] for cat in categories}


# 加载COCO图像的说明文字
def get_captions_for_image(image_id, coco_captions):
    """获取特定图像的所有说明文字"""
    ann_ids = coco_captions.getAnnIds(imgIds=image_id)
    anns = coco_captions.loadAnns(ann_ids)
    return [ann['caption'] for ann in anns]


# 转换图像为张量
def img_to_tensor(img, device=None):
    """将OpenCV图像转换为PyTorch张量"""
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1) / 255.0).float()
    if device is not None:
        img_tensor = img_tensor.to(device)
    return img_tensor.unsqueeze(0)  # 添加批次维度


# 转换张量为图像
def tensor_to_img(tensor):
    """将PyTorch张量转换为OpenCV图像"""
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    img_np = tensor.squeeze(0).permute(1, 2, 0).numpy() * 255
    return np.clip(img_np, 0, 255).astype(np.uint8)


# 调整图像大小
def resize_image(image, target_size=None, scale_factor=None):
    """
    调整图像大小

    参数:
    - image: 输入图像
    - target_size: 目标大小 (宽, 高) 或 None
    - scale_factor: 缩放因子 或 None

    返回:
    - 调整大小后的图像
    """
    if target_size is not None:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    elif scale_factor is not None:
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    return image


# 创建数据集划分 (训练/验证/测试)
def create_dataset_splits(coco_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    创建训练、验证和测试集划分

    参数:
    - coco_dir: COCO数据集根目录
    - output_dir: 输出目录
    - train_ratio: 训练集比例
    - val_ratio: 验证集比例
    - seed: 随机种子

    返回:
    - 包含训练、验证和测试集图像ID的字典
    """
    set_seed(seed)

    # 计算测试集比例
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "训练和验证集比例总和不能超过1"

    # 初始化COCO API (使用训练集2017)
    instance_file = os.path.join(coco_dir, "annotations", "instances_train2017.json")
    coco_instance = COCO(instance_file)

    # 获取所有图像ID并随机打乱
    image_ids = list(coco_instance.imgs.keys())
    random.shuffle(image_ids)

    # 计算各集合的大小
    total_images = len(image_ids)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)

    # 划分数据集
    train_ids = image_ids[:train_size]
    val_ids = image_ids[train_size:train_size + val_size]
    test_ids = image_ids[train_size + val_size:]

    print(f"数据集划分: 训练集 {len(train_ids)}张, 验证集 {len(val_ids)}张, 测试集 {len(test_ids)}张")

    # 创建各数据集目录
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(os.path.join(split_dir, "hr"), exist_ok=True)

    # 返回划分的ID列表
    return {"train": train_ids, "val": val_ids, "test": test_ids}


# 处理单个图像并创建训练样本
def process_image(img_id, coco_instance, coco_captions, category_map, coco_dir, output_dir, split_name,
                  jpeg_quality=90, resize_scale=None, split_info=None):
    """
    处理单个图像并创建训练样本

    参数:
    - img_id: 图像ID
    - coco_instance: COCO实例API对象
    - coco_captions: COCO说明文字API对象
    - category_map: 类别ID到类别名称的映射
    - coco_dir: COCO数据集目录
    - output_dir: 输出目录
    - split_name: 数据集划分名称 (train/val/test)
    - jpeg_quality: JPEG压缩质量 (1-100)
    - resize_scale: 缩放比例 (可选)
    - split_info: 数据集划分信息 (用于记录)

    返回:
    - 处理的样本信息列表
    """
    try:
        # 获取图像信息
        img_info = coco_instance.loadImgs(img_id)[0]

        # 确定图像路径
        img_path = os.path.join(coco_dir, "train2017", img_info['file_name'])

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图像: {img_path}")
            return []

        # 检查是否是彩色图像
        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"跳过非彩色图像: {img_path}, shape: {image.shape}")
            return []

        # 可选: 调整图像大小以节省空间
        if resize_scale is not None and resize_scale != 1.0:
            image = resize_image(image, scale_factor=resize_scale)

        # 获取图像的captions
        captions = get_captions_for_image(img_id, coco_captions)
        if not captions:
            return []  # 没有captions跳过

        # 随机选择一个caption
        caption = random.choice(captions)

        # 获取图像中的所有实例标注
        ann_ids = coco_instance.getAnnIds(imgIds=img_id)
        anns = coco_instance.loadAnns(ann_ids)

        # 确定输出路径
        image_id = f"{img_id:012d}"
        split_dir = os.path.join(output_dir, split_name)
        hr_path = os.path.join(split_dir, "hr", f"{image_id}.jpg")

        # 保存高分辨率图像 (原始图像)
        cv2.imwrite(hr_path, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        # 记录标注的对象类别
        objects = []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in category_map:
                objects.append(category_map[cat_id])

        # 记录样本信息
        sample_info = {
            "image_id": image_id,
            "original_id": img_id,
            "caption": caption,
            "objects": list(set(objects)),  # 去重
            "hr_path": hr_path,
            "split": split_name
        }

        # 如果有split_info，添加到对应的列表
        if split_info is not None:
            split_info.append(sample_info)

        return [sample_info]

    except Exception as e:
        print(f"处理图像 {img_id} 时出错: {str(e)}")
        traceback.print_exc()
        return []


# 批处理图像 (利用多线程)
def batch_process_images(image_ids, coco_instance, coco_captions, category_map, coco_dir, output_dir, split_name,
                         jpeg_quality=90, resize_scale=None, split_info=None, max_workers=8):
    """
    批量处理图像 (可以利用多线程加速)

    参数:
    - image_ids: 图像ID列表
    - coco_instance: COCO实例API对象
    - coco_captions: COCO说明文字API对象
    - category_map: 类别ID到类别名称的映射
    - coco_dir: COCO数据集目录
    - output_dir: 输出目录
    - split_name: 数据集划分名称 (train/val/test)
    - jpeg_quality: JPEG压缩质量 (1-100)
    - resize_scale: 缩放比例 (可选)
    - split_info: 数据集划分信息 (用于记录)
    - max_workers: 最大工作线程数
    """
    all_processed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for img_id in image_ids:
            future = executor.submit(
                process_image,
                img_id=img_id,
                coco_instance=coco_instance,
                coco_captions=coco_captions,
                category_map=category_map,
                coco_dir=coco_dir,
                output_dir=output_dir,
                split_name=split_name,
                jpeg_quality=jpeg_quality,
                resize_scale=resize_scale,
                split_info=None  # 不在线程中直接修改，避免竞争条件
            )
            futures.append(future)

        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"处理 {split_name} 图像"):
            result = future.result()
            if result:
                all_processed.extend(result)
                if split_info is not None:
                    split_info.extend(result)

    return all_processed


# 主函数：处理COCO数据集并创建文本引导超分辨率训练数据
def create_tgsr_dataset(coco_dir, output_dir, jpeg_quality=90, resize_scale=None, max_workers=8, max_images=None):
    """
    处理COCO数据集并创建文本引导超分辨率训练数据

    参数:
    - coco_dir: COCO数据集根目录
    - output_dir: 输出目录
    - jpeg_quality: JPEG压缩质量 (1-100)
    - resize_scale: 缩放比例 (可选，如0.5表示缩小到50%)
    - max_workers: 最大工作线程数
    - max_images: 每个划分最大处理的图像数量，默认为None表示处理所有图像

    返回:
    - 包含所有样本信息的字典
    """
    set_seed(42)

    print(f"创建文本引导超分辨率数据集 - JPEG质量: {jpeg_quality}, 缩放比例: {resize_scale if resize_scale else '原始尺寸'}")

    # 创建数据集划分
    splits = create_dataset_splits(coco_dir, output_dir)

    # 初始化COCO实例API
    instance_file = os.path.join(coco_dir, "annotations", "instances_train2017.json")
    coco_instance = COCO(instance_file)

    # 初始化COCO说明文字API
    caption_file = os.path.join(coco_dir, "annotations", "captions_train2017.json")
    coco_captions = COCO(caption_file)

    # 获取类别映射
    category_map = get_category_mapping(coco_instance)

    # 存储每个划分的样本信息
    split_samples = {"train": [], "val": [], "test": []}

    # 处理训练、验证和测试集
    for split_name in ["train", "val", "test"]:
        print(f"\n处理 {split_name} 集...")
        
        # 如果指定了最大图像数量，则限制处理的图像数量
        if max_images is not None:
            image_ids = splits[split_name][:max_images]
            print(f"限制处理{split_name}集的前{max_images}张图像（共{len(splits[split_name])}张）")
        else:
            image_ids = splits[split_name]

        # 使用批处理图像加速
        batch_process_images(
            image_ids=image_ids,
            coco_instance=coco_instance,
            coco_captions=coco_captions,
            category_map=category_map,
            coco_dir=coco_dir,
            output_dir=output_dir,
            split_name=split_name,
            jpeg_quality=jpeg_quality,
            resize_scale=resize_scale,
            split_info=split_samples[split_name],
            max_workers=max_workers
        )

    # 保存每个划分的提示映射为JSON文件
    for split_name in split_samples:
        prompts_file = os.path.join(output_dir, f"{split_name}_captions.json")
        with open(prompts_file, 'w') as f:
            json.dump(split_samples[split_name], f, indent=2)

        print(f"{split_name} 集: 处理了 {len(split_samples[split_name])} 个样本")

    # 统计训练集中的对象分布
    object_counts = {}
    for item in split_samples["train"]:
        for obj in item["objects"]:
            if obj not in object_counts:
                object_counts[obj] = 0
            object_counts[obj] += 1

    # 输出对象分布
    print("\n训练集对象分布:")
    for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"{obj}: {count} 样本")

    # 创建汇总信息
    all_samples = {
        "train": split_samples["train"],
        "val": split_samples["val"],
        "test": split_samples["test"]
    }

    # 保存汇总信息
    all_samples_file = os.path.join(output_dir, "all_samples.json")
    with open(all_samples_file, 'w') as f:
        json.dump(all_samples, f, indent=2)

    print(f"\n数据集创建完成! 共处理 {sum(len(samples) for samples in split_samples.values())} 个样本")
    print(f"数据集保存在: {output_dir}")

    return all_samples


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从COCO2017数据集创建文本引导超分辨率数据集')
    parser.add_argument('--coco_dir', type=str, default='./COCO2017', help='COCO数据集根目录路径')
    parser.add_argument('--output_dir', type=str, default='./tgsr_dataset_hr_only', help='输出数据集目录路径')
    parser.add_argument('--jpeg_quality', type=int, default=100, help='JPEG压缩质量 (1-100)')
    parser.add_argument('--resize_scale', type=float, default=None, help='图像缩放比例 (可选，如0.5表示缩小到50%)')
    parser.add_argument('--max_workers', type=int, default=8, help='最大工作线程数')
    parser.add_argument('--max_images', type=int, default=None, help='每个划分最大处理的图像数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建数据集
    all_samples = create_tgsr_dataset(
        coco_dir=args.coco_dir,
        output_dir=args.output_dir,
        jpeg_quality=args.jpeg_quality,
        resize_scale=args.resize_scale,
        max_workers=args.max_workers,
        max_images=args.max_images
    ) 