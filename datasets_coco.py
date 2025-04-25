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
from pycocotools import mask as mask_util
from concurrent.futures import ThreadPoolExecutor, as_completed

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 为重现性设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# 解压COCO数据集
def extract_zip(zip_path, extract_to):
    print(f"正在解压 {zip_path} 到 {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"解压完成: {zip_path}")


# 获取类别ID到名称的映射
def get_category_mapping(coco):
    categories = coco.loadCats(coco.getCatIds())
    return {cat['id']: cat['name'] for cat in categories}


# 获取图像的说明文字
def get_captions_for_image(image_id, coco_captions):
    ann_ids = coco_captions.getAnnIds(imgIds=image_id)
    anns = coco_captions.loadAnns(ann_ids)
    return [ann['caption'] for ann in anns]


# 编码掩码为RLE格式
def encode_mask(mask):
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(mask)
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle


# 转换图像为张量
def img_to_tensor(img, device=None):
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1) / 255.0).float()
    if device is not None:
        img_tensor = img_tensor.to(device)
    return img_tensor.unsqueeze(0)  # 添加批次维度


# 转换张量为图像
def tensor_to_img(tensor):
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    img_np = tensor.squeeze(0).permute(1, 2, 0).numpy() * 255
    return np.clip(img_np, 0, 255).astype(np.uint8)


# 调整图像大小
def resize_image(image, target_size=None, scale_factor=None):
    if target_size is not None:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    elif scale_factor is not None:
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
    return image


# 创建数据集划分
def create_dataset_splits(coco_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    set_seed(seed)

    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "训练和验证集比例总和不能超过1"

    # 初始化COCO API
    instance_file = os.path.join(coco_dir, "annotations", "instances_train2017.json")
    coco_instance = COCO(instance_file)

    # 获取所有图像ID并打乱
    image_ids = list(coco_instance.imgs.keys())
    random.shuffle(image_ids)

    # 计算各集合大小
    total_images = len(image_ids)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)

    # 划分数据集
    train_ids = image_ids[:train_size]
    val_ids = image_ids[train_size:train_size + val_size]
    test_ids = image_ids[train_size + val_size:]

    print(f"数据集划分: 训练集 {len(train_ids)}张, 验证集 {len(val_ids)}张, 测试集 {len(test_ids)}张")

    # 创建目录
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(os.path.join(split_dir, "hr"), exist_ok=True)

    return {"train": train_ids, "val": val_ids, "test": test_ids}


# 处理单个图像
def process_image(img_id, coco_instance, coco_captions, category_map, coco_dir, output_dir, split_name,
                  jpeg_quality=90, resize_scale=None, split_info=None):
    try:
        # 获取图像信息
        img_info = coco_instance.loadImgs(img_id)[0]
        
        # 获取原始尺寸
        original_width = img_info['width']
        original_height = img_info['height']

        # 图像路径
        img_path = os.path.join(coco_dir, "train2017", img_info['file_name'])

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图像: {img_path}")
            return []

        # 检查是否彩色图像
        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"跳过非彩色图像: {img_path}, shape: {image.shape}")
            return []

        # 获取captions
        captions = get_captions_for_image(img_id, coco_captions)
        if not captions:
            return []

        # 合并captions
        combined_caption = " ".join(captions)

        # 获取实例标注
        ann_ids = coco_instance.getAnnIds(imgIds=img_id)
        anns = coco_instance.loadAnns(ann_ids)

        # 创建对象信息
        objects_info = []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in category_map:
                obj_name = category_map[cat_id]
                
                # 获取掩码
                mask = coco_instance.annToMask(ann)
                
                if np.sum(mask) > 0:
                    # 边界框
                    bbox = ann.get('bbox', [0, 0, 0, 0])  # [x, y, width, height]
                    
                    # 保存对象信息
                    objects_info.append({
                        'category': obj_name,
                        'bbox': bbox,
                        'area': float(ann.get('area', 0)),
                        'mask_encoded': encode_mask(mask)
                    })

        # 输出路径
        image_id = f"{img_id:012d}"
        split_dir = os.path.join(output_dir, split_name)
        hr_path = os.path.join(split_dir, "hr", f"{image_id}.jpg")

        # 调整图像大小
        resized_width = original_width
        resized_height = original_height
        if resize_scale is not None and resize_scale != 1.0:
            image = resize_image(image, scale_factor=resize_scale)
            resized_width = int(original_width * resize_scale)
            resized_height = int(original_height * resize_scale)

        # 保存高分辨率图像
        cv2.imwrite(hr_path, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        # 记录样本信息
        sample_info = {
            "image_id": image_id,
            "original_id": img_id,
            "caption": combined_caption,
            "objects": objects_info,
            "hr_path": hr_path,
            "original_width": original_width,
            "original_height": original_height,
            "split": split_name
        }

        if split_info is not None:
            split_info.append(sample_info)

        return [sample_info]

    except Exception as e:
        print(f"处理图像 {img_id} 时出错: {str(e)}")
        traceback.print_exc()
        return []


# 批处理图像
def batch_process_images(image_ids, coco_instance, coco_captions, category_map, coco_dir, output_dir, split_name,
                         jpeg_quality=90, resize_scale=None, split_info=None, max_workers=8):
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
                split_info=None
            )
            futures.append(future)

        # 显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"处理 {split_name} 图像"):
            result = future.result()
            if result:
                all_processed.extend(result)
                if split_info is not None:
                    split_info.extend(result)

    return all_processed


# 创建文本引导超分数据集
def create_tgsr_dataset(coco_dir, output_dir, jpeg_quality=90, resize_scale=None, max_workers=8, max_images=None):
    set_seed(42)

    print(f"创建文本引导超分辨率数据集 - JPEG质量: {jpeg_quality}, 缩放比例: {resize_scale if resize_scale else '原始尺寸'}")

    # 创建数据集划分
    splits = create_dataset_splits(coco_dir, output_dir)

    # 初始化COCO API
    instance_file = os.path.join(coco_dir, "annotations", "instances_train2017.json")
    coco_instance = COCO(instance_file)

    caption_file = os.path.join(coco_dir, "annotations", "captions_train2017.json")
    coco_captions = COCO(caption_file)

    # 获取类别映射
    category_map = get_category_mapping(coco_instance)

    # 存储样本信息
    split_samples = {"train": [], "val": [], "test": []}

    # 处理各个数据集
    for split_name in ["train", "val", "test"]:
        print(f"\n处理 {split_name} 集...")
        
        # 限制图像数量
        if max_images is not None:
            image_ids = splits[split_name][:max_images]
            print(f"限制处理{split_name}集的前{max_images}张图像（共{len(splits[split_name])}张）")
        else:
            image_ids = splits[split_name]

        # 批处理图像
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

    # 保存提示映射
    for split_name in split_samples:
        prompts_file = os.path.join(output_dir, f"{split_name}_captions.json")
        with open(prompts_file, 'w') as f:
            json.dump(split_samples[split_name], f, indent=2)

        print(f"{split_name} 集: 处理了 {len(split_samples[split_name])} 个样本")

    # 统计对象分布
    object_counts = {}
    for item in split_samples["train"]:
        for obj in item["objects"]:
            category = obj["category"]
            if category not in object_counts:
                object_counts[category] = 0
            object_counts[category] += 1

    # 输出分布
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
    parser.add_argument('--coco_dir', type=str, default='/root/autodl-tmp/COCO2017', help='COCO数据集根目录路径')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/tgsr_dataset_hr_only', help='输出数据集目录路径')
    parser.add_argument('--jpeg_quality', type=int, default=100, help='JPEG压缩质量 (1-100)')
    parser.add_argument('--resize_scale', type=float, default=None, help='图像缩放比例 (可选，如0.5表示缩小到50%)')
    parser.add_argument('--max_workers', type=int, default=8, help='最大工作线程数')
    parser.add_argument('--max_images', type=int, default=None, help='每个划分最大处理的图像数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    # 解压COCO数据集
    coco_zip_dir = "/root/autodl-pub/COCO2017/"
    extract_to = "/root/autodl-tmp/COCO2017/"

    # 解压所有COCO压缩文件
    # for zip_file in os.listdir(coco_zip_dir):
    #     if zip_file.endswith('.zip'):
    #         zip_path = os.path.join(coco_zip_dir, zip_file)
    #         extract_zip(zip_path, extract_to)
    
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