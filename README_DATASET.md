# 文本引导超分辨率 (TGSR) 数据集生成与训练指南

本文档提供了如何从COCO2017数据集生成文本引导超分辨率训练数据集，以及如何使用这些数据训练TGSR模型的说明。

## 数据集生成

我们使用COCO2017数据集作为基础，提取图像、说明文字和对象标签来创建文本引导超分辨率训练数据集。
在训练过程中，我们采用RealESRGAN风格的降质方法动态生成低分辨率图像，而不是预先生成它们。

### 前提条件

1. 下载COCO2017数据集，包括：
   - train2017.zip (图像)
   - annotations_trainval2017.zip (标注)

2. 将数据集解压到指定目录，例如：`/root/autodl-tmp/COCO2017`，确保以下路径结构：
   ```
   COCO2017/
   ├── annotations/
   │   ├── instances_train2017.json
   │   └── captions_train2017.json
   └── train2017/
       └── [图像文件]
   ```

### 生成数据集

从命令行直接运行数据集生成脚本：

```bash
python datasets_coco.py \
    --coco_dir /path/to/COCO2017 \
    --output_dir /path/to/output \
    --jpeg_quality 100 \
    --max_workers 8
```

参数说明：
- `--coco_dir`: COCO2017数据集根目录路径
- `--output_dir`: 输出数据集目录路径
- `--jpeg_quality`: JPEG压缩质量 (1-100)
- `--resize_scale`: 图像缩放比例 (可选，如0.5表示缩小到50%)
- `--max_workers`: 最大工作线程数
- `--max_images`: 每个划分最大处理的图像数量 (可选，用于快速测试)

### 使用脚本一键生成数据集并训练

我们提供了一个脚本，可以自动生成数据集并启动训练：

```bash
# 先给脚本添加执行权限
chmod +x scripts/start_training.sh

# 运行脚本
./scripts/start_training.sh
```

这个脚本会：
1. 检查是否已存在生成的数据集，如果不存在则从COCO2017生成
2. 启动TensorBoard用于监控训练过程
3. 开始训练模型

## 训练模型

训练使用RealESRGAN风格的动态退化生成低分辨率图像，对于文本区域使用较轻的退化以保留文本特征。

### 配置说明

训练配置文件位于 `options/train_tgsr_x4.yml`，主要参数包括：

- **数据集参数**：指定HR图像路径和文本提示JSON文件
- **退化参数**：控制生成LR图像的退化强度、噪声、JPEG压缩等
- **训练参数**：批量大小、学习率、损失函数等
- **验证参数**：验证频率、保存图像等
- **测试参数**：完整测试集评估频率

### 恢复训练

如果训练中断，可以使用以下命令恢复训练：

```bash
python tgsr/train.py --opt options/train_tgsr_x4.yml --auto_resume --memory_opt
```

## 输出目录结构

训练过程中，模型会生成以下输出：

```
experiments/
├── models/           # 模型检查点
├── training_states/  # 训练状态（用于恢复训练）
└── visualization/    # 可视化结果（验证/测试图像）

logs/                 # 训练日志
```

## TensorBoard监控

训练启动后，可以通过以下地址访问TensorBoard：

```
http://localhost:6006
```

TensorBoard中可以查看：
- 训练损失曲线
- 验证指标（PSNR/SSIM）
- 训练/验证/测试图像样本
- 学习率变化 