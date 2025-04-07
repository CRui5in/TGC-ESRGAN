# TGSR测试模块

本测试模块用于评估TGSR模型在提升开放词汇目标检测能力上的效果。

## 功能特点

- 支持使用COCO2017验证集测试超分辨率质量
- 支持使用GroundingDINO测试目标检测效果
- 提供两种提示词生成方式：基于类别和基于图像描述
- 支持注意力热力图可视化，包含颜色图例
- 自动计算PSNR、SSIM和mAP指标
- 支持图像退化处理

## 文件结构

- `config.py`: 测试配置类
- `dataset.py`: 数据集加载和提示词生成
- `evaluator.py`: 超分辨率和目标检测评估器
- `test_tgsr.py`: 主测试脚本
- `run_test.sh`: 运行脚本

## 安装依赖

在运行测试前，确保安装以下依赖：

```bash
pip install torch torchvision tqdm matplotlib opencv-python pillow
# 如果需要目标检测功能，请安装GroundingDINO
pip install groundingdino
```

## 使用方法

### 1. 准备数据和模型

- 确保COCO2017数据集已下载至指定目录
- 确保TGSR模型权重已准备好
- 如果需要目标检测，请准备GroundingDINO模型和配置文件

### 2. 修改配置

编辑`run_test.sh`中的路径配置：

```bash
COCO_PATH="/path/to/COCO2017"  # COCO数据集路径
MODEL_PATH="/path/to/model.pth"  # TGSR模型路径
OUTPUT_DIR="/path/to/output"  # 输出目录
DETECTION_MODEL_PATH="/path/to/groundingdino.pth"  # GroundingDINO模型路径
DETECTION_CONFIG_PATH="/path/to/config.py"  # GroundingDINO配置路径
```

### 3. 运行测试

```bash
cd TGSR/test
bash run_test.sh
```

或者单独运行特定的测试：

```bash
python test_tgsr.py --coco_path /path/to/COCO2017 --model_path /path/to/model.pth --prompt_type category
```

### 4. 参数说明

主要参数说明：

- `--coco_path`: COCO数据集路径
- `--model_path`: TGSR模型路径
- `--output_dir`: 输出目录
- `--prompt_type`: 提示词类型，可选'category'或'caption'
- `--apply_degradation`: 是否应用图像退化
- `--use_ema`: 是否使用EMA模型
- `--detection_model_path`: GroundingDINO模型路径
- `--detection_config_path`: GroundingDINO配置文件路径
- `--box_threshold`: 检测框置信度阈值
- `--text_threshold`: 文本匹配阈值

## 输出结果

测试完成后，将在输出目录生成以下文件和目录：

- `sr_results/`: 超分辨率结果图像
- `detection_results/`: 目标检测结果图像
- `attention_maps/`: 注意力热力图
- `metrics.json`: 评估指标结果
- `test_xxxx.log`: 测试日志

## 注意事项

1. 确保图像退化参数与训练时一致
2. GroundingDINO模型需要单独下载和配置
3. 如果没有提供GroundingDINO模型，仍然可以测试超分辨率效果
4. 注意力热力图带有颜色图例，可直观观察模型关注区域

# TGSR 测试指南（Windows环境）

## 代码优化摘要

为了确保TGSR模型在Windows环境下能够顺利运行，我们对测试代码进行了以下优化：

1. **路径处理优化**：
   - 使用 `os.path.normpath()` 规范化所有文件路径
   - 根据操作系统自动调整路径分隔符（`/` 或 `\`）
   - 添加文件路径存在性检查，并提供更友好的错误提示

2. **文件操作增强**：
   - 添加 `encoding='utf-8'` 参数处理中文文件名和内容
   - 为所有文件操作添加异常处理
   - 优化图像保存和加载逻辑

3. **多进程优化**：
   - 在Windows环境下限制数据加载器的工作线程数量，避免多进程问题
   - 默认将Windows环境下的 `num_workers` 设置为0

4. **GPU内存管理**：
   - 添加GPU内存释放与错误恢复机制
   - 优化Torch Tensor到Numpy数组的转换过程

5. **日志系统优化**：
   - 添加详细的日志记录，包括操作系统信息、路径信息等
   - 将错误和警告信息统一记录到日志系统中

6. **可视化功能增强**：
   - 为Windows环境设置Matplotlib的非交互式后端 (`Agg`)
   - 改进注意力热力图和目标检测结果的可视化方法

7. **AI模型集成**：
   - 优化GroundingDINO模型的加载和使用方式，更好地支持Transformers库
   - 添加模型可用性检查，提供更友好的错误反馈

8. **代码执行方式**：
   - 添加直接运行Python脚本的方法，无需使用shell脚本
   - 提供在Python代码中调用测试函数的示例

本指南将帮助您在Windows环境下正确运行TGSR（Text-Guided Super-Resolution）模型的测试脚本。

## 环境准备

1. 确保已安装 PyTorch 和其他必要依赖
2. 确保已下载并正确放置以下文件：
   - TGSR模型权重文件（默认位置：`TGSR/experiments/models/net_g_250000.pth`）
   - 本地CLIP模型（默认位置：`clip-vit-base-patch32`）
   - COCO2017数据集（如需进行完整测试）

## 目录结构

```
项目根目录
├── TGSR
│   ├── experiments
│   │   └── models
│   │       └── net_g_250000.pth (模型权重文件)
│   └── test
│       ├── config.py
│       ├── dataset.py
│       ├── evaluator.py
│       └── test_tgsr.py
├── clip-vit-base-patch32 (CLIP模型目录)
└── COCO2017 (可选，用于完整测试)
    ├── annotations
    │   ├── instances_val2017.json
    │   └── captions_val2017.json
    └── val2017
        └── (验证集图像)
```

## 运行测试脚本

### 方法1：直接运行测试（推荐）

打开命令提示符或PowerShell，切换到项目根目录，然后执行：

```
python -m TGSR.test.test_tgsr --model_path=TGSR/experiments/models/net_g_250000.pth --clip_model_path=clip-vit-base-patch32
```

如果需要指定COCO数据集路径：

```
python -m TGSR.test.test_tgsr --coco_path=COCO2017路径 --model_path=TGSR/experiments/models/net_g_250000.pth --clip_model_path=clip-vit-base-patch32
```

### 方法2：在Python代码中运行

```python
from TGSR.test.test_tgsr import run_tgsr_test

# 使用类别名称作为提示词运行测试
run_tgsr_test(prompt_type='category', apply_degradation=False)

# 使用图像描述作为提示词运行测试
# run_tgsr_test(prompt_type='caption', apply_degradation=False)
```

## 参数说明

测试脚本支持以下命令行参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --coco_path | COCO数据集路径 | 当前目录下的COCO2017文件夹 |
| --model_path | TGSR模型权重路径 | TGSR/experiments/models/net_g_250000.pth |
| --clip_model_path | 本地CLIP模型路径 | clip-vit-base-patch32 |
| --output_dir | 结果输出目录 | results |
| --prompt_type | 提示词类型：category或caption | category |
| --apply_degradation | 是否应用图像退化处理 | 不启用 |
| --use_ema | 是否使用EMA模型 | 不启用 |
| --detection_model_name | 目标检测模型名称 | IDEA-Research/grounding-dino-base |
| --box_threshold | 检测框置信度阈值 | 0.4 |
| --text_threshold | 文本匹配阈值 | 0.3 |

## 输出结果

测试完成后，将在指定的输出目录（默认为`results`）生成以下内容：

1. `sr_results`：超分辨率结果图像
2. `detection_results`：目标检测结果图像
3. `attention_maps`：注意力热力图（如启用）
4. `metrics.json`：评估指标结果
5. `test_<时间戳>.log`：测试日志

## 常见问题

### 1. 找不到模块

确保您在项目根目录运行脚本，或者已将项目路径添加到`PYTHONPATH`环境变量中。

### 2. CUDA内存不足

如果遇到CUDA内存不足的问题，可尝试以下解决方法：
- 减小batch_size（目前默认为1）
- 在CPU上运行测试（添加参数`--device=cpu`）

### 3. 找不到模型文件

确保模型文件放置在正确位置，或使用`--model_path`参数明确指定模型路径。

### 4. 图像显示问题

如果遇到图像无法显示或保存的问题，请确保安装了PIL库：
```
pip install pillow
``` 