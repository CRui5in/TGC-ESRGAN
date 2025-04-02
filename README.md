# TGSR: Text-Guided Super-Resolution

TGSR是一个基于文本引导的超分辨率模型框架，基于BasicSR框架开发，参考了Real-ESRGAN的结构。通过融合文本条件信息，能够生成更符合文本描述的超分辨率图像。

## 特性

- 文本引导的超分辨率
- 基于BasicSR的灵活训练框架
- GradCAM注意力可视化
- TensorBoard实时监控
- 支持多GPU训练(DP/DDP)
- 支持断点续训

## 安装

### 依赖

- Python >= 3.7
- PyTorch >= 1.7
- BasicSR >= 1.4.0
- CLIP (OpenAI)

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/CRui5in/TGSR.git
cd TGSR

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

## 数据准备

TGSR需要成对的低分辨率/高分辨率图像以及对应的文本描述。文本描述需要以下格式保存在文本文件中：

```
image1.png    高分辨率的猫图像
image2.png    清晰的自然风景
```

## 使用方法

### 训练

使用以下命令启动训练：

```bash
# 使用默认配置文件训练
sh scripts/start_training.sh

# 指定配置文件训练
sh scripts/start_training.sh --opt options/my_config.yml

# 使用特定数量的GPU
sh scripts/start_training.sh --gpu 2

# 启用内存优化
sh scripts/start_training.sh --memory_opt
```

### 恢复训练

从检查点恢复训练：

```bash
sh scripts/resume_train.sh --checkpoint experiments/train_TGSR_x4/models/50000_G.pth
```

### 停止训练

```bash
sh scripts/stop_train.sh
```

### 清理日志

```bash
sh scripts/clean_logs.sh
```

## 可视化

训练过程中会自动启动TensorBoard，可以通过访问 http://localhost:6006 查看训练进度、损失曲线和生成的图像。可视化内容包括：

- 训练/验证损失
- 生成的SR图像与GT对比
- 注意力热力图
- PSNR/SSIM等评估指标

## 配置文件说明

配置文件位于`options/`目录，主要包括以下几个部分：

- 数据集配置
- 网络配置
- 训练参数
- 验证参数
- 文本编码器配置

详细配置请参考`options/train_tgsr_x4.yml`中的注释。

## 引用

如果您使用了TGSR框架，请引用以下项目：

```
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```

## 许可证

本项目遵循MIT许可证 