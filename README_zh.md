# 工业级滑块验证码识别系统

<div align="center">

[English](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/README.md) | [简体中文](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/README_zh.md)

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于深度学习的高精度滑块验证码识别解决方案，采用改进的CenterNet架构，在真实验证码数据集上达到80%准确率。

</div>

## 📋 项目概述

本项目是一个工业级的滑块验证码识别系统，通过深度学习方法解决传统模板匹配算法的准确率瓶颈。系统基于**30多万张**合成验证码图片训练，采用轻量级CNN架构，在保证高精度的同时实现了实时推理能力。

### 🎯 核心特性

- **高精度识别**：真实验证码7px误差准确率达80%，5px误差准确率达73%
- **实时推理**：GPU推理 1.30ms（RTX 5090），CPU推理 5.21ms（AMD Ryzen 9 9950X），支持实时应用
- **轻量架构**：仅3.5M参数，模型文件约36MB
- **工业级设计**：完整的数据生成、训练、评估管线
- **亚像素精度**：采用CenterNet offset机制实现亚像素级定位

### 🖼️ 识别效果展示

#### 真实验证码数据集识别效果

![真实数据集识别效果](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/results/best_model_evaluation/real_captchas/visualizations/sample_0031.png?raw=true)

*图示：在某网站真实验证码上的识别效果，红色圆圈标记缺口位置，蓝色圆圈标记滑块位置*

#### 测试集识别效果

![测试集识别效果](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/results/best_model_evaluation/test_dataset/visualizations/sample_0014.png?raw=true)

*图示：在合成测试集上的识别效果，展示了模型对不同形状和光照条件的适应能力*

## 🚀 快速开始

### 环境要求

```bash
# Python 3.8+
pip install -r requirements.txt
```

### 安装方式

#### 可直接使用 pip 安装

```bash
pip install Sider_CAPTCHA_Solver
```

### 基础使用

使用 pip 安装后，可以直接导入并使用：

#### 1. 基础预测 - 获取滑动距离

```python
from sider_captcha_solver import CaptchaPredictor

# 初始化预测器
predictor = CaptchaPredictor(
    model_path='best',  # 使用内置最佳模型，或指定自定义模型路径
    device='auto'       # 自动选择 GPU/CPU
)

# 预测单张图片
result = predictor.predict('path/to/captcha.png')

# 获取滑动距离
if result['slider_x'] and result['gap_x']:
    sliding_distance = result['gap_x'] - result['slider_x']
    print(f"滑动距离: {sliding_distance:.2f} px")
    print(f"缺口位置: ({result['gap_x']:.2f}, {result['gap_y']:.2f})")
    print(f"滑块位置: ({result['slider_x']:.2f}, {result['slider_y']:.2f})")
else:
    print("检测失败")
```

#### 2. 批量处理 - 处理文件夹中的多张图片

```python
from sider_captcha_solver import CaptchaPredictor
import glob
import os

# 初始化预测器
predictor = CaptchaPredictor(model_path='best', device='auto')

# 批量处理验证码
captcha_folder = 'path/to/captchas'

for img_path in glob.glob(os.path.join(captcha_folder, '*.png')):
    result = predictor.predict(img_path)

    if result['slider_x'] and result['gap_x']:
        distance = result['gap_x'] - result['slider_x']
        confidence = (result['slider_confidence'] + result['gap_confidence']) / 2
        print(f"{os.path.basename(img_path)}: 滑动 {distance:.1f} px (置信度: {confidence:.3f})")
    else:
        print(f"{os.path.basename(img_path)}: 检测失败")
```

#### 3. 可视化与调试

```python
from sider_captcha_solver import CaptchaPredictor
import matplotlib.pyplot as plt

# 初始化预测器
predictor = CaptchaPredictor(model_path='best', device='auto')

# 测试图片路径
test_image = 'path/to/captcha.png'

# 生成并保存预测可视化
predictor.visualize_prediction(
    test_image,
    save_path='prediction_result.png',  # 保存路径
    show=True                           # 显示窗口
)

# 生成热力图可视化（查看模型内部激活）
predictor.visualize_heatmaps(
    test_image,
    save_path='heatmap_result.png',    # 保存 4 宫格热力图
    show=True
)

# 对比不同阈值的效果
thresholds = [0.0, 0.1, 0.3, 0.5]
fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 4))

for idx, threshold in enumerate(thresholds):
    # 使用不同阈值创建预测器
    pred = CaptchaPredictor(model_path='best', hm_threshold=threshold)
    result = pred.predict(test_image)

    # 可视化到子图
    ax = axes[idx]
    img = plt.imread(test_image)
    ax.imshow(img)
    ax.set_title(f'Threshold={threshold}')

    if result['slider_x'] and result['gap_x']:
        ax.plot(result['slider_x'], result['slider_y'], 'bo', markersize=10)
        ax.plot(result['gap_x'], result['gap_y'], 'ro', markersize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('threshold_comparison.png')
plt.show()
```

#### 4. 完整的生产环境示例

```python
from sider_captcha_solver import CaptchaPredictor
import logging
import time
from typing import Optional, Dict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaptchaSolver:
    """生产环境的验证码求解器封装"""

    def __init__(self, model_path: str = 'best', device: str = 'auto'):
        self.predictor = CaptchaPredictor(
            model_path=model_path,
            device=device,
            hm_threshold=0.1  # 平衡准确率和召回率
        )
        logger.info(f"验证码求解器初始化完成，设备: {device}")

    def solve(self, image_path: str, max_retries: int = 3) -> Optional[Dict]:
        """
        求解验证码，支持重试机制

        Args:
            image_path: 验证码图片路径
            max_retries: 最大重试次数

        Returns:
            包含滑动距离和置信度的字典，失败返回 None
        """
        for attempt in range(max_retries):
            try:
                # 记录开始时间
                start_time = time.time()

                # 执行预测
                result = self.predictor.predict(image_path)

                # 计算耗时
                elapsed_time = (time.time() - start_time) * 1000

                # 检查结果有效性
                if result['slider_x'] and result['gap_x']:
                    sliding_distance = result['gap_x'] - result['slider_x']
                    confidence = (result['slider_confidence'] + result['gap_confidence']) / 2

                    logger.info(f"求解成功: 距离={sliding_distance:.1f}px, "
                              f"置信度={confidence:.3f}, 耗时={elapsed_time:.1f}ms")

                    return {
                        'success': True,
                        'sliding_distance': sliding_distance,
                        'confidence': confidence,
                        'elapsed_ms': elapsed_time,
                        'details': result
                    }
                else:
                    logger.warning(f"第 {attempt + 1} 次尝试失败：未检测到有效结果")

            except Exception as e:
                logger.error(f"第 {attempt + 1} 次尝试出错: {str(e)}")

            # 如果不是最后一次尝试，稍作延迟
            if attempt < max_retries - 1:
                time.sleep(0.1)

        logger.error(f"求解失败：已达到最大重试次数 {max_retries}")
        return None

# 使用示例
if __name__ == "__main__":
    solver = CaptchaSolver()

    # 求解单个验证码
    result = solver.solve('path/to/captcha.png')

    if result and result['success']:
        print(f"滑动距离: {result['sliding_distance']:.1f} px")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"处理时间: {result['elapsed_ms']:.1f} ms")
    else:
        print("验证码求解失败")
```

### 进阶功能

#### 1. 自定义模型和配置

```python
from sider_captcha_solver import CaptchaPredictor
import torch

# 使用自己训练的模型
custom_predictor = CaptchaPredictor(
    model_path='path/to/your_trained_model.pth',
    device='cuda:0',    # 指定 GPU
    hm_threshold=0.15   # 根据模型特性调整
)

# 检查模型信息
if torch.cuda.is_available():
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# 预测
result = custom_predictor.predict('captcha.png')
```

#### 2. 性能基准测试

```python
from sider_captcha_solver import CaptchaPredictor
import time
import numpy as np

# 初始化预测器
predictor = CaptchaPredictor(model_path='best', device='auto')

# 测试图片列表
test_images = ['captcha1.png', 'captcha2.png', 'captcha3.png']

# 预热（首次推理较慢）
_ = predictor.predict(test_images[0])

# 性能测试
times = []
for _ in range(10):  # 每张图片测试 10 次
    for img_path in test_images:
        start = time.time()
        result = predictor.predict(img_path)
        elapsed = (time.time() - start) * 1000  # 转换为毫秒
        times.append(elapsed)

# 统计结果
print(f"平均推理时间: {np.mean(times):.2f} ms")
print(f"最快: {np.min(times):.2f} ms")
print(f"最慢: {np.max(times):.2f} ms")
print(f"标准差: {np.std(times):.2f} ms")
print(f"FPS: {1000 / np.mean(times):.1f}")
```

## 📊 数据生成流程

### 1. 数据采集

从Pixabay下载10个类别的高质量图片作为背景：Minecraft、Pixel Food、Block Public Square、Block Illustration、Backgrounds、Buildings、Nature、Anime Cityscape、Abstract Geometric Art等。每个类别最多200张，共计约2千张原始图片。

### 2. 验证码生成逻辑

```
原始图片(2千余张) → Resize(320×160) → 挖洞生成
                                        ↓
                              11种形状 × 3种尺寸 × 4个位置
                                        ↓
                              每张原图生成132个验证码
                                        ↓
                              总计生成354,024张训练图片
```

**拼图形状设计**：

- 5种普通拼图形状（四边凹凸平组合）
- 6种特殊形状（圆形、正方形、三角形、六边形、五边形、五角星）

**随机参数**：

- 拼图尺寸：40-70像素（3种随机尺寸）
- 位置分布：x轴在滑块宽度+10px之外，避免重叠
- 光照效果：随机添加光照变化增强鲁棒性

### 3. 数据集划分

- 训练集：90%（基于原图划分，避免数据泄露）
- 测试集：10%（测试集 1）
- 真实验证码测试集：100张网易易盾验证码（测试集 2）

## 🏗️ 网络架构

### 模型结构

```
输入 (3×160×320)
    │
    ├─ Stem Conv (3×3, stride=2) ──────→ 32×80×160
    │
    ├─ ResBlock Stage-1 (×2, stride=2) ─→ 64×40×80
    │
    ├─ ResBlock Stage-2 (×2, stride=2) ─→ 128×20×40
    │
    ├─ ResBlock Stage-3 (×2, stride=2) ─→ 256×10×20
    │
    ├─ Neck (1×1 Conv) ─────────────────→ 128×10×20
    │
    ├─ UpConv-1 (3×3, stride=2) ────────→ 64×20×40
    │
    ├─ UpConv-2 (3×3, stride=2) ────────→ 64×40×80
    │
    └─┬─ Gap Detection Head ────┐
        │   ├─ Heatmap (1×40×80)   │
        │   └─ Offset (2×40×80)    │
        │                              │
        └─ Piece Detection Head ───┤
             ├─ Heatmap (1×40×80)   │
             └─ Offset (2×40×80)    │
```

### 关键设计

- **骨干网络**：ResNet18-Lite，删除全局池化层和全连接层
- **检测头**：双分支CenterNet设计，分别检测缺口和滑块中心
- **损失函数**：Focal Loss（热力图）+ L1 Loss（偏移回归）
- **下采样率**：4倍，输出分辨率80×40
- **激活函数**：ReLU（除输出层外）
- **归一化**：BatchNorm

### 模型参数

| 组件              | 参数量       | 说明            |
| --------------- | --------- | ------------- |
| Backbone        | ~3.0M     | ResNet18-Lite |
| Neck + UpConv   | ~0.3M     | 特征融合与上采样      |
| Detection Heads | ~0.2M     | 双分支检测头        |
| **总计**          | **~3.5M** | FP32模型约36MB   |

## 📈 性能指标

### 准确率（基于滑动距离误差）

| 数据集     | 5px阈值   | 7px阈值   | 最佳Epoch |
| ------- | ------- | ------- | ------- |
| 测试集（生成） | 99.4%   | 99.4%   | 16      |
| 真实验证码   | **73%** | **80%** | 15/16   |

### 推理性能

| 硬件                | 推理时间   | FPS | 批处理（×32） |
| ----------------- | ------ | --- | -------- |
| RTX 5090          | 1.30ms | 771 | 11.31ms  |
| AMD Ryzen 9 9950X | 5.21ms | 192 | 144.89ms |

### 平均绝对误差（MAE）

- 测试集：滑块 0.30px，缺口 1.14px
- 真实验证码：滑块 2.84px，缺口 9.98px

## 🛠️ 主要功能

### 1. 数据生成

- 自动下载Pixabay图片
- 批量生成滑块验证码
- 支持多种拼图形状

### 2. 模型训练

- 自动学习率调度
- 训练过程可视化

### 3. 推理部署

- 支持批量预测
- REST API接口
- 支持热图可视化

### 4. 评估分析

- 训练曲线分析

## ⚠️ 免责声明

**本项目仅供学习和研究使用，不得用于任何商业或非法用途。**

1. 本项目旨在促进计算机视觉和深度学习技术的学术研究
2. 使用者需遵守相关法律法规，不得将本项目用于绕过网站安全机制
3. 因使用本项目产生的任何法律责任由使用者自行承担
4. 请勿将本项目用于任何可能损害他人利益的行为

## 📁 项目结构

```
Sider_CAPTCHA_Solver/
│
├── configs/                       # 配置文件
│   └── config.yaml               # 项目配置
│
├── data/                          # 数据目录
│   ├── captchas/                  # 生成的验证码（354,024张）
│   │   └── Pic*.png              # 格式：Pic{XXXX}_Bgx{X}Bgy{Y}_Sdx{X}Sdy{Y}_{hash}.png
│   ├── raw_images/                # 原始图片（2000张）
│   ├── real_captchas/             # 真实验证码测试集
│   │   └── annotated/             # 标注数据（100张）
│   ├── annotations.json           # 训练集标注文件
│   ├── test_annotations.json      # 测试集标注文件
│   ├── generation_stats.json      # 生成统计信息
│   └── dataset_split_stats.json   # 数据集划分统计
│
├── logs/                          # 日志文件
│   ├── training_accuracy_curves_all.png    # 训练准确率曲线
│   ├── accuracy_comparison.png             # 测试集vs真实数据对比
│   ├── training_analysis_report.txt        # 训练分析报告
│   ├── training_accuracy_results.csv       # 准确率CSV数据
│   ├── training_accuracy_results.json      # 准确率JSON数据
│   ├── evaluation_*.log                    # 评估日志
│   ├── training_log.txt                    # 训练日志
│   └── benchmark_results_*.json            # 性能基准测试结果
│
├── results/                       # 评估结果
│   └── best_model_evaluation/     # 最佳模型评估
│       ├── test_dataset/          # 测试集结果
│       │   ├── evaluation_results.json     # 评估指标
│       │   └── visualizations/             # 可视化结果（100张）
│       ├── real_captchas/         # 真实验证码结果
│       │   ├── evaluation_results.json     # 评估指标
│       │   └── visualizations/             # 可视化结果（50张）
│       └── summary_report.json    # 汇总报告
│
├── scripts/                       # 核心脚本
│   ├── annotation/                # 标注工具
│   │   ├── annotate_captchas_matplotlib.py  # Matplotlib标注界面
│   │   └── annotate_captchas_web.py         # Web标注界面
│   │
│   ├── data_generation/           # 数据生成脚本
│   │   ├── geometry_generator.py  # 几何形状生成器
│   │   └── puzzle_background_generator.py   # 拼图背景生成器
│   │
│   ├── training/                  # 训练相关
│   │   ├── train.py              # 主训练脚本
│   │   ├── dataset.py            # PyTorch数据集类
│   │   └── analyze_training.py   # 训练分析工具
│   │
│   ├── inference/                 # 推理相关
│   │   └── predict.py            # 预测接口（CaptchaPredictor类）
│   │
│   ├── evaluation/                # 评估脚本
│   │   └── evaluate_model.py      # 综合评估工具（支持多种模式）
│   │
│   ├── download_images.py         # Pixabay图片下载脚本
│   ├── generate_captchas.py       # 批量验证码生成脚本
│   └── split_dataset.py           # 数据集划分脚本
│
├── src/                          # 源代码
│   ├── __init__.py
│   │
│   ├── checkpoints/               # 模型权重文件
│   │   ├── best_model.pth         # 最佳模型（最高准确率）
│   │   ├── checkpoint_epoch_0001.pth ~ checkpoint_epoch_0020.pth  # 各epoch检查点
│   │   ├── latest_checkpoint.pth  # 最新检查点
│   │   ├── training_log_*.txt     # 训练日志
│   │   └── logs/                  # TensorBoard日志
│   │       └── events.out.tfevents.*
│   │
│   ├── captcha_generator/         # 验证码生成模块
│   │   ├── __init__.py
│   │   ├── batch_generator.py    # 批量生成器
│   │   ├── lighting_effects.py   # 光照效果
│   │   ├── simple_puzzle_generator.py  # 拼图生成器
│   │   └── slider_effects.py     # 滑块效果
│   │
│   ├── data_collection/           # 数据采集模块
│   │   ├── __init__.py
│   │   └── pixabay_downloader.py # Pixabay下载器
│   │
│   ├── models/                   # 模型定义
│   │   ├── __init__.py
│   │   ├── captcha_solver.py     # CaptchaSolver主模型
│   │   ├── centernet_heads.py    # CenterNet检测头
│   │   ├── losses.py             # 损失函数（Focal Loss + L1）
│   │   └── resnet18_lite.py      # ResNet18-Lite骨干网络
│   │
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       └── logger.py             # 日志工具
│
├── tests/                        # 测试脚本
│   ├── benchmark_inference.py     # 推理性能基准测试
│   ├── merge_real_captchas.py     # 真实验证码合并工具
│   ├── test_all_puzzle_shapes.py  # 全部拼图形状测试
│   ├── test_captcha_generation.py # 验证码生成测试
│   ├── test_darkness_levels.py    # 亮度级别测试
│   ├── test_distance_error_visualization.py  # 距离误差可视化
│   ├── test_generate_captchas.py  # 生成功能测试
│   ├── test_model.py             # 模型单元测试
│   ├── test_model_architecture.py # 模型架构测试
│   ├── test_real_captchas.py     # 真实验证码测试
│   └── test_slider_effects.py    # 滑块效果测试
│
├── outputs/                      # 测试输出文件
│   └── *.png                     # 各种测试结果图片
│
├── api_example.py                # API使用示例
├── requirements.txt              # 依赖包列表
├── README.md                     # 英文说明文档
├── README_zh.md                  # 中文说明文档
└── CLAUDE.md                     # 项目技术规范
```

## 🔧 技术栈

- **深度学习框架**：PyTorch 2.0+
- **图像处理**：OpenCV, Pillow
- **数据处理**：NumPy, Pandas
- **可视化**：Matplotlib, Seaborn
- **Web框架**：FastAPI
- **其他**：tqdm, requests, psutil

## 📞 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。

---

<div align="center">
<i>本项目遵循MIT协议，仅供学习研究使用</i>
</div>