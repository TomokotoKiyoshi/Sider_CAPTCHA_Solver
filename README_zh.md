# 工业级滑块验证码识别系统

<div align="center">

[English](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/README.md) | [简体中文](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/README_zh.md)

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/badge/PyPI-v1.0.3-blue.svg)](https://pypi.org/project/sider-captcha-solver/)
[![GitHub version](https://img.shields.io/badge/GitHub-v1.1.0-blue.svg)](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver)

一个基于深度学习的高精度滑块验证码识别解决方案，采用**Lite-HRNet-18-FPN**架构和改进的CenterNet检测头，在真实验证码数据集上达到90%+准确率。

**最新版本**: v1.1.0

</div>

## 🆕 更新日志

### v1.1.0 (2025-08-17) - 最新版本

- 🏗️ **全新架构升级**：
  - 采用**Lite-HRNet-18-FPN**骨干网络，提升特征提取能力
  - 双分支检测头设计，分别处理缺口和滑块检测
  - 4通道输入支持（RGB + Padding Mask），有效防止padding区域误检，支持多种图片大小输入
- 🎯 **精度大幅提升**：
  - 真实验证码准确率提升至**90%+**（7px误差阈值）
  - MAE降至约1px，亚像素精度定位
- ⚡ **性能优化**：
  - 模型参数仅3.3M，FP32模型约12.53MB
  - GPU推理速度<3ms（RTX 5090）
  - CPU推理12-14ms（AMD 9950X）
- 🛡️ **增强的抗混淆特性**：
  - 加入拼图中央挖空混淆
  - 修改了拼图混淆的生成的概率逻辑，改为生成固定样本数

### v1.0.3 (2025-07-27)

- 🛡️ **增强的抗混淆特性**：
  - 缺口旋转（0.5-1.8°随机旋转，50%概率）
  - 滑块柏林噪声（40-80%强度，50%概率）
  - 混淆缺口（±10-30°旋转，60%概率）
  - 缺口高光效果（30%概率）
- 📊 **模型性能提升**：
  - 真实验证码**85%+准确率**，抗干扰能力增强
  - 更好的对抗样本防御能力
  - 复杂场景下更稳定的预测

### v1.0.2 (2025-07-21) - 初始发布

- 🚀 首次公开发布
- 📦 基础滑块验证码识别
- 🎯 真实验证码7px误差80%准确率
- 💡 支持11种拼图形状（5种常规+6种特殊）
- ⚡ 快速推理：GPU 2ms，CPU 5ms

## 📑 目录

- [📋 项目概述](#-项目概述)
  - [🎯 核心特性](#-核心特性)
  - [🖼️ 识别效果展示](#️-识别效果展示)
- [🚀 快速开始](#-快速开始)
  - [环境要求](#环境要求)
  - [安装方式](#安装方式)
  - [基础使用](#基础使用)
- [📊 数据生成流程](#-数据生成流程)
- [🏗️ 网络架构](#️-网络架构)
- [📈 性能指标](#-性能指标)
- [🛠️ 主要功能](#️-主要功能)
- [⚠️ 免责声明](#️-免责声明)
- [📁 项目结构](#-项目结构)
- [🔧 技术栈](#-技术栈)
- [📞 联系方式](#-联系方式)

## 📋 项目概述

本项目是一个工业级的滑块验证码识别系统，通过深度学习方法解决传统模板匹配算法的准确率瓶颈。系统基于**30多万张**合成验证码图片训练，采用轻量级CNN架构，在保证高精度的同时实现了实时推理能力。

### 🎯 核心特性

- **高精度识别**：真实验证码7px误差准确率达**90%+**（v1.1.0），MAE约1px
- **增强抗混淆能力**：支持缺口旋转、滑块柏林噪声、混淆缺口、缺口高光效果
- **实时推理**：GPU推理 <3ms（RTX 3050），CPU推理 12-14ms（i5-1240P），支持实时应用
- **轻量架构**：仅3.5M参数，模型文件约14MB（FP32）
- **工业级设计**：完整的数据生成、训练、评估管线
- **亚像素精度**：采用CenterNet offset机制实现亚像素级定位

### 🖼️ 识别效果展示

#### 真实验证码数据集识别效果

![真实数据集识别效果](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/results/1.0.3/real_captchas/visualizations/sample_0000.png)

*图示：在某网站真实验证码上的识别效果，红色圆圈标记缺口位置，蓝色圆圈标记滑块位置*

#### 测试集识别效果

![测试集识别效果](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/results/1.0.3/test_dataset/visualizations/sample_0025.png)

## 🚀 快速开始

### 环境要求

```bash
# Python 3.8+
pip install -r requirements.txt
```

### 安装方式

#### 可直接使用 pip 安装

```bash
pip install sider-captcha-solver  # 安装 v1.0.3 版本（v1.1.0即将发布）
```

### 基础使用

使用 pip 安装后，可以直接导入并使用：（目前最新版本仍在完善，尚未上传Pypi）

#### 1. 获取滑动距离和详细信息

```python
from sider_captcha_solver import solve

# 获取详细信息
result = solve('data/captchas/Pic0001_Bgx84Bgy206_Sdx25Sdy206_1a85adc2.png', detailed=True)
print(f"Sliding distance: {result['distance']:.2f} px")
print(f"Gap position: {result['gap']}")
print(f"Slider position: {result['slider']}")
print(f"Confidence: {result['confidence']:.2f}")
```

#### 2. 可视化预测结果

```python
from sider_captcha_solver import visualize

# 可视化预测结果（自动预测并标注）
visualize('data/captchas/Pic0001_Bgx84Bgy206_Sdx25Sdy206_1a85adc2.png', save_path='result.png', show=True)
```

## 📊 数据生成流程

### 1. 数据采集

从Pixabay下载10个类别的高质量图片作为背景：Minecraft、Pixel Food、Block Public Square、Block Illustration、Backgrounds、Buildings、Nature、Anime Cityscape、Abstract Geometric Art等。每个类别最多200张，共计约2千张原始图片。
外加计算机合成的数百张混淆图片

### 2. 验证码生成逻辑

```
原始图片(2千余张) → 随机Resize(280×140 ~ 480×240) → 添加抗混淆特性 → 挖洞生成
                                                              ↓
                                               16种形状 × 3种尺寸 × 多个随机位置
                                                              ↓
                                                    每张原图生成100+个验证码
                                                              ↓
                                                    总计生成约26万张训练图片
```

**拼图形状设计**：

- 10种普通拼图形状（四边凹凸平组合的81种中选取）
- 6种特殊形状（圆形、正方形、三角形、六边形、五边形、五角星）

**抗混淆增强特性**（v1.1.0新增）：

- **缺口旋转**（10%概率）：0.5-1.8°随机旋转
- **柏林噪声**（20%概率）：滑块表面40-80%强度噪声纹理
- **混淆缺口**（30%概率）：±10-30°旋转的干扰缺口
- **缺口高光**（10%概率）：传统阴影效果改为高光效果

**随机参数**：

- 图片尺寸：280×140 ~ 480×240像素随机
- 拼图尺寸：30-50像素（3种随机尺寸）
- 位置分布：滑块x∈[15,35]，缺口x∈[65,305]，确保不重叠
- 光照效果：随机添加光照变化增强鲁棒性

### 3. 数据集划分

- 训练集：80%（基于原图划分，避免数据泄露）
- 测试集：10%
- 验证集：10%

## 🏗️ 网络架构

### 模型结构 (v1.1.0 - Lite-HRNet-18-FPN)

```
输入 [B,4,256,512] (RGB + Padding Mask)
    │
    ├─ Stage1 (Stem)
    │  ├─ Conv1: 3×3,s=2,32c → [B,32,128,256]
    │  ├─ Conv2: 3×3,s=2,32c → [B,32,64,128] @1/4
    │  └─ LiteBlock×2 → X1=[B,32,64,128]
    │
    ├─ Stage2 (双分支 + 融合)
    │  ├─ 分支1 @1/4: [B,32,64,128] ──┐
    │  ├─ 分支2 @1/8: [B,64,32,64]  ──┤
    │  └─ 跨尺度融合 CRF → X2_1/4, X2_1/8
    │
    ├─ Stage3 (三分支 + 融合)
    │  ├─ 分支1 @1/4: [B,32,64,128] ──┐
    │  ├─ 分支2 @1/8: [B,64,32,64]  ──┤
    │  ├─ 分支3 @1/16: [B,128,16,32] ──┤
    │  └─ 跨尺度融合 CRF → X3_1/4, X3_1/8, X3_1/16
    │
    ├─ Stage4 (四分支 + 融合)
    │  ├─ 分支1 @1/4: [B,32,64,128] ──┐
    │  ├─ 分支2 @1/8: [B,64,32,64]  ──┤
    │  ├─ 分支3 @1/16: [B,128,16,32] ──┤
    │  ├─ 分支4 @1/32: [B,256,8,16]  ──┤
    │  └─ 跨尺度融合 CRF → B1,B2,B3,B4
    │
    ├─ LiteFPN (特征金字塔融合)
    │  ├─ 侧连接：统一到128通道
    │  ├─ 自顶向下：逐级上采样融合
    │  └─ 输出：Hf=[B,128,64,128] @1/4
    │
    └─ 预测头 (基于统一特征Hf)
       ├─ Gap Head (缺口检测)
       │  ├─ Heatmap: 1×1 Conv → [B,1,64,128]
       │  └─ Offset: 1×1 Conv → [B,2,64,128]
       └─ Slider Head (滑块检测)
          ├─ Heatmap: 1×1 Conv → [B,1,64,128]
          └─ Offset: 1×1 Conv → [B,2,64,128]
```

### 关键设计

- **骨干网络**：Lite-HRNet-18，多分辨率并行处理架构
- **核心模块**：LiteBlock (MBConv风格，e=2扩展因子)
- **特征融合**：LiteFPN金字塔网络，统一到128通道
- **输入增强**：4通道输入（RGB + Padding Mask），防止边缘误检
- **检测头**：双分支CenterNet设计，统一在1/4分辨率预测
- **损失函数**：改进Focal Loss（α=2,γ=4）+ L1 Loss（λ=1.0）
- **解码策略**：Soft-argmax + 亚像素偏移校正

### 模型参数

| 组件              | 参数量       | 说明                    |
| --------------- | --------- | --------------------- |
| Backbone        | ~2.8M     | Lite-HRNet-18多分辨率主干    |
| LiteFPN         | ~0.5M     | 特征金字塔网络（128通道）        |
| Detection Heads | ~0.2M     | 双分支检测头（各3个1×1卷积）     |
| **总计**          | **~3.3M** | FP32模型约14MB          |

## 📈 性能指标

### 准确率（基于滑动距离误差）

| 数据集     | 2px阈值   | 5px阈值   | 7px阈值   | MAE     |
| ------- | ------- | ------- | ------- | ------- |
| 测试集（生成） | 95%+    | 99.5%   | 99.9%   | ~0.5px  |
| 真实验证码   | **70%** | **85%** | **90%** | **~1px** |

### 推理性能

| 硬件          | 推理时间      | FPS   | 内存占用   |
| ----------- | --------- | ----- | ------- |
| RTX 5090    | <3ms      | 330+  | ~200MB  |
| AMD 9950X   | ~42ms     | ~24   | ~150MB  |

### 关键性能指标

- **MAE（平均绝对误差）**：约1px（真实验证码）
- **命中率@±2px**：≥95%（合成数据集）
- **推理延迟**：CPU ~42ms（实测），GPU < 3ms
- **模型大小**：12.5MB（FP32），~7MB（FP16）

## 🛠️ 主要功能

### 1. 数据生成

- 自动下载Pixabay图片
- 批量生成滑块验证码
- 支持多种拼图形状
- 生成多种混淆

### 2. 模型训练

- 自动学习率调度
- TensorBoard 训练过程可视化

### 3. 推理部署

- 支持预测不同形状的拼图（普通拼图形状，圆形，三角形，方形等）
- 支持不同大小的验证码图片的输入
- 支持热图可视化

### 4. 评估分析

- TensorBoard可视化

## ⚠️ 免责声明

**本项目仅供学习和研究使用，不得用于任何商业或非法用途。**

1. 本项目旨在促进计算机视觉和深度学习技术的学术研究
2. 使用者需遵守相关法律法规，不得将本项目用于绕过网站安全机制
3. 因使用本项目产生的任何法律责任由使用者自行承担
4. 请勿将本项目用于任何可能损害他人利益的行为

## 📞 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。

---

<div align="center">
<i>本项目遵循MIT协议，仅供学习研究使用</i>
</div>