# Industrial-Grade Slider CAPTCHA Recognition System

<div align="center">

[English](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/README.md) | [简体中文](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/README_zh.md)

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/badge/PyPI-v1.0.3-blue.svg)](https://pypi.org/project/sider-captcha-solver/)
[![GitHub version](https://img.shields.io/badge/GitHub-v1.1.0-blue.svg)](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver)

A high-precision slider CAPTCHA recognition solution based on deep learning, using **Lite-HRNet-18-FPN** architecture and improved CenterNet detection heads. Achieves 91.8% hit rate within 7px and 85.7% within 5px on real CAPTCHA datasets, with 99%+ hit rate within 5px on synthetic validation sets.

**Latest Version**: v1.1.0

</div>

## 🆕 Changelog

### v1.1.0 (2025-08-17) - Latest Version

- 🏗️ **Architecture Upgrade**:
  - Adopted **Lite-HRNet-18-FPN** backbone network for enhanced feature extraction
  - Dual-branch detection head design for gap and slider detection separately
  - 4-channel input support (RGB + Padding Mask) to prevent padding area misdetection, supporting various image sizes
- 🎯 **Significant Accuracy Improvement**:
  - Real CAPTCHA accuracy improved to **90%+** (7px error threshold)
  - MAE reduced to ~1px with sub-pixel precision localization
- ⚡ **Performance Optimization**:
  - Model parameters only 3.3M, FP32 model ~12.53MB
  - GPU inference speed <3ms (RTX 5090)
  - CPU inference 12-14ms (AMD 9950X)
- 🛡️ **Enhanced Anti-Confusion Features**:
  - Added central hollow confusion for puzzle pieces
  - Modified confusion generation probability logic to fixed sample counts

### v1.0.3 (2025-07-27)

- 🛡️ **Enhanced Anti-Confusion Features**:
  - Gap rotation (0.5-1.8° random rotation, 50% probability)
  - Slider Perlin noise (40-80% intensity, 50% probability)
  - Confusion gaps (±10-30° rotation, 60% probability)
  - Gap highlight effects (30% probability)
- 📊 **Model Performance Improvement**:
  - Real CAPTCHA **85%+ accuracy**, enhanced anti-interference capability
  - Better adversarial sample defense
  - More stable predictions in complex scenarios

### v1.0.2 (2025-07-21) - Initial Release

- 🚀 First public release
- 📦 Basic slider CAPTCHA recognition
- 🎯 Real CAPTCHA 80% accuracy at 7px error
- 💡 Support for 11 puzzle shapes (5 regular + 6 special)
- ⚡ Fast inference: GPU 2ms, CPU 5ms

## 📑 Table of Contents

- [📋 Project Overview](#-project-overview)
  - [🎯 Core Features](#-core-features)
  - [🖼️ Recognition Effect Demo](#️-recognition-effect-demo)
- [🚀 Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
- [📊 Data Generation Pipeline](#-data-generation-pipeline)
- [🏗️ Network Architecture](#️-network-architecture)
- [📈 Performance Metrics](#-performance-metrics)
- [🛠️ Main Features](#️-main-features)
- [⚠️ Disclaimer](#️-disclaimer)
- [📞 Contact](#-contact)

## 📋 Project Overview

This project is an industrial-grade slider CAPTCHA recognition system that solves the accuracy bottleneck of traditional template matching algorithms through deep learning methods. The system is trained on **over 300,000** synthetic CAPTCHA images, using a lightweight CNN architecture that achieves real-time inference while maintaining high accuracy.

### 🎯 Core Features

- **High-Precision Recognition**: Real CAPTCHA accuracy reaches **90%+** at 7px error (v1.1.0), MAE ~1.5px
- **Enhanced Anti-Confusion**: Supports gap rotation, slider Perlin noise, confusion gaps, gap highlight effects
- **Real-Time Inference**: Lightweight model supports real-time applications
- **Lightweight Architecture**: Only 3.5M parameters, model file ~14MB (FP32)
- **Industrial-Grade Design**: Complete data generation, training, and evaluation pipeline
- **Sub-Pixel Precision**: Achieves sub-pixel level localization using CenterNet offset mechanism

### 🖼️ Recognition Effect Demo

#### Real CAPTCHA Dataset Recognition Effect

![Real Dataset Recognition Effect](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/results/1.1.0/site2/visualizations/Pic0002_Bgx182Bgy157_Sdx34Sdy157_5a4abbb4_result.png)

*Figure: Recognition effect on real website CAPTCHAs, red circle marks gap position, blue circle marks slider position*

#### Test Set Recognition Effect

![Test Set Recognition Effect](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/results/1.1.0/test/visualizations/Pic2002_Bgx210Bgy43_Sdx25Sdy43_4ed0c0fd_result.png)

## 🚀 Quick Start

### Installation

#### Install via pip

```bash
# Please refer to Releases\v1.0.3\api_example.py in the project root directory for usage instructions of version v1.0.3.
pip install sider-captcha-solver  # Install v1.0.3 (v1.1.0 coming soon)
```

### Basic Usage

After pip installation, you can directly import and use: (Latest version still being refined, not yet uploaded to PyPI)

#### 1. Get Sliding Distance and Details

```python
from sider_captcha_solver import solve

# Get detailed information
result = solve('data/captchas/Pic0001_Bgx84Bgy206_Sdx25Sdy206_1a85adc2.png', detailed=True)
print(f"Sliding distance: {result['distance']:.2f} px")
print(f"Gap position: {result['gap']}")
print(f"Slider position: {result['slider']}")
print(f"Confidence: {result['confidence']:.2f}")
```

#### 2. Visualize Prediction Results

```python
from sider_captcha_solver import visualize

# Visualize prediction results (auto predict and annotate)
visualize('data/captchas/Pic0001_Bgx84Bgy206_Sdx25Sdy206_1a85adc2.png', save_path='result.png', show=True)
```

## 📊 Data Generation Pipeline

### 1. Data Collection

Downloaded high-quality images from Pixabay across 10 categories as backgrounds: Minecraft, Pixel Food, Block Public Square, Block Illustration, Backgrounds, Buildings, Nature, Anime Cityscape, Abstract Geometric Art, etc. Maximum 200 images per category, totaling about 2,000 original images.
Plus hundreds of computer-generated confusion images.

### 2. CAPTCHA Generation Logic

```
Original Images (2000+) → Random Resize (280×140 ~ 480×240) → Add Anti-Confusion Features → Hole Generation
                                                                            ↓
                                                         16 shapes × 3 sizes × multiple random positions
                                                                            ↓
                                                      Each original generates 100+ CAPTCHAs
                                                                            ↓
                                                      Total ~260,000 training images generated
```

**Puzzle Shape Design**:

- 10 regular puzzle shapes (selected from 81 combinations of four-sided concave-convex-flat)
- 6 special shapes (circle, square, triangle, hexagon, pentagon, pentagram)

**Anti-Confusion Enhancement Features** (v1.1.0 additions):

- **Gap Rotation** (10% probability): 0.5-1.8° random rotation
- **Perlin Noise** (20% probability): Slider surface 40-80% intensity noise texture
- **Confusion Gaps** (30% probability): ±10-30° rotated interference gaps
- **Gap Highlight** (10% probability): Traditional shadow effect changed to highlight effect

**Random Parameters**:

- Image size: 280×140 ~ 480×240 pixels random
- Puzzle size: 30-50 pixels (3 random sizes)
- Position distribution: Slider x∈[15,35], Gap x∈[65,305], ensuring no overlap
- Lighting effects: Randomly add lighting variations for enhanced robustness

### 3. Dataset Split

- Training set: 80% (split by original images to avoid data leakage)
- Test set: 10%
- Validation set: 10%

## 🏗️ Network Architecture

### Model Structure (v1.1.0 - Lite-HRNet-18-FPN)

```
Input [B,4,256,512] (RGB + Padding Mask)
    │
    ├─ Stage1 (Stem)
    │  ├─ Conv1: 3×3,s=2,32c → [B,32,128,256]
    │  ├─ Conv2: 3×3,s=2,32c → [B,32,64,128] @1/4
    │  └─ LiteBlock×2 → X1=[B,32,64,128]
    │
    ├─ Stage2 (Dual-branch + Fusion)
    │  ├─ Branch1 @1/4: [B,32,64,128] ──┐
    │  ├─ Branch2 @1/8: [B,64,32,64]  ──┤
    │  └─ Cross-scale fusion CRF → X2_1/4, X2_1/8
    │
    ├─ Stage3 (Triple-branch + Fusion)
    │  ├─ Branch1 @1/4: [B,32,64,128] ──┐
    │  ├─ Branch2 @1/8: [B,64,32,64]  ──┤
    │  ├─ Branch3 @1/16: [B,128,16,32] ──┤
    │  └─ Cross-scale fusion CRF → X3_1/4, X3_1/8, X3_1/16
    │
    ├─ Stage4 (Quad-branch + Fusion)
    │  ├─ Branch1 @1/4: [B,32,64,128] ──┐
    │  ├─ Branch2 @1/8: [B,64,32,64]  ──┤
    │  ├─ Branch3 @1/16: [B,128,16,32] ──┤
    │  ├─ Branch4 @1/32: [B,256,8,16]  ──┤
    │  └─ Cross-scale fusion CRF → B1,B2,B3,B4
    │
    ├─ LiteFPN (Feature Pyramid Network)
    │  ├─ Lateral connections: Unify to 128 channels
    │  ├─ Top-down: Progressive upsampling and fusion
    │  └─ Output: Hf=[B,128,64,128] @1/4
    │
    └─ Prediction Heads (Based on unified feature Hf)
       ├─ Gap Head (Gap detection)
       │  ├─ Heatmap: 1×1 Conv → [B,1,64,128]
       │  └─ Offset: 1×1 Conv → [B,2,64,128]
       └─ Slider Head (Slider detection)
          ├─ Heatmap: 1×1 Conv → [B,1,64,128]
          └─ Offset: 1×1 Conv → [B,2,64,128]
```

### Key Design

- **Backbone Network**: Lite-HRNet-18, multi-resolution parallel processing architecture
- **Core Module**: LiteBlock (MBConv style, e=2 expansion factor)
- **Feature Fusion**: LiteFPN pyramid network, unified to 128 channels
- **Input Enhancement**: 4-channel input (RGB + Padding Mask) to prevent edge misdetection
- **Detection Heads**: Dual-branch CenterNet design, unified prediction at 1/4 resolution
- **Loss Function**: Improved Focal Loss (α=2, γ=4) + L1 Loss (λ=1.0)
- **Decoding Strategy**: Soft-argmax + sub-pixel offset correction

### Model Parameters

| Component       | Parameters | Description                           |
| --------------- | ---------- | ------------------------------------- |
| Backbone        | ~2.8M      | Lite-HRNet-18 multi-resolution backbone |
| LiteFPN         | ~0.5M      | Feature pyramid network (128 channels)  |
| Detection Heads | ~0.2M      | Dual-branch detection heads (3×1×1 convs each) |
| **Total**       | **~3.3M**  | FP32 model ~14MB                      |

## 📈 Performance Metrics

### Accuracy (Based on Sliding Distance Error)

| Dataset          | 5px Threshold | 7px Threshold | MAE     |
| ---------------- | ------------- | ------------- | ------- |
| Test Set (Synthetic) | 99.5%        | 99.9%        | ~0.5px  |
| Real CAPTCHAs    | **85%**       | **90%**       | **~1px** |

### Inference Performance

| Hardware     | Inference Time | FPS   | Memory Usage |
| ------------ | -------------- | ----- | ------------ |
| RTX 5090     | <3ms          | 330+  | ~200MB       |
| AMD 9950X    | ~42ms         | ~24   | ~150MB       |

### Key Performance Indicators

- **MAE (Mean Absolute Error)**: ~1px (real CAPTCHAs)
- **Hit Rate @±2px**: ≥95% (synthetic dataset)
- **Inference Latency**: CPU ~42ms (measured), GPU < 3ms
- **Model Size**: 12.5MB (FP32), ~7MB (FP16)

## 🛠️ Main Features

### 1. Data Generation

- Automatic Pixabay image download
- Batch slider CAPTCHA generation
- Support for multiple puzzle shapes
- Generate various confusions

### 2. Model Training

- Automatic learning rate scheduling
- TensorBoard training visualization

### 3. Inference Deployment

- Support prediction of different puzzle shapes (regular puzzles, circles, triangles, squares, etc.)
- Support different sizes of CAPTCHA image inputs
- Support heatmap visualization

### 4. Evaluation Analysis

- TensorBoard visualization

## ⚠️ Disclaimer

**This project is for learning and research purposes only, and must not be used for any commercial or illegal purposes.**

1. This project aims to promote academic research in computer vision and deep learning technologies
2. Users must comply with relevant laws and regulations and must not use this project to bypass website security mechanisms
3. Any legal liability arising from the use of this project shall be borne by the user
4. Please do not use this project for any actions that may harm the interests of others

## 📞 Contact

For questions or suggestions, please submit an Issue or Pull Request.

---

<div align="center">
<i>This project follows the MIT license and is for learning and research purposes only</i>
</div>