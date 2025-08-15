# 预处理输出文档

## 目录
- [概述](#概述)
- [输出目录结构](#输出目录结构)
- [文件格式说明](#文件格式说明)
- [预处理流程](#预处理流程)
- [坐标系统](#坐标系统)
- [使用示例](#使用示例)
- [配置说明](#配置说明)

## 概述

本文档详细说明了滑块验证码数据集预处理系统的输出格式、文件位置和预处理方法。预处理系统将原始验证码图像转换为模型训练所需的标准化格式。

### 核心特性
- **批处理优化**：每64个样本保存为一个NPY文件，减少文件I/O
- **流式处理**：避免内存累积，支持大规模数据集
- **标准化格式**：统一的512×256输入尺寸，保持宽高比
- **子像素精度**：通过offset机制实现亚像素级定位

## 输出目录结构

```
data/processed/
├── images/
│   ├── train/                     # 训练集图像数据
│   │   ├── train_0000.npy        # 批次0：包含64张图像
│   │   ├── train_0001.npy        # 批次1：包含64张图像
│   │   └── ...
│   ├── val/                       # 验证集图像数据
│   │   ├── val_0000.npy
│   │   └── ...
│   └── test/                      # 测试集图像数据
│       ├── test_0000.npy
│       └── ...
├── labels/
│   ├── train/                     # 训练集标签数据
│   │   ├── train_0000_heatmaps.npy   # 热图
│   │   ├── train_0000_offsets.npy    # 偏移量
│   │   ├── train_0000_weights.npy    # 权重
│   │   ├── train_0000_meta.json      # 元数据
│   │   └── ...
│   ├── val/                       # 验证集标签数据
│   │   └── ...
│   └── test/                      # 测试集标签数据
│       └── ...
└── split_info/                    # 数据集划分信息
    ├── dataset_split_info.json    # 划分统计
    ├── train_sample_ids.json      # 训练集样本ID
    ├── val_sample_ids.json        # 验证集样本ID
    └── test_sample_ids.json       # 测试集样本ID
```

## 文件格式说明

### 1. 图像文件 (`{split}_{batch_id:04d}.npy`)

**格式**: NumPy数组
**维度**: `(batch_size, 4, 256, 512)`
**数据类型**: `float32`

**通道说明**:
- Channel 0-2: RGB图像通道（归一化到0-1）
- Channel 3: Padding mask（0=padding, 1=valid）

```python
# 批次结构
batch_images = {
    'shape': (64, 4, 256, 512),  # (B, C, H, W)
    'dtype': 'float32',
    'channels': {
        0: 'Red channel',
        1: 'Green channel', 
        2: 'Blue channel',
        3: 'Padding mask'
    }
}
```

### 2. 热图文件 (`{split}_{batch_id:04d}_heatmaps.npy`)

**格式**: NumPy数组
**维度**: `(batch_size, 2, 64, 128)`
**数据类型**: `float32`

**通道说明**:
- Channel 0: 缺口（gap）中心点热图
- Channel 1: 滑块（slider）中心点热图

热图使用高斯分布生成，峰值为1.0，标准差σ=1.5（栅格单位）。

### 3. 偏移量文件 (`{split}_{batch_id:04d}_offsets.npy`)

**格式**: NumPy数组
**维度**: `(batch_size, 4, 64, 128)`
**数据类型**: `float32`

**通道说明**:
- Channel 0: 缺口X方向偏移（-0.5 ~ 0.5）
- Channel 1: 缺口Y方向偏移（-0.5 ~ 0.5）
- Channel 2: 滑块X方向偏移（-0.5 ~ 0.5）
- Channel 3: 滑块Y方向偏移（-0.5 ~ 0.5）

偏移量仅在热图峰值位置有效，用于亚像素精度定位。

### 4. 权重文件 (`{split}_{batch_id:04d}_weights.npy`)

**格式**: NumPy数组
**维度**: `(batch_size, 2, 64, 128)`
**数据类型**: `float32`

**通道说明**:
- Channel 0: 缺口位置权重（1.0=正样本，0.0=负样本）
- Channel 1: 滑块位置权重（1.0=正样本，0.0=负样本）

用于损失函数计算时的样本加权。

### 5. 元数据文件 (`{split}_{batch_id:04d}_meta.json`)

**格式**: JSON
**结构**:
```json
{
    "batch_id": 0,
    "batch_size": 64,
    "samples": [
        {
            "index": 0,
            "sample_id": "Pic0001_Bgx150Bgy80_Sdx25Sdy80_abc123",
            "pic_id": "Pic0001",
            "letterbox_params": {
                "scale": 0.5,
                "dx": 0,
                "dy": 0,
                "w_scaled": 256,
                "h_scaled": 128,
                "w_letterbox": 512,
                "h_letterbox": 256
            },
            "grid_coords": {
                "gap": [37, 20],
                "slider": [6, 20]
            },
            "offsets_meta": {
                "gap": [0.25, 0.0],
                "slider": [0.125, 0.0]
            },
            "original_coords": {
                "gap": [150, 80],
                "slider": [25, 80]
            }
        },
        // ... 更多样本
    ]
}
```

## 预处理流程

### 1. Letterbox变换
将原始图像（320×160）调整到标准输入尺寸（512×256），保持宽高比：

```python
# 原始图像: 320×160
# 目标尺寸: 512×256

# 计算缩放比例
scale = min(512/320, 256/160) = min(1.6, 1.6) = 1.6

# 缩放后尺寸
scaled_w = 320 * 1.6 = 512
scaled_h = 160 * 1.6 = 256

# 无需padding（完美匹配）
```

### 2. 坐标转换
三级坐标系统：

1. **原始坐标系** (320×160)
   - 原始标注坐标
   
2. **Letterbox坐标系** (512×256)  
   - 缩放后的坐标
   - 公式: `letterbox_coord = original_coord * scale + offset`
   
3. **栅格坐标系** (128×64)
   - 下采样4倍的特征图坐标
   - 公式: `grid_coord = letterbox_coord / 4`

### 3. 热图生成
使用高斯分布在栅格坐标系生成热图：

```python
def generate_gaussian_heatmap(center, size=(64, 128), sigma=1.5):
    """
    生成高斯热图
    center: (x, y) 栅格坐标
    size: 热图尺寸 (H, W)
    sigma: 高斯分布标准差
    """
    heatmap = np.zeros(size)
    # 计算高斯分布
    for y in range(size[0]):
        for x in range(size[1]):
            dist = ((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2)
            heatmap[y, x] = np.exp(-dist)
    return heatmap
```

### 4. 偏移量计算
记录亚像素偏移以提高精度：

```python
# 精确坐标到栅格坐标的转换
precise_coord = 150.4  # 原始坐标
grid_coord = precise_coord / 4 = 37.6  # 栅格坐标

# 分解为整数和小数部分
grid_int = 37  # 栅格位置
offset = 0.6 - 0.5 = 0.1  # 偏移量（归一化到-0.5~0.5）
```

## 坐标系统

### 坐标恢复公式

从预测结果恢复原始坐标：

```python
# 1. 从热图获取峰值位置（栅格坐标）
grid_x, grid_y = np.unravel_index(heatmap.argmax(), heatmap.shape)

# 2. 加上偏移量获得精确栅格坐标
precise_grid_x = grid_x + 0.5 + offset_x  # offset_x ∈ [-0.5, 0.5]
precise_grid_y = grid_y + 0.5 + offset_y

# 3. 转换到Letterbox坐标系
letterbox_x = precise_grid_x * 4
letterbox_y = precise_grid_y * 4

# 4. 转换回原始坐标系
original_x = (letterbox_x - dx) / scale
original_y = (letterbox_y - dy) / scale
```

## 使用示例

### 1. 生成预处理数据

```bash
# 使用配置文件生成所有数据集
python scripts/data_generation/preprocess_dataset.py --auto-split

# 仅生成训练集
python scripts/data_generation/preprocess_dataset.py --split train

# 自定义配置
python scripts/data_generation/preprocess_dataset.py \
    --config custom_config.yaml \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### 2. 加载预处理数据

```python
import numpy as np
import json

# 加载批次数据
batch_id = 0
split = 'train'

# 加载图像
images = np.load(f'data/processed/images/{split}/{split}_{batch_id:04d}.npy')
print(f"Images shape: {images.shape}")  # (64, 4, 256, 512)

# 加载标签
heatmaps = np.load(f'data/processed/labels/{split}/{split}_{batch_id:04d}_heatmaps.npy')
offsets = np.load(f'data/processed/labels/{split}/{split}_{batch_id:04d}_offsets.npy')
weights = np.load(f'data/processed/labels/{split}/{split}_{batch_id:04d}_weights.npy')

# 加载元数据
with open(f'data/processed/labels/{split}/{split}_{batch_id:04d}_meta.json', 'r') as f:
    meta = json.load(f)

# 获取单个样本
sample_idx = 0
sample_image = images[sample_idx]  # (4, 256, 512)
sample_meta = meta['samples'][sample_idx]

print(f"Sample ID: {sample_meta['sample_id']}")
print(f"Original gap coords: {sample_meta['original_coords']['gap']}")
print(f"Grid gap coords: {sample_meta['grid_coords']['gap']}")
```

### 3. 可视化预处理结果

```python
import matplotlib.pyplot as plt

def visualize_sample(images, heatmaps, meta, sample_idx=0):
    """可视化单个样本"""
    # 获取样本数据
    img = images[sample_idx, :3].transpose(1, 2, 0)  # (H, W, 3)
    gap_heatmap = heatmaps[sample_idx, 0]  # (64, 128)
    slider_heatmap = heatmaps[sample_idx, 1]
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示图像
    axes[0].imshow(img)
    axes[0].set_title('Preprocessed Image')
    
    # 显示缺口热图
    axes[1].imshow(gap_heatmap, cmap='hot')
    axes[1].set_title('Gap Heatmap')
    
    # 显示滑块热图
    axes[2].imshow(slider_heatmap, cmap='hot')
    axes[2].set_title('Slider Heatmap')
    
    plt.tight_layout()
    plt.show()
```

## 配置说明

预处理配置文件位于 `config/preprocessing_config.yaml`：

### 关键配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `letterbox.target_size` | [512, 256] | 目标图像尺寸 [宽, 高] |
| `letterbox.fill_value` | 255 | Padding填充值（白色） |
| `coordinate.downsample` | 4 | 下采样率 |
| `heatmap.sigma` | 1.5 | 高斯分布标准差 |
| `dataset.batch_size` | 64 | 批次大小 |
| `dataset.num_workers` | 12 | 并行进程数 |
| `data_split.split_ratio` | {train: 0.8, val: 0.1, test: 0.1} | 数据集划分比例 |


## 总结

预处理系统提供了一套完整的数据准备流程，将原始验证码图像转换为适合深度学习模型训练的标准化格式。通过批处理、流式处理和多进程并行，实现了高效的大规模数据集处理能力。