# 神经网络训练标签规范文档

## 一、最小必需标签（per sample）

### 1. 路径与尺寸

- **piece_path**：拼图图像路径
  - 尺寸：$H_p = W_p \in [40, 60]$ 像素（正方形）
  
- **bg_path**：带缺口背景图路径
  - 尺寸：160×320 像素
  
- **comp_path**：合成图路径（拼图叠加在背景上）
  - 尺寸：160×320 像素

### 2. 合成图中拼图的中心坐标（像素）

```python
comp.piece_center = (x_pc, y_pc)
```

**说明**：这是阶段A（合成图里找拼图）用于监督的真值。用于训练2D相关热力图和亚像素精修。

### 3. 背景图中真实缺口的中心坐标（像素）

```python
bg.gap_center = (x_gc, y_gc)
```

**说明**：这是阶段B（背景里找缺口）用于监督的真值。用于训练1D横向相关和最终定位。

### 4. 真实缺口相对拼图的微旋转与尺度

```python
gap.delta_theta_deg = float  # 一般为生成时对"真缺口"施加的 0.5–1.8°，有符号
gap.scale = float           # 若未缩放就写 1.0
```

**说明**：用于位姿回归头的监督信号。

> **注意**：以上1-4项就能训练"合成图2D粗定位 + 背景1D粗定位 + 位姿微回归 + 亚像素精修"。其它信息都能从图像本身或这些元数据里推导。

## 二、推荐增强标签（强烈建议，有则显著提升判别）

### 1. "旋转假缺口"列表（负样本）

```python
bg.fake_gaps = [
    {
        "center": (x_f1, y_f1),
        "delta_theta_deg": θ_f1,  # ±10–30°
        "scale": s_f1
    },
    {
        "center": (x_f2, y_f2),
        "delta_theta_deg": θ_f2,
        "scale": s_f2
    },
    # ... 1-3个假缺口
]
```

**说明**：这些是生成时添加的1-3个假缺口（旋转±10–30°、缩放）；训练时用于难负样本与间隔损失（OriScore margin）。

## 三、标签文件格式建议

### JSON格式示例

```json
{
    "sample_id": "Pic0001_Bgx120Bgy70_Sdx30Sdy70_hash",
    "paths": {
        "piece": "path/to/piece.png",
        "background": "path/to/background.png",
        "composite": "path/to/composite.png"
    },
    "labels": {
        "comp_piece_center": [x_pc, y_pc],
        "bg_gap_center": [x_gc, y_gc],
        "gap_pose": {
            "delta_theta_deg": 0.8,
            "scale": 1.0
        }
    },
    "augmented_labels": {
        "fake_gaps": [
            {
                "center": [x_f1, y_f1],
                "delta_theta_deg": 15.0,
                "scale": 0.9
            }
        ]
    }
}
```

### CSV格式示例

| sample_id | piece_path | bg_path | comp_path | x_pc | y_pc | x_gc | y_gc | delta_theta | scale | fake_gaps_json |
|-----------|------------|---------|-----------|------|------|------|------|-------------|-------|----------------|
| Pic0001_... | path1 | path2 | path3 | 30 | 70 | 120 | 70 | 0.8 | 1.0 | [{...}] |

## 四、数据加载器实现建议

```python
class CaptchaDataset:
    def __init__(self, label_file):
        # 加载标签文件
        self.labels = self._load_labels(label_file)
    
    def __getitem__(self, idx):
        sample = self.labels[idx]
        
        # 加载图像
        piece = load_image(sample['paths']['piece'])
        background = load_image(sample['paths']['background'])
        composite = load_image(sample['paths']['composite'])
        
        # 提取标签
        labels = {
            'comp_center': sample['labels']['comp_piece_center'],
            'bg_center': sample['labels']['bg_gap_center'],
            'delta_theta': sample['labels']['gap_pose']['delta_theta_deg'],
            'scale': sample['labels']['gap_pose']['scale']
        }
        
        # 如果有增强标签
        if 'augmented_labels' in sample:
            labels['fake_gaps'] = sample['augmented_labels']['fake_gaps']
        
        return piece, background, composite, labels
```