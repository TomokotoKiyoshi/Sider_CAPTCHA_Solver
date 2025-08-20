# 真实数据集微调落地方案

## 项目背景
本方案旨在使用真实验证码数据对现有的滑块验证码识别模型进行微调，以提升模型在实际应用场景中的准确率和鲁棒性。

## 一、数据准备阶段

### 1.1 真实数据收集与标注

#### 数据收集
- **现有数据源**：
  - `data/real_captchas/site1/` - 网站1的真实验证码
  - `data/real_captchas/site2/` - 网站2的真实验证码
  
- **新数据采集**：
  ```bash
  # 使用数据采集脚本
  python scripts/data_generation/capture_captcha.py \
      --site target_website \
      --output data/real_captchas/new_site \
      --count 1000
  ```

#### 数据标注
- **标注工具**：
  ```bash
  # 使用matplotlib标注工具（支持Spyder等环境）
  python scripts/annotation/annotate_captchas_matplotlib.py \
      --input data/real_captchas/merged \
      --output data/real_captchas/annotated \
      --max_images 1000
  ```

- **标注要求**：
  - 精确标注缺口中心点坐标
  - 精确标注滑块中心点坐标
  - 建议标注500-1000张高质量样本
  - 确保Y轴坐标的准确性（用于验证对齐效果）

### 1.2 数据预处理

#### 创建真实数据预处理脚本
创建文件：`scripts/data_generation/preprocess_real_data.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实验证码数据预处理脚本
将标注的真实数据转换为训练所需的NPY格式
"""
import numpy as np
import cv2
import json
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.letterbox_transform import LetterboxTransform
from src.preprocessing.heatmap_generator import HeatmapGenerator

class RealDataPreprocessor:
    def __init__(self, config_path="config/preprocessing_config.yaml"):
        # 加载配置
        self.config = self.load_config(config_path)
        self.letterbox = LetterboxTransform(self.config)
        self.heatmap_gen = HeatmapGenerator(self.config)
        
    def process_batch(self, image_paths, annotations):
        """处理一批真实数据"""
        processed_images = []
        processed_labels = []
        
        for img_path, annot in zip(image_paths, annotations):
            # 1. 读取图像
            image = cv2.imread(str(img_path))
            
            # 2. Letterbox变换
            transformed = self.letterbox.transform(image)
            
            # 3. 生成热力图标签
            heatmaps = self.heatmap_gen.generate(
                gap_coord=annot['gap'],
                slider_coord=annot['slider']
            )
            
            # 4. 计算子像素偏移
            offsets = self.calculate_offsets(annot)
            
            processed_images.append(transformed)
            processed_labels.append({
                'heatmaps': heatmaps,
                'offsets': offsets,
                'coords': annot
            })
            
        return processed_images, processed_labels
```

#### 数据增强策略
- **轻微增强**（保持真实性）：
  - 亮度调整：±10%
  - 对比度调整：±10%
  - 轻微高斯噪声：σ=1-2
  - 不建议使用几何变换（会影响坐标准确性）

### 1.3 数据集划分
```bash
# 执行数据预处理和划分
python scripts/data_generation/preprocess_real_data.py \
    --input data/real_captchas/annotated \
    --output data/real_processed \
    --split-ratio 0.8:0.1:0.1  # 训练:验证:测试
    --augment light  # 轻微数据增强
```

## 二、模型配置调整

### 2.1 创建微调配置文件

创建 `config/finetune_config.yaml`：

```yaml
# 基于training_config.yaml的微调配置
# 继承基础配置，仅修改必要参数

# 数据配置
data:
  processed_dir: "data/real_processed"  # 真实数据处理后的目录
  # 混合训练时的数据源
  synthetic_dir: "data/processed"       # 合成数据目录（可选）
  mix_ratio: 0.3                        # 真实数据占比（30%真实+70%合成）
  
model:
  name: "Lite-HRNet-18+LiteFPN"
  input_channels: 4
  model_path: "src.models.lite_hrnet_18_fpn"
  # 预训练模型路径
  pretrained: "src/checkpoints/1.1.0/best_model.pth"
  
optimizer:
  name: AdamW
  lr: 3.0e-5              # 降低10倍，防止灾难性遗忘
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.01      # 减小正则化
  clip_grad_norm: 1.0
  ema_decay: 0.999
  
sched:
  warmup_epochs: 1
  cosine_min_lr: 3.0e-7
  epochs: 20              # 减少总训练轮数
  
train:
  batch_size: 32          # 减小批次大小（真实数据量较少）
  gradient_accumulation_steps: 4  # 增加累积，有效批次=128
  amp: true
  channels_last: true
  num_workers: 4
  
  # 微调特定设置
  freeze_backbone: true   # 第一阶段冻结骨干网络
  freeze_epochs: 5        # 冻结训练的轮数
  
eval:
  metrics: [mae_px, rmse_px, hit_le_2px, hit_le_5px, y_axis_consistency]
  select_by: hit_le_5px   # 保持5像素精度选择
  
  # Y轴一致性评估
  y_axis_metrics:
    enabled: true
    threshold: 2.0        # Y轴差异阈值（像素）
    
  early_stopping:
    min_epochs: 5         # 更早允许早停
    patience: 5           # 减少等待轮数
    min_delta: 0.1        # 最小改善阈值
    
checkpoints:
  save_dir: "src/checkpoints/finetune_v1"
  save_interval: 1
  save_best: true
  
logging:
  log_dir: "logs/finetune"
  tensorboard_dir: "runs/finetune"
```

### 2.2 损失函数权重调整

创建 `config/loss_finetune.yaml`：

```yaml
# 微调专用损失函数配置
# 针对真实数据特点调整权重

focal_loss:
  alpha: 1.5
  beta: 4.0
  pos_threshold: 0.8
  eps: 1.0e-8

offset_loss:
  beta: 1.0

hard_negative_loss:
  margin: 0.7              # 提高margin，真实数据干扰更多
  reduction: mean

angle_loss:
  pos_threshold: 0.7
  eps: 1.0e-8

padding_bce_loss:
  enabled: true
  eps: 1.0e-8
  max_loss_per_pixel: 10.0

total_loss:
  weights:
    heatmap: 1.0
    offset: 1.5          # 增加偏移权重，提高精度
    hard_negative: 2.0   # 增加，应对真实数据的复杂背景
    angle: 0.3           # 降低，真实数据旋转较少
    padding_bce: 0.3     # 降低，真实数据padding模式不同
    
  # Y轴一致性损失（新增）
  y_consistency:
    enabled: true
    weight: 0.5          # Y轴一致性损失权重
    threshold: 4.0       # 4像素容差
```

## 三、训练策略

### 3.1 两阶段微调法

#### 第一阶段：冻结骨干网络（5 epochs）
- 只微调检测头（stage6_dual_head）
- 快速适应真实数据分布
- 防止骨干网络过度调整

```python
# 实现冻结逻辑
def freeze_backbone(model):
    """冻结骨干网络，只训练检测头"""
    for name, param in model.named_parameters():
        if 'stage6' not in name:  # 只有stage6（检测头）可训练
            param.requires_grad = False
```

#### 第二阶段：全网络微调（15 epochs）
- 解冻所有层
- 使用更小的学习率（1e-5）
- 精细调整整个网络

```python
def unfreeze_all(model):
    """解冻所有层"""
    for param in model.parameters():
        param.requires_grad = True
```

### 3.2 混合训练策略（推荐）

#### 数据混合加载器
```python
class MixedDataLoader:
    """混合真实和合成数据的加载器"""
    def __init__(self, real_loader, synthetic_loader, mix_ratio=0.3):
        self.real_loader = real_loader
        self.synthetic_loader = synthetic_loader
        self.mix_ratio = mix_ratio
        
    def __iter__(self):
        for real_batch, synth_batch in zip(self.real_loader, self.synthetic_loader):
            if random.random() < self.mix_ratio:
                yield real_batch  # 30%概率返回真实数据
            else:
                yield synth_batch  # 70%概率返回合成数据
```

#### 优势
- 保持模型泛化能力
- 避免过拟合到少量真实数据
- 利用大量合成数据的多样性

## 四、实施步骤

### 4.1 环境准备
```bash
# 确保环境配置正确
pip install -r requirements.txt

# 验证GPU可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 4.2 数据准备
```bash
# Step 1: 标注真实数据
python scripts/annotation/annotate_captchas_matplotlib.py \
    --input data/real_captchas/site1 \
    --output data/real_captchas/annotated/site1 \
    --max_images 500

python scripts/annotation/annotate_captchas_matplotlib.py \
    --input data/real_captchas/site2 \
    --output data/real_captchas/annotated/site2 \
    --max_images 500

# Step 2: 合并标注数据
python scripts/data_generation/merge_annotations.py \
    --input data/real_captchas/annotated \
    --output data/real_captchas/merged_annotations.json

# Step 3: 预处理真实数据
python scripts/data_generation/preprocess_real_data.py \
    --annotations data/real_captchas/merged_annotations.json \
    --output data/real_processed \
    --batch_size 100 \
    --split_ratio 0.8:0.1:0.1
```

### 4.3 微调训练

#### 创建微调脚本
创建 `scripts/training/finetune.py`：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实数据微调训练脚本
基于train.py修改，支持两阶段微调
"""
import argparse
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import create_lite_hrnet_18_fpn
from src.training.training_engine import TrainingEngine
from src.training.validator import Validator

def main():
    parser = argparse.ArgumentParser(description='Finetune on Real Data')
    parser.add_argument('--config', default='config/finetune_config.yaml')
    parser.add_argument('--stage', choices=['freeze', 'unfreeze'], required=True)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建模型
    model = create_lite_hrnet_18_fpn(config['model'])
    
    # 加载预训练权重
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 加载基础模型权重
        checkpoint = torch.load(config['model']['pretrained'])
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 根据阶段冻结/解冻层
    if args.stage == 'freeze':
        freeze_backbone(model)
        print("Stage 1: Backbone frozen, training detection head only")
    else:
        unfreeze_all(model)
        config['optimizer']['lr'] = 1e-5  # 降低学习率
        print("Stage 2: All layers unfrozen, fine-tuning entire network")
    
    # 训练流程
    engine = TrainingEngine(model, config, device='cuda')
    validator = Validator(config, device='cuda')
    
    # 开始训练...
    # (省略具体训练循环，参考train.py)
```

#### 执行微调
```bash
# 第一阶段：冻结骨干网络（5 epochs）
python scripts/training/finetune.py \
    --config config/finetune_config.yaml \
    --stage freeze \
    --epochs 5

# 第二阶段：全网络微调（15 epochs）
python scripts/training/finetune.py \
    --config config/finetune_config.yaml \
    --stage unfreeze \
    --resume src/checkpoints/finetune_v1/epoch_005.pth \
    --epochs 15
```

### 4.4 混合训练（可选）
```bash
# 使用混合数据训练
python scripts/training/finetune.py \
    --config config/finetune_config.yaml \
    --use_mixed_data \
    --mix_ratio 0.3 \
    --synthetic_data data/processed \
    --real_data data/real_processed
```

## 五、评估与监控

### 5.1 关键评估指标

#### 主要指标
- **5像素命中率**（hit@5px）：目标 >95%
- **2像素命中率**（hit@2px）：目标 >85%
- **平均绝对误差**（MAE）：目标 <2px
- **Y轴一致性率**：目标 >98%（差异<2px）

#### Y轴一致性指标
```python
def calculate_y_consistency(predictions):
    """计算Y轴一致性指标"""
    gap_y = predictions['gap_coords'][:, 1]
    slider_y = predictions['slider_coords'][:, 1]
    
    y_diff = torch.abs(gap_y - slider_y)
    
    return {
        'y_diff_mean': y_diff.mean().item(),
        'y_diff_max': y_diff.max().item(),
        'y_consistent_rate_2px': (y_diff <= 2.0).float().mean().item() * 100,
        'y_consistent_rate_4px': (y_diff <= 4.0).float().mean().item() * 100,
    }
```

### 5.2 评估脚本使用

#### 在真实测试集上评估
```bash
# 评估微调后的模型
python scripts/validation/evaluate_real_captchas.py \
    --model src/checkpoints/finetune_v1/best_model.pth \
    --dataset data/real_captchas/test \
    --output results/finetune_evaluation.json
```

#### 对比评估（基础模型 vs 微调模型）
```bash
# 评估基础模型
python scripts/validation/evaluate_real_captchas.py \
    --model src/checkpoints/1.1.0/best_model.pth \
    --dataset data/real_captchas/test \
    --output results/baseline_evaluation.json

# 生成对比报告
python scripts/analysis/compare_models.py \
    --baseline results/baseline_evaluation.json \
    --finetuned results/finetune_evaluation.json \
    --output results/comparison_report.md
```

### 5.3 实时监控

#### TensorBoard监控
```bash
# 启动TensorBoard
tensorboard --logdir runs/finetune --port 6006
```

监控指标：
- 训练/验证损失曲线
- 各项精度指标趋势
- Y轴一致性指标
- 学习率变化

#### 关键监控点
1. **合成数据性能**：不应显著下降（<5%）
2. **真实数据性能**：应持续提升
3. **过拟合检测**：验证集性能不应持续下降
4. **Y轴对齐效果**：一致性率应提升

## 六、注意事项

### 6.1 避免过拟合

#### 预防措施
1. **数据增强**：但不过度，保持真实性
2. **Dropout**：在检测头添加Dropout层（p=0.1-0.2）
3. **早停机制**：验证集性能不再提升时停止
4. **正则化**：适当的权重衰减（weight_decay=0.01）
5. **混合训练**：使用合成数据保持泛化能力

#### 过拟合检测
- 训练损失持续下降，验证损失上升
- 训练集精度远高于验证集（差异>10%）
- 在新数据上性能急剧下降

### 6.2 Y轴对齐保证

#### 解码逻辑优化（已实现）
- 4像素误差范围内扫描热力图
- 置信度加权的Y轴选择
- 确保滑块和缺口在同一水平线

#### 训练时强化
- 添加Y轴一致性损失
- 增加Y轴对齐的样本权重
- 监控Y轴一致性指标

### 6.3 增量改进策略

#### 持续优化流程
1. **收集失败案例**
   ```bash
   python scripts/analysis/collect_failures.py \
       --model src/checkpoints/finetune_v1/best_model.pth \
       --threshold 5.0 \
       --output data/failure_cases
   ```

2. **重新标注**
   ```bash
   python scripts/annotation/annotate_failures.py \
       --input data/failure_cases \
       --output data/failure_cases_annotated
   ```

3. **增量微调**
   ```bash
   python scripts/training/incremental_finetune.py \
       --model src/checkpoints/finetune_v1/best_model.pth \
       --new_data data/failure_cases_annotated \
       --epochs 5
   ```

#### 版本管理
- 保留多个模型版本
- A/B测试不同版本
- 记录每个版本的性能指标

## 七、预期效果

### 7.1 性能提升目标

| 指标 | 基础模型 | 微调目标 | 提升幅度 |
|------|---------|---------|---------|
| 真实数据 hit@5px | 85% | 95%+ | +10% |
| 真实数据 hit@2px | 70% | 85%+ | +15% |
| 真实数据 MAE | 3.5px | <2.0px | -43% |
| Y轴一致性率 | 80% | 98%+ | +18% |
| 处理时间 | 45ms | <50ms | <+11% |

### 7.2 鲁棒性增强
- 更好处理真实场景的噪声
- 适应不同光照条件
- 处理模糊和变形
- 应对复杂背景干扰

### 7.3 Y轴一致性
- 确保滑块和缺口在同一水平线
- 减少因Y轴偏差导致的验证失败
- 提高整体通过率

## 八、后续优化方向

### 8.1 主动学习
```python
class ActiveLearning:
    """主动学习策略"""
    def select_samples(self, predictions, threshold=0.7):
        """选择低置信度样本进行标注"""
        low_confidence = predictions['confidence'] < threshold
        return predictions[low_confidence]
```

### 8.2 域适应
- 针对不同网站风格微调
- 建立多个专门模型
- 自动选择最佳模型

### 8.3 在线学习
- 部署后持续学习
- 从用户反馈中改进
- 自动更新模型权重

### 8.4 模型压缩
- 知识蒸馏到更小模型
- 量化和剪枝
- 边缘设备部署

## 九、风险管理

### 9.1 潜在风险
1. **数据偏差**：真实数据可能有偏向性
2. **标注质量**：人工标注可能有误差
3. **过拟合**：在少量数据上过度拟合
4. **性能退化**：在某些场景下性能下降

### 9.2 缓解措施
1. **多样化数据源**：收集不同来源的数据
2. **标注验证**：多人交叉验证标注
3. **正则化技术**：使用各种防过拟合方法
4. **回滚机制**：保留基础模型作为备份

## 十、实施时间表

| 阶段 | 任务 | 预计时间 | 里程碑 |
|------|------|---------|--------|
| 第1周 | 数据收集与标注 | 5天 | 1000张标注数据 |
| 第2周 | 数据预处理与验证 | 3天 | NPY格式数据集 |
| 第2周 | 第一阶段微调 | 2天 | 冻结骨干模型 |
| 第3周 | 第二阶段微调 | 3天 | 全网络微调模型 |
| 第3周 | 评估与优化 | 2天 | 性能报告 |
| 第4周 | 增量改进 | 3天 | 最终模型 |
| 第4周 | 部署准备 | 2天 | 生产就绪 |

## 附录A：相关文件清单

### 配置文件
- `config/finetune_config.yaml` - 微调训练配置
- `config/loss_finetune.yaml` - 微调损失函数配置
- `config/preprocessing_config.yaml` - 数据预处理配置

### 脚本文件
- `scripts/training/finetune.py` - 微调训练主脚本
- `scripts/data_generation/preprocess_real_data.py` - 真实数据预处理
- `scripts/annotation/annotate_captchas_matplotlib.py` - 数据标注工具
- `scripts/validation/evaluate_real_captchas.py` - 真实数据评估
- `scripts/analysis/compare_models.py` - 模型对比分析

### 数据目录
- `data/real_captchas/` - 原始真实数据
- `data/real_captchas/annotated/` - 标注后的数据
- `data/real_processed/` - 预处理后的NPY数据
- `data/failure_cases/` - 失败案例收集

### 模型目录
- `src/checkpoints/1.1.0/` - 基础模型
- `src/checkpoints/finetune_v1/` - 微调模型

## 附录B：常见问题解答

### Q1: 需要多少真实数据？
A: 建议至少500-1000张高质量标注数据。更多数据会带来更好效果，但边际收益递减。

### Q2: 微调会影响原有性能吗？
A: 使用混合训练策略可以最小化影响。建议保持70%合成数据，确保泛化能力。

### Q3: 如何处理不同网站的验证码？
A: 可以训练多个专门模型，或使用域适应技术。建议先从最常见的类型开始。

### Q4: Y轴对齐是否会影响X轴精度？
A: 新的对齐算法在4像素范围内扫描，能找到最佳X位置，不会显著影响X轴精度。

### Q5: 如何验证微调效果？
A: 使用独立的测试集，对比基础模型和微调模型的各项指标，特别关注目标场景的性能。

---

*文档版本: 1.0*  
*更新日期: 2024*  
*作者: AI Assistant*