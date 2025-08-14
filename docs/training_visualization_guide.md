# Lite-HRNet-18+LiteFPN 训练与可视化指南

## 📋 目录

- [模型架构概述](#模型架构概述)
- [训练目标与指标](#训练目标与指标)
- [训练配置](#训练配置)
- [训练流程](#训练流程)
- [模型保存策略](#模型保存策略)
- [可视化监控](#可视化监控)
- [TensorBoard 监控面板](#tensorboard-监控面板)
- [结果评估与分析](#结果评估与分析)
- [故障排查指南](#故障排查指南)

---

## 🏗️ 模型架构概述

### 模型规格

- **模型名称**: Lite-HRNet-18+LiteFPN
- **参数量**: 3.28M (目标3.5M，误差-6.1%)
- **模型大小**: 
  - FP32: 12.53 MB
  - FP16: 6.27 MB
  - INT8: 3.13 MB
- **推理速度**: 
  - CPU (i5-1240P): 12-14ms
  - GPU (RTX 3050): <3ms

### 架构组成

```
输入 [B, 4, 256, 512] (RGB + padding mask)
    ↓
Stem (Stage1): 20,480 参数 (0.6%)
    ↓
Stage2 (双分支): 100,864 参数 (3.1%)
    ↓
Stage3 (三分支): 512,416 参数 (15.6%)
    ↓
Stage4 (四分支): 1,701,568 参数 (51.8%)
    ↓
LiteFPN (Stage5): 653,318 参数 (19.9%)
    ↓
DualHead (Stage6): 296,200 参数 (9.0%)
    ↓
输出:
- heatmap_gap [B, 1, 64, 128]
- heatmap_slider [B, 1, 64, 128]
- offset_gap [B, 2, 64, 128]
- offset_slider [B, 2, 64, 128]
- angle [B, 2, 64, 128] (可选)
```

---

## 🎯 训练目标与指标

### 主要目标

- **MAE (Mean Absolute Error)**: < 5px
- **精度@5px**: > 98%
- **推理速度**: CPU < 15ms

### 评估指标体系

| 指标          | 目标值     | 计算方法                         | 重要性       |
| ----------- | ------- | ---------------------------- | --------- |
| **MAE**     | < 5px   | `mean(                       | pred - gt |
| **精度@5px**  | > 98%   | `sum(distance < 5) / total`  | 🔴 关键     |
| **精度@3px**  | > 85%   | `sum(distance < 3) / total`  | 🟡 重要     |
| **精度@10px** | > 99.5% | `sum(distance < 10) / total` | 🟢 参考     |
| **缺口MAE**   | < 5px   | 仅计算缺口位置误差                    | 🔴 关键     |
| **滑块MAE**   | < 5px   | 仅计算滑块位置误差                    | 🔴 关键     |
| **角度误差**    | < 2°    | 旋转缺口的角度误差                    | 🟢 参考     |

---

## ⚙️ 训练配置

### 超参数设置

```yaml
# config/training_config.yaml
model:
  name: "Lite-HRNet-18+LiteFPN"
  input_channels: 4  # RGB + padding mask

training:
  # 优化器配置
  optimizer:
    type: "AdamW"
    lr: 3e-4  # 基础学习率
    betas: [0.9, 0.999]
    eps: 1e-8
    weight_decay: 0.05

  # 学习率调度
  scheduler:
    type: "CosineAnnealingWarmRestarts"
    T_0: 10  # 第一个周期的epoch数
    T_mult: 2  # 周期倍增因子
    eta_min: 3e-6  # 最小学习率
    warmup_epochs: 5  # 预热epoch数
    warmup_lr: 1e-5  # 预热起始学习率

  # 训练参数
  batch_size: 16  # 基础批次大小
  num_epochs: 160
  gradient_clip: 1.0  # 梯度裁剪

  # EMA配置
  ema:
    enabled: true
    decay: 0.9997
    update_interval: 1

  # 数据增强
  augmentation:
    # 几何变换
    random_rotate: [-5, 5]  # 度
    random_scale: [0.9, 1.1]
    random_translate: [-10, 10]  # 像素

    # 颜色增强
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1

    # 噪声与模糊
    gaussian_noise: 0.01
    gaussian_blur: [0, 2]

    # 混淆增强（专门针对验证码）
    confusing_gap_prob: 0.3
    perlin_noise_prob: 0.2
    gap_rotation_prob: 0.1
    gap_highlight_prob: 0.1

loss:
  # 热力图损失 (Focal Loss)
  heatmap:
    type: "FocalLoss"
    alpha: 2
    beta: 4
    weight: 1.0

  # 偏移损失 (L1 Loss)
  offset:
    type: "L1Loss"
    weight: 1.0

  # 角度损失 (可选)
  angle:
    type: "L1Loss"
    weight: 0.5

validation:
  interval: 1  # 每个epoch验证一次
  save_best: true
  metrics: ["mae", "accuracy@5px", "accuracy@3px"]

checkpoints:
  save_dir: "checkpoints/1.1.0"
  save_interval: 1  # 每个epoch保存
  keep_last: 5  # 保留最近5个checkpoint
  save_best: true  # 保存最佳模型
```

### 批次大小自适应

```python
# 根据GPU内存自动调整批次大小
def get_adaptive_batch_size(gpu_memory_gb):
    if gpu_memory_gb >= 24:
        return 64
    elif gpu_memory_gb >= 16:
        return 32
    elif gpu_memory_gb >= 8:
        return 16
    else:
        return 8
```

---

## 🚀 训练流程

### 1. 数据准备阶段

```bash
# 生成验证码数据集
python scripts/generate_captchas.py \
    --num_images 2000 \
    --output_dir data/captchas \
    --enable_confusing_gaps \
    --enable_rotation \
    --enable_perlin_noise

# 划分数据集 (9:1)
python scripts/split_dataset.py \
    --input_dir data/captchas \
    --train_ratio 0.9 \
    --seed 42
```

### 2. 训练启动

```bash
# 基础训练
python scripts/training/train.py \
    --config config/training_config.yaml \
    --resume_from checkpoints/1.1.0/last.pth

# 多GPU训练
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/training/train.py \
    --config config/training_config.yaml \
    --distributed

# 混合精度训练 (节省内存，加速训练)
python scripts/training/train.py \
    --config config/training_config.yaml \
    --amp \
    --amp_level O1
```

### 3. 训练监控

```bash
# 启动TensorBoard
tensorboard --logdir runs/1.1.0 --port 6006

# 实时查看训练日志
tail -f logs/training_$(date +%Y%m%d).log
```

---

## 💾 模型保存策略

### 保存路径结构

```
checkpoints/1.1.0/
├── epoch_001.pth      # 每个epoch的checkpoint
├── epoch_002.pth
├── ...
├── epoch_160.pth
├── best_model.pth     # 测试集最佳模型
├── last.pth          # 最新模型（用于恢复训练）
└── config.yaml       # 训练配置备份
```

### Checkpoint 内容

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'ema_state_dict': ema.state_dict() if use_ema else None,
    'metrics': {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_mae': val_mae,
        'val_acc_5px': val_acc_5px,
        'val_acc_3px': val_acc_3px,
        'best_mae': best_mae,
    },
    'config': config,
    'random_state': {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'random': random.getstate(),
    }
}
```

### 保存触发条件

1. **每个Epoch结束**: 保存为 `epoch_XXX.pth`
2. **验证集最佳**: MAE最低时保存为 `best_model.pth`
3. **训练中断**: 自动保存为 `last.pth`
4. **定期保存**: 每30分钟自动保存一次

---

## 📊 可视化监控

### 1. 训练曲线可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_theme(style="darkgrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Loss曲线
axes[0, 0].plot(epochs, train_losses, 'b-', label='训练集Loss', linewidth=2)
axes[0, 0].plot(epochs, val_losses, 'r-', label='验证集Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('损失函数变化曲线')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. MAE曲线
axes[0, 1].plot(epochs, train_mae, 'b-', label='训练集MAE', linewidth=2)
axes[0, 1].plot(epochs, val_mae, 'r-', label='验证集MAE', linewidth=2)
axes[0, 1].axhline(y=5, color='g', linestyle='--', label='目标: 5px')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE (pixels)')
axes[0, 1].set_title('平均绝对误差变化曲线')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 精度曲线
axes[1, 0].plot(epochs, acc_3px, 'g-', label='精度@3px', linewidth=2)
axes[1, 0].plot(epochs, acc_5px, 'b-', label='精度@5px', linewidth=2)
axes[1, 0].plot(epochs, acc_10px, 'r-', label='精度@10px', linewidth=2)
axes[1, 0].axhline(y=98, color='k', linestyle='--', label='目标: 98%')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].set_title('不同阈值下的精度变化')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 学习率变化
axes[1, 1].plot(epochs, learning_rates, 'orange', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('学习率调度')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 2. 误差分布分析

```python
# 误差分布直方图
plt.figure(figsize=(12, 5))

# 缺口误差分布
plt.subplot(1, 2, 1)
plt.hist(gap_errors, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=5, color='r', linestyle='--', label='目标阈值: 5px')
plt.xlabel('误差 (pixels)')
plt.ylabel('概率密度')
plt.title(f'缺口位置误差分布\nMAE: {gap_mae:.2f}px')
plt.legend()

# 滑块误差分布
plt.subplot(1, 2, 2)
plt.hist(slider_errors, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
plt.axvline(x=5, color='r', linestyle='--', label='目标阈值: 5px')
plt.xlabel('误差 (pixels)')
plt.ylabel('概率密度')
plt.title(f'滑块位置误差分布\nMAE: {slider_mae:.2f}px')
plt.legend()

plt.tight_layout()
plt.savefig('results/error_distribution.png', dpi=300, bbox_inches='tight')
```

### 3. 混淆矩阵（针对困难样本）

```python
# 按误差等级分类的混淆矩阵
error_ranges = ['0-3px', '3-5px', '5-10px', '>10px']
confusion_matrix = np.array([
    [850, 50, 20, 5],   # 简单样本
    [30, 40, 10, 2],    # 混淆缺口
    [10, 8, 15, 3],     # 旋转缺口
    [2, 1, 3, 1]        # 噪声样本
])

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=error_ranges,
            yticklabels=['简单', '混淆缺口', '旋转缺口', '噪声样本'])
plt.title('不同样本类型的误差分布')
plt.xlabel('误差范围')
plt.ylabel('样本类型')
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
```

---

## 📈 TensorBoard 监控面板

### 1. Scalars（标量监控）

```python
# 在训练代码中添加
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'runs/1.1.0/exp_{timestamp}')

# 每个step记录
writer.add_scalar('Loss/train', train_loss, global_step)
writer.add_scalar('Loss/val', val_loss, global_step)
writer.add_scalar('Metrics/MAE', mae, global_step)
writer.add_scalar('Metrics/Accuracy_5px', acc_5px, global_step)
writer.add_scalar('Metrics/Accuracy_3px', acc_3px, global_step)
writer.add_scalar('Learning_Rate', current_lr, global_step)

# 分组记录
writer.add_scalars('Loss_Comparison', {
    'train': train_loss,
    'validation': val_loss
}, global_step)

writer.add_scalars('MAE_Components', {
    'gap': gap_mae,
    'slider': slider_mae,
    'total': total_mae
}, global_step)
```

### 2. Images（图像监控）

```python
# 可视化预测结果
def visualize_predictions(images, targets, predictions, writer, step):
    batch_size = min(4, images.size(0))  # 显示4个样本

    fig, axes = plt.subplots(batch_size, 3, figsize=(12, batch_size*3))

    for i in range(batch_size):
        # 原图
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        axes[i, 0].imshow(img[:, :, :3])  # 只显示RGB通道
        axes[i, 0].set_title('输入图像')

        # 真实热力图
        gt_heatmap = targets['heatmap_gap'][i].cpu().numpy()
        axes[i, 1].imshow(gt_heatmap[0], cmap='hot')
        axes[i, 1].set_title('真实热力图')

        # 预测热力图
        pred_heatmap = predictions['heatmap_gap'][i].cpu().numpy()
        axes[i, 2].imshow(pred_heatmap[0], cmap='hot')
        axes[i, 2].set_title('预测热力图')

        # 添加坐标点
        gt_coord = targets['gap_coords'][i].cpu().numpy()
        pred_coord = predictions['gap_coords'][i].cpu().numpy()

        axes[i, 0].plot(gt_coord[0], gt_coord[1], 'go', markersize=10, label='真实')
        axes[i, 0].plot(pred_coord[0], pred_coord[1], 'rx', markersize=10, label='预测')
        axes[i, 0].legend()

    plt.tight_layout()
    writer.add_figure('Predictions', fig, step)
    plt.close()

# 每10个epoch可视化一次
if epoch % 10 == 0:
    visualize_predictions(images, targets, outputs, writer, epoch)
```

### 3. Histograms（直方图监控）

```python
# 监控权重和梯度分布
for name, param in model.named_parameters():
    if param.requires_grad:
        writer.add_histogram(f'Weights/{name}', param.data, epoch)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

# 监控激活值分布
def hook_fn(module, input, output):
    writer.add_histogram(f'Activations/{module.__class__.__name__}', 
                         output.detach(), epoch)

# 注册钩子
for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        module.register_forward_hook(hook_fn)
```

### 4. Graph（模型结构）

```python
# 可视化模型结构
dummy_input = torch.randn(1, 4, 256, 512).to(device)
writer.add_graph(model, dummy_input)
```

### 5. Custom Scalars（自定义面板）

```python
# 创建自定义监控面板
layout = {
    "训练监控": {
        "损失函数": ["Multiline", ["Loss/train", "Loss/val"]],
        "学习率": ["Multiline", ["Learning_Rate"]],
    },
    "性能指标": {
        "MAE": ["Multiline", ["Metrics/MAE", "MAE_Components/gap", "MAE_Components/slider"]],
        "精度": ["Multiline", ["Metrics/Accuracy_3px", "Metrics/Accuracy_5px"]],
    },
    "模型分析": {
        "梯度范数": ["Multiline", ["Gradients/norm"]],
        "权重更新": ["Multiline", ["Weights/update_ratio"]],
    }
}

writer.add_custom_scalars(layout)
```

### 6. PR Curves（精确率-召回率曲线）

```python
# 计算不同阈值下的PR曲线
def compute_pr_curve(predictions, targets, thresholds):
    precisions = []
    recalls = []

    for threshold in thresholds:
        pred_positive = predictions > threshold
        true_positive = pred_positive & targets

        precision = true_positive.sum() / pred_positive.sum()
        recall = true_positive.sum() / targets.sum()

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

# 添加到TensorBoard
writer.add_pr_curve('PR_Curve', targets, predictions, epoch)
```

---

## 🎯 结果评估与分析

### 1. 最终评估指标

```python
# 测试集完整评估
def evaluate_final_model(model, test_loader):
    results = {
        'mae': {'gap': [], 'slider': [], 'total': []},
        'accuracy': {'3px': 0, '5px': 0, '10px': 0},
        'error_distribution': [],
        'inference_time': [],
        'difficult_samples': []
    }

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # 预测
            outputs = model(batch['image'])
            predictions = model.decode_predictions(outputs)

            # 计算误差
            gap_error = torch.abs(predictions['gap_coords'] - batch['gap_coords'])
            slider_error = torch.abs(predictions['slider_coords'] - batch['slider_coords'])

            # 统计结果
            results['mae']['gap'].extend(gap_error.mean(dim=1).cpu().numpy())
            results['mae']['slider'].extend(slider_error.mean(dim=1).cpu().numpy())

            # 困难样本识别
            if batch.get('has_confusing_gap', False):
                results['difficult_samples'].append({
                    'type': 'confusing_gap',
                    'error': gap_error.mean().item()
                })

    return results
```

### 2. 结果报告生成

```markdown
# 模型评估报告

## 基础指标
- **总体MAE**: 3.85 px ✅ (目标: <5px)
- **精度@5px**: 98.7% ✅ (目标: >98%)
- **精度@3px**: 87.3% ✅ (目标: >85%)
- **推理速度**: 12.5ms (CPU) ✅

## 分项指标
| 组件 | MAE | 精度@5px | 精度@3px |
|-----|-----|----------|----------|
| 缺口 | 3.92px | 98.5% | 86.8% |
| 滑块 | 3.78px | 98.9% | 87.8% |

## 困难样本表现
| 样本类型 | 占比 | MAE | 精度@5px |
|---------|-----|-----|----------|
| 普通样本 | 70% | 3.2px | 99.5% |
| 混淆缺口 | 15% | 5.1px | 95.2% |
| 旋转缺口 | 10% | 4.8px | 96.8% |
| 噪声样本 | 5% | 6.2px | 92.1% |

## 训练过程
- **总训练时间**: 8h 35min
- **最佳Epoch**: 142/160
- **最终Loss**: 0.0234
- **过拟合程度**: 轻微（train/val loss差异<10%）
```

### 3. 失败案例分析

```python
# 收集失败案例
def analyze_failures(predictions, targets, threshold=10):
    failures = []

    for i in range(len(predictions)):
        error = np.abs(predictions[i] - targets[i]).mean()
        if error > threshold:
            failures.append({
                'index': i,
                'error': error,
                'prediction': predictions[i],
                'target': targets[i],
                'image_path': image_paths[i]
            })

    # 分析失败原因
    failure_reasons = {
        'extreme_rotation': 0,
        'heavy_noise': 0,
        'multiple_gaps': 0,
        'edge_position': 0,
        'low_contrast': 0
    }

    for failure in failures:
        # 分析具体原因...
        pass

    return failures, failure_reasons
```

---

## 🔧 故障排查指南

### 常见问题及解决方案

| 问题          | 可能原因     | 解决方案                |
| ----------- | -------- | ------------------- |
| **Loss不下降** | 学习率过小/过大 | 调整学习率，检查数据标签        |
| **过拟合严重**   | 数据增强不足   | 增加数据增强，使用dropout    |
| **MAE突然增大** | 梯度爆炸     | 减小学习率，增加梯度裁剪        |
| **显存溢出**    | 批次过大     | 减小batch_size，使用梯度累积 |
| **训练速度慢**   | 数据加载瓶颈   | 增加num_workers，使用缓存  |
| **精度提升停滞**  | 达到模型容量上限 | 增加模型复杂度，改进架构        |

### 训练日志监控

```bash
# 实时监控关键指标
watch -n 1 'tail -20 logs/training.log | grep -E "Loss|MAE|Acc"'

# 检查GPU使用率
nvidia-smi -l 1

# 监控内存使用
htop

# 检查数据加载速度
python scripts/benchmark_dataloader.py
```

### 调试技巧

1. **梯度检查**
   
   ```python
   # 检查梯度范数
   for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100:
            print(f"Warning: Large gradient in {name}: {grad_norm}")
   ```

2. **数据验证**
   
   ```python
   # 验证数据标签
   def validate_dataset(dataset):
    errors = []
    for i, sample in enumerate(dataset):
        if not (0 <= sample['gap_x'] <= 512):
            errors.append(f"Sample {i}: Invalid gap_x")
        if not (0 <= sample['gap_y'] <= 256):
            errors.append(f"Sample {i}: Invalid gap_y")
    return errors
   ```

3. **模型输出检查**
   
   ```python
   # 检查输出范围
   outputs = model(batch)
   assert outputs['heatmap_gap'].min() >= 0
   assert outputs['heatmap_gap'].max() <= 1
   assert outputs['offset_gap'].abs().max() <= 0.5
   ```