# Lite-HRNet-18+LiteFPN 训练系统架构设计

## 📋 目录

- [系统架构概览](#系统架构概览)
- [核心模块设计](#核心模块设计)
- [训练脚本实现](#训练脚本实现)
- [验证评估系统](#验证评估系统)
- [可视化监控系统](#可视化监控系统)
- [配置管理系统](#配置管理系统)
- [性能优化策略](#性能优化策略)

---

## 🏗️ 系统架构概览

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     训练系统主控制器                           │
│                  scripts/training/train.py                   │
└────────┬────────────────────────────────────────┬───────────┘
         │                                        │
    ┌────▼────────┐                      ┌───────▼──────────┐
    │  配置管理器   │                      │   数据管道系统     │
    │ ConfigManager│                      │ DataPipeline     │
    └─────────────┘                      └──────────────────┘
         │                                        │
    ┌────▼────────────────────────────────────────▼───────────┐
    │                     训练循环控制器                         │
    │                    TrainingEngine                        │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
    │  │ Optimizer│  │Scheduler │  │   EMA    │  │  AMP    ││
    │  └──────────┘  └──────────┘  └──────────┘  └─────────┘│
    └────────┬────────────────────────────────────┬───────────┘
             │                                    │
    ┌────────▼──────────┐              ┌─────────▼──────────┐
    │   验证评估系统      │              │   可视化监控系统     │
    │   Validator       │              │   Visualizer        │
    │ ┌──────────────┐ │              │ ┌────────────────┐ │
    │ │ Metrics      │ │              │ │  TensorBoard   │ │
    │ │ EarlyStopping│ │              │ │  Logging       │ │
    │ │ Checkpointing│ │              │ │  Profiling     │ │
    │ └──────────────┘ │              │ └────────────────┘ │
    └───────────────────┘              └────────────────────┘
```

### 核心设计原则

1. **模块化设计**: 每个功能独立成模块，便于维护和扩展
2. **配置驱动**: 所有参数通过YAML配置文件管理
3. **容错性**: 支持训练中断恢复和异常处理
4. **可扩展性**: 易于添加新的度量指标和可视化功能
5. **性能优化**: 支持混合精度、数据并行和内存优化

---

## 🔧 核心模块设计

### 1. 配置管理器 (ConfigManager)

```python
# scripts/training/config_manager.py
import yaml
from pathlib import Path
from typing import Dict, Any
import torch

class ConfigManager:
    """配置管理器 - 负责加载、验证和管理所有配置参数"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._setup_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self):
        """验证配置完整性"""
        required_keys = ['model', 'optimizer', 'sched', 'train', 'eval']
        for key in required_keys:
            assert key in self.config, f"Missing required config: {key}"
    
    def _setup_paths(self):
        """创建必要的目录结构"""
        paths = [
            Path(self.config['checkpoints']['save_dir']),
            Path(self.config['logging']['log_dir']),
            Path(self.config['logging']['tensorboard_dir'])
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_device(self) -> torch.device:
        """获取训练设备"""
        device_str = self.config['hardware']['device']
        if device_str == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
```

### 2. 数据管道系统 (DataPipeline)

```python
# scripts/training/data_pipeline.py
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
import numpy as np
from pathlib import Path

class CaptchaDataset(Dataset):
    """滑块验证码数据集"""
    
    def __init__(self, data_dir: str, mode: str = 'train'):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """加载数据样本"""
        # 实现数据加载逻辑
        pass
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """返回单个样本"""
        sample = self.samples[idx]
        # 返回格式：
        # {
        #     'image': tensor,         # [4, 256, 512]
        #     'gap_coords': tensor,    # [2]
        #     'slider_coords': tensor, # [2]
        #     'has_rotation': bool,
        #     'has_noise': bool
        # }
        return sample

class DataPipeline:
    """数据管道管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.train_loader = None
        self.val_loader = None
        
    def setup(self):
        """初始化数据加载器"""
        train_dataset = CaptchaDataset('data/train', mode='train')
        val_dataset = CaptchaDataset('data/val', mode='val')
        
        # 训练数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['num_workers'],
            pin_memory=self.config['train']['pin_memory'],
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # 验证数据加载器
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
```

### 3. 训练引擎 (TrainingEngine)

```python
# scripts/training/training_engine.py
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, Any
import time

class TrainingEngine:
    """训练引擎 - 核心训练循环管理"""
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 优化器设置
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # 混合精度训练
        self.use_amp = config['train']['amp'] == 'bf16'
        self.scaler = GradScaler() if self.use_amp else None
        
        # EMA设置
        self.ema = self._setup_ema() if config['optimizer']['ema_decay'] else None
        
        # 内存布局优化
        if config['train']['channels_last']:
            self.model = self.model.to(memory_format=torch.channels_last)
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """设置优化器"""
        opt_cfg = self.config['optimizer']
        return AdamW(
            self.model.parameters(),
            lr=opt_cfg['lr'],
            betas=opt_cfg['betas'],
            eps=opt_cfg['eps'],
            weight_decay=opt_cfg['weight_decay']
        )
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        sched_cfg = self.config['sched']
        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=sched_cfg['warmup_epochs'],
            T_mult=2,
            eta_min=sched_cfg['cosine_min_lr']
        )
    
    def _setup_ema(self):
        """设置指数移动平均"""
        from copy import deepcopy
        ema_model = deepcopy(self.model)
        ema_model.eval()
        return ema_model
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        metrics = {'loss': 0, 'gap_mae': 0, 'slider_mae': 0}
        
        for batch_idx, batch in enumerate(dataloader):
            # 数据传输到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播 (带混合精度)
            with autocast(enabled=self.use_amp, dtype=torch.bfloat16):
                outputs = self.model(batch['image'])
                loss = self._compute_loss(outputs, batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['optimizer']['clip_grad_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['optimizer']['clip_grad_norm']
                )
                self.optimizer.step()
            
            # 更新EMA
            if self.ema:
                self._update_ema()
            
            # 记录指标
            metrics['loss'] += loss.item()
            
        # 平均指标
        num_batches = len(dataloader)
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return metrics
    
    def _compute_loss(self, outputs: Dict, targets: Dict) -> torch.Tensor:
        """计算损失函数"""
        # 实现CenterNet损失
        pass
    
    def _update_ema(self):
        """更新EMA模型"""
        decay = self.config['optimizer']['ema_decay']
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), self.model.parameters()):
                ema_p.data.mul_(decay).add_(model_p.data, alpha=1-decay)
```

### 4. 验证评估系统 (Validator)

```python
# scripts/training/validator.py
import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np

class Validator:
    """验证评估系统"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.best_metric = -float('inf')
        self.patience_counter = 0
        self.metrics_history = []
        
        # 第二道防护指标
        if 'second_guard' in config['eval']['early_stopping']:
            guard_cfg = config['eval']['early_stopping']['second_guard']
            self.second_guard_metric = guard_cfg['metric']
            self.second_guard_mode = guard_cfg['mode']
            self.second_guard_min_delta = guard_cfg['min_delta']
            self.second_guard_best = float('inf') if guard_cfg['mode'] == 'min' else -float('inf')
        else:
            self.second_guard_metric = None
        
    def validate(self, model: nn.Module, dataloader, epoch: int) -> Dict[str, float]:
        """执行验证"""
        model.eval()
        
        # 初始化指标收集器
        mae_gap_list = []
        mae_slider_list = []
        hit_le_1px = 0
        hit_le_2px = 0
        hit_le_5px = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = model(batch['image'])
                predictions = model.decode_predictions(outputs)
                
                # 计算误差
                gap_error = torch.abs(
                    predictions['gap_coords'] - batch['gap_coords']
                ).mean(dim=1)  # [B]
                
                slider_error = torch.abs(
                    predictions['slider_coords'] - batch['slider_coords']
                ).mean(dim=1)  # [B]
                
                # 收集MAE
                mae_gap_list.extend(gap_error.cpu().numpy())
                mae_slider_list.extend(slider_error.cpu().numpy())
                
                # 计算命中率
                total_error = (gap_error + slider_error) / 2
                hit_le_1px += (total_error <= 1).sum().item()
                hit_le_2px += (total_error <= 2).sum().item()
                hit_le_5px += (total_error <= 5).sum().item()
                total_samples += batch['image'].size(0)
        
        # 汇总指标
        metrics = {
            'mae_px': np.mean(mae_gap_list + mae_slider_list),
            'rmse_px': np.sqrt(np.mean(np.square(mae_gap_list + mae_slider_list))),
            'hit_le_1px': hit_le_1px / total_samples * 100,
            'hit_le_2px': hit_le_2px / total_samples * 100,
            'hit_le_5px': hit_le_5px / total_samples * 100,
            'gap_mae': np.mean(mae_gap_list),
            'slider_mae': np.mean(mae_slider_list)
        }
        
        # 保存指标历史
        self.metrics_history.append(metrics)
        
        # 早停检查
        select_metric = metrics[self.config['eval']['select_by']]
        if self._check_early_stopping(select_metric, epoch):
            metrics['early_stop'] = True
        
        return metrics
    
    def _check_early_stopping(self, metric: float, epoch: int) -> bool:
        """检查早停条件"""
        cfg = self.config['eval']['early_stopping']
        
        # 未达到最小训练轮数
        if epoch < cfg['min_epochs']:
            return False
        
        # 主指标检查（hit_le_5px）
        has_primary_improvement = False
        if metric > self.best_metric:
            self.best_metric = metric
            self.patience_counter = 0
            has_primary_improvement = True
        else:
            self.patience_counter += 1
        
        # 第二道防护检查（mae_px）
        if self.second_guard_metric and epoch >= cfg['min_epochs']:
            current_guard_metric = self.metrics_history[-1][self.second_guard_metric]
            
            if self.second_guard_mode == 'min':
                # 对于mae_px，越小越好
                improvement = self.second_guard_best - current_guard_metric
                if improvement > self.second_guard_min_delta:
                    self.second_guard_best = current_guard_metric
                    # 第二指标有显著改善，重置耐心计数
                    self.patience_counter = max(0, self.patience_counter - 3)
                    print(f"Second guard: {self.second_guard_metric} improved by {improvement:.3f}px")
            else:
                # 对于其他指标，越大越好
                improvement = current_guard_metric - self.second_guard_best
                if improvement > self.second_guard_min_delta:
                    self.second_guard_best = current_guard_metric
                    self.patience_counter = max(0, self.patience_counter - 3)
                    print(f"Second guard: {self.second_guard_metric} improved by {improvement:.3f}")
        
        # 检查耐心值
        if self.patience_counter >= cfg['patience']:
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best hit@5px: {self.best_metric:.2f}%")
            if self.second_guard_metric:
                print(f"Best {self.second_guard_metric}: {self.second_guard_best:.3f}")
            return True
        
        return False
```

### 5. 可视化监控系统 (Visualizer)

```python
# scripts/training/visualizer.py
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from typing import Dict, Any
from pathlib import Path

class Visualizer:
    """可视化监控系统"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.writer = SummaryWriter(config['logging']['tensorboard_dir'])
        self.log_dir = Path(config['logging']['log_dir'])
        
    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """记录标量指标"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{key}', value, step)
    
    def log_learning_rate(self, lr: float, step: int):
        """记录学习率"""
        self.writer.add_scalar('lr/current', lr, step)
    
    def log_histograms(self, model: nn.Module, step: int):
        """记录权重和梯度直方图"""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # 权重直方图
                self.writer.add_histogram(
                    f'hist/weight_{name}', 
                    param.data, 
                    step
                )
                # 梯度直方图
                self.writer.add_histogram(
                    f'hist/grad_{name}', 
                    param.grad, 
                    step
                )
    
    def log_predictions(self, images, predictions, targets, step: int):
        """可视化预测结果"""
        # 选择前N个样本
        num_vis = min(4, images.size(0))
        
        for i in range(num_vis):
            # 绘制预测和真实位置
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)[:, :, :3]  # 只取RGB
            
            # 真实位置（绿色）
            gt_gap = targets['gap_coords'][i].cpu().numpy()
            cv2.circle(img, tuple(gt_gap.astype(int)), 5, (0, 255, 0), -1)
            
            # 预测位置（红色）
            pred_gap = predictions['gap_coords'][i].cpu().numpy()
            cv2.circle(img, tuple(pred_gap.astype(int)), 5, (255, 0, 0), -1)
            
            # 添加到TensorBoard
            self.writer.add_image(
                f'vis/overlay_gap_{i}', 
                img.transpose(2, 0, 1), 
                step
            )
    
    def log_failure_cases(self, failures: List[Dict], step: int):
        """记录失败案例"""
        for i, failure in enumerate(failures[:self.config['eval']['vis_fail_k']]):
            # 可视化失败案例
            self.writer.add_text(
                f'failures/case_{i}',
                f"Error: {failure['error']:.2f}px",
                step
            )
    
    def close(self):
        """关闭writer"""
        self.writer.close()
```

---

## 📝 训练脚本实现

### 主训练脚本

```python
# scripts/training/train.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lite-HRNet-18+LiteFPN 训练主脚本
"""
import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import create_lite_hrnet_18_fpn
from config_manager import ConfigManager
from data_pipeline import DataPipeline
from training_engine import TrainingEngine
from validator import Validator
from visualizer import Visualizer

def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                       default='config/training_config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # 初始化配置
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    device = config_manager.get_device()
    
    # 初始化模型
    model = create_lite_hrnet_18_fpn(config['model'])
    
    # 初始化数据管道
    data_pipeline = DataPipeline(config)
    data_pipeline.setup()
    
    # 初始化训练组件
    engine = TrainingEngine(model, config, device)
    validator = Validator(config, device)
    visualizer = Visualizer(config)
    
    # 恢复训练（如果需要）
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # 训练循环
    for epoch in range(start_epoch, config['sched']['epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config['sched']['epochs']}")
        print(f"{'='*50}")
        
        # 训练阶段
        train_metrics = engine.train_epoch(
            data_pipeline.train_loader, 
            epoch
        )
        
        # 验证阶段
        val_metrics = validator.validate(
            model, 
            data_pipeline.val_loader, 
            epoch
        )
        
        # 记录到TensorBoard
        visualizer.log_scalars(train_metrics, epoch, 'train')
        visualizer.log_scalars(val_metrics, epoch, 'val')
        visualizer.log_learning_rate(
            engine.scheduler.get_last_lr()[0], 
            epoch
        )
        
        # 更新学习率
        engine.scheduler.step()
        
        # 保存检查点
        if epoch % config['checkpoints']['save_interval'] == 0:
            save_checkpoint(model, engine, epoch, config)
        
        # 保存最佳模型
        if val_metrics[config['eval']['select_by']] == validator.best_metric:
            save_best_model(model, config)
        
        # 早停检查
        if val_metrics.get('early_stop', False):
            print("Early stopping triggered!")
            break
        
        # 打印进度
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val MAE: {val_metrics['mae_px']:.2f}px")
        print(f"Val hit@5px: {val_metrics['hit_le_5px']:.2f}%")
    
    visualizer.close()
    print("Training completed!")

def save_checkpoint(model, engine, epoch, config):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': engine.optimizer.state_dict(),
        'scheduler_state_dict': engine.scheduler.state_dict(),
    }
    
    # 保存epoch检查点
    path = Path(config['checkpoints']['save_dir']) / f"epoch_{epoch:03d}.pth"
    torch.save(checkpoint, path)
    
    # 保存最新检查点
    latest_path = Path(config['checkpoints']['save_dir']) / "last.pth"
    torch.save(checkpoint, latest_path)

def save_best_model(model, config):
    """保存最佳模型"""
    path = Path(config['checkpoints']['save_dir']) / "best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, path)
    print(f"Saved best model to {path}")

if __name__ == "__main__":
    main()
```

---

## 🚀 启动命令

### 基础训练
```bash
python scripts/training/train.py --config config/training_config.yaml
```

### 恢复训练
```bash
python scripts/training/train.py \
    --config config/training_config.yaml \
    --resume checkpoints/1.1.0/last.pth
```

### 监控训练
```bash
# 启动TensorBoard
tensorboard --logdir logs/tensorboard/1.1.0 --port 6006

# 实时查看日志
tail -f logs/log-1.1.0/training.log
```

---

## 📊 性能优化策略

### 1. 数据加载优化
- **预取因子**: `prefetch_factor=2`
- **持久化workers**: `persistent_workers=True`
- **固定内存**: `pin_memory=True`
- **并行workers**: `num_workers=24`

### 2. 训练优化
- **混合精度**: BFloat16自动混合精度
- **梯度累积**: 支持小显存大batch训练
- **内存布局**: Channels-last优化
- **梯度裁剪**: 防止梯度爆炸

### 3. 模型优化
- **EMA**: 指数移动平均提升稳定性
- **早停机制**: 防止过拟合
- **学习率调度**: Cosine退火with warm restarts

---

## 📈 监控指标

### 核心指标
- **MAE**: 平均绝对误差 < 2px
- **RMSE**: 均方根误差 < 3px
- **hit@1px**: 1像素内命中率 > 90%
- **hit@2px**: 2像素内命中率 > 95%
- **hit@5px**: 5像素内命中率 > 99%（选择指标）

### TensorBoard监控项
- 损失曲线：`loss/train`, `loss/val`
- 评估指标：`metrics/mae_px`, `metrics/hit_le_5px`
- 学习率：`lr/current`
- 可视化：`vis/overlay_gap`, `vis/overlay_slider`
- 直方图：`hist/grad_*`, `hist/weight_*`

---

## 🔧 故障排查

### 常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| OOM错误 | Batch size过大 | 减小batch_size或使用梯度累积 |
| Loss不下降 | 学习率不合适 | 调整学习率或检查数据 |
| 验证指标震荡 | 过拟合 | 增加数据增强或正则化 |
| 训练速度慢 | 数据加载瓶颈 | 增加num_workers |

### 调试命令
```bash
# GPU监控
watch -n 1 nvidia-smi

# 查看训练日志
grep -E "epoch|loss|mae" logs/log-1.1.0/training.log | tail -20

# 检查checkpoint
python -c "import torch; print(torch.load('checkpoints/1.1.0/last.pth').keys())"
```