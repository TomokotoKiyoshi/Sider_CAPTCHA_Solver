#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMN-R3-FP 模型训练脚本
保存checkpoint到: src/checkpoints/v1.1.0/
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import cv2

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.pmn_r3_fp import PMN_R3_FP
from src.models.loss import PMN_R3_FP_Loss


class CaptchaDataset(Dataset):
    """滑块验证码数据集"""
    
    def __init__(self, annotations, data_dir, transform=None, is_train=True):
        """
        Args:
            annotations: 标注列表
            data_dir: 数据根目录
            transform: 数据增强
            is_train: 是否训练模式
        """
        self.annotations = annotations
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """加载单个样本"""
        ann = self.annotations[idx]
        
        # 加载验证码图片
        img_path = self.data_dir / "captchas" / ann['filename']
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Cannot load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 提取坐标
        gap_x = ann['bg_x']
        gap_y = ann['bg_y']
        slider_x = ann['sd_x']
        slider_y = ann['sd_y']
        
        # 获取形状和大小
        shape = ann.get('shape', 'circle')
        size = ann.get('size', 40)
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 创建目标
        target = {
            'gap_coord': torch.tensor([gap_x, gap_y], dtype=torch.float32),
            'slider_coord': torch.tensor([slider_x, slider_y], dtype=torch.float32),
            'shape': shape,
            'size': size,
            'filename': ann['filename']
        }
        
        # 数据增强（如果需要）
        if self.transform and self.is_train:
            image, target = self.apply_augmentation(image, target)
        
        return image, target
    
    def apply_augmentation(self, image, target):
        """应用数据增强"""
        # 随机颜色抖动
        if torch.rand(1) > 0.5:
            brightness = 0.2 * (torch.rand(1) - 0.5)
            image = torch.clamp(image + brightness, 0, 1)
        
        # 随机噪声
        if torch.rand(1) > 0.5:
            noise = torch.randn_like(image) * 0.01
            image = torch.clamp(image + noise, 0, 1)
        
        return image, target


class Trainer:
    """PMN-R3-FP 训练器"""
    
    def __init__(self, config):
        """初始化训练器"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 处理配置 - 支持字典和对象两种格式
        if isinstance(config, dict):
            # 字典格式
            checkpoint_dir = config.get('checkpoint_dir', 'src/checkpoints/v1.1.0')
            model_config = config.get('model', {})
            loss_weights = config.get('loss_weights', {})
            learning_rate = config.get('learning_rate', 1e-4)
            weight_decay = config.get('weight_decay', 1e-4)
            num_epochs = config.get('num_epochs', 100)
            scheduler_eta_min = config.get('scheduler', {}).get('eta_min', 1e-6)
            use_amp = config.get('use_amp', True)
        else:
            # TrainingConfig对象或其他对象
            checkpoint_dir = getattr(config, 'checkpoint_dir', 'src/checkpoints/v1.1.0')
            model_config = {}  # PMN_R3_FP使用默认配置
            loss_weights = {}  # 使用默认权重
            learning_rate = config.optimizer.lr if hasattr(config, 'optimizer') else 1e-4
            weight_decay = config.optimizer.weight_decay if hasattr(config, 'optimizer') else 1e-4
            num_epochs = getattr(config, 'epochs', 100)
            scheduler_eta_min = 1e-6
            use_amp = getattr(config, 'mixed_precision', True)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        self.model = PMN_R3_FP(model_config).to(self.device)
        
        # 多GPU支持
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # 损失函数
        self.criterion = PMN_R3_FP_Loss(loss_weights)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=scheduler_eta_min
        )
        
        # 混合精度训练
        self.scaler = GradScaler() if use_amp else None
        
        # 保存一些配置值供后续使用
        self.num_epochs = num_epochs
        self.keep_last_n = config.get('keep_last_n', 10) if isinstance(config, dict) else 10
        self.early_stopping_patience = (config.get('early_stopping', {}).get('patience', 20) 
                                       if isinstance(config, dict) 
                                       else getattr(config, 'early_stopping_patience', 20))
        
        # 最佳指标
        self.best_mae = float('inf')
        self.start_epoch = 0
        
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch [{epoch}/{self.num_epochs}]')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # 准备目标
            gap_coords = torch.stack([t['gap_coord'] for t in targets]).to(self.device)
            slider_coords = torch.stack([t['slider_coord'] for t in targets]).to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, gap_coords, slider_coords)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, gap_coords, slider_coords)
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # 计算MAE
            with torch.no_grad():
                gap_pred = outputs['gap_heatmap'].argmax(dim=(2,3))  # 简化版本
                slider_pred = outputs['piece_heatmap'].argmax(dim=(2,3))
                
                # 将索引转换为坐标（需要根据实际输出调整）
                gap_pred_coords = self.index_to_coords(gap_pred, scale=4)
                slider_pred_coords = self.index_to_coords(slider_pred, scale=4)
                
                gap_mae = (gap_pred_coords - gap_coords).abs().mean()
                slider_mae = (slider_pred_coords - slider_coords).abs().mean()
                mae = (gap_mae + slider_mae) / 2
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae.item():.2f}px'
            })
        
        return total_loss / num_batches, total_mae / num_batches
    
    def validate(self, dataloader, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        hit_at_2px = 0
        num_samples = 0
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc='Validation'):
                images = images.to(self.device)
                gap_coords = torch.stack([t['gap_coord'] for t in targets]).to(self.device)
                slider_coords = torch.stack([t['slider_coord'] for t in targets]).to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, gap_coords, slider_coords)
                
                # 计算指标
                gap_pred = outputs['gap_heatmap'].argmax(dim=(2,3))
                slider_pred = outputs['piece_heatmap'].argmax(dim=(2,3))
                
                gap_pred_coords = self.index_to_coords(gap_pred, scale=4)
                slider_pred_coords = self.index_to_coords(slider_pred, scale=4)
                
                gap_error = (gap_pred_coords - gap_coords).abs()
                slider_error = (slider_pred_coords - slider_coords).abs()
                
                mae = (gap_error.mean() + slider_error.mean()) / 2
                
                # 计算命中率@2px
                gap_hit = (gap_error.max(dim=1)[0] <= 2).float()
                slider_hit = (slider_error.max(dim=1)[0] <= 2).float()
                hit = (gap_hit + slider_hit) / 2
                
                total_loss += loss.item() * images.size(0)
                total_mae += mae.item() * images.size(0)
                hit_at_2px += hit.sum().item()
                num_samples += images.size(0)
        
        avg_loss = total_loss / num_samples
        avg_mae = total_mae / num_samples
        avg_hit = hit_at_2px / num_samples * 100
        
        return avg_loss, avg_mae, avg_hit
    
    def index_to_coords(self, indices, scale=4):
        """将热力图索引转换为坐标"""
        # 简化实现，实际需要根据模型输出调整
        coords = indices.float() * scale
        return coords
    
    def save_checkpoint(self, epoch, val_mae, is_best):
        """保存checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_mae': self.best_mae,
            'val_mae': val_mae,
            'config': self.config
        }
        
        # 保存当前epoch
        epoch_path = self.checkpoint_dir / f'epoch_{epoch:02d}.pth'
        torch.save(state, epoch_path)
        print(f'  ✓ Saved: {epoch_path.name}')
        
        # 如果是最优模型，额外保存
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            shutil.copy(str(epoch_path), str(best_path))
            print(f'  ✓ New best! Saved: {best_path.name}')
        
        # 清理旧的checkpoint（保留最近N个）
        if self.keep_last_n > 0:
            self.cleanup_old_checkpoints(epoch)
    
    def cleanup_old_checkpoints(self, current_epoch):
        """清理旧的checkpoint"""
        keep_n = self.keep_last_n
        epoch_files = sorted(self.checkpoint_dir.glob('epoch_*.pth'))
        
        if len(epoch_files) > keep_n:
            for f in epoch_files[:-keep_n]:
                f.unlink()
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_mae = checkpoint.get('best_mae', float('inf'))
        print(f"Resumed from epoch {checkpoint['epoch']}, best MAE: {self.best_mae:.2f}px")
    
    def train(self, train_loader, val_loader, test_loader=None):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"Training PMN-R3-FP Model")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"{'='*60}\n")
        
        # 早停计数器
        patience_counter = 0
        patience = self.early_stopping_patience
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # 训练
            train_loss, train_mae = self.train_epoch(train_loader, epoch + 1)
            
            # 验证
            val_loss, val_mae, val_hit = self.validate(val_loader, epoch + 1)
            
            # 学习率调度
            self.scheduler.step()
            
            # 打印结果
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f}px")
            print(f"  Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}px, Hit@2px: {val_hit:.1f}%")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # 保存checkpoint
            is_best = val_mae < self.best_mae
            if is_best:
                self.best_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch + 1, val_mae, is_best)
            
            # 早停检查
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # 最终测试（如果有测试集）
        if test_loader:
            print(f"\n{'='*60}")
            print("Final Testing on Test Set")
            print(f"{'='*60}")
            
            # 加载最佳模型
            best_path = self.checkpoint_dir / 'best_model.pth'
            if best_path.exists():
                checkpoint = torch.load(best_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            test_loss, test_mae, test_hit = self.validate(test_loader, -1)
            print(f"Test Results:")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  MAE: {test_mae:.2f}px")
            print(f"  Hit@2px: {test_hit:.1f}%")
            
            # 保存测试结果
            results = {
                'test_loss': test_loss,
                'test_mae': test_mae,
                'test_hit_at_2px': test_hit,
                'best_val_mae': self.best_mae,
                'timestamp': datetime.now().isoformat()
            }
            
            results_path = self.checkpoint_dir / 'test_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Results saved to: {results_path}")


def load_data(config):
    """加载数据集"""
    # 处理配置 - 支持字典和对象两种格式
    if isinstance(config, dict):
        # 如果是字典格式
        split_file = Path(config.get('data', {}).get('split_file', 'data/splits/data_split.json'))
        data_dir = config.get('data', {}).get('data_dir', '.')
        batch_size = config.get('batch_size', 16)
        num_workers = config.get('num_workers', 4)
    else:
        # 如果是TrainingConfig对象或其他对象
        split_file = Path('data/splits/data_split.json')  # 使用默认路径
        data_dir = '.'
        if hasattr(config, 'data'):
            batch_size = config.data.batch_size if hasattr(config.data, 'batch_size') else 16
            num_workers = config.data.num_workers if hasattr(config.data, 'num_workers') else 4
        else:
            batch_size = getattr(config, 'batch_size', 16)
            num_workers = getattr(config, 'num_workers', 4)
    
    # 检查分割文件是否存在，如果不存在则尝试运行分割脚本
    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        print("Trying to generate split file...")
        
        # 尝试运行分割脚本
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, 'scripts/data_generation/run_split.py'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0:
                print(f"Failed to generate split file: {result.stderr}")
                # 使用备用路径
                alt_split_file = Path('data/metadata/all_annotations.json')
                if alt_split_file.exists():
                    print(f"Using annotations file directly: {alt_split_file}")
                    with open(alt_split_file, 'r') as f:
                        all_annotations = json.load(f)
                    
                    # 手动分割数据
                    n = len(all_annotations)
                    train_end = int(n * 0.8)
                    val_end = train_end + int(n * 0.1)
                    
                    split_data = {
                        'splits': {
                            'train': {'annotations': all_annotations[:train_end]},
                            'val': {'annotations': all_annotations[train_end:val_end]},
                            'test': {'annotations': all_annotations[val_end:]}
                        }
                    }
                else:
                    raise FileNotFoundError(f"Cannot find data files")
        except Exception as e:
            print(f"Error running split script: {e}")
            raise
    else:
        with open(split_file, 'r') as f:
            split_data = json.load(f)
    
    # 创建数据集
    train_dataset = CaptchaDataset(
        split_data['splits']['train']['annotations'],
        data_dir,
        transform=True,
        is_train=True
    )
    
    val_dataset = CaptchaDataset(
        split_data['splits']['val']['annotations'],
        data_dir,
        transform=False,
        is_train=False
    )
    
    test_dataset = CaptchaDataset(
        split_data['splits']['test']['annotations'],
        data_dir,
        transform=False,
        is_train=False
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train PMN-R3-FP Model')
    parser.add_argument('--config', type=str, default='config/training_config.py',
                       help='Path to training config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ids (e.g., "0,1,2")')
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 默认配置
    config = {
        # 数据配置
        'data': {
            'split_file': 'data/splits/data_split.json',
            'data_dir': '.'
        },
        
        # 模型配置
        'model': {
            'backbone_channels': [32, 64, 128, 256],
            'fpn_channels': 256,
            'num_classes': 1,
            'use_se2': True,
            'use_sdf': True
        },
        
        # 训练配置
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'use_amp': True,
        
        # 损失权重
        'loss_weights': {
            'region_obj': 1.0,      # Region objectness损失
            'region_ctr': 1.0,      # Region centerness损失
            'region_loc': 1.0,      # Region location损失
            'region_scale': 1.0,    # Region scale损失
            'shape_mask': 1.0,      # Shape掩码损失
            'shape_sdf': 0.5,       # Shape SDF损失
            'matching': 2.0,        # 匹配损失
            'geometry': 1.0         # 几何参数损失
        },
        
        # 学习率调度
        'scheduler': {
            'eta_min': 1e-6
        },
        
        # 早停
        'early_stopping': {
            'patience': 20,
            'min_delta': 0.001
        },
        
        # 保存设置
        'checkpoint_dir': 'src/checkpoints/v1.1.0',
        'keep_last_n': 10
    }
    
    # 如果有配置文件，加载并更新
    config_path = Path(args.config)
    if config_path.exists() and config_path.suffix == '.py':
        # 动态加载Python配置文件
        import importlib.util
        spec = importlib.util.spec_from_file_location("training_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        if hasattr(config_module, 'TrainingConfig'):
            # 直接使用TrainingConfig对象，而不是转换为字典
            config = config_module.TrainingConfig()
            print(f"Loaded TrainingConfig from {config_path}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    train_loader, val_loader, test_loader = load_data(config)
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(train_loader, val_loader, test_loader)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best model saved at: {Path(config['checkpoint_dir']) / 'best_model.pth'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()