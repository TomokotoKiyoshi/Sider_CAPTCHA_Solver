# -*- coding: utf-8 -*-
"""
训练工具函数
包含配置加载、检查点管理、日志等
"""
import os
import json
import shutil
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

import yaml
import torch
import numpy as np


def load_config(config_path: str) -> Dict:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建必要的目录
    Path(config['checkpoint']['save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['logging']['log_dir']).mkdir(parents=True, exist_ok=True)
    
    return config


def set_seed(seed: int, deterministic: bool = True):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
        deterministic: 是否启用确定性模式
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: Dict,
    epoch: int,
    checkpoint_dir: str,
    is_best: bool = False,
    keep_last_n: int = 10
):
    """
    保存检查点
    
    Args:
        state: 模型状态字典
        epoch: 当前epoch
        checkpoint_dir: 保存目录
        is_best: 是否为最佳模型
        keep_last_n: 保留最近N个检查点
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存当前epoch
    epoch_path = checkpoint_dir / f'epoch_{epoch:03d}.pth'
    torch.save(state, epoch_path)
    print(f'  ✓ Saved checkpoint: {epoch_path.name}')
    
    # 如果是最佳模型，额外保存
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        shutil.copy(str(epoch_path), str(best_path))
        print(f'  ✓ New best model saved: {best_path.name}')
    
    # 清理旧的检查点
    if keep_last_n > 0:
        cleanup_old_checkpoints(checkpoint_dir, keep_last_n)


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int):
    """
    清理旧的检查点，保留最近的N个
    
    Args:
        checkpoint_dir: 检查点目录
        keep_last_n: 保留的数量
    """
    epoch_files = sorted(checkpoint_dir.glob('epoch_*.pth'))
    
    if len(epoch_files) > keep_last_n:
        for f in epoch_files[:-keep_last_n]:
            f.unlink()
            print(f'  Removed old checkpoint: {f.name}')


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda'
) -> Dict:
    """
    加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
        
    Returns:
        检查点信息
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 加载优化器状态
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', float('inf')),
        'config': checkpoint.get('config', {})
    }
    
    print(f"Loaded checkpoint from epoch {info['epoch']}")
    if 'best_metric' in checkpoint:
        print(f"Best metric: {info['best_metric']:.4f}")
    
    return info


class EarlyStopping:
    """早停机制"""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        """
        Args:
            patience: 耐心值（多少个epoch没有改善就停止）
            min_delta: 最小改善量
            mode: 'min'或'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            value: 当前指标值
            
        Returns:
            是否触发早停
        """
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Logger:
    """简单的日志记录器"""
    
    def __init__(self, log_dir: str, experiment_name: str = None):
        """
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f"{experiment_name}_{timestamp}" if experiment_name else timestamp
        self.log_file = self.log_dir / f"{name}.log"
        
        # 保存指标历史
        self.history = {
            'train': [],
            'val': [],
            'test': []
        }
    
    def log(self, message: str, print_msg: bool = True):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
        
        # 打印到控制台
        if print_msg:
            print(message)
    
    def log_metrics(self, phase: str, epoch: int, metrics: Dict[str, float]):
        """记录指标"""
        # 添加epoch信息
        metrics['epoch'] = epoch
        
        # 保存到历史
        self.history[phase].append(metrics)
        
        # 格式化输出
        metric_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items() if k != 'epoch'])
        self.log(f"[{phase.upper()}] Epoch {epoch}: {metric_str}")
    
    def save_history(self):
        """保存训练历史"""
        history_file = self.log_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def get_optimizer(config: Dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        config: 配置字典
        model: 模型
        
    Returns:
        优化器
    """
    opt_config = config['optimizer']
    name = opt_config['name'].lower()
    
    if name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=opt_config.get('betas', [0.9, 0.999]),
            eps=opt_config.get('eps', 1e-8)
        )
    elif name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=opt_config.get('betas', [0.9, 0.999]),
            eps=opt_config.get('eps', 1e-8)
        )
    elif name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizer


def get_scheduler(config: Dict, optimizer: torch.optim.Optimizer):
    """
    创建学习率调度器
    
    Args:
        config: 配置字典
        optimizer: 优化器
        
    Returns:
        调度器
    """
    sch_config = config['scheduler']
    name = sch_config['name'].lower()
    
    if name == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_config['T_max'],
            eta_min=sch_config.get('eta_min', 1e-6)
        )
    elif name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_config['step_size'],
            gamma=sch_config.get('gamma', 0.1)
        )
    elif name == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sch_config['milestones'],
            gamma=sch_config.get('gamma', 0.1)
        )
    elif name == 'reducelronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sch_config.get('mode', 'min'),
            factor=sch_config.get('factor', 0.5),
            patience=sch_config.get('patience', 10),
            min_lr=sch_config.get('min_lr', 1e-7)
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")
    
    return scheduler