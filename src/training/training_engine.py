#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练引擎 - 核心训练循环管理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
from copy import deepcopy
import numpy as np
import yaml
from pathlib import Path

# 导入完整的损失函数
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.models.loss_calculation.total_loss import create_total_loss


class TrainingEngine:
    """
    训练引擎 - 核心训练循环管理
    
    功能：
    1. 优化器和调度器管理
    2. 混合精度训练(BFloat16)
    3. EMA模型管理
    4. 训练循环执行
    5. 梯度裁剪
    6. 损失计算
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 config: Dict, 
                 device: torch.device):
        """
        初始化训练引擎
        
        Args:
            model: 模型
            config: 配置字典
            device: 训练设备
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logging.getLogger('TrainingEngine')
        
        # 加载损失函数配置并创建总损失函数
        self.loss_fn = self._setup_loss_function()
        
        # 优化器设置
        self.optimizer = self._setup_optimizer()
        
        # 学习率调度器
        self.scheduler = self._setup_scheduler()
        
        # 混合精度训练
        self.use_amp = config['train'].get('amp', 'none') == 'bf16'
        if self.use_amp:
            # BFloat16需要特殊处理
            self.scaler = GradScaler(enabled=False)  # BFloat16不需要GradScaler
            self.amp_dtype = torch.bfloat16
            self.logger.info("使用BFloat16混合精度训练")
        else:
            self.scaler = None
            self.amp_dtype = None
            self.logger.info("不使用混合精度训练")
        
        # EMA设置
        self.ema_decay = config['optimizer'].get('ema_decay', 0)
        if self.ema_decay > 0:
            self.ema_model = self._setup_ema()
            self.logger.info(f"启用EMA，decay={self.ema_decay}")
        else:
            self.ema_model = None
            self.logger.info("未启用EMA")
        
        # 内存布局优化
        if config['train'].get('channels_last', False):
            self.model = self.model.to(memory_format=torch.channels_last)
            self.logger.info("使用channels_last内存布局")
        
        # 梯度裁剪
        self.clip_grad_norm = config['optimizer'].get('clip_grad_norm', 1.0)
        
        # 梯度累积
        self.gradient_accumulation_steps = config['train'].get('gradient_accumulation_steps', 1)
        self.accumulation_counter = 0
        
        # 训练统计
        self.global_step = 0
        self.epoch = 0
    
    def _setup_loss_function(self):
        """设置损失函数"""
        # 加载损失函数配置
        loss_config_path = project_root / 'config' / 'loss.yaml'
        if not loss_config_path.exists():
            # 使用默认配置
            self.logger.warning(f"损失配置文件不存在: {loss_config_path}，使用默认配置")
            loss_config = {
                'focal_loss': {
                    'alpha': 1.5,
                    'beta': 4.0,
                    'pos_threshold': 0.8,
                    'eps': 1e-8
                },
                'offset_loss': {
                    'beta': 1.0
                },
                'hard_negative_loss': {
                    'margin': 0.2,
                    'score_type': 'bilinear',
                    'neighborhood_size': 3
                },
                'angle_loss': {
                    'weight': 1.0
                },
                'total_loss': {
                    'use_angle': False,  # 默认不使用角度损失
                    'weights': {
                        'heatmap': 1.0,
                        'offset': 1.0,
                        'hard_negative': 0.5,
                        'angle': 0.5
                    }
                }
            }
        else:
            with open(loss_config_path, 'r', encoding='utf-8') as f:
                loss_config = yaml.safe_load(f)
        
        # 创建总损失函数
        loss_fn = create_total_loss(loss_config)
        self.logger.info("使用完整的TotalLoss（包含硬负样本损失和角度损失）")
        self.logger.info(f"损失权重: {loss_config['total_loss']['weights']}")
        
        return loss_fn
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """设置优化器"""
        opt_cfg = self.config['optimizer']
        
        # 创建AdamW优化器
        optimizer = AdamW(
            self.model.parameters(),
            lr=opt_cfg['lr'],
            betas=opt_cfg.get('betas', [0.9, 0.999]),
            eps=opt_cfg.get('eps', 1e-8),
            weight_decay=opt_cfg.get('weight_decay', 0.05)
        )
        
        self.logger.info(f"优化器: AdamW (lr={opt_cfg['lr']}, wd={opt_cfg['weight_decay']})")
        
        return optimizer
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        sched_cfg = self.config['sched']
        
        # 使用Cosine Annealing with Warm Restarts
        scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=sched_cfg.get('warmup_epochs', 5),
            T_mult=2,
            eta_min=sched_cfg.get('cosine_min_lr', 3e-6)
        )
        
        self.logger.info(f"调度器: CosineAnnealingWarmRestarts")
        
        return scheduler
    
    def _setup_ema(self) -> nn.Module:
        """设置指数移动平均模型"""
        ema_model = deepcopy(self.model)
        ema_model.eval()
        
        # 禁用梯度
        for param in ema_model.parameters():
            param.requires_grad = False
        
        return ema_model
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前epoch
            
        Returns:
            训练指标字典
        """
        self.model.train()
        self.epoch = epoch
        
        # 初始化指标
        metrics = {
            'loss': 0.0,
            'focal_loss': 0.0,
            'offset_loss': 0.0,
            'hard_negative_loss': 0.0,
            'angle_loss': 0.0,
            'gap_mae': 0.0,
            'slider_mae': 0.0,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        num_batches = len(dataloader)
        log_interval = self.config['logging'].get('log_interval', 10)
        
        # 批次循环
        for batch_idx, batch in enumerate(dataloader):
            # 数据传输到设备
            batch = self._batch_to_device(batch)
            
            # 前向传播
            loss, batch_metrics = self._forward_step(batch)
            
            # 反向传播
            self._backward_step(loss)
            
            # 更新EMA
            if self.ema_model is not None:
                self._update_ema()
            
            # 累积指标
            for key in batch_metrics:
                if key in metrics:
                    metrics[key] += batch_metrics[key]
            
            # 更新全局步数
            self.global_step += 1
            
            # 定期日志
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = metrics['loss'] / (batch_idx + 1)
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                    f"Loss: {avg_loss:.4f}, LR: {metrics['lr']:.6f}"
                )
        
        # 平均指标
        for key in metrics:
            if key != 'lr':
                metrics[key] /= num_batches
        
        return metrics
    
    def _batch_to_device(self, batch: Dict) -> Dict:
        """将批次数据传输到设备"""
        device_batch = {}
        # 检查是否启用非阻塞传输
        non_blocking = self.config['train'].get('non_blocking', True)
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=non_blocking)
                # 应用channels_last（如果启用）
                if self.config['train'].get('channels_last', False) and len(value.shape) == 4:
                    device_batch[key] = device_batch[key].to(memory_format=torch.channels_last)
            else:
                device_batch[key] = value
        return device_batch
    
    def _forward_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        前向传播步骤
        
        Args:
            batch: 批次数据
            
        Returns:
            损失和指标
        """
        # 混合精度上下文
        if self.use_amp:
            # 兼容旧版本PyTorch的autocast API
            with autocast(dtype=self.amp_dtype):
                outputs = self.model(batch['image'])
                loss, loss_dict = self._compute_loss(outputs, batch)
        else:
            outputs = self.model(batch['image'])
            loss, loss_dict = self._compute_loss(outputs, batch)
        
        # 计算预测误差（用于监控）
        with torch.no_grad():
            predictions = self.model.decode_predictions(outputs)
            gap_error = torch.abs(
                predictions['gap_coords'] - batch['gap_coords']
            ).mean().item()
            slider_error = torch.abs(
                predictions['slider_coords'] - batch['slider_coords']
            ).mean().item()
        
        # 整合指标
        metrics = {
            'loss': loss.item(),
            'focal_loss': loss_dict.get('focal_loss', 0.0),
            'offset_loss': loss_dict.get('offset_loss', 0.0),
            'hard_negative_loss': loss_dict.get('hard_negative_loss', 0.0),
            'angle_loss': loss_dict.get('angle_loss', 0.0),
            'gap_mae': gap_error,
            'slider_mae': slider_error
        }
        
        return loss, metrics
    
    def _compute_loss(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失函数（使用完整的TotalLoss）
        
        包含：
        - Focal Loss用于热力图
        - L1 Loss用于偏移量  
        - Hard Negative Loss用于混淆缺口抑制
        - Angle Loss用于微旋转（可选）
        
        Args:
            outputs: 模型输出
            targets: 目标值（已包含预计算的热力图和偏移量）
            
        Returns:
            总损失和各项损失
        """
        # 准备TotalLoss需要的输入格式
        predictions = {
            'heatmap_gap': outputs['heatmap_gap'],  # [B, 1, H, W]
            'heatmap_piece': outputs['heatmap_slider'],  # [B, 1, H, W] 
            'offset': torch.cat([  # [B, 4, H, W]
                outputs['offset_gap'],  # [B, 2, H, W]
                outputs['offset_slider']  # [B, 2, H, W]
            ], dim=1)
        }
        
        # 如果模型输出角度预测，添加到predictions
        if 'angle' in outputs:
            predictions['angle'] = outputs['angle']  # [B, 2, H, W]
        
        # 准备目标格式
        loss_targets = {
            'heatmap_gap': targets['heatmap_gap'].unsqueeze(1),  # [B, 1, H, W]
            'heatmap_piece': targets['heatmap_slider'].unsqueeze(1),  # [B, 1, H, W]
            'offset': torch.cat([  # [B, 4, H, W]
                targets['offset_gap'],  # [B, 2, H, W]
                targets['offset_slider']  # [B, 2, H, W]
            ], dim=1)
        }
        
        # 添加权重掩码（如果有）
        if 'weight_gap' in targets and 'weight_slider' in targets:
            # 使用gap和slider权重的平均值作为统一掩码
            loss_targets['mask'] = (targets['weight_gap'] + targets['weight_slider']) / 2.0
            if loss_targets['mask'].dim() == 3:  # [B, H, W]
                loss_targets['mask'] = loss_targets['mask'].unsqueeze(1)  # [B, 1, H, W]
        
        # 准备混淆缺口信息（用于硬负样本损失）
        batch_size = predictions['heatmap_gap'].shape[0]
        gap_centers = []
        fake_centers_list = []
        
        # 从targets中提取真实缺口中心和混淆缺口中心
        if 'gap_coords' in targets:
            # gap_coords是原图坐标，需要转换到特征图坐标
            gap_centers = targets['gap_coords'] / 4.0  # [B, 2]
            loss_targets['gap_center'] = gap_centers
        else:
            # 如果没有提供坐标，从热力图中提取峰值位置
            gap_centers = self._extract_peak_coords(targets['heatmap_gap'])
            loss_targets['gap_center'] = gap_centers
        
        # 提取混淆缺口坐标（如果有）
        if 'confusing_gaps' in targets:
            # confusing_gaps应该是每个批次样本的混淆缺口列表
            loss_targets['fake_centers'] = targets['confusing_gaps']
        else:
            # 如果没有混淆缺口信息，提供空列表
            loss_targets['fake_centers'] = [[] for _ in range(batch_size)]
        
        # 如果有角度目标，添加到loss_targets
        if 'angle' in targets:
            loss_targets['angle'] = targets['angle']  # [B, 2, H, W]
        
        # 使用TotalLoss计算损失
        total_loss, loss_dict = self.loss_fn(predictions, loss_targets)
        
        # 提取各项损失值用于记录
        result_dict = {
            'focal_loss': loss_dict.get('heatmap', 0.0).item() if isinstance(loss_dict.get('heatmap', 0.0), torch.Tensor) else loss_dict.get('heatmap', 0.0),
            'offset_loss': loss_dict.get('offset', 0.0).item() if isinstance(loss_dict.get('offset', 0.0), torch.Tensor) else loss_dict.get('offset', 0.0),
            'hard_negative_loss': loss_dict.get('hard_negative', 0.0).item() if isinstance(loss_dict.get('hard_negative', 0.0), torch.Tensor) else loss_dict.get('hard_negative', 0.0),
            'angle_loss': loss_dict.get('angle', 0.0).item() if isinstance(loss_dict.get('angle', 0.0), torch.Tensor) else loss_dict.get('angle', 0.0)
        }
        
        return total_loss, result_dict
    
    def _extract_peak_coords(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        从热力图中提取峰值坐标
        
        Args:
            heatmap: 热力图 [B, H, W]
            
        Returns:
            峰值坐标 [B, 2] (x, y)
        """
        batch_size = heatmap.shape[0]
        coords = torch.zeros(batch_size, 2, device=heatmap.device)
        
        for b in range(batch_size):
            # 找到最大值位置
            flat_idx = heatmap[b].argmax()
            y = flat_idx // heatmap.shape[2]
            x = flat_idx % heatmap.shape[2]
            coords[b] = torch.tensor([x, y], device=heatmap.device, dtype=torch.float32)
        
        return coords
    
    def _generate_targets(self, 
                         coords: torch.Tensor, 
                         shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成目标热力图和偏移量
        
        Args:
            coords: 坐标 [B, 2]
            shape: 输出形状 [B, 1, H, W]
            
        Returns:
            热力图和偏移量目标
        """
        batch_size, _, height, width = shape
        device = coords.device
        
        # 初始化
        heatmap = torch.zeros(batch_size, 1, height, width, device=device)
        offset = torch.zeros(batch_size, 2, height, width, device=device)
        
        # 缩放因子（原图到特征图）
        scale = 4.0  # 根据模型下采样率
        
        for b in range(batch_size):
            # 转换到特征图坐标
            x = coords[b, 0] / scale
            y = coords[b, 1] / scale
            
            # 整数坐标
            x_int = int(x)
            y_int = int(y)
            
            # 确保在范围内
            if 0 <= x_int < width and 0 <= y_int < height:
                # 生成高斯热力图
                sigma = 2.0
                for i in range(max(0, y_int-3*int(sigma)), min(height, y_int+3*int(sigma)+1)):
                    for j in range(max(0, x_int-3*int(sigma)), min(width, x_int+3*int(sigma)+1)):
                        dist = ((i - y) ** 2 + (j - x) ** 2) / (2 * sigma ** 2)
                        heatmap[b, 0, i, j] = torch.exp(-dist)
                
                # 偏移量
                offset[b, 0, y_int, x_int] = x - x_int - 0.5
                offset[b, 1, y_int, x_int] = y - y_int - 0.5
        
        return heatmap, offset
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                   alpha: float = 2.0, beta: float = 4.0, 
                   weight_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        CenterNet变体的Focal Loss
        
        Args:
            pred: 预测热力图
            target: 目标热力图
            alpha, beta: 超参数
            
        Returns:
            损失值
        """
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        # 正样本损失
        pos_loss = -torch.log(pred + 1e-12) * torch.pow(1 - pred, alpha) * pos_mask
        
        # 负样本损失
        neg_weights = torch.pow(1 - target, beta)
        neg_loss = -torch.log(1 - pred + 1e-12) * torch.pow(pred, alpha) * neg_weights * neg_mask
        
        # 应用权重掩码（如果提供）
        if weight_mask is not None:
            if weight_mask.dim() == 3:  # [B, H, W]
                weight_mask = weight_mask.unsqueeze(1)  # [B, 1, H, W]
            pos_loss = pos_loss * weight_mask
            neg_loss = neg_loss * weight_mask
        
        # 归一化
        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            loss = neg_loss
        else:
            loss = (pos_loss + neg_loss) / num_pos
        
        return loss
    
    def _offset_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    mask: torch.Tensor) -> torch.Tensor:
        """
        偏移量L1损失
        
        Args:
            pred: 预测偏移量 [B, 2, H, W]
            target: 目标偏移量 [B, 2, H, W]
            mask: 正样本掩码 [B, 1, H, W]
            
        Returns:
            损失值
        """
        # 只在正样本位置计算损失
        pos_mask = (mask > 0.5).float().expand_as(pred)
        
        loss = F.l1_loss(pred * pos_mask, target * pos_mask, reduction='sum')
        
        # 归一化
        num_pos = pos_mask.sum() / 2  # 除以2因为有x,y两个通道
        if num_pos > 0:
            loss = loss / num_pos
        
        return loss
    
    def _backward_step(self, loss: torch.Tensor):
        """反向传播步骤（支持梯度累积）"""
        # 如果使用梯度累积，需要缩放损失
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        # 累积计数器
        self.accumulation_counter += 1
        
        if self.use_amp and self.amp_dtype != torch.bfloat16:
            # 标准混合精度（FP16）
            self.scaler.scale(loss).backward()
            
            # 只在累积完成时更新权重
            if self.accumulation_counter % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            # BFloat16或FP32
            loss.backward()
            
            # 只在累积完成时更新权重
            if self.accumulation_counter % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
    
    def _update_ema(self):
        """更新EMA模型"""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            decay = self.ema_decay
            for ema_param, model_param in zip(
                self.ema_model.parameters(), 
                self.model.parameters()
            ):
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1-decay)
    
    def step_scheduler(self):
        """更新学习率调度器"""
        self.scheduler.step()
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def save_checkpoint(self) -> Dict:
        """
        保存检查点
        
        Returns:
            检查点字典
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch
        }
        
        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        
        return checkpoint
    
    def load_checkpoint(self, checkpoint: Dict):
        """
        加载检查点
        
        Args:
            checkpoint: 检查点字典
        """
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        if self.ema_model is not None and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        
        self.logger.info(f"从epoch {self.epoch}恢复训练")


if __name__ == "__main__":
    # 测试训练引擎
    print("训练引擎模块测试")
    print("注：需要配合模型和数据加载器使用")