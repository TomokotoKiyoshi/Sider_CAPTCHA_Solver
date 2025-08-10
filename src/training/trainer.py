# -*- coding: utf-8 -*-
"""
核心训练器模块
封装模型训练的核心逻辑
"""
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .metrics import MetricTracker, extract_coordinates_from_heatmap
from .utils import EarlyStopping, Logger, save_checkpoint


class Trainer:
    """PMN-R3-FP 训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict,
        logger: Logger,
        device: str = 'cuda'
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            config: 配置字典
            logger: 日志记录器
            device: 设备
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.logger = logger
        self.device = device
        
        # 训练配置
        self.use_amp = config['training'].get('use_amp', True)
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)
        
        # 混合精度训练
        self.scaler = GradScaler() if self.use_amp else None
        
        # 早停机制
        if config['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=config['early_stopping']['patience'],
                min_delta=config['early_stopping']['min_delta'],
                mode=config['early_stopping']['mode']
            )
        else:
            self.early_stopping = None
        
        # 指标跟踪
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        
        # 最佳指标
        self.best_metric = float('inf') if config['early_stopping']['mode'] == 'min' else float('-inf')
        self.start_epoch = 0
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            训练指标字典
        """
        self.model.train()
        self.train_metrics.reset()
        
        # 进度条
        pbar = tqdm(
            dataloader,
            desc=f'Epoch [{epoch}/{self.config["training"]["num_epochs"]}]',
            dynamic_ncols=True
        )
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # 数据移到设备
            images = images.to(self.device)
            gap_coords = targets['gap_coord'].to(self.device)
            slider_coords = targets['slider_coord'].to(self.device)
            
            # 混合精度训练
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, gap_coords, slider_coords)
                    loss = loss / self.accumulation_steps
            else:
                outputs = self.model(images)
                loss = self._compute_loss(outputs, gap_coords, slider_coords)
                loss = loss / self.accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # 梯度裁剪
                if self.gradient_clip > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                # 优化器步进
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # 提取预测坐标
            with torch.no_grad():
                gap_pred = self._extract_coords(outputs, 'gap')
                slider_pred = self._extract_coords(outputs, 'slider')
            
            # 更新指标
            self.train_metrics.update(
                loss.item() * self.accumulation_steps,
                gap_pred,
                gap_coords,
                slider_pred,
                slider_coords,
                images.size(0)
            )
            
            # 更新进度条
            if batch_idx % 10 == 0:
                metrics = self.train_metrics.get_metrics()
                pbar.set_postfix({
                    'loss': f"{metrics.get('loss', 0):.4f}",
                    'mae': f"{metrics.get('mae', 0):.2f}px"
                })
        
        return self.train_metrics.get_metrics()
    
    def validate(self, dataloader, epoch: int = -1) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            dataloader: 验证数据加载器
            epoch: 当前epoch（用于显示）
            
        Returns:
            验证指标字典
        """
        self.model.eval()
        self.val_metrics.reset()
        
        desc = 'Validation' if epoch < 0 else f'Validation [Epoch {epoch}]'
        pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True)
        
        with torch.no_grad():
            for images, targets in pbar:
                # 数据移到设备
                images = images.to(self.device)
                gap_coords = targets['gap_coord'].to(self.device)
                slider_coords = targets['slider_coord'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self._compute_loss(outputs, gap_coords, slider_coords)
                
                # 提取预测坐标
                gap_pred = self._extract_coords(outputs, 'gap')
                slider_pred = self._extract_coords(outputs, 'slider')
                
                # 更新指标
                self.val_metrics.update(
                    loss.item(),
                    gap_pred,
                    gap_coords,
                    slider_pred,
                    slider_coords,
                    images.size(0)
                )
                
                # 更新进度条
                metrics = self.val_metrics.get_metrics()
                pbar.set_postfix({
                    'mae': f"{metrics.get('mae', 0):.2f}px",
                    'hit@2px': f"{metrics.get('hit_at_2px', 0):.1f}%"
                })
        
        return self.val_metrics.get_metrics()
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        checkpoint_dir: str,
        test_loader=None
    ):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            checkpoint_dir: 检查点保存目录
            test_loader: 测试数据加载器（可选）
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        self.logger.log("=" * 60)
        self.logger.log("Starting Training")
        self.logger.log(f"Device: {self.device}")
        self.logger.log(f"Checkpoint directory: {checkpoint_dir}")
        self.logger.log("=" * 60)
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch + 1)
            
            # 验证
            val_metrics = self.validate(val_loader, epoch + 1)
            
            # 学习率调度
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        self.scheduler.step(val_metrics['mae'])
                    else:
                        self.scheduler.step()
            
            # 记录指标
            self.logger.log_metrics('train', epoch + 1, train_metrics)
            self.logger.log_metrics('val', epoch + 1, val_metrics)
            
            # 打印epoch总结
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.logger.log(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
            self.logger.log(f"  Time: {epoch_time:.2f}s")
            self.logger.log(f"  Learning Rate: {current_lr:.6f}")
            self.logger.log(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                          f"MAE: {train_metrics['mae']:.2f}px")
            self.logger.log(f"  Val - Loss: {val_metrics['loss']:.4f}, "
                          f"MAE: {val_metrics['mae']:.2f}px, "
                          f"Hit@2px: {val_metrics['hit_at_2px']:.1f}%")
            
            # 检查是否为最佳模型
            metric_value = val_metrics[self.config['early_stopping']['metric'].replace('val_', '')]
            is_best = self._is_better(metric_value)
            
            if is_best:
                self.best_metric = metric_value
                self.logger.log(f"  New best {self.config['early_stopping']['metric']}: "
                              f"{metric_value:.4f}")
            
            # 保存检查点
            state = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_metric': self.best_metric,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': self.config
            }
            
            save_checkpoint(
                state,
                epoch + 1,
                str(checkpoint_dir),
                is_best,
                self.config['checkpoint']['keep_last_n']
            )
            
            # 早停检查
            if self.early_stopping:
                if self.early_stopping(metric_value):
                    self.logger.log(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
        
        # 保存训练历史
        self.logger.save_history()
        
        # 最终测试
        if test_loader:
            self._run_final_test(test_loader, checkpoint_dir)
        
        self.logger.log("\n" + "=" * 60)
        self.logger.log("Training Completed!")
        self.logger.log(f"Best model saved at: {checkpoint_dir / 'best_model.pth'}")
        self.logger.log("=" * 60)
    
    def _compute_loss(self, outputs: Dict, gap_coords: torch.Tensor, 
                     slider_coords: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        # 构建目标字典
        targets = {
            'gap_coord': gap_coords,
            'slider_coord': slider_coords
        }
        
        # 计算损失
        loss_dict = self.criterion(outputs, targets)
        
        # 返回总损失
        if isinstance(loss_dict, dict):
            return loss_dict['total']
        return loss_dict
    
    def _extract_coords(self, outputs: Dict, target_type: str) -> torch.Tensor:
        """从模型输出提取坐标"""
        if target_type == 'gap':
            heatmap = outputs.get('gap_heatmap', outputs.get('heatmap_gap'))
            offset = outputs.get('gap_offset', outputs.get('offset_gap'))
        else:  # slider/piece
            heatmap = outputs.get('piece_heatmap', outputs.get('heatmap_piece'))
            offset = outputs.get('piece_offset', outputs.get('offset_piece'))
        
        # 从热力图提取坐标
        coords = extract_coordinates_from_heatmap(heatmap, offset, scale=4)
        return coords
    
    def _is_better(self, metric_value: float) -> bool:
        """判断指标是否更好"""
        mode = self.config['early_stopping']['mode']
        if mode == 'min':
            return metric_value < self.best_metric
        else:
            return metric_value > self.best_metric
    
    def _run_final_test(self, test_loader, checkpoint_dir: Path):
        """运行最终测试"""
        self.logger.log("\n" + "=" * 60)
        self.logger.log("Running Final Test on Test Set")
        self.logger.log("=" * 60)
        
        # 加载最佳模型
        best_path = checkpoint_dir / 'best_model.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.log(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        # 运行测试
        test_metrics = self.validate(test_loader)
        
        # 记录和保存结果
        self.logger.log_metrics('test', -1, test_metrics)
        self.logger.log("\nTest Results:")
        self.logger.log(f"  MAE: {test_metrics['mae']:.2f}px")
        self.logger.log(f"  Gap MAE: {test_metrics['gap_mae']:.2f}px")
        self.logger.log(f"  Slider MAE: {test_metrics['slider_mae']:.2f}px")
        self.logger.log(f"  Hit@2px: {test_metrics['hit_at_2px']:.1f}%")
        self.logger.log(f"  Hit@5px: {test_metrics['hit_at_5px']:.1f}%")
        
        # 保存测试结果到JSON
        import json
        from datetime import datetime
        
        results = {
            'test_metrics': test_metrics,
            'best_val_metric': self.best_metric,
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        
        results_path = checkpoint_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.log(f"  Results saved to: {results_path}")