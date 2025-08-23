#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练引擎 - 核心训练循环管理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
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
    3. 训练循环执行
    4. 梯度裁剪
    5. 损失计算
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 config: Dict, 
                 device: torch.device,
                 visualizer=None):
        """
        初始化训练引擎
        
        Args:
            model: 模型
            config: 配置字典
            device: 训练设备
            visualizer: 可视化器（可选）
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.visualizer = visualizer
        self.logger = logging.getLogger('TrainingEngine')
        
        # 加载损失函数配置并创建总损失函数
        self.loss_fn = self._setup_loss_function()
        
        # 优化器设置
        self.optimizer = self._setup_optimizer()
        
        # 学习率调度器
        self.scheduler = self._setup_scheduler()
        
        # 混合精度训练 - 支持布尔值和字符串配置
        amp_config = config['train'].get('amp', False)
        
        # 智能解析AMP配置
        if amp_config == True or amp_config == 'true' or amp_config == 'bf16':
            # RTX 40/50系列优先使用BFloat16
            self.use_amp = True
            self.amp_dtype = torch.bfloat16
            self.scaler = GradScaler(enabled=False)  # BFloat16不需要GradScaler
            self.logger.info("使用BFloat16混合精度训练")
        elif amp_config == 'fp16':
            # 兼容旧GPU的FP16模式
            self.use_amp = True
            self.amp_dtype = torch.float16
            self.scaler = GradScaler(enabled=True)  # FP16需要GradScaler
            self.logger.info("使用FP16混合精度训练")
        else:
            # 不使用混合精度
            self.use_amp = False
            self.scaler = None
            self.amp_dtype = None
            self.logger.info("不使用混合精度训练")
        
        # EMA已移除
        
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
            'padding_bce_loss': 0.0,
            'y_axis_loss': 0.0,
            'y_axis_row_ce': 0.0,
            'y_axis_emd': 0.0,
            'gap_mae': 0.0,
            'slider_mae': 0.0,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        num_batches = len(dataloader)
        log_interval = self.config['logging'].get('log_interval', 10)
        
        # 时间追踪
        import time
        epoch_start_time = time.time()
        batch_times = []
        
        # 批次循环
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            # 数据传输到设备
            batch = self._batch_to_device(batch)
            
            
            # 前向传播
            loss, batch_metrics = self._forward_step(batch)
            
            # 反向传播
            self._backward_step(loss)
            
            # 累积指标 - 处理tensor
            for key in batch_metrics:
                if key in metrics and key != 'lr':
                    if torch.is_tensor(batch_metrics[key]):
                        metrics[key] += batch_metrics[key].detach()  # detach避免梯度累积
                    else:
                        metrics[key] += batch_metrics[key]
            
            # 更新全局步数
            self.global_step += 1
            
            # 记录批次时间
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # 定期日志
            if (batch_idx + 1) % log_interval == 0:
                # 计算平均值 - 只在日志时同步一次
                batch_count = batch_idx + 1
                # 使用.item()转换为标量，但只在需要打印时
                avg_loss = (metrics['loss'] / batch_count).item() if torch.is_tensor(metrics['loss']) else metrics['loss'] / batch_count
                avg_focal = (metrics['focal_loss'] / batch_count).item() if torch.is_tensor(metrics['focal_loss']) else metrics['focal_loss'] / batch_count
                avg_offset = (metrics['offset_loss'] / batch_count).item() if torch.is_tensor(metrics['offset_loss']) else metrics['offset_loss'] / batch_count
                avg_hard_neg = (metrics['hard_negative_loss'] / batch_count).item() if torch.is_tensor(metrics['hard_negative_loss']) else metrics['hard_negative_loss'] / batch_count
                avg_angle = (metrics['angle_loss'] / batch_count).item() if torch.is_tensor(metrics['angle_loss']) else metrics['angle_loss'] / batch_count
                avg_padding_bce = (metrics['padding_bce_loss'] / batch_count).item() if torch.is_tensor(metrics['padding_bce_loss']) else metrics['padding_bce_loss'] / batch_count
                avg_y_axis = (metrics['y_axis_loss'] / batch_count).item() if torch.is_tensor(metrics['y_axis_loss']) else metrics['y_axis_loss'] / batch_count
                
                # 计算时间估计
                avg_batch_time = sum(batch_times) / len(batch_times)
                remaining_batches = num_batches - batch_count
                eta_seconds = remaining_batches * avg_batch_time
                
                # 格式化ETA
                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.0f}s"
                elif eta_seconds < 3600:
                    eta_str = f"{eta_seconds/60:.1f}min"
                else:
                    hours = int(eta_seconds // 3600)
                    minutes = int((eta_seconds % 3600) // 60)
                    eta_str = f"{hours}h{minutes}m"
                
                # 计算速度
                samples_per_second = self.config['train']['batch_size'] / avg_batch_time
                
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                    f"Loss: {avg_loss:.4f}, "
                    f"Focal: {avg_focal:.4f}, "
                    f"Offset: {avg_offset:.4f}, "
                    f"HardNeg: {avg_hard_neg:.4f}, "
                    f"Angle: {avg_angle:.4f}, "
                    f"PadBCE: {avg_padding_bce:.4f}, "
                    f"YAxis: {avg_y_axis:.4f}, "
                    f"LR: {metrics['lr']:.6f} | "
                    f"ETA: {eta_str} | "
                    f"Speed: {samples_per_second:.0f} samples/s"
                )
                
                # 如果有visualizer，记录到TensorBoard
                if hasattr(self, 'visualizer') and self.visualizer is not None:
                    # 使用epoch内的批次索引作为步数，确保每个epoch图表重新开始
                    epoch_batch_step = batch_idx + 1  # 从1开始
                    
                    # 批次级别的损失（每个epoch重新绘制）
                    self.visualizer.writer.add_scalar(f'batch_epoch{epoch}/loss', avg_loss, epoch_batch_step)
                    self.visualizer.writer.add_scalar(f'batch_epoch{epoch}/focal_loss', avg_focal, epoch_batch_step)
                    self.visualizer.writer.add_scalar(f'batch_epoch{epoch}/offset_loss', avg_offset, epoch_batch_step)
                    self.visualizer.writer.add_scalar(f'batch_epoch{epoch}/hard_negative_loss', avg_hard_neg, epoch_batch_step)
                    self.visualizer.writer.add_scalar(f'batch_epoch{epoch}/angle_loss', avg_angle, epoch_batch_step)
                    self.visualizer.writer.add_scalar(f'batch_epoch{epoch}/padding_bce_loss', avg_padding_bce, epoch_batch_step)
                    
                    # 全局步数的损失（连续图表）
                    self.visualizer.writer.add_scalar('batch_global/loss', avg_loss, self.global_step)
                    self.visualizer.writer.add_scalar('batch_global/focal_loss', avg_focal, self.global_step)
                    self.visualizer.writer.add_scalar('batch_global/offset_loss', avg_offset, self.global_step)
                    self.visualizer.writer.add_scalar('batch_global/hard_negative_loss', avg_hard_neg, self.global_step)
                    self.visualizer.writer.add_scalar('batch_global/angle_loss', avg_angle, self.global_step)
                    self.visualizer.writer.add_scalar('batch_global/padding_bce_loss', avg_padding_bce, self.global_step)
                    
                    # 训练速度
                    self.visualizer.writer.add_scalar('training/speed_samples_per_sec', samples_per_second, self.global_step)
                    self.visualizer.flush()
        
        # 平均指标 - 最后转换为标量
        for key in metrics:
            if key != 'lr':
                if torch.is_tensor(metrics[key]):
                    metrics[key] = (metrics[key] / num_batches).item()
                else:
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
            # 使用新版本PyTorch的autocast API
            with autocast(device_type='cuda', dtype=self.amp_dtype):
                outputs = self.model(batch['image'])
                            
                loss, loss_dict = self._compute_loss(outputs, batch)
        else:
            outputs = self.model(batch['image'])
                        
            loss, loss_dict = self._compute_loss(outputs, batch)
        
        
        # 计算预测误差（用于监控） - 不使用.item()避免GPU同步！
        with torch.no_grad():
            predictions = self.model.decode_predictions(outputs, input_images=batch['image'])
            gap_error = torch.abs(
                predictions['gap_coords'] - batch['gap_coords']
            ).mean()  # 保持为tensor，不调用.item()
            slider_error = torch.abs(
                predictions['slider_coords'] - batch['slider_coords']
            ).mean()  # 保持为tensor，不调用.item()
        
        # 整合指标 - 保持为tensor，延迟同步
        metrics = {
            'loss': loss,  # 保持为tensor
            'focal_loss': loss_dict.get('focal_loss', torch.tensor(0.0, device=self.device)),
            'offset_loss': loss_dict.get('offset_loss', torch.tensor(0.0, device=self.device)),
            'hard_negative_loss': loss_dict.get('hard_negative_loss', torch.tensor(0.0, device=self.device)),
            'angle_loss': loss_dict.get('angle_loss', torch.tensor(0.0, device=self.device)),
            'padding_bce_loss': loss_dict.get('padding_bce', torch.tensor(0.0, device=self.device)),
            'y_axis_loss': loss_dict.get('y_axis', torch.tensor(0.0, device=self.device)),
            'y_axis_row_ce': loss_dict.get('y_axis_row_ce', torch.tensor(0.0, device=self.device)),
            'y_axis_emd': loss_dict.get('y_axis_emd', torch.tensor(0.0, device=self.device)),
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
        
        # 添加权重掩码（必须存在，因为输入包含padding mask）
        if 'weight_gap' not in targets or 'weight_slider' not in targets:
            raise ValueError("数据批次缺少权重掩码（weight_gap/weight_slider），这是必需的因为输入包含padding mask通道")
        
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
        
        # 添加滑块中心（用于Y轴损失）
        if 'slider_coords' in targets:
            # slider_coords是原图坐标，需要转换到特征图坐标
            slider_centers = targets['slider_coords'] / 4.0  # [B, 2]
            loss_targets['slider_center'] = slider_centers
        else:
            # 如果没有提供坐标，从热力图中提取峰值位置
            slider_centers = self._extract_peak_coords(targets['heatmap_slider'])
            loss_targets['slider_center'] = slider_centers
        
        # 提取混淆缺口坐标（必须存在）
        if 'confusing_gaps' not in targets:
            raise ValueError("数据批次缺少 'confusing_gaps' 字段，无法计算硬负样本损失。请确保数据预处理正确生成了混淆缺口信息。")
        
        # 转换混淆缺口格式：从嵌套列表转换为张量列表
        # confusing_gaps: [[样本1的假缺口], [样本2的假缺口], ...]
        # 需要转换为：[所有样本的第1个假缺口张量, 所有样本的第2个假缺口张量, ...]
        confusing_gaps = targets['confusing_gaps']
        batch_size = len(confusing_gaps)
        
        # 找出最大的假缺口数量
        max_fake_gaps = max(len(gaps) for gaps in confusing_gaps) if confusing_gaps else 0
        
        # 构建张量列表
        fake_centers_list = []
        
        if max_fake_gaps > 0:
            for gap_idx in range(max_fake_gaps):
                batch_fake_centers = []
                for sample_idx in range(batch_size):
                    if gap_idx < len(confusing_gaps[sample_idx]):
                        # 该样本有这个假缺口
                        batch_fake_centers.append(confusing_gaps[sample_idx][gap_idx])
                    else:
                        # 该样本没有这个假缺口，使用一个远离的点或重复最后一个
                        if len(confusing_gaps[sample_idx]) > 0:
                            batch_fake_centers.append(confusing_gaps[sample_idx][-1])
                        else:
                            # 没有假缺口的样本，使用一个默认值（远离实际缺口的位置）
                            batch_fake_centers.append([0.0, 0.0])
                
                # 转换为张量 [B, 2]
                fake_centers_tensor = torch.tensor(batch_fake_centers, dtype=torch.float32, device=self.device)
                fake_centers_list.append(fake_centers_tensor)
        
        loss_targets['fake_centers'] = fake_centers_list
        
        # 提取角度信息（必须存在）
        if 'angle' not in targets and 'gap_angles' not in targets:
            raise ValueError("数据批次缺少 'angle' 或 'gap_angles' 字段，无法计算角度损失。请确保数据预处理正确生成了角度信息。")
        
        # 获取角度值（1维张量，每个样本一个角度）
        angle_values = targets.get('angle', targets.get('gap_angles'))  # [B]
        
        # 检查是否有非零角度
        has_rotation = (angle_values != 0).any()
        
        if has_rotation:
            # 只有在有旋转角度时才准备角度张量
            # 将角度值转换为4维张量 [B, 2, H, W]
            # 角度表示为 (sin(θ), cos(θ))
            batch_size = angle_values.shape[0]
            height, width = predictions['heatmap_gap'].shape[2:]  # 特征图尺寸
            
            # 计算sin和cos
            sin_values = torch.sin(angle_values)  # [B]
            cos_values = torch.cos(angle_values)  # [B]
            
            # 扩展为4维张量
            sin_map = sin_values.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
            cos_map = cos_values.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
            
            # 合并为 [B, 2, H, W]
            angle_tensor = torch.cat([sin_map, cos_map], dim=1)
            
            loss_targets['angle'] = angle_tensor  # 传递4维角度张量
        else:
            # 没有旋转角度，不计算角度损失
            loss_targets['angle'] = None
        
        # 使用TotalLoss计算损失
        total_loss, loss_dict = self.loss_fn(predictions, loss_targets)
        
        # 提取各项损失值用于记录 - 保持为tensor，不调用.item()
        result_dict = {
            'focal_loss': loss_dict.get('heatmap', torch.tensor(0.0, device=self.device)),
            'offset_loss': loss_dict.get('offset', torch.tensor(0.0, device=self.device)),
            'hard_negative_loss': loss_dict.get('hard_negative', torch.tensor(0.0, device=self.device)),
            'angle_loss': loss_dict.get('angle', torch.tensor(0.0, device=self.device)),
            'padding_bce': loss_dict.get('padding_bce', torch.tensor(0.0, device=self.device)),
            'y_axis': loss_dict.get('y_axis', torch.tensor(0.0, device=self.device)),
            'y_axis_row_ce': loss_dict.get('y_axis_row_ce', torch.tensor(0.0, device=self.device)),
            'y_axis_emd': loss_dict.get('y_axis_emd', torch.tensor(0.0, device=self.device))
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
        
        if self.use_amp and self.amp_dtype == torch.float16:
            # FP16模式需要使用GradScaler
            self.scaler.scale(loss).backward()
            
            # 只在累积完成时更新权重
            if self.accumulation_counter % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
        else:
            # BFloat16或FP32模式
            loss.backward()
            
            # 只在累积完成时更新权重
            if self.accumulation_counter % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
    
    
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
        
        
        self.logger.info(f"从epoch {self.epoch}恢复训练")


if __name__ == "__main__":
    # 测试训练引擎
    print("训练引擎模块测试")
    print("注：需要配合模型和数据加载器使用")