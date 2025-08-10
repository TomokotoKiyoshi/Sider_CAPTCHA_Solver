# -*- coding: utf-8 -*-
"""
评估指标模块
计算MAE、命中率等关键指标
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn
import numpy as np


class MetricTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.total_loss = 0.0
        self.total_mae = 0.0
        self.total_gap_mae = 0.0
        self.total_slider_mae = 0.0
        self.hit_at_2px = 0
        self.hit_at_5px = 0
        self.num_samples = 0
        self.num_batches = 0
    
    def update(
        self,
        loss: float,
        gap_pred: torch.Tensor,
        gap_target: torch.Tensor,
        slider_pred: torch.Tensor,
        slider_target: torch.Tensor,
        batch_size: int
    ):
        """
        更新指标
        
        Args:
            loss: 批次损失
            gap_pred: 缺口预测坐标 [B, 2]
            gap_target: 缺口真实坐标 [B, 2]
            slider_pred: 滑块预测坐标 [B, 2]
            slider_target: 滑块真实坐标 [B, 2]
            batch_size: 批次大小
        """
        with torch.no_grad():
            # 计算误差
            gap_error = (gap_pred - gap_target).abs()
            slider_error = (slider_pred - slider_target).abs()
            
            # MAE（平均绝对误差）
            gap_mae = gap_error.mean().item()
            slider_mae = slider_error.mean().item()
            mae = (gap_mae + slider_mae) / 2
            
            # 命中率计算
            gap_hit_2px = (gap_error.max(dim=1)[0] <= 2).float().sum().item()
            slider_hit_2px = (slider_error.max(dim=1)[0] <= 2).float().sum().item()
            hit_2px = (gap_hit_2px + slider_hit_2px) / 2
            
            gap_hit_5px = (gap_error.max(dim=1)[0] <= 5).float().sum().item()
            slider_hit_5px = (slider_error.max(dim=1)[0] <= 5).float().sum().item()
            hit_5px = (gap_hit_5px + slider_hit_5px) / 2
            
            # 更新累计值
            self.total_loss += loss * batch_size
            self.total_mae += mae * batch_size
            self.total_gap_mae += gap_mae * batch_size
            self.total_slider_mae += slider_mae * batch_size
            self.hit_at_2px += hit_2px
            self.hit_at_5px += hit_5px
            self.num_samples += batch_size
            self.num_batches += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """获取平均指标"""
        if self.num_samples == 0:
            return {}
        
        return {
            'loss': self.total_loss / self.num_samples,
            'mae': self.total_mae / self.num_samples,
            'gap_mae': self.total_gap_mae / self.num_samples,
            'slider_mae': self.total_slider_mae / self.num_samples,
            'hit_at_2px': self.hit_at_2px / self.num_samples * 100,
            'hit_at_5px': self.hit_at_5px / self.num_samples * 100
        }


def extract_coordinates_from_heatmap(
    heatmap: torch.Tensor,
    offset: torch.Tensor = None,
    scale: int = 4
) -> torch.Tensor:
    """
    从热力图提取坐标
    
    Args:
        heatmap: 热力图 [B, 1, H, W]
        offset: 偏移量 [B, 2, H, W] (可选)
        scale: 下采样倍数
        
    Returns:
        坐标 [B, 2]
    """
    B, _, H, W = heatmap.shape
    device = heatmap.device
    
    # 找到热力图最大值位置
    heatmap_flat = heatmap.view(B, -1)
    max_idx = heatmap_flat.argmax(dim=1)
    
    # 转换为2D坐标
    y_idx = max_idx // W
    x_idx = max_idx % W
    
    # 基础坐标
    coords = torch.stack([x_idx, y_idx], dim=1).float()
    
    # 应用偏移量（如果有）
    if offset is not None:
        batch_idx = torch.arange(B, device=device)
        offset_x = offset[batch_idx, 0, y_idx, x_idx]
        offset_y = offset[batch_idx, 1, y_idx, x_idx]
        coords[:, 0] += offset_x
        coords[:, 1] += offset_y
    
    # 缩放到原始尺寸
    coords *= scale
    
    return coords


def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算平均绝对误差（MAE）
    
    Args:
        pred: 预测坐标 [B, 2]
        target: 真实坐标 [B, 2]
        
    Returns:
        MAE值
    """
    return (pred - target).abs().mean().item()


def calculate_hit_rate(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 2.0
) -> float:
    """
    计算命中率
    
    Args:
        pred: 预测坐标 [B, 2]
        target: 真实坐标 [B, 2]
        threshold: 距离阈值（像素）
        
    Returns:
        命中率（百分比）
    """
    error = (pred - target).abs()
    max_error = error.max(dim=1)[0]
    hit_rate = (max_error <= threshold).float().mean().item() * 100
    return hit_rate


class CombinedMetric:
    """组合指标计算器"""
    
    def __init__(self, thresholds=[2, 5, 10]):
        """
        Args:
            thresholds: 命中率阈值列表
        """
        self.thresholds = thresholds
    
    def compute(
        self,
        gap_pred: torch.Tensor,
        gap_target: torch.Tensor,
        slider_pred: torch.Tensor,
        slider_target: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # MAE
        gap_mae = calculate_mae(gap_pred, gap_target)
        slider_mae = calculate_mae(slider_pred, slider_target)
        metrics['gap_mae'] = gap_mae
        metrics['slider_mae'] = slider_mae
        metrics['mae'] = (gap_mae + slider_mae) / 2
        
        # 命中率
        for threshold in self.thresholds:
            gap_hit = calculate_hit_rate(gap_pred, gap_target, threshold)
            slider_hit = calculate_hit_rate(slider_pred, slider_target, threshold)
            metrics[f'gap_hit@{threshold}px'] = gap_hit
            metrics[f'slider_hit@{threshold}px'] = slider_hit
            metrics[f'hit@{threshold}px'] = (gap_hit + slider_hit) / 2
        
        return metrics