# -*- coding: utf-8 -*-
"""
总损失函数（Total Loss）
组合所有子损失函数，提供统一的损失计算接口
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from .focal_loss import FocalLoss, create_focal_loss
from .offset_loss import OffsetLoss, create_offset_loss
from .hard_negative_loss import HardNegativeLoss, create_hard_negative_loss
from .angle_loss import AngleLoss, create_angle_loss


class TotalLoss(nn.Module):
    """
    总损失函数
    
    组合多个子损失：
    1. Focal Loss（热力图损失）- 用于缺口和滑块中心预测
    2. Offset Loss（偏移损失）- 用于亚像素精度
    3. Hard Negative Loss（硬负样本损失）- 用于假缺口抑制
    4. Angle Loss（角度损失）- 用于微旋转预测（可选）
    
    总损失公式：
    L_total = λ_heat * (L_focal_gap + L_focal_piece)
            + λ_offset * (L_offset_gap + L_offset_piece)
            + λ_hn * L_hard_negative
            + λ_angle * L_angle  (if enabled)
    """
    
    def __init__(self,
                 focal_config: dict = None,
                 offset_config: dict = None,
                 hard_negative_config: dict = None,
                 angle_config: dict = None,
                 loss_weights: dict = None,
                 use_angle: bool = False,
                 use_hard_negative: bool = True):
        """
        Args:
            focal_config: Focal Loss配置
            offset_config: Offset Loss配置
            hard_negative_config: Hard Negative Loss配置
            angle_config: Angle Loss配置
            loss_weights: 各损失的权重
            use_angle: 是否使用角度损失
            use_hard_negative: 是否使用硬负样本损失
        """
        super().__init__()
        
        # 默认配置
        self.focal_config = focal_config or {
            'loss_type': 'modified_focal',
            'alpha': 2.0,
            'beta': 4.0,
            'pos_threshold': 0.9,
            'norm_by_pos': True
        }
        
        self.offset_config = offset_config or {
            'loss_class': 'weighted_offset',
            'loss_type': 'smooth_l1',
            'beta': 1.0,
            'pos_threshold': 0.7,
            'use_heatmap_weight': True
        }
        
        self.hard_negative_config = hard_negative_config or {
            'loss_class': 'hard_negative',
            'margin': 0.2,
            'score_type': 'bilinear',
            'neighborhood_size': 3,
            'reduction': 'mean'
        }
        
        self.angle_config = angle_config or {
            'loss_class': 'weighted_angle',
            'loss_type': 'cosine',
            'pos_threshold': 0.7,
            'temperature': 1.0
        }
        
        # 损失权重（根据文档设置）
        self.loss_weights = loss_weights or {
            'heatmap': 1.0,      # 热力图损失权重
            'offset': 1.0,       # 偏移损失权重（文档中λ=1.0）
            'hard_negative': 0.5, # 硬负样本损失权重
            'angle': 0.5         # 角度损失权重
        }
        
        # 使用标志
        self.use_angle = use_angle
        self.use_hard_negative = use_hard_negative
        
        # 创建子损失函数
        self.focal_loss = create_focal_loss(self.focal_config)
        self.offset_loss = create_offset_loss(self.offset_config)
        
        if self.use_hard_negative:
            self.hard_negative_loss = create_hard_negative_loss(self.hard_negative_config)
        
        if self.use_angle:
            self.angle_loss = create_angle_loss(self.angle_config)
        
        # 损失记录（用于日志）
        self.loss_history = {}
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播计算总损失
        
        Args:
            predictions: 模型预测，包含：
                - 'heatmap_gap': 缺口热力图 [B, 1, H, W]
                - 'heatmap_piece': 滑块热力图 [B, 1, H, W]
                - 'offset': 偏移预测 [B, 4, H, W]
                - 'angle': 角度预测 [B, 2, H, W]（可选）
            
            targets: 目标值，包含：
                - 'heatmap_gap': 缺口高斯热图 [B, 1, H, W]
                - 'heatmap_piece': 滑块高斯热图 [B, 1, H, W]
                - 'offset': 目标偏移 [B, 4, H, W]
                - 'angle': 目标角度 [B, 2, H, W]（可选）
                - 'mask': 有效区域掩码 [B, 1, H, W]（可选）
                - 'gap_center': 真实缺口中心 [B, 2]（用于硬负样本）
                - 'fake_centers': 假缺口中心列表（用于硬负样本）
        
        Returns:
            total_loss: 总损失（标量）
            loss_dict: 各子损失的值（用于日志记录）
        """
        loss_dict = {}
        
        # 获取掩码
        mask = targets.get('mask', None)
        
        # ========== 1. 热力图损失（Focal Loss） ==========
        # 缺口热力图损失
        loss_heat_gap = self.focal_loss(
            predictions['heatmap_gap'],
            targets['heatmap_gap'],
            mask
        )
        loss_dict['heat_gap'] = loss_heat_gap
        
        # 滑块热力图损失
        loss_heat_piece = self.focal_loss(
            predictions['heatmap_piece'],
            targets['heatmap_piece'],
            mask
        )
        loss_dict['heat_piece'] = loss_heat_piece
        
        # 合并热力图损失
        loss_heatmap = loss_heat_gap + loss_heat_piece
        loss_dict['heatmap'] = loss_heatmap
        
        # ========== 2. 偏移损失（Offset Loss） ==========
        # 使用真实热力图作为正样本掩码
        heatmap_combined = torch.cat([
            targets['heatmap_gap'],
            targets['heatmap_piece']
        ], dim=1)  # [B, 2, H, W]
        
        loss_offset = self.offset_loss(
            predictions['offset'],
            targets['offset'],
            heatmap_combined,
            mask
        )
        loss_dict['offset'] = loss_offset
        
        # ========== 3. 硬负样本损失（可选） ==========
        if self.use_hard_negative and 'fake_centers' in targets:
            loss_hard_neg = self.hard_negative_loss(
                predictions['heatmap_gap'],
                targets['gap_center'],
                targets['fake_centers'],
                mask
            )
            loss_dict['hard_negative'] = loss_hard_neg
        else:
            loss_hard_neg = torch.tensor(0.0, device=predictions['heatmap_gap'].device, requires_grad=True)
            loss_dict['hard_negative'] = loss_hard_neg
        
        # ========== 4. 角度损失（可选） ==========
        if self.use_angle and 'angle' in predictions:
            loss_angle = self.angle_loss(
                predictions['angle'],
                targets['angle'],
                targets['heatmap_gap'],  # 使用缺口热力图作为监督区域
                mask
            )
            loss_dict['angle'] = loss_angle
        else:
            loss_angle = torch.tensor(0.0, device=predictions['heatmap_gap'].device, requires_grad=True)
            loss_dict['angle'] = loss_angle
        
        # ========== 5. 计算总损失 ==========
        total_loss = (
            self.loss_weights['heatmap'] * loss_heatmap +
            self.loss_weights['offset'] * loss_offset +
            self.loss_weights['hard_negative'] * loss_hard_neg +
            self.loss_weights['angle'] * loss_angle
        )
        
        loss_dict['total'] = total_loss
        
        # 更新损失历史
        self._update_history(loss_dict)
        
        return total_loss, loss_dict
    
    def _update_history(self, loss_dict: Dict[str, torch.Tensor]):
        """更新损失历史记录"""
        for key, value in loss_dict.items():
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(value.item() if torch.is_tensor(value) else value)
    
    def get_loss_summary(self) -> Dict[str, float]:
        """获取损失统计摘要"""
        summary = {}
        for key, values in self.loss_history.items():
            if len(values) > 0:
                summary[f'{key}_mean'] = sum(values) / len(values)
                summary[f'{key}_min'] = min(values)
                summary[f'{key}_max'] = max(values)
        return summary
    
    def reset_history(self):
        """重置损失历史"""
        self.loss_history = {}


class AdaptiveTotalLoss(TotalLoss):
    """
    自适应总损失函数
    
    扩展功能：
    1. 动态调整损失权重
    2. 基于训练进度的权重调度
    3. 基于损失值的自适应平衡
    """
    
    def __init__(self,
                 focal_config: dict = None,
                 offset_config: dict = None,
                 hard_negative_config: dict = None,
                 angle_config: dict = None,
                 initial_weights: dict = None,
                 weight_schedule: str = 'constant',
                 balance_losses: bool = True,
                 use_angle: bool = False,
                 use_hard_negative: bool = True):
        """
        Args:
            initial_weights: 初始损失权重
            weight_schedule: 权重调度策略 ('constant', 'linear', 'cosine')
            balance_losses: 是否自动平衡损失
            其他参数同基类
        """
        super().__init__(
            focal_config, offset_config, hard_negative_config, angle_config,
            initial_weights, use_angle, use_hard_negative
        )
        
        self.initial_weights = self.loss_weights.copy()
        self.weight_schedule = weight_schedule
        self.balance_losses = balance_losses
        self.current_epoch = 0
        self.total_epochs = 100  # 默认值
        
        # 用于自适应平衡的移动平均
        self.loss_ema = {}
        self.ema_alpha = 0.99
    
    def update_epoch(self, epoch: int, total_epochs: int):
        """
        更新当前epoch，用于权重调度
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
        """
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        
        # 更新权重
        if self.weight_schedule == 'linear':
            # 线性调度
            progress = epoch / total_epochs
            for key in self.loss_weights:
                if key == 'heatmap':
                    # 热力图权重逐渐降低
                    self.loss_weights[key] = self.initial_weights[key] * (1 - 0.3 * progress)
                elif key == 'offset':
                    # 偏移权重逐渐增加
                    self.loss_weights[key] = self.initial_weights[key] * (1 + 0.5 * progress)
                elif key == 'hard_negative':
                    # 硬负样本权重在中期增加
                    if progress < 0.5:
                        self.loss_weights[key] = self.initial_weights[key] * (1 + progress)
                    else:
                        self.loss_weights[key] = self.initial_weights[key] * 1.5
        
        elif self.weight_schedule == 'cosine':
            # 余弦退火
            import numpy as np
            progress = epoch / total_epochs
            cosine_factor = (1 + np.cos(np.pi * progress)) / 2
            
            for key in self.loss_weights:
                if key == 'angle':
                    # 角度损失后期增强
                    self.loss_weights[key] = self.initial_weights[key] * (2 - cosine_factor)
                else:
                    self.loss_weights[key] = self.initial_weights[key]
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算自适应总损失
        """
        # 计算基础损失
        total_loss, loss_dict = super().forward(predictions, targets)
        
        # 自适应损失平衡
        if self.balance_losses and len(self.loss_ema) > 0:
            # 更新EMA
            for key in ['heatmap', 'offset', 'hard_negative', 'angle']:
                if key in loss_dict:
                    if key not in self.loss_ema:
                        self.loss_ema[key] = loss_dict[key].item()
                    else:
                        self.loss_ema[key] = (
                            self.ema_alpha * self.loss_ema[key] +
                            (1 - self.ema_alpha) * loss_dict[key].item()
                        )
            
            # 计算平衡权重
            mean_loss = sum(self.loss_ema.values()) / len(self.loss_ema)
            for key in self.loss_ema:
                if self.loss_ema[key] > 0:
                    balance_factor = mean_loss / self.loss_ema[key]
                    balance_factor = max(0.5, min(2.0, balance_factor))  # 限制范围
                    self.loss_weights[key] *= balance_factor
        
        # 重新计算总损失（使用平衡后的权重）
        if self.balance_losses:
            total_loss = sum(
                self.loss_weights.get(key, 1.0) * loss_dict.get(key, 0)
                for key in ['heatmap', 'offset', 'hard_negative', 'angle']
            )
            loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


def create_total_loss(config: dict) -> TotalLoss:
    """
    工厂函数：根据配置创建总损失函数
    
    Args:
        config: 配置字典，包含：
            - loss_class: 'total' 或 'adaptive_total'
            - focal_config: Focal Loss配置
            - offset_config: Offset Loss配置
            - hard_negative_config: Hard Negative Loss配置
            - angle_config: Angle Loss配置
            - loss_weights: 损失权重
            - use_angle: 是否使用角度损失
            - use_hard_negative: 是否使用硬负样本损失
            - weight_schedule: 权重调度策略（仅adaptive）
            - balance_losses: 是否自动平衡（仅adaptive）
    
    Returns:
        TotalLoss实例
    """
    loss_class = config.get('loss_class', 'total')
    
    base_params = {
        'focal_config': config.get('focal_config'),
        'offset_config': config.get('offset_config'),
        'hard_negative_config': config.get('hard_negative_config'),
        'angle_config': config.get('angle_config'),
        'loss_weights': config.get('loss_weights'),
        'use_angle': config.get('use_angle', False),
        'use_hard_negative': config.get('use_hard_negative', True)
    }
    
    if loss_class == 'adaptive_total':
        return AdaptiveTotalLoss(
            **base_params,
            initial_weights=base_params['loss_weights'],
            weight_schedule=config.get('weight_schedule', 'constant'),
            balance_losses=config.get('balance_losses', True)
        )
    else:
        return TotalLoss(**base_params)