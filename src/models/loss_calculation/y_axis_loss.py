#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Y-Axis Loss for Slider CAPTCHA
确保gap和piece中心位于不同的y轴位置，抑制"假同行"现象
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class YAxisLoss(nn.Module):
    """
    Y轴损失函数
    
    包含两部分：
    1. Row CE Loss: 行分类损失，确保预测正确的y坐标
    2. EMD Loss: 分布约束，确保gap和piece在不同y轴位置
    
    Args:
        tau: Softmax温度参数 (default: 1.0)
        eps: 数值稳定性参数 (default: 1e-8)
        log_stretch: 是否使用对数伸展 (default: True)
        row_ce_weight: 行分类损失权重 (default: 1.0)
        emd_weight: EMD损失权重 (default: 0.3)
    """
    
    def __init__(self, 
                 tau: float = 1.0,
                 eps: float = 1e-8,
                 log_stretch: bool = True,
                 row_ce_weight: float = 1.0,
                 emd_weight: float = 0.3):
        super().__init__()
        self.tau = tau
        self.eps = eps
        self.log_stretch = log_stretch
        self.row_ce_weight = row_ce_weight
        self.emd_weight = emd_weight
        
    def forward(self,
                heatmap_gap: torch.Tensor,
                heatmap_piece: torch.Tensor,
                gap_y: torch.Tensor,
                piece_y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算Y轴损失
        
        Args:
            heatmap_gap: 缺口热力图 [B, 1, H, W] 或 [B, H, W]
            heatmap_piece: 滑块热力图 [B, 1, H, W] 或 [B, H, W]
            gap_y: 缺口真实y坐标 [B] (原始分辨率，需要/4)
            piece_y: 滑块真实y坐标 [B] (原始分辨率，需要/4)
            
        Returns:
            total_loss: 总Y轴损失
            loss_dict: 各分项损失字典
        """
        # 确保热力图是3维的 [B, H, W]
        if heatmap_gap.dim() == 4:
            heatmap_gap = heatmap_gap.squeeze(1)
        if heatmap_piece.dim() == 4:
            heatmap_piece = heatmap_piece.squeeze(1)
            
        batch_size, height, width = heatmap_gap.shape
        device = heatmap_gap.device
        
        # ========== 0. 热力图到行分布 ==========
        # 计算每行平均能量
        row_energy_gap = heatmap_gap.mean(dim=2)  # [B, H]
        row_energy_piece = heatmap_piece.mean(dim=2)  # [B, H]
        
        # 对数伸展（可选）
        if self.log_stretch:
            row_energy_gap = torch.log(row_energy_gap + self.eps)
            row_energy_piece = torch.log(row_energy_piece + self.eps)
        
        # Softmax归一化为行分布
        row_dist_gap = F.softmax(row_energy_gap / self.tau, dim=1)  # [B, H]
        row_dist_piece = F.softmax(row_energy_piece / self.tau, dim=1)  # [B, H]
        
        # ========== 1. 行分类损失 (Row CE) ==========
        # 将真实y坐标转换为1/4分辨率的行索引
        # 假设输入坐标是原始分辨率(256)，需要除以4得到热力图分辨率(64)
        target_y_gap = (gap_y / 4.0).long().clamp(0, height - 1)  # [B]
        target_y_piece = (piece_y / 4.0).long().clamp(0, height - 1)  # [B]
        
        # 计算交叉熵损失
        # 需要先取对数，因为row_dist已经是概率分布
        log_row_dist_gap = torch.log(row_dist_gap + self.eps)  # [B, H]
        log_row_dist_piece = torch.log(row_dist_piece + self.eps)  # [B, H]
        
        # 使用gather获取目标位置的对数概率
        row_ce_gap = -log_row_dist_gap.gather(1, target_y_gap.unsqueeze(1)).squeeze(1)  # [B]
        row_ce_piece = -log_row_dist_piece.gather(1, target_y_piece.unsqueeze(1)).squeeze(1)  # [B]
        
        # 总行分类损失
        row_ce_loss = (row_ce_gap + row_ce_piece).mean()
        
        # ========== 2. EMD损失 ==========
        # 计算累积分布函数 (CDF)
        cdf_gap = torch.cumsum(row_dist_gap, dim=1)  # [B, H]
        cdf_piece = torch.cumsum(row_dist_piece, dim=1)  # [B, H]
        
        # EMD = L1距离的平均
        emd_loss = torch.abs(cdf_gap - cdf_piece).mean()
        
        # ========== 3. 总Y轴损失 ==========
        total_loss = self.row_ce_weight * row_ce_loss + self.emd_weight * emd_loss
        
        # 构建损失字典
        loss_dict = {
            'y_axis_total': total_loss,
            'y_axis_row_ce': row_ce_loss,
            'y_axis_emd': emd_loss,
            'y_axis_row_ce_gap': row_ce_gap.mean(),
            'y_axis_row_ce_piece': row_ce_piece.mean()
        }
        
        return total_loss, loss_dict


def create_y_axis_loss(config: Dict) -> YAxisLoss:
    """
    工厂函数：根据配置创建Y轴损失函数
    
    Args:
        config: Y轴损失配置
        
    Returns:
        YAxisLoss实例
    """
    return YAxisLoss(
        tau=config.get('tau', 1.0),
        eps=config.get('eps', 1e-8),
        log_stretch=config.get('log_stretch', True),
        row_ce_weight=config.get('row_ce_weight', 1.0),
        emd_weight=config.get('emd_weight', 0.3)
    )