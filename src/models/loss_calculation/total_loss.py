# -*- coding: utf-8 -*-
"""
总损失函数（Total Loss）
组合所有子损失函数，提供统一的损失计算接口
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from .focal_loss import create_focal_loss
from .offset_loss import create_offset_loss
from .hard_negative_loss import create_hard_negative_loss
from .angle_loss import create_angle_loss


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
            + λ_offset * L_offset
            + λ_hn * L_hard_negative
            + λ_angle * L_angle  (if enabled)
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 从loss.yaml加载的完整配置
        """
        super().__init__()
        
        # 创建必需的损失函数
        self.focal_loss = create_focal_loss(config['focal_loss'])
        self.offset_loss = create_offset_loss(config['offset_loss'])
        self.hard_negative_loss = create_hard_negative_loss(config['hard_negative_loss'])
        
        # 角度损失是可选的
        self.use_angle = config['total_loss'].get('use_angle', False)
        if self.use_angle:
            self.angle_loss = create_angle_loss(config['angle_loss'])
        
        # 损失权重
        self.weights = config['total_loss']['weights']
    
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
                - 'mask': 有效区域掩码 [B, 1, H, W]（必需）
                - 'gap_center': 真实缺口中心 [B, 2]
                - 'fake_centers': 假缺口中心列表
        
        Returns:
            total_loss: 总损失（标量）
            loss_dict: 各子损失的值（用于日志记录）
        """
        loss_dict = {}
        # mask是必需的，因为输入包含padding mask通道
        if 'mask' not in targets:
            raise ValueError("targets缺少必需的'mask'字段")
        mask = targets['mask']
        
        # ========== 1. 热力图损失（Focal Loss） ==========
        loss_heat_gap = self.focal_loss(
            predictions['heatmap_gap'],
            targets['heatmap_gap'],
            mask
        )
        
        loss_heat_piece = self.focal_loss(
            predictions['heatmap_piece'],
            targets['heatmap_piece'],
            mask
        )
        
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
        # 只有在有混淆缺口时才计算硬负样本损失
        if 'fake_centers' in targets and targets['fake_centers'] and len(targets['fake_centers']) > 0:
            try:
                loss_hard_neg = self.hard_negative_loss(
                    predictions['heatmap_gap'],
                    targets['gap_center'],
                    targets['fake_centers'],
                    mask
                )
                # 检查是否有效
                if torch.isnan(loss_hard_neg) or torch.isinf(loss_hard_neg):
                    loss_hard_neg = torch.tensor(0.0, device=predictions['heatmap_gap'].device, requires_grad=True)
            except Exception as e:
                loss_hard_neg = torch.tensor(0.0, device=predictions['heatmap_gap'].device, requires_grad=True)
        else:
            # 没有混淆缺口时，硬负样本损失为0
            loss_hard_neg = torch.tensor(0.0, device=predictions['heatmap_gap'].device, requires_grad=True)
        
        loss_dict['hard_negative'] = loss_hard_neg
        
        # ========== 4. 角度损失（可选） ==========
        if self.use_angle and 'angle' in predictions and 'angle' in targets and targets['angle'] is not None:
            try:
                # 只有在有角度信息时才计算角度损失
                loss_angle = self.angle_loss(
                    predictions['angle'],
                    targets['angle'],
                    targets['heatmap_gap'],  # 使用缺口热力图作为监督区域
                    mask
                )
                # 检查是否有效
                if torch.isnan(loss_angle) or torch.isinf(loss_angle):
                    loss_angle = torch.tensor(0.0, device=predictions['heatmap_gap'].device, requires_grad=True)
            except Exception as e:
                loss_angle = torch.tensor(0.0, device=predictions['heatmap_gap'].device, requires_grad=True)
        else:
            # 不使用角度损失时，或没有旋转角度时，设为0
            loss_angle = torch.tensor(0.0, device=predictions['heatmap_gap'].device, requires_grad=True)
        
        loss_dict['angle'] = loss_angle
        
        # ========== 5. 计算总损失 ==========
        valid_losses = []
        
        # 必须有的损失（focal和offset）
        if not torch.isnan(loss_heatmap) and not torch.isinf(loss_heatmap):
            valid_losses.append(self.weights['heatmap'] * loss_heatmap)
            
        if not torch.isnan(loss_offset) and not torch.isinf(loss_offset):
            valid_losses.append(self.weights['offset'] * loss_offset)
        
        # 可选损失（hard_negative和angle）
        # 只有非零时才加入
        if loss_hard_neg.item() > 0 and not torch.isnan(loss_hard_neg) and not torch.isinf(loss_hard_neg):
            valid_losses.append(self.weights['hard_negative'] * loss_hard_neg)
            
        if self.use_angle and loss_angle.item() > 0 and not torch.isnan(loss_angle) and not torch.isinf(loss_angle):
            valid_losses.append(self.weights['angle'] * loss_angle)
        
        # 计算总损失
        if len(valid_losses) > 0:
            total_loss = sum(valid_losses)
        else:
            # 所有损失都无效时返回默认值
            total_loss = torch.tensor(1.0, device=predictions['heatmap_gap'].device, requires_grad=True)
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


def create_total_loss(config: Dict) -> TotalLoss:
    """
    工厂函数：根据配置创建总损失函数
    
    Args:
        config: 完整的损失配置（从loss.yaml加载）
    
    Returns:
        TotalLoss实例
    """
    return TotalLoss(config)