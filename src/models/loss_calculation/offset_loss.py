# -*- coding: utf-8 -*-
"""
子像素偏移损失（Offset Loss）实现
用于回归中心点在栅格内的连续偏移，实现亚像素精度定位
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# No Optional needed anymore


class OffsetLoss(nn.Module):
    """
    子像素偏移损失 - Smooth L1（正样本内）
    
    预测张量：O=[B,4,64,128]，四通道分别是(du_g, dv_g, du_p, dv_p)
    网络输出通过tanh×0.5映射到[-0.5, 0.5]
    
    数学公式：
    L_off = Σ(w_g * l_off(g)) / (ε + Σw_g)
    
    其中：
    - l_off(g) = SmoothL1(d̂_x(g) - d*_x(g)) + SmoothL1(d̂_y(g) - d*_y(g))
    - w_g: 基于热图值的权重
    """
    
    def __init__(self,
                 beta: float = 1.0,
                 eps: float = 1e-6):
        """
        Args:
            beta: Smooth L1的平滑参数
            eps: 防止除零的小常数
        """
        super().__init__()
        self.beta = beta
        self.eps = eps
        
        # Smooth L1损失
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction='none')
    
    def forward(self,
                pred_offset: torch.Tensor,
                target_offset: torch.Tensor,
                heatmap: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算偏移损失
        
        Args:
            pred_offset: 预测偏移 O=[B,4,64,128]
                        通道顺序：(du_g, dv_g, du_p, dv_p)
            target_offset: 真实偏移 [B,4,64,128]
            heatmap: 真实高斯热图 [B,2,64,128]
                    第0通道为gap热图，第1通道为piece热图
            mask: 有效区域掩码 W_1/4 [B,1,H,W]（必需）
        
        Returns:
            标量损失值
        """
        B, _, H, W = pred_offset.shape
        
        # 分别计算gap和piece的偏移损失
        total_loss = 0.0
        
        # Gap偏移损失
        # 读取对应预测：d̂_x(g) = O[b,0,:,:], d̂_y(g) = O[b,1,:,:]
        pred_dx_g = pred_offset[:, 0:1, :, :]  # [B,1,H,W]
        pred_dy_g = pred_offset[:, 1:2, :, :]  # [B,1,H,W]
        target_dx_g = target_offset[:, 0:1, :, :]
        target_dy_g = target_offset[:, 1:2, :, :]
        
        # Gap热图作为权重 w_g
        gap_heatmap = heatmap[:, 0:1, :, :]  # [B,1,H,W]
        w_g = gap_heatmap * mask  # 直接应用mask
        
        # 检查权重和不为0
        w_g_sum = w_g.sum()
        if w_g_sum < self.eps:
            # 权重过小时返回小的常数损失
            return torch.tensor(0.01, device=pred_offset.device, requires_grad=True)
        
        # 计算单样本Smooth L1
        # l_off(g) = SmoothL1(d̂_x(g) - d*_x(g)) + SmoothL1(d̂_y(g) - d*_y(g))
        loss_dx_g = self.smooth_l1(pred_dx_g, target_dx_g)
        loss_dy_g = self.smooth_l1(pred_dy_g, target_dy_g)
        l_off_g = (loss_dx_g + loss_dy_g) * w_g
        
        # L_off_g = Σ(w_g * l_off(g)) / Σw_g
        gap_loss = l_off_g.sum() / w_g_sum
        total_loss = total_loss + gap_loss
        
        # Piece偏移损失
        # 读取对应预测：d̂_x(p) = O[b,2,:,:], d̂_y(p) = O[b,3,:,:]
        pred_dx_p = pred_offset[:, 2:3, :, :]  # [B,1,H,W]
        pred_dy_p = pred_offset[:, 3:4, :, :]  # [B,1,H,W]
        target_dx_p = target_offset[:, 2:3, :, :]
        target_dy_p = target_offset[:, 3:4, :, :]
        
        # Piece热图作为权重 w_p
        piece_heatmap = heatmap[:, 1:2, :, :]  # [B,1,H,W]
        w_p = piece_heatmap * mask  # 直接应用mask
        
        # 检查权重和不为0
        w_p_sum = w_p.sum()
        if w_p_sum < self.eps:
            # 权重过小时返回gap损失加上小的常数
            return gap_loss + torch.tensor(0.01, device=pred_offset.device, requires_grad=True)
        
        # 计算单样本Smooth L1
        # l_off(p) = SmoothL1(d̂_x(p) - d*_x(p)) + SmoothL1(d̂_y(p) - d*_y(p))
        loss_dx_p = self.smooth_l1(pred_dx_p, target_dx_p)
        loss_dy_p = self.smooth_l1(pred_dy_p, target_dy_p)
        l_off_p = (loss_dx_p + loss_dy_p) * w_p
        
        # L_off_p = Σ(w_p * l_off(p)) / Σw_p
        piece_loss = l_off_p.sum() / w_p_sum
        total_loss = total_loss + piece_loss
        
        return total_loss


def create_offset_loss(config: dict) -> OffsetLoss:
    """
    工厂函数：根据配置创建偏移损失
    
    Args:
        config: 配置字典，必须包含：
            - beta: Smooth L1的平滑参数
            - eps: 防止除零的小常数（可选，默认1e-6）
    Returns:
        OffsetLoss实例
    """
    # 必须提供的参数
    required_params = ['beta']
    missing_params = [p for p in required_params if p not in config]
    if missing_params:
        raise KeyError(f"Missing required parameters: {missing_params}")
    
    return OffsetLoss(
        beta=config['beta'],
        eps=config.get('eps', 1e-6)  # eps可以有默认值
    )