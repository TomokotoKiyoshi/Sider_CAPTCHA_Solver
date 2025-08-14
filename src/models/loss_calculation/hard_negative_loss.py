# -*- coding: utf-8 -*-
"""
硬负样本抑制损失（Hard Negative Loss）实现
用于显式抑制混淆缺口（假缺口）的响应
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class HardNegativeLoss(nn.Module):
    """
    Margin Ranking损失用于假缺口抑制
    
    强制真实缺口的响应比假缺口高至少一个margin值。
    
    数学公式：
    L_hn = 1/K * Σ_{k=1}^K max(0, m - s^+ + s_k^-)
    
    其中：
    - s^+: 真实缺口位置的得分
    - s_k^-: 第k个假缺口位置的得分
    - m: margin值，强制的最小得分差
    - K: 假缺口的数量（1-3个）
    """
    
    def __init__(self,
                 margin: float = 0.2,
                 reduction: str = 'mean'):
        """
        Args:
            margin: 真假缺口之间的最小得分差（m=0.2~0.3）
            reduction: 归约方式 ('mean', 'sum')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self,
                heatmap: torch.Tensor,
                true_centers: torch.Tensor,
                fake_centers: List[torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播计算硬负样本损失
        
        Args:
            heatmap: 预测的缺口热力图 [B, 1, H, W]
            true_centers: 真实缺口中心坐标 [B, 2]，格式(u_g, v_g)
            fake_centers: 假缺口中心坐标列表，每个元素 [B, 2]
            mask: 有效区域掩码（W_1/4）
        
        Returns:
            标量损失值
        """
        B = heatmap.shape[0]
        device = heatmap.device
        
        # 计算真实缺口的得分 s^+
        pos_scores = self._bilinear_sample(heatmap, true_centers)
        
        # 计算假缺口的得分 s_k^-
        neg_scores_list = []
        for fake_center in fake_centers:
            neg_score = self._bilinear_sample(heatmap, fake_center)
            neg_scores_list.append(neg_score)
        
        if len(neg_scores_list) == 0:
            # 没有假缺口，返回0损失
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 堆叠假缺口得分 [B, K]
        neg_scores = torch.stack(neg_scores_list, dim=1)
        K = neg_scores.shape[1]
        
        # 计算margin ranking损失
        # L = max(0, m - s^+ + s^-)
        pos_scores = pos_scores.unsqueeze(1)  # [B, 1]
        losses = torch.clamp(
            self.margin - pos_scores + neg_scores,
            min=0
        )  # [B, K]
        
        # 只在有效区域计算损失
        if mask is not None:
            # 检查中心点是否在有效区域
            # 这里简化处理：假设传入的中心点都在有效区域
            pass
        
        # 归约：1/K * Σ
        loss = losses.sum(dim=1) / K  # [B]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _bilinear_sample(self,
                        heatmap: torch.Tensor,
                        centers: torch.Tensor) -> torch.Tensor:
        """
        双线性插值采样
        
        Args:
            heatmap: [B, 1, H, W]
            centers: [B, 2]，栅格坐标(u, v)
        
        Returns:
            采样值 [B]
        """
        B = centers.shape[0]
        _, _, H, W = heatmap.shape
        
        # 归一化坐标到[-1, 1]
        u = centers[:, 0:1]  # [B, 1]
        v = centers[:, 1:2]  # [B, 1]
        
        u_norm = 2.0 * u / (W - 1) - 1.0
        v_norm = 2.0 * v / (H - 1) - 1.0
        
        # 构建采样网格 [B, 1, 1, 2]
        grid = torch.cat([u_norm, v_norm], dim=1)
        grid = grid.unsqueeze(1).unsqueeze(1)
        
        # 采样
        sampled = F.grid_sample(
            heatmap,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # 返回 [B]
        return sampled.squeeze()
    


def create_hard_negative_loss(config: dict) -> HardNegativeLoss:
    """
    工厂函数：根据配置创建硬负样本损失
    
    Args:
        config: 配置字典，必须包含：
            - margin: margin值
            - score_type: 得分计算方式
            - neighborhood_size: 邻域大小
            - reduction: 归约方式
    
    Returns:
        HardNegativeLoss实例
    """
    # 必须提供的参数
    required_params = ['margin', 'reduction']
    missing_params = [p for p in required_params if p not in config]
    if missing_params:
        raise KeyError(f"Missing required parameters: {missing_params}")
    
    return HardNegativeLoss(
        margin=config['margin'],
        reduction=config['reduction']
    )