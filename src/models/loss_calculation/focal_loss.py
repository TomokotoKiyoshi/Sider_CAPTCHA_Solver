# -*- coding: utf-8 -*-
"""
CenterNet风格的Focal Loss实现
用于热力图的训练，处理正负样本不平衡问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    CenterNet风格的Focal Loss
    
    用于目标检测中心点预测的损失函数，通过加权机制处理：
    1. 正负样本不平衡问题
    2. 难易样本的关注度调节
    
    数学公式：
    L = -1/N_pos * Σ {
        (1-P)^α * log(P)           if Y ≥ t_pos (正样本)
        (1-Y)^β * P^α * log(1-P)   otherwise (负样本)
    }
    
    其中：
    - P: 预测的概率图
    - Y: 真实的高斯热图
    - α: 正样本的聚焦参数
    - β: 负样本的距离权重参数
    - t_pos: 正样本阈值
    """
    
    def __init__(self,
                 alpha: float = 1.5,
                 beta: float = 4.0,
                 pos_threshold: float = 0.9,
                 eps: float = 1e-8):
        """
        Args:
            alpha: 正样本聚焦参数，控制难易样本的关注度
                   较大的值会更关注困难的正样本
            beta: 负样本距离权重参数，根据到中心的距离加权
                  较大的值会降低远离中心的负样本权重
            pos_threshold: 正样本阈值，Y >= pos_threshold视为正样本
            eps: 防止log(0)的小常数
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_threshold = pos_threshold
        self.eps = eps
    
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播计算Focal Loss
        
        Args:
            pred: 预测的热力图 [B, 1, H, W]，已经过Sigmoid激活，值域(0,1)
            target: 真实的高斯热图 [B, 1, H, W]，值域[0,1]
            mask: 有效区域掩码 [B, 1, H, W]，1表示有效，0表示padding
        
        Returns:
            标量损失值
        """
        # 确保输入维度正确
        assert pred.dim() == 4 and target.dim() == 4, \
            f"Expected 4D tensors, got pred: {pred.dim()}D, target: {target.dim()}D"
        assert pred.shape == target.shape, \
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        
        # 防止数值不稳定
        pred = torch.clamp(pred, min=self.eps, max=1-self.eps)
        
        # 识别正负样本
        pos_mask = (target >= self.pos_threshold).float()  # 正样本掩码
        neg_mask = 1 - pos_mask  # 负样本掩码
        
        # 正样本损失：(1-P)^α * log(P)
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred)
        pos_loss = pos_loss * pos_mask  # 只在正样本位置计算
        
        # 负样本损失：(1-Y)^β * P^α * log(1-P)
        # (1-Y)^β: 距离权重，离中心越远权重越小
        neg_weights = torch.pow(1 - target, self.beta)
        neg_loss = -neg_weights * torch.pow(pred, self.alpha) * torch.log(1 - pred)
        neg_loss = neg_loss * neg_mask  # 只在负样本位置计算
        
        # 组合正负样本损失
        loss = pos_loss + neg_loss
        
        # 应用有效区域掩码
        loss = loss * mask
        pos_mask = pos_mask * mask  # 同时更新正样本掩码
        
        # 归一化：按照文档公式，使用正样本数N_pos
        # L = -1/N_pos * Σ(loss)
        num_pos = pos_mask.sum()
        
        # 检查是否有正样本
        if num_pos == 0:
            raise ValueError(
                f"No positive samples detected! Please check:\n"
                f"1. Target heatmap max value: {target.max().item():.4f}\n"
                f"2. Positive threshold: {self.pos_threshold}\n"
                f"3. Gaussian sigma parameter (may be too large)\n"
                f"4. Center point coordinates"
            )
        
        total_loss = loss.sum() / num_pos
        
        return total_loss


def create_focal_loss(config: dict) -> FocalLoss:
    """
    工厂函数：根据配置创建Focal Loss
    
    Args:
        config: 配置字典，必须包含：
            - alpha: 正样本聚焦参数
            - beta: 负样本距离权重参数
            - pos_threshold: 正样本阈值
            可选参数：
            - eps: 数值稳定性常数（默认1e-8）
    
    Returns:
        FocalLoss实例
    
    Raises:
        KeyError: 如果缺少必要的配置参数
    """
    # 必须提供的参数
    required_params = ['alpha', 'beta', 'pos_threshold']
    missing_params = [p for p in required_params if p not in config]
    if missing_params:
        raise KeyError(f"Missing required parameters: {missing_params}")
    
    return FocalLoss(
        alpha=config['alpha'],
        beta=config['beta'],
        pos_threshold=config['pos_threshold'],
        eps=config.get('eps', 1e-8)  # eps可以有默认值
    )