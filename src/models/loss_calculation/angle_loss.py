# -*- coding: utf-8 -*-
"""
角度损失（Angle Loss）实现
用于预测缺口的微小旋转角度（0.5-1.8度）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleLoss(nn.Module):
    """
    角度损失函数
    
    作用：建模0.5–1.8°的微旋，抑制"形似但角度偏"的假峰。
    
    预测：θ=[B,2,64,128]（L2归一化后表示(sin̂θ, coŝθ)）
    标签：(sinθ_g, cosθ_g)（来自合成器）
    
    仅在gap中心邻域监督：M_ang = 1(Y_gap > 0.7)
    
    屏蔽归一化：
    L_ang = Σ[1 - (sin̂θ*sinθ_g + coŝθ*cosθ_g)] * M_ang * W_1/4 / (Σ(M_ang*W_1/4)+ε)
    """
    
    def __init__(self,
                 pos_threshold: float = 0.7,
                 eps: float = 1e-8):
        """
        Args:
            pos_threshold: 正样本阈值，热力图>该值才计算角度损失
            eps: 防止除零的小常数
        """
        super().__init__()
        self.pos_threshold = pos_threshold
        self.eps = eps
    
    def forward(self,
                pred_angle: torch.Tensor,
                target_angle: torch.Tensor,
                heatmap: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算角度损失
        
        Args:
            pred_angle: 预测的角度 [B, 2, H, W]
                       通道0: sin θ
                       通道1: cos θ
                       已经过L2归一化
            target_angle: 目标角度 [B, 2, H, W]
                         格式同pred_angle
            heatmap: 缺口热力图 [B, 1, H, W]
                    用于确定监督区域
            mask: 有效区域掩码 W_1/4 [B, 1, H, W]
        
        Returns:
            标量损失值
        """
        # 确保预测已归一化（L2归一化）
        pred_angle = F.normalize(pred_angle, p=2, dim=1)
        
        # 创建监督掩码（仅在gap中心邻域）
        M_ang = (heatmap > self.pos_threshold).float()
        
        # 提取sin和cos分量
        sin_pred = pred_angle[:, 0:1, :, :]
        cos_pred = pred_angle[:, 1:2, :, :]
        sin_target = target_angle[:, 0:1, :, :]
        cos_target = target_angle[:, 1:2, :, :]
        
        # 计算余弦相似度损失：1 - (sin̂θ*sinθ_g + coŝθ*cosθ_g)
        cosine_sim = sin_pred * sin_target + cos_pred * cos_target
        loss = (1 - cosine_sim) * M_ang
        
        # 应用有效区域掩码W_1/4
        loss = loss * mask
        weight = M_ang * mask
        
        # 屏蔽归一化
        weight_sum = weight.sum()
        loss_sum = loss.sum()
        
        # L_ang = Σ[loss * M_ang * W_1/4] / (Σ(M_ang * W_1/4) + ε)
        return loss_sum / (weight_sum + self.eps)


def create_angle_loss(config: dict) -> AngleLoss:
    """
    工厂函数：根据配置创建角度损失
    
    Args:
        config: 配置字典，必须包含：
            - pos_threshold: 正样本阈值
            - eps: 防止除零的小常数（可选）
    
    Returns:
        AngleLoss实例
    """
    # 必须提供的参数
    required_params = ['pos_threshold']
    missing_params = [p for p in required_params if p not in config]
    if missing_params:
        raise KeyError(f"Missing required parameters: {missing_params}")
    
    return AngleLoss(
        pos_threshold=config['pos_threshold'],
        eps=config.get('eps', 1e-8)
    )