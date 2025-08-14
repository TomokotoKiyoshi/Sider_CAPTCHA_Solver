# -*- coding: utf-8 -*-
"""
角度损失（Angle Loss）实现
用于预测缺口的微小旋转角度（0.5-1.8度）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AngleLoss(nn.Module):
    """
    角度损失函数
    
    用于回归缺口的微小旋转角度，通过预测(sin θ, cos θ)
    来表示角度，并使用余弦相似度作为损失。
    
    特点：
    1. 使用(sin θ, cos θ)表示避免角度周期性问题
    2. L2归一化确保sin²θ + cos²θ = 1
    3. 仅在缺口中心邻域监督
    4. 支持加权损失计算
    """
    
    def __init__(self,
                 loss_type: str = 'cosine',
                 pos_threshold: float = 0.7,
                 temperature: float = 1.0,
                 reduction: str = 'mean'):
        """
        Args:
            loss_type: 损失类型
                      'cosine': 1 - cos(θ_pred, θ_true)
                      'l2': L2距离损失
                      'smooth_l1': Smooth L1损失
            pos_threshold: 正样本阈值，热力图>该值才计算角度损失
            temperature: 温度参数，控制损失的尺度
            reduction: 归约方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.loss_type = loss_type
        self.pos_threshold = pos_threshold
        self.temperature = temperature
        self.reduction = reduction
        
        # 验证参数
        assert loss_type in ['cosine', 'l2', 'smooth_l1'], \
            f"Invalid loss_type: {loss_type}"
    
    def forward(self,
                pred_angle: torch.Tensor,
                target_angle: torch.Tensor,
                heatmap: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            mask: 有效区域掩码 [B, 1, H, W]
        
        Returns:
            标量损失值
        """
        B, _, H, W = pred_angle.shape
        
        # 验证输入
        assert pred_angle.shape[1] == 2, \
            f"Expected 2 channels for (sin, cos), got {pred_angle.shape[1]}"
        assert pred_angle.shape == target_angle.shape, \
            f"Shape mismatch: pred {pred_angle.shape} vs target {target_angle.shape}"
        
        # 确保预测已归一化（可选的安全检查）
        pred_angle = F.normalize(pred_angle, p=2, dim=1)
        
        # 创建监督掩码（仅在缺口中心附近）
        angle_mask = (heatmap > self.pos_threshold).float()
        
        # 应用有效区域掩码
        if mask is not None:
            angle_mask = angle_mask * mask
        
        # 计算损失
        if self.loss_type == 'cosine':
            # 余弦相似度损失：1 - cos(θ_pred, θ_true)
            loss = self._cosine_loss(pred_angle, target_angle, angle_mask)
        elif self.loss_type == 'l2':
            # L2距离损失
            loss = self._l2_loss(pred_angle, target_angle, angle_mask)
        else:  # 'smooth_l1'
            # Smooth L1损失
            loss = self._smooth_l1_loss(pred_angle, target_angle, angle_mask)
        
        # 应用温度缩放
        loss = loss / self.temperature
        
        # 归约
        if self.reduction == 'mean':
            num_valid = angle_mask.sum() + 1e-8
            return loss.sum() / num_valid
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
    
    def _cosine_loss(self,
                     pred: torch.Tensor,
                     target: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """
        余弦相似度损失
        
        L = 1 - (sin_pred * sin_true + cos_pred * cos_true)
        
        Args:
            pred: [B, 2, H, W]
            target: [B, 2, H, W]
            mask: [B, 1, H, W]
        
        Returns:
            损失图 [B, 1, H, W]
        """
        # 提取sin和cos分量
        sin_pred = pred[:, 0:1, :, :]
        cos_pred = pred[:, 1:2, :, :]
        sin_target = target[:, 0:1, :, :]
        cos_target = target[:, 1:2, :, :]
        
        # 计算余弦相似度
        cosine_sim = sin_pred * sin_target + cos_pred * cos_target
        
        # 损失 = 1 - 余弦相似度
        loss = (1 - cosine_sim) * mask
        
        return loss
    
    def _l2_loss(self,
                 pred: torch.Tensor,
                 target: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        """
        L2距离损失
        
        Args:
            pred: [B, 2, H, W]
            target: [B, 2, H, W]
            mask: [B, 1, H, W]
        
        Returns:
            损失图 [B, 1, H, W]
        """
        # 计算L2距离
        diff = pred - target  # [B, 2, H, W]
        l2_dist = torch.norm(diff, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 应用掩码
        loss = l2_dist * mask
        
        return loss
    
    def _smooth_l1_loss(self,
                        pred: torch.Tensor,
                        target: torch.Tensor,
                        mask: torch.Tensor,
                        beta: float = 1.0) -> torch.Tensor:
        """
        Smooth L1损失
        
        Args:
            pred: [B, 2, H, W]
            target: [B, 2, H, W]
            mask: [B, 1, H, W]
            beta: 平滑参数
        
        Returns:
            损失图 [B, 1, H, W]
        """
        # 计算Smooth L1
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # Smooth L1公式
        smooth_l1 = torch.where(
            abs_diff < beta,
            0.5 * diff ** 2 / beta,
            abs_diff - 0.5 * beta
        )
        
        # 对sin和cos分量求和
        loss = smooth_l1.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 应用掩码
        loss = loss * mask
        
        return loss
    
    @staticmethod
    def angle_to_sincos(angle_deg: torch.Tensor) -> torch.Tensor:
        """
        将角度转换为(sin θ, cos θ)表示
        
        Args:
            angle_deg: 角度（度） [B] 或 [B, H, W]
        
        Returns:
            (sin θ, cos θ) 表示，增加了通道维度
        """
        # 转换为弧度
        angle_rad = angle_deg * (torch.pi / 180.0)
        
        # 计算sin和cos
        sin_theta = torch.sin(angle_rad)
        cos_theta = torch.cos(angle_rad)
        
        # 根据输入维度调整输出
        if angle_deg.dim() == 1:  # [B]
            # 扩展为 [B, 2]
            sincos = torch.stack([sin_theta, cos_theta], dim=1)
        elif angle_deg.dim() == 3:  # [B, H, W]
            # 扩展为 [B, 2, H, W]
            sincos = torch.stack([sin_theta, cos_theta], dim=1)
        else:
            raise ValueError(f"Unsupported angle dimension: {angle_deg.dim()}")
        
        return sincos
    
    @staticmethod
    def sincos_to_angle(sincos: torch.Tensor) -> torch.Tensor:
        """
        将(sin θ, cos θ)转换回角度
        
        Args:
            sincos: [B, 2] 或 [B, 2, H, W]
                   通道0: sin θ
                   通道1: cos θ
        
        Returns:
            角度（度）
        """
        if sincos.dim() == 2:  # [B, 2]
            sin_theta = sincos[:, 0]
            cos_theta = sincos[:, 1]
        elif sincos.dim() == 4:  # [B, 2, H, W]
            sin_theta = sincos[:, 0, :, :]
            cos_theta = sincos[:, 1, :, :]
        else:
            raise ValueError(f"Unsupported sincos dimension: {sincos.dim()}")
        
        # 使用atan2计算角度
        angle_rad = torch.atan2(sin_theta, cos_theta)
        
        # 转换为度
        angle_deg = angle_rad * (180.0 / torch.pi)
        
        return angle_deg


class WeightedAngleLoss(AngleLoss):
    """
    加权角度损失
    
    扩展功能：
    1. 根据角度大小加权（大角度权重更高）
    2. 根据热力图值加权
    3. 时序加权（训练后期更重视角度）
    """
    
    def __init__(self,
                 loss_type: str = 'cosine',
                 pos_threshold: float = 0.7,
                 temperature: float = 1.0,
                 angle_weight: bool = True,
                 heatmap_weight: bool = True,
                 weight_power: float = 1.0,
                 reduction: str = 'mean'):
        """
        Args:
            loss_type: 损失类型
            pos_threshold: 正样本阈值
            temperature: 温度参数
            angle_weight: 是否根据角度大小加权
            heatmap_weight: 是否使用热力图值加权
            weight_power: 权重的幂次
            reduction: 归约方式
        """
        super().__init__(loss_type, pos_threshold, temperature, reduction)
        self.angle_weight = angle_weight
        self.heatmap_weight = heatmap_weight
        self.weight_power = weight_power
    
    def forward(self,
                pred_angle: torch.Tensor,
                target_angle: torch.Tensor,
                heatmap: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算加权角度损失
        
        Args:
            pred_angle: 预测角度 [B, 2, H, W]
            target_angle: 目标角度 [B, 2, H, W]
            heatmap: 热力图 [B, 1, H, W]
            mask: 有效区域掩码 [B, 1, H, W]
        
        Returns:
            加权损失值
        """
        # 确保预测已归一化
        pred_angle = F.normalize(pred_angle, p=2, dim=1)
        
        # 基础监督掩码
        angle_mask = (heatmap > self.pos_threshold).float()
        
        # 计算权重
        weight = angle_mask.clone()
        
        if self.heatmap_weight:
            # 使用热力图值作为权重
            weight = weight * torch.pow(heatmap, self.weight_power)
        
        if self.angle_weight:
            # 根据目标角度大小加权（大角度更重要）
            target_angle_deg = self.sincos_to_angle(target_angle)
            angle_weight = torch.abs(target_angle_deg) / 1.8  # 归一化到[0,1]
            if target_angle_deg.dim() == 3:  # [B, H, W]
                angle_weight = angle_weight.unsqueeze(1)  # [B, 1, H, W]
            weight = weight * torch.pow(angle_weight, self.weight_power)
        
        # 应用有效区域掩码
        if mask is not None:
            weight = weight * mask
        
        # 计算基础损失
        if self.loss_type == 'cosine':
            base_loss = self._cosine_loss(pred_angle, target_angle, torch.ones_like(weight))
        elif self.loss_type == 'l2':
            base_loss = self._l2_loss(pred_angle, target_angle, torch.ones_like(weight))
        else:
            base_loss = self._smooth_l1_loss(pred_angle, target_angle, torch.ones_like(weight))
        
        # 应用权重
        loss = base_loss * weight / self.temperature
        
        # 归约
        if self.reduction == 'mean':
            num_valid = weight.sum() + 1e-8
            return loss.sum() / num_valid
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_angle_loss(config: dict) -> AngleLoss:
    """
    工厂函数：根据配置创建角度损失
    
    Args:
        config: 配置字典，包含：
            - loss_class: 'angle' 或 'weighted_angle'
            - loss_type: 'cosine', 'l2', 'smooth_l1'
            - pos_threshold: 正样本阈值
            - temperature: 温度参数
            - angle_weight: 是否根据角度大小加权（仅weighted）
            - heatmap_weight: 是否使用热力图加权（仅weighted）
    
    Returns:
        AngleLoss实例
    """
    loss_class = config.get('loss_class', 'angle')
    
    base_params = {
        'loss_type': config.get('loss_type', 'cosine'),
        'pos_threshold': config.get('pos_threshold', 0.7),
        'temperature': config.get('temperature', 1.0),
        'reduction': config.get('reduction', 'mean')
    }
    
    if loss_class == 'weighted_angle':
        return WeightedAngleLoss(
            **base_params,
            angle_weight=config.get('angle_weight', True),
            heatmap_weight=config.get('heatmap_weight', True),
            weight_power=config.get('weight_power', 1.0)
        )
    else:
        return AngleLoss(**base_params)