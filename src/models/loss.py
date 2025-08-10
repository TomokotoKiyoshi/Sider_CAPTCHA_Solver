"""
PMN-R3-FP Loss Functions
损失函数实现 - 包括Focal Loss, SDF Loss, 匹配损失等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import sys
import os

# 添加配置路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.model_config import get_model_config

# 获取配置单例实例
model_config = get_model_config()


class FocalLoss(nn.Module):
    """Focal Loss - 用于处理类别不平衡"""
    
    def __init__(self, alpha=None, gamma=None):
        """
        Args:
            alpha: 平衡因子 (None则从配置读取)
            gamma: 聚焦参数 (None则从配置读取)
        """
        super().__init__()
        
        # 从配置文件读取参数
        if alpha is None:
            alpha = model_config.focal_alpha
        
        if gamma is None:
            gamma = model_config.focal_gamma
        
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] - 预测概率图
            target: [B, 1, H, W] - 目标热力图
        Returns:
            loss: 标量损失值
        """
        # 展平
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 计算focal权重
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # 计算alpha权重
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # 计算BCE损失
        bce_loss = -target * torch.log(pred + 1e-8) - (1 - target) * torch.log(1 - pred + 1e-8)
        
        # 应用权重
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.mean()


class CenterNetLoss(nn.Module):
    """CenterNet风格的损失函数"""
    
    def __init__(self, alpha=None, beta=None):
        """
        Args:
            alpha: 正样本focal参数 (None则从配置读取)
            beta: 负样本focal参数 (None则从配置读取)
        """
        super().__init__()
        
        # 从配置文件读取参数
        if alpha is None:
            alpha = model_config.centernet_alpha  # AttributeError: 'ModelConfig' object has no attribute 'centernet_alpha'
        
        if beta is None:
            beta = model_config.centernet_beta  # AttributeError: 'ModelConfig' object has no attribute 'centernet_beta'
        
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] - 预测热力图
            target: [B, 1, H, W] - 目标热力图 (高斯渲染)
        Returns:
            loss: 标量损失值
        """
        pos_mask = (target == 1).float()
        neg_mask = (target < 1).float()
        
        # 正样本损失
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred + 1e-8) * pos_mask
        
        # 负样本损失
        neg_loss = -torch.pow(pred, self.alpha) * torch.pow(1 - target, self.beta) * torch.log(1 - pred + 1e-8) * neg_mask
        
        # 归一化
        num_pos = pos_mask.sum()
        if num_pos == 0:
            loss = neg_loss.sum()
        else:
            loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        
        return loss


class SDFLoss(nn.Module):
    """符号距离场损失"""
    
    def __init__(self, truncation=None):
        """
        Args:
            truncation: SDF截断距离 (None则从配置读取)
        """
        super().__init__()
        
        # 从配置文件读取参数
        if truncation is None:
            truncation = model_config.sdf_truncation  # AttributeError: 'ModelConfig' object has no attribute 'sdf_truncation'
        
        self.truncation = truncation
    
    def forward(self, pred_sdf, target_sdf, mask=None):
        """
        Args:
            pred_sdf: [B, 1, H, W] - 预测SDF
            target_sdf: [B, 1, H, W] - 目标SDF
            mask: [B, 1, H, W] - 有效区域掩码
        Returns:
            loss: 标量损失值
        """
        # 截断SDF值
        pred_sdf = torch.clamp(pred_sdf, -self.truncation, self.truncation)
        target_sdf = torch.clamp(target_sdf, -self.truncation, self.truncation)
        
        # L1损失
        l1_loss = F.l1_loss(pred_sdf, target_sdf, reduction='none')
        
        # 应用掩码
        if mask is not None:
            l1_loss = l1_loss * mask
            return l1_loss.sum() / (mask.sum() + 1e-8)
        else:
            return l1_loss.mean()


class MatchingLoss(nn.Module):
    """匹配损失 - 用于SE(2)变换器"""
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, match_scores, geometry, target_matches, target_geometry):
        """
        Args:
            match_scores: [B, N_piece, N_gap] - 预测匹配得分
            geometry: [B, N_piece, N_gap, 3] - 预测几何参数
            target_matches: [B, N_piece, N_gap] - 目标匹配矩阵
            target_geometry: [B, N_piece, N_gap, 3] - 目标几何参数
        Returns:
            Dict containing individual losses
        """
        # 匹配得分损失 (二元交叉熵)
        match_loss = self.bce_loss(match_scores, target_matches)
        
        # 几何参数损失 (仅对正确匹配计算)
        valid_mask = target_matches > 0.5
        
        if valid_mask.sum() > 0:
            pred_geom = geometry[valid_mask]
            target_geom = target_geometry[valid_mask]
            geometry_loss = self.l1_loss(pred_geom, target_geom)
        else:
            geometry_loss = torch.tensor(0.0, device=geometry.device)
        
        return {
            'match_loss': match_loss,
            'geometry_loss': geometry_loss
        }


class PMN_R3_FP_Loss(nn.Module):
    """PMN-R3-FP完整损失函数"""
    
    def __init__(self, loss_weights=None):
        """
        Args:
            loss_weights: 各项损失的权重 (None则从配置读取)
        """
        super().__init__()
        
        # 从配置文件读取权重
        if loss_weights is None:
            loss_weights = model_config.loss_weights  # AttributeError: 'ModelConfig' object has no attribute 'loss_weights'
        self.loss_weights = loss_weights
        
        # 损失函数 - 全部使用配置文件的值
        self.focal_loss = FocalLoss()  # 自动从配置读取
        self.centernet_loss = CenterNetLoss()  # 自动从配置读取
        self.sdf_loss = SDFLoss()  # 自动从配置读取
        self.matching_loss = MatchingLoss()
        self.l1_loss = nn.L1Loss()
    
    def generate_gaussian_heatmap(self, centers, shape, sigma=None):
        """
        生成高斯热力图
        Args:
            centers: [N, 2] - 中心点坐标
            shape: (H, W) - 热力图尺寸
            sigma: 高斯标准差 (None则从配置读取)
        Returns:
            heatmap: [1, H, W] - 热力图
        """
        if sigma is None:
            sigma = model_config.gaussian_sigma  # AttributeError: 'ModelConfig' object has no attribute 'gaussian_sigma'
        H, W = shape
        heatmap = torch.zeros(1, H, W, device=centers.device)
        
        for center in centers:
            cx, cy = center
            cx, cy = int(cx), int(cy)
            
            # 生成高斯核
            size = int(6 * sigma + 1)
            x = torch.arange(0, size, device=centers.device) - size // 2
            y = torch.arange(0, size, device=centers.device) - size // 2
            x, y = torch.meshgrid(x, y, indexing='ij')
            
            gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
            gaussian = gaussian / gaussian.max()
            
            # 计算边界
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(W, cx + size // 2 + 1)
            y2 = min(H, cy + size // 2 + 1)
            
            # 裁剪高斯核
            gx1 = max(0, size // 2 - cx)
            gy1 = max(0, size // 2 - cy)
            gx2 = gx1 + (x2 - x1)
            gy2 = gy1 + (y2 - y1)
            
            # 应用高斯核
            heatmap[0, y1:y2, x1:x2] = torch.max(
                heatmap[0, y1:y2, x1:x2],
                gaussian[gy1:gy2, gx1:gx2]
            )
        
        return heatmap
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict - 模型预测结果
            targets: Dict - 目标标签
        Returns:
            Dict containing total loss and individual losses
        """
        losses = {}
        
        # Region支路损失
        if 'region_predictions' in predictions:
            region_preds = predictions['region_predictions']
            region_targets = targets.get('region_targets', {})
            
            # Objectness损失
            if 'objectness' in region_preds and 'objectness' in region_targets:
                obj_loss = 0
                for pred, target in zip(region_preds['objectness'], region_targets['objectness']):
                    obj_loss += self.focal_loss(pred, target)
                losses['region_obj'] = obj_loss / len(region_preds['objectness'])
            
            # Centerness损失
            if 'centerness' in region_preds and 'centerness' in region_targets:
                ctr_loss = 0
                for pred, target in zip(region_preds['centerness'], region_targets['centerness']):
                    ctr_loss += F.mse_loss(pred, target)
                losses['region_ctr'] = ctr_loss / len(region_preds['centerness'])
            
            # Location损失
            if 'location' in region_preds and 'location' in region_targets:
                loc_loss = 0
                for pred, target in zip(region_preds['location'], region_targets['location']):
                    mask = region_targets['objectness'][0] > 0.5  # 仅对正样本计算
                    if mask.sum() > 0:
                        loc_loss += F.l1_loss(pred[mask], target[mask])
                losses['region_loc'] = loc_loss / len(region_preds['location'])
            
            # Scale损失
            if 'scale' in region_preds and 'scale' in region_targets:
                scale_loss = 0
                for pred, target in zip(region_preds['scale'], region_targets['scale']):
                    mask = region_targets['objectness'][0] > 0.5
                    if mask.sum() > 0:
                        scale_loss += F.l1_loss(pred[mask], target[mask])
                losses['region_scale'] = scale_loss / len(region_preds['scale'])
        
        # Shape支路损失
        if 'shape_mask' in predictions and 'shape_mask' in targets:
            losses['shape_mask'] = self.focal_loss(
                predictions['shape_mask'],
                targets['shape_mask']
            )
        
        if 'sdf_map' in predictions and 'sdf_map' in targets:
            losses['shape_sdf'] = self.sdf_loss(
                predictions['sdf_map'],
                targets['sdf_map'],
                mask=targets.get('sdf_mask', None)
            )
        
        # 匹配损失
        if 'match_scores' in predictions and 'match_targets' in targets:
            match_losses = self.matching_loss(
                predictions['match_scores'],
                predictions['geometry'],
                targets['match_targets'],
                targets['geometry_targets']
            )
            losses['matching'] = match_losses['match_loss']
            losses['geometry'] = match_losses['geometry_loss']
        
        # 计算总损失
        total_loss = 0
        for key, loss in losses.items():
            weight = self.loss_weights.get(key, 1.0)
            total_loss += weight * loss
        
        losses['total'] = total_loss
        
        return losses


if __name__ == "__main__":
    # 测试代码
    print("Loading configuration from model_config.yaml...")
    print(f"Focal Loss - alpha: {model_config.focal_alpha}, gamma: {model_config.focal_gamma}")
    print(f"CenterNet Loss - alpha: {model_config.centernet_alpha}, beta: {model_config.centernet_beta}")
    print(f"SDF Loss - truncation: {model_config.sdf_truncation}")
    print(f"Gaussian sigma: {model_config.gaussian_sigma}")
    
    # 测试Focal Loss - 使用配置文件的值
    focal_loss = FocalLoss()  # 自动从配置读取
    pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 30:35, 30:35] = 1
    
    loss = focal_loss(pred, target)
    print(f"\nFocal Loss: {loss.item():.4f}")
    
    # 测试CenterNet Loss - 使用配置文件的值
    centernet_loss = CenterNetLoss()  # 自动从配置读取
    pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
    target = torch.zeros(2, 1, 64, 64)
    # 模拟高斯热力图
    target[:, :, 30:35, 30:35] = torch.exp(-torch.randn(2, 1, 5, 5).abs())
    
    loss = centernet_loss(pred, target)
    print(f"CenterNet Loss: {loss.item():.4f}")
    
    # 测试SDF Loss - 使用配置文件的值
    sdf_loss = SDFLoss()  # 自动从配置读取
    pred_sdf = torch.randn(2, 1, 64, 64) * 5
    target_sdf = torch.randn(2, 1, 64, 64) * 5
    
    loss = sdf_loss(pred_sdf, target_sdf)
    print(f"SDF Loss: {loss.item():.4f}")
    
    # 测试匹配损失
    matching_loss = MatchingLoss()
    match_scores = torch.sigmoid(torch.randn(2, 5, 10))
    geometry = torch.randn(2, 5, 10, 3)
    target_matches = torch.zeros(2, 5, 10)
    target_matches[:, 0, 0] = 1  # 第一个滑块匹配第一个缺口
    target_geometry = torch.randn(2, 5, 10, 3)
    
    losses = matching_loss(match_scores, geometry, target_matches, target_geometry)
    print(f"Matching Loss: {losses['match_loss'].item():.4f}")
    print(f"Geometry Loss: {losses['geometry_loss'].item():.4f}")
    
    # 测试完整损失 - 使用配置文件的权重
    pmn_loss = PMN_R3_FP_Loss()  # 自动从配置读取权重
    print(f"\nLoss weights from config: {pmn_loss.loss_weights}")