# -*- coding: utf-8 -*-
"""
Loss Functions
CenterNet损失函数实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CenterNetLoss(nn.Module):
    """
    CenterNet损失函数
    包含Focal Loss和L1 Loss
    """
    
    def __init__(self, alpha=2, gamma=4, offset_weight=1.0):
        """
        Args:
            alpha: Focal loss的alpha参数
            gamma: Focal loss的gamma参数
            offset_weight: 偏移损失的权重
        """
        super(CenterNetLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.offset_weight = offset_weight
    
    def forward(self, predictions, targets):
        """
        计算损失
        
        Args:
            predictions: 模型输出字典
            targets: 目标字典，包含:
                - gap_heatmap_gt: [B, 1, H, W] 缺口热力图真值
                - gap_offset_gt: [B, 2, H, W] 缺口偏移真值
                - gap_mask: [B, 1, H, W] 缺口正样本掩码
                - piece_heatmap_gt: [B, 1, H, W] 滑块热力图真值
                - piece_offset_gt: [B, 2, H, W] 滑块偏移真值
                - piece_mask: [B, 1, H, W] 滑块正样本掩码
        
        Returns:
            损失字典
        """
        # 计算Focal Loss
        gap_focal_loss = self._focal_loss(
            predictions['gap_heatmap'], 
            targets['gap_heatmap_gt']
        )
        
        piece_focal_loss = self._focal_loss(
            predictions['piece_heatmap'], 
            targets['piece_heatmap_gt']
        )
        
        # 计算Offset L1 Loss (仅在正样本位置)
        gap_offset_loss = self._offset_loss(
            predictions['gap_offset'],
            targets['gap_offset_gt'],
            targets['gap_mask']
        )
        
        piece_offset_loss = self._offset_loss(
            predictions['piece_offset'],
            targets['piece_offset_gt'],
            targets['piece_mask']
        )
        
        # 总损失
        total_loss = (gap_focal_loss + piece_focal_loss + 
                     self.offset_weight * (gap_offset_loss + piece_offset_loss))
        
        return {
            'total_loss': total_loss,
            'gap_focal_loss': gap_focal_loss,
            'piece_focal_loss': piece_focal_loss,
            'gap_offset_loss': gap_offset_loss,
            'piece_offset_loss': piece_offset_loss
        }
    
    def _focal_loss(self, pred, gt):
        """
        CenterNet变体的Focal Loss
        """
        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        
        neg_weights = torch.pow(1 - gt, self.gamma)
        
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_mask
        
        num_pos = pos_mask.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        
        return loss
    
    def _offset_loss(self, pred, gt, mask):
        """
        偏移L1损失，仅在正样本位置计算
        """
        num_pos = mask.float().sum() + 1e-4
        
        # 扩展mask到2个通道
        mask = mask.expand_as(pred)
        
        # 计算L1损失
        loss = F.l1_loss(pred * mask, gt * mask, reduction='sum')
        loss = loss / num_pos
        
        return loss


def generate_gaussian_target(heatmap_size, centers, sigma=2):
    """
    生成高斯热力图目标
    
    Args:
        heatmap_size: (H, W) 热力图尺寸
        centers: [(x, y), ...] 中心点坐标列表 (热力图坐标系)
        sigma: 高斯核标准差
    
    Returns:
        heatmap: [H, W] 热力图
    """
    H, W = heatmap_size
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    for cx, cy in centers:
        cx, cy = int(cx), int(cy)
        
        # 计算高斯核大小
        radius = int(sigma * 3)
        
        # 生成坐标网格
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        
        # 计算高斯值
        gaussian = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
        
        # 确定在热力图中的位置
        x_min = max(0, cx - radius)
        x_max = min(W, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(H, cy + radius + 1)
        
        # 确定在高斯核中的位置
        g_x_min = max(0, radius - cx)
        g_x_max = g_x_min + (x_max - x_min)
        g_y_min = max(0, radius - cy)
        g_y_max = g_y_min + (y_max - y_min)
        
        # 取最大值
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(
            heatmap[y_min:y_max, x_min:x_max],
            gaussian[g_y_min:g_y_max, g_x_min:g_x_max]
        )
    
    return heatmap


def prepare_targets(annotations, heatmap_size=(40, 80), down_ratio=4, sigma=2):
    """
    准备训练目标
    
    Args:
        annotations: 标注信息列表
        heatmap_size: 热力图大小
        down_ratio: 下采样倍数
        sigma: 高斯核标准差
    
    Returns:
        目标字典
    """
    batch_size = len(annotations)
    H, W = heatmap_size
    
    # 初始化目标张量
    gap_heatmap_gt = torch.zeros(batch_size, 1, H, W)
    gap_offset_gt = torch.zeros(batch_size, 2, H, W)
    gap_mask = torch.zeros(batch_size, 1, H, W)
    
    piece_heatmap_gt = torch.zeros(batch_size, 1, H, W)
    piece_offset_gt = torch.zeros(batch_size, 2, H, W)
    piece_mask = torch.zeros(batch_size, 1, H, W)
    
    for idx, ann in enumerate(annotations):
        # 转换到热力图坐标
        gap_cx = ann['gap_x'] / down_ratio
        gap_cy = ann['gap_y'] / down_ratio
        piece_cx = ann['piece_x'] / down_ratio
        piece_cy = ann['piece_y'] / down_ratio
        
        # 生成热力图
        gap_hm = generate_gaussian_target(heatmap_size, [(gap_cx, gap_cy)], sigma)
        piece_hm = generate_gaussian_target(heatmap_size, [(piece_cx, piece_cy)], sigma)
        
        gap_heatmap_gt[idx, 0] = torch.from_numpy(gap_hm)
        piece_heatmap_gt[idx, 0] = torch.from_numpy(piece_hm)
        
        # 计算偏移和掩码
        gap_cx_int, gap_cy_int = int(gap_cx), int(gap_cy)
        piece_cx_int, piece_cy_int = int(piece_cx), int(piece_cy)
        
        # 确保坐标在范围内
        if 0 <= gap_cx_int < W and 0 <= gap_cy_int < H:
            gap_offset_gt[idx, 0, gap_cy_int, gap_cx_int] = gap_cx - gap_cx_int
            gap_offset_gt[idx, 1, gap_cy_int, gap_cx_int] = gap_cy - gap_cy_int
            gap_mask[idx, 0, gap_cy_int, gap_cx_int] = 1
        
        if 0 <= piece_cx_int < W and 0 <= piece_cy_int < H:
            piece_offset_gt[idx, 0, piece_cy_int, piece_cx_int] = piece_cx - piece_cx_int
            piece_offset_gt[idx, 1, piece_cy_int, piece_cx_int] = piece_cy - piece_cy_int
            piece_mask[idx, 0, piece_cy_int, piece_cx_int] = 1
    
    return {
        'gap_heatmap_gt': gap_heatmap_gt,
        'gap_offset_gt': gap_offset_gt,
        'gap_mask': gap_mask,
        'piece_heatmap_gt': piece_heatmap_gt,
        'piece_offset_gt': piece_offset_gt,
        'piece_mask': piece_mask
    }


if __name__ == "__main__":
    # 测试损失函数
    loss_fn = CenterNetLoss()
    
    # 模拟预测和目标
    batch_size = 2
    predictions = {
        'gap_heatmap': torch.sigmoid(torch.randn(batch_size, 1, 40, 80)),
        'gap_offset': torch.randn(batch_size, 2, 40, 80),
        'piece_heatmap': torch.sigmoid(torch.randn(batch_size, 1, 40, 80)),
        'piece_offset': torch.randn(batch_size, 2, 40, 80)
    }
    
    # 模拟标注
    annotations = [
        {'gap_x': 120, 'gap_y': 70, 'piece_x': 30, 'piece_y': 70},
        {'gap_x': 200, 'gap_y': 90, 'piece_x': 50, 'piece_y': 90}
    ]
    
    # 准备目标
    targets = prepare_targets(annotations)
    
    # 计算损失
    losses = loss_fn(predictions, targets)
    
    print("损失计算结果:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")