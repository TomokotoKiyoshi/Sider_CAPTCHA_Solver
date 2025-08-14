# -*- coding: utf-8 -*-
"""
损失计算工具函数
提供高斯热图生成、坐标转换、掩码处理等辅助功能
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Union


def generate_gaussian_heatmap(centers: torch.Tensor,
                             shape: Tuple[int, int],
                             sigma: float = 1.5,
                             device: str = 'cpu') -> torch.Tensor:
    """
    生成高斯热图
    
    Args:
        centers: 中心点坐标 [N, 2] 或 [B, N, 2]
                格式：(x, y) 或 (u, v)，栅格坐标系
        shape: 热图形状 (H, W)
        sigma: 高斯标准差（栅格单位）
        device: 设备
    
    Returns:
        高斯热图 [H, W] 或 [B, N, H, W]
    """
    H, W = shape
    
    # 处理不同维度的输入
    if centers.dim() == 2:  # [N, 2]
        centers = centers.unsqueeze(0)  # [1, N, 2]
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    B, N, _ = centers.shape
    
    # 创建坐标网格
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 扩展维度以支持批处理
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 初始化热图
    heatmaps = torch.zeros(B, N, H, W, device=device)
    
    # 为每个中心生成高斯分布
    for i in range(N):
        cx = centers[:, i, 0:1].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        cy = centers[:, i, 1:2].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        
        # 计算高斯分布
        gaussian = torch.exp(
            -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2)
        )
        
        heatmaps[:, i, :, :] = gaussian.squeeze(1)
    
    # 处理输出维度
    if squeeze_batch:
        heatmaps = heatmaps.squeeze(0)  # [N, H, W]
    
    return heatmaps


def create_padding_mask(input_shape: Tuple[int, int],
                       padded_shape: Tuple[int, int],
                       downsample: int = 4,
                       pooling: str = 'avg') -> torch.Tensor:
    """
    创建padding掩码并下采样到特征图分辨率
    
    Args:
        input_shape: 原始输入形状 (H_orig, W_orig)
        padded_shape: padding后的形状 (H_pad, W_pad)
        downsample: 下采样率
        pooling: 池化方式 ('avg' 或 'max')
    
    Returns:
        下采样后的掩码 [1, H_down, W_down]
        0表示padding区域，1表示有效区域
    """
    H_orig, W_orig = input_shape
    H_pad, W_pad = padded_shape
    
    # 创建原始分辨率掩码
    # 需要确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = torch.zeros(1, 1, H_pad, W_pad, device=device)
    mask[:, :, :H_orig, :W_orig] = 1.0
    
    # 下采样到特征图分辨率
    if pooling == 'avg':
        # 平均池化（软掩码）
        mask_down = F.avg_pool2d(mask, kernel_size=downsample, stride=downsample)
    else:  # 'max'
        # 最大池化（硬掩码）
        mask_down = F.max_pool2d(mask, kernel_size=downsample, stride=downsample)
    
    return mask_down.squeeze(0)  # [1, H_down, W_down]


def coordinate_transform(coords: torch.Tensor,
                        mode: str,
                        scale: float = 4.0) -> torch.Tensor:
    """
    坐标系转换
    
    Args:
        coords: 坐标 [B, N, 2] 或 [N, 2]
        mode: 转换模式
              'pixel_to_grid': 原图坐标 -> 栅格坐标
              'grid_to_pixel': 栅格坐标 -> 原图坐标
              'grid_to_offset': 栅格坐标 -> 栅格+偏移
              'offset_to_grid': 栅格+偏移 -> 栅格坐标
        scale: 缩放因子（下采样率）
    
    Returns:
        转换后的坐标
    """
    if mode == 'pixel_to_grid':
        # 原图坐标 -> 栅格坐标
        return coords / scale
    
    elif mode == 'grid_to_pixel':
        # 栅格坐标 -> 原图坐标
        return coords * scale
    
    elif mode == 'grid_to_offset':
        # 栅格坐标 -> 栅格索引 + 偏移
        grid_int = torch.floor(coords)
        grid_offset = coords - grid_int - 0.5  # [-0.5, 0.5]
        return grid_int, grid_offset
    
    elif mode == 'offset_to_grid':
        # 栅格索引 + 偏移 -> 栅格坐标
        # coords应该是(grid_int, grid_offset)的元组
        grid_int, grid_offset = coords
        return grid_int + grid_offset + 0.5
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    计算边界框IoU（用于评估）
    
    Args:
        box1: 边界框1 [B, 4] (x1, y1, x2, y2)
        box2: 边界框2 [B, 4] (x1, y1, x2, y2)
    
    Returns:
        IoU值 [B]
    """
    # 计算交集
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算并集
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = area1 + area2 - inter_area
    
    # 计算IoU
    iou = inter_area / (union_area + 1e-8)
    
    return iou


def nms_heatmap(heatmap: torch.Tensor,
                kernel_size: int = 3,
                threshold: float = 0.01) -> torch.Tensor:
    """
    对热力图进行非极大值抑制
    
    Args:
        heatmap: 输入热力图 [B, C, H, W]
        kernel_size: 池化核大小
        threshold: 阈值，低于此值的响应被抑制
    
    Returns:
        NMS后的热力图 [B, C, H, W]
    """
    # 最大池化找到局部最大值
    padding = (kernel_size - 1) // 2
    max_pool = F.max_pool2d(heatmap, kernel_size, stride=1, padding=padding)
    
    # 保留局部最大值
    peak_mask = (heatmap == max_pool).float()
    
    # 应用阈值
    peak_mask = peak_mask * (heatmap > threshold).float()
    
    # 应用掩码
    nms_heatmap = heatmap * peak_mask
    
    return nms_heatmap


def extract_peaks(heatmap: torch.Tensor,
                 threshold: float = 0.1,
                 nms: bool = True,
                 nms_kernel: int = 3,
                 top_k: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从热力图中提取峰值点
    
    Args:
        heatmap: 热力图 [B, C, H, W]
        threshold: 响应阈值
        nms: 是否应用NMS
        nms_kernel: NMS核大小
        top_k: 保留前k个峰值
    
    Returns:
        peaks: 峰值坐标 [B, C, K, 2]
        scores: 峰值得分 [B, C, K]
    """
    B, C, H, W = heatmap.shape
    
    # 应用NMS
    if nms:
        heatmap = nms_heatmap(heatmap, nms_kernel, threshold)
    
    # 展平并找到top-k
    heatmap_flat = heatmap.view(B, C, -1)
    scores, indices = torch.topk(heatmap_flat, k=min(top_k, heatmap_flat.shape[-1]), dim=-1)
    
    # 转换索引为坐标
    y_coords = indices // W
    x_coords = indices % W
    peaks = torch.stack([x_coords, y_coords], dim=-1)  # [B, C, K, 2]
    
    # 过滤低于阈值的峰值
    valid_mask = scores > threshold
    peaks = peaks * valid_mask.unsqueeze(-1)
    scores = scores * valid_mask
    
    return peaks, scores


def soft_argmax(heatmap: torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
    """
    软argmax：可微分的峰值定位
    
    Args:
        heatmap: 热力图 [B, C, H, W]
        temperature: 温度参数，控制软化程度
    
    Returns:
        坐标 [B, C, 2] (x, y)
    """
    B, C, H, W = heatmap.shape
    
    # 创建坐标网格
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=heatmap.device),
        torch.arange(W, dtype=torch.float32, device=heatmap.device),
        indexing='ij'
    )
    
    # 应用温度并归一化
    heatmap_soft = torch.softmax(heatmap.view(B, C, -1) / temperature, dim=-1)
    heatmap_soft = heatmap_soft.view(B, C, H, W)
    
    # 加权平均计算坐标
    x_coords = (heatmap_soft * x_grid.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3))
    y_coords = (heatmap_soft * y_grid.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3))
    
    coords = torch.stack([x_coords, y_coords], dim=-1)  # [B, C, 2]
    
    return coords


def compute_distance_error(pred: torch.Tensor,
                          target: torch.Tensor,
                          scale: float = 4.0) -> torch.Tensor:
    """
    计算预测和目标之间的距离误差
    
    Args:
        pred: 预测坐标 [B, N, 2]
        target: 目标坐标 [B, N, 2]
        scale: 缩放因子，用于转换到原图坐标系
    
    Returns:
        距离误差 [B, N]
    """
    # 转换到原图坐标系
    pred_pixel = pred * scale
    target_pixel = target * scale
    
    # 计算欧氏距离
    distance = torch.norm(pred_pixel - target_pixel, p=2, dim=-1)
    
    return distance


class GaussianHeatmapGenerator:
    """
    高斯热图生成器（优化版本）
    预计算高斯核，提高生成效率
    """
    
    def __init__(self,
                 sigma: float = 1.5,
                 radius_factor: float = 3.0):
        """
        Args:
            sigma: 高斯标准差
            radius_factor: 半径因子，radius = sigma * radius_factor
        """
        self.sigma = sigma
        self.radius_factor = radius_factor
        self.radius = int(sigma * radius_factor)
        
        # 预计算高斯核
        self.gaussian_kernel = self._create_gaussian_kernel()
    
    def _create_gaussian_kernel(self) -> torch.Tensor:
        """预计算高斯核"""
        size = 2 * self.radius + 1
        kernel = torch.zeros(size, size)
        center = self.radius
        
        for i in range(size):
            for j in range(size):
                dist_sq = (i - center) ** 2 + (j - center) ** 2
                kernel[i, j] = np.exp(-dist_sq / (2 * self.sigma ** 2))
        
        return kernel
    
    def generate(self,
                centers: torch.Tensor,
                shape: Tuple[int, int]) -> torch.Tensor:
        """
        生成热图（使用预计算的高斯核）
        
        Args:
            centers: 中心点坐标 [N, 2]
            shape: 热图形状 (H, W)
        
        Returns:
            热图 [H, W]
        """
        H, W = shape
        heatmap = torch.zeros(H, W, device=centers.device)
        
        for center in centers:
            x, y = int(center[0]), int(center[1])
            
            # 计算核的有效区域
            x_min = max(0, x - self.radius)
            x_max = min(W, x + self.radius + 1)
            y_min = max(0, y - self.radius)
            y_max = min(H, y + self.radius + 1)
            
            # 计算核的对应区域
            kernel_x_min = max(0, self.radius - x)
            kernel_x_max = kernel_x_min + (x_max - x_min)
            kernel_y_min = max(0, self.radius - y)
            kernel_y_max = kernel_y_min + (y_max - y_min)
            
            # 应用高斯核
            if x_min < x_max and y_min < y_max:
                kernel_region = self.gaussian_kernel[
                    kernel_y_min:kernel_y_max,
                    kernel_x_min:kernel_x_max
                ].to(heatmap.device)
                
                heatmap[y_min:y_max, x_min:x_max] = torch.maximum(
                    heatmap[y_min:y_max, x_min:x_max],
                    kernel_region
                )
        
        return heatmap