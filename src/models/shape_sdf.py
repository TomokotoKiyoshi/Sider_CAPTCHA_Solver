"""
PMN-R3-FP Shape/SDF Module
形状预测和符号距离场(SDF)模块
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


class CoordConv2d(nn.Module):
    """坐标卷积层 - 添加空间坐标信息"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            coords: [B, 2, H, W] - 归一化坐标 [-1, 1]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 生成归一化坐标网格
        y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        
        coords = torch.cat([x_coords, y_coords], dim=1)  # [B, 2, H, W]
        
        return coords


class EdgePriorExtractor(nn.Module):
    """边缘先验提取器"""
    
    def __init__(self):
        super().__init__()
        
        # Sobel算子
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # LoG算子 (Laplacian of Gaussian)
        log_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 注册为buffer (不参与梯度更新)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('log_kernel', log_kernel)
    
    def forward(self, rgb_input):
        """
        Args:
            rgb_input: [B, 3, H, W]
        Returns:
            edges: Dict with 'full', 'half', 'quarter' resolutions
        """
        # 转换为灰度图 (使用配置的RGB权重)
        r_weight, g_weight, b_weight = model_config.edge_rgb_weights
        gray = r_weight * rgb_input[:, 0:1] + g_weight * rgb_input[:, 1:2] + b_weight * rgb_input[:, 2:3]
        
        # 计算Sobel边缘
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge_sobel = torch.sqrt(edge_x**2 + edge_y**2)
        
        # 计算LoG边缘
        edge_log = F.conv2d(gray, self.log_kernel, padding=1)
        
        # 组合边缘特征
        edges_full = torch.cat([edge_x, edge_y, edge_log], dim=1)  # [B, 3, H, W]
        
        # 多尺度下采样
        edges_half = F.avg_pool2d(edges_full, 2)      # [B, 3, H/2, W/2]
        edges_quarter = F.avg_pool2d(edges_half, 2)    # [B, 3, H/4, W/4]
        
        return {
            'full': edges_full,      # [B, 3, 256, 512]
            'half': edges_half,      # [B, 3, 128, 256]
            'quarter': edges_quarter  # [B, 3, 64, 128]
        }


class ShapeBranch(nn.Module):
    """Shape支路 - 高分辨率形状预测"""
    
    def __init__(self):
        super().__init__()
        
        # 边缘提取器
        self.edge_extractor = EdgePriorExtractor()
        
        # 从配置获取通道数
        ch_quarter = model_config.shape_channels_quarter
        ch_half = model_config.shape_channels_half
        ch_full = model_config.shape_channels_full
        
        # 1/4分辨率处理
        self.conv_quarter = nn.Sequential(
            nn.Conv2d(64 + 3, ch_quarter, 3, 1, 1),  # s1特征 + 边缘
            nn.BatchNorm2d(ch_quarter),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_quarter, ch_quarter, 3, 1, 1),
            nn.BatchNorm2d(ch_quarter),
            nn.ReLU(inplace=True)
        )
        
        # 1/2分辨率处理
        self.conv_half = nn.Sequential(
            nn.Conv2d(ch_quarter + 3, ch_half, 3, 1, 1),  # 上采样特征 + 边缘
            nn.BatchNorm2d(ch_half),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_half, ch_half, 3, 1, 1),
            nn.BatchNorm2d(ch_half),
            nn.ReLU(inplace=True)
        )
        
        # 全分辨率处理
        self.conv_full = nn.Sequential(
            nn.Conv2d(ch_half + 3, ch_full, 3, 1, 1),   # 上采样特征 + 边羘
            nn.BatchNorm2d(ch_full),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_full, ch_full, 3, 1, 1),
            nn.BatchNorm2d(ch_full),
            nn.ReLU(inplace=True)
        )
        
        # 形状分类头
        self.shape_head = nn.Conv2d(ch_full, 1, 1, 1, 0)  # 二值掩码
        
        # SDF回归头
        self.sdf_head = nn.Conv2d(ch_full, 1, 1, 1, 0)    # 符号距离场
    
    def forward(self, rgb_input, s1_feature):
        """
        Args:
            rgb_input: [B, 3, 256, 512] - RGB输入
            s1_feature: [B, 64, 64, 128] - 来自骨干网络的s1特征
        Returns:
            Dict containing:
                - shape_mask: [B, 1, 256, 512] - 形状掩码
                - sdf_map: [B, 1, 256, 512] - SDF图
        """
        # 提取边缘先验
        edges = self.edge_extractor(rgb_input)
        
        # 1/4分辨率处理
        x_quarter = torch.cat([s1_feature, edges['quarter']], dim=1)
        x_quarter = self.conv_quarter(x_quarter)  # [B, 128, 64, 128]
        
        # 上采样到1/2分辨率
        x_half = F.interpolate(x_quarter, scale_factor=2, mode='bilinear', align_corners=False)
        x_half = torch.cat([x_half, edges['half']], dim=1)
        x_half = self.conv_half(x_half)  # [B, 64, 128, 256]
        
        # 上采样到全分辨率
        x_full = F.interpolate(x_half, scale_factor=2, mode='bilinear', align_corners=False)
        x_full = torch.cat([x_full, edges['full']], dim=1)
        x_full = self.conv_full(x_full)  # [B, 32, 256, 512]
        
        # 预测形状掩码
        shape_mask = torch.sigmoid(self.shape_head(x_full))  # [B, 1, 256, 512]
        
        # 预测SDF (使用配置的范围)
        sdf_map = torch.tanh(self.sdf_head(x_full)) * model_config.sdf_range  # [B, 1, 256, 512]
        
        return {
            'shape_mask': shape_mask,
            'sdf_map': sdf_map,
            'features': x_full  # 用于后续处理
        }


class ROIAlignExtractor(nn.Module):
    """ROI对齐特征提取器"""
    
    def __init__(self, output_size=None, spatial_scale=None):
        """
        Args:
            output_size: ROI输出尺寸 (None则从配置读取)
            spatial_scale: 特征图相对于原图的缩放比例 (None则从配置读取)
        """
        super().__init__()
        
        # 从配置文件读取参数
        if output_size is None:
            if not hasattr(model_config, 'roi_shape_size'):
                raise ValueError("roi_shape_size not found in model_config.yaml")
            output_size = model_config.roi_shape_size
        
        if spatial_scale is None:
            if not hasattr(model_config, 'roi_shape_scale'):
                raise ValueError("roi_shape_scale not found in model_config.yaml")
            spatial_scale = model_config.roi_shape_scale
        
        self.output_size = output_size
        self.spatial_scale = spatial_scale
    
    def forward(self, features, proposals):
        """
        Args:
            features: [B, C, H, W] - 特征图
            proposals: [B, N, 5] - 候选框 (cx, cy, w, h, score)
        Returns:
            roi_features: [B*N, C, output_size, output_size]
        """
        B, N, _ = proposals.shape
        device = features.device
        
        # 转换proposals格式 (cx,cy,w,h) -> (x1,y1,x2,y2)
        cx = proposals[..., 0]
        cy = proposals[..., 1]
        w = proposals[..., 2]
        h = proposals[..., 3]
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # 创建批次索引
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, N)
        
        # 组合ROI格式 [batch_idx, x1, y1, x2, y2]
        rois = torch.stack([
            batch_indices.reshape(-1),
            x1.reshape(-1) * self.spatial_scale,
            y1.reshape(-1) * self.spatial_scale,
            x2.reshape(-1) * self.spatial_scale,
            y2.reshape(-1) * self.spatial_scale
        ], dim=1)
        
        # 使用torchvision的roi_align (需要安装torchvision)
        try:
            from torchvision.ops import roi_align
            roi_features = roi_align(
                features,
                rois,
                output_size=(self.output_size, self.output_size),
                spatial_scale=1.0,  # 已经在rois中处理了scale
                aligned=True
            )
        except ImportError:
            # 简化版本的ROI提取 (用于测试)
            roi_features = self._simple_roi_extract(features, proposals)
        
        return roi_features
    
    def _simple_roi_extract(self, features, proposals):
        """简化的ROI提取 (备用方案)"""
        B, C, H, W = features.shape
        N = proposals.shape[1]
        device = features.device
        
        roi_features = []
        
        for b in range(B):
            for n in range(N):
                cx, cy, w, h = proposals[b, n, :4]
                
                # 计算ROI边界
                x1 = max(0, int((cx - w/2) * self.spatial_scale))
                y1 = max(0, int((cy - h/2) * self.spatial_scale))
                x2 = min(W, int((cx + w/2) * self.spatial_scale))
                y2 = min(H, int((cy + h/2) * self.spatial_scale))
                
                # 提取并调整大小
                roi = features[b:b+1, :, y1:y2, x1:x2]
                roi = F.adaptive_avg_pool2d(roi, (self.output_size, self.output_size))
                roi_features.append(roi)
        
        return torch.cat(roi_features, dim=0)


class SDFDecoder(nn.Module):
    """SDF解码器 - 从ROI特征解码SDF"""
    
    def __init__(self, in_channels=None, hidden_dim=None):
        """
        Args:
            in_channels: 输入通道数 (None则从配置读取)
            hidden_dim: 隐藏层维度 (None则从配置读取)
        """
        super().__init__()
        
        # 从配置文件读取参数
        if in_channels is None:
            if not hasattr(model_config, 'sdf_decoder_in_channels'):
                raise ValueError("sdf_decoder_in_channels not found in model_config.yaml")
            in_channels = model_config.sdf_decoder_in_channels
        
        if hidden_dim is None:
            if not hasattr(model_config, 'sdf_decoder_hidden_dim'):
                raise ValueError("sdf_decoder_hidden_dim not found in model_config.yaml")
            hidden_dim = model_config.sdf_decoder_hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),  # 下采样
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),  # 下采样
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 3, 2, 1, 1),  # 上采样
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 3, 2, 1, 1),  # 上采样
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim//2, 1, 1, 1, 0)  # SDF输出
        )
    
    def forward(self, roi_features):
        """
        Args:
            roi_features: [B*N, C, 64, 64]
        Returns:
            sdf: [B*N, 1, 64, 64]
        """
        encoded = self.encoder(roi_features)
        sdf = self.decoder(encoded)
        sdf = torch.tanh(sdf) * model_config.sdf_range  # 使用配置的范围
        
        return sdf


if __name__ == "__main__":
    # 测试代码
    
    # 测试Shape支路
    shape_branch = ShapeBranch()
    rgb_input = torch.randn(2, 3, 256, 512)
    s1_feature = torch.randn(2, 64, 64, 128)
    
    shape_output = shape_branch(rgb_input, s1_feature)
    print("Shape Branch Output:")
    print(f"  Shape mask: {shape_output['shape_mask'].shape}")
    print(f"  SDF map: {shape_output['sdf_map'].shape}")
    
    # 测试ROI提取 - 使用配置文件的值
    print("\nLoading configuration from model_config.yaml...")
    print(f"ROI shape size: {model_config.roi_shape_size}")
    print(f"ROI shape scale: {model_config.roi_shape_scale}")
    print(f"SDF range: {model_config.sdf_range}")
    print(f"SDF decoder channels: in={model_config.sdf_decoder_in_channels}, hidden={model_config.sdf_decoder_hidden_dim}")
    
    roi_extractor = ROIAlignExtractor()  # 现在会自动使用配置文件的值
    features = torch.randn(2, 32, 64, 128)
    proposals = torch.randn(2, 10, 5)  # 10个候选框
    proposals[..., :4] = proposals[..., :4].abs() * 100 + 50  # 确保正值
    
    roi_features = roi_extractor(features, proposals)
    print(f"\nROI Features shape: {roi_features.shape}")
    
    # 测试SDF解码器 - 使用配置文件的值
    # 注意: 这里模拟的输入通道数为32，但实际配置中是35
    sdf_decoder = SDFDecoder(in_channels=32)  # 测试时使用指定值
    sdf = sdf_decoder(roi_features)
    print(f"SDF output shape: {sdf.shape}")