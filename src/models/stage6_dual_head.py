# -*- coding: utf-8 -*-
"""
Stage6 - 双头预测网络（Dual Head）
在统一的1/4分辨率特征图上进行密集预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from .modules import ConvBNAct


class HeatmapHead(nn.Module):
    """
    热力图预测头
    用于预测缺口中心和拼图中心的概率分布
    
    结构：3×3,128→64 → ReLU → 1×1,64→1 → Sigmoid
    """
    
    def __init__(self,
                 in_channels: int = 128,
                 mid_channels: int = 64):
        """
        Args:
            in_channels: 输入通道数（来自LiteFPN的Hf）
            mid_channels: 中间层通道数
        """
        super().__init__()
        
        # 第一层：3×3卷积降维
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            act_type='relu',
            bias=True  # 热力图头使用bias
        )
        
        # 第二层：1×1卷积输出单通道热力图
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        # Sigmoid激活，输出概率
        self.sigmoid = nn.Sigmoid()
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 最后一层使用较小的初始化
        nn.init.normal_(self.conv2.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv2.bias, -2.19)  # -log((1-0.1)/0.1) for 0.1 prior
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, 128, 64, 128]
        
        Returns:
            热力图 [B, 1, 64, 128]，值域(0, 1)
        """
        x = self.conv1(x)        # [B, 64, 64, 128]
        x = self.conv2(x)        # [B, 1, 64, 128]
        heatmap = self.sigmoid(x)  # [B, 1, 64, 128]
        
        return heatmap


class OffsetHead(nn.Module):
    """
    子像素偏移预测头
    预测缺口和拼图中心在栅格内的连续偏移
    
    结构：3×3,128→64 → ReLU → 1×1,64→4
    """
    
    def __init__(self,
                 in_channels: int = 128,
                 mid_channels: int = 64,
                 num_points: int = 2,
                 use_tanh: bool = True):
        """
        Args:
            in_channels: 输入通道数
            mid_channels: 中间层通道数
            num_points: 预测点的数量（2个：缺口和拼图）
            use_tanh: 是否使用tanh限制输出范围到[-0.5, 0.5]
        """
        super().__init__()
        
        self.num_points = num_points
        self.use_tanh = use_tanh
        
        # 第一层：3×3卷积降维
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            act_type='relu',
            bias=True
        )
        
        # 第二层：1×1卷积输出偏移
        # 每个点需要2个通道（du, dv），总共num_points*2个通道
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=num_points * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        # 可选的tanh激活，限制偏移范围
        if self.use_tanh:
            self.activation = nn.Tanh()
            self.scale = 0.5  # tanh输出[-1,1]，缩放到[-0.5,0.5]
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 偏移预测初始化为0附近
        nn.init.normal_(self.conv2.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, 128, 64, 128]
        
        Returns:
            偏移图 [B, 4, 64, 128]
            通道顺序：(du_gap, dv_gap, du_piece, dv_piece)
            值域：[-0.5, 0.5]（栅格单位）
        """
        x = self.conv1(x)        # [B, 64, 64, 128]
        offset = self.conv2(x)   # [B, 4, 64, 128]
        
        if self.use_tanh:
            offset = self.activation(offset) * self.scale  # [-0.5, 0.5]
        
        return offset


class AngleHead(nn.Module):
    """
    角度预测头（可选）
    预测缺口的微小旋转角度
    
    结构：3×3,128→64 → ReLU → 1×1,64→2 → L2 normalize
    输出：(sin θ, cos θ)
    """
    
    def __init__(self,
                 in_channels: int = 128,
                 mid_channels: int = 64):
        """
        Args:
            in_channels: 输入通道数
            mid_channels: 中间层通道数
        """
        super().__init__()
        
        # 第一层：3×3卷积降维
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            act_type='relu',
            bias=True
        )
        
        # 第二层：1×1卷积输出sin和cos
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,  # (sin θ, cos θ)
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 初始化为接近(sin(0), cos(0)) = (0, 1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, 128, 64, 128]
        
        Returns:
            角度图 [B, 2, 64, 128]
            通道0: sin θ
            通道1: cos θ
            经过L2归一化，满足sin²θ + cos²θ = 1
        """
        x = self.conv1(x)        # [B, 64, 64, 128]
        angle = self.conv2(x)    # [B, 2, 64, 128]
        
        # L2归一化，确保sin²θ + cos²θ = 1
        angle = F.normalize(angle, p=2, dim=1)
        
        return angle


class DualHead(nn.Module):
    """
    Stage6 - 双头预测网络
    
    在统一的1/4分辨率特征图上进行所有密集预测
    
    输入：
        Hf: [B, 128, 64, 128] (来自Stage5 LiteFPN)
    
    输出：
        predictions: 包含所有预测结果的字典
            - 'heatmap_gap': [B, 1, 64, 128] 缺口中心热力图
            - 'heatmap_piece': [B, 1, 64, 128] 拼图中心热力图
            - 'offset': [B, 4, 64, 128] 子像素偏移
            - 'angle': [B, 2, 64, 128] 角度预测（可选）
    """
    
    def __init__(self,
                 in_channels: int = 128,
                 mid_channels: int = 64,
                 use_angle: bool = False,
                 use_tanh_offset: bool = True):
        """
        Args:
            in_channels: 输入通道数（来自LiteFPN）
            mid_channels: 中间层通道数
            use_angle: 是否使用角度预测头
            use_tanh_offset: 偏移头是否使用tanh限制范围
        """
        super().__init__()
        
        self.use_angle = use_angle
        
        # ========== 双中心热力图头 ==========
        # 缺口中心热力图
        self.heatmap_gap = HeatmapHead(
            in_channels=in_channels,
            mid_channels=mid_channels
        )
        
        # 拼图中心热力图
        self.heatmap_piece = HeatmapHead(
            in_channels=in_channels,
            mid_channels=mid_channels
        )
        
        # ========== 子像素偏移头 ==========
        # 预测两个点的偏移：缺口和拼图
        self.offset = OffsetHead(
            in_channels=in_channels,
            mid_channels=mid_channels,
            num_points=2,
            use_tanh=use_tanh_offset
        )
        
        # ========== 角度头（可选） ==========
        if self.use_angle:
            self.angle = AngleHead(
                in_channels=in_channels,
                mid_channels=mid_channels
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, 128, 64, 128]（来自Stage5 LiteFPN）
        
        Returns:
            predictions: 预测结果字典
                - 'heatmap_gap': [B, 1, 64, 128] 缺口中心热力图
                - 'heatmap_piece': [B, 1, 64, 128] 拼图中心热力图
                - 'offset': [B, 4, 64, 128] 子像素偏移
                - 'angle': [B, 2, 64, 128] 角度预测（如果use_angle=True）
        """
        predictions = {}
        
        # 热力图预测
        predictions['heatmap_gap'] = self.heatmap_gap(x)
        predictions['heatmap_piece'] = self.heatmap_piece(x)
        
        # 偏移预测
        predictions['offset'] = self.offset(x)
        
        # 角度预测（可选）
        if self.use_angle:
            predictions['angle'] = self.angle(x)
        
        return predictions


def create_stage6_dual_head(config: dict):
    """
    创建Stage6 双头预测网络的工厂函数
    
    Args:
        config: 配置字典，必须包含:
            - in_channels: 输入通道数
            - mid_channels: 中间层通道数
            - use_angle: 是否使用角度预测
            - use_tanh_offset: 是否使用tanh限制偏移范围
    
    Returns:
        DualHead模块实例
    """
    # 默认配置
    default_config = {
        'in_channels': 128,
        'mid_channels': 64,
        'use_angle': True,
        'use_tanh_offset': True
    }
    
    # 合并用户配置
    final_config = {**default_config, **config}
    
    return DualHead(
        in_channels=final_config['in_channels'],
        mid_channels=final_config['mid_channels'],
        use_angle=final_config['use_angle'],
        use_tanh_offset=final_config['use_tanh_offset']
    )