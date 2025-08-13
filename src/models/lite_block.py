# -*- coding: utf-8 -*-
"""
LiteBlock模块
Lite-HRNet的核心轻量级残差块实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SiLU(nn.Module):
    """
    SiLU激活函数 (Swish)
    SiLU(x) = x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class _ConvBNAct(nn.Module):
    """
    Conv + BatchNorm + Activation 组合模块
    内部使用，不对外暴露
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 act_type: str = 'silu'):
        super().__init__()
        
        # 卷积层
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias
        )
        
        # BatchNorm层
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        
        # 激活函数
        if act_type == 'silu':
            self.act = _SiLU()
        elif act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'none':
            self.act = nn.Identity()
        else:
            raise ValueError(f"不支持的激活函数类型: {act_type}")
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LiteBlock(nn.Module):
    """
    轻量级残差块 (MobileNetV2风格)
    结构: PW-Expand → DWConv → PW-Project → Residual
    
    这是Lite-HRNet的核心构建块，用于Stage1-4的主体部分
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: int = 2,
                 stride: int = 1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            expansion: 扩张倍率
            stride: 步长
        """
        super().__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_channels = in_channels * expansion
        
        # 1. Pointwise Expand: 1×1卷积扩张通道
        self.pw_expand = _ConvBNAct(
            in_channels, hidden_channels,
            kernel_size=1, padding=0,
            act_type='silu'
        )
        
        # 2. Depthwise Conv: 3×3深度可分离卷积
        self.dw_conv = _ConvBNAct(
            hidden_channels, hidden_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=hidden_channels,  # 深度可分离
            act_type='silu'
        )
        
        # 3. Pointwise Project: 1×1卷积投影回原通道数
        self.pw_project = _ConvBNAct(
            hidden_channels, out_channels,
            kernel_size=1, padding=0,
            act_type='none'  # 最后一层不用激活函数
        )
    
    def forward(self, x):
        identity = x
        
        # MobileNetV2风格的inverted residual
        x = self.pw_expand(x)
        x = self.dw_conv(x)
        x = self.pw_project(x)
        
        # 残差连接
        if self.use_residual:
            x = x + identity
        
        return x
