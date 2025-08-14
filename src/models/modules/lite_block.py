# -*- coding: utf-8 -*-
"""
LiteBlock模块
Lite-HRNet的核心轻量级残差块实现
"""
import torch
import torch.nn as nn
from typing import Optional
from .conv_bn_act import ConvBNAct


class LiteBlock(nn.Module):
    """
    轻量级残差块 (MobileNetV2风格 Inverted Residual)
    
    结构: PW-Expand → DWConv → PW-Project → Residual
    
    激活函数配置:
    - PW-Expand后: BN + SiLU激活
    - DWConv后: BN + SiLU激活
    - PW-Project后: 仅BN，无激活函数（线性输出）
    
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
            expansion: 扩张倍率 (默认2)
            stride: 步长 (默认1)
        """
        super().__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_channels = in_channels * expansion
        
        # 1. Pointwise Expand: 1×1卷积扩张通道
        self.pw_expand = ConvBNAct(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act_type='silu'
            # padding自动计算为0
        )
        
        # 2. Depthwise Conv: 3×3深度可分离卷积
        self.dw_conv = ConvBNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=stride,
            groups=hidden_channels,  # 深度可分离
            act_type='silu'
            # padding自动计算为1
        )
        
        # 3. Pointwise Project: 1×1卷积投影回原通道数
        # 重要: Project层只有BN，没有激活函数（线性输出）
        # 这是MobileNetV2的标准设计，有助于保持特征的表达能力
        self.pw_project = ConvBNAct(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act_type='none'  # 线性输出，仅BN无激活
            # padding自动计算为0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            
        Returns:
            输出张量 [B, C_out, H', W']
        """
        identity = x
        
        # MobileNetV2风格的inverted residual
        x = self.pw_expand(x)
        x = self.dw_conv(x)
        x = self.pw_project(x)
        
        # 残差连接
        if self.use_residual:
            x = x + identity
        
        return x
    
    def get_params_count(self) -> dict:
        """
        计算参数量统计
        
        Returns:
            包含各层参数量的字典
        """
        expand_params = sum(p.numel() for p in self.pw_expand.parameters())
        dw_params = sum(p.numel() for p in self.dw_conv.parameters())
        project_params = sum(p.numel() for p in self.pw_project.parameters())
        total_params = expand_params + dw_params + project_params
        
        return {
            'expand': expand_params,
            'depthwise': dw_params,
            'project': project_params,
            'total': total_params
        }


# 只暴露必要的接口
__all__ = ['LiteBlock']