# -*- coding: utf-8 -*-
"""
DepthwiseSeparableConv 模块
深度可分离卷积的标准实现
"""
import torch
import torch.nn as nn
from typing import Optional
from .conv_bn_act import ConvBNAct


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积模块
    
    结构: Depthwise Conv → Pointwise Conv
    用于减少参数量和计算量，同时保持特征提取能力
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dw_act_type: str = 'silu',
                 pw_act_type: str = 'silu',
                 padding: Optional[int] = None):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 深度卷积的卷积核大小 (默认3)
            stride: 步长 (默认1)
            dw_act_type: Depthwise卷积的激活函数类型 (默认'silu')
            pw_act_type: Pointwise卷积的激活函数类型 (默认'silu')
            padding: 填充大小，如果为None则自动计算
        """
        super().__init__()
        
        # 如果没有指定padding，则自动计算
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        # Depthwise Convolution: 每个通道独立卷积
        # groups=in_channels 使得每个输入通道只与一个卷积核卷积
        self.dw_conv = ConvBNAct(
            in_channels=in_channels,
            out_channels=in_channels,  # 保持通道数不变
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,  # 关键：深度可分离
            act_type=dw_act_type,
            padding=padding
        )
        
        # Pointwise Convolution: 1×1卷积用于通道混合
        # 这里进行通道数的变换
        self.pw_conv = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act_type=pw_act_type
            # padding自动计算为0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            
        Returns:
            输出张量 [B, C_out, H', W']
            其中 H' = (H + 2*padding - kernel_size) / stride + 1
        """
        x = self.dw_conv(x)  # Depthwise: [B, C_in, H', W']
        x = self.pw_conv(x)  # Pointwise: [B, C_out, H', W']
        return x
    
    def get_params_count(self) -> dict:
        """
        计算参数量统计
        
        Returns:
            包含depthwise和pointwise参数量的字典
        """
        dw_params = sum(p.numel() for p in self.dw_conv.parameters())
        pw_params = sum(p.numel() for p in self.pw_conv.parameters())
        total_params = dw_params + pw_params
        
        return {
            'depthwise': dw_params,
            'pointwise': pw_params,
            'total': total_params,
            'reduction_ratio': f"{total_params / (dw_params * self.dw_conv.conv.in_channels):.2%}"
        }


# 只暴露必要的接口
__all__ = ['DepthwiseSeparableConv']