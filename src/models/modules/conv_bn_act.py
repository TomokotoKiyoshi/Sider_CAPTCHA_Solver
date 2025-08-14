# -*- coding: utf-8 -*-
"""
ConvBNAct模块
Conv + BatchNorm + Activation 组合层
"""
import torch
import torch.nn as nn
from typing import Optional


class ConvBNAct(nn.Module):
    """
    Conv + BatchNorm + Activation 组合模块
    标准的卷积-批归一化-激活函数组合
    """
    
    # 支持的激活函数类型
    SUPPORTED_ACTIVATIONS = {
        'silu': lambda: nn.SiLU(inplace=True),
        'relu': lambda: nn.ReLU(inplace=True),
        'relu6': lambda: nn.ReLU6(inplace=True),
        'none': lambda: nn.Identity(),
        'identity': lambda: nn.Identity()
    }
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 groups: int,
                 act_type: str,
                 bias: bool = False,
                 bn_eps: float = 1e-5,
                 bn_momentum: float = 0.1,
                 padding: Optional[int] = None):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数  
            kernel_size: 卷积核大小
            stride: 步长
            groups: 分组卷积的组数
            act_type: 激活函数类型
            bias: 是否使用偏置 (默认False，因为使用BatchNorm时不需要)
            bn_eps: BatchNorm的epsilon参数 (默认1e-5，PyTorch标准值)
            bn_momentum: BatchNorm的momentum参数 (默认0.1，PyTorch标准值)
            padding: 填充，如果为None则自动计算
        """
        super().__init__()
        
        # 自动计算padding
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        # 卷积层
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias
        )
        
        # BatchNorm层
        self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        
        # 激活函数
        if act_type not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"不支持的激活函数类型: {act_type}. "
                f"支持的类型: {list(self.SUPPORTED_ACTIVATIONS.keys())}"
            )
        self.act = self.SUPPORTED_ACTIVATIONS[act_type]()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# 只暴露ConvBNAct接口
__all__ = ['ConvBNAct']