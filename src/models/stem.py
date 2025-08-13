# -*- coding: utf-8 -*-
"""
Stem 模块
Lite-HRNet-18的特征提取阶段 (Stage1)
"""
import torch
import torch.nn as nn
from .lite_block import LiteBlock


class Stem(nn.Module):
    """
    Stem - 特征提取阶段 (Stage1)
    
    功能：将输入从 [B, 4, 256, 512] 降采样到 [B, 32, 64, 128]
    结构：Conv1 → Conv2 → LiteBlock×2
    """
    
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 2):
        """
        Args:
            in_channels: 输入通道数 (RGB + padding mask)
            out_channels: 输出通道数
            expansion: LiteBlock扩张倍率
        """
        super().__init__()
        
        # Conv1: 3×3, stride=2, 32通道
        # [B, 4, 256, 512] → [B, 32, 128, 256]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.SiLU(inplace=True)
        )
        
        # Conv2: 3×3, stride=2, 32通道
        # [B, 32, 128, 256] → [B, 32, 64, 128]
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.SiLU(inplace=True)
        )
        
        # LiteBlock×2: 保持分辨率不变
        # [B, 32, 64, 128] → [B, 32, 64, 128]
        self.lite_blocks = nn.Sequential(
            LiteBlock(out_channels, out_channels, expansion=expansion, stride=1),
            LiteBlock(out_channels, out_channels, expansion=expansion, stride=1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [B, 4, H, W]，其中H和W应该能被4整除
            
        Returns:
            输出张量 [B, 32, H/4, W/4]
        """
        # 第一次降采样: 1/2分辨率
        x = self.conv1(x)
        
        # 第二次降采样: 1/4分辨率  
        x = self.conv2(x)
        
        # 特征增强（保持分辨率）
        x = self.lite_blocks(x)
        
        return x


def create_stem(config: dict):
    """
    创建Stem模块的工厂函数
    
    Args:
        config: 配置字典，必须包含以下键：
            - in_channels: 输入通道数
            - out_channels: 输出通道数
            - expansion: LiteBlock扩张倍率
    
    Returns:
        Stem模块实例
    
    Raises:
        ValueError: 配置缺少必要的键
    """
    if config is None:
        raise ValueError("必须提供配置字典")
    
    required_keys = ['in_channels', 'out_channels', 'expansion']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置缺少必要的键: {key}")
    
    in_channels = config['in_channels']
    out_channels = config['out_channels']
    expansion = config['expansion']
    
    return Stem(in_channels, out_channels, expansion)