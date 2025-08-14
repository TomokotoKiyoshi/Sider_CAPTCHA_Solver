# -*- coding: utf-8 -*-
"""
Stage2 模块
Lite-HRNet的双分支结构实现，包含跨分辨率融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .modules import ConvBNAct, DepthwiseSeparableConv, LiteBlock


class CrossResolutionFusion(nn.Module):
    """
    跨分辨率融合模块 (CRF-2)
    实现HRNet风格的双向特征融合：对齐→相加→平滑
    """
    
    def __init__(self,
                 channels_1_4: int = 32,
                 channels_1_8: int = 64):
        """
        Args:
            channels_1_4: 1/4分辨率分支的通道数
            channels_1_8: 1/8分辨率分支的通道数
        """
        super().__init__()
        
        # 到1/4分支的融合路径
        # 上采样1/8到1/4：双线性插值 + 1×1通道对齐
        self.up_1_8_to_1_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNAct(
                in_channels=channels_1_8,
                out_channels=channels_1_4,
                kernel_size=1,
                stride=1,
                groups=1,
                act_type='none'
            )
        )
        
        # 1/4分支SiLU平滑卷积
        self.smooth_1_4 = ConvBNAct(
            in_channels=channels_1_4,
            out_channels=channels_1_4,
            kernel_size=3,
            stride=1,
            groups=1,
            act_type='silu'
        )
        
        # 到1/8分支的融合路径
        # 下采样1/4到1/8：DWConv s=2 + PW通道对齐
        self.down_1_4_to_1_8 = DepthwiseSeparableConv(
            channels_1_4, channels_1_8,
            kernel_size=3, stride=2, padding=1
        )
        
        # 1/8分支SiLU平滑卷积
        self.smooth_1_8 = ConvBNAct(
            in_channels=channels_1_8,
            out_channels=channels_1_8,
            kernel_size=3,
            stride=1,
            groups=1,
            act_type='silu'
        )
    
    def forward(self, x_1_4: torch.Tensor, x_1_8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_1_4: 1/4分辨率特征 [B, 32, 64, 128]
            x_1_8: 1/8分辨率特征 [B, 64, 32, 64]
        
        Returns:
            融合后的特征:
            - y_1_4: [B, 32, 64, 128]
            - y_1_8: [B, 64, 32, 64]
        """
        # 融合到1/4分支
        x_1_8_up = self.up_1_8_to_1_4(x_1_8)
        y_1_4 = x_1_4 + x_1_8_up
        y_1_4 = self.smooth_1_4(y_1_4)
        
        # 融合到1/8分支
        x_1_4_down = self.down_1_4_to_1_8(x_1_4)
        y_1_8 = x_1_8 + x_1_4_down
        y_1_8 = self.smooth_1_8(y_1_8)
        
        return y_1_4, y_1_8


class Stage2(nn.Module):
    """
    Stage2 - 双分支结构
    
    功能：建立并维护1/4和1/8两个分辨率分支，进行跨分辨率融合
    输入：[B, 32, 64, 128] (来自Stage1)
    输出：两个分支的特征
        - Y1: [B, 32, 64, 128] (1/4分辨率)
        - Y2: [B, 64, 32, 64] (1/8分辨率)
    """
    
    def __init__(self,
                 in_channels: int,
                 channels: List[int],
                 num_blocks: List[int],
                 expansion: int):
        """
        Args:
            in_channels: 输入通道数
            channels: 各分支通道数列表 [1/4分辨率, 1/8分辨率]
            num_blocks: 各分支的LiteBlock数量列表
            expansion: LiteBlock扩张倍率
        """
        super().__init__()
        
        # 从配置提取参数
        channels_1_4 = channels[0]
        channels_1_8 = channels[1]
        blocks_1_4 = num_blocks[0]
        blocks_1_8 = num_blocks[1]
        
        # Transition: 从Stage1建立双分支
        # T2.b1_keep: 保持1/4分支
        self.t2_b1_keep = ConvBNAct(
            in_channels=in_channels,
            out_channels=channels_1_4,
            kernel_size=1,
            stride=1,
            groups=1,
            act_type='silu'
        )
        
        # T2.b2_down: 新建1/8分支
        self.t2_b2_down = DepthwiseSeparableConv(
            in_channels, channels_1_8,
            kernel_size=3, stride=2
        )
        
        # 每分支的残差块
        # 1/4分支: LiteBlock×num_blocks[0]
        self.branch_1_4 = nn.Sequential(*[
            LiteBlock(channels_1_4, channels_1_4, expansion=expansion, stride=1)
            for _ in range(blocks_1_4)
        ])
        
        # 1/8分支: LiteBlock×num_blocks[1]
        self.branch_1_8 = nn.Sequential(*[
            LiteBlock(channels_1_8, channels_1_8, expansion=expansion, stride=1)
            for _ in range(blocks_1_8)
        ])
        
        # 跨分辨率融合
        self.fusion = CrossResolutionFusion(channels_1_4, channels_1_8)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, 32, 64, 128]
        
        Returns:
            融合后的特征列表:
            - [0]: Y1 [B, 32, 64, 128] (1/4分辨率)
            - [1]: Y2 [B, 64, 32, 64] (1/8分辨率)
        """
        # Transition: 建立双分支
        s2_1_4_in = self.t2_b1_keep(x)    # [B, 32, 64, 128]
        s2_1_8_in = self.t2_b2_down(x)    # [B, 64, 32, 64]
        
        # 每分支处理
        s2_1_4_mid = self.branch_1_4(s2_1_4_in)  # [B, 32, 64, 128]
        s2_1_8_mid = self.branch_1_8(s2_1_8_in)  # [B, 64, 32, 64]
        
        # 跨分辨率融合
        y1, y2 = self.fusion(s2_1_4_mid, s2_1_8_mid)
        
        return [y1, y2]


def create_stage2(config: dict):
    """
    创建Stage2模块的工厂函数
    
    Args:
        config: 配置字典，必须包含:
            - in_channels: 输入通道数
            - channels: 各分支通道数列表 [1/4分辨率, 1/8分辨率]
            - num_blocks: 各分支的LiteBlock数量列表
            - expansion: LiteBlock扩张倍率
    
    Returns:
        Stage2模块实例
    
    Raises:
        ValueError: 缺少必要的配置参数
    """
    required_keys = ['in_channels', 'channels', 'num_blocks', 'expansion']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Stage2配置缺少必要的参数: {key}")
    
    return Stage2(
        in_channels=config['in_channels'],
        channels=config['channels'],
        num_blocks=config['num_blocks'],
        expansion=config['expansion']
    )


