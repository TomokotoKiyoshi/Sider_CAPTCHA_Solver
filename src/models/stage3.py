# -*- coding: utf-8 -*-
"""
Stage3 模块
Lite-HRNet的三分支结构实现，包含跨分辨率融合CRF-3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .modules import ConvBNAct, DepthwiseSeparableConv, LiteBlock


class CrossResolutionFusion3(nn.Module):
    """
    跨分辨率融合模块 (CRF-3)
    实现HRNet风格的三向特征融合：对齐→相加→平滑
    """
    
    def __init__(self,
                 channels_1_4: int = 32,
                 channels_1_8: int = 64,
                 channels_1_16: int = 128):
        """
        Args:
            channels_1_4: 1/4分辨率分支的通道数
            channels_1_8: 1/8分辨率分支的通道数
            channels_1_16: 1/16分辨率分支的通道数
        """
        super().__init__()
        
        # ========== 汇到1/4分支的融合路径 ==========
        # 1/8 → 1/4: 上采样×2 + 通道对齐
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
        
        # 1/16 → 1/4: 上采样×4 + 通道对齐
        self.up_1_16_to_1_4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            ConvBNAct(
                in_channels=channels_1_16,
                out_channels=channels_1_4,
                kernel_size=1,
                stride=1,
                groups=1,
                act_type='none'
            )
        )
        
        # 1/4分支平滑卷积
        self.smooth_1_4 = ConvBNAct(
            in_channels=channels_1_4,
            out_channels=channels_1_4,
            kernel_size=3,
            stride=1,
            groups=1,
            act_type='silu'
        )
        
        # ========== 汇到1/8分支的融合路径 ==========
        # 1/4 → 1/8: 下采样×2
        self.down_1_4_to_1_8 = DepthwiseSeparableConv(
            channels_1_4, channels_1_8,
            kernel_size=3, stride=2, padding=1
        )
        
        # 1/16 → 1/8: 上采样×2 + 通道对齐
        self.up_1_16_to_1_8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNAct(
                in_channels=channels_1_16,
                out_channels=channels_1_8,
                kernel_size=1,
                stride=1,
                groups=1,
                act_type='none'
            )
        )
        
        # 1/8分支平滑卷积
        self.smooth_1_8 = ConvBNAct(
            in_channels=channels_1_8,
            out_channels=channels_1_8,
            kernel_size=3,
            stride=1,
            groups=1,
            act_type='silu'
        )
        
        # ========== 汇到1/16分支的融合路径 ==========
        # 1/4 → 1/16: 下采样×4 (分两步stride=2)
        self.down_1_4_to_1_16 = nn.Sequential(
            # 第一步: 1/4 → 1/8
            DepthwiseSeparableConv(
                channels_1_4, channels_1_4,
                kernel_size=3, stride=2
            ),
            # 第二步: 1/8 → 1/16
            DepthwiseSeparableConv(
                channels_1_4, channels_1_16,
                kernel_size=3, stride=2
            )
        )
        
        # 1/8 → 1/16: 下采样×2
        self.down_1_8_to_1_16 = DepthwiseSeparableConv(
            channels_1_8, channels_1_16,
            kernel_size=3, stride=2
        )
        
        # 1/16分支平滑卷积
        self.smooth_1_16 = ConvBNAct(
            in_channels=channels_1_16,
            out_channels=channels_1_16,
            kernel_size=3,
            stride=1,
            groups=1,
            act_type='silu'
        )
    
    def forward(self, 
                x_1_4: torch.Tensor, 
                x_1_8: torch.Tensor,
                x_1_16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_1_4: 1/4分辨率特征 [B, 32, 64, 128]
            x_1_8: 1/8分辨率特征 [B, 64, 32, 64]
            x_1_16: 1/16分辨率特征 [B, 128, 16, 32]
        
        Returns:
            融合后的特征:
            - t1: [B, 32, 64, 128] (1/4分辨率)
            - t2: [B, 64, 32, 64] (1/8分辨率)
            - t3: [B, 128, 16, 32] (1/16分辨率)
        """
        # 汇到1/4分支
        x_1_8_to_1_4 = self.up_1_8_to_1_4(x_1_8)
        x_1_16_to_1_4 = self.up_1_16_to_1_4(x_1_16)
        t1 = x_1_4 + x_1_8_to_1_4 + x_1_16_to_1_4
        t1 = self.smooth_1_4(t1)
        
        # 汇到1/8分支
        x_1_4_to_1_8 = self.down_1_4_to_1_8(x_1_4)
        x_1_16_to_1_8 = self.up_1_16_to_1_8(x_1_16)
        t2 = x_1_8 + x_1_4_to_1_8 + x_1_16_to_1_8
        t2 = self.smooth_1_8(t2)
        
        # 汇到1/16分支
        x_1_4_to_1_16 = self.down_1_4_to_1_16(x_1_4)
        x_1_8_to_1_16 = self.down_1_8_to_1_16(x_1_8)
        t3 = x_1_16 + x_1_4_to_1_16 + x_1_8_to_1_16
        t3 = self.smooth_1_16(t3)
        
        return t1, t2, t3


class Stage3(nn.Module):
    """
    Stage3 - 三分支结构
    
    功能：建立1/16分支，维护三个分辨率分支，进行跨分辨率融合
    输入：两个分支的特征 (来自Stage2)
        - Y1: [B, 32, 64, 128] (1/4分辨率)
        - Y2: [B, 64, 32, 64] (1/8分辨率)
    输出：三个分支的特征
        - T1: [B, 32, 64, 128] (1/4分辨率)
        - T2: [B, 64, 32, 64] (1/8分辨率)
        - T3: [B, 128, 16, 32] (1/16分辨率)
    """
    
    def __init__(self,
                 channels: List[int],
                 num_blocks: List[int],
                 expansion: int):
        """
        Args:
            channels: 各分支通道数列表 [1/4分辨率, 1/8分辨率, 1/16分辨率]
            num_blocks: 各分支的LiteBlock数量列表
            expansion: LiteBlock扩张倍率
        """
        super().__init__()
        
        # 从配置提取参数
        channels_1_4 = channels[0]   # 32
        channels_1_8 = channels[1]   # 64
        channels_1_16 = channels[2]  # 128
        blocks_1_4 = num_blocks[0]   # 3
        blocks_1_8 = num_blocks[1]   # 3
        blocks_1_16 = num_blocks[2]  # 3
        
        # Transition3: 建立1/16分支
        # T3.b3_from_1_8: 从1/8下采到1/16
        self.t3_b3_from_1_8 = DepthwiseSeparableConv(
            channels_1_8, channels_1_16,
            kernel_size=3, stride=2, padding=1
        )
        
        # 三个分支的残差块
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
        
        # 1/16分支: LiteBlock×num_blocks[2]
        self.branch_1_16 = nn.Sequential(*[
            LiteBlock(channels_1_16, channels_1_16, expansion=expansion, stride=1)
            for _ in range(blocks_1_16)
        ])
        
        # 跨分辨率融合CRF-3
        self.fusion = CrossResolutionFusion3(channels_1_4, channels_1_8, channels_1_16)
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征列表
               - [0]: Y1 [B, 32, 64, 128] (1/4分辨率)
               - [1]: Y2 [B, 64, 32, 64] (1/8分辨率)
        
        Returns:
            融合后的特征列表:
            - [0]: T1 [B, 32, 64, 128] (1/4分辨率)
            - [1]: T2 [B, 64, 32, 64] (1/8分辨率)
            - [2]: T3 [B, 128, 16, 32] (1/16分辨率)
        """
        y1, y2 = x[0], x[1]
        
        # Transition3: 建立1/16分支
        z1_in = y1                           # [B, 32, 64, 128]
        z2_in = y2                           # [B, 64, 32, 64]
        z3_in = self.t3_b3_from_1_8(y2)     # [B, 128, 16, 32]
        
        # 每分支处理
        z1_mid = self.branch_1_4(z1_in)     # [B, 32, 64, 128]
        z2_mid = self.branch_1_8(z2_in)     # [B, 64, 32, 64]
        z3_mid = self.branch_1_16(z3_in)    # [B, 128, 16, 32]
        
        # 跨分辨率融合
        t1, t2, t3 = self.fusion(z1_mid, z2_mid, z3_mid)
        
        return [t1, t2, t3]


def create_stage3(config: dict):
    """
    创建Stage3模块的工厂函数
    
    Args:
        config: 配置字典，必须包含:
            - channels: 各分支通道数列表 [1/4分辨率, 1/8分辨率, 1/16分辨率]
            - num_blocks: 各分支的LiteBlock数量列表
            - expansion: LiteBlock扩张倍率
    
    Returns:
        Stage3模块实例
    
    Raises:
        ValueError: 缺少必要的配置参数
    """
    required_keys = ['channels', 'num_blocks', 'expansion']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Stage3配置缺少必要的参数: {key}")
    
    # 验证列表长度
    if len(config['channels']) != 3:
        raise ValueError("Stage3 channels必须包含3个元素 [1/4分辨率, 1/8分辨率, 1/16分辨率]")
    
    if len(config['num_blocks']) != 3:
        raise ValueError("Stage3 num_blocks必须包含3个元素")
    
    return Stage3(
        channels=config['channels'],
        num_blocks=config['num_blocks'],
        expansion=config['expansion']
    )