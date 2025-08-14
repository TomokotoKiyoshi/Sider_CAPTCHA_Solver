# -*- coding: utf-8 -*-
"""
Stage4 模块
Lite-HRNet的四分支结构实现，包含跨分辨率融合CRF-4
这是主干网络的最终阶段，提供多尺度特征供下游使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .modules import ConvBNAct, DepthwiseSeparableConv, LiteBlock


class CrossResolutionFusion4(nn.Module):
    """
    跨分辨率融合模块 (CRF-4)
    实现HRNet风格的四向特征融合：对齐→相加→平滑
    """
    
    def __init__(self,
                 channels_1_4: int = 32,
                 channels_1_8: int = 64,
                 channels_1_16: int = 128,
                 channels_1_32: int = 256):
        """
        Args:
            channels_1_4: 1/4分辨率分支的通道数
            channels_1_8: 1/8分辨率分支的通道数
            channels_1_16: 1/16分辨率分支的通道数
            channels_1_32: 1/32分辨率分支的通道数
        """
        super().__init__()
        
        # ========== 汇到1/4分支的融合路径 ==========
        # 1/8 → 1/4: 上采样×2
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
        
        # 1/16 → 1/4: 上采样×4
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
        
        # 1/32 → 1/4: 上采样×8
        self.up_1_32_to_1_4 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            ConvBNAct(
                in_channels=channels_1_32,
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
        
        # 1/16 → 1/8: 上采样×2
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
        
        # 1/32 → 1/8: 上采样×4
        self.up_1_32_to_1_8 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            ConvBNAct(
                in_channels=channels_1_32,
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
        # 1/4 → 1/16: 下采样×4 (分两步)
        self.down_1_4_to_1_16 = nn.Sequential(
            DepthwiseSeparableConv(
                channels_1_4, channels_1_4,
                kernel_size=3, stride=2
            ),
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
        
        # 1/32 → 1/16: 上采样×2
        self.up_1_32_to_1_16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNAct(
                in_channels=channels_1_32,
                out_channels=channels_1_16,
                kernel_size=1,
                stride=1,
                groups=1,
                act_type='none'
            )
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
        
        # ========== 汇到1/32分支的融合路径 ==========
        # 1/4 → 1/32: 下采样×8 (分三步)
        self.down_1_4_to_1_32 = nn.Sequential(
            DepthwiseSeparableConv(
                channels_1_4, channels_1_4,
                kernel_size=3, stride=2
            ),
            DepthwiseSeparableConv(
                channels_1_4, channels_1_4,
                kernel_size=3, stride=2
            ),
            DepthwiseSeparableConv(
                channels_1_4, channels_1_32,
                kernel_size=3, stride=2
            )
        )
        
        # 1/8 → 1/32: 下采样×4 (分两步)
        self.down_1_8_to_1_32 = nn.Sequential(
            DepthwiseSeparableConv(
                channels_1_8, channels_1_8,
                kernel_size=3, stride=2
            ),
            DepthwiseSeparableConv(
                channels_1_8, channels_1_32,
                kernel_size=3, stride=2
            )
        )
        
        # 1/16 → 1/32: 下采样×2
        self.down_1_16_to_1_32 = DepthwiseSeparableConv(
            channels_1_16, channels_1_32,
            kernel_size=3, stride=2, padding=1
        )
        
        # 1/32分支平滑卷积
        self.smooth_1_32 = ConvBNAct(
            in_channels=channels_1_32,
            out_channels=channels_1_32,
            kernel_size=3,
            stride=1,
            groups=1,
            act_type='silu'
        )
    
    def forward(self, 
                x_1_4: torch.Tensor, 
                x_1_8: torch.Tensor,
                x_1_16: torch.Tensor,
                x_1_32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_1_4: 1/4分辨率特征 [B, 32, 64, 128]
            x_1_8: 1/8分辨率特征 [B, 64, 32, 64]
            x_1_16: 1/16分辨率特征 [B, 128, 16, 32]
            x_1_32: 1/32分辨率特征 [B, 256, 8, 16]
        
        Returns:
            融合后的特征:
            - b1: [B, 32, 64, 128] (1/4分辨率)
            - b2: [B, 64, 32, 64] (1/8分辨率)
            - b3: [B, 128, 16, 32] (1/16分辨率)
            - b4: [B, 256, 8, 16] (1/32分辨率)
        """
        # 汇到1/4分支
        x_1_8_to_1_4 = self.up_1_8_to_1_4(x_1_8)
        x_1_16_to_1_4 = self.up_1_16_to_1_4(x_1_16)
        x_1_32_to_1_4 = self.up_1_32_to_1_4(x_1_32)
        b1 = x_1_4 + x_1_8_to_1_4 + x_1_16_to_1_4 + x_1_32_to_1_4
        b1 = self.smooth_1_4(b1)
        
        # 汇到1/8分支
        x_1_4_to_1_8 = self.down_1_4_to_1_8(x_1_4)
        x_1_16_to_1_8 = self.up_1_16_to_1_8(x_1_16)
        x_1_32_to_1_8 = self.up_1_32_to_1_8(x_1_32)
        b2 = x_1_8 + x_1_4_to_1_8 + x_1_16_to_1_8 + x_1_32_to_1_8
        b2 = self.smooth_1_8(b2)
        
        # 汇到1/16分支
        x_1_4_to_1_16 = self.down_1_4_to_1_16(x_1_4)
        x_1_8_to_1_16 = self.down_1_8_to_1_16(x_1_8)
        x_1_32_to_1_16 = self.up_1_32_to_1_16(x_1_32)
        b3 = x_1_16 + x_1_4_to_1_16 + x_1_8_to_1_16 + x_1_32_to_1_16
        b3 = self.smooth_1_16(b3)
        
        # 汇到1/32分支
        x_1_4_to_1_32 = self.down_1_4_to_1_32(x_1_4)
        x_1_8_to_1_32 = self.down_1_8_to_1_32(x_1_8)
        x_1_16_to_1_32 = self.down_1_16_to_1_32(x_1_16)
        b4 = x_1_32 + x_1_4_to_1_32 + x_1_8_to_1_32 + x_1_16_to_1_32
        b4 = self.smooth_1_32(b4)
        
        return b1, b2, b3, b4


class Stage4(nn.Module):
    """
    Stage4 - 四分支结构
    
    功能：建立1/32分支，维护四个分辨率分支，进行跨分辨率融合
    输入：三个分支的特征 (来自Stage3)
        - T1: [B, 32, 64, 128] (1/4分辨率)
        - T2: [B, 64, 32, 64] (1/8分辨率)
        - T3: [B, 128, 16, 32] (1/16分辨率)
    输出：四个分支的特征（主干最终输出）
        - B1: [B, 32, 64, 128] (1/4分辨率)
        - B2: [B, 64, 32, 64] (1/8分辨率)
        - B3: [B, 128, 16, 32] (1/16分辨率)
        - B4: [B, 256, 8, 16] (1/32分辨率)
    """
    
    def __init__(self,
                 channels: List[int],
                 num_blocks: List[int],
                 expansion: int):
        """
        Args:
            channels: 各分支通道数列表 [1/4, 1/8, 1/16, 1/32分辨率]
            num_blocks: 各分支的LiteBlock数量列表
            expansion: LiteBlock扩张倍率
        """
        super().__init__()
        
        # 从配置提取参数
        channels_1_4 = channels[0]   # 32
        channels_1_8 = channels[1]   # 64
        channels_1_16 = channels[2]  # 128
        channels_1_32 = channels[3]  # 256
        blocks_1_4 = num_blocks[0]   # 2
        blocks_1_8 = num_blocks[1]   # 2
        blocks_1_16 = num_blocks[2]  # 2
        blocks_1_32 = num_blocks[3]  # 2
        
        # Transition4: 建立1/32分支
        # T4.b4_from_1_16: 从1/16下采到1/32
        self.t4_b4_from_1_16 = DepthwiseSeparableConv(
            channels_1_16, channels_1_32,
            kernel_size=3, stride=2, padding=1
        )
        
        # 四个分支的残差块
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
        
        # 1/32分支: LiteBlock×num_blocks[3]
        self.branch_1_32 = nn.Sequential(*[
            LiteBlock(channels_1_32, channels_1_32, expansion=expansion, stride=1)
            for _ in range(blocks_1_32)
        ])
        
        # 跨分辨率融合CRF-4
        self.fusion = CrossResolutionFusion4(
            channels_1_4, channels_1_8, channels_1_16, channels_1_32
        )
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征列表
               - [0]: T1 [B, 32, 64, 128] (1/4分辨率)
               - [1]: T2 [B, 64, 32, 64] (1/8分辨率)
               - [2]: T3 [B, 128, 16, 32] (1/16分辨率)
        
        Returns:
            融合后的特征列表（主干最终输出）:
            - [0]: B1 [B, 32, 64, 128] (1/4分辨率)
            - [1]: B2 [B, 64, 32, 64] (1/8分辨率)
            - [2]: B3 [B, 128, 16, 32] (1/16分辨率)
            - [3]: B4 [B, 256, 8, 16] (1/32分辨率)
        """
        t1, t2, t3 = x[0], x[1], x[2]
        
        # Transition4: 建立1/32分支
        u1_in = t1                             # [B, 32, 64, 128]
        u2_in = t2                             # [B, 64, 32, 64]
        u3_in = t3                             # [B, 128, 16, 32]
        u4_in = self.t4_b4_from_1_16(t3)      # [B, 256, 8, 16]
        
        # 每分支处理
        u1_mid = self.branch_1_4(u1_in)       # [B, 32, 64, 128]
        u2_mid = self.branch_1_8(u2_in)       # [B, 64, 32, 64]
        u3_mid = self.branch_1_16(u3_in)      # [B, 128, 16, 32]
        u4_mid = self.branch_1_32(u4_in)      # [B, 256, 8, 16]
        
        # 跨分辨率融合
        b1, b2, b3, b4 = self.fusion(u1_mid, u2_mid, u3_mid, u4_mid)
        
        return [b1, b2, b3, b4]


def create_stage4(config: dict):
    """
    创建Stage4模块的工厂函数
    
    Args:
        config: 配置字典，必须包含:
            - channels: 各分支通道数列表 [1/4, 1/8, 1/16, 1/32分辨率]
            - num_blocks: 各分支的LiteBlock数量列表
            - expansion: LiteBlock扩张倍率
    
    Returns:
        Stage4模块实例
    
    Raises:
        ValueError: 缺少必要的配置参数
    """
    required_keys = ['channels', 'num_blocks', 'expansion']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Stage4配置缺少必要的参数: {key}")
    
    # 验证列表长度
    if len(config['channels']) != 4:
        raise ValueError("Stage4 channels必须包含4个元素 [1/4, 1/8, 1/16, 1/32分辨率]")
    
    if len(config['num_blocks']) != 4:
        raise ValueError("Stage4 num_blocks必须包含4个元素")
    
    return Stage4(
        channels=config['channels'],
        num_blocks=config['num_blocks'],
        expansion=config['expansion']
    )