# -*- coding: utf-8 -*-
"""
Stage5 - LiteFPN模块
轻量级特征金字塔网络，统一多尺度特征到128通道并聚合到1/4分辨率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .modules import ConvBNAct


class FusionModule(nn.Module):
    """
    特征融合模块
    支持三种融合方式：直接相加、可学习权重、通道注意力
    功能：融合两个特征图，输出平滑后的特征图
    """
    
    def __init__(self,
                 channels: int = 128,
                 fusion_type: str = 'add'):
        """
        Args:
            channels: 特征通道数
            fusion_type: 融合类型 ('add', 'weighted', 'attention')
        """
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'weighted':
            # BiFPN风格的可学习权重
            self.weight_a = nn.Parameter(torch.ones(1))
            self.weight_b = nn.Parameter(torch.ones(1))
            self.epsilon = 1e-4
        elif fusion_type == 'attention':
            # 简单的通道注意力（SE风格）
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channels * 2, channels // 4, 1)
            self.fc2 = nn.Conv2d(channels // 4, channels * 2, 1)
            self.sigmoid = nn.Sigmoid()
        
        # 融合后的平滑卷积
        self.smooth = ConvBNAct(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            groups=1,
            act_type='silu'
        )
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        融合两个特征图
        
        Args:
            a: 第一个特征图 [B, C, H, W]
            b: 第二个特征图 [B, C, H, W]
        
        Returns:
            融合后的特征图 [B, C, H, W]
        """
        if self.fusion_type == 'add':
            # === 直接相加融合 (最轻量，零参数) ===
            # 简单地将两个特征图逐元素相加
            # 优点：计算快速，无额外参数
            # 缺点：不同尺度特征的贡献比例固定为1:1，不可学习
            # 适用场景：特征已经过良好的归一化，语义层级相近
            fused = a + b
            
        elif self.fusion_type == 'weighted':
            # === BiFPN风格的可学习权重融合 (推荐用于size confusion场景) ===
            # 为每个输入特征学习一个标量权重，通过softplus保证非负
            # 优点：能自适应调整不同分辨率特征的贡献比例
            # 缺点：增加2个可学习参数
            # 适用场景：处理混淆缺口、尺寸干扰等复杂验证码
            
            # Step 1: 使用softplus(x) = log(1 + exp(x))确保权重非负
            w_a = F.softplus(self.weight_a)
            w_b = F.softplus(self.weight_b)
            w_sum = w_a + w_b + self.epsilon  # epsilon防止除零
            
            # Step 2: 归一化权重，使其和为1
            w_a_norm = w_a / w_sum
            w_b_norm = w_b / w_sum
            
            # Step 3: 加权融合
            fused = w_a_norm * a + w_b_norm * b
            
        elif self.fusion_type == 'attention':
            # === SE风格的通道注意力融合 (适用于强噪声场景) ===
            # 通过学习通道级的注意力权重来动态调整融合
            # 优点：能够根据内容自适应地调整每个通道的重要性
            # 缺点：计算开销较大，参数量增加
            # 适用场景：柏林噪声、高光干扰等需要选择性融合的场景
            
            # Step 1: 拼接两个特征图
            concat = torch.cat([a, b], dim=1)  # [B, 2C, H, W]
            
            # Step 2: 全局平均池化，获取每个通道的全局统计信息
            gap = self.global_pool(concat)  # [B, 2C, 1, 1]
            
            # Step 3: 通过两层全连接学习通道注意力权重
            # 先降维到C/4（瓶颈层），再升维回2C
            attn = self.fc1(gap)       # [B, C/4, 1, 1]
            attn = F.relu(attn)        # 激活
            attn = self.fc2(attn)      # [B, 2C, 1, 1]
            attn = self.sigmoid(attn)  # 归一化到[0,1]
            
            # Step 4: 分别应用注意力权重到两个输入
            attn_a, attn_b = torch.split(attn, a.shape[1], dim=1)
            fused = a * attn_a + b * attn_b
        else:
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")
        
        # 融合后的平滑卷积（3×3 Conv + BN + SiLU）
        # 作用：对齐统计分布，抑制上采样产生的棋盘效应
        return self.smooth(fused)


class LiteFPN(nn.Module):
    """
    Stage5 - 轻量级特征金字塔网络（LiteFPN）
    
    功能：统一低分辨率语义到128通道，逐级上采样回1/4分辨率，与高分辨率分支融合
    
    输入：四个分支的特征（来自Stage4）
        - B1: [B, 32, 64, 128] (1/4分辨率)
        - B2: [B, 64, 32, 64] (1/8分辨率)
        - B3: [B, 128, 16, 32] (1/16分辨率)
        - B4: [B, 256, 8, 16] (1/32分辨率)
    
    输出：
        主特征：
        - Hf: [B, 128, 64, 128] (1/4分辨率，用于密集头)
        
        中间金字塔（可选，用于辅助监督）：
        - P3_td: [B, 128, 32, 64] (1/8分辨率)
        - P4_td: [B, 128, 16, 32] (1/16分辨率)
        - P5: [B, 128, 8, 16] (1/32分辨率)
    """
    
    def __init__(self,
                 in_channels: List[int] = [32, 64, 128, 256],
                 fpn_channels: int = 128,
                 fusion_type: str = 'add',
                 return_pyramid: bool = False):
        """
        Args:
            in_channels: 输入各分支的通道数 [1/4, 1/8, 1/16, 1/32]
            fpn_channels: FPN统一通道数（默认128）
            fusion_type: 融合类型 ('add', 'weighted', 'attention')
            return_pyramid: 是否返回中间金字塔特征
        """
        super().__init__()
        
        self.fpn_channels = fpn_channels
        self.fusion_type = fusion_type
        self.return_pyramid = return_pyramid
        
        # ========== 1. 侧连接（统一到128通道） ==========
        # L5: 1×1(256→128) for B4
        self.lateral_1_32 = ConvBNAct(
            in_channels=in_channels[3],  # 256
            out_channels=fpn_channels,   # 128
            kernel_size=1,
            stride=1,
            groups=1,
            act_type='none',  # 侧连接不加激活
            bias=False
        )
        
        # L4: 1×1(128→128) for B3
        self.lateral_1_16 = ConvBNAct(
            in_channels=in_channels[2],  # 128
            out_channels=fpn_channels,   # 128
            kernel_size=1,
            stride=1,
            groups=1,
            act_type='none',
            bias=False
        )
        
        # L3: 1×1(64→128) for B2
        self.lateral_1_8 = ConvBNAct(
            in_channels=in_channels[1],  # 64
            out_channels=fpn_channels,   # 128
            kernel_size=1,
            stride=1,
            groups=1,
            act_type='none',
            bias=False
        )
        
        # L2: 1×1(32→128) for B1
        self.lateral_1_4 = ConvBNAct(
            in_channels=in_channels[0],  # 32
            out_channels=fpn_channels,   # 128
            kernel_size=1,
            stride=1,
            groups=1,
            act_type='none',
            bias=False
        )
        
        # ========== 2. 平滑卷积 ==========
        # P5平滑（最低层减少噪声）
        self.smooth_p5 = ConvBNAct(
            in_channels=fpn_channels,
            out_channels=fpn_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            act_type='silu'
        )
        
        # ========== 3. 融合模块 ==========
        # 1/32 → 1/16 融合
        self.fuse_1_16 = FusionModule(fpn_channels, fusion_type)
        
        # 1/16 → 1/8 融合
        self.fuse_1_8 = FusionModule(fpn_channels, fusion_type)
        
        # 1/8 → 1/4 融合（与高分辨率分支）
        self.fuse_1_4 = FusionModule(fpn_channels, fusion_type)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征列表
                - [0]: B1 [B, 32, 64, 128] (1/4分辨率)
                - [1]: B2 [B, 64, 32, 64] (1/8分辨率)
                - [2]: B3 [B, 128, 16, 32] (1/16分辨率)
                - [3]: B4 [B, 256, 8, 16] (1/32分辨率)
        
        Returns:
            如果return_pyramid=False:
                Hf: [B, 128, 64, 128] (1/4分辨率主特征)
            如果return_pyramid=True:
                (Hf, [P3_td, P4_td, P5])：主特征和中间金字塔
        """
        b1, b2, b3, b4 = features[0], features[1], features[2], features[3]
        
        # ========== 1. 侧连接：统一到128通道 ==========
        p5 = self.lateral_1_32(b4)  # [B, 128, 8, 16]
        p4 = self.lateral_1_16(b3)  # [B, 128, 16, 32]
        p3 = self.lateral_1_8(b2)   # [B, 128, 32, 64]
        hr = self.lateral_1_4(b1)   # [B, 128, 64, 128]
        
        # ========== 2. 自顶向下逐级上采样与融合 ==========
        # 2.1 从1/32到1/16
        # 先平滑P5
        p5_smooth = self.smooth_p5(p5)  # [B, 128, 8, 16]
        
        # 上采样到1/16
        u5 = F.interpolate(
            p5_smooth,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )  # [B, 128, 16, 32]
        
        # 融合得到P4_td
        p4_td = self.fuse_1_16(u5, p4)  # [B, 128, 16, 32]
        
        # 2.2 从1/16到1/8
        # 上采样到1/8
        u4 = F.interpolate(
            p4_td,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )  # [B, 128, 32, 64]
        
        # 融合得到P3_td
        p3_td = self.fuse_1_8(u4, p3)  # [B, 128, 32, 64]
        
        # 2.3 从1/8到1/4（与高分辨率分支融合）
        # 上采样到1/4
        u3 = F.interpolate(
            p3_td,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )  # [B, 128, 64, 128]
        
        # 与高分辨率分支融合得到主特征
        hf = self.fuse_1_4(hr, u3)  # [B, 128, 64, 128]
        
        # 返回结果
        if self.return_pyramid:
            # 返回主特征和中间金字塔
            pyramid = [p3_td, p4_td, p5]
            return hf, pyramid
        else:
            # 只返回主特征
            return hf


def create_stage5_lite_fpn(config: dict):
    """
    创建Stage5 LiteFPN模块的工厂函数
    
    Args:
        config: 配置字典，必须包含:
            - in_channels: 输入各分支的通道数
            - fpn_channels: FPN统一通道数
            - fusion_type: 融合类型
            - return_pyramid: 是否返回中间金字塔
    
    Returns:
        LiteFPN模块实例
    """
    # 默认配置
    default_config = {
        'in_channels': [32, 64, 128, 256],
        'fpn_channels': 128,
        'fusion_type': 'add',
        'return_pyramid': False
    }
    
    # 合并用户配置
    final_config = {**default_config, **config}
    
    return LiteFPN(
        in_channels=final_config['in_channels'],
        fpn_channels=final_config['fpn_channels'],
        fusion_type=final_config['fusion_type'],
        return_pyramid=final_config['return_pyramid']
    )