# -*- coding: utf-8 -*-
"""
Models模块 - PMN-R3-FP 滑块验证码识别模型组件

包含以下核心组件：
- HRBackbone: 高分辨率多尺度骨干网络 (HRNet-style)
- FPN_PAN: 特征金字塔网络 + 路径聚合网络
- SE2Transformer: SE(2)群等变交叉注意力变换器
- ShapeBranch: 形状感知分支 (含SDF解码器)
- PMN_R3_FP: 完整的PuzzleMatchNet模型
- 相关损失函数和工具函数
"""

# 骨干网络
from .backbone import HRBackbone, SharedStem, BasicBlock, HighResolutionModule

# 特征金字塔网络
from .fpn_pan import FPN_PAN, FPN, PAN, ProposalGenerator

# SE(2)变换器
from .se2_transformer import (
    SE2Transformer, 
    SE2Attention, 
    SE2CrossAttention,
    GeometricRefinement,
    PositionalEncoding
)

# 形状感知模块
from .shape_sdf import (
    ShapeBranch,
    SDFDecoder,
    EdgePriorExtractor,
    ROIAlignExtractor,
    CoordConv2d
)

# 损失函数
from .loss import (
    FocalLoss,
    CenterNetLoss,
    SDFLoss,
    MatchingLoss,
    PMN_R3_FP_Loss
)

# 完整模型
from .pmn_r3_fp import (
    PMN_R3_FP,
    InputPreprocessor,
    create_model
)

__all__ = [
    # 骨干网络
    'HRBackbone',
    'SharedStem',
    'BasicBlock',
    'HighResolutionModule',
    
    # FPN-PAN
    'FPN_PAN',
    'FPN',
    'PAN',
    'ProposalGenerator',
    
    # SE(2)变换器
    'SE2Transformer',
    'SE2Attention',
    'SE2CrossAttention',
    'GeometricRefinement',
    'PositionalEncoding',
    
    # 形状感知模块
    'ShapeBranch',
    'SDFDecoder',
    'EdgePriorExtractor',
    'ROIAlignExtractor',
    'CoordConv2d',
    
    # 损失函数
    'FocalLoss',
    'CenterNetLoss',
    'SDFLoss',
    'MatchingLoss',
    'PMN_R3_FP_Loss',
    
    # 完整模型
    'PMN_R3_FP',
    'InputPreprocessor',
    'create_model'
]

# 模型版本信息
__version__ = '1.0.0'
__author__ = 'PMN-R3-FP Team'