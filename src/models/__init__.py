# -*- coding: utf-8 -*-
"""
模型模块
Lite-HRNet-18+LiteFPN 滑块验证码识别模型
"""

# 主模型接口
from .lite_hrnet_18_fpn import LiteHRNet18FPN, create_lite_hrnet_18_fpn

__all__ = [
    # 主模型类
    'LiteHRNet18FPN',
    
    # 模型创建函数
    'create_lite_hrnet_18_fpn',
]