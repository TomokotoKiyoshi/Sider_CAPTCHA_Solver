# -*- coding: utf-8 -*-
"""
光照效果模块

包含两种光照效果处理：
- gap_lighting: 缺口光照效果（阴影和高光）
- slider_lighting: 滑块光照效果（3D凸起效果）
"""

from .gap_lighting import apply_gap_lighting, apply_gap_highlighting
from .slider_lighting import apply_slider_lighting, create_slider_frame, composite_slider

__all__ = [
    # 缺口光照效果
    'apply_gap_lighting',
    'apply_gap_highlighting',
    
    # 滑块光照效果
    'apply_slider_lighting',
    'create_slider_frame',
    'composite_slider',
]