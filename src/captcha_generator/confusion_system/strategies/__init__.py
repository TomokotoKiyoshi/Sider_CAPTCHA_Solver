# -*- coding: utf-8 -*-
"""
混淆策略模块
"""
from .highlight import HighlightConfusion
from .rotation import RotationConfusion
from .perlin_noise import PerlinNoiseConfusion
from .confusing_gap import ConfusingGapConfusion
from .hollow_center import HollowCenterConfusion

__all__ = [
    'HighlightConfusion',
    'RotationConfusion',
    'PerlinNoiseConfusion',
    'ConfusingGapConfusion',
    'HollowCenterConfusion',
]