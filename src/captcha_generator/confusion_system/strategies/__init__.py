# -*- coding: utf-8 -*-
"""
混淆策略模块
"""
from .highlight import HighlightConfusion
from .gap_edge_highlight import GapEdgeHighlightConfusion
from .rotation import RotationConfusion
from .perlin_noise import PerlinNoiseConfusion
from .confusing_gap import ConfusingGapConfusion
from .hollow_center import HollowCenterConfusion
from .circular_confusing_gap import CircularConfusingGapConfusion

__all__ = [
    'HighlightConfusion',
    'GapEdgeHighlightConfusion',
    'RotationConfusion',
    'PerlinNoiseConfusion',
    'ConfusingGapConfusion',
    'HollowCenterConfusion',
    'CircularConfusingGapConfusion',
]