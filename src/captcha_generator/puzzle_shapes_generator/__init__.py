# -*- coding: utf-8 -*-
"""
拼图形状生成器模块

包含两种拼图形状生成器：
- create_common_puzzle_piece: 生成带有凸起/凹陷边缘的常规拼图
- create_special_puzzle_piece: 生成特殊形状（圆形、正方形、三角形、六边形）
"""

from .create_common_puzzle_piece import create_common_puzzle_piece
from .create_special_puzzle_piece import create_special_puzzle_piece

__all__ = [
    'create_common_puzzle_piece',
    'create_special_puzzle_piece',
]