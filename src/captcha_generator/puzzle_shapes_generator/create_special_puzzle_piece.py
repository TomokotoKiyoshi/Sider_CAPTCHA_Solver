# -*- coding: utf-8 -*-
"""
文件名: create_special_puzzle_piece.py

功能描述:
    本模块提供 create_special_puzzle_piece 函数，用于根据指定的形状类型生成带有透明背景的白色图案，
    图案绘制在一个 RGBA 格式的 NumPy 数组上，可用于拼图块生成、图形遮罩、图标渲染、图形识别训练等用途。

支持的形状类型包括:
    - 'circle'      : 圆形
    - 'rectangle'   : 直角矩形（无圆角）
    - 'rounded_rect': 圆角矩形
    - 'triangle'    : 等腰三角形（顶点朝上）
    - 'polygon'     : 随机多边形（顶点数从配置读取）

使用方法示例:
    >>> img = create_special_puzzle_piece('circle', size=80)
    >>> img = create_special_puzzle_piece('rectangle', size=80)
    >>> img = create_special_puzzle_piece('rounded_rect', size=80)
    >>> img = create_special_puzzle_piece('polygon', size=80)  # 随机多边形

创建日期: 2025年7月30日
更新日期: 2025年8月 - 添加直角矩形支持并规范化命名
"""

import numpy as np
import cv2
from typing import Tuple, List
from .shapes import draw_rectangle, draw_rounded_rectangle, draw_triangle, draw_polygon
from ...config.dataset_config import get_dataset_config

def create_special_puzzle_piece(shape_type: str, size: int = 60) -> np.ndarray:
    """
    根据指定的形状类型生成一个带透明背景的白色图案
    
    参数:
        shape_type (str): 形状类型，可选值包括:
                         'circle', 'rectangle', 'rounded_rect', 'triangle', 'polygon'
        size (int): 图像大小（正方形边长，单位像素），默认为60

    返回:
        canvas (np.ndarray): 生成的 RGBA 图像（带透明通道的白色图形）
    
    异常:
        ValueError: 当提供不支持的形状类型时
    """
    # 参数验证
    supported_shapes = ['circle', 'rectangle', 'rounded_rect', 'triangle', 'polygon']
    if shape_type not in supported_shapes:
        raise ValueError(f"Unsupported shape type: '{shape_type}'. Supported types: {supported_shapes}")
    
    # 创建透明画布
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    
    # 留出1像素边距，确保形状不会被裁剪
    margin = 1
    usable_size = size - 2 * margin
    
    # 定义填充颜色（白色，完全不透明）
    fill_color = (255, 255, 255, 255)
    
    if shape_type == 'circle':
        # 圆形：直径等于可用尺寸
        radius = usable_size // 2
        cv2.circle(canvas, (center, center), radius, fill_color, -1)
    
    elif shape_type == 'rectangle':
        # 直角矩形：无圆角，内缩2像素避免边缘裁剪
        draw_rectangle(canvas, center, usable_size, fill_color)
        
    elif shape_type == 'rounded_rect':
        # 圆角矩形
        draw_rounded_rectangle(canvas, center, usable_size, fill_color)
        
    elif shape_type == 'triangle':
        # 等边三角形
        draw_triangle(canvas, center, usable_size, size, margin, fill_color)
        
    elif shape_type == 'polygon':
        # 随机多边形 (从配置读取所有参数)
        config = get_dataset_config()
        draw_polygon(canvas, center, usable_size, fill_color, 
                    min_vertices=config.POLYGON_MIN_VERTICES,
                    max_vertices=config.POLYGON_MAX_VERTICES,
                    angle_perturbation_divisor=config.POLYGON_ANGLE_PERTURBATION_DIVISOR,
                    radius_min=config.POLYGON_RADIUS_MIN,
                    radius_max=config.POLYGON_RADIUS_MAX)
        
    return canvas


