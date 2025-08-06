# -*- coding: utf-8 -*-
"""
文件名: create_special_puzzle_piece.py

功能描述:
    本模块提供 create_special_puzzle_piece 函数，用于根据指定的形状类型生成带有透明背景的白色图案，
    图案绘制在一个 RGBA 格式的 NumPy 数组上，可用于拼图块生成、图形遮罩、图标渲染、图形识别训练等用途。

支持的形状类型包括:
    - 'circle'   ：圆形
    - 'square'   ：正方形
    - 'triangle' ：等腰三角形（顶点朝上）
    - 'hexagon'  ：正六边形

使用方法示例:
    >>> img = create_special_puzzle_piece('circle', size=80)

创建日期: 2025年7月30日
"""


import numpy as np
import cv2

def create_special_puzzle_piece(shape_type: str, size: int = 60) -> np.ndarray:
    """
    根据指定的形状类型生成一个带透明背景的白色图案
    
    参数:
        shape_type (str): 形状类型，可选值包括 'circle', 'square', 'triangle', 'hexagon'
        size (int): 图像大小（正方形边长，单位像素），默认为60

    返回:
        canvas (np.ndarray): 生成的 RGBA 图像（带透明通道的白色图形）
    """
    # 创建透明画布
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    
    # 留出1像素边距，确保形状不会被裁剪
    margin = 1
    usable_size = size - 2 * margin
    
    if shape_type == 'circle':
        # 圆形：直径等于可用尺寸
        radius = usable_size // 2
        cv2.circle(canvas, (center, center), radius, (255, 255, 255, 255), -1)
        
    elif shape_type == 'square':
        # 正方形：边长等于可用尺寸
        half_size = usable_size // 2
        top_left = (center - half_size, center - half_size)
        bottom_right = (center + half_size, center + half_size)
        cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255, 255), -1)
        
    elif shape_type == 'triangle':
        # 等边三角形：底边等于可用尺寸，高度为 sqrt(3)/2 * 底边
        half_base = usable_size // 2
        # 高度 = sqrt(3)/2 * base ≈ 0.866 * base
        height = int(usable_size * 0.866)
        # 调整三角形使其在画布中居中
        y_offset = (usable_size - height) // 2
        
        pts = np.array([
            [center, margin + y_offset],  # 顶点
            [margin, size - margin - y_offset],  # 左下角
            [size - margin, size - margin - y_offset]  # 右下角
        ], np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    elif shape_type == 'hexagon':
        # 正六边形：使其内切于正方形
        # 对于正六边形，如果要内切于正方形，半径 = 边长 / 2
        radius = usable_size // 2
        angles = np.linspace(0, 2 * np.pi, 7) + np.pi / 6  # 旋转30度使平边朝上
        pts = []
        for angle in angles[:-1]:
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    return canvas