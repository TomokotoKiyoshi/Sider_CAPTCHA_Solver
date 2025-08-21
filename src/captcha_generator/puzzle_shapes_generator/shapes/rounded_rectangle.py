# -*- coding: utf-8 -*-
"""
圆角矩形绘制模块

功能描述:
    提供圆角矩形的绘制功能，用于生成带有圆滑边角的矩形形状。
    
创建日期: 2025年8月
"""

import numpy as np
import cv2
from typing import Tuple


def draw_rounded_rectangle(canvas: np.ndarray, center: int, usable_size: int, 
                          fill_color: Tuple[int, int, int, int]) -> None:
    """
    绘制圆角矩形
    
    参数:
        canvas: 画布（RGBA格式的NumPy数组）
        center: 中心点坐标（正方形画布的中心）
        usable_size: 可用尺寸（矩形的大小）
        fill_color: 填充颜色（RGBA格式）
    """
    half_size = usable_size // 2
    corner_radius = int(usable_size * 0.15)  # 圆角半径为边长的15%
    
    # 创建圆角矩形的边界
    # 留出2像素边框供距离变换正确计算
    x1, y1 = center - half_size + 2, center - half_size + 2
    x2, y2 = center + half_size - 2, center + half_size - 2
    
    # 绘制主体矩形（去掉角落）
    # 水平矩形
    cv2.rectangle(canvas, 
                 (x1 + corner_radius, y1), 
                 (x2 - corner_radius + 1, y2 + 1), 
                 fill_color, -1)
    # 垂直矩形
    cv2.rectangle(canvas, 
                 (x1, y1 + corner_radius), 
                 (x2 + 1, y2 - corner_radius + 1), 
                 fill_color, -1)
    
    # 绘制四个圆角
    corners = [
        ((x1 + corner_radius, y1 + corner_radius), 180, 270),  # 左上角
        ((x2 - corner_radius + 1, y1 + corner_radius), 270, 360),  # 右上角
        ((x1 + corner_radius, y2 - corner_radius + 1), 90, 180),  # 左下角
        ((x2 - corner_radius + 1, y2 - corner_radius + 1), 0, 90),  # 右下角
    ]
    
    for center_pos, start_angle, end_angle in corners:
        cv2.ellipse(canvas, center_pos,
                   (corner_radius, corner_radius), 
                   0, start_angle, end_angle, 
                   fill_color, -1)