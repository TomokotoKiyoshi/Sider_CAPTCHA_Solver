# -*- coding: utf-8 -*-
"""
直角矩形绘制模块

功能描述:
    提供直角矩形的绘制功能，用于生成无圆角的矩形形状。
    矩形会向内缩进2像素，确保边缘不会被裁剪。
    
创建日期: 2025年8月
"""

import numpy as np
import cv2
from typing import Tuple


def draw_rectangle(canvas: np.ndarray, center: int, usable_size: int, 
                  fill_color: Tuple[int, int, int, int], margin: int = 2) -> None:
    """
    绘制直角矩形（无圆角）
    
    参数:
        canvas: 画布（RGBA格式的NumPy数组）
        center: 中心点坐标（正方形画布的中心）
        usable_size: 可用尺寸（矩形的大小）
        fill_color: 填充颜色（RGBA格式）
        margin: 边缘内缩距离（像素），默认2像素，避免边缘被裁剪
    """
    # 计算半尺寸并应用内缩
    half_size = usable_size // 2 - margin
    
    # 确保矩形不会太小
    if half_size <= 0:
        half_size = 1
    
    # 计算矩形的四个顶点坐标
    x1 = center - half_size
    y1 = center - half_size
    x2 = center + half_size
    y2 = center + half_size
    
    # 绘制填充的矩形
    cv2.rectangle(canvas, (x1, y1), (x2, y2), fill_color, -1)