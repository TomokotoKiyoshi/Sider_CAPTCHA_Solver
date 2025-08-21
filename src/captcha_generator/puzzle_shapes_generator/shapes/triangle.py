# -*- coding: utf-8 -*-
"""
三角形绘制模块

功能描述:
    提供等边三角形的绘制功能，支持生成随机旋转的三角形形状。
    三角形可以随机旋转0°、90°、180°、270°，实现四个方向的朝向。
    
创建日期: 2025年8月
更新日期: 2025年8月 - 添加随机旋转功能
"""

import numpy as np
import cv2
from typing import Tuple
import random


def draw_triangle(canvas: np.ndarray, center: int, usable_size: int, 
                 size: int, margin: int, fill_color: Tuple[int, int, int, int]) -> None:
    """
    绘制等边三角形（支持随机旋转）
    
    参数:
        canvas: 画布（RGBA格式的NumPy数组）
        center: 中心点坐标（正方形画布的中心）
        usable_size: 可用尺寸（三角形的大小）
        size: 画布大小（像素）
        margin: 边距（像素）
        fill_color: 填充颜色（RGBA格式）
    """
    # 随机选择旋转角度：0°（朝上）、90°（朝右）、180°（朝下）、270°（朝左）
    rotation_angle = random.choice([0, 90, 180, 270])
    
    # 高度 = sqrt(3)/2 * base ≈ 0.866 * base
    height = int(usable_size * 0.866)
    # 调整三角形使其在画布中居中
    y_offset = (usable_size - height) // 2
    
    # 基础三角形顶点（0°朝上）
    if rotation_angle == 0:
        # 顶点朝上
        pts = np.array([
            [center, margin + y_offset],  # 顶点
            [margin, size - margin - y_offset],  # 左下角
            [size - margin, size - margin - y_offset]  # 右下角
        ], np.int32)
    
    elif rotation_angle == 90:
        # 顶点朝右
        pts = np.array([
            [size - margin - y_offset, center],  # 右侧顶点
            [margin + y_offset, margin],  # 左上角
            [margin + y_offset, size - margin]  # 左下角
        ], np.int32)
    
    elif rotation_angle == 180:
        # 顶点朝下
        pts = np.array([
            [center, size - margin - y_offset],  # 底部顶点
            [size - margin, margin + y_offset],  # 右上角
            [margin, margin + y_offset]  # 左上角
        ], np.int32)
    
    elif rotation_angle == 270:
        # 顶点朝左
        pts = np.array([
            [margin + y_offset, center],  # 左侧顶点
            [size - margin - y_offset, size - margin],  # 右下角
            [size - margin - y_offset, margin]  # 右上角
        ], np.int32)
    
    cv2.fillPoly(canvas, [pts], fill_color)