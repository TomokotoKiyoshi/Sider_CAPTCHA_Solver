# -*- coding: utf-8 -*-
"""
随机多边形绘制模块

功能描述:
    提供随机多边形的绘制功能，支持生成5-12个顶点的多边形。
    生成的多边形保证是凸多边形，顶点均匀分布在圆周上。
    
创建日期: 2025年8月
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import random


def draw_polygon(canvas: np.ndarray, center: int, usable_size: int, 
                fill_color: Tuple[int, int, int, int], 
                min_vertices: int = 5, max_vertices: int = 12,
                angle_perturbation_divisor: float = 10,
                radius_min: float = 0.8, radius_max: float = 1.0) -> None:
    """
    绘制随机多边形
    
    参数:
        canvas: 画布（RGBA格式的NumPy数组）
        center: 中心点坐标（正方形画布的中心）
        usable_size: 可用尺寸（多边形的大小）
        fill_color: 填充颜色（RGBA格式）
        min_vertices: 最小顶点数（默认5）
        max_vertices: 最大顶点数（默认12）
        angle_perturbation_divisor: 角度扰动除数（越大越规则，默认10）
        radius_min: 最小半径比例（默认0.8）
        radius_max: 最大半径比例（默认1.0）
    """
    # 从配置的范围内随机选择顶点数
    num_vertices = random.randint(min_vertices, max_vertices)
    
    radius = usable_size // 2
    
    # 生成不规则但凸多边形的顶点
    # 策略1：在圆周上均匀分布角度，但添加小的随机扰动
    base_angles = np.linspace(0, 2 * np.pi, num_vertices + 1)[:-1]
    
    # 添加随机扰动，使多边形不那么规则
    # 扰动范围由配置参数控制
    angle_step = 2 * np.pi / num_vertices
    max_perturbation = angle_step / angle_perturbation_divisor
    
    angles = []
    for i, angle in enumerate(base_angles):
        # 添加随机扰动
        perturbation = random.uniform(-max_perturbation, max_perturbation)
        angles.append(angle + perturbation)
    
    # 确保角度是递增的（保持凸性）
    angles.sort()
    
    # 生成顶点坐标，半径也可以有轻微变化
    pts = []
    for angle in angles:
        # 半径变化范围由配置参数控制
        r_variation = random.uniform(radius_min, radius_max)
        r = radius * r_variation
        
        x = int(center + r * np.cos(angle))
        y = int(center + r * np.sin(angle))
        pts.append([x, y])
    
    pts = np.array(pts, np.int32)
    cv2.fillPoly(canvas, [pts], fill_color)


def draw_regular_polygon(canvas: np.ndarray, center: int, usable_size: int, 
                         fill_color: Tuple[int, int, int, int], 
                         num_vertices: int) -> None:
    """
    绘制正多边形（所有边和角都相等）
    
    参数:
        canvas: 画布（RGBA格式的NumPy数组）
        center: 中心点坐标（正方形画布的中心）
        usable_size: 可用尺寸（多边形的大小）
        fill_color: 填充颜色（RGBA格式）
        num_vertices: 顶点数量（5-12）
    """
    # 确保顶点数在合理范围内
    num_vertices = max(5, min(12, num_vertices))
    
    radius = usable_size // 2
    
    # 计算起始角度，使得多边形有一条边平行于底部（对于偶数边）
    # 或有一个顶点朝上（对于奇数边）
    if num_vertices % 2 == 0:
        start_angle = np.pi / num_vertices
    else:
        start_angle = np.pi / 2
    
    # 生成正多边形的顶点
    angles = np.linspace(0, 2 * np.pi, num_vertices + 1)[:-1] + start_angle
    
    pts = []
    for angle in angles:
        x = int(center + radius * np.cos(angle))
        y = int(center + radius * np.sin(angle))
        pts.append([x, y])
    
    pts = np.array(pts, np.int32)
    cv2.fillPoly(canvas, [pts], fill_color)