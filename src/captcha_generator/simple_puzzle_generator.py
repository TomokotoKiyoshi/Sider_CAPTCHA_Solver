# -*- coding: utf-8 -*-
"""
简洁的拼图形状生成器
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import itertools


def create_puzzle_piece(piece_size=60, knob_radius=12,
                       edges=('flat', 'convex', 'flat', 'concave')):
    """
    创建拼图块掩码
    edges: (上, 右, 下, 左) 每个值为 'flat', 'convex', 'concave'
    返回: RGBA图像 (uint8)
    """
    h = w = piece_size
    # 画布留出凸起的空间
    pad = knob_radius + 2
    H = h + 2 * pad
    W = w + 2 * pad
    canvas = np.zeros((H, W), dtype=np.uint8)

    # 绘制中央正方形
    cv2.rectangle(canvas,
                  (pad, pad),
                  (pad + w, pad + h),
                  255,
                  thickness=-1)

    # 辅助函数：绘制凸起或凹陷
    def apply_knob(mask, center, r, mode):
        if mode == 'flat':
            return
        knob = np.zeros_like(mask)
        cv2.circle(knob, center, r, 255, thickness=-1)
        if mode == 'convex':
            # 添加凸起
            mask[:] = cv2.bitwise_or(mask, knob)
        elif mode == 'concave':
            # 减去凹陷
            mask[:] = cv2.bitwise_and(mask, cv2.bitwise_not(knob))

    # 上边
    apply_knob(canvas, (pad + w // 2, pad), knob_radius, edges[0])
    # 右边
    apply_knob(canvas, (pad + w, pad + h // 2), knob_radius, edges[1])
    # 下边
    apply_knob(canvas, (pad + w // 2, pad + h), knob_radius, edges[2])
    # 左边
    apply_knob(canvas, (pad, pad + h // 2), knob_radius, edges[3])

    # 转换为RGBA
    rgba = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGBA)
    rgba[:, :, 3] = canvas  # alpha通道

    return rgba


def edges_to_numeric(edges: Tuple[str, str, str, str]) -> Tuple[int, int, int, int]:
    """将边缘类型转换为数字表示"""
    mapping = {'concave': 0, 'flat': 1, 'convex': 2}
    return tuple(mapping[e] for e in edges)


def numeric_to_edges(nums: Tuple[int, int, int, int]) -> Tuple[str, str, str, str]:
    """将数字表示转换为边缘类型"""
    mapping = {0: 'concave', 1: 'flat', 2: 'convex'}
    return tuple(mapping[n] for n in nums)


def get_all_puzzle_combinations() -> List[Tuple[str, str, str, str]]:
    """获取所有81种拼图组合"""
    edge_types = ['concave', 'flat', 'convex']
    return list(itertools.product(edge_types, repeat=4))


def generate_special_shape(shape_type: str, size: int = 60) -> np.ndarray:
    """生成特殊形状"""
    # 创建透明画布
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    
    if shape_type == 'circle':
        cv2.circle(canvas, (center, center), int(size * 0.4), (255, 255, 255, 255), -1)
        
    elif shape_type == 'square':
        s = int(size * 0.6)
        top_left = (center - s // 2, center - s // 2)
        bottom_right = (center + s // 2, center + s // 2)
        cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255, 255), -1)
        
    elif shape_type == 'triangle':
        pts = np.array([
            [center, int(center - size * 0.35)],
            [int(center - size * 0.3), int(center + size * 0.25)],
            [int(center + size * 0.3), int(center + size * 0.25)]
        ], np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    elif shape_type == 'hexagon':
        angles = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
        radius = size * 0.35
        pts = []
        for angle in angles[:-1]:
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    elif shape_type == 'pentagon':
        angles = np.linspace(0, 2 * np.pi, 6) + np.pi / 2
        radius = size * 0.35
        pts = []
        for angle in angles[:-1]:
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    elif shape_type == 'star':
        # 五角星
        outer_radius = size * 0.35
        inner_radius = size * 0.15
        pts = []
        for i in range(10):
            angle = i * np.pi / 5 - np.pi / 2
            if i % 2 == 0:
                radius = outer_radius
            else:
                radius = inner_radius
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
    
    return canvas