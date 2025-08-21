# -*- coding: utf-8 -*-
"""
文件名: create_common_puzzle_piece.py

功能概述:
    本模块提供 create_common_puzzle_piece 函数，用于根据指定的四边边缘类型
    （'flat'、'convex'、'concave'）生成具有相应凸起/凹陷结构的拼图块掩码图像。
    输出为灰度图转为 RGBA 图像（含透明通道），可用于拼图形状的掩膜生成。

主要功能:
    - 自动在拼图中心绘制实心正方形作为主体；
    - 根据参数在上下左右四边添加圆形“凸起”或“凹陷”；
    - 输出带透明背景的 RGBA 图像，白色区域表示拼图块形状。

输入参数（所有参数均为必须参数）:
    - piece_size (int): 整个拼图的边长（包括凸起部分）；
    - knob_radius_ratio (float): 凸起半径与中央正方形边长的比例；
    - edges (Tuple[str, str, str, str]): 四边边缘类型（顺序为：上、右、下、左），
      每个元素取值为 'flat'（平边）、'convex'（凸起）、'concave'（凹陷）之一。

返回值:
    - 一个 NumPy ndarray 对象，shape 为 (H, W, 4)，数据类型为 uint8，
      表示带透明通道的 RGBA 图像。

示例用法:
    >>> img = create_common_puzzle_piece(60, 0.25, ('convex', 'flat', 'concave', 'flat'))

创建日期: 2025年7月30日
"""
import cv2
import numpy as np

def create_common_puzzle_piece(piece_size: int, knob_radius_ratio: float, 
                       edges: tuple[str, str, str, str]):
    """
    创建拼图块掩码
    
    Args:
        piece_size (int): 整个拼图的边长（包括凸起部分）- 必须参数
        knob_radius_ratio (float): 凸起半径与中央正方形边长的比例 - 必须参数
        edges (tuple[str, str, str, str]): (上, 右, 下, 左) 每个值为 'flat', 'convex', 'concave' - 必须参数
    
    返回: 
        np.ndarray: RGBA图像 (uint8)，shape为(piece_size, piece_size, 4)
    """
    # 整个画布大小
    H = W = piece_size
    canvas = np.zeros((H, W), dtype=np.uint8)
    
    # 计算中央正方形的边长和knob_radius
    # 中央正方形需要为凸起预留空间
    # center_square_size + 2*knob_radius = piece_size
    # knob_radius = center_square_size * knob_radius_ratio
    # 所以：center_square_size = piece_size / (1 + 2*knob_radius_ratio)
    center_square_size = int(piece_size / (1 + 2 * knob_radius_ratio))
    knob_radius = int(center_square_size * knob_radius_ratio)
    
    # 计算中央正方形的起始位置（居中）
    square_offset = (piece_size - center_square_size) // 2
    
    # 绘制中央正方形
    cv2.rectangle(canvas,
                  (square_offset, square_offset),
                  (square_offset + center_square_size, square_offset + center_square_size),
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
    apply_knob(canvas, (piece_size // 2, square_offset), knob_radius, edges[0])
    # 右边
    apply_knob(canvas, (square_offset + center_square_size, piece_size // 2), knob_radius, edges[1])
    # 下边
    apply_knob(canvas, (piece_size // 2, square_offset + center_square_size), knob_radius, edges[2])
    # 左边
    apply_knob(canvas, (square_offset, piece_size // 2), knob_radius, edges[3])

    # 转换为RGBA
    rgba = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGBA)
    rgba[:, :, 3] = canvas  # alpha通道

    return rgba     # 返回RGBA格式图像: 拼图部分(255,255,255,255), 背景部分(0,0,0,0)
