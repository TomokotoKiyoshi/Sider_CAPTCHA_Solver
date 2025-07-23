# -*- coding: utf-8 -*-
"""
滑块光照效果 - 使用白色蒙版实现
"""
import cv2
import numpy as np


def apply_slider_lighting(puzzle_piece, highlight_strength=0, edge_highlight=100, 
                         directional_highlight=30, edge_width=5, decay_factor=2.0):
    """
    为滑块添加3D凸起光照效果（使用白色蒙版）
    
    Args:
        puzzle_piece: 拼图块图像（RGBA）
        highlight_strength: 基础高光强度（未使用，保留用于兼容）
        edge_highlight: 边缘高光强度（映射到蒙版不透明度）
        directional_highlight: 方向性高光强度
        edge_width: 受光照影响的边缘宽度（像素）
        decay_factor: 衰减系数，控制从边缘到中心的透明度衰减速度
    
    Returns:
        应用光照效果后的滑块
    """
    h, w = puzzle_piece.shape[:2]
    result = puzzle_piece.copy()
    
    # 获取alpha通道
    alpha = puzzle_piece[:, :, 3]
    
    # 创建白色蒙版（初始透明）
    white_mask = np.zeros((h, w, 4), dtype=np.float32)
    white_mask[:, :, :3] = 255  # 白色
    
    # 1. 计算距离变换 - 找出每个像素到边缘的距离
    dist_transform = cv2.distanceTransform(alpha, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # 将edge_highlight映射到不透明度（0-100映射到0-0.8）
    max_opacity = min(edge_highlight / 100.0 * 0.8, 0.8)
    
    # 2. 创建边缘蒙版透明度（限制影响区域）
    # 边缘最不透明，向中心逐渐变透明
    edge_opacity = np.zeros((h, w), dtype=np.float32)
    
    if dist_transform.max() > 0:
        # 只在边缘宽度范围内应用效果
        edge_mask = dist_transform <= edge_width
        
        # 在边缘区域内，使用非线性衰减
        # 距离归一化到0-1（0是边缘，1是edge_width处）
        edge_distances = np.clip(dist_transform / edge_width, 0, 1)
        
        # 使用sigmoid函数实现平滑的非线性衰减
        # 在前30%几乎不衰减，之后快速衰减
        # sigmoid函数：1 / (1 + exp(-k*(x-x0)))
        # 这里我们使用一个变形的sigmoid来实现所需效果
        
        # 参数调整：
        # - shift控制转折点位置（0.5表示50%）
        # - steepness控制衰减的陡峭程度（降低以获得更平缓的衰减）
        shift = 0.5
        steepness = 4.0 * decay_factor
        
        # 使用tanh函数的变形，创造一个S型曲线
        # 先将距离映射到合适的范围
        x = (edge_distances - shift) * steepness
        
        # 使用1 - tanh来创建从1到0的衰减
        # tanh在[-3, 3]范围内变化最明显，所以我们调整steepness
        decay_curve = (1.0 - np.tanh(x)) * 0.5
        
        # 应用最大不透明度
        opacity_map = decay_curve * max_opacity
        
        # 应用到边缘区域
        edge_opacity[edge_mask] = opacity_map[edge_mask]
    
    # 3. 创建方向性光照（限制在边缘区域内）
    # 左上和右下更不透明
    directional_opacity = np.zeros((h, w), dtype=np.float32)
    directional_strength = directional_highlight / 100.0  # 转换到0-1范围
    
    if dist_transform.max() > 0 and directional_strength > 0:
        # 只在边缘宽度范围内应用方向性光照
        edge_mask = dist_transform <= edge_width
        
        # 创建坐标网格
        yy, xx = np.mgrid[0:h, 0:w]
        # 中心点
        center_x, center_y = w / 2, h / 2
        
        # 计算每个像素相对于中心的角度
        angles = np.arctan2(yy - center_y, xx - center_x)
        
        # 左上方向（约-135度）和右下方向（约45度）
        # 使用更尖锐的函数创建更明显的方向性
        # 将余弦函数的结果进行幂运算以增强对比度
        left_top_factor = np.maximum(0, np.cos(angles + 3*np.pi/4))
        right_bottom_factor = np.maximum(0, np.cos(angles - np.pi/4))
        
        # 使用幂函数增强方向性（值越大，方向性越明显）
        sharpness = 3.0
        left_top_factor = left_top_factor ** sharpness
        right_bottom_factor = right_bottom_factor ** sharpness
        
        # 组合两个方向的高光，并确保最大值为1
        directional_factor = np.minimum(1.0, left_top_factor + right_bottom_factor)
        
        # 在边缘区域内，根据距离衰减（使用相同的分段衰减）
        edge_distances = np.clip(dist_transform / edge_width, 0, 1)
        
        # 使用与边缘相同的tanh衰减函数
        shift = 0.5
        steepness = 4.0 * decay_factor
        
        # 计算衰减曲线
        x = (edge_distances - shift) * steepness
        decay_map = (1.0 - np.tanh(x)) * 0.5
        
        # 应用方向性效果
        directional_opacity[edge_mask] = (directional_factor[edge_mask] * 
                                         directional_strength * 
                                         decay_map[edge_mask] * 
                                         max_opacity)
    
    # 4. 组合边缘和方向性效果
    # 使用加权组合，让方向性效果更明显
    # 边缘光照作为基础，方向性光照作为额外增强
    total_opacity = edge_opacity + directional_opacity * 0.5
    # 确保不超过最大值
    total_opacity = np.minimum(total_opacity, max_opacity)
    
    # 5. 设置白色蒙版的alpha通道
    white_mask[:, :, 3] = total_opacity * 255
    
    # 6. 将白色蒙版叠加到原图上
    # 只在拼图形状内叠加
    puzzle_mask = alpha > 0
    
    for c in range(3):  # BGR通道
        # 计算叠加后的颜色
        # result = original * (1 - opacity) + white * opacity
        original = result[:, :, c].astype(np.float32)
        white = white_mask[:, :, c]
        opacity = total_opacity
        
        blended = original * (1 - opacity) + white * opacity
        result[:, :, c] = np.where(puzzle_mask, 
                                  np.clip(blended, 0, 255).astype(np.uint8),
                                  result[:, :, c])
    
    return result


def create_slider_frame(slider_width=60, slider_height=160, 
                       border_width=3, border_color=(200, 200, 200)):
    """
    创建滑块框架（透明背景）
    
    Args:
        slider_width: 滑块宽度
        slider_height: 滑块高度
        border_width: 边框宽度
        border_color: 边框颜色 (B, G, R)
    
    Returns:
        滑块框架图像 (RGBA)
    """
    # 创建透明画布
    frame = np.zeros((slider_height, slider_width, 4), dtype=np.uint8)
    
    # 创建边框掩码
    border_mask = np.zeros((slider_height, slider_width), dtype=np.uint8)
    
    # 绘制外边框（只有边框线条）
    cv2.rectangle(border_mask, 
                  (0, 0), 
                  (slider_width-1, slider_height-1),
                  255, 
                  border_width)
    
    # 设置边框颜色和透明度
    frame[:, :, :3] = border_color  # BGR颜色
    frame[:, :, 3] = border_mask     # Alpha通道（只有边框不透明）
    
    # 添加顶部滑动条效果
    cv2.rectangle(frame,
                  (border_width, border_width),
                  (slider_width-border_width-1, border_width+20),
                  (*border_color, 100),
                  -1)
    
    return frame


def composite_slider(background, puzzle_piece, slider_position, slider_frame=None):
    """
    将拼图块合成到滑块中
    
    Args:
        background: 背景图像
        puzzle_piece: 带光照效果的拼图块
        slider_position: 滑块位置 (x, y) - 中心坐标
        slider_frame: 滑块框架（可选）
    
    Returns:
        合成后的图像
    """
    result = background.copy()
    piece_h, piece_w = puzzle_piece.shape[:2]
    
    # 计算拼图块在滑块中的位置（居中）
    slider_x, slider_y = slider_position
    
    # 如果有滑块框架，先绘制框架
    if slider_frame is not None:
        frame_h, frame_w = slider_frame.shape[:2]
        frame_x = slider_x - frame_w // 2
        frame_y = slider_y - frame_h // 2
        
        # 确保框架在图像范围内
        if 0 <= frame_x < result.shape[1] - frame_w and 0 <= frame_y < result.shape[0] - frame_h:
            # 混合框架
            roi = result[frame_y:frame_y+frame_h, frame_x:frame_x+frame_w]
            alpha = slider_frame[:, :, 3] / 255.0
            
            for c in range(3):
                roi[:, :, c] = (slider_frame[:, :, c] * alpha + 
                               roi[:, :, c] * (1 - alpha)).astype(np.uint8)
    
    # 计算拼图块位置（在滑块中心）
    piece_x = slider_x - piece_w // 2
    piece_y = slider_y - piece_h // 2
    
    # 计算实际可绘制的区域（处理超出边界的情况）
    # 源区域（拼图块）
    src_x1 = max(0, -piece_x)
    src_y1 = max(0, -piece_y)
    src_x2 = min(piece_w, result.shape[1] - piece_x)
    src_y2 = min(piece_h, result.shape[0] - piece_y)
    
    # 目标区域（背景）
    dst_x1 = max(0, piece_x)
    dst_y1 = max(0, piece_y)
    dst_x2 = min(result.shape[1], piece_x + piece_w)
    dst_y2 = min(result.shape[0], piece_y + piece_h)
    
    # 如果有有效的区域可以绘制
    if src_x2 > src_x1 and src_y2 > src_y1:
        # 获取源和目标区域
        src_region = puzzle_piece[src_y1:src_y2, src_x1:src_x2]
        dst_region = result[dst_y1:dst_y2, dst_x1:dst_x2]
        
        # 混合拼图块
        alpha = src_region[:, :, 3:4] / 255.0
        
        for c in range(3):
            dst_region[:, :, c] = (src_region[:, :, c] * alpha[:, :, 0] + 
                                  dst_region[:, :, c] * (1 - alpha[:, :, 0])).astype(np.uint8)
    
    return result