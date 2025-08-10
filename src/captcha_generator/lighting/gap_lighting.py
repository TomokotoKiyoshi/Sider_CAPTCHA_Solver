# -*- coding: utf-8 -*-
"""
拼图缺口光照效果
"""
import cv2
import numpy as np


def apply_gap_lighting(background, x, y, mask_alpha, mask_h, mask_w,
                      base_darkness=40, edge_darkness=50, directional_darkness=20,
                      outer_edge_darkness=0):
    """
    为拼图缺口应用3D光照效果
    
    Args:
        background: 背景图片
        x, y: 缺口位置
        mask_alpha: 拼图掩码的alpha通道
        mask_h, mask_w: 掩码尺寸
        base_darkness: 基础暗度（整个缺口区域的基础变暗程度）
        edge_darkness: 边缘暗度（边缘额外的变暗程度）
        directional_darkness: 方向性暗度（左上和右下的额外变暗程度）
        outer_edge_darkness: 外边缘暗度（缺口外围的阴影强度）
    
    Returns:
        应用光照效果后的背景
    """
    # 边界检查
    bg_h, bg_w = background.shape[:2]
    if x < 0:
        raise ValueError(f"缺口x坐标不能为负: x={x}")
    if y < 0:
        raise ValueError(f"缺口y坐标不能为负: y={y}")
    if x + mask_w > bg_w:
        raise ValueError(f"缺口超出右边界: x={x}, mask_w={mask_w}, bg_w={bg_w}")
    if y + mask_h > bg_h:
        raise ValueError(f"缺口超出下边界: y={y}, mask_h={mask_h}, bg_h={bg_h}")
    if mask_alpha.shape != (mask_h, mask_w):
        raise ValueError(f"mask_alpha形状不匹配: expected ({mask_h}, {mask_w}), got {mask_alpha.shape}")
    
    # 创建工作副本
    result = background.copy()
    
    # 保存原始缺口区域的内容
    gap_region = result[y:y+mask_h, x:x+mask_w].copy()
    
    # 1. 计算距离变换 - 找出每个像素到边缘的距离
    # 注意：不能直接对带hollow的mask使用distanceTransform
    # 需要先创建一个二值mask，只标记实体部分
    solid_mask = (mask_alpha > 128).astype(np.uint8) * 255
    dist_transform = cv2.distanceTransform(solid_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # 2. 创建基础阴影层
    # 整个缺口区域都比背景暗
    base_shadow = np.full((mask_h, mask_w), base_darkness, dtype=np.float32)
    
    # 3. 创建边缘阴影渐变
    # 边缘最暗，向中心逐渐变亮
    if dist_transform.max() > 0:
        # 归一化距离（0-1）
        norm_dist = dist_transform / dist_transform.max()
        # 边缘阴影强度：边缘(0)最暗，中心(1)最亮
        edge_shadow = (1.0 - norm_dist) * edge_darkness
    else:
        edge_shadow = np.zeros((mask_h, mask_w), dtype=np.float32)
    
    # 4. 创建方向性光照
    # 左上和右下更暗（模拟光线被遮挡）
    directional_shadow = np.zeros((mask_h, mask_w), dtype=np.float32)
    
    # 创建坐标网格
    yy, xx = np.mgrid[0:mask_h, 0:mask_w]
    # 中心点
    center_x, center_y = mask_w / 2, mask_h / 2
    
    # 计算每个像素相对于中心的角度
    angles = np.arctan2(yy - center_y, xx - center_x)
    
    # 左上方向（约-135度）和右下方向（约45度）
    # 使用余弦函数创建平滑过渡
    left_top_factor = np.cos(angles + 3*np.pi/4) * 0.5 + 0.5  # 0到1
    right_bottom_factor = np.cos(angles - np.pi/4) * 0.5 + 0.5  # 0到1
    
    # 组合两个方向的阴影
    directional_factor = np.maximum(left_top_factor, right_bottom_factor)
    directional_shadow = directional_factor * directional_darkness  # 方向性阴影强度
    
    # 5. 组合所有阴影效果
    total_shadow = base_shadow + edge_shadow + directional_shadow
    
    # 6. 在缺口区域应用阴影效果并真正挖空
    # 重要：只在alpha > 0的区域创建缺口，保留hollow效果
    # 先保存原始背景区域
    original_region = result[y:y+mask_h, x:x+mask_w].copy()
    
    # 创建一个阴影层
    shadow_layer = np.zeros((mask_h, mask_w, 3), dtype=np.float32)
    for c in range(3):
        shadow_layer[:, :, c] = -total_shadow
    
    # 对于缺口实体部分（alpha > 128）：应用强阴影模拟深度
    # 对于缺口边缘（0 < alpha <= 128）：应用渐变阴影
    # 对于hollow区域（alpha = 0）：保持原背景
    for c in range(3):
        channel = result[y:y+mask_h, x:x+mask_w, c].astype(np.float32)
        
        # 根据alpha值混合：
        # - alpha=255: 完全应用阴影（模拟深缺口）
        # - alpha=0 (hollow): 保持原背景
        # - 中间值: 渐变过渡
        mask_factor = mask_alpha / 255.0
        
        # 应用阴影创建深度效果
        result[y:y+mask_h, x:x+mask_w, c] = np.clip(
            channel + shadow_layer[:, :, c] * mask_factor, 
            0, 255
        ).astype(np.uint8)
    
    # 8. 在缺口边缘外添加微弱的阴影（增强3D效果）
    # 只有当outer_edge_darkness > 0时才处理
    if outer_edge_darkness > 0:
        # 创建稍大的掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_alpha = cv2.dilate(mask_alpha, kernel, iterations=2)
        outer_edge = cv2.bitwise_xor(dilated_alpha, mask_alpha)
        
        # 外边缘阴影
        for c in range(3):
            channel = result[y:y+mask_h, x:x+mask_w, c].astype(np.float32)
            edge_factor = outer_edge / 255.0
            darkened = channel - (outer_edge_darkness * edge_factor)
            result[y:y+mask_h, x:x+mask_w, c] = np.clip(darkened, 0, 255).astype(np.uint8)
    
    # 9. 对缺口区域进行轻微模糊，模拟景深效果
    # 只对实体部分（非hollow区域）应用模糊
    solid_region = mask_alpha > 128  # 使用更高的阈值，排除渐变边缘
    blurred_result = cv2.GaussianBlur(result[y:y+mask_h, x:x+mask_w], (3, 3), 0.8)
    
    # 只在实体缺口内应用轻微模糊，保留hollow区域的背景
    for c in range(3):
        result[y:y+mask_h, x:x+mask_w, c] = np.where(
            solid_region,
            blurred_result[:, :, c],
            result[y:y+mask_h, x:x+mask_w, c]
        )
    
    # 不要挖空！保留缺口内容，只是变暗了
    return result



def apply_gap_highlighting(background, x, y, mask_alpha, mask_h, mask_w,
                          base_lightness=40, edge_lightness=60, directional_lightness=30,
                          outer_edge_lightness=20):
    """
    为拼图缺口应用3D高光效果（变亮而非变暗）
    
    Args:
        background: 背景图片
        x, y: 缺口位置
        mask_alpha: 拼图掩码的alpha通道
        mask_h, mask_w: 掩码尺寸
        base_lightness: 基础亮度（整个缺口区域的基础变亮程度）
        edge_lightness: 边缘亮度（边缘额外的变亮程度）
        directional_lightness: 方向性亮度（右下和左上的额外变亮程度）
        outer_edge_lightness: 外边缘亮度（缺口外围的高光强度）
    
    Returns:
        应用高光效果后的背景
    """
    # 边界检查
    bg_h, bg_w = background.shape[:2]
    if x < 0:
        raise ValueError(f"缺口x坐标不能为负: x={x}")
    if y < 0:
        raise ValueError(f"缺口y坐标不能为负: y={y}")
    if x + mask_w > bg_w:
        raise ValueError(f"缺口超出右边界: x={x}, mask_w={mask_w}, bg_w={bg_w}")
    if y + mask_h > bg_h:
        raise ValueError(f"缺口超出下边界: y={y}, mask_h={mask_h}, bg_h={bg_h}")
    if mask_alpha.shape != (mask_h, mask_w):
        raise ValueError(f"mask_alpha形状不匹配: expected ({mask_h}, {mask_w}), got {mask_alpha.shape}")
    
    # 创建工作副本
    result = background.copy()
    
    # 保存原始缺口区域的内容
    gap_region = result[y:y+mask_h, x:x+mask_w].copy()
    
    # 1. 计算距离变换 - 找出每个像素到边缘的距离
    # 注意：不能直接对带hollow的mask使用distanceTransform
    # 需要先创建一个二值mask，只标记实体部分
    solid_mask = (mask_alpha > 128).astype(np.uint8) * 255
    dist_transform = cv2.distanceTransform(solid_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # 2. 创建基础高光层
    # 整个缺口区域都比背景亮
    base_highlight = np.full((mask_h, mask_w), base_lightness, dtype=np.float32)
    
    # 3. 创建边缘高光渐变
    # 边缘最亮，向中心逐渐变暗
    if dist_transform.max() > 0:
        # 归一化距离（0-1）
        norm_dist = dist_transform / dist_transform.max()
        # 边缘高光强度：边缘(0)最亮，中心(1)最暗
        edge_highlight = (1.0 - norm_dist) * edge_lightness
    else:
        edge_highlight = np.zeros((mask_h, mask_w), dtype=np.float32)
    
    # 4. 创建方向性光照
    # 右下和左上更亮（模拟光线照射）
    directional_highlight = np.zeros((mask_h, mask_w), dtype=np.float32)
    
    # 创建坐标网格
    yy, xx = np.mgrid[0:mask_h, 0:mask_w]
    # 中心点
    center_x, center_y = mask_w / 2, mask_h / 2
    
    # 计算每个像素相对于中心的角度
    angles = np.arctan2(yy - center_y, xx - center_x)
    
    # 右下方向（约45度）和左上方向（约-135度）- 与原来相反
    # 使用余弦函数创建平滑过渡
    right_bottom_factor = np.cos(angles - np.pi/4) * 0.5 + 0.5  # 0到1
    left_top_factor = np.cos(angles + 3*np.pi/4) * 0.5 + 0.5  # 0到1
    
    # 组合两个方向的高光
    directional_factor = np.maximum(right_bottom_factor, left_top_factor)
    directional_highlight = directional_factor * directional_lightness
    
    # 5. 组合所有高光效果
    total_highlight = base_highlight + edge_highlight + directional_highlight
    
    # 6. 在缺口区域应用高光效果并保持hollow透明
    # 重要：只在alpha > 0的区域应用高光，保留hollow效果
    # 创建高光层
    highlight_layer = np.zeros((mask_h, mask_w, 3), dtype=np.float32)
    for c in range(3):
        highlight_layer[:, :, c] = total_highlight
    
    # 应用高光效果
    for c in range(3):
        channel = result[y:y+mask_h, x:x+mask_w, c].astype(np.float32)
        
        # 根据alpha值混合：
        # - alpha=255: 完全应用高光（缺口边缘变亮）
        # - alpha=0 (hollow): 保持原背景
        # - 中间值: 渐变过渡
        mask_factor = mask_alpha / 255.0
        
        # 应用高光创建3D效果
        result[y:y+mask_h, x:x+mask_w, c] = np.clip(
            channel + highlight_layer[:, :, c] * mask_factor,
            0, 255
        ).astype(np.uint8)
    
    # 8. 在缺口边缘外添加微弱的高光（增强3D效果）
    if outer_edge_lightness > 0:
        # 创建稍大的掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_alpha = cv2.dilate(mask_alpha, kernel, iterations=2)
        outer_edge = cv2.bitwise_xor(dilated_alpha, mask_alpha)
        
        # 外边缘高光
        for c in range(3):
            channel = result[y:y+mask_h, x:x+mask_w, c].astype(np.float32)
            edge_factor = outer_edge / 255.0
            lightened = channel + (outer_edge_lightness * edge_factor)
            result[y:y+mask_h, x:x+mask_w, c] = np.clip(lightened, 0, 255).astype(np.uint8)
    
    # 9. 对缺口区域进行轻微模糊，模拟景深效果
    # 只对实体部分（非hollow区域）应用模糊
    solid_region = mask_alpha > 128
    blurred_result = cv2.GaussianBlur(result[y:y+mask_h, x:x+mask_w], (3, 3), 0.8)
    
    # 只在实体缺口内应用轻微模糊，保留hollow区域的背景
    for c in range(3):
        result[y:y+mask_h, x:x+mask_w, c] = np.where(
            solid_region,
            blurred_result[:, :, c],
            result[y:y+mask_h, x:x+mask_w, c]
        )
    
    return result


def add_puzzle_piece_shadow(puzzle, shadow_offset=(2, 2), shadow_opacity=0.3):
    """
    为拼图块添加阴影效果
    
    Args:
        puzzle: 拼图块图像（RGBA）
        shadow_offset: 阴影偏移(x, y)
        shadow_opacity: 阴影透明度
    
    Returns:
        带阴影的拼图块
    """
    h, w = puzzle.shape[:2]
    offset_x, offset_y = shadow_offset
    
    # 创建更大的画布容纳阴影
    canvas_h = h + abs(offset_y) + 5
    canvas_w = w + abs(offset_x) + 5
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    
    # 计算拼图位置
    puzzle_x = max(0, -offset_x) + 2
    puzzle_y = max(0, -offset_y) + 2
    
    # 创建阴影
    shadow = np.zeros((h, w, 4), dtype=np.uint8)
    shadow[:, :, 3] = puzzle[:, :, 3]  # 复制alpha通道
    shadow[:, :, :3] = 0  # 黑色阴影
    
    # 模糊阴影
    shadow[:, :, 3] = cv2.GaussianBlur(shadow[:, :, 3], (5, 5), 2)
    shadow[:, :, 3] = (shadow[:, :, 3] * shadow_opacity).astype(np.uint8)
    
    # 放置阴影
    shadow_x = puzzle_x + offset_x
    shadow_y = puzzle_y + offset_y
    canvas[shadow_y:shadow_y+h, shadow_x:shadow_x+w] = shadow
    
    # 放置拼图块
    # 需要正确混合alpha通道
    roi = canvas[puzzle_y:puzzle_y+h, puzzle_x:puzzle_x+w]
    alpha = puzzle[:, :, 3] / 255.0
    
    for c in range(3):
        roi[:, :, c] = (puzzle[:, :, c] * alpha + 
                       roi[:, :, c] * (1 - alpha)).astype(np.uint8)
    roi[:, :, 3] = np.maximum(roi[:, :, 3], puzzle[:, :, 3])
    
    canvas[puzzle_y:puzzle_y+h, puzzle_x:puzzle_x+w] = roi
    
    return canvas