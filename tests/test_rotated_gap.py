# -*- coding: utf-8 -*-
"""
测试旋转缺口的验证码生成
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import random
from src.captcha_generator.simple_puzzle_generator import create_puzzle_piece
from src.captcha_generator.lighting_effects import apply_gap_lighting
from src.captcha_generator.slider_effects import apply_slider_lighting, create_slider_frame, composite_slider

def rotate_image(image, angle, center=None):
    """
    旋转图像
    
    Args:
        image: 输入图像（可以是BGR或BGRA）
        angle: 旋转角度（度）
        center: 旋转中心，None表示使用图像中心
    
    Returns:
        旋转后的图像
    """
    h, w = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后的边界框
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵以适应新的图像大小
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # 执行旋转
    if image.shape[2] == 4:  # BGRA
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    else:  # BGR
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
    
    return rotated

def generate_rotated_gap_captchas(num_images=5):
    """
    生成带有旋转缺口的验证码
    
    Args:
        num_images: 生成图片数量
    """
    # 创建输出目录
    output_dir = Path(__file__).parent.parent / "test_output" / "rotated_gaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取背景图片
    img_path = Path(__file__).parent.parent / "data" / "raw" / "nature" / "nature_001.jpg"
    if not img_path.exists():
        # 如果找不到，创建测试图像
        img = np.ones((160, 320, 3), dtype=np.uint8) * 100
        # 添加一些纹理
        for _ in range(50):
            x = random.randint(0, 320)
            y = random.randint(0, 160)
            cv2.circle(img, (x, y), random.randint(5, 20), 
                      (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)), -1)
    else:
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (320, 160))
    
    # 旋转角度列表（0.5-2度）
    rotation_angles = [0, 0.5, 1.0, 1.5, 2.0, -1.0]  # 添加一个负角度
    
    # 创建预览画布 - 修正尺寸，添加间距
    rows = 2
    cols = 3
    img_width = 320
    img_height = 160
    padding = 20  # 图片之间的间距
    
    canvas_width = cols * img_width + (cols + 1) * padding
    canvas_height = rows * img_height + (rows + 1) * padding + 40  # 额外空间给标题
    preview_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 添加标题
    cv2.putText(preview_canvas, "Rotated Gap Test (0.5-2 degrees)", 
               (canvas_width // 2 - 150, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    print(f"Generating 6 rotated gap CAPTCHAs...")
    
    for i in range(num_images):
        # 创建拼图形状
        size = 60
        puzzle_mask = create_puzzle_piece(
            piece_size=size,
            knob_radius=int(size * 0.2),
            edges=('convex', 'concave', 'convex', 'concave')
        )
        
        mask_h, mask_w = puzzle_mask.shape[:2]
        alpha = puzzle_mask[:, :, 3]
        
        # 随机位置（确保不超出边界）
        x = random.randint(80, min(200, 320 - mask_w))
        y = random.randint(20, min(80, 160 - mask_h))
        
        # 使用预定的旋转角度
        rotation_angle = rotation_angles[i % len(rotation_angles)]
        
        # 创建工作副本
        background = img.copy()
        
        # 创建滑块（始终使用原始未旋转的形状）
        puzzle = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
        img_region = img[y:y+mask_h, x:x+mask_w]
        
        for c in range(3):
            puzzle[:, :, c] = img_region[:, :, c]
        puzzle[:, :, 3] = alpha
        
        if rotation_angle != 0:
            # 只旋转缺口的掩码，不旋转滑块
            rotated_mask = rotate_image(puzzle_mask, rotation_angle)
            rotated_h, rotated_w = rotated_mask.shape[:2]
            rotated_alpha = rotated_mask[:, :, 3]
            
            # 调整位置以确保旋转后的掩码在图像范围内
            cx_offset = (rotated_w - mask_w) // 2
            cy_offset = (rotated_h - mask_h) // 2
            
            # 调整x, y位置
            new_x = max(0, x - cx_offset)
            new_y = max(0, y - cy_offset)
            
            # 确保不超出背景边界
            if new_x + rotated_w > 320:
                new_x = 320 - rotated_w
            if new_y + rotated_h > 160:
                new_y = 160 - rotated_h
            
            # 应用旋转后的缺口光照效果
            background = apply_gap_lighting(background, new_x, new_y, rotated_alpha, rotated_h, rotated_w)
        else:
            # 不旋转的情况
            background = apply_gap_lighting(background, x, y, alpha, mask_h, mask_w)
        
        # 应用滑块光照
        puzzle = apply_slider_lighting(puzzle)
        
        # 滑块位置（始终与原始位置y对齐）
        slider_x = 20
        slider_y = y
        
        # 创建滑块边框
        slider_frame = create_slider_frame(60, 160)
        
        # 合成最终图像
        sd_center_x = slider_x + 30
        sd_center_y = slider_y + puzzle.shape[0] // 2
        
        final_image = composite_slider(background, puzzle, (sd_center_x, sd_center_y), slider_frame)
        
        # 保存图片
        filename = f"rotated_gap_{i+1:02d}_angle_{rotation_angle:.1f}.png"
        cv2.imwrite(str(output_dir / filename), final_image)
        
        # 添加到预览画布
        if i < 6:
            row = i // 3
            col = i % 3
            # 计算位置，包含间距
            y_offset = 40 + padding + row * (img_height + padding)
            x_offset = padding + col * (img_width + padding)
            preview_canvas[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = final_image
            
            # 添加角度标注
            cv2.putText(preview_canvas, f"Angle: {rotation_angle:.1f} deg", 
                       (x_offset + 10, y_offset + 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 保存预览图
    cv2.imwrite(str(output_dir / "rotated_gaps_preview.png"), preview_canvas)
    
    print(f"Generated {num_images} images")
    print(f"Output directory: {output_dir}")
    print(f"Preview saved as: rotated_gaps_preview.png")

if __name__ == "__main__":
    generate_rotated_gap_captchas(6)