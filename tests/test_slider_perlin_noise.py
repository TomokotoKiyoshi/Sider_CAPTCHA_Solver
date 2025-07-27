# -*- coding: utf-8 -*-
"""
测试滑块柏林噪声效果
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

def generate_perlin_noise(shape, scale=100, octaves=4, persistence=0.5, lacunarity=2.0):
    """
    生成柏林噪声
    
    Args:
        shape: 输出形状 (height, width)
        scale: 噪声的缩放因子
        octaves: 噪声的层数
        persistence: 每一层的振幅衰减
        lacunarity: 每一层的频率增长
    
    Returns:
        柏林噪声图 (0-1范围)
    """
    height, width = shape
    noise = np.zeros(shape)
    
    # 生成基础随机噪声
    base_noise = np.random.rand(height // scale + 2, width // scale + 2)
    
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    for i in range(octaves):
        # 创建当前层的采样点
        y_coords = np.linspace(0, height / scale * frequency, height)
        x_coords = np.linspace(0, width / scale * frequency, width)
        
        # 使用双线性插值来平滑噪声
        layer = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                # 计算在base_noise中的位置
                y_pos = y_coords[y]
                x_pos = x_coords[x]
                
                # 获取周围的四个点
                y0 = int(y_pos)
                y1 = y0 + 1
                x0 = int(x_pos)
                x1 = x0 + 1
                
                # 确保索引在范围内
                y0 = min(y0, base_noise.shape[0] - 1)
                y1 = min(y1, base_noise.shape[0] - 1)
                x0 = min(x0, base_noise.shape[1] - 1)
                x1 = min(x1, base_noise.shape[1] - 1)
                
                # 双线性插值
                fx = x_pos - x0
                fy = y_pos - y0
                
                v1 = base_noise[y0, x0] * (1 - fx) + base_noise[y0, x1] * fx
                v2 = base_noise[y1, x0] * (1 - fx) + base_noise[y1, x1] * fx
                layer[y, x] = v1 * (1 - fy) + v2 * fy
        
        noise += layer * amplitude
        max_value += amplitude
        
        amplitude *= persistence
        frequency *= lacunarity
        
        # 为下一层生成新的随机噪声
        base_noise = np.random.rand(int(height // scale * frequency) + 2, 
                                   int(width // scale * frequency) + 2)
    
    # 归一化到0-1范围
    noise = noise / max_value
    
    # 应用平滑函数使噪声更自然
    noise = np.sin(noise * np.pi - np.pi/2) * 0.5 + 0.5
    
    return noise

def apply_perlin_noise_to_slider(slider, noise_alpha_percentage):
    """
    为滑块应用柏林噪声
    
    Args:
        slider: 滑块图像 (BGRA)
        noise_alpha_percentage: 噪声覆盖的alpha百分比 (0-100)
    
    Returns:
        应用噪声后的滑块
    """
    h, w = slider.shape[:2]
    result = slider.copy()
    
    # 生成柏林噪声
    noise = generate_perlin_noise((h, w), scale=20, octaves=3, persistence=0.6)
    
    # 创建噪声颜色（灰白色调）
    noise_color = np.ones((h, w, 3), dtype=np.uint8)
    base_gray = random.randint(180, 220)  # 随机灰度基准
    
    for c in range(3):
        # 为每个通道添加轻微的变化
        channel_variation = random.randint(-10, 10)
        noise_color[:, :, c] = (base_gray + channel_variation + noise * 30).astype(np.uint8)
    
    # 只在滑块的有效区域（alpha > 0）应用噪声
    slider_mask = slider[:, :, 3] > 0
    
    # 计算噪声的alpha值
    noise_alpha = noise * (noise_alpha_percentage / 100.0) * 255
    
    # 在滑块区域内应用噪声
    for c in range(3):
        # 使用alpha混合公式
        result[:, :, c] = np.where(
            slider_mask,
            (result[:, :, c] * (255 - noise_alpha) + noise_color[:, :, c] * noise_alpha) / 255,
            result[:, :, c]
        ).astype(np.uint8)
    
    return result

def generate_noise_gradient_test():
    """生成噪声梯度测试图"""
    # 创建输出目录
    output_dir = Path(__file__).parent.parent / "test_output" / "slider_noise"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取背景图片
    img_path = Path(__file__).parent.parent / "data" / "raw" / "nature" / "nature_001.jpg"
    if not img_path.exists():
        # 如果找不到，创建测试图像
        img = np.ones((160, 320, 3), dtype=np.uint8) * 100
        for _ in range(50):
            x = random.randint(0, 320)
            y = random.randint(0, 160)
            cv2.circle(img, (x, y), random.randint(5, 20), 
                      (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)), -1)
    else:
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (320, 160))
    
    # 创建预览画布 (2行3列，只显示6张)
    rows = 2
    cols = 3
    img_width = 320
    img_height = 160
    padding = 20  # 图片之间的间距
    
    canvas_width = cols * img_width + (cols + 1) * padding
    canvas_height = rows * img_height + (rows + 1) * padding + 40  # 额外空间给标题
    preview_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 添加标题
    cv2.putText(preview_canvas, "Perlin Noise Gradient Test", 
               (canvas_width // 2 - 120, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 噪声强度梯度（选择6个有代表性的值）
    noise_levels = [30, 40, 50, 70, 85, 100]
    
    print("Generating slider noise gradient test...")
    
    for i, noise_level in enumerate(noise_levels):
        # 创建拼图形状
        size = 60
        puzzle_mask = create_puzzle_piece(
            piece_size=size,
            knob_radius=int(size * 0.2),
            edges=('convex', 'concave', 'convex', 'concave')
        )
        
        mask_h, mask_w = puzzle_mask.shape[:2]
        alpha = puzzle_mask[:, :, 3]
        
        # 位置
        x = 130
        y = 50
        
        # 创建背景（应用缺口）
        background = apply_gap_lighting(img, x, y, alpha, mask_h, mask_w)
        
        # 创建滑块
        puzzle = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
        img_region = img[y:y+mask_h, x:x+mask_w]
        
        for c in range(3):
            puzzle[:, :, c] = img_region[:, :, c]
        puzzle[:, :, 3] = alpha
        
        # 应用滑块光照
        puzzle = apply_slider_lighting(puzzle)
        
        # 应用柏林噪声
        puzzle = apply_perlin_noise_to_slider(puzzle, noise_level)
        
        # 滑块位置
        slider_x = 20
        slider_y = y
        
        # 创建滑块边框
        slider_frame = create_slider_frame(60, 160)
        
        # 合成最终图像
        sd_center_x = slider_x + 30
        sd_center_y = slider_y + puzzle.shape[0] // 2
        
        final_image = composite_slider(background, puzzle, (sd_center_x, sd_center_y), slider_frame)
        
        # 保存单独的图片
        filename = f"slider_noise_{noise_level}percent.png"
        cv2.imwrite(str(output_dir / filename), final_image)
        
        # 添加到预览画布
        if i < 6:  # 只显示前6张
            row = i // 3
            col = i % 3
            # 计算位置，包含间距
            y_offset = 40 + padding + row * (img_height + padding)
            x_offset = padding + col * (img_width + padding)
            preview_canvas[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = final_image
            
            # 添加噪声强度标注
            cv2.putText(preview_canvas, f"{noise_level}% Noise", 
                       (x_offset + 120, y_offset + 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 保存预览图
    cv2.imwrite(str(output_dir / "noise_gradient_preview.png"), preview_canvas)
    
    print(f"Generated {len(noise_levels)} images with different noise levels")
    print(f"Output directory: {output_dir}")
    print(f"Preview saved as: noise_gradient_preview.png")

if __name__ == "__main__":
    generate_noise_gradient_test()