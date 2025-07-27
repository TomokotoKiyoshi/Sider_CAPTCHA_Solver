# -*- coding: utf-8 -*-
"""
测试不同级别的亮度和暗度效果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.captcha_generator.simple_puzzle_generator import create_puzzle_piece
from src.captcha_generator.lighting_effects import apply_gap_lighting, apply_gap_highlighting

def test_brightness_darkness_levels(image_path):
    """测试20个级别的亮度和20个级别的暗度"""
    
    # 读取图片
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        # 如果是numpy数组
        img = image_path
    
    if img is None:
        print(f"Cannot read image from: {image_path}")
        return
    
    # 调整图片大小到320x160
    img = cv2.resize(img, (320, 160))
    
    # 创建输出目录（在项目根目录下的test_output）
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "test_output" / "brightness_darkness_levels"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建拼图块（用于测试缺口效果）
    piece_size = 50
    knob_radius = int(piece_size * 0.2)
    puzzle_mask = create_puzzle_piece(
        piece_size=piece_size,
        knob_radius=knob_radius,
        edges=('convex', 'concave', 'convex', 'concave')
    )
    
    mask_h, mask_w = puzzle_mask.shape[:2]
    alpha = puzzle_mask[:, :, 3]
    
    # 定义位置（确保在图片中央）
    x = (320 - mask_w) // 2
    y = (160 - mask_h) // 2
    
    # 创建20个级别的暗度效果
    print("Creating darkness levels...")
    darkness_samples = []
    
    for i in range(20):
        # 计算参数（从轻微到极暗）
        factor = i / 19.0  # 0到1
        
        base_darkness = int(10 + factor * 40)      # 10-50
        edge_darkness = int(20 + factor * 60)      # 20-80
        directional_darkness = int(10 + factor * 30)  # 10-40
        outer_edge_darkness = int(5 + factor * 10)    # 5-15
        
        # 应用效果
        result = apply_gap_lighting(
            img.copy(), x, y, alpha, mask_h, mask_w,
            base_darkness=base_darkness,
            edge_darkness=edge_darkness,
            directional_darkness=directional_darkness,
            outer_edge_darkness=outer_edge_darkness
        )
        
        darkness_samples.append(result)
    
    # 创建20个级别的亮度效果
    print("Creating brightness levels...")
    brightness_samples = []
    
    for i in range(20):
        # 计算参数（从轻微到极亮）
        factor = i / 19.0  # 0到1
        
        base_lightness = int(10 + factor * 40)      # 10-50
        edge_lightness = int(20 + factor * 60)      # 20-80
        directional_lightness = int(10 + factor * 30)  # 10-40
        outer_edge_lightness = int(5 + factor * 20)    # 5-25
        
        # 应用效果
        result = apply_gap_highlighting(
            img.copy(), x, y, alpha, mask_h, mask_w,
            base_lightness=base_lightness,
            edge_lightness=edge_lightness,
            directional_lightness=directional_lightness,
            outer_edge_lightness=outer_edge_lightness
        )
        
        brightness_samples.append(result)
    
    # 创建暗度级别图（4列x5行）
    print("Creating darkness levels visualization...")
    cols = 4
    rows = 5
    cell_width = 320
    cell_height = 160
    padding = 15  # 图片之间的间距
    label_height = 30  # 标签高度
    
    grid_width = cols * cell_width + (cols - 1) * padding
    grid_height = rows * (cell_height + label_height) + (rows - 1) * padding
    darkness_grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # 纯白色背景
    
    for i, sample in enumerate(darkness_samples):
        row = i // cols
        col = i % cols
        
        # 计算位置（包括标签空间）
        y_start = row * (cell_height + label_height + padding) + label_height
        x_start = col * (cell_width + padding)
        
        # 添加标签到图片上方
        factor = i / 19.0  # 0到1
        base_darkness = int(10 + factor * 40)
        edge_darkness = int(20 + factor * 60)
        
        label_text = f"Level D{i+1:02d}: Base={base_darkness}, Edge={edge_darkness}"
        label_y = y_start - label_height + 20
        label_x = x_start + 10
        
        cv2.putText(darkness_grid, label_text, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # 放置图片（去除原有标签）
        clean_sample = sample.copy()
        # 清除原有标签区域（左上角）
        clean_sample[0:35, 0:100] = img[0:35, 0:100]
        darkness_grid[y_start:y_start+cell_height, x_start:x_start+cell_width] = clean_sample
    
    darkness_output = output_dir / "darkness_20_levels.png"
    cv2.imwrite(str(darkness_output), darkness_grid)
    print(f"Darkness levels saved to: {darkness_output}")
    
    # 创建亮度级别图（4列x5行）
    print("Creating brightness levels visualization...")
    brightness_grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # 纯白色背景
    
    for i, sample in enumerate(brightness_samples):
        row = i // cols
        col = i % cols
        
        # 计算位置（包括标签空间）
        y_start = row * (cell_height + label_height + padding) + label_height
        x_start = col * (cell_width + padding)
        
        # 添加标签到图片上方
        factor = i / 19.0  # 0到1
        base_lightness = int(10 + factor * 40)
        edge_lightness = int(20 + factor * 60)
        
        label_text = f"Level B{i+1:02d}: Base={base_lightness}, Edge={edge_lightness}"
        label_y = y_start - label_height + 20
        label_x = x_start + 10
        
        cv2.putText(brightness_grid, label_text, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # 放置图片（去除原有标签）
        clean_sample = sample.copy()
        # 清除原有标签区域（左上角）
        clean_sample[0:35, 0:100] = img[0:35, 0:100]
        brightness_grid[y_start:y_start+cell_height, x_start:x_start+cell_width] = clean_sample
    
    brightness_output = output_dir / "brightness_20_levels.png"
    cv2.imwrite(str(brightness_output), brightness_grid)
    print(f"Brightness levels saved to: {brightness_output}")
    
    # 创建对比图（选择部分级别展示）
    print("Creating comparison visualization...")
    
    # 选择5个代表性级别：1, 5, 10, 15, 20
    selected_indices = [0, 4, 9, 14, 19]
    comparison_samples = []
    
    # 添加原图
    original = img.copy()
    comparison_samples.append(original)
    
    # 添加选定的暗度级别
    for idx in selected_indices:
        comparison_samples.append(darkness_samples[idx])
    
    # 添加选定的亮度级别
    for idx in selected_indices:
        comparison_samples.append(brightness_samples[idx])
    
    # 创建对比网格（3行x4列）
    comp_rows = 3
    comp_cols = 4
    comp_grid_width = comp_cols * cell_width + (comp_cols - 1) * padding
    comp_grid_height = comp_rows * (cell_height + label_height) + (comp_rows - 1) * padding
    comparison_grid = np.ones((comp_grid_height, comp_grid_width, 3), dtype=np.uint8) * 255  # 纯白色背景
    
    # 放置样本
    positions = [
        (0, 0),  # Original
        (0, 1), (0, 2), (0, 3), (1, 0), (1, 1),  # Darkness samples
        (1, 2), (1, 3), (2, 0), (2, 1), (2, 2)   # Brightness samples
    ]
    
    # 标签
    labels = [
        "Original",
        "Darkness D01", "Darkness D05", "Darkness D10", "Darkness D15", "Darkness D20",
        "Brightness B01", "Brightness B05", "Brightness B10", "Brightness B15", "Brightness B20"
    ]
    
    for i, (sample, (row, col)) in enumerate(zip(comparison_samples, positions)):
        # 计算位置（包括标签空间）
        y_start = row * (cell_height + label_height + padding) + label_height
        x_start = col * (cell_width + padding)
        
        # 添加标签到图片上方
        label_text = labels[i]
        label_y = y_start - label_height + 20
        label_x = x_start + 10
        
        cv2.putText(comparison_grid, label_text, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        # 放置图片（去除原有标签）
        clean_sample = sample.copy()
        # 清除原有标签区域（左上角）
        clean_sample[0:35, 0:100] = img[0:35, 0:100]
        comparison_grid[y_start:y_start+cell_height, x_start:x_start+cell_width] = clean_sample
    
    comparison_output = output_dir / "comparison_selected_levels.png"
    cv2.imwrite(str(comparison_output), comparison_grid)
    print(f"Comparison saved to: {comparison_output}")
    
    # 创建参数变化图表
    print("\nParameter progression:")
    print("Darkness levels:")
    print("Level 1:  Base=10,  Edge=20,  Directional=10,  Outer=5")
    print("Level 10: Base=28,  Edge=48,  Directional=24,  Outer=9")
    print("Level 20: Base=50,  Edge=80,  Directional=40,  Outer=15")
    print("\nBrightness levels:")
    print("Level 1:  Base=10,  Edge=20,  Directional=10,  Outer=5")
    print("Level 10: Base=28,  Edge=48,  Directional=24,  Outer=14")
    print("Level 20: Base=50,  Edge=80,  Directional=40,  Outer=25")
    
    print("\nAll visualizations created successfully!")
    print(f"Output directory: {output_dir}")


def create_tile_pattern_from_reference():
    """创建类似于参考图片的彩色瓷砖图案"""
    width, height = 320, 160
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 定义瓷砖颜色（基于参考图片的色调）
    tile_colors = [
        # 蓝绿色系
        (180, 200, 190),  # 浅蓝绿
        (140, 180, 170),  # 中蓝绿
        (100, 150, 140),  # 深蓝绿
        (160, 190, 200),  # 天蓝
        
        # 粉色系
        (200, 160, 170),  # 浅粉
        (180, 140, 150),  # 中粉
        (160, 120, 130),  # 深粉
        (210, 180, 190),  # 淡粉
        
        # 灰色系
        (180, 180, 180),  # 浅灰
        (150, 150, 150),  # 中灰
        (120, 120, 120),  # 深灰
        (200, 200, 190),  # 米灰
        
        # 黄褐色系
        (200, 180, 140),  # 浅黄
        (180, 160, 120),  # 中黄
        (220, 200, 160),  # 米黄
        
        # 其他颜色
        (140, 160, 180),  # 灰蓝
        (180, 160, 140),  # 棕褐
        (200, 190, 180),  # 浅褐
    ]
    
    # 创建菱形瓷砖图案
    tile_size = 16  # 瓷砖大小
    
    # 绘制菱形瓷砖
    for row in range(-2, height // tile_size + 2):
        for col in range(-2, width // tile_size + 2):
            # 计算菱形中心
            center_x = col * tile_size + (row % 2) * (tile_size // 2)
            center_y = row * tile_size
            
            # 随机选择颜色
            color = tile_colors[np.random.randint(0, len(tile_colors))]
            
            # 添加颜色变化
            variation = np.random.randint(-15, 15, 3)
            color_array = np.clip(np.array(color) + variation, 0, 255)
            color = tuple(int(c) for c in color_array)
            
            # 绘制菱形
            points = np.array([
                [center_x, center_y - tile_size//2],
                [center_x + tile_size//2, center_y],
                [center_x, center_y + tile_size//2],
                [center_x - tile_size//2, center_y]
            ], np.int32)
            
            cv2.fillPoly(img, [points], color)
            
            # 添加边框效果
            border_color_array = np.clip(np.array(color) - 30, 0, 255)
            border_color = tuple(int(c) for c in border_color_array)
            cv2.polylines(img, [points], True, border_color, 1)
    
    # 添加纹理和老化效果
    # 1. 添加细微噪声
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 2. 添加一些污渍效果
    for _ in range(20):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(5, 15)
        intensity = np.random.randint(-20, -5)
        
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        mask = cv2.GaussianBlur(mask, (radius*2+1, radius*2+1), 0)
        mask = mask.astype(float) / 255.0
        
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c] + (intensity * mask).astype(int), 0, 255)
    
    return img


def test_single_image_effects():
    """测试单张图片的暗化和亮化效果"""
    
    # 创建输出目录
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建类似参考图片的彩色瓷砖图案
    img = create_tile_pattern_from_reference()
    
    # 创建拼图块
    size = 55
    puzzle_mask = create_puzzle_piece(
        piece_size=size,
        knob_radius=int(size * 0.2),
        edges=('convex', 'concave', 'convex', 'concave')
    )
    
    mask_h, mask_w = puzzle_mask.shape[:2]
    alpha = puzzle_mask[:, :, 3]
    
    # 缺口位置（偏右）
    x = 180
    y = (160 - mask_h) // 2
    
    # 1. 生成暗化效果（默认参数）
    img_dark = apply_gap_lighting(
        img.copy(), x, y, alpha, mask_h, mask_w
        # 使用默认参数
    )
    
    # 2. 生成亮化效果（medium强度）
    img_bright = apply_gap_highlighting(
        img.copy(), x, y, alpha, mask_h, mask_w,
        base_lightness=30,
        edge_lightness=45,
        directional_lightness=20,
        outer_edge_lightness=15
    )
    
    # 保存两张图片
    cv2.imwrite(str(output_dir / "pic0001_dark.png"), img_dark)
    cv2.imwrite(str(output_dir / "pic0001_bright.png"), img_bright)
    
    print(f"Dark effect saved to: {output_dir / 'pic0001_dark.png'}")
    print(f"Bright effect saved to: {output_dir / 'pic0001_bright.png'}")


def test_pic0001_grids():
    """生成两张5x4的网格图，分别展示20个暗化和亮化效果"""
    import random
    
    # 创建输出目录
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "test_output" / "pic0003_effects"  # 改为003
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建背景图
    background = create_tile_pattern_from_reference()
    
    # 定义网格参数
    cols = 4
    rows = 5
    cell_width = 320
    cell_height = 160
    padding = 15  # 图片间隙
    label_height = 30  # 标题高度
    
    # 计算网格大小
    grid_width = cols * cell_width + (cols - 1) * padding
    grid_height = rows * (cell_height + label_height) + (rows - 1) * padding
    
    # 创建两个大图（白色背景）
    dark_grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    bright_grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # 定义不同的形状
    shapes = [
        ('convex', 'concave', 'convex', 'concave'),
        ('flat', 'convex', 'flat', 'concave'),
        ('concave', 'flat', 'convex', 'flat'),
        ('convex', 'flat', 'concave', 'convex'),
    ]
    
    # 生成20个效果（递增梯度）
    for i in range(20):
        row = i // cols
        col = i % cols
        
        # 计算梯度因子（0到1）
        gradient_factor = i / 19.0  # 0到1的渐变
        
        # 为每个位置生成不同的参数
        # 随机选择形状
        shape = shapes[i % len(shapes)]
        
        # 随机大小（40-70之间）
        size = random.randint(40, 70)
        
        # 创建拼图块
        puzzle_mask = create_puzzle_piece(
            piece_size=size,
            knob_radius=int(size * 0.2),
            edges=shape
        )
        
        mask_h, mask_w = puzzle_mask.shape[:2]
        alpha = puzzle_mask[:, :, 3]
        
        # 随机位置（确保在合理范围内）
        x = random.randint(60, 320 - mask_w - 20)
        y = random.randint(20, 160 - mask_h - 20)
        
        # 生成暗化效果（递增的暗度）
        # 从轻微到极暗
        base_darkness = int(10 + gradient_factor * 40)      # 10-50
        edge_darkness = int(20 + gradient_factor * 60)      # 20-80
        directional_darkness = int(10 + gradient_factor * 30)  # 10-40
        
        img_dark = apply_gap_lighting(
            background.copy(), x, y, alpha, mask_h, mask_w,
            base_darkness=base_darkness,
            edge_darkness=edge_darkness,
            directional_darkness=directional_darkness,
            outer_edge_darkness=0  # 保持为0
        )
        
        # 生成亮化效果（递增的亮度）
        # 从轻微到极亮
        base_lightness = int(10 + gradient_factor * 40)      # 10-50
        edge_lightness = int(20 + gradient_factor * 60)      # 20-80
        directional_lightness = int(10 + gradient_factor * 30)  # 10-40
        outer_edge_lightness = 0    # 必须为0，不在外围添加效果
        
        img_bright = apply_gap_highlighting(
            background.copy(), x, y, alpha, mask_h, mask_w,
            base_lightness=base_lightness,
            edge_lightness=edge_lightness,
            directional_lightness=directional_lightness,
            outer_edge_lightness=outer_edge_lightness
        )
        
        # 计算网格位置（包括间隙和标题）
        y_start = row * (cell_height + label_height + padding) + label_height
        x_start = col * (cell_width + padding)
        
        # 添加标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 暗化标题
        dark_title = f"#{i+1:02d} Dark Level:{i+1}/20 (B:{base_darkness},E:{edge_darkness})"
        # 亮化标题
        bright_title = f"#{i+1:02d} Bright Level:{i+1}/20 (B:{base_lightness},E:{edge_lightness})"
        
        title_y = y_start - label_height + 20
        title_x = x_start + 10
        
        # 在暗化网格添加标题
        cv2.putText(dark_grid, dark_title, (title_x, title_y), 
                   font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        
        # 在亮化网格添加标题
        cv2.putText(bright_grid, bright_title, (title_x, title_y), 
                   font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        
        # 放置图片
        dark_grid[y_start:y_start+cell_height, x_start:x_start+cell_width] = img_dark
        bright_grid[y_start:y_start+cell_height, x_start:x_start+cell_width] = img_bright
        
        print(f"Generated effect {i+1}/20 - Level {i+1} - Dark(B:{base_darkness},E:{edge_darkness}) Bright(B:{base_lightness},E:{edge_lightness})")
    
    # 保存两张大图
    dark_path = output_dir / "pic0003_dark_5x4.png"
    bright_path = output_dir / "pic0003_bright_5x4.png"
    
    cv2.imwrite(str(dark_path), dark_grid)
    cv2.imwrite(str(bright_path), bright_grid)
    
    print(f"\nDark effects grid (5x4) saved to: {dark_path}")
    print(f"Bright effects grid (5x4) saved to: {bright_path}")


def main():
    """主函数"""
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            test_single_image_effects()
        elif sys.argv[1] == "grid":
            test_pic0001_grids()
    else:
        print("Creating colorful tile pattern similar to the reference image...")
        
        # 创建类似参考图片的彩色瓷砖图案
        img = create_tile_pattern_from_reference()
        
        # 运行测试
        test_brightness_darkness_levels(img)


if __name__ == "__main__":
    main()