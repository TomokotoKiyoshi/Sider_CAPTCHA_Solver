# -*- coding: utf-8 -*-
"""
测试滑块在背景图内的显示效果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.captcha_generator.confusion_system.generator import CaptchaGenerator
from src.captcha_generator.confusion_system.strategies import (
    PerlinNoiseConfusion,
    HollowCenterConfusion
)


def create_display_with_slider_in_background(background: np.ndarray, slider: np.ndarray, 
                                           slider_pos: tuple, gap_pos: tuple) -> np.ndarray:
    """
    创建滑块在背景内的展示图像
    
    Args:
        background: 背景图（带缺口）
        slider: 滑块图
        slider_pos: 滑块目标位置（应该与gap位置对齐时的位置）
        gap_pos: 缺口位置
        
    Returns:
        合成后的图像
    """
    result = background.copy()
    h, w = slider.shape[:2]
    
    # 计算滑块左上角位置
    x = slider_pos[0] - w // 2
    y = slider_pos[1] - h // 2
    
    # 确保不超出边界
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(result.shape[1], x + w)
    y2 = min(result.shape[0], y + h)
    
    # 实际可绘制区域
    actual_w = x2 - x1
    actual_h = y2 - y1
    
    # 源区域（滑块）的对应部分
    src_x1 = max(0, -x)
    src_y1 = max(0, -y)
    src_x2 = src_x1 + actual_w
    src_y2 = src_y1 + actual_h
    
    # 如果滑块有alpha通道，使用alpha混合
    if slider.shape[2] == 4:
        alpha = slider[src_y1:src_y2, src_x1:src_x2, 3] / 255.0
        for c in range(3):
            result[y1:y2, x1:x2, c] = (
                result[y1:y2, x1:x2, c] * (1 - alpha) + 
                slider[src_y1:src_y2, src_x1:src_x2, c] * alpha
            ).astype(np.uint8)
    else:
        result[y1:y2, x1:x2] = slider[src_y1:src_y2, src_x1:src_x2]
    
    # 绘制引导线（可选）
    # 从滑块中心到缺口中心
    cv2.arrowedLine(result, 
                   (slider_pos[0], slider_pos[1]), 
                   (gap_pos[0], gap_pos[1]),
                   (0, 255, 0), 2, tipLength=0.1)
    
    return result


def test_slider_positions():
    """测试不同的滑块位置"""
    # 使用Abstract_Geometric_Art目录下的图片
    img_dir = Path(__file__).parent.parent / "data" / "raw" / "Abstract_Geometric_Art"
    img1 = cv2.imread(str(img_dir / "Pic0001.png"))
    img2 = cv2.imread(str(img_dir / "Pic0002.png"))
    
    generator = CaptchaGenerator()
    
    # 测试配置 - 使用符合CLAUDE.md的滑块位置范围[15, 35]
    test_configs = [
        {
            'name': 'left_position',
            'image': img1,
            'gap_pos': (100, 80),
            'slider_pos': (25, 80),  # 滑块在标准左侧位置
            'shape': 'circle',
            'size': 50
        },
        {
            'name': 'gap_near_center',
            'image': img1,
            'gap_pos': (200, 80),
            'slider_pos': (30, 80),  # 滑块在标准位置，缺口在中间
            'shape': 'square',
            'size': 50
        },
        {
            'name': 'gap_right_side',
            'image': img2,
            'gap_pos': (250, 80),
            'slider_pos': (20, 80),  # 滑块在标准位置，缺口在右侧
            'shape': 'triangle',
            'size': 50
        },
        {
            'name': 'different_y_position',
            'image': img2,
            'gap_pos': (200, 100),
            'slider_pos': (25, 100),  # 不同的y坐标
            'shape': 'hexagon',
            'size': 50
        }
    ]
    
    output_dir = Path(__file__).parent / "slider_in_background_output"
    output_dir.mkdir(exist_ok=True)
    
    # 创建大画布显示所有结果
    canvas_rows = 2
    canvas_cols = 2
    cell_height = 200
    cell_width = 320
    canvas = np.ones((cell_height * canvas_rows, cell_width * canvas_cols, 3), 
                     dtype=np.uint8) * 240
    
    for i, config in enumerate(test_configs):
        print(f"\nTesting {config['name']}...")
        
        # 生成验证码
        result = generator.generate(
            image=config['image'],
            puzzle_shape=config['shape'],
            puzzle_size=config['size'],
            gap_position=config['gap_pos'],
            slider_position=config['slider_pos'],
            apply_shadow=True,
            confusion_strategies=[]
        )
        
        # 创建滑块在背景内的显示
        display = create_display_with_slider_in_background(
            result.background,
            result.slider,
            config['slider_pos'],
            config['gap_pos']
        )
        
        # 添加标注
        cv2.putText(display, config['name'], (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display, f"Gap: {config['gap_pos']}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(display, f"Slider: {config['slider_pos']}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # 保存单独的图片
        cv2.imwrite(str(output_dir / f"{config['name']}.png"), display)
        
        # 添加到画布
        row = i // canvas_cols
        col = i % canvas_cols
        y1 = row * cell_height
        y2 = y1 + 160
        x1 = col * cell_width
        x2 = x1 + cell_width
        
        canvas[y1:y2, x1:x2] = display
        
        # 添加标题
        title_y = y2 + 20
        cv2.putText(canvas, config['name'].replace('_', ' ').title(), 
                   (x1 + 10, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 保存汇总图
    cv2.imwrite(str(output_dir / "summary_slider_in_background.png"), canvas)
    
    print(f"\nTest completed! Results saved in: {output_dir}")
    print("Check summary_slider_in_background.png for all effects")


def test_slider_movement_animation():
    """创建滑块移动的动画序列"""
    # 使用Abstract_Geometric_Art目录下的图片
    img_dir = Path(__file__).parent.parent / "data" / "raw" / "Abstract_Geometric_Art"
    img = cv2.imread(str(img_dir / "Pic0001.png"))
    
    generator = CaptchaGenerator()
    
    # 生成一个验证码
    gap_pos = (250, 80)
    initial_slider_pos = (25, 80)
    
    result = generator.generate(
        image=img,
        puzzle_shape='circle',
        puzzle_size=50,
        gap_position=gap_pos,
        slider_position=initial_slider_pos,
        apply_shadow=True,
        confusion_strategies=[HollowCenterConfusion({'hollow_ratio': 0.4})]
    )
    
    output_dir = Path(__file__).parent / "slider_movement_frames"
    output_dir.mkdir(exist_ok=True)
    
    # 创建移动序列
    num_frames = 10
    for i in range(num_frames):
        # 计算当前滑块位置（从左到右移动）
        progress = i / (num_frames - 1)
        current_x = int(initial_slider_pos[0] + (gap_pos[0] - initial_slider_pos[0]) * progress)
        current_pos = (current_x, initial_slider_pos[1])
        
        # 创建当前帧
        frame = create_display_with_slider_in_background(
            result.background,
            result.slider,
            current_pos,
            gap_pos
        )
        
        # 添加进度信息
        cv2.putText(frame, f"Frame {i+1}/{num_frames}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"X: {current_x}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 保存帧
        cv2.imwrite(str(output_dir / f"frame_{i:02d}.png"), frame)
    
    print(f"\nAnimation frames saved to: {output_dir}")
    print("You can use these frames to create a GIF animation")


if __name__ == "__main__":
    print("Testing slider display in background...")
    test_slider_positions()
    
    print("\n\nCreating slider movement animation sequence...")
    test_slider_movement_animation()