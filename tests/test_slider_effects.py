# -*- coding: utf-8 -*-
"""
测试滑块光照效果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.captcha_generator.simple_puzzle_generator import create_puzzle_piece
from src.captcha_generator.lighting_effects import apply_gap_lighting
from src.captcha_generator.slider_effects import (
    apply_slider_lighting, create_slider_frame, composite_slider
)
import random


def test_slider_lighting():
    """测试滑块光照效果"""
    # 查找测试图片
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return
    
    # 获取一张测试图片
    image_files = []
    for category_dir in data_dir.iterdir():
        if category_dir.is_dir():
            images = list(category_dir.glob("*.png"))
            if images:
                image_files.append(images[0])
                break
    
    if not image_files:
        print("No images found!")
        return
    
    # 读取图片
    test_image = image_files[0]
    img = cv2.imread(str(test_image))
    img = cv2.resize(img, (320, 160))
    
    # 生成拼图形状
    edges = ('convex', 'flat', 'concave', 'flat')
    puzzle_mask = create_puzzle_piece(piece_size=60, knob_radius=12, edges=edges)
    mask_h, mask_w = puzzle_mask.shape[:2]
    
    # 拼图位置
    x, y = 150, 50
    
    # 提取拼图块
    puzzle = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
    img_region = img[y:y+mask_h, x:x+mask_w]
    
    # 应用掩码
    alpha = puzzle_mask[:, :, 3]
    for c in range(3):
        puzzle[:, :, c] = img_region[:, :, c]
    puzzle[:, :, 3] = alpha
    
    # 创建背景（带凹陷效果）
    background = apply_gap_lighting(img, x, y, alpha, mask_h, mask_w)
    
    # 应用滑块光照效果（使用默认参数）
    lit_puzzle = apply_slider_lighting(puzzle.copy())
    
    # 创建显示 - 需要将BGR转换为RGB用于显示
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 原始拼图块（BGR->RGB）
    ax1 = plt.subplot(2, 3, 1)
    puzzle_rgb = puzzle.copy()
    puzzle_rgb[:, :, :3] = cv2.cvtColor(puzzle[:, :, :3], cv2.COLOR_BGR2RGB)
    ax1.imshow(puzzle_rgb)
    ax1.set_title('Original Puzzle Piece')
    ax1.axis('off')
    
    # 2. 默认光照效果（BGR->RGB）
    ax2 = plt.subplot(2, 3, 2)
    lit_puzzle_rgb = lit_puzzle.copy()
    lit_puzzle_rgb[:, :, :3] = cv2.cvtColor(lit_puzzle[:, :, :3], cv2.COLOR_BGR2RGB)
    ax2.imshow(lit_puzzle_rgb)
    ax2.set_title('Default Lighting\n(Edge:5px, Decay:3.0)')
    ax2.axis('off')
    
    # 3. 强光照效果
    ax3 = plt.subplot(2, 3, 3)
    strong_lit = apply_slider_lighting(puzzle.copy(), 
                                     highlight_strength=0,
                                     edge_highlight=100,
                                     directional_highlight=30,
                                     edge_width=5,
                                     decay_factor=2.0)
    strong_lit_rgb = strong_lit.copy()
    strong_lit_rgb[:, :, :3] = cv2.cvtColor(strong_lit[:, :, :3], cv2.COLOR_BGR2RGB)
    ax3.imshow(strong_lit_rgb)
    ax3.set_title('Strong Lighting\n(Edge:100, Decay:2.0)')
    ax3.axis('off')
    
    # 4. 背景（带凹陷）
    ax4 = plt.subplot(2, 3, 4)
    bg_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    ax4.imshow(bg_rgb)
    ax4.set_title('Background with Gap Depression')
    ax4.axis('off')
    
    # 5. 滑块框架测试
    ax5 = plt.subplot(2, 3, 5)
    
    # 创建背景（用于显示效果）
    frame_bg = np.ones((160, 60, 3), dtype=np.uint8) * 100
    
    # 使用composite_slider函数正确合成
    slider_center_x = 30
    slider_center_y = 80
    
    # 创建滑块框架
    slider_frame = create_slider_frame()
    
    # 使用composite_slider合成（这个函数会正确处理拼图块和框架）
    slider_demo = composite_slider(frame_bg, lit_puzzle, (slider_center_x, slider_center_y), slider_frame)
    
    # BGR转RGB显示
    slider_demo_rgb = cv2.cvtColor(slider_demo, cv2.COLOR_BGR2RGB)
    
    ax5.imshow(slider_demo_rgb)
    ax5.set_title('Slider with Frame')
    ax5.axis('off')
    
    # 6. 完整效果演示
    ax6 = plt.subplot(2, 3, 6)
    # 使用原始背景
    full_demo = bg_rgb.copy()
    
    # 滑块在左边
    slider_width = 60
    # 滑块框架应该在左侧 10px + slider_width 的范围内
    # 也就是说，滑块框架的左边缘可以在 0 到 10px 之间随机
    # 这样整个框架（宽度60px）会在 0-70px 的范围内
    np.random.seed(None)  # 确保每次运行都有不同的随机数
    slider_x = np.random.randint(0, 11)  # 框架左边缘位置：0到10px之间
    print(f"Random slider_x: {slider_x}")  # 调试输出
    
    # 滑块y位置：与缺口中心对齐
    gap_center_y = y + mask_h // 2
    
    # 合成拼图块到滑块中心
    slider_center_x = slider_x + slider_width // 2
    slider_center_y = gap_center_y
    
    # 创建滑块框架
    slider_frame_full = create_slider_frame(slider_width, full_demo.shape[0])
    
    # 使用composite_slider合成（需要先转换为BGR）
    full_demo_bgr = cv2.cvtColor(full_demo, cv2.COLOR_RGB2BGR)
    full_demo_bgr = composite_slider(full_demo_bgr, lit_puzzle, (slider_center_x, slider_center_y), slider_frame_full)
    
    # 转回RGB显示
    full_demo = cv2.cvtColor(full_demo_bgr, cv2.COLOR_BGR2RGB)
    
    ax6.imshow(full_demo)
    ax6.set_title(f'Complete CAPTCHA Demo\nSlider at x={slider_x}')
    ax6.axis('off')
    
    plt.suptitle('Slider 3D Lighting Effects Test', fontsize=16)
    plt.tight_layout()
    
    # 保存结果
    output_dir = Path(__file__).parent.parent / "outputs"
    output_path = output_dir / "slider_lighting_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.show()


def test_lighting_comparison():
    """测试不同光照参数的效果对比"""
    # 创建一个简单的测试拼图
    edges = ('convex', 'convex', 'convex', 'convex')
    puzzle_mask = create_puzzle_piece(piece_size=80, knob_radius=16, edges=edges)
    
    # 创建纯色拼图用于更好地展示光照效果
    h, w = puzzle_mask.shape[:2]
    puzzle = np.zeros((h, w, 4), dtype=np.uint8)
    # 使用中等灰色
    puzzle[:, :, :3] = 128
    puzzle[:, :, 3] = puzzle_mask[:, :, 3]
    
    # 不同的光照配置（固定边缘宽度为5px）
    lighting_configs = [
        {"highlight_strength": 0, "edge_highlight": 40, "directional_highlight": 10, "decay_factor": 4.0},
        {"highlight_strength": 0, "edge_highlight": 60, "directional_highlight": 20, "decay_factor": 3.0},
        {"highlight_strength": 0, "edge_highlight": 80, "directional_highlight": 25, "decay_factor": 2.5},
        {"highlight_strength": 0, "edge_highlight": 100, "directional_highlight": 30, "decay_factor": 2.0},
    ]
    
    labels = ['Subtle', 'Default', 'Strong', 'Very Strong']
    
    # 创建对比图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for ax, config, label in zip(axes, lighting_configs, labels):
        # 固定边缘宽度为5px
        lit_puzzle = apply_slider_lighting(puzzle.copy(), edge_width=5, **config)
        lit_puzzle_rgb = lit_puzzle.copy()
        lit_puzzle_rgb[:, :, :3] = cv2.cvtColor(lit_puzzle[:, :, :3], cv2.COLOR_BGR2RGB)
        ax.imshow(lit_puzzle_rgb)
        ax.set_title(f'{label}\nE:{config["edge_highlight"]} '
                     f'D:{config["directional_highlight"]} '
                     f'Decay:{config["decay_factor"]}')
        ax.axis('off')
    
    plt.suptitle('Slider Lighting Parameters Comparison', fontsize=16)
    plt.tight_layout()
    
    # 保存
    output_dir = Path(__file__).parent.parent / "outputs"
    output_path = output_dir / "slider_lighting_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Testing slider lighting effects...")
    test_slider_lighting()
    
    print("\nTesting lighting parameters comparison...")
    test_lighting_comparison()