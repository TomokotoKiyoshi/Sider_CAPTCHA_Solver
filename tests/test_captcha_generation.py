# -*- coding: utf-8 -*-
"""
测试在真实图片上生成拼图验证码
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.captcha_generator.simple_puzzle_generator import create_puzzle_piece
from src.captcha_generator.lighting_effects import apply_gap_lighting, add_puzzle_piece_shadow
import random


def generate_captcha_from_image(image_path, puzzle_size=60, knob_radius=12):
    """
    从图片生成拼图验证码
    
    Args:
        image_path: 图片路径
        puzzle_size: 拼图大小
        knob_radius: 凹凸半径
        
    Returns:
        background: 带凹陷效果的背景图（缺口部分变暗但保留内容）
        puzzle: 拼图块
        position: 拼图位置 (x, y)
        edges: 拼图形状
    """
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # 调整图片大小到 320x160
    img = cv2.resize(img, (320, 160))
    
    # 随机选择拼图形状（从81种中选1种）
    edge_types = ['concave', 'flat', 'convex']
    edges = tuple(random.choice(edge_types) for _ in range(4))
    
    # 创建拼图掩码
    puzzle_mask = create_puzzle_piece(piece_size=puzzle_size, 
                                     knob_radius=knob_radius, 
                                     edges=edges)
    
    # 获取拼图掩码的实际大小
    mask_h, mask_w = puzzle_mask.shape[:2]
    
    # 随机选择拼图位置（确保拼图在图片范围内）
    # x轴要留出空间避免太靠近左边（滑块宽度+10px）
    min_x = puzzle_size + 10
    max_x = img.shape[1] - mask_w
    max_y = img.shape[0] - mask_h
    
    x = random.randint(min_x, max_x)
    y = random.randint(0, max_y)
    
    # 提取拼图块
    puzzle = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
    
    # 从原图中提取对应区域
    img_region = img[y:y+mask_h, x:x+mask_w]
    
    # 应用掩码
    alpha = puzzle_mask[:, :, 3]
    for c in range(3):  # BGR通道
        puzzle[:, :, c] = img_region[:, :, c]
    puzzle[:, :, 3] = alpha
    
    # 创建背景并应用光照效果
    background = apply_gap_lighting(img, x, y, alpha, mask_h, mask_w)
    
    # 为拼图块添加阴影效果
    puzzle = add_puzzle_piece_shadow(puzzle, shadow_offset=(3, 3), shadow_opacity=0.4)
    
    # 返回拼图中心位置
    center_x = x + mask_w // 2
    center_y = y + mask_h // 2
    
    return background, puzzle, (center_x, center_y), edges


def test_single_image():
    """测试单张图片的拼图生成"""
    # 查找一张测试图片
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        print("Please run the image download script first:")
        print("  python scripts/download_images.py")
        return
    
    image_files = []
    # 遍历所有类别文件夹
    for category_dir in data_dir.iterdir():
        if category_dir.is_dir():
            # 获取该类别下的图片
            images = list(category_dir.glob("*.png"))
            if images:
                image_files.extend(images)
    
    if not image_files:
        print("No images found in data/raw directory!")
        print("Please run the image download script first:")
        print("  python scripts/download_images.py")
        return
    
    # 随机选择一张图片
    test_image = random.choice(image_files)
    print(f"Testing with image: {test_image}")
    
    # 生成拼图验证码
    background, puzzle, position, edges = generate_captcha_from_image(test_image)
    
    # 显示结果
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 原始图片
    original = cv2.imread(str(test_image))
    original = cv2.resize(original, (320, 160))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 背景（带凹陷效果）
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    axes[1].imshow(background_rgb)
    axes[1].set_title(f'Background with Depression Effect\nPosition: ({position[0]}, {position[1]})')
    axes[1].axis('off')
    
    plt.suptitle('Slider CAPTCHA Generation Test', fontsize=16)
    plt.tight_layout()
    
    # 保存结果到outputs目录
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "captcha_generation_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved test result to: {output_path}")
    
    plt.show()


def test_multiple_shapes():
    """测试同一张图片生成多种形状"""
    # 查找测试图片
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        print("Please run the image download script first:")
        print("  python scripts/download_images.py")
        return
    
    image_files = []
    for category_dir in data_dir.iterdir():
        if category_dir.is_dir():
            images = list(category_dir.glob("*.png"))
            if images:
                image_files.append(images[0])  # 每个类别取第一张
                break
    
    if not image_files:
        print("No images found in data/raw directory!")
        print("Please run the image download script first:")
        print("  python scripts/download_images.py")
        return
    
    test_image = image_files[0]
    print(f"Testing multiple shapes with: {test_image}")
    
    # 生成6种不同形状
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    shape_examples = [
        ('flat', 'flat', 'flat', 'flat'),
        ('convex', 'convex', 'convex', 'convex'),
        ('concave', 'concave', 'concave', 'concave'),
        ('flat', 'convex', 'flat', 'concave'),
        ('convex', 'flat', 'concave', 'flat'),
        ('concave', 'convex', 'concave', 'convex'),
    ]
    
    for ax, edges in zip(axes.flat, shape_examples):
        # 创建固定形状的拼图
        img = cv2.imread(str(test_image))
        img = cv2.resize(img, (320, 160))
        
        puzzle_mask = create_puzzle_piece(piece_size=60, knob_radius=12, edges=edges)
        mask_h, mask_w = puzzle_mask.shape[:2]
        
        # 固定位置用于比较
        x, y = 150, 50
        
        # 创建背景并应用光照效果
        alpha = puzzle_mask[:, :, 3]
        background = apply_gap_lighting(img, x, y, alpha, mask_h, mask_w)
        
        # 显示
        background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        ax.imshow(background_rgb)
        ax.set_title(f'{edges[0][:3]}, {edges[1][:3]}, {edges[2][:3]}, {edges[3][:3]}', 
                     fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Different Puzzle Shapes on Same Image', fontsize=16)
    plt.tight_layout()
    
    # 保存到outputs目录
    output_dir = Path(__file__).parent.parent / "outputs"
    output_path = output_dir / "multiple_shapes_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved multiple shapes test to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Testing CAPTCHA generation...")
    test_single_image()
    print("\nTesting multiple shapes...")
    test_multiple_shapes()
    print("\nDone!")