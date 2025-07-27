# -*- coding: utf-8 -*-
"""
测试不同暗度级别的效果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import matplotlib.pyplot as plt
from src.captcha_generator.simple_puzzle_generator import create_puzzle_piece
from src.captcha_generator.lighting_effects import apply_gap_lighting


def test_darkness_levels():
    """测试不同暗度参数的效果"""
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
                image_files.append(images[0])
                break
    
    if not image_files:
        print("No images found in data/raw directory!")
        print("Please run the image download script first:")
        print("  python scripts/download_images.py")
        return
    
    # 读取图片
    test_image = image_files[0]
    img = cv2.imread(str(test_image))
    img = cv2.resize(img, (320, 160))
    
    # 创建拼图掩码
    edges = ('convex', 'flat', 'concave', 'flat')
    puzzle_mask = create_puzzle_piece(piece_size=60, knob_radius=12, edges=edges)
    mask_h, mask_w = puzzle_mask.shape[:2]
    alpha = puzzle_mask[:, :, 3]
    
    # 固定位置
    x, y = 130, 50
    
    # 不同的暗度配置
    darkness_configs = [
        # (base, edge, directional, outer_edge)
        (10, 20, 8, 0),      # 浅（无外边缘）
        (25, 35, 15, 0),     # 默认（无外边缘）
        (40, 50, 20, 0),     # 深（无外边缘）
        (25, 35, 15, 8),     # 默认（有外边缘）
    ]
    
    labels = ['Light', 'Default', 'Dark', 'Default (With Outer)']
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Different Darkness Levels for Gap Depression', fontsize=16)
    
    for ax, config, label in zip(axes.flat, darkness_configs, labels):
        # 应用光照效果
        result = apply_gap_lighting(
            img, x, y, alpha, mask_h, mask_w,
            base_darkness=config[0],
            edge_darkness=config[1],
            directional_darkness=config[2],
            outer_edge_darkness=config[3]
        )
        
        # 显示
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        ax.imshow(result_rgb)
        ax.set_title(f'{label}\nBase={config[0]}, Edge={config[1]}, Dir={config[2]}, Outer={config[3]}')
        ax.axis('off')
    
    plt.tight_layout()
    
    # 保存到项目根目录的outputs文件夹
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "darkness_levels_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.show()


def test_custom_darkness():
    """测试自定义暗度参数"""
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
                image_files.append(images[0])
                break
    
    if not image_files:
        print("No images found in data/raw directory!")
        print("Please run the image download script first:")
        print("  python scripts/download_images.py")
        return
    
    # 读取图片
    test_image = image_files[0]
    img = cv2.imread(str(test_image))
    img = cv2.resize(img, (320, 160))
    
    # 创建拼图
    edges = ('concave', 'convex', 'concave', 'convex')
    puzzle_mask = create_puzzle_piece(piece_size=60, knob_radius=12, edges=edges)
    mask_h, mask_w = puzzle_mask.shape[:2]
    alpha = puzzle_mask[:, :, 3]
    
    x, y = 130, 50
    
    # 自定义参数
    result = apply_gap_lighting(
        img, x, y, alpha, mask_h, mask_w,
        base_darkness=30,         # 基础暗度
        edge_darkness=40,         # 边缘额外暗度
        directional_darkness=10,  # 方向性暗度（左上、右下）
        outer_edge_darkness=6     # 外边缘柔和阴影
    )
    
    # 显示对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # 原图
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.set_title('Original')
    ax1.axis('off')
    
    # 效果图
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    ax2.imshow(result_rgb)
    ax2.set_title('With Custom Darkness Parameters')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Testing different darkness levels...")
    test_darkness_levels()
    
    print("\nTesting custom darkness parameters...")
    test_custom_darkness()