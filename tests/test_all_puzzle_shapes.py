# -*- coding: utf-8 -*-
"""
测试脚本：生成所有拼图形状的2D展示
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from src.captcha_generator.simple_puzzle_generator import (
    create_puzzle_piece, get_all_puzzle_combinations, 
    generate_special_shape, numeric_to_edges
)


def main():
    """生成所有拼图形状展示"""
    # 获取所有81种组合
    all_combinations = get_all_puzzle_combinations()
    
    # 创建图形
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('All 81 Puzzle Shapes + 6 Special Shapes', fontsize=20)
    
    # 网格参数
    cols = 9
    rows = 10  # 9行普通拼图 + 1行特殊形状
    
    # 绘制81种普通拼图
    for idx, edges in enumerate(all_combinations):
        row = idx // cols
        col = idx % cols
        
        # 创建子图
        ax = plt.subplot(rows, cols, idx + 1)
        
        # 生成拼图
        piece = create_puzzle_piece(piece_size=50, knob_radius=10, edges=edges)
        
        # 创建白色背景
        h, w = piece.shape[:2]
        background = np.ones((h, w, 3), dtype=np.uint8) * 240  # 浅灰色背景
        
        # 创建彩色拼图（而不是白色）
        colored_piece = np.zeros((h, w, 3), dtype=np.uint8)
        alpha = piece[:, :, 3]
        
        # 设置拼图颜色为蓝色
        colored_piece[:, :, 0] = 100  # B
        colored_piece[:, :, 1] = 150  # G
        colored_piece[:, :, 2] = 200  # R
        
        # 应用alpha通道
        alpha_3d = alpha[:, :, np.newaxis] / 255.0
        result = colored_piece * alpha_3d + background * (1 - alpha_3d)
        background = result.astype(np.uint8)
        
        # 显示
        ax.imshow(background)
        ax.axis('off')
        
        # 添加编号
        ax.text(0.5, -0.1, f'{idx+1}', transform=ax.transAxes, 
                ha='center', fontsize=8)
    
    # 绘制6种特殊形状
    special_shapes = ['circle', 'square', 'triangle', 'hexagon', 'pentagon', 'star']
    start_idx = 81
    
    for i, shape in enumerate(special_shapes):
        ax = plt.subplot(rows, cols, start_idx + i + 1)
        
        # 生成特殊形状
        shape_img = generate_special_shape(shape, size=60)
        
        # 创建灰色背景
        h, w = shape_img.shape[:2]
        background = np.ones((h, w, 3), dtype=np.uint8) * 240  # 浅灰色背景
        
        # 创建彩色形状（而不是白色）
        colored_shape = np.zeros((h, w, 3), dtype=np.uint8)
        alpha = shape_img[:, :, 3]
        
        # 设置形状颜色为蓝色
        colored_shape[:, :, 0] = 100  # B
        colored_shape[:, :, 1] = 150  # G
        colored_shape[:, :, 2] = 200  # R
        
        # 应用alpha通道
        alpha_3d = alpha[:, :, np.newaxis] / 255.0
        result = colored_shape * alpha_3d + background * (1 - alpha_3d)
        background = result.astype(np.uint8)
        
        # 显示
        ax.imshow(background)
        ax.axis('off')
        
        # 添加标签
        ax.text(0.5, -0.1, shape, transform=ax.transAxes, 
                ha='center', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图片到outputs目录
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "all_puzzle_shapes_2d.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.show()
    
    # 生成示例展示
    create_examples()


def create_examples():
    """创建一些示例拼图展示"""
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle('Example Puzzle Pieces', fontsize=16)
    
    # 示例配置
    examples = [
        ('flat', 'flat', 'flat', 'flat'),      # 普通方块
        ('convex', 'convex', 'convex', 'convex'),  # 全凸
        ('concave', 'concave', 'concave', 'concave'),  # 全凹
        ('flat', 'convex', 'flat', 'concave'),  # 混合1
        ('convex', 'flat', 'concave', 'flat'),  # 混合2
        ('concave', 'convex', 'concave', 'convex'),  # 混合3
    ]
    
    for i, (ax, edges) in enumerate(zip(axes.flat, examples)):
        piece = create_puzzle_piece(piece_size=80, knob_radius=16, edges=edges)
        
        # 创建白色背景
        h, w = piece.shape[:2]
        background = np.ones((h, w, 3), dtype=np.uint8) * 240  # 浅灰色背景
        
        # 创建彩色拼图（而不是白色）
        colored_piece = np.zeros((h, w, 3), dtype=np.uint8)
        alpha = piece[:, :, 3]
        
        # 设置拼图颜色为蓝色
        colored_piece[:, :, 0] = 100  # B
        colored_piece[:, :, 1] = 150  # G
        colored_piece[:, :, 2] = 200  # R
        
        # 应用alpha通道
        alpha_3d = alpha[:, :, np.newaxis] / 255.0
        result = colored_piece * alpha_3d + background * (1 - alpha_3d)
        background = result.astype(np.uint8)
        
        ax.imshow(background)
        ax.axis('off')
        ax.set_title(f'({edges[0][:3]}, {edges[1][:3]}, {edges[2][:3]}, {edges[3][:3]})', 
                     fontsize=10)
    
    plt.tight_layout()
    
    # 保存到outputs目录
    output_dir = Path(__file__).parent.parent / "outputs"
    output_path = output_dir / "example_puzzle_pieces.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved examples to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()