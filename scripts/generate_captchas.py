# -*- coding: utf-8 -*-
"""
批量生成滑块验证码数据集 - 优化的并行版本
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import random
import hashlib
from tqdm import tqdm
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from src.captcha_generator.simple_puzzle_generator import create_puzzle_piece, generate_special_shape
from src.captcha_generator.lighting_effects import apply_gap_lighting
from src.captcha_generator.slider_effects import apply_slider_lighting, create_slider_frame, composite_slider


def generate_captchas_for_image(args):
    """为单张图片生成所有验证码（进程池函数）"""
    img_path, output_dir, pic_index, shapes_info, sizes, slider_width = args
    
    # 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        return [], {}
    
    img = cv2.resize(img, (320, 160))
    
    annotations = []
    stats = {'shapes_used': {}, 'sizes_used': {}}
    
    # 解析形状信息
    shapes = []
    for shape_str in shapes_info:
        if shape_str.startswith('(') and shape_str.endswith(')'):
            # 普通形状 - 转换字符串回元组
            shape = eval(shape_str)
            shapes.append(shape)
        else:
            # 特殊形状
            shapes.append(shape_str)
    
    # 为每种形状、每种大小生成4个位置
    for shape in shapes:
        for size in sizes:
            for pos_idx in range(4):
                # 创建拼图
                if isinstance(shape, tuple):
                    puzzle_mask = create_puzzle_piece(
                        piece_size=size,
                        knob_radius=int(size * 0.2),
                        edges=shape
                    )
                else:
                    # 特殊形状
                    puzzle_mask = generate_special_shape(shape, size)
                
                mask_h, mask_w = puzzle_mask.shape[:2]
                
                # 生成随机位置
                min_x = slider_width + 10
                max_x = 320 - mask_w
                max_y = 160 - mask_h
                
                x = random.randint(min_x, max_x)
                y = random.randint(0, max_y)
                
                # 提取拼图块
                puzzle = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
                img_region = img[y:y+mask_h, x:x+mask_w]
                
                alpha = puzzle_mask[:, :, 3]
                for c in range(3):
                    puzzle[:, :, c] = img_region[:, :, c]
                puzzle[:, :, 3] = alpha
                
                # 应用滑块光照
                puzzle = apply_slider_lighting(
                    puzzle,
                    edge_highlight=80,
                    directional_highlight=30,
                    edge_width=5,
                    decay_factor=2.0
                )
                
                # 创建背景
                background = apply_gap_lighting(img, x, y, alpha, mask_h, mask_w)
                
                # 计算中心坐标
                bg_center_x = x + mask_w // 2
                bg_center_y = y + mask_h // 2
                
                # 滑块位置
                slider_x = random.randint(0, 10)
                sd_center_x = slider_x + slider_width // 2
                sd_center_y = bg_center_y
                
                # 创建最终图片
                slider_frame = create_slider_frame(slider_width, 160)
                final_image = composite_slider(background, puzzle, (sd_center_x, sd_center_y), slider_frame)
                
                # 生成文件名
                positions = f"{bg_center_x}{bg_center_y}{sd_center_x}{sd_center_y}"
                hash_value = hashlib.md5(positions.encode()).hexdigest()[:8]
                filename = f"Pic{pic_index:04d}_Bgx{bg_center_x}Bgy{bg_center_y}_Sdx{sd_center_x}Sdy{sd_center_y}_{hash_value}.png"
                
                # 保存图片
                output_path = output_dir / filename
                cv2.imwrite(str(output_path), final_image)
                
                # 记录标注
                annotations.append({
                    'filename': filename,
                    'bg_center': [bg_center_x, bg_center_y],
                    'sd_center': [sd_center_x, sd_center_y],
                    'shape': str(shape),
                    'size': size,
                    'hash': hash_value
                })
                
                # 更新统计
                shape_key = str(shape)
                stats['shapes_used'][shape_key] = stats['shapes_used'].get(shape_key, 0) + 1
                stats['sizes_used'][size] = stats['sizes_used'].get(size, 0) + 1
    
    return annotations, stats


def generate_dataset_parallel(input_dir, output_dir, max_workers=None, 
                            max_images=None, selected_subdirs=None):
    """并行生成数据集
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        max_workers: 最大工作进程数
        max_images: 最大处理图片数（None表示处理所有）
        selected_subdirs: 要处理的子目录列表（None表示处理所有）
    """
    start_time = datetime.now()
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片（包括子文件夹）
    image_files = []
    subdirs = []
    
    # 首先检查是否有子目录
    for item in input_dir.iterdir():
        if item.is_dir():
            # 如果指定了子目录，只处理指定的
            if selected_subdirs is None or item.name in selected_subdirs:
                subdirs.append(item)
    
    # 如果有子目录，从子目录中收集图片
    if subdirs:
        print(f"Found {len(subdirs)} subdirectories:")
        for subdir in subdirs:
            print(f"  - {subdir.name}")
            subdir_images = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                subdir_images.extend(list(subdir.glob(ext)))
            if subdir_images:
                print(f"    Found {len(subdir_images)} images")
                image_files.extend(subdir_images)
    else:
        # 否则直接从当前目录获取图片
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(list(input_dir.glob(ext)))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    # 如果指定了最大图片数，进行限制
    if max_images and len(image_files) > max_images:
        # 随机选择图片以确保多样性
        random.shuffle(image_files)
        image_files = image_files[:max_images]
        print(f"\nLimited to {max_images} images (randomly selected)")
    
    print(f"\nTotal images to process: {len(image_files)}")
    
    # 生成形状组合
    edge_types = ['concave', 'flat', 'convex']
    normal_shapes = []
    for top in edge_types:
        for right in edge_types:
            for bottom in edge_types:
                for left in edge_types:
                    normal_shapes.append((top, right, bottom, left))
    
    # 随机选择5种普通形状
    selected_normal = random.sample(normal_shapes, 5)
    
    # 6种特殊形状
    special_shapes = ['circle', 'square', 'triangle', 'hexagon', 'pentagon', 'star']
    
    # 组合所有形状（5种普通+6种特殊=11种）
    all_shapes = [str(s) for s in selected_normal] + special_shapes
    
    # 这里只是打印信息，实际的尺寸会在每张图片处理时随机生成
    print(f"\nWill generate random sizes between 40 and 70 for each image")
    slider_width = 60
    
    # 计算每张图片生成的验证码数量
    captchas_per_image = len(all_shapes) * 3 * 4  # 11*3*4=132张
    total_captchas = len(image_files) * captchas_per_image
    
    # 准备任务
    tasks = []
    for i, img_path in enumerate(image_files):
        pic_idx = i + 1  # Pic编号从1开始
        # 为每张图片生成独立的3个随机尺寸
        img_sizes = sorted(random.sample(range(40, 71), 3))
        print(f"  Image {pic_idx}: sizes = {img_sizes}")
        tasks.append((img_path, output_dir, pic_idx, all_shapes, img_sizes, slider_width))
    
    # 确定进程数
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)
    
    print(f"\nUsing {max_workers} worker processes")
    print(f"Will generate {total_captchas} CAPTCHAs")
    print(f"  - {len(all_shapes)} shapes (5 normal + 6 special)")
    print(f"  - 3 random sizes per image (40-70)")
    print(f"  - 4 positions per shape/size combination")
    
    # 并行处理
    all_annotations = []
    total_stats = {'shapes_used': {}, 'sizes_used': {}, 'total_images': 0}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_img = {executor.submit(generate_captchas_for_image, task): task[0] 
                        for task in tasks}
        
        # 处理完成的任务
        for future in tqdm(as_completed(future_to_img), total=len(tasks), desc="Processing images"):
            try:
                annotations, stats = future.result()
                all_annotations.extend(annotations)
                
                # 合并统计
                for shape, count in stats['shapes_used'].items():
                    total_stats['shapes_used'][shape] = total_stats['shapes_used'].get(shape, 0) + count
                for size, count in stats['sizes_used'].items():
                    total_stats['sizes_used'][size] = total_stats['sizes_used'].get(size, 0) + count
                    
            except Exception as e:
                img_path = future_to_img[future]
                print(f"\nError processing {img_path}: {e}")
    
    # 保存标注
    annotations_path = output_dir / 'annotations.json'
    with open(annotations_path, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    
    # 更新统计
    end_time = datetime.now()
    total_stats['total_images'] = len(all_annotations)
    total_stats['generation_time'] = str(end_time - start_time)
    
    # 保存统计
    stats_path = output_dir / 'generation_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(total_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\nGeneration completed!")
    print(f"Total CAPTCHAs generated: {total_stats['total_images']}")
    print(f"Time taken: {total_stats['generation_time']}")
    print(f"Annotations saved to: {annotations_path}")
    print(f"Statistics saved to: {stats_path}")


def main():
    """主函数"""
    import argparse
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    parser = argparse.ArgumentParser(description='Generate CAPTCHA dataset (README compliant)')
    parser.add_argument('--input-dir', type=str, 
                        default=str(project_root / 'data' / 'raw'),
                        help='Input directory containing raw images')
    parser.add_argument('--output-dir', type=str, 
                        default=str(project_root / 'data' / 'captchas'),
                        help='Output directory for generated CAPTCHAs')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')
    parser.add_argument('--subdirs', nargs='*', default=None,
                        help='Specific subdirectories to process (default: all)')
    
    args = parser.parse_args()
    
    # 生成数据集
    generate_dataset_parallel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.workers,
        max_images=args.max_images,
        selected_subdirs=args.subdirs
    )


if __name__ == "__main__":
    main()