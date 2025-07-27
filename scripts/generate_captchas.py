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
from src.captcha_generator.lighting_effects import apply_gap_lighting, apply_gap_highlighting
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
    base_gray = np.random.randint(180, 220)  # 随机灰度基准
    
    for c in range(3):
        # 为每个通道添加轻微的变化
        channel_variation = np.random.randint(-10, 10)
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


def get_random_level_params(level_min=5, level_max=15):
    """
    获取随机级别的光照参数
    
    Args:
        level_min: 最小级别（默认5）
        level_max: 最大级别（默认15）
    
    Returns:
        tuple: (base, edge, directional) 参数值
    """
    level = random.randint(level_min, level_max)
    # 基于 Level 1-20 的梯度公式计算
    gradient_factor = (level - 1) / 19.0
    base = int(10 + gradient_factor * 40)      # 10-50
    edge = int(20 + gradient_factor * 60)      # 20-80
    directional = int(10 + gradient_factor * 30)  # 10-40
    return base, edge, directional


def check_overlap(x1, y1, w1, h1, x2, y2, w2, h2, min_distance=10):
    """
    检查两个矩形是否重叠或距离太近
    
    Args:
        x1, y1, w1, h1: 第一个矩形的位置和大小
        x2, y2, w2, h2: 第二个矩形的位置和大小
        min_distance: 最小距离要求
    
    Returns:
        True如果重叠或太近，False如果位置合适
    """
    # 扩展矩形边界以包含最小距离
    x1_min = x1 - min_distance
    y1_min = y1 - min_distance
    x1_max = x1 + w1 + min_distance
    y1_max = y1 + h1 + min_distance
    
    x2_min = x2
    y2_min = y2
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    # 检查是否重叠
    return not (x2_max < x1_min or x2_min > x1_max or 
                y2_max < y1_min or y2_min > y1_max)


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
                
                # 判断是否应用柏林噪声（用于抗混淆特性）
                apply_noise = False
                if pos_idx == 0 and shape == shapes[0] and size == sizes[0]:
                    # 第一个组合作为抗混淆特性：应用柏林噪声
                    apply_noise = True
                    noise_percentage = random.uniform(30, 50)  # 30-50%强度
                    puzzle = apply_perlin_noise_to_slider(puzzle, noise_percentage)
                
                # 判断是否旋转缺口（用于抗混淆特性）
                rotation_angle = 0  # 默认不旋转
                gap_highlighted = False  # 记录是否使用了变亮效果
                
                if pos_idx == 1 and shape == shapes[0] and size == sizes[0]:
                    # 第二个组合作为抗混淆特性：旋转缺口
                    rotation_angle = random.uniform(0.3, 0.7)  # 0.3-0.7度
                    # 随机决定旋转方向（顺时针或逆时针）
                    if random.random() < 0.5:
                        rotation_angle = -rotation_angle
                    
                    # 旋转拼图掩码
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
                    
                    # 应用旋转后的缺口光照效果（变暗） - 使用随机 Level 5-15
                    base_darkness, edge_darkness, directional_darkness = get_random_level_params(5, 15)
                    background = apply_gap_lighting(img, new_x, new_y, rotated_alpha, rotated_h, rotated_w,
                                                  base_darkness=base_darkness,
                                                  edge_darkness=edge_darkness,
                                                  directional_darkness=directional_darkness,
                                                  outer_edge_darkness=0)
                    
                    # 计算旋转后的中心坐标
                    bg_center_x = new_x + rotated_w // 2
                    bg_center_y = new_y + rotated_h // 2
                else:
                    # 不旋转，使用原始的缺口
                    # 判断是否应用高光效果（用于抗混淆特性）
                    if pos_idx == 2 and shape == shapes[0] and size == sizes[0]:
                        # 第三个组合作为抗混淆特性：缺口高光效果
                        gap_highlighted = True
                        # 应用缺口高光效果（变亮）- medium强度
                        background = apply_gap_highlighting(img, x, y, alpha, mask_h, mask_w,
                                                          base_lightness=30, edge_lightness=45,
                                                          directional_lightness=20, outer_edge_lightness=15)
                    else:
                        # 应用缺口光照效果（变暗） - 使用随机 Level 5-15
                        base_darkness, edge_darkness, directional_darkness = get_random_level_params(5, 15)
                        background = apply_gap_lighting(img, x, y, alpha, mask_h, mask_w,
                                                      base_darkness=base_darkness,
                                                      edge_darkness=edge_darkness,
                                                      directional_darkness=directional_darkness,
                                                      outer_edge_darkness=0)
                    
                    # 计算中心坐标
                    bg_center_x = x + mask_w // 2
                    bg_center_y = y + mask_h // 2
                
                # 滑块位置
                slider_x = random.randint(0, 10)
                sd_center_x = slider_x + slider_width // 2
                sd_center_y = bg_center_y
                
                # 判断是否生成混淆缺口（用于抗混淆特性）
                confusion_gap_info = None
                if pos_idx == 3 and shape == shapes[0] and size == sizes[0]:
                    # 第四个组合作为抗混淆特性：混淆缺口
                    # 随机选择旋转角度（±10到±30度）
                    angle_magnitude = random.uniform(10, 30)
                    confusion_angle = angle_magnitude if random.random() < 0.5 else -angle_magnitude
                    
                    # 生成一个较小的混淆缺口（比真实滑块小）
                    confusion_size = int(size * random.uniform(0.7, 0.9))  # 原尺寸的70%-90%
                    
                    # 创建混淆缺口的掩码
                    if isinstance(shape, tuple):
                        confusion_puzzle_mask = create_puzzle_piece(
                            piece_size=confusion_size,
                            knob_radius=int(confusion_size * 0.2),
                            edges=shape
                        )
                    else:
                        confusion_puzzle_mask = generate_special_shape(shape, confusion_size)
                    
                    # 旋转混淆缺口掩码
                    confusion_mask = rotate_image(confusion_puzzle_mask, confusion_angle)
                    confusion_h, confusion_w = confusion_mask.shape[:2]
                    confusion_alpha = confusion_mask[:, :, 3]
                    
                    # 尝试找到合适的混淆缺口位置
                    confusion_placed = False
                    max_attempts = 100
                    
                    for _ in range(max_attempts):
                        # 使用智能位置生成策略
                        if random.random() < 0.5:
                            # 50%概率在右侧区域（远离左侧滑块）
                            conf_x = random.randint(max(100, slider_x + slider_width + 20), 
                                                  max(100, min(320 - confusion_w, 250)))
                        else:
                            # 50%概率在其他区域
                            conf_x = random.randint(0, max(0, 320 - confusion_w))
                        conf_y = random.randint(0, max(0, 160 - confusion_h))
                        
                        # 检查是否与真实缺口重叠或太近
                        real_gap_x = bg_center_x - mask_w // 2
                        real_gap_y = bg_center_y - mask_h // 2
                        if not check_overlap(real_gap_x, real_gap_y, mask_w, mask_h, 
                                           conf_x, conf_y, confusion_w, confusion_h, 10):
                            # 检查是否与滑块重叠或太近
                            if not check_overlap(slider_x, y, slider_width, mask_h,
                                               conf_x, conf_y, confusion_w, confusion_h, 10):
                                # 检查是否超出边界
                                if conf_x >= 0 and conf_y >= 0 and conf_x + confusion_w <= 320 and conf_y + confusion_h <= 160:
                                    # 应用混淆缺口
                                    # 混淆缺口必须与真实缺口使用相同的效果（都亮或都暗）
                                    if gap_highlighted:
                                        # 如果真实缺口是变亮的，混淆缺口也要变亮 - 使用随机 Level 5-15
                                        base_lightness, edge_lightness, directional_lightness = get_random_level_params(5, 15)
                                        background = apply_gap_highlighting(
                                            background, conf_x, conf_y, confusion_alpha, confusion_h, confusion_w,
                                            base_lightness=base_lightness,
                                            edge_lightness=edge_lightness,
                                            directional_lightness=directional_lightness,
                                            outer_edge_lightness=0
                                        )
                                    else:
                                        # 如果真实缺口是变暗的，混淆缺口也要变暗 - 使用随机 Level 5-15
                                        base_darkness, edge_darkness, directional_darkness = get_random_level_params(5, 15)
                                        background = apply_gap_lighting(
                                            background, conf_x, conf_y, confusion_alpha, confusion_h, confusion_w,
                                            base_darkness=base_darkness,
                                            edge_darkness=edge_darkness,
                                            directional_darkness=directional_darkness,
                                            outer_edge_darkness=0
                                        )
                                    confusion_placed = True
                                    confusion_gap_info = {
                                        'position': [conf_x + confusion_w // 2, conf_y + confusion_h // 2],
                                        'angle': confusion_angle,
                                        'size': [confusion_w, confusion_h],
                                        'original_size': confusion_size
                                    }
                                    break
                
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
                annotation = {
                    'filename': filename,
                    'bg_center': [bg_center_x, bg_center_y],
                    'sd_center': [sd_center_x, sd_center_y],
                    'shape': str(shape),
                    'size': size,
                    'hash': hash_value,
                    'rotation_angle': rotation_angle,  # 记录旋转角度
                    'gap_highlighted': gap_highlighted,  # 记录是否使用了变亮效果
                    'perlin_noise_applied': apply_noise,  # 记录是否应用了柏林噪声
                    'anti_confusion_type': None  # 记录抗混淆特性类型
                }
                
                # 标记抗混淆特性类型
                if pos_idx == 0 and shape == shapes[0] and size == sizes[0]:
                    annotation['anti_confusion_type'] = 'perlin_noise'
                elif pos_idx == 1 and shape == shapes[0] and size == sizes[0]:
                    annotation['anti_confusion_type'] = 'gap_rotation'
                elif pos_idx == 2 and shape == shapes[0] and size == sizes[0]:
                    annotation['anti_confusion_type'] = 'gap_highlighting'
                elif pos_idx == 3 and shape == shapes[0] and size == sizes[0]:
                    annotation['anti_confusion_type'] = 'confusion_gap'
                
                # 如果有混淆缺口，记录其信息
                if confusion_gap_info is not None:
                    annotation['confusion_gap'] = confusion_gap_info
                
                annotations.append(annotation)
                
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
    # 原始：11形状 × 3尺寸 × 4位置 = 132张
    # 抗混淆特性：4种（柏林噪声、缺口旋转、缺口高光、混淆缺口）使用第一个形状的第一个尺寸的前4个位置
    # 总计：132张（其中前4张带有抗混淆特性）
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
    print(f"Will generate {total_captchas} CAPTCHAs per image set")
    print(f"  - {len(all_shapes)} shapes (5 normal + 6 special)")
    print(f"  - 3 random sizes per image (40-70)")
    print(f"  - 4 positions per shape/size combination")
    print(f"  - 4 anti-confusion features:")
    print(f"    * Perlin noise (30-50% strength)")
    print(f"    * Gap rotation (0.3-0.7 degrees)")
    print(f"    * Gap highlighting (medium intensity)")
    print(f"    * Confusion gap (10-30 degrees, 70-90% of original size)")
    
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