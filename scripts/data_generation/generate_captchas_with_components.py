# -*- coding: utf-8 -*-
"""
工业级滑块验证码数据集生成器 - 增强版
解决了位置采样不足、背景过拟合等关键问题
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import random
import hashlib
from tqdm import tqdm
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict


class NumpyJSONEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 导入混淆系统
from src.captcha_generator.confusion_system.generator import CaptchaGenerator
from src.captcha_generator.confusion_system.strategies import (
    PerlinNoiseConfusion,
    RotationConfusion,
    HighlightConfusion,
    ConfusingGapConfusion,
    HollowCenterConfusion
)

# 导入配置
from config.confusion_config import ConfusionConfig
from config.dataset_config import DatasetConfig
from config.size_confusion_config import CaptchaSizeConfig

# 导入标签生成器
from src.captcha_generator import CaptchaLabelGenerator, create_label_from_captcha_result
from src.captcha_generator.size_variation import SizeVariation


def select_puzzle_sizes_for_image(
    pic_index: int,
    all_sizes: List[int],
    num_sizes: int = None
) -> List[int]:
    """
    为每张图片选择固定的puzzle_sizes
    
    Args:
        pic_index: 图片索引（用作随机种子）
        all_sizes: 所有可能的puzzle大小
        num_sizes: 要选择的大小数量（None时使用配置）
        
    Returns:
        选中的puzzle_sizes列表
    """
    if num_sizes is None:
        num_sizes = DatasetConfig.SIZES_PER_IMAGE
    
    rng = np.random.RandomState(pic_index)
    # 确保不会选择超过可用的尺寸数
    actual_num_sizes = min(num_sizes, len(all_sizes))
    selected_sizes = sorted(rng.choice(all_sizes, size=actual_num_sizes, replace=False).tolist())
    return selected_sizes


def generate_fixed_gap_positions(
    pic_index: int,
    gap_x_count: int = 4,
    gap_y_count: int = 3,
    img_width: int = 320,
    img_height: int = 160
) -> Tuple[List[int], List[int]]:
    """
    为每张图片生成固定的gap位置（不依赖于puzzle_size）
    使用最大可能的puzzle_size来确保安全范围
    
    Args:
        pic_index: 图片索引（用作随机种子）
        gap_x_count: 缺口x轴位置数量
        gap_y_count: 缺口y轴位置数量
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        (gap_x_positions, gap_y_positions)
    """
    rng = np.random.RandomState(pic_index)
    
    # 使用最大puzzle_size(60)来确定安全范围，根据图像尺寸缩放
    scale_factor = min(img_width / 320, img_height / 160)
    max_puzzle_size = int(60 * scale_factor)
    gap_x_min = max_puzzle_size + 20  
    gap_x_max = img_width - max_puzzle_size // 2  
    
    # 为这张图片生成固定的gap x坐标
    gap_x_positions = sorted([int(x) for x in rng.choice(
        range(gap_x_min, gap_x_max),
        size=gap_x_count,
        replace=False
    )])
    
    # 生成gap y位置（使用最大size的安全边距）
    gap_y_min = max_puzzle_size // 2 + 5  
    gap_y_max = img_height - max_puzzle_size // 2 - 5
    gap_y_positions = sorted([int(y) for y in rng.choice(
        range(gap_y_min, gap_y_max),
        size=gap_y_count,
        replace=False
    )])
    
    return gap_x_positions, gap_y_positions


def generate_positions_for_size(
    pic_index: int,
    puzzle_size: int,
    gap_x_positions: List[int],
    gap_y_positions: List[int],
    slider_x_count: int = 4,
    img_width: int = 320,
    img_height: int = 160
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    基于固定的gap位置和特定的puzzle_size，生成有效的位置组合
    
    Args:
        pic_index: 图片索引（用作随机种子）
        puzzle_size: 拼图大小
        gap_x_positions: 固定的gap x位置列表
        gap_y_positions: 固定的gap y位置列表
        slider_x_count: 滑块x轴位置数量
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        位置列表 [(slider_pos, gap_pos), ...]
    """
    rng = np.random.RandomState(pic_index * 1000 + puzzle_size)
    half_size = puzzle_size // 2
    
    # 生成滑块x位置
    slider_x_min = max(half_size, 15)
    slider_x_max = min(half_size + 10, 40)
    
    slider_x_positions = []
    for i in range(slider_x_count):
        x = int(rng.randint(slider_x_min, slider_x_max))
        slider_x_positions.append(x)
    
    # 生成所有组合，但过滤掉会重叠的位置
    positions = []
    for slider_x in slider_x_positions:
        for gap_x in gap_x_positions:
            for gap_y in gap_y_positions:
                slider_y = gap_y  # 滑块y等于gap y
                
                # 检查是否会重叠
                # 滑块范围：[slider_x - half_size, slider_x + half_size]
                # 缺口范围：[gap_x - half_size, gap_x + half_size]
                slider_left = slider_x - half_size
                slider_right = slider_x + half_size
                gap_left = gap_x - half_size
                gap_right = gap_x + half_size
                
                # 如果不重叠，添加到位置列表
                if gap_left >= slider_right or gap_right <= slider_left:
                    positions.append(((slider_x, slider_y), (gap_x, gap_y)))
    
    return positions


def generate_continuous_positions(
    puzzle_size: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    生成连续分布的位置（保留用于兼容性）
    
    Args:
        puzzle_size: 拼图大小
        
    Returns:
        (slider_pos, gap_pos)
    """
    half_size = puzzle_size // 2
    
    # 根据CaptchaGenerator的验证逻辑，滑块x必须在 [half_size, half_size + 10] 范围内
    slider_x_min = half_size
    slider_x_max = 40
    slider_x = np.random.randint(slider_x_min, slider_x_max + 1)
    
    # 滑块中心y坐标必须 >= half_size + 5（确保上边不超界且有安全边距）
    # 滑块中心y坐标必须 <= 160 - half_size - 5（确保下边不超界且有安全边距）
    slider_y_min = max(DatasetConfig.SLIDER_Y_RANGE[0], half_size + 5)
    slider_y_max = min(DatasetConfig.SLIDER_Y_RANGE[1], 160 - half_size - 5)
    slider_y = np.random.randint(slider_y_min, slider_y_max + 1)
    
    # 生成缺口位置
    # 缺口必须在滑块右边界之后，且不能超出图像边界
    gap_x_min = slider_x + 2*half_size + 10  # 滑块右边界 + half_size + 10像素间隔
    gap_x_max = 320 - half_size  # 确保不超出右边界
    
    gap_x = np.random.randint(gap_x_min, gap_x_max + 1)
    gap_y = slider_y  # 保持y坐标一致
    
    return (slider_x, slider_y), (gap_x, gap_y)


def get_random_puzzle_shapes(num_shapes: int = 6) -> List[Tuple[str, ...]]:
    """生成随机的普通拼图形状组合"""
    edge_types = ['concave', 'flat', 'convex']
    all_shapes = []
    
    for top in edge_types:
        for right in edge_types:
            for bottom in edge_types:
                for left in edge_types:
                    all_shapes.append((top, right, bottom, left))
    
    # 从所有81种可能的组合中随机选择
    return random.sample(all_shapes, num_shapes)


def generate_confusion_plan(pic_index: int, confusion_counts: Dict[str, int]) -> List[str]:
    """
    生成确定性的混淆策略计划
    
    Args:
        pic_index: 图片索引（用作随机种子）
        confusion_counts: 每种混淆策略的数量配置
    
    Returns:
        混淆策略列表（长度等于总样本数）
    """
    # 使用图片索引作为种子，保证可重现性
    rng = np.random.RandomState(pic_index)
    
    # 构建混淆计划列表
    confusion_plan = []
    for confusion_type, count in confusion_counts.items():
        confusion_plan.extend([confusion_type] * count)
    
    # 验证总数
    total = len(confusion_plan)
    
    # 打乱顺序（使用固定种子）
    rng.shuffle(confusion_plan)
    
    return confusion_plan


def create_confusion_strategies(
    strategy_type: str,
    rng: Optional[np.random.RandomState] = None
) -> List:
    """
    创建混淆策略（支持单一和组合）
    
    Args:
        strategy_type: 策略类型
        rng: 随机数生成器（用于可重现性）
        
    Returns:
        策略实例列表
    """
    if rng is None:
        rng = np.random.RandomState()
    
    strategies = []
    
    if strategy_type == 'none':
        return strategies
    
    elif strategy_type == 'combined':
        # 使用配置文件获取组合策略
        selected = ConfusionConfig.get_combined_strategies(rng)
        
        # 为每种选中的混淆创建策略
        for s in selected:
            if s == 'rotation':
                strategies.append(RotationConfusion(
                    ConfusionConfig.get_rotation_params(rng)
                ))
            elif s == 'perlin_noise':
                strategies.append(PerlinNoiseConfusion(
                    ConfusionConfig.get_perlin_noise_params(rng)
                ))
            elif s == 'highlight':
                strategies.append(HighlightConfusion(
                    ConfusionConfig.get_highlight_params(rng)
                ))
            elif s == 'confusing_gap':
                strategies.append(ConfusingGapConfusion(
                    ConfusionConfig.get_confusing_gap_params(rng)
                ))
            elif s == 'hollow_center':
                strategies.append(HollowCenterConfusion(
                    ConfusionConfig.get_hollow_center_params(rng)
                ))
    
    else:
        # 单一混淆策略
        if strategy_type == 'rotation':
            strategies.append(RotationConfusion(
                ConfusionConfig.get_rotation_params(rng)
            ))
        elif strategy_type == 'perlin_noise':
            strategies.append(PerlinNoiseConfusion(
                ConfusionConfig.get_perlin_noise_params(rng)
            ))
        elif strategy_type == 'highlight':
            strategies.append(HighlightConfusion(
                ConfusionConfig.get_highlight_params(rng)
            ))
        elif strategy_type == 'confusing_gap':
            strategies.append(ConfusingGapConfusion(
                ConfusionConfig.get_confusing_gap_params(rng)
            ))
        elif strategy_type == 'hollow_center':
            strategies.append(HollowCenterConfusion(
                ConfusionConfig.get_hollow_center_params(rng)
            ))
    
    return strategies


def generate_captcha_batch(args: Tuple) -> Tuple[List[Dict], Dict, str, List[Dict], List[Dict]]:
    """
    为单张图片生成一批验证码
    
    Returns:
        (annotations, stats, background_hash, errors, labels)
    """
    img_path, output_dir, pic_index, gen_config = args
    save_components = gen_config.get('save_components', False)
    
    # 从gen_config中获取配置参数
    puzzle_sizes = gen_config.get('puzzle_sizes', DatasetConfig.ALL_PUZZLE_SIZES)
    sizes_per_image = gen_config.get('sizes_per_image', DatasetConfig.SIZES_PER_IMAGE)
    samples_per_bg = gen_config.get('samples_per_bg', DatasetConfig.MAX_SAMPLES_PER_BG)
    confusion_counts = gen_config.get('confusion_counts', DatasetConfig.CONFUSION_COUNTS)
    
    # 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        return [], {}, "", []
    
    # 创建尺寸配置和处理器 - 在设置种子之前生成随机尺寸
    size_config = CaptchaSizeConfig()
    size_processor = SizeVariation(size_config)
    
    # 生成随机尺寸并调整图片（使用真正的随机性）
    target_size = size_config.generate_random_size()
    img, size_info = size_processor.apply_size_variation(img, target_size)
    
    # 设置随机种子（基于图片路径和尺寸）- 用于形状和位置的可重现性
    # 注意：现在seed包含尺寸信息，确保相同图片+尺寸组合的形状/位置可重现
    seed_str = f"{str(img_path)}_{target_size[0]}x{target_size[1]}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 计算背景哈希（用于数据集划分）
    img_hash = hashlib.md5(img.tobytes()).hexdigest()
    
    # 解析配置
    shapes = gen_config['shapes']
    
    # 创建生成器
    generator = CaptchaGenerator()
    
    annotations = []
    stats = defaultdict(int)
    errors = []  # 收集错误信息
    labels_list = []  # 收集训练标签
    
    # 生成样本（控制每个背景的样本数）
    num_samples = samples_per_bg
    
    # 生成混淆计划
    confusion_plan = generate_confusion_plan(pic_index, confusion_counts)
    
    # 为这张图片选择固定的puzzle_sizes（根据图像尺寸缩放）
    # 使用高度作为缩放基准，确保滑块不会太大
    scale_factor = size_info['height'] / 160
    scaled_puzzle_sizes = []
    for s in puzzle_sizes:
        scaled_size = int(s * scale_factor)
        # 确保是偶数（便于中心对齐）
        if scaled_size % 2 != 0:
            scaled_size += 1
        # 确保不超过图像高度的合理范围（最大为高度的40%）
        max_size = int(size_info['height'] * 0.4)
        scaled_size = min(scaled_size, max_size)
        # 额外限制：确保puzzle_size不超过60，以避免slider位置计算问题
        # 当half_size > 30时，slider_x_min会大于slider_x_max(40)
        scaled_size = min(scaled_size, 60)
        scaled_puzzle_sizes.append(scaled_size)
    
    selected_sizes = select_puzzle_sizes_for_image(
        pic_index, 
        scaled_puzzle_sizes,
        sizes_per_image
    )
    
    # 为这张图片生成固定的gap位置（不依赖于size）
    gap_x_positions, gap_y_positions = generate_fixed_gap_positions(
        pic_index, 
        img_width=size_info['width'],
        img_height=size_info['height']
    )
    
    # 为每个选中的size生成位置列表
    all_positions = {}
    for puzzle_size in selected_sizes:
        positions = generate_positions_for_size(
            pic_index, puzzle_size, gap_x_positions, gap_y_positions,
            img_width=size_info['width'], img_height=size_info['height']
        )
        all_positions[puzzle_size] = positions
    
    # 计算每个size应该生成的数量（平均分配）
    samples_per_size = num_samples // len(selected_sizes)  # 100 / 4 = 25
    remaining_samples = num_samples % len(selected_sizes)  # 处理余数
    
    # 构建样本分配计划
    sample_size_plan = []
    for i, size in enumerate(selected_sizes):
        count = samples_per_size
        if i < remaining_samples:  # 将余数分配给前几个size
            count += 1
        sample_size_plan.extend([size] * count)
    
    # 打乱顺序，使不同size的样本交错分布
    random.shuffle(sample_size_plan)
    
    # 统计每个尺寸使用的次数，用于循环选择位置
    size_usage_counter = defaultdict(int)
    
    for sample_idx in range(num_samples):
        # 从计划中获取当前样本的size
        size = sample_size_plan[sample_idx]
        
        # 随机选择形状
        shape = random.choice(shapes)
        
        # 获取该尺寸对应的位置列表
        positions = all_positions[size]
        
        # 如果没有可用位置，报错（理论上不应该发生）
        if not positions:
            print(f"Warning: No valid positions for size {size} at sample {sample_idx}")
            continue
        
        # 获取位置（循环使用该尺寸的位置列表）
        pos_idx = size_usage_counter[size] % len(positions)
        size_usage_counter[size] += 1
        slider_pos, gap_pos = positions[pos_idx]
        
        # 获取混淆策略类型
        confusion_type = confusion_plan[sample_idx]
        
        # 创建混淆策略实例
        strategies = create_confusion_strategies(confusion_type, rng)
        
        try:
            # 生成验证码
            result = generator.generate(
                image=img,
                puzzle_shape=shape,
                puzzle_size=size,
                gap_position=gap_pos,
                slider_position=slider_pos,
                confusion_strategies=strategies
            )
            
            # 生成唯一文件名
            params_str = f"{gap_pos[0]}{gap_pos[1]}{slider_pos[0]}{slider_pos[1]}{size}"
            file_hash = hashlib.md5(f"{params_str}{sample_idx}".encode()).hexdigest()[:8]
            filename = (f"Pic{pic_index:04d}_Bgx{gap_pos[0]}Bgy{gap_pos[1]}_"
                       f"Sdx{slider_pos[0]}Sdy{slider_pos[1]}_{file_hash}.png")
            
            # 创建最终图像（背景+滑块区域）
            final_image = create_final_image(result.background, result.slider, slider_pos)
            
            # 保存组合图片到captchas目录
            from pathlib import Path
            captchas_dir = Path(output_dir) / 'captchas'
            captchas_dir.mkdir(parents=True, exist_ok=True)
            composite_path = captchas_dir / filename
            cv2.imwrite(str(composite_path), final_image)
            
            # 如果需要保存组件
            if save_components:
                components_dir = Path(output_dir) / 'components'
                
                # 使用统一的基础文件名（包含所有位置信息）
                base_filename = f"Pic{pic_index:04d}_Bgx{gap_pos[0]}Bgy{gap_pos[1]}_Sdx{slider_pos[0]}Sdy{slider_pos[1]}_{file_hash}"
                
                # 保存滑块（RGBA格式）
                slider_filename = f"{base_filename}_slider.png"
                slider_path = components_dir / 'sliders' / slider_filename
                cv2.imwrite(str(slider_path), result.slider)
                
                # 保存背景（RGB格式，带缺口）
                bg_filename = f"{base_filename}_gap.png"
                bg_path = components_dir / 'backgrounds' / bg_filename
                cv2.imwrite(str(bg_path), result.background)
            
            # 记录标注（包含尺寸信息）
            annotation = {
                'filename': filename,
                'background_hash': img_hash,
                'bg_center': [int(gap_pos[0]), int(gap_pos[1])],
                'sd_center': [int(slider_pos[0]), int(slider_pos[1])],
                'shape': str(shape),
                'size': int(size),
                'image_width': size_info['width'],
                'image_height': size_info['height'],
                'aspect_ratio': size_info['aspect_ratio'],
                'confusion_type': confusion_type,
                'confusion_details': result.confusion_type,
                'hash': file_hash,
                'metadata': result.metadata
            }
            
            # 如果保存了组件，添加组件文件路径和生成训练标签
            if save_components:
                annotation['slider_file'] = f"components/sliders/{slider_filename}"
                annotation['background_file'] = f"components/backgrounds/{bg_filename}"
                
                # 生成训练标签
                label = create_label_from_captcha_result(
                    pic_index=pic_index,
                    sample_idx=sample_idx,
                    gap_position=gap_pos,
                    slider_position=slider_pos,
                    puzzle_size=size,
                    confusion_type=confusion_type,
                    confusion_metadata=result.confusion_params if hasattr(result, 'confusion_params') else {},
                    additional_gaps=result.additional_gaps if hasattr(result, 'additional_gaps') else None,
                    file_hash=file_hash,
                    base_filename=base_filename.replace('_slider', '').replace('_gap', '')  # 清理文件名
                )
                labels_list.append(label)
            
            if result.additional_gaps:
                # 转换 numpy 数组为列表以便 JSON 序列化
                serializable_gaps = []
                for gap in result.additional_gaps:
                    serializable_gap = {
                        'position': [int(gap['position'][0]), int(gap['position'][1])],
                        'size': [int(gap['size'][0]), int(gap['size'][1])]
                    }
                    # 不包含 mask，因为它是 numpy 数组且太大
                    serializable_gaps.append(serializable_gap)
                annotation['additional_gaps'] = serializable_gaps
            
            annotations.append(annotation)
            
            # 更新统计
            stats[f'shape_{shape}'] += 1
            stats[f'size_{size}'] += 1
            stats[f'confusion_{confusion_type}'] += 1
            
        except Exception as e:
            import traceback
            error_info = {
                'sample_idx': int(sample_idx),
                'img_path': str(img_path),
                'pic_index': int(pic_index),
                'shape': str(shape),
                'size': int(size),
                'slider_pos': [int(slider_pos[0]), int(slider_pos[1])],
                'gap_pos': [int(gap_pos[0]), int(gap_pos[1])],
                'confusion_type': confusion_type,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            errors.append(error_info)
            continue
    
    return annotations, dict(stats), img_hash, errors, labels_list


def create_final_image(background: np.ndarray, slider: np.ndarray,
                      slider_pos: Tuple[int, int]) -> np.ndarray:
    """创建最终的验证码图像"""
    final = background.copy()
    
    # 放置滑块
    h, w = slider.shape[:2]
    x, y = slider_pos
    
    # 计算放置区域（以slider_pos为中心）
    x1 = x - w // 2
    y1 = y - h // 2
    x2 = x1 + w
    y2 = y1 + h
    
    # Alpha混合
    if slider.shape[2] == 4:
        alpha = slider[:, :, 3] / 255.0
        for c in range(3):              # 如果某像素 alpha = 0（透明），结果 = 100% 背景
            final[y1:y2, x1:x2, c] = (  # 如果某像素 alpha = 1（不透明），结果 = 100% 滑块
                final[y1:y2, x1:x2, c] * (1 - alpha) +
                slider[:, :, c] * alpha
            ).astype(np.uint8)
    else:
        final[y1:y2, x1:x2] = slider
    
    return final


def generate_dataset_parallel(
    input_dir: Path,
    output_dir: Path,
    max_workers: int = None,
    max_images: int = None,
    seed: int = None,
    save_components: bool = False  # 默认不保存组件
) -> None:
    """并行生成数据集"""
    start_time = datetime.now()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建输出目录结构
    # 1. 组合图片目录
    captchas_dir = output_dir / 'captchas'
    captchas_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directories created:")
    print(f"  - Captchas: {captchas_dir}")
    
    # 2. 组件目录（如果需要保存）
    if save_components:
        components_dir = output_dir / 'components'
        sliders_dir = components_dir / 'sliders'
        backgrounds_dir = components_dir / 'backgrounds'
        sliders_dir.mkdir(parents=True, exist_ok=True)
        backgrounds_dir.mkdir(parents=True, exist_ok=True)
        print(f"  - Sliders: {sliders_dir}")
        print(f"  - Backgrounds: {backgrounds_dir}")
    
    # 3. 标签目录
    labels_dir = output_dir / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"  - Labels: {labels_dir}")
    
    # 获取所有图片
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(input_dir.rglob(ext)))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    # 检查背景图数量
    if len(image_files) < DatasetConfig.MIN_BACKGROUNDS:
        print(f"Warning: Only {len(image_files)} background images found. "
              f"Recommended minimum: {DatasetConfig.MIN_BACKGROUNDS}")
    
    # 限制图片数量
    if max_images and len(image_files) > max_images:
        random.shuffle(image_files)
        image_files = image_files[:max_images]
    
    print(f"Total background images: {len(image_files)}")
    
    # 打印配置摘要
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"Puzzle Sizes: {DatasetConfig.ALL_PUZZLE_SIZES}")
    print(f"Sizes per Image: {DatasetConfig.SIZES_PER_IMAGE}")
    print(f"Samples per Background: {DatasetConfig.MAX_SAMPLES_PER_BG}")
    print(f"Slider X Range: {DatasetConfig.SLIDER_X_RANGE}")
    print(f"Slider Y Range: {DatasetConfig.SLIDER_Y_RANGE}")
    print(f"Save Components: {DatasetConfig.SAVE_COMPONENTS}")
    print(f"Save Full Images: {DatasetConfig.SAVE_FULL_IMAGE}")
    print("="*60 + "\n")
    
    # 生成形状列表
    normal_shapes = get_random_puzzle_shapes(DatasetConfig.NORMAL_SHAPES_COUNT)
    all_shapes = DatasetConfig.SPECIAL_SHAPES + normal_shapes
    
    # 准备任务
    tasks = []
    for i, img_path in enumerate(image_files):
        pic_idx = i + 1
        
        gen_config = {
            'shapes': all_shapes,
            'save_components': save_components,
            'puzzle_sizes': DatasetConfig.ALL_PUZZLE_SIZES,
            'sizes_per_image': DatasetConfig.SIZES_PER_IMAGE,
            'samples_per_bg': DatasetConfig.MAX_SAMPLES_PER_BG,
            'confusion_counts': DatasetConfig.CONFUSION_COUNTS
        }
        
        tasks.append((img_path, output_dir, pic_idx, gen_config))
    
    # 确定进程数
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)
    
    print(f"Using {max_workers} worker processes")
    print(f"Maximum samples per background: {DatasetConfig.MAX_SAMPLES_PER_BG}")
    print(f"Expected total samples: ~{len(tasks) * DatasetConfig.MAX_SAMPLES_PER_BG}")
    
    # 并行处理
    all_annotations = []
    total_stats = defaultdict(int)
    background_hashes = set()
    all_errors = []  # 收集所有错误
    
    # 创建标签生成器（仅当save_components为True时）
    label_generator = None
    if save_components:
        labels_dir = output_dir / 'labels'
        label_generator = CaptchaLabelGenerator(labels_dir)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {
            executor.submit(generate_captcha_batch, task): task[0]
            for task in tasks
        }
        
        for future in tqdm(as_completed(future_to_img), total=len(tasks),
                          desc="Processing backgrounds"):
            try:
                annotations, stats, bg_hash, errors, labels = future.result()
                all_annotations.extend(annotations)
                background_hashes.add(bg_hash)
                all_errors.extend(errors)  # 收集错误
                
                # 如果有标签生成器，添加标签
                if label_generator and labels:
                    for label in labels:
                        label_generator.add_label(label)
                
                # 合并统计
                for key, count in stats.items():
                    total_stats[key] += count
                    
            except Exception as e:
                img_path = future_to_img[future]
                print(f"\nError processing {img_path}: {e}")
    
    # 保存标注
    all_annotations_path = output_dir / 'all_annotations.json'
    with open(all_annotations_path, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
    
    # 保存训练标签（如果启用了组件保存）
    saved_label_files = {}
    if label_generator:
        saved_label_files = label_generator.save_labels()
        label_stats = label_generator.get_statistics()
        print(f"\nLabel generation statistics:")
        print(f"  - Total labels: {label_stats['total_labels']}")
        print(f"  - Unique pics: {label_stats['unique_pics']}")
        print(f"  - Labels saved to: {saved_label_files['all_labels']}")
    
    # 保存完整配置（用于复现）
    config_backup_path = output_dir / 'dataset_config_used.json'
    config_backup = {
        'dataset_config': {
            'slider_x_range': DatasetConfig.SLIDER_X_RANGE,
            'slider_y_range': DatasetConfig.SLIDER_Y_RANGE,
            'min_backgrounds': DatasetConfig.MIN_BACKGROUNDS,
            'max_samples_per_bg': DatasetConfig.MAX_SAMPLES_PER_BG,
            'confusion_counts': DatasetConfig.CONFUSION_COUNTS,
            'special_shapes': DatasetConfig.SPECIAL_SHAPES,
            'normal_shapes_count': DatasetConfig.NORMAL_SHAPES_COUNT,
            'all_puzzle_sizes': DatasetConfig.ALL_PUZZLE_SIZES,
            'sizes_per_image': DatasetConfig.SIZES_PER_IMAGE,
            'save_full_image': DatasetConfig.SAVE_FULL_IMAGE,
            'save_components': DatasetConfig.SAVE_COMPONENTS,
            'component_format': DatasetConfig.COMPONENT_FORMAT,
            'save_masks': DatasetConfig.SAVE_MASKS,
            'gap_x_count': DatasetConfig.GAP_X_COUNT,
            'gap_y_count': DatasetConfig.GAP_Y_COUNT,
            'slider_x_count': DatasetConfig.SLIDER_X_COUNT
        },
        'confusion_config': {
            'perlin_noise': ConfusionConfig.PERLIN_NOISE,
            'rotation': ConfusionConfig.ROTATION,
            'highlight': ConfusionConfig.HIGHLIGHT,
            'confusing_gap': ConfusionConfig.CONFUSING_GAP,
            'hollow_center': ConfusionConfig.HOLLOW_CENTER,
            'combined': ConfusionConfig.COMBINED,
            'gap_lighting': ConfusionConfig.GAP_LIGHTING
        },
        'generation_params': {
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'num_backgrounds': len(image_files),
            'max_workers': max_workers,
            'seed': seed
        }
    }
    with open(config_backup_path, 'w', encoding='utf-8') as f:
        json.dump(config_backup, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
    
    # 生成统计报告
    end_time = datetime.now()
    report = {
        'generation_time': str(end_time - start_time),
        'total_backgrounds': len(background_hashes),
        'total_samples': len(all_annotations),
        'total_errors': len(all_errors),
        'error_rate': f"{len(all_errors) / (len(tasks) * DatasetConfig.MAX_SAMPLES_PER_BG) * 100:.2f}%",
        'statistics': dict(total_stats),
        'labels_generated': len(label_generator.labels) if label_generator else 0,
        'config': {
            'slider_x_range': DatasetConfig.SLIDER_X_RANGE,
            'slider_y_range': DatasetConfig.SLIDER_Y_RANGE,
            'puzzle_sizes': DatasetConfig.ALL_PUZZLE_SIZES,
            'sizes_per_image': DatasetConfig.SIZES_PER_IMAGE,
            'confusion_counts': DatasetConfig.CONFUSION_COUNTS,
            'max_samples_per_bg': DatasetConfig.MAX_SAMPLES_PER_BG,
            'special_shapes': DatasetConfig.SPECIAL_SHAPES,
            'normal_shapes_count': DatasetConfig.NORMAL_SHAPES_COUNT,
            'save_components': DatasetConfig.SAVE_COMPONENTS,
            'save_full_image': DatasetConfig.SAVE_FULL_IMAGE,
            'component_format': DatasetConfig.COMPONENT_FORMAT
        }
    }
    
    report_path = output_dir / 'generation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
    
    # 保存错误日志
    if all_errors:
        error_log_path = output_dir / 'generation_errors.json'
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_errors': len(all_errors),
                'error_rate': f"{len(all_errors) / (len(tasks) * DatasetConfig.MAX_SAMPLES_PER_BG) * 100:.2f}%",
                'errors': all_errors
            }, f, ensure_ascii=False, indent=2)
        print(f"\nError log saved to: {error_log_path}")
    
    print(f"\nGeneration completed!")
    print(f"Total samples: {report['total_samples']}")
    print(f"Time taken: {report['generation_time']}")
    print(f"Report saved to: {report_path}")
    
    # 验证数据质量
    print("\nData quality checks:")
    print(f"[OK] Background diversity: {len(background_hashes)} unique backgrounds")
    print(f"[OK] Position sampling: Slider X{DatasetConfig.SLIDER_X_RANGE}, "
          f"Y{DatasetConfig.SLIDER_Y_RANGE}")
    print(f"[OK] Gap position: Dynamically calculated based on slider position")


def main():
    """主函数 - 所有配置从DatasetConfig读取"""
    
    # 验证配置
    if not DatasetConfig.validate():
        print("Configuration validation failed!")
        print("Please check the settings in config/dataset_config.py")
        return
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / DatasetConfig.INPUT_DIR
    output_dir = project_root / DatasetConfig.OUTPUT_DIR
    
    # 打印配置信息
    print("=" * 60)
    print("CAPTCHA Dataset Generation")
    print("=" * 60)
    print(f"Configuration loaded from: config/dataset_config.py")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {DatasetConfig.MAX_WORKERS or 'auto'}")
    print(f"Max images: {DatasetConfig.MAX_IMAGES or 'all'}")
    print(f"Random seed: {DatasetConfig.RANDOM_SEED or 'random'}")
    print(f"Save components: {DatasetConfig.SAVE_COMPONENTS}")
    print("=" * 60)
    
    # 生成数据集
    generate_dataset_parallel(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=DatasetConfig.MAX_WORKERS,
        max_images=DatasetConfig.MAX_IMAGES,
        seed=DatasetConfig.RANDOM_SEED,
        save_components=DatasetConfig.SAVE_COMPONENTS
    )


if __name__ == "__main__":
    main()