# -*- coding: utf-8 -*-
"""
测试版滑块验证码数据集生成器 - 简化版
直接输出图片到指定目录，不生成标签和元数据
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
import argparse
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import yaml


# 导入混淆系统
from src.captcha_generator.confusion_system.generator import CaptchaGenerator
from src.captcha_generator.confusion_system.strategies import (
    PerlinNoiseConfusion,
    RotationConfusion,
    HighlightConfusion,
    ConfusingGapConfusion,
    HollowCenterConfusion,
    GapEdgeHighlightConfusion
)

# 导入配置类
from src.config.size_confusion_config import SizeConfusionConfig

# 导入几何中心计算工具
from src.captcha_generator.utils.geometric_center import (
    calculate_geometric_center,
    calculate_absolute_geometric_center
)

# 导入尺寸变化
from src.captcha_generator.size_variation import SizeVariation


class TestDatasetConfig:
    """测试数据集配置类 - 从YAML文件加载配置"""
    
    def __init__(self, config_path: str):
        """
        初始化配置
        
        Args:
            config_path: YAML配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 数据集配置
        dataset = self.config['dataset']
        self.INPUT_DIR = dataset['paths']['input_dir']
        self.OUTPUT_DIR = dataset['paths']['output_dir']
        
        # 处理配置
        self.MAX_WORKERS = dataset['processing']['max_workers']
        self.MAX_IMAGES = dataset['processing']['max_images']
        self.RANDOM_SEED = dataset['processing']['random_seed']
        
        # 规模配置
        self.MIN_BACKGROUNDS = dataset['scale']['min_backgrounds']
        self.MAX_BACKGROUNDS = dataset['scale'].get('max_backgrounds', None)  # 新增：最大背景数量限制
        self.CAPTCHAS_PER_IMAGE = dataset['scale']['captchas_per_image']
        
        # 混淆策略配置
        self.CONFUSION_COUNTS = dataset['confusion_counts']
        
        # 形状配置
        self.SPECIAL_SHAPES = dataset['shapes']['special']
        self.NORMAL_SHAPES_COUNT = dataset['shapes']['normal_count']
        
        # 拼图尺寸配置
        self.ALL_PUZZLE_SIZES = dataset['puzzle_sizes']['all_sizes']
        self.SIZES_PER_IMAGE = dataset['puzzle_sizes']['sizes_per_image']
        
        # 网格位置配置
        self.GAP_X_COUNT = dataset['grid_positions']['gap_x_positions']
        self.GAP_Y_COUNT = dataset['grid_positions']['gap_y_positions']
        self.SLIDER_X_COUNT = dataset['grid_positions']['slider_x_positions']
        
        # 保存选项（简化版固定设置）
        self.SAVE_FULL_IMAGE = True
        self.SAVE_COMPONENTS = False
        self.COMPONENT_FORMAT = 'png'
        self.SAVE_MASKS = False


class TestConfusionConfig:
    """测试混淆配置类 - 从YAML文件加载配置"""
    
    def __init__(self, config_path: str):
        """
        初始化配置
        
        Args:
            config_path: YAML配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 获取混淆策略配置
        strategies = self.config['confusion_strategies']
        
        self.PERLIN_NOISE = strategies['perlin_noise']
        self.ROTATION = strategies['rotation']
        self.HIGHLIGHT = strategies['highlight']
        self.CONFUSING_GAP = strategies['confusing_gap']
        self.HOLLOW_CENTER = strategies['hollow_center']
        self.GAP_EDGE_HIGHLIGHT = strategies['gap_edge_highlight']
        self.COMBINED = strategies['combined']
        
        # 光照效果配置
        self.GAP_LIGHTING = self.config.get('lighting_effects', {})
    
    def get_rotation_params(self, rng=None):
        """获取旋转参数"""
        if rng is None:
            rng = np.random.RandomState()
        return {
            'rotation_angle': rng.uniform(
                self.ROTATION['rotation_angle']['min'],
                self.ROTATION['rotation_angle']['max']
            )
        }
    
    def get_perlin_noise_params(self, rng=None):
        """获取Perlin噪声参数"""
        if rng is None:
            rng = np.random.RandomState()
        return {
            'noise_strength': rng.uniform(
                self.PERLIN_NOISE['noise_strength']['min'],
                self.PERLIN_NOISE['noise_strength']['max']
            ),
            'noise_scale': self.PERLIN_NOISE['noise_scale']
        }
    
    def get_highlight_params(self, rng=None):
        """获取高光参数"""
        if rng is None:
            rng = np.random.RandomState()
        return {
            'base_lightness': rng.randint(
                self.HIGHLIGHT['base_lightness']['min'],
                self.HIGHLIGHT['base_lightness']['max'] + 1
            ),
            'edge_lightness': rng.randint(
                self.HIGHLIGHT['edge_lightness']['min'],
                self.HIGHLIGHT['edge_lightness']['max'] + 1
            ),
            'directional_lightness': rng.randint(
                self.HIGHLIGHT['directional_lightness']['min'],
                self.HIGHLIGHT['directional_lightness']['max'] + 1
            ),
            'outer_edge_lightness': self.HIGHLIGHT.get('outer_edge_lightness', 0)
        }
    
    def get_confusing_gap_params(self, rng=None):
        """获取混淆缺口参数"""
        if rng is None:
            rng = np.random.RandomState()
        
        num_gaps_range = self.CONFUSING_GAP['num_confusing_gaps']
        num_gaps = rng.randint(num_gaps_range[0], num_gaps_range[1] + 1)
        
        return {
            'num_confusing_gaps': num_gaps,
            'confusing_type': self.CONFUSING_GAP['confusing_type'],
            'rotation_range': [
                self.CONFUSING_GAP['rotation_range']['min'],
                self.CONFUSING_GAP['rotation_range']['max']
            ],
            'scale_range': [
                self.CONFUSING_GAP['scale_range']['min'],
                self.CONFUSING_GAP['scale_range']['max']
            ],
            'different_y_no_transform_prob': self.CONFUSING_GAP.get('different_y_no_transform_prob', 0.5)
        }
    
    def get_hollow_center_params(self, rng=None):
        """获取空心中心参数"""
        if rng is None:
            rng = np.random.RandomState()
        return {
            'scale': rng.uniform(
                self.HOLLOW_CENTER['scale']['min'],
                self.HOLLOW_CENTER['scale']['max']
            )
        }
    
    def get_gap_edge_highlight_params(self, rng=None):
        """获取缺口边缘高光参数"""
        if rng is None:
            rng = np.random.RandomState()
        return {
            'edge_lightness': rng.randint(
                self.GAP_EDGE_HIGHLIGHT['edge_lightness']['min'],
                self.GAP_EDGE_HIGHLIGHT['edge_lightness']['max'] + 1
            ),
            'edge_width': rng.randint(
                self.GAP_EDGE_HIGHLIGHT['edge_width']['min'],
                self.GAP_EDGE_HIGHLIGHT['edge_width']['max'] + 1
            ),
            'decay_factor': rng.uniform(
                self.GAP_EDGE_HIGHLIGHT['decay_factor']['min'],
                self.GAP_EDGE_HIGHLIGHT['decay_factor']['max']
            )
        }
    
    def get_combined_strategies(self, rng=None):
        """获取组合策略"""
        if rng is None:
            rng = np.random.RandomState()
        
        num_strategies_range = self.COMBINED['num_strategies']
        num_strategies = rng.randint(num_strategies_range[0], num_strategies_range[1] + 1)
        
        available = self.COMBINED['available_strategies']
        selected = rng.choice(available, size=min(num_strategies, len(available)), replace=False)
        
        return selected.tolist()


def select_puzzle_sizes_for_image(
    pic_index: int,
    all_sizes: List[int],
    num_sizes: int = None,
    DatasetConfig = None
) -> List[int]:
    """为每张图片选择固定的puzzle_sizes"""
    if num_sizes is None:
        num_sizes = DatasetConfig.SIZES_PER_IMAGE
    
    rng = np.random.RandomState(pic_index)
    actual_num_sizes = min(num_sizes, len(all_sizes))
    selected_sizes = sorted(rng.choice(all_sizes, size=actual_num_sizes, replace=False).tolist())
    return selected_sizes


def generate_grid_positions(
    puzzle_size: int,
    img_width: int = 320,
    img_height: int = 160,
    gap_x_count: int = 4,
    gap_y_count: int = 3,
    slider_x_count: int = 3
) -> Dict[str, List[int]]:
    """生成固定的网格位置"""
    half_size = puzzle_size // 2
    
    # 计算Y轴网格位置
    y_min = half_size + 10
    y_max = img_height - half_size - 10
    y_grid = np.linspace(y_min, y_max, gap_y_count, dtype=int).tolist()
    
    # 计算滑块X轴网格位置
    slider_x_min = half_size
    slider_x_max = half_size + 10
    slider_x_grid = np.linspace(slider_x_min, slider_x_max, slider_x_count, dtype=int).tolist()
    
    # 计算缺口X轴网格位置
    gap_x_min = slider_x_max + 2*half_size + 10
    gap_x_max = img_width - half_size - 10
    gap_x_grid = np.linspace(gap_x_min, gap_x_max, gap_x_count, dtype=int).tolist()
    
    return {
        'slider_x_grid': slider_x_grid,
        'gap_x_grid': gap_x_grid,
        'y_grid': y_grid
    }


def generate_positions_for_size(
    pic_index: int,
    puzzle_size: int,
    img_width: int = 320,
    img_height: int = 160,
    gap_x_count: int = 4,
    gap_y_count: int = 3,
    slider_x_count: int = 3,
    size_config = None
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """生成所有网格位置组合"""
    # 使用配置中的目标尺寸
    if img_width is None or img_height is None:
        if size_config:
            img_width, img_height = size_config.target_size
        else:
            img_width, img_height = 512, 256
    
    # 生成网格
    grid = generate_grid_positions(
        puzzle_size, img_width, img_height,
        gap_x_count, gap_y_count, slider_x_count
    )
    
    # 生成所有组合
    positions = []
    for slider_x in grid['slider_x_grid']:
        for gap_x in grid['gap_x_grid']:
            for y in grid['y_grid']:
                slider_pos = (slider_x, y)
                gap_pos = (gap_x, y)
                positions.append((slider_pos, gap_pos))
    
    return positions


def get_random_puzzle_shapes(num_shapes: int = 6) -> List[Tuple[str, ...]]:
    """生成随机的普通拼图形状组合"""
    edge_types = ['concave', 'flat', 'convex']
    all_shapes = []
    
    for top in edge_types:
        for right in edge_types:
            for bottom in edge_types:
                for left in edge_types:
                    all_shapes.append((top, right, bottom, left))
    
    return random.sample(all_shapes, num_shapes)


def generate_confusion_plan(pic_index: int, confusion_counts: Dict[str, int]) -> List[str]:
    """生成确定性的混淆策略计划"""
    rng = np.random.RandomState(pic_index)
    
    confusion_plan = []
    for confusion_type, count in confusion_counts.items():
        confusion_plan.extend([confusion_type] * count)
    
    rng.shuffle(confusion_plan)
    return confusion_plan


def create_confusion_strategies(
    strategy_type: str,
    rng: Optional[np.random.RandomState] = None,
    shape: Optional[str] = None,
    hollow_center_remaining: Optional[int] = None,
    ConfusionConfig = None,
    DatasetConfig = None
) -> List:
    """创建混淆策略"""
    if rng is None:
        rng = np.random.RandomState()
    
    strategies = []
    
    # 判断是否为特殊形状
    is_special_shape = False
    if shape is not None and DatasetConfig:
        is_special_shape = shape in DatasetConfig.SPECIAL_SHAPES
    
    if strategy_type == 'none':
        return strategies
    
    elif strategy_type == 'combined':
        selected = ConfusionConfig.get_combined_strategies(rng)
        
        if hollow_center_remaining is not None and hollow_center_remaining <= 0:
            selected = [s for s in selected if s != 'hollow_center']
        
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
                if is_special_shape and (hollow_center_remaining is None or hollow_center_remaining > 0):
                    strategies.append(HollowCenterConfusion(
                        ConfusionConfig.get_hollow_center_params(rng)
                    ))
            elif s == 'gap_edge_highlight':
                strategies.append(GapEdgeHighlightConfusion(
                    ConfusionConfig.get_gap_edge_highlight_params(rng)
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
            if is_special_shape:
                strategies.append(HollowCenterConfusion(
                    ConfusionConfig.get_hollow_center_params(rng)
                ))
        elif strategy_type == 'gap_edge_highlight':
            strategies.append(GapEdgeHighlightConfusion(
                ConfusionConfig.get_gap_edge_highlight_params(rng)
            ))
    
    return strategies


def create_final_image(background: np.ndarray, slider: np.ndarray,
                      slider_pos: Tuple[int, int]) -> np.ndarray:
    """创建最终的验证码图像"""
    final = background.copy()
    
    # 放置滑块
    h, w = slider.shape[:2]
    x, y = slider_pos
    
    # 计算放置区域
    x1 = x - w // 2
    y1 = y - h // 2
    x2 = x1 + w
    y2 = y1 + h
    
    # Alpha混合
    if slider.shape[2] == 4:
        alpha = slider[:, :, 3] / 255.0
        for c in range(3):
            final[y1:y2, x1:x2, c] = (
                final[y1:y2, x1:x2, c] * (1 - alpha) +
                slider[:, :, c] * alpha
            ).astype(np.uint8)
    else:
        final[y1:y2, x1:x2] = slider
    
    return final


def generate_captcha_batch(args: Tuple) -> Tuple[int, List[str]]:
    """
    为单张图片生成一批验证码（简化版）
    
    Returns:
        (generated_count, filenames)
    """
    img_path, output_dir, pic_index, gen_config = args
    
    # 从gen_config中获取配置参数
    DatasetConfig = gen_config['DatasetConfig']
    ConfusionConfig = gen_config['ConfusionConfig']
    size_config = gen_config['size_config']
    
    puzzle_sizes = DatasetConfig.ALL_PUZZLE_SIZES
    sizes_per_image = DatasetConfig.SIZES_PER_IMAGE
    captchas_per_image = DatasetConfig.CAPTCHAS_PER_IMAGE
    confusion_counts = DatasetConfig.CONFUSION_COUNTS
    
    # 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        return 0, []
    
    # 创建尺寸配置和处理器
    size_config_local = SizeConfusionConfig()
    size_processor = SizeVariation(size_config_local)
    
    # 生成随机尺寸并调整图片
    target_size = size_config_local.generate_random_size()
    img, size_info = size_processor.apply_size_variation(img, target_size)
    
    # 设置随机种子
    seed_str = f"{str(img_path)}_{target_size[0]}x{target_size[1]}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 解析配置
    shapes = gen_config['shapes']
    
    # 创建生成器
    generator = CaptchaGenerator()
    
    generated_count = 0
    filenames = []
    
    # 生成样本
    num_samples = captchas_per_image
    
    # 生成混淆计划
    confusion_plan = generate_confusion_plan(pic_index, confusion_counts)
    
    # 为这张图片选择固定的puzzle_sizes
    selected_sizes = select_puzzle_sizes_for_image(
        pic_index, 
        puzzle_sizes,
        sizes_per_image,
        DatasetConfig
    )
    
    # 为每个选中的size动态生成位置列表
    all_positions = {}
    for puzzle_size in selected_sizes:
        positions = generate_positions_for_size(
            pic_index, 
            puzzle_size,
            img_width=size_info['width'], 
            img_height=size_info['height'],
            gap_x_count=DatasetConfig.GAP_X_COUNT,
            gap_y_count=DatasetConfig.GAP_Y_COUNT,
            slider_x_count=DatasetConfig.SLIDER_X_COUNT,
            size_config=size_config
        )
        all_positions[puzzle_size] = positions
    
    # 计算每个size应该生成的数量
    samples_per_size = num_samples // len(selected_sizes)
    remaining_samples = num_samples % len(selected_sizes)
    
    # 构建样本分配计划
    sample_size_plan = []
    for i, size in enumerate(selected_sizes):
        count = samples_per_size
        if i < remaining_samples:
            count += 1
        sample_size_plan.extend([size] * count)
    
    # 打乱顺序
    random.shuffle(sample_size_plan)
    
    # 统计每个尺寸使用的次数
    size_usage_counter = defaultdict(int)
    
    # 追踪hollow_center的使用情况
    hollow_center_used = 0
    hollow_center_quota = confusion_counts.get('hollow_center', 0)
    
    for sample_idx in range(num_samples):
        # 从计划中获取当前样本的size
        size = sample_size_plan[sample_idx]
        
        # 获取混淆策略类型
        confusion_type = confusion_plan[sample_idx]
        
        # 根据混淆策略类型智能选择形状
        if confusion_type == 'hollow_center':
            shape = random.choice(DatasetConfig.SPECIAL_SHAPES)
        elif confusion_type == 'combined':
            temp_rng = np.random.RandomState(seed + sample_idx)
            combined_strategies = ConfusionConfig.get_combined_strategies(temp_rng)
            if 'hollow_center' in combined_strategies:
                shape = random.choice(DatasetConfig.SPECIAL_SHAPES)
            else:
                shape = random.choice(shapes)
        else:
            shape = random.choice(shapes)
        
        # 获取该尺寸对应的位置列表
        positions = all_positions[size]
        
        if not positions:
            print(f"Warning: No valid positions for size {size} at sample {sample_idx}")
            continue
        
        # 获取位置
        pos_idx = size_usage_counter[size] % len(positions)
        size_usage_counter[size] += 1
        slider_pos, gap_pos = positions[pos_idx]
        
        # 计算剩余的hollow_center配额
        hollow_remaining = hollow_center_quota - hollow_center_used
        
        # 创建混淆策略实例
        strategies = create_confusion_strategies(
            confusion_type, rng, 
            shape=shape,
            hollow_center_remaining=hollow_remaining,
            ConfusionConfig=ConfusionConfig,
            DatasetConfig=DatasetConfig
        )
        
        # 更新hollow_center使用计数
        if confusion_type == 'hollow_center':
            hollow_center_used += 1
        elif confusion_type == 'combined':
            for strategy in strategies:
                if strategy.name == 'hollow_center':
                    hollow_center_used += 1
                    break
        
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
            
            # 计算几何中心坐标
            gap_geometric_center = calculate_absolute_geometric_center(
                result.gap_image.image if hasattr(result, 'gap_image') and hasattr(result.gap_image, 'image') else result.slider,
                gap_pos,
                shape
            )
            
            slider_geometric_center = calculate_absolute_geometric_center(
                result.slider,
                slider_pos,
                shape
            )
            
            # 生成唯一文件名
            params_str = f"{gap_geometric_center[0]}{gap_geometric_center[1]}{slider_geometric_center[0]}{slider_geometric_center[1]}{size}"
            file_hash = hashlib.md5(f"{params_str}{sample_idx}".encode()).hexdigest()[:8]
            filename = (f"Pic{pic_index:04d}_Bgx{gap_geometric_center[0]}Bgy{gap_geometric_center[1]}_"
                       f"Sdx{slider_geometric_center[0]}Sdy{slider_geometric_center[1]}_{file_hash}.png")
            
            # 创建最终图像
            final_image = create_final_image(result.background, result.slider, slider_pos)
            
            # 直接保存到输出目录（不创建子目录）
            output_path = output_dir / filename
            cv2.imwrite(str(output_path), final_image)
            
            filenames.append(filename)
            generated_count += 1
            
        except Exception as e:
            print(f"  Failed to generate CAPTCHA {sample_idx+1}/{num_samples}: {type(e).__name__}: {str(e)}")
            continue
    
    return generated_count, filenames


def generate_dataset_parallel(
    input_dir: Path,
    output_dir: Path,
    config_path: str,
    max_workers: int = None,
    max_images: int = None,
    seed: int = None
) -> None:
    """并行生成数据集（简化版）"""
    start_time = datetime.now()
    
    # 加载配置
    DatasetConfig = TestDatasetConfig(config_path)
    ConfusionConfig = TestConfusionConfig(config_path)
    
    # 加载尺寸配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    size_config = config_data.get('size_confusion', {})
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 创建输出目录（直接创建，不创建子目录）
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
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
    
    # 应用配置中的最大背景数量限制
    if DatasetConfig.MAX_BACKGROUNDS and len(image_files) > DatasetConfig.MAX_BACKGROUNDS:
        print(f"Limiting backgrounds to {DatasetConfig.MAX_BACKGROUNDS} (configured in yaml)")
        random.shuffle(image_files)
        image_files = image_files[:DatasetConfig.MAX_BACKGROUNDS]
    
    # 命令行参数覆盖配置文件
    if max_images and len(image_files) > max_images:
        print(f"Further limiting to {max_images} images (command line override)")
        random.shuffle(image_files)
        image_files = image_files[:max_images]
    
    print(f"Total background images: {len(image_files)}")
    
    # 打印配置摘要
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"Config file: {config_path}")
    print(f"Puzzle Sizes: {DatasetConfig.ALL_PUZZLE_SIZES}")
    print(f"Sizes per Image: {DatasetConfig.SIZES_PER_IMAGE}")
    print(f"CAPTCHAs per Image: {DatasetConfig.CAPTCHAS_PER_IMAGE}")
    print(f"Output directory: {output_dir}")
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
            'DatasetConfig': DatasetConfig,
            'ConfusionConfig': ConfusionConfig,
            'size_config': size_config
        }
        
        tasks.append((img_path, output_dir, pic_idx, gen_config))
    
    # 确定进程数
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)
    
    print(f"Using {max_workers} worker processes")
    print(f"CAPTCHAs per image: {DatasetConfig.CAPTCHAS_PER_IMAGE}")
    print(f"Expected total CAPTCHAs: ~{len(tasks) * DatasetConfig.CAPTCHAS_PER_IMAGE}")
    
    # 并行处理
    total_generated = 0
    all_filenames = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {
            executor.submit(generate_captcha_batch, task): task[0]
            for task in tasks
        }
        
        for future in tqdm(as_completed(future_to_img), total=len(tasks),
                          desc="Processing backgrounds"):
            try:
                generated_count, filenames = future.result()
                total_generated += generated_count
                all_filenames.extend(filenames)
                    
            except Exception as e:
                img_path = future_to_img[future]
                print(f"\nError processing {img_path}: {e}")
    
    # 打印简单统计
    end_time = datetime.now()
    print(f"\nGeneration completed!")
    print(f"Total samples generated: {total_generated}")
    print(f"Time taken: {end_time - start_time}")
    print(f"Output directory: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试版验证码数据集生成器')
    parser.add_argument('--config', type=str, 
                       default='captcha_test_config.yaml',
                       help='配置文件路径（默认为当前目录下的captcha_test_config.yaml）')
    parser.add_argument('--max-images', type=int, default=None, 
                       help='限制处理的图片数量（用于测试）')
    parser.add_argument('--test-mode', action='store_true',
                       help='测试模式：默认只处理10张图片')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='覆盖配置文件中的输出目录')
    args = parser.parse_args()
    
    # 获取配置文件路径
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Please create captcha_test_config.yaml in the current directory")
        return
    
    # 加载配置
    DatasetConfig = TestDatasetConfig(str(config_path))
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / DatasetConfig.INPUT_DIR
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / DatasetConfig.OUTPUT_DIR
    
    # 确定最终的 max_images 值
    max_images_to_use = DatasetConfig.MAX_IMAGES
    
    # 如果启用测试模式
    if args.test_mode:
        max_images_to_use = 10
        print("\n" + "="*60)
        print("测试模式已启用")
        print(f"  - 将处理最多 10 张图片")
        print(f"  - 每张图片生成 {DatasetConfig.CAPTCHAS_PER_IMAGE} 个验证码")
        print(f"  - 预计总共生成 {10 * DatasetConfig.CAPTCHAS_PER_IMAGE} 个验证码")
        print("="*60 + "\n")
    
    # 命令行参数具有最高优先级
    if args.max_images is not None:
        max_images_to_use = args.max_images
        if not args.test_mode:
            print(f"\n限制模式：最多处理 {args.max_images} 张图片")
            print(f"预计生成 {args.max_images * DatasetConfig.CAPTCHAS_PER_IMAGE} 个验证码\n")
    
    # 打印配置信息
    print("=" * 60)
    print("Test CAPTCHA Dataset Generation")
    print("=" * 60)
    print(f"Configuration loaded from: {config_path}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {DatasetConfig.MAX_WORKERS or 'auto'}")
    print(f"Max images: {max_images_to_use or 'all'}")
    print(f"Random seed: {DatasetConfig.RANDOM_SEED or 'random'}")
    print("=" * 60)
    
    # 生成数据集
    generate_dataset_parallel(
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=str(config_path),
        max_workers=DatasetConfig.MAX_WORKERS,
        max_images=max_images_to_use,
        seed=DatasetConfig.RANDOM_SEED
    )


if __name__ == "__main__":
    main()