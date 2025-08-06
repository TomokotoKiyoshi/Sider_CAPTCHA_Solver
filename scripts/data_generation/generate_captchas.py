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
import argparse
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# 导入混淆系统
from src.captcha_generator.confusion_system.generator import CaptchaGenerator
from src.captcha_generator.confusion_system.strategies import (
    PerlinNoiseConfusion,
    RotationConfusion,
    HighlightConfusion,
    ConfusingGapConfusion,
    HollowCenterConfusion
)


class CaptchaDatasetConfig:
    """数据集生成配置"""
    # 位置采样范围
    SLIDER_X_RANGE = (15, 35)     # 滑块x范围
    SLIDER_Y_RANGE = (30, 130)    # 滑块y范围
    
    # 数据集规模
    MIN_BACKGROUNDS = 2000        # 最少背景图数量
    MAX_SAMPLES_PER_BG = 100       # 每个背景最多样本数
    
    
    # 混淆策略概率分布
    CONFUSION_DISTRIBUTION = {
        'none': 0.2,      # 20% 无混淆（基准）
        'single': 0.5,    # 50% 单一混淆
        'combined': 0.3   # 30% 组合混淆
    }
    
    # 形状配置
    SPECIAL_SHAPES = ['circle', 'square', 'triangle', 'hexagon']
    NORMAL_SHAPES_COUNT = 6  # 普通拼图形状数量
    PUZZLE_SIZES = list(range(30, 62, 2)) # 拼图大小范围（30-60，步长2）


def generate_continuous_positions(
    puzzle_size: int,
    config: CaptchaDatasetConfig = CaptchaDatasetConfig()
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    生成连续分布的位置（而非离散点）
    
    Args:
        puzzle_size: 拼图大小
        config: 配置对象
        
    Returns:
        (slider_pos, gap_pos)
    """
    half_size = puzzle_size // 2
    
    # 根据CaptchaGenerator的验证逻辑，滑块x必须在 [half_size, half_size + 10] 范围内
    slider_x_min = half_size
    slider_x_max = half_size + 10
    slider_x = np.random.randint(slider_x_min, slider_x_max + 1)
    
    # 滑块中心y坐标必须 >= half_size（确保上边不超界）
    # 滑块中心y坐标必须 <= 160 - half_size（确保下边不超界）
    slider_y_min = max(config.SLIDER_Y_RANGE[0], half_size)
    slider_y_max = min(config.SLIDER_Y_RANGE[1], 160 - half_size)
    slider_y = np.random.randint(slider_y_min, slider_y_max + 1)
    
    # 生成缺口位置
    # 缺口必须在滑块右边界之后，且不能超出图像边界
    gap_x_min = slider_x + half_size + 10  # 滑块右边界 + 10像素间隔
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


def create_confusion_strategies(
    strategy_type: str,
    config: Dict[str, Any],
    rng: Optional[np.random.RandomState] = None
) -> List:
    """
    创建混淆策略
    
    Args:
        strategy_type: 策略类型
        config: 策略配置
        rng: 随机数生成器（用于可重现性）
        
    Returns:
        策略实例列表
    """
    if rng is None:
        rng = np.random.RandomState()
    
    strategies = []
    
    if strategy_type == 'none':
        return strategies
    
    elif strategy_type == 'perlin_noise':
        strategies.append(PerlinNoiseConfusion({
            'noise_strength': config.get('noise_strength', rng.uniform(0.4, 0.8)),
            'noise_scale': config.get('noise_scale', 0.1)
        }))
    
    elif strategy_type == 'rotation':
        angle_range = config.get('angle_range', (0.5, 1.5))
        # 从范围中随机选择一个角度
        rotation_angle = rng.uniform(angle_range[0], angle_range[1])
        strategies.append(RotationConfusion({
            'rotation_angle': rotation_angle
        }))
    
    elif strategy_type == 'highlight':
        strategies.append(HighlightConfusion({
            'base_lightness': config.get('base_lightness', rng.randint(20, 40)),
            'edge_lightness': config.get('edge_lightness', rng.randint(40, 60)),
            'directional_lightness': config.get('directional_lightness', rng.randint(15, 30)),
            'outer_edge_lightness': 0
        }))
    
    elif strategy_type == 'confusing_gap':
        strategies.append(ConfusingGapConfusion({
            'num_confusing_gaps': config.get('num_confusing_gaps', int(rng.choice([1, 2]))),
            'confusing_type': config.get('confusing_type', 'mixed'),
            'rotation_range': config.get('rotation_range', (10, 30)),
            'scale_range': config.get('scale_range', (0.8, 1.2))
        }))
    
    elif strategy_type == 'hollow_center':
        strategies.append(HollowCenterConfusion({
            'scale': config.get('scale', rng.uniform(0.3, 0.5))
        }))
    
    elif strategy_type == 'single':
        # 随机选择一种策略
        available = ['perlin_noise', 'rotation', 'highlight', 'hollow_center', 'confusing_gap']
        selected = rng.choice(available)
        strategies.extend(create_confusion_strategies(selected, {}, rng))
    
    elif strategy_type == 'combined':
        # 组合2-3种策略
        available = ['perlin_noise', 'rotation', 'highlight', 'hollow_center', 'confusing_gap']
        num_strategies = rng.choice([2, 3])
        selected = rng.choice(available, num_strategies, replace=False)
        for s in selected:
            strategies.extend(create_confusion_strategies(s, {}, rng))
    
    return strategies


def generate_captcha_batch(args: Tuple) -> Tuple[List[Dict], Dict, str, List[Dict]]:
    """
    为单张图片生成一批验证码
    
    Returns:
        (annotations, stats, background_hash, errors)
    """
    img_path, output_dir, pic_index, gen_config = args
    
    # 设置随机种子（基于图片路径）
    seed = int(hashlib.md5(str(img_path).encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 读取并调整图片
    img = cv2.imread(str(img_path))
    if img is None:
        return [], {}, "", []
    
    img = cv2.resize(img, (320, 160))
    
    # 计算背景哈希（用于数据集划分）
    img_hash = hashlib.md5(img.tobytes()).hexdigest()
    
    # 解析配置
    dataset_config = gen_config['dataset_config']
    shapes = gen_config['shapes']
    
    # 创建生成器
    generator = CaptchaGenerator()
    
    annotations = []
    stats = defaultdict(int)
    errors = []  # 收集错误信息
    
    # 生成样本（控制每个背景的样本数）
    num_samples = dataset_config.MAX_SAMPLES_PER_BG
    
    for sample_idx in range(num_samples):
        # 随机选择形状和大小
        shape = random.choice(shapes)
        size = random.choice(dataset_config.PUZZLE_SIZES)
        
        # 生成连续位置
        slider_pos, gap_pos = generate_continuous_positions(size, dataset_config)
        
        # 决定混淆策略类型
        confusion_type = np.random.choice(
            list(dataset_config.CONFUSION_DISTRIBUTION.keys()),
            p=list(dataset_config.CONFUSION_DISTRIBUTION.values())
        )
        
        # 创建混淆策略
        strategies = create_confusion_strategies(confusion_type, {}, rng)
        
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
            
            # 保存图片
            output_path = output_dir / filename
            
            # 创建最终图像（背景+滑块区域）
            final_image = create_final_image(result.background, result.slider, slider_pos)
            cv2.imwrite(str(output_path), final_image)
            
            # 记录标注
            annotation = {
                'filename': filename,
                'background_hash': img_hash,
                'bg_center': list(gap_pos),
                'sd_center': list(slider_pos),
                'shape': str(shape),
                'size': size,
                'confusion_type': confusion_type,
                'confusion_details': result.confusion_type,
                'hash': file_hash,
                'metadata': result.metadata
            }
            
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
                'sample_idx': sample_idx,
                'img_path': str(img_path),
                'pic_index': pic_index,
                'shape': str(shape),
                'size': size,
                'slider_pos': list(slider_pos),
                'gap_pos': list(gap_pos),
                'confusion_type': confusion_type,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            errors.append(error_info)
            continue
    
    return annotations, dict(stats), img_hash, errors


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
    seed: int = None
) -> None:
    """并行生成数据集"""
    start_time = datetime.now()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(input_dir.rglob(ext)))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    # 检查背景图数量
    if len(image_files) < CaptchaDatasetConfig.MIN_BACKGROUNDS:
        print(f"Warning: Only {len(image_files)} background images found. "
              f"Recommended minimum: {CaptchaDatasetConfig.MIN_BACKGROUNDS}")
    
    # 限制图片数量
    if max_images and len(image_files) > max_images:
        random.shuffle(image_files)
        image_files = image_files[:max_images]
    
    print(f"Total background images: {len(image_files)}")
    
    # 准备配置
    dataset_config = CaptchaDatasetConfig()
    
    # 生成形状列表
    normal_shapes = get_random_puzzle_shapes(dataset_config.NORMAL_SHAPES_COUNT)
    all_shapes = dataset_config.SPECIAL_SHAPES + normal_shapes
    
    # 准备任务
    tasks = []
    for i, img_path in enumerate(image_files):
        pic_idx = i + 1
        
        gen_config = {
            'dataset_config': dataset_config,
            'shapes': all_shapes
        }
        
        tasks.append((img_path, output_dir, pic_idx, gen_config))
    
    # 确定进程数
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)
    
    print(f"Using {max_workers} worker processes")
    print(f"Maximum samples per background: {dataset_config.MAX_SAMPLES_PER_BG}")
    print(f"Expected total samples: ~{len(tasks) * dataset_config.MAX_SAMPLES_PER_BG}")
    
    # 并行处理
    all_annotations = []
    total_stats = defaultdict(int)
    background_hashes = set()
    all_errors = []  # 收集所有错误
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {
            executor.submit(generate_captcha_batch, task): task[0]
            for task in tasks
        }
        
        for future in tqdm(as_completed(future_to_img), total=len(tasks),
                          desc="Processing backgrounds"):
            try:
                annotations, stats, bg_hash, errors = future.result()
                all_annotations.extend(annotations)
                background_hashes.add(bg_hash)
                all_errors.extend(errors)  # 收集错误
                
                # 合并统计
                for key, count in stats.items():
                    total_stats[key] += count
                    
            except Exception as e:
                img_path = future_to_img[future]
                print(f"\nError processing {img_path}: {e}")
    
    # 保存标注
    all_annotations_path = output_dir / 'all_annotations.json'
    with open(all_annotations_path, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    
    # 生成统计报告
    end_time = datetime.now()
    report = {
        'generation_time': str(end_time - start_time),
        'total_backgrounds': len(background_hashes),
        'total_samples': len(all_annotations),
        'total_errors': len(all_errors),
        'error_rate': f"{len(all_errors) / (len(tasks) * dataset_config.MAX_SAMPLES_PER_BG) * 100:.2f}%",
        'statistics': dict(total_stats),
        'config': {
            'slider_x_range': dataset_config.SLIDER_X_RANGE,
            'puzzle_sizes': dataset_config.PUZZLE_SIZES,
            'confusion_distribution': dataset_config.CONFUSION_DISTRIBUTION
        }
    }
    
    report_path = output_dir / 'generation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 保存错误日志
    if all_errors:
        error_log_path = output_dir / 'generation_errors.json'
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_errors': len(all_errors),
                'error_rate': f"{len(all_errors) / (len(tasks) * dataset_config.MAX_SAMPLES_PER_BG) * 100:.2f}%",
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
    print(f"[OK] Position sampling: Slider X{dataset_config.SLIDER_X_RANGE}, "
          f"Y{dataset_config.SLIDER_Y_RANGE}")
    print(f"[OK] Gap position: Dynamically calculated based on slider position")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Generate industrial-grade CAPTCHA dataset with continuous sampling'
    )
    parser.add_argument('--input-dir', type=str, default='data/raw',
                        help='Input directory containing background images')
    parser.add_argument('--output-dir', type=str, default='data/captchas',
                        help='Output directory for generated CAPTCHAs')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of background images to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    
    # 生成数据集
    generate_dataset_parallel(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=args.workers,
        max_images=args.max_images,
        seed=args.seed
    )


if __name__ == "__main__":
    main()