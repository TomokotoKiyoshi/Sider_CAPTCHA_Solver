#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分割执行脚本 - 使用新的配置系统
按原始图片ID分割，避免数据泄漏
"""
import sys
import json
from pathlib import Path
from glob import glob
import argparse
import random

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.data_split_config import DataSplitConfig


def split_dataset_with_annotations(annotations, config):
    """
    根据标注信息分割数据集
    
    Args:
        annotations: 标注列表
        config: 配置对象
        
    Returns:
        分割结果字典
    """
    # 按图片ID分组
    image_groups = {}
    for ann in annotations:
        # 提取图片ID (如 Pic0001)
        pic_id = ann['filename'].split('_')[0]
        
        if pic_id not in image_groups:
            image_groups[pic_id] = []
        image_groups[pic_id].append(ann)
    
    # 获取所有唯一的图片ID
    unique_pic_ids = list(image_groups.keys())
    print(f"Found {len(unique_pic_ids)} unique source images")
    
    # 设置随机种子并打乱
    random.seed(config.seed)
    if config.shuffle:
        random.shuffle(unique_pic_ids)
    
    # 计算分割点
    n_pics = len(unique_pic_ids)
    train_end = int(n_pics * config.train_ratio)
    val_end = train_end + int(n_pics * config.val_ratio)
    
    # 分割图片ID
    train_pic_ids = unique_pic_ids[:train_end]
    val_pic_ids = unique_pic_ids[train_end:val_end]
    test_pic_ids = unique_pic_ids[val_end:]
    
    # 收集每个集合的标注
    train_annotations = []
    val_annotations = []
    test_annotations = []
    
    for pic_id in train_pic_ids:
        train_annotations.extend(image_groups[pic_id])
    
    for pic_id in val_pic_ids:
        val_annotations.extend(image_groups[pic_id])
    
    for pic_id in test_pic_ids:
        test_annotations.extend(image_groups[pic_id])
    
    # 打印统计信息
    print(f"\nDataset split results:")
    print(f"Train set: {len(train_pic_ids):4d} images -> {len(train_annotations):6d} samples")
    print(f"Val set:   {len(val_pic_ids):4d} images -> {len(val_annotations):6d} samples")
    print(f"Test set:  {len(test_pic_ids):4d} images -> {len(test_annotations):6d} samples")
    print(f"Total:     {len(unique_pic_ids):4d} images -> {len(annotations):6d} samples")
    
    # 验证没有数据泄漏
    train_set = set(train_pic_ids)
    val_set = set(val_pic_ids)
    test_set = set(test_pic_ids)
    
    if train_set & val_set:
        raise ValueError("Train and validation sets overlap!")
    if train_set & test_set:
        raise ValueError("Train and test sets overlap!")
    if val_set & test_set:
        raise ValueError("Validation and test sets overlap!")
    
    print("[OK] No dataset overlap, no data leakage risk")
    
    return {
        "train": {
            "annotations": train_annotations,
            "captchas": [ann['filename'] for ann in train_annotations],
            "count": len(train_annotations)
        },
        "val": {
            "annotations": val_annotations,
            "captchas": [ann['filename'] for ann in val_annotations],
            "count": len(val_annotations)
        },
        "test": {
            "annotations": test_annotations,
            "captchas": [ann['filename'] for ann in test_annotations],
            "count": len(test_annotations)
        },
        "pic_ids": {
            "train": train_pic_ids,
            "val": val_pic_ids,
            "test": test_pic_ids
        }
    }


def main():
    """执行数据集分割 - 使用config中的默认配置"""
    parser = argparse.ArgumentParser(
        description='Split CAPTCHA dataset by image ID to avoid data leakage'
    )
    
    # 创建配置（使用默认值）
    config = DataSplitConfig()
    
    # 只提供覆盖默认值的选项
    parser.add_argument('--annotations-file', type=str, 
                       default='data/metadata/all_annotations.json',
                       help='Annotations JSON file')
    parser.add_argument('--input-dir', type=str, 
                       default=config.source_dir,
                       help=f'Input directory (default: {config.source_dir})')
    parser.add_argument('--output-file', type=str,
                       default=config.output_file,
                       help=f'Output JSON file (default: {config.output_file})')
    parser.add_argument('--train-ratio', type=float, default=None,
                       help=f'Override train ratio (config: {config.train_ratio})')
    parser.add_argument('--val-ratio', type=float, default=None,
                       help=f'Override val ratio (config: {config.val_ratio})')
    parser.add_argument('--test-ratio', type=float, default=None,
                       help=f'Override test ratio (config: {config.test_ratio})')
    parser.add_argument('--seed', type=int, default=None,
                       help=f'Override random seed (config: {config.seed})')
    
    args = parser.parse_args()
    
    # 只在用户指定时覆盖配置
    if args.train_ratio is not None:
        config.train_ratio = args.train_ratio
    if args.val_ratio is not None:
        config.val_ratio = args.val_ratio
    if args.test_ratio is not None:
        config.test_ratio = args.test_ratio
    if args.seed is not None:
        config.seed = args.seed
    
    # 验证比例
    total = config.train_ratio + config.val_ratio + config.test_ratio
    if abs(total - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0 (current: {total})")
        return 1
    
    print("=" * 60)
    print("CAPTCHA Dataset Splitter")
    print("=" * 60)
    print(f"Using configuration from: config/data_split_config.py")
    
    # 加载标注文件
    annotations_path = Path(project_root) / args.annotations_file
    if not annotations_path.exists():
        print(f"Error: Annotations file not found: {annotations_path}")
        return 1
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"Loaded {len(annotations)} annotations from {annotations_path}")
    
    # 获取图片文件列表
    input_dir = Path(project_root) / args.input_dir
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    # 从标注中获取文件列表
    file_list = [str(input_dir / ann['filename']) for ann in annotations]
    unique_files = list(set(file_list))
    
    print(f"Input directory: {input_dir}")
    print(f"Found {len(unique_files)} unique PNG files")
    print(f"Split ratios: train={config.train_ratio}, val={config.val_ratio}, test={config.test_ratio}")
    print(f"Random seed: {config.seed}")
    print("-" * 60)
    
    # 执行分割（基于标注）
    try:
        split_result = split_dataset_with_annotations(annotations, config)
        
        # 保存结果
        output_file = Path(project_root) / args.output_file
        config.save_split_result(split_result, str(output_file))
        
        print("\n" + "=" * 60)
        print("SUCCESS: Dataset split completed!")
        print("=" * 60)
        print(f"Results saved to: {output_file}")
        
        # 打印一些示例
        print("\nSample files from each split:")
        for split_name in ['train', 'val', 'test']:
            if split_result[split_name] and split_result[split_name]['captchas']:
                captcha_samples = split_result[split_name]['captchas'][:3]
                print(f"\n{split_name.upper()} (first 3 captchas):")
                for sample in captcha_samples:
                    print(f"  - {Path(sample).name}")
                print(f"  (Each CAPTCHA has corresponding slider and background components)")
        
        return 0
        
    except Exception as e:
        print(f"\nError during split: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())