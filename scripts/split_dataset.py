#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本 - 简化版
将生成的验证码数据集划分为训练集和测试集
"""
import sys
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigLoader

def main():
    """主函数"""
    print("=" * 60)
    print("CAPTCHA Dataset Splitter")
    print("=" * 60)
    
    # 加载配置
    config_loader = ConfigLoader()
    
    # 获取路径配置
    input_dir = Path(project_root) / config_loader.get('captcha_config.data_split.paths.input_dir', 'data/generated')
    train_dir = Path(project_root) / config_loader.get('captcha_config.data_split.paths.train_dir', 'data/train')
    test_dir = Path(project_root) / config_loader.get('captcha_config.data_split.paths.test_dir', 'data/test')
    
    # 获取划分比例
    train_ratio = config_loader.get('captcha_config.data_split.split_ratio.train', 0.9)
    test_ratio = config_loader.get('captcha_config.data_split.split_ratio.test', 0.1)
    
    print(f"Input directory: {input_dir}")
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Split ratio - Train: {train_ratio:.1%}, Test: {test_ratio:.1%}")
    print("-" * 60)
    
    # 检查输入目录
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please run generate_captchas_with_components.py first")
        return 1
    
    # 加载标注文件
    annotations_file = input_dir / 'metadata' / 'all_annotations.json'
    if not annotations_file.exists():
        print(f"Error: Annotations file not found: {annotations_file}")
        return 1
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"Loaded {len(annotations)} annotations")
    
    # 按图片ID分组（避免数据泄漏）
    image_groups = defaultdict(list)
    for ann in annotations:
        # 提取图片ID (如 Pic0001)
        pic_id = ann['filename'].split('_')[0]
        image_groups[pic_id].append(ann)
    
    # 获取所有唯一的图片ID并打乱
    unique_pic_ids = list(image_groups.keys())
    print(f"Found {len(unique_pic_ids)} unique source images")
    
    random.seed(42)  # 固定种子以确保可重现
    random.shuffle(unique_pic_ids)
    
    # 计算分割点
    n_pics = len(unique_pic_ids)
    train_end = int(n_pics * train_ratio)
    
    # 分割图片ID
    train_pic_ids = unique_pic_ids[:train_end]
    test_pic_ids = unique_pic_ids[train_end:]
    
    # 收集每个集合的标注
    train_annotations = []
    test_annotations = []
    
    for pic_id in train_pic_ids:
        train_annotations.extend(image_groups[pic_id])
    
    for pic_id in test_pic_ids:
        test_annotations.extend(image_groups[pic_id])
    
    # 打印统计信息
    print("\nDataset split results:")
    print(f"Train set: {len(train_pic_ids):4d} images -> {len(train_annotations):6d} samples")
    print(f"Test set:  {len(test_pic_ids):4d} images -> {len(test_annotations):6d} samples")
    print(f"Total:     {len(unique_pic_ids):4d} images -> {len(annotations):6d} samples")
    
    # 创建输出目录
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制文件到对应目录
    print("\nCopying files...")
    
    # 复制训练集
    train_count = 0
    for ann in train_annotations:
        src = input_dir / ann['filename']
        dst = train_dir / ann['filename']
        if src.exists():
            shutil.copy2(src, dst)
            train_count += 1
            if train_count % 1000 == 0:
                print(f"  Copied {train_count}/{len(train_annotations)} training files...")
    
    # 复制测试集
    test_count = 0
    for ann in test_annotations:
        src = input_dir / ann['filename']
        dst = test_dir / ann['filename']
        if src.exists():
            shutil.copy2(src, dst)
            test_count += 1
            if test_count % 1000 == 0:
                print(f"  Copied {test_count}/{len(test_annotations)} test files...")
    
    print(f"\nCopied {train_count} files to train directory")
    print(f"Copied {test_count} files to test directory")
    
    # 保存标注文件
    train_ann_file = train_dir / 'annotations.json'
    test_ann_file = test_dir / 'annotations.json'
    
    with open(train_ann_file, 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open(test_ann_file, 'w') as f:
        json.dump(test_annotations, f, indent=2)
    
    print(f"\nSaved training annotations to: {train_ann_file}")
    print(f"Saved test annotations to: {test_ann_file}")
    
    # 保存分割信息
    split_info = {
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "train_images": len(train_pic_ids),
        "test_images": len(test_pic_ids),
        "train_samples": len(train_annotations),
        "test_samples": len(test_annotations),
        "total_images": len(unique_pic_ids),
        "total_samples": len(annotations),
        "train_pic_ids": train_pic_ids,
        "test_pic_ids": test_pic_ids
    }
    
    split_info_file = Path(project_root) / 'data' / 'split_info.json'
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nSaved split information to: {split_info_file}")
    
    print("\n" + "=" * 60)
    print("Dataset splitting completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())