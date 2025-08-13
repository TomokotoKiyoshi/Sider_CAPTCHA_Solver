#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess Training Dataset (Optimized Multi-process Version)

Highly optimized version with:
- Single preprocessor initialization per worker
- Batch processing mode
- Optimized serialization
- Dynamic chunksize adjustment
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import argparse
import yaml
from src.preprocessing.dataset_generator import DatasetGeneratorMPOptimized


def main():
    # 先加载配置文件
    config_path = Path(__file__).parents[1] / 'config' / 'preprocessing_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 从配置文件获取默认路径
    default_paths = config.get('paths', {})
    
    parser = argparse.ArgumentParser(description='Generate CAPTCHA training dataset (Optimized)')
    parser.add_argument('--config', type=str, default=str(config_path),
                       help='Config file path')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split')
    # 可选参数，用于覆盖配置文件中的设置
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Override number of workers from config')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    # 重新加载配置文件（如果指定了不同的配置）
    if args.config != str(config_path):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        default_paths = config.get('paths', {})
    
    # 从配置文件获取路径
    data_root = default_paths.get('data_root', 'data')
    output_dir = default_paths.get('output_dir', 'data/processed')
    labels_path = default_paths.get('labels_file', 'data/labels/all_labels.json')
    
    print("=" * 60)
    print("CAPTCHA Dataset Preprocessing (Optimized Multi-process Version)")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Data root: {data_root}")
    print(f"  Output dir: {output_dir}")
    print(f"  Labels file: {labels_path}")
    print(f"  Split: {args.split}")
    print("\nOptimizations:")
    print("✓ Single preprocessor initialization per worker")
    print("✓ Batch processing to reduce overhead")
    print("✓ Optimized data serialization")
    print("✓ Dynamic chunksize adjustment")
    print("=" * 60)
    
    # 创建优化的多进程生成器
    generator = DatasetGeneratorMPOptimized(
        data_root=data_root,
        output_dir=output_dir,
        config_path=args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 生成数据集
    if args.max_samples:
        print(f"\\nTest mode: Processing only {args.max_samples} samples")
    generator.generate_dataset(labels_path, split=args.split, max_samples=args.max_samples)
    
    print("\n" + "=" * 60)
    print("Dataset generation completed!")
    print("=" * 60)
    print(f"\nOutput file structure:")
    print(f"{output_dir}/")
    print(f"├── images/          # Preprocessed image NPY files")
    print(f"│   └── {args.split}_*.npy    # Each file contains {generator.batch_size} images")
    print(f"├── labels/          # Labels and metadata")
    print(f"│   ├── {args.split}_*.npz    # Compressed label arrays (heatmaps, offsets, masks)")
    print(f"│   └── {args.split}_*_meta.json  # Metadata (coordinates, transform params, etc.)")
    print(f"└── {args.split}_index.json   # Dataset index file")
    print(f"\nData format description:")
    print(f"- Images: [batch_size, 4, 256, 512] (4 channels = RGB + padding_mask)")
    print(f"- Heatmaps: [batch_size, 2, 64, 128] (gap + slider)")
    print(f"- Offsets: [batch_size, 4, 64, 128] (sub-pixel precision)")
    print(f"- Masks: [batch_size, 64, 128] (valid region weights)")


if __name__ == "__main__":
    main()