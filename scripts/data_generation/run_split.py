#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分割执行脚本 - 使用新的配置系统
按原始图片ID分割，避免数据泄漏
"""
import sys
from pathlib import Path
from glob import glob
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.data_split_config import DataSplitConfig


def main():
    """执行数据集分割 - 使用config中的默认配置"""
    parser = argparse.ArgumentParser(
        description='Split CAPTCHA dataset by image ID to avoid data leakage'
    )
    
    # 创建配置（使用默认值）
    config = DataSplitConfig()
    
    # 只提供覆盖默认值的选项
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
    
    # 获取图片文件列表
    input_dir = Path(project_root) / args.input_dir
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    pattern = str(input_dir / '*.png')
    file_list = glob(pattern)
    
    if not file_list:
        print(f"Error: No PNG files found in {input_dir}")
        return 1
    
    print(f"Input directory: {input_dir}")
    print(f"Found {len(file_list)} PNG files")
    print(f"Split ratios: train={config.train_ratio}, val={config.val_ratio}, test={config.test_ratio}")
    print(f"Random seed: {config.seed}")
    print("-" * 60)
    
    # 执行分割
    try:
        split_result = config.split_by_image(file_list)
        
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
                print(f"  (每个验证码都有对应的滑块和背景组件)")
        
        return 0
        
    except Exception as e:
        print(f"\nError during split: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())