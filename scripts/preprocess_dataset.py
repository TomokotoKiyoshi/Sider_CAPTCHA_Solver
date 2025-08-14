#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAPTCHA数据集预处理脚本 - 流式版本
使用流式写入，彻底解决内存问题
"""
import sys
import os
from pathlib import Path
import argparse
from datetime import datetime
import psutil
import gc

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# 导入流式数据集生成器
from src.preprocessing.dataset_generator import StreamingDatasetGenerator


def monitor_memory():
    """监控内存使用"""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    mem_percent = process.memory_percent()
    return mem_mb, mem_percent


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CAPTCHA Dataset Preprocessing (Streaming Version)')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test', 'all'],
                       help='Dataset split to generate')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to preprocessing config file')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of worker processes')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for saving')
    
    args = parser.parse_args()
    
    # 显示启动信息
    print("=" * 80)
    print("CAPTCHA Dataset Preprocessing (Streaming Version)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示内存信息
    mem_mb, mem_percent = monitor_memory()
    print(f"Initial memory: {mem_mb:.2f} MB ({mem_percent:.1f}%)")
    
    # 如果没有指定配置文件，使用默认配置
    if args.config is None:
        config_path = project_root / "config" / "preprocessing_config.yaml"
    else:
        config_path = Path(args.config)
    
    # 加载配置文件
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 从配置文件读取路径（支持相对路径）
    def resolve_path(path_str):
        path = Path(path_str)
        if not path.is_absolute():
            path = project_root / path
        return path
    
    data_root = resolve_path(config['paths']['data_root'])
    output_dir = resolve_path(config['paths']['output_dir'])
    labels_file = resolve_path(config['paths']['labels_file'])
    
    print(f"\nConfiguration:")
    print(f"  Config file: {config_path}")
    print(f"  Data root: {data_root}")
    print(f"  Output dir: {output_dir}")
    print(f"  Labels file: {labels_file}")
    print(f"  Split: {args.split}")
    
    # 检查标签文件是否存在
    if not labels_file.exists():
        print(f"\n❌ Error: Labels file not found: {labels_file}")
        print("Please run the captcha generation script first.")
        return 1
    
    print(f"\nOptimizations:")
    print(f"✅ Streaming write with reusable buffers")
    print(f"✅ No intermediate lists or accumulation")
    print(f"✅ Separate file saves (no zip overhead)")
    print(f"✅ Memory usage is constant and predictable")
    print("=" * 80)
    
    try:
        # 创建流式生成器
        generator = StreamingDatasetGenerator(
            data_root=str(data_root),
            output_dir=str(output_dir),
            config_path=str(config_path),
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # 生成数据集
        if args.split == 'all':
            splits = ['train', 'val', 'test']
        else:
            splits = [args.split]
        
        for split in splits:
            print(f"\n📝 Processing {split} split...")
            
            # 显示处理前内存
            mem_before_mb, mem_before_percent = monitor_memory()
            print(f"Memory before: {mem_before_mb:.2f} MB ({mem_before_percent:.1f}%)")
            
            # 生成数据集
            generator.generate_dataset(
                str(labels_file),
                split=split,
                max_samples=args.max_samples
            )
            
            # 显示处理后内存
            mem_after_mb, mem_after_percent = monitor_memory()
            print(f"Memory after: {mem_after_mb:.2f} MB ({mem_after_percent:.1f}%)")
            print(f"Memory delta: {mem_after_mb - mem_before_mb:+.2f} MB")
            
            # 强制垃圾回收
            gc.collect(2)
        
        # 最终内存统计
        final_mem_mb, final_mem_percent = monitor_memory()
        print(f"\n" + "=" * 80)
        print(f"✅ All processing completed successfully!")
        print(f"Final memory: {final_mem_mb:.2f} MB ({final_mem_percent:.1f}%)")
        print(f"Total memory growth: {final_mem_mb - mem_mb:+.2f} MB")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())