#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAPTCHA数据集预处理脚本 - 流式版本
使用流式写入，彻底解决内存问题
"""
import sys
from pathlib import Path
import argparse
from datetime import datetime
import psutil
import gc

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[2]  # 现在需要向上两层到达项目根目录
sys.path.insert(0, str(project_root))

# 导入流式数据集生成器和数据集划分器
from src.preprocessing.dataset_generator import StreamingDatasetGenerator
from src.preprocessing.dataset_splitter import DatasetSplitter


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
    parser.add_argument('--auto-split', action='store_true',
                       help='Automatically split dataset by pic_id')
    parser.add_argument('--train-ratio', type=float, default=None,
                       help='Training set ratio (overrides config file)')
    parser.add_argument('--val-ratio', type=float, default=None,
                       help='Validation set ratio (overrides config file)')
    parser.add_argument('--test-ratio', type=float, default=None,
                       help='Test set ratio (overrides config file)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for split (overrides config file)')
    parser.add_argument('--shuffle', type=lambda x: x.lower() in ['true', '1', 'yes', 'on'],
                       default=None,
                       help='Whether to shuffle data before splitting (overrides config file, true/false)')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Disable shuffling (equivalent to --shuffle false)')
    
    args = parser.parse_args()
    
    # 处理no-shuffle参数
    if args.no_shuffle:
        args.shuffle = False
    
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
    
    # 从配置文件读取数据集划分参数
    data_split_config = config.get('data_split', {})
    split_ratios = data_split_config.get('split_ratio', {})
    split_options = data_split_config.get('options', {})
    
    # 命令行参数优先级高于配置文件
    train_ratio = args.train_ratio if args.train_ratio is not None else split_ratios.get('train', 0.8)
    val_ratio = args.val_ratio if args.val_ratio is not None else split_ratios.get('val', 0.1)
    test_ratio = args.test_ratio if args.test_ratio is not None else split_ratios.get('test', 0.1)
    seed = args.seed if args.seed is not None else split_options.get('random_seed', 42)
    
    # 判断是否启用自动划分（命令行参数 > 配置文件）
    # 注意：如果命令行有--auto-split，则为True；否则读取配置文件
    auto_split_enabled = args.auto_split or split_options.get('auto_split', False)
    
    # 显示配置来源信息
    print(f"\n📋 Configuration Source:")
    print(f"  Config file: {config_path}")
    print(f"  Auto-split setting in config: {split_options.get('auto_split', False)}")
    print(f"  Command line --auto-split: {args.auto_split}")
    print(f"  Final auto_split_enabled: {auto_split_enabled}")
    
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
    if auto_split_enabled:
        print(f"✅ Auto-split by pic_id enabled (train:{train_ratio}, val:{val_ratio}, test:{test_ratio}, seed:{seed})")
        print(f"   Source: {'command line --auto-split' if args.auto_split else f'config file (auto_split: true)'}")
        print(f"   Will generate: train/val/test datasets")
    else:
        print(f"⚠️  Auto-split disabled - will only generate '{args.split}' dataset")
        print(f"   To enable auto-split: set auto_split: true in config or use --auto-split")
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
        
        # 如果启用自动划分
        if auto_split_enabled:
            print("\n🔄 Auto-splitting dataset by pic_id...")
            
            # 加载所有标签
            import json
            with open(labels_file, 'r', encoding='utf-8') as f:
                all_labels = json.load(f)
            
            # 如果指定了最大样本数，先截取
            if args.max_samples:
                all_labels = all_labels[:args.max_samples]
                print(f"Limited to {args.max_samples} samples for testing")
            
            # 获取shuffle配置（命令行参数 > 配置文件）
            shuffle = getattr(args, 'shuffle', None)
            if shuffle is None:
                shuffle = split_options.get('shuffle', True)
            
            # 创建数据集划分器（使用从配置文件或命令行获取的参数）
            splitter = DatasetSplitter(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                shuffle=shuffle
            )
            
            # 执行划分
            split_result = splitter.split_by_pic_id(all_labels)
            
            # 保存划分信息（使用配置文件中的路径）
            split_info_dir = config.get('output_structure', {}).get('split_info_subdir', 'split_info')
            splitter.save_split_info(split_result, str(output_dir / split_info_dir))
            
            # 得到 split_result 后，尽快释放大表
            del all_labels
            gc.collect()
            
            # 保证固定次序，且每次处理后从 split_result 弹出释放引用
            for split_name in ("train", "val", "test"):
                split_labels = split_result.pop(split_name, [])
                if not split_labels:
                    continue  # 跳过空的split
                    
                print(f"\n📝 Processing {split_name} split ({len(split_labels)} samples)...")
                mem_before_mb, mem_before_percent = monitor_memory()
                print(f"Memory before: {mem_before_mb:.2f} MB ({mem_before_percent:.1f}%)")
                
                # 生成数据集（传入预划分的标签）
                generator.generate_dataset(
                    str(labels_file),
                    split=split_name,
                    labels_subset=split_labels
                )
                
                # 释放当前 split 的列表
                del split_labels
                gc.collect()
                
                mem_after_mb, mem_after_percent = monitor_memory()
                print(f"Memory after: {mem_after_mb:.2f} MB ({mem_after_percent:.1f}%)")
                print(f"Memory delta: {mem_after_mb - mem_before_mb:+.2f} MB")
            
            # 最后清空字典，二次回收
            split_result.clear()
            gc.collect()
        else:
            # 原始逻辑：不自动划分
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