#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速训练测试脚本 - 只使用5%的数据进行快速验证
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import torch
from pathlib import Path
import glob
import random

# 导入训练脚本的主函数
from scripts.training.train import main as train_main

def limit_npy_files(data_dir, percentage=0.05):
    """限制NPY文件数量到指定百分比"""
    for split in ['train', 'val', 'test']:
        # 图像文件
        image_dir = Path(data_dir) / 'processed' / 'images' / split
        if image_dir.exists():
            image_files = sorted(glob.glob(str(image_dir / '*.npy')))
            num_keep = max(1, int(len(image_files) * percentage))
            files_to_keep = random.sample(image_files, min(num_keep, len(image_files)))
            
            # 删除未选中的文件（实际上是重命名，避免真的删除）
            for f in image_files:
                if f not in files_to_keep:
                    os.rename(f, f + '.bak')
            
            print(f"{split}集: 保留 {len(files_to_keep)}/{len(image_files)} 个图像文件")
            
            # 对应的标签文件
            label_dir = Path(data_dir) / 'processed' / 'labels' / split
            if label_dir.exists():
                for f in image_files:
                    if f not in files_to_keep:
                        base_name = Path(f).stem
                        for label_file in glob.glob(str(label_dir / f"{base_name}*.npy")):
                            os.rename(label_file, label_file + '.bak')
                        meta_file = label_dir / f"{base_name}_meta.json"
                        if meta_file.exists():
                            os.rename(str(meta_file), str(meta_file) + '.bak')

def restore_npy_files(data_dir):
    """恢复所有备份的NPY文件"""
    for split in ['train', 'val', 'test']:
        # 恢复图像文件
        image_dir = Path(data_dir) / 'processed' / 'images' / split
        if image_dir.exists():
            for f in glob.glob(str(image_dir / '*.npy.bak')):
                os.rename(f, f[:-4])  # 移除.bak后缀
        
        # 恢复标签文件
        label_dir = Path(data_dir) / 'processed' / 'labels' / split
        if label_dir.exists():
            for f in glob.glob(str(label_dir / '*.npy.bak')):
                os.rename(f, f[:-4])
            for f in glob.glob(str(label_dir / '*.json.bak')):
                os.rename(f, f[:-4])

def main():
    """快速训练测试主函数"""
    parser = argparse.ArgumentParser(description='快速训练测试（使用部分数据）')
    parser.add_argument('--percentage', type=float, default=0.05,
                       help='使用数据的百分比 (默认: 0.05即5%%)')
    parser.add_argument('--epochs', type=int, default=2,
                       help='训练轮数 (默认: 2)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小 (默认: 16)')
    args, unknown = parser.parse_known_args()
    
    # 数据目录
    data_dir = Path(__file__).parent.parent.parent / 'data'
    
    try:
        print(f"\n{'='*60}")
        print(f"快速训练测试 - 使用 {args.percentage*100:.1f}% 数据")
        print(f"{'='*60}\n")
        
        # 限制数据量
        limit_npy_files(data_dir, args.percentage)
        
        # 修改系统参数以使用更少的epoch
        sys.argv = [sys.argv[0]]  # 清除原有参数
        sys.argv.extend(['--epochs', str(args.epochs)])
        sys.argv.extend(['--batch-size', str(args.batch_size)])
        sys.argv.extend(unknown)  # 添加其他未知参数
        
        # 运行训练
        train_main()
        
    finally:
        # 恢复所有文件
        print("\n恢复所有数据文件...")
        restore_npy_files(data_dir)
        print("恢复完成")

if __name__ == "__main__":
    main()