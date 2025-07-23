# -*- coding: utf-8 -*-
"""
数据集划分脚本
按照Pic{XXXX}进行分类，避免同一张图片同时出现在训练集与测试集
训练/测试 = 9/1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import shutil
from collections import defaultdict
import random
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def extract_pic_id(filename):
    """
    从文件名中提取Pic ID
    例如: Pic0001_Bgx120Bgy70_Sdx30Sdy70_{hash}.png -> 1
    """
    pic_part = filename.split('_')[0]  # Pic0001
    pic_id = int(pic_part[3:])  # 提取数字部分
    return pic_id


def process_file_batch(args):
    """处理一批文件的复制/移动操作"""
    files, captcha_dir, train_dir, test_dir, move_files = args
    
    results = []
    for filename, dataset in files:
        src_path = captcha_dir / filename
        
        if dataset == 'train':
            dst_path = train_dir / filename
        else:
            dst_path = test_dir / filename
        
        if src_path.exists():
            try:
                if move_files:
                    shutil.move(str(src_path), str(dst_path))
                else:
                    shutil.copy2(src_path, dst_path)
                results.append((filename, True, None))
            except Exception as e:
                results.append((filename, False, str(e)))
        else:
            results.append((filename, False, "File not found"))
    
    return results


def split_dataset_optimized(captcha_dir, train_dir, test_dir, train_ratio=0.9, seed=42, move_files=False, num_workers=None):
    """
    优化的数据集划分函数（支持大数据集）
    
    Args:
        captcha_dir: 验证码目录
        train_dir: 训练集输出目录
        test_dir: 测试集输出目录
        train_ratio: 训练集比例
        seed: 随机种子
        move_files: 是否移动文件而不是复制（对大数据集更快）
    """
    captcha_dir = Path(captcha_dir)
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    
    # 检查输入目录
    if not captcha_dir.exists():
        print(f"Error: Captcha directory not found: {captcha_dir}")
        return
    
    # 读取标注文件
    annotations_path = captcha_dir / "annotations.json"
    if not annotations_path.exists():
        print(f"Error: Annotations file not found: {annotations_path}")
        return
    
    print("Loading annotations...")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"Total annotations: {len(annotations)}")
    
    # 按Pic ID分组
    print("Grouping by Pic ID...")
    pic_groups = defaultdict(list)
    for ann in tqdm(annotations, desc="Grouping annotations"):
        pic_id = extract_pic_id(ann['filename'])
        pic_groups[pic_id].append(ann)
    
    print(f"Total unique Pic IDs: {len(pic_groups)}")
    
    # 获取所有Pic ID并打乱
    random.seed(seed)
    pic_ids = list(pic_groups.keys())
    random.shuffle(pic_ids)
    
    # 计算训练集和测试集的Pic ID数量
    train_count = int(len(pic_ids) * train_ratio)
    
    # 划分Pic ID
    train_pic_ids = set(pic_ids[:train_count])
    test_pic_ids = set(pic_ids[train_count:])
    
    print(f"\nDataset split:")
    print(f"  Train Pic IDs: {len(train_pic_ids)}")
    print(f"  Test Pic IDs: {len(test_pic_ids)}")
    
    # 创建输出目录
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备文件操作
    train_annotations = []
    test_annotations = []
    train_files = []
    test_files = []
    
    for pic_id, anns in pic_groups.items():
        if pic_id in train_pic_ids:
            train_annotations.extend(anns)
            train_files.extend([(ann['filename'], 'train') for ann in anns])
        else:
            test_annotations.extend(anns)
            test_files.extend([(ann['filename'], 'test') for ann in anns])
    
    print(f"  Train samples: {len(train_annotations)}")
    print(f"  Test samples: {len(test_annotations)}")
    
    # 处理文件
    operation = "Moving" if move_files else "Copying"
    print(f"\n{operation} files...")
    
    # 合并所有文件操作
    all_files = train_files + test_files
    
    # 确定工作进程数
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)
    
    # 将文件分成批次
    batch_size = max(100, len(all_files) // (num_workers * 10))  # 每批至少100个文件
    file_batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]
    
    print(f"Using {num_workers} workers to process {len(file_batches)} batches")
    
    # 使用多进程处理
    error_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 准备任务
        futures = []
        for batch in file_batches:
            future = executor.submit(process_file_batch, 
                                   (batch, captcha_dir, train_dir, test_dir, move_files))
            futures.append(future)
        
        # 处理结果
        with tqdm(total=len(all_files), desc=f"{operation} files") as pbar:
            for future in as_completed(futures):
                try:
                    results = future.result()
                    for filename, success, error in results:
                        if not success:
                            error_count += 1
                            if error == "File not found":
                                print(f"\nWarning: Image not found: {filename}")
                            else:
                                print(f"\nError processing {filename}: {error}")
                        pbar.update(1)
                except Exception as e:
                    print(f"\nBatch processing error: {e}")
                    pbar.update(batch_size)
    
    if error_count > 0:
        print(f"\nWarning: {error_count} files failed to {operation.lower()}")
    
    # 保存划分后的标注文件
    print("\nSaving annotations...")
    train_ann_path = train_dir / "annotations.json"
    with open(train_ann_path, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, ensure_ascii=False, indent=2)
    
    test_ann_path = test_dir / "annotations.json"
    with open(test_ann_path, 'w', encoding='utf-8') as f:
        json.dump(test_annotations, f, ensure_ascii=False, indent=2)
    
    # 生成数据集统计信息
    stats = {
        'total_images': len(annotations),
        'total_pic_ids': len(pic_groups),
        'train_pic_ids': len(train_pic_ids),
        'test_pic_ids': len(test_pic_ids),
        'train_samples': len(train_annotations),
        'test_samples': len(test_annotations),
        'train_ratio': train_ratio,
        'test_ratio': 1 - train_ratio,
        'random_seed': seed,
        'operation': 'move' if move_files else 'copy'
    }
    
    # 保存统计信息到data目录
    stats_path = Path(train_dir).parent / "dataset_split_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("\nDataset split completed!")
    print(f"Train annotations saved to: {train_ann_path}")
    print(f"Test annotations saved to: {test_ann_path}")
    print(f"Statistics saved to: {stats_path}")
    
    # 验证数据无泄漏
    print("\nVerifying no data leakage...")
    train_pic_ids_check = set()
    for ann in train_annotations:
        pic_id = extract_pic_id(ann['filename'])
        train_pic_ids_check.add(pic_id)
    
    test_pic_ids_check = set()
    for ann in test_annotations:
        pic_id = extract_pic_id(ann['filename'])
        test_pic_ids_check.add(pic_id)
    
    overlap = train_pic_ids_check & test_pic_ids_check
    if overlap:
        print(f"ERROR: Data leakage detected! Overlapping Pic IDs: {overlap}")
    else:
        print("SUCCESS: No data leakage. Train and test sets have no overlapping Pic IDs.")
    
    # 如果是移动操作，可以选择删除原目录
    if move_files:
        print(f"\nNote: Original files have been moved from {captcha_dir}")
        print("You may want to delete the empty source directory if no longer needed.")


def split_dataset(captcha_dir, output_dir, train_ratio=0.9, seed=42, num_workers=None):
    """
    划分数据集
    
    Args:
        captcha_dir: 验证码目录（包含所有验证码图片和annotations.json）
        output_dir: 输出目录
        train_ratio: 训练集比例（默认0.9）
        seed: 随机种子
    """
    captcha_dir = Path(captcha_dir)
    output_dir = Path(output_dir)
    
    # 检查输入目录
    if not captcha_dir.exists():
        print(f"Error: Captcha directory not found: {captcha_dir}")
        return
    
    # 读取标注文件
    annotations_path = captcha_dir / "annotations.json"
    if not annotations_path.exists():
        print(f"Error: Annotations file not found: {annotations_path}")
        return
    
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"Total annotations: {len(annotations)}")
    
    # 按Pic ID分组
    pic_groups = defaultdict(list)
    for ann in annotations:
        pic_id = extract_pic_id(ann['filename'])
        pic_groups[pic_id].append(ann)
    
    print(f"Total unique Pic IDs: {len(pic_groups)}")
    
    # 获取所有Pic ID并打乱
    random.seed(seed)
    pic_ids = list(pic_groups.keys())
    random.shuffle(pic_ids)
    
    # 计算训练集和测试集的Pic ID数量
    train_count = int(len(pic_ids) * train_ratio)
    
    # 划分Pic ID
    train_pic_ids = set(pic_ids[:train_count])
    test_pic_ids = set(pic_ids[train_count:])
    
    print(f"Train Pic IDs: {len(train_pic_ids)}")
    print(f"Test Pic IDs: {len(test_pic_ids)}")
    
    # 创建输出目录
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 划分标注
    train_annotations = []
    test_annotations = []
    
    for pic_id, anns in pic_groups.items():
        if pic_id in train_pic_ids:
            train_annotations.extend(anns)
        else:
            test_annotations.extend(anns)
    
    print(f"Train samples: {len(train_annotations)}")
    print(f"Test samples: {len(test_annotations)}")
    
    # 复制图片文件
    print("\nCopying images...")
    
    # 准备所有文件操作
    train_files = [(ann['filename'], 'train') for ann in train_annotations]
    test_files = [(ann['filename'], 'test') for ann in test_annotations]
    all_files = train_files + test_files
    
    # 确定工作进程数
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)
    
    # 将文件分成批次
    batch_size = max(100, len(all_files) // (num_workers * 10))
    file_batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]
    
    print(f"Using {num_workers} workers to process {len(file_batches)} batches")
    
    # 使用多进程处理
    error_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 准备任务
        futures = []
        for batch in file_batches:
            future = executor.submit(process_file_batch, 
                                   (batch, captcha_dir, train_dir, test_dir, False))  # False = copy
            futures.append(future)
        
        # 处理结果
        with tqdm(total=len(all_files), desc="Copying files") as pbar:
            for future in as_completed(futures):
                try:
                    results = future.result()
                    for filename, success, error in results:
                        if not success:
                            error_count += 1
                            if error == "File not found":
                                print(f"\nWarning: Image not found: {filename}")
                            else:
                                print(f"\nError processing {filename}: {error}")
                        pbar.update(1)
                except Exception as e:
                    print(f"\nBatch processing error: {e}")
                    pbar.update(batch_size)
    
    if error_count > 0:
        print(f"\nWarning: {error_count} files failed to copy")
    
    # 保存划分后的标注文件
    train_ann_path = train_dir / "annotations.json"
    with open(train_ann_path, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, ensure_ascii=False, indent=2)
    
    test_ann_path = test_dir / "annotations.json"
    with open(test_ann_path, 'w', encoding='utf-8') as f:
        json.dump(test_annotations, f, ensure_ascii=False, indent=2)
    
    # 生成数据集统计信息
    stats = {
        'total_images': len(annotations),
        'total_pic_ids': len(pic_groups),
        'train_pic_ids': len(train_pic_ids),
        'test_pic_ids': len(test_pic_ids),
        'train_samples': len(train_annotations),
        'test_samples': len(test_annotations),
        'train_ratio': train_ratio,
        'test_ratio': 1 - train_ratio,
        'random_seed': seed
    }
    
    # 保存统计信息
    stats_path = output_dir / "split_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("\nDataset split completed!")
    print(f"Train annotations saved to: {train_ann_path}")
    print(f"Test annotations saved to: {test_ann_path}")
    print(f"Statistics saved to: {stats_path}")
    
    # 验证数据无泄漏
    print("\nVerifying no data leakage...")
    train_pic_ids_check = set()
    for ann in train_annotations:
        pic_id = extract_pic_id(ann['filename'])
        train_pic_ids_check.add(pic_id)
    
    test_pic_ids_check = set()
    for ann in test_annotations:
        pic_id = extract_pic_id(ann['filename'])
        test_pic_ids_check.add(pic_id)
    
    overlap = train_pic_ids_check & test_pic_ids_check
    if overlap:
        print(f"ERROR: Data leakage detected! Overlapping Pic IDs: {overlap}")
    else:
        print("SUCCESS: No data leakage. Train and test sets have no overlapping Pic IDs.")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split CAPTCHA dataset into train/test sets')
    parser.add_argument('--captcha-dir', type=str, default='data/captchas',
                        help='Directory containing all CAPTCHA images and annotations')
    parser.add_argument('--train-dir', type=str, default='data/train',
                        help='Output directory for training set')
    parser.add_argument('--test-dir', type=str, default='data/test',
                        help='Output directory for test set')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Training set ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--move', action='store_true',
                        help='Move files instead of copying (faster for large datasets)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto)')
    
    args = parser.parse_args()
    
    # 设置路径
    project_root = Path(__file__).parent.parent
    captcha_dir = project_root / args.captcha_dir
    
    # 创建输出目录结构
    output_dir = project_root / "data"
    train_dir = project_root / args.train_dir
    test_dir = project_root / args.test_dir
    
    # 执行划分（传入单独的训练和测试目录）
    split_dataset_optimized(captcha_dir, train_dir, test_dir, args.train_ratio, args.seed, args.move, args.workers)


if __name__ == "__main__":
    main()