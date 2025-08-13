# -*- coding: utf-8 -*-
"""
优化的多进程数据集生成器
解决性能问题：预处理器重复初始化、数据序列化开销等
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from datetime import datetime
import hashlib
from multiprocessing import Pool, Manager
from functools import partial
import warnings
import cv2
from PIL import Image

# 抑制警告
warnings.filterwarnings('ignore')

# 全局预处理器实例（每个进程一个）
_global_preprocessor = None

def init_worker(config: Dict):
    """
    初始化工作进程，创建预处理器实例
    每个进程只创建一次，避免重复初始化开销
    """
    global _global_preprocessor
    from .preprocessor import TrainingPreprocessor
    _global_preprocessor = TrainingPreprocessor(config)
    # 不要在工作进程中打印，会干扰进度条

def process_single_sample_optimized(label: Dict) -> Optional[Dict[str, Any]]:
    """
    处理单个样本的优化函数
    使用全局预处理器，避免重复创建
    """
    try:
        # 使用全局预处理器
        global _global_preprocessor
        if _global_preprocessor is None:
            return None
            
        # 构建图像路径（使用label中的data_root）
        data_root = Path(label['_data_root'])
        image_path = data_root / label['paths']['composite']
        
        if not image_path.exists():
            return None
        
        # 提取坐标信息
        gap_center = tuple(label['labels']['bg_gap_center'])
        slider_center = tuple(label['labels']['comp_piece_center'])
        gap_angle = label['labels']['gap_pose'].get('delta_theta_deg', 0.0)
        
        # 处理混淆缺口
        fake_gaps = []
        if 'fake_gaps' in label['labels']:
            for fake_gap in label['labels']['fake_gaps']:
                fake_gaps.append(tuple(fake_gap['center']))
        
        # 使用预处理器处理图像
        result = _global_preprocessor.preprocess(
            str(image_path),
            gap_center=gap_center,
            slider_center=slider_center,
            confusing_gaps=fake_gaps if fake_gaps else None,
            gap_angle=gap_angle
        )
        
        # 只返回必要的数据，减少序列化开销
        return {
            'sample_id': label.get('sample_id', 'unknown'),
            'input': result['input'].numpy(),  # 转为numpy减少序列化开销
            'heatmaps': result['heatmaps'].numpy(),
            'offsets': result['offsets'].numpy(),
            'weight_mask': result['weight_mask'].numpy(),
            'transform_params': result['transform_params'],
            'gap_grid': result['gap_grid'],
            'gap_offset': result['gap_offset'],
            'slider_grid': result['slider_grid'],
            'slider_offset': result['slider_offset'],
            'confusing_gaps': result.get('confusing_gaps', []),
            'gap_angle': result.get('gap_angle', 0.0)
        }
    except Exception as e:
        print(f"Failed to process sample {label.get('sample_id', 'unknown')}: {e}")
        return None


def process_batch_optimized(batch_labels: List[Dict]) -> List[Optional[Dict]]:
    """
    批量处理样本，减少函数调用开销
    """
    results = []
    for label in batch_labels:
        result = process_single_sample_optimized(label)
        results.append(result)
    return results


class DatasetGeneratorMPOptimized:
    """
    优化的多进程数据集生成器
    主要优化：
    1. 每个进程只初始化一次预处理器
    2. 批量处理减少函数调用开销
    3. 优化数据序列化
    4. 动态调整chunksize
    """
    
    def __init__(self,
                 data_root: str,
                 output_dir: str,
                 config_path: Optional[str] = None,
                 batch_size: int = None,
                 num_workers: Optional[int] = None):
        """
        初始化优化的多进程数据集生成器
        """
        from .config_loader import load_config
        from multiprocessing import cpu_count
        
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        
        # 加载配置
        self.config = load_config(config_path)
        preprocessing_config = self.config['preprocessing']
        
        # 从配置文件读取batch_size
        if batch_size is not None:
            self.batch_size = batch_size
        elif 'dataset' in self.config and 'batch_size' in self.config['dataset']:
            self.batch_size = self.config['dataset']['batch_size']
        else:
            self.batch_size = 128
        
        # 设置工作进程数
        if num_workers is not None:
            self.num_workers = num_workers
        elif 'dataset' in self.config and 'num_workers' in self.config['dataset']:
            self.num_workers = self.config['dataset']['num_workers']
            print(f"Using num_workers from config: {self.num_workers}")
        else:
            self.num_workers = max(1, cpu_count() - 1)
            print(f"Using default num_workers: {self.num_workers} (CPU cores - 1)")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'labels').mkdir(exist_ok=True)
        
        # 记录配置信息
        self.target_size = preprocessing_config['letterbox']['target_size']
        self.downsample = preprocessing_config['coordinate']['downsample']
        
        # 计算输出维度
        self.input_shape = (4, self.target_size[1], self.target_size[0])
        self.grid_size = (
            self.target_size[1] // self.downsample,
            self.target_size[0] // self.downsample
        )
        
        print(f"Optimized Multi-process dataset generator initialized")
        print(f"  Data root: {self.data_root}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Worker processes: {self.num_workers}")
        print(f"  Target size: {self.target_size}")
        print(f"  Grid size: {self.grid_size}")
    
    def load_labels(self, labels_path: str) -> List[Dict]:
        """加载标签文件"""
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} labels")
        
        # 在标签中添加data_root路径，避免序列化整个路径对象
        for label in labels:
            label['_data_root'] = str(self.data_root)
        
        return labels
    
    def generate_dataset(self, labels_path: str, split: str = 'train', max_samples: Optional[int] = None):
        """
        使用优化的多进程生成数据集
        """
        print(f"\nGenerating {split} dataset (Optimized Version)...")
        print(f"Using {self.num_workers} parallel processes")
        
        # 加载标签
        labels = self.load_labels(labels_path)
        
        # 如果指定了最大样本数，截取标签
        if max_samples is not None and max_samples < len(labels):
            labels = labels[:max_samples]
            print(f"Limiting to {max_samples} samples for testing")
        
        # 准备批量处理
        processed_samples = []
        failed_count = 0
        
        # 动态计算最优chunksize
        # 对于I/O密集型任务，较小的chunksize更好
        samples_per_worker = len(labels) // self.num_workers
        chunksize = min(10, max(1, samples_per_worker // 10))
        print(f"Using chunksize: {chunksize}")
        
        # 创建进程池，使用初始化函数
        with Pool(
            processes=self.num_workers,
            initializer=init_worker,
            initargs=(self.config,)
        ) as pool:
            
            # 使用imap_unordered获得更好的性能
            batch_counter = 0
            with tqdm(total=len(labels), desc=f"Processing {split} samples") as pbar:
                # 批量处理模式
                if chunksize > 1:
                    # 将标签分组
                    label_batches = [labels[i:i+chunksize] 
                                   for i in range(0, len(labels), chunksize)]
                    
                    for batch_results in pool.imap_unordered(
                        process_batch_optimized, 
                        label_batches
                    ):
                        for processed in batch_results:
                            if processed is not None:
                                processed_samples.append(processed)
                            else:
                                failed_count += 1
                        pbar.update(len(batch_results))
                        
                        # 当积累够一个批次时，立即保存
                        if len(processed_samples) >= self.batch_size:
                            self._save_single_batch(processed_samples[:self.batch_size], split, batch_counter)
                            processed_samples = processed_samples[self.batch_size:]
                            batch_counter += 1
                else:
                    # 单个处理模式
                    for processed in pool.imap_unordered(
                        process_single_sample_optimized, 
                        labels,
                        chunksize=1
                    ):
                        if processed is not None:
                            processed_samples.append(processed)
                        else:
                            failed_count += 1
                        pbar.update(1)
                        
                        # 当积累够一个批次时，立即保存
                        if len(processed_samples) >= self.batch_size:
                            self._save_single_batch(processed_samples[:self.batch_size], split, batch_counter)
                            processed_samples = processed_samples[self.batch_size:]
                            batch_counter += 1
        
        # 保存剩余的样本
        if processed_samples:
            self._save_single_batch(processed_samples, split, batch_counter)
            
        print(f"\nDataset generation completed:")
        print(f"  Success: {len(labels) - failed_count}")
        print(f"  Failed: {failed_count}")
        
        # 生成索引文件
        if len(labels) - failed_count > 0:
            self._generate_index(split)
    
    def _save_batches(self, samples: List[Dict], split: str):
        """保存处理后的样本为批量文件"""
        num_batches = (len(samples) + self.batch_size - 1) // self.batch_size
        
        for batch_id in range(num_batches):
            start_idx = batch_id * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]
            
            self._save_single_batch(batch_samples, split, batch_id)
    
    def _save_single_batch(self, batch_samples: List[Dict], split: str, batch_id: int):
        """保存单个批次"""
        batch_size = len(batch_samples)
        
        # 准备批量数组
        images = np.zeros((batch_size, *self.input_shape), dtype=np.float32)
        heatmaps = np.zeros((batch_size, 2, *self.grid_size), dtype=np.float32)
        offsets = np.zeros((batch_size, 4, *self.grid_size), dtype=np.float32)
        weight_masks = np.zeros((batch_size, *self.grid_size), dtype=np.float32)
        
        # 元数据
        metadata = {
            'batch_id': batch_id,
            'batch_size': batch_size,
            'sample_ids': [],
            'transform_params': [],
            'grid_coords': [],
            'offsets_meta': [],
            'confusing_gaps': [],
            'gap_angles': []
        }
        
        # 填充数组
        for i, sample in enumerate(batch_samples):
            images[i] = sample['input']
            heatmaps[i] = sample['heatmaps']
            offsets[i] = sample['offsets']
            weight_masks[i] = sample['weight_mask']
            
            metadata['sample_ids'].append(sample['sample_id'])
            metadata['transform_params'].append(sample['transform_params'])
            metadata['grid_coords'].append({
                'gap': sample['gap_grid'],
                'slider': sample['slider_grid']
            })
            metadata['offsets_meta'].append({
                'gap': sample['gap_offset'],
                'slider': sample['slider_offset']
            })
            metadata['confusing_gaps'].append(sample.get('confusing_gaps', []))
            metadata['gap_angles'].append(sample.get('gap_angle', 0.0))
        
        # 保存文件
        image_path = self.output_dir / 'images' / f'{split}_{batch_id:04d}.npy'
        labels_path = self.output_dir / 'labels' / f'{split}_{batch_id:04d}.npz'
        meta_path = self.output_dir / 'labels' / f'{split}_{batch_id:04d}_meta.json'
        
        # 保存图像
        np.save(image_path, images)
        
        # 保存标签（压缩）
        np.savez_compressed(
            labels_path,
            heatmaps=heatmaps,
            offsets=offsets,
            weight_masks=weight_masks
        )
        
        # 保存元数据
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved batch {batch_id}: {batch_size} samples")
    
    def _generate_index(self, split: str):
        """生成数据集索引文件"""
        image_files = sorted((self.output_dir / 'images').glob(f'{split}_*.npy'))
        label_files = sorted((self.output_dir / 'labels').glob(f'{split}_*.npz'))
        meta_files = sorted((self.output_dir / 'labels').glob(f'{split}_*_meta.json'))
        
        index = {
            'split': split,
            'num_batches': len(image_files),
            'batch_size': self.batch_size,
            'total_samples': 0,
            'input_shape': self.input_shape,
            'grid_size': self.grid_size,
            'batches': []
        }
        
        for img_file, label_file, meta_file in zip(image_files, label_files, meta_files):
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            batch_info = {
                'batch_id': meta['batch_id'],
                'batch_size': meta['batch_size'],
                'image_file': img_file.name,
                'label_file': label_file.name,
                'meta_file': meta_file.name
            }
            index['batches'].append(batch_info)
            index['total_samples'] += meta['batch_size']
        
        # 保存索引
        index_path = self.output_dir / f"{split}_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        
        print(f"\nGenerated index file: {index_path}")
        print(f"  Total batches: {len(image_files)}")
        print(f"  Total samples: {index['total_samples']}")