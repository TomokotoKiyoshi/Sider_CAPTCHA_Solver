# -*- coding: utf-8 -*-
"""
数据加载器
用于训练时高效加载NPY格式的数据集
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader


class CaptchaDataset(Dataset):
    """
    验证码数据集类
    支持高效的批量加载和内存管理
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 cache_batches: int = 2):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录（包含images和labels文件夹）
            split: 数据集划分 ('train', 'val', 'test')
            cache_batches: 缓存的批次数量，用于加速访问
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.cache_batches = cache_batches
        
        # 加载索引文件
        index_path = self.data_dir / f"{split}_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        with open(index_path, 'r') as f:
            self.index = json.load(f)
        
        self.total_samples = self.index['total_samples']
        self.batches = self.index['batches']
        self.input_shape = tuple(self.index['input_shape'])
        self.grid_size = tuple(self.index['grid_size'])
        
        # 构建样本索引映射
        self._build_sample_index()
        
        # 批次缓存
        self.cache = {}
        self.cache_order = []
        
        print(f"Dataset initialized: {split}")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Batches: {len(self.batches)}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Grid size: {self.grid_size}")
    
    def _build_sample_index(self):
        """
        构建全局样本索引到批次的映射
        """
        self.sample_map = []
        current_idx = 0
        
        for batch_idx, batch_info in enumerate(self.batches):
            batch_size = batch_info['batch_size']
            for local_idx in range(batch_size):
                self.sample_map.append({
                    'batch_idx': batch_idx,
                    'local_idx': local_idx,
                    'global_idx': current_idx + local_idx
                })
            current_idx += batch_size
    
    def _load_batch(self, batch_idx: int) -> Dict[str, Any]:
        """
        加载一个批次的数据
        
        Args:
            batch_idx: 批次索引
        
        Returns:
            批次数据字典
        """
        # 检查缓存
        if batch_idx in self.cache:
            # 更新访问顺序
            self.cache_order.remove(batch_idx)
            self.cache_order.append(batch_idx)
            return self.cache[batch_idx]
        
        # 加载数据
        batch_info = self.batches[batch_idx]
        
        # 加载图像
        image_path = self.data_dir / batch_info['image_file']
        images = np.load(image_path)
        
        # 加载标签
        label_path = self.data_dir / batch_info['label_file']
        labels = np.load(label_path)
        
        # 加载元数据
        meta_path = self.data_dir / batch_info['meta_file']
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        batch_data = {
            'images': images,
            'heatmaps': labels['heatmaps'],
            'offsets': labels['offsets'],
            'weight_masks': labels['weight_masks'],
            'metadata': metadata
        }
        
        # 更新缓存
        if len(self.cache) >= self.cache_batches:
            # 移除最旧的缓存
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        self.cache[batch_idx] = batch_data
        self.cache_order.append(batch_idx)
        
        return batch_data
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            样本字典，包含input和labels
        """
        # 获取样本映射信息
        sample_info = self.sample_map[idx]
        batch_idx = sample_info['batch_idx']
        local_idx = sample_info['local_idx']
        
        # 加载批次数据
        batch_data = self._load_batch(batch_idx)
        
        # 提取样本
        sample = {
            'input': torch.from_numpy(batch_data['images'][local_idx]).float(),
            'heatmaps': torch.from_numpy(batch_data['heatmaps'][local_idx]).float(),
            'offsets': torch.from_numpy(batch_data['offsets'][local_idx]).float(),
            'weight_mask': torch.from_numpy(batch_data['weight_masks'][local_idx]).float(),
            'sample_id': batch_data['metadata']['samples'][local_idx]['sample_id']
        }
        
        return sample
    
    def get_batch_metadata(self, batch_idx: int) -> Dict:
        """
        获取批次的元数据
        
        Args:
            batch_idx: 批次索引
        
        Returns:
            元数据字典
        """
        meta_path = self.data_dir / self.batches[batch_idx]['meta_file']
        with open(meta_path, 'r') as f:
            return json.load(f)


class CaptchaDataLoader:
    """
    验证码数据加载器工厂类
    """
    
    @staticmethod
    def create_loader(data_dir: str,
                      split: str = 'train',
                      batch_size: int = 32,
                      num_workers: int = 4,
                      shuffle: bool = None,
                      cache_batches: int = 2) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            data_dir: 数据目录
            split: 数据集划分
            batch_size: 批量大小
            num_workers: 工作进程数
            shuffle: 是否打乱，None时训练集打乱，其他不打乱
            cache_batches: 缓存批次数
        
        Returns:
            PyTorch DataLoader
        """
        if shuffle is None:
            shuffle = (split == 'train')
        
        dataset = CaptchaDataset(
            data_dir=data_dir,
            split=split,
            cache_batches=cache_batches
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')  # 训练时丢弃不完整批次
        )
        
        return loader
