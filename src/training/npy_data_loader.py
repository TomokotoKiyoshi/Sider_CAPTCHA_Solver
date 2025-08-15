#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPY批次数据加载器 - 用于加载预处理的npy格式数据
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import logging
from typing import Dict, List, Optional, Any


class NPYBatchDataset(Dataset):
    """
    从预处理的NPY批次文件加载数据集
    
    新数据结构（支持train/val/test子目录）：
    - images/
        - train/: train_XXXX.npy (训练批次图像)
        - val/: val_XXXX.npy (验证批次图像)
        - test/: test_XXXX.npy (测试批次图像)
    - labels/
        - train/: 
            - train_XXXX_heatmaps.npy (热力图)
            - train_XXXX_offsets.npy (偏移量)
            - train_XXXX_weights.npy (权重)
            - train_XXXX_meta.json (元数据)
        - val/: val_XXXX_*.npy
        - test/: test_XXXX_*.npy
    """
    
    def __init__(self, 
                 data_root: str = "data/processed",
                 mode: str = 'train',
                 index_file: Optional[str] = None):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            mode: 'train', 'val' 或 'test'
            index_file: 索引文件路径 (可选，如果不提供则扫描目录)
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.logger = logging.getLogger(f'NPYBatchDataset.{mode}')
        
        # 如果提供了索引文件，从索引加载
        if index_file:
            index_path = self.data_root / index_file
            if not index_path.exists():
                # 尝试在split_info目录查找
                index_path = self.data_root / "split_info" / index_file
                if not index_path.exists():
                    raise FileNotFoundError(f"索引文件不存在: {index_file}")
            
            with open(index_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
            
            self.batches = self.index['batches']
            self.num_batches = len(self.batches)
            self.batch_size = self.index.get('batch_size', 64)
            self.total_samples = self.index.get('total_samples', self.num_batches * self.batch_size)
        else:
            # 扫描目录以查找批次文件
            self._scan_directory()
        
        self.logger.info(f"加载了 {self.num_batches} 个批次，共 {self.total_samples} 个样本")
    
    def _scan_directory(self):
        """扫描目录以查找批次文件"""
        # 构建图像目录路径（新结构：images/train/）
        image_dir = self.data_root / "images" / self.mode
        
        if not image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        
        # 查找所有npy文件
        npy_files = sorted(image_dir.glob(f"{self.mode}_*.npy"))
        
        if not npy_files:
            raise FileNotFoundError(f"在 {image_dir} 中未找到任何 {self.mode}_*.npy 文件")
        
        self.batches = []
        total_samples = 0
        
        for npy_file in npy_files:
            # 提取批次ID（例如：train_0000.npy -> 0000）
            batch_id_str = npy_file.stem.replace(f"{self.mode}_", "")
            try:
                batch_id = int(batch_id_str)
            except ValueError:
                self.logger.warning(f"跳过无效文件名: {npy_file.name}")
                continue
            
            # 检查对应的标签文件是否存在
            label_dir = self.data_root / "labels" / self.mode
            prefix = f"{self.mode}_{batch_id:04d}"
            
            required_files = [
                label_dir / f"{prefix}_heatmaps.npy",
                label_dir / f"{prefix}_offsets.npy",
                label_dir / f"{prefix}_weights.npy",
                label_dir / f"{prefix}_meta.json"
            ]
            
            if all(f.exists() for f in required_files):
                # 读取元数据以获取批次大小
                with open(required_files[-1], 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    batch_size = meta.get('batch_size', 64)
                
                # 丢弃不完整的批次（从配置或索引文件获取期望的批次大小）
                expected_batch_size = self.index.get('batch_size', 64) if hasattr(self, 'index') and self.index else 64
                if batch_size < expected_batch_size:
                    self.logger.info(f"丢弃不完整批次 {batch_id}，大小为 {batch_size}（期望 {expected_batch_size}）")
                    continue
                
                self.batches.append({
                    'batch_id': batch_id,
                    'batch_size': batch_size,
                    'image_file': str(npy_file.name)
                })
                total_samples += batch_size
            else:
                missing = [f for f in required_files if not f.exists()]
                self.logger.warning(f"批次 {batch_id} 缺少文件: {missing}")
        
        self.num_batches = len(self.batches)
        self.batch_size = self.batches[0]['batch_size'] if self.batches else 64
        self.total_samples = total_samples
        
        # 按批次ID排序
        self.batches.sort(key=lambda x: x['batch_id'])
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        """
        获取一个批次
        
        Returns:
            包含图像和标签的字典
        """
        batch_info = self.batches[idx]
        batch_id = batch_info['batch_id']
        
        # 构建文件路径（新结构：在mode子目录下）
        prefix = f"{self.mode}_{batch_id:04d}"
        image_path = self.data_root / "images" / self.mode / f"{prefix}.npy"
        heatmap_path = self.data_root / "labels" / self.mode / f"{prefix}_heatmaps.npy"
        offset_path = self.data_root / "labels" / self.mode / f"{prefix}_offsets.npy"
        weight_path = self.data_root / "labels" / self.mode / f"{prefix}_weights.npy"
        meta_path = self.data_root / "labels" / self.mode / f"{prefix}_meta.json"
        
        # 加载数据
        try:
            images = np.load(image_path)  # [B, 4, 256, 512]
            heatmaps = np.load(heatmap_path)  # [B, 2, 64, 128]
            offsets = np.load(offset_path)  # [B, 4, 64, 128]
            weights = np.load(weight_path)  # [B, 1, 64, 128]
            
            # 加载元数据
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            # 转换为张量
            images = torch.from_numpy(images).float()
            heatmaps = torch.from_numpy(heatmaps).float()
            offsets = torch.from_numpy(offsets).float()
            weights = torch.from_numpy(weights).float()
            
            # 提取坐标信息（从grid_coords和offsets_meta计算）
            # meta中包含grid_coords和offsets_meta，每个样本都有对应的信息
            batch_size = batch_info['batch_size']
            gap_coords_list = []
            slider_coords_list = []
            
            for i in range(batch_size):
                if i < len(meta.get('grid_coords', [])):
                    # 获取网格坐标和偏移量
                    grid_gap = meta['grid_coords'][i]['gap']  # [x, y]
                    grid_slider = meta['grid_coords'][i]['slider']  # [x, y]
                    offset_gap = meta['offsets_meta'][i]['gap']  # [dx, dy]
                    offset_slider = meta['offsets_meta'][i]['slider']  # [dx, dy]
                    
                    # 计算letterbox空间坐标: coord = 4 * (grid + 0.5 + offset)
                    gap_x = 4 * (grid_gap[0] + 0.5 + offset_gap[0])
                    gap_y = 4 * (grid_gap[1] + 0.5 + offset_gap[1])
                    slider_x = 4 * (grid_slider[0] + 0.5 + offset_slider[0])
                    slider_y = 4 * (grid_slider[1] + 0.5 + offset_slider[1])
                    
                    gap_coords_list.append([gap_x, gap_y])
                    slider_coords_list.append([slider_x, slider_y])
                else:
                    # 如果没有足够的元数据，使用默认值
                    gap_coords_list.append([0.0, 0.0])
                    slider_coords_list.append([0.0, 0.0])
            
            gap_coords = torch.tensor(gap_coords_list, dtype=torch.float32)
            slider_coords = torch.tensor(slider_coords_list, dtype=torch.float32)
            
            # weights 形状是 [B, 1, 64, 128]，squeeze 去掉通道维度得到 [B, 64, 128]
            weight_mask = weights.squeeze(1)  # [B, 64, 128]
            
            return {
                'image': images,
                'heatmap_gap': heatmaps[:, 0],
                'heatmap_slider': heatmaps[:, 1],
                'offset_gap': offsets[:, :2],
                'offset_slider': offsets[:, 2:],
                'weight_gap': weight_mask,  # [B, 64, 128] - gap和slider共享同一个权重掩码
                'weight_slider': weight_mask,  # [B, 64, 128] - gap和slider共享同一个权重掩码
                'gap_coords': gap_coords,
                'slider_coords': slider_coords,
                'batch_size': batch_info['batch_size']
            }
            
        except Exception as e:
            self.logger.error(f"加载批次 {batch_id} 失败: {e}")
            # 返回空批次
            return self._get_empty_batch(batch_info['batch_size'])
    
    def _get_empty_batch(self, batch_size):
        """创建空批次用于错误处理"""
        return {
            'image': torch.zeros(batch_size, 4, 256, 512),
            'heatmap_gap': torch.zeros(batch_size, 64, 128),
            'heatmap_slider': torch.zeros(batch_size, 64, 128),
            'offset_gap': torch.zeros(batch_size, 2, 64, 128),
            'offset_slider': torch.zeros(batch_size, 2, 64, 128),
            'weight_gap': torch.zeros(batch_size, 64, 128),
            'weight_slider': torch.zeros(batch_size, 64, 128),
            'gap_coords': torch.zeros(batch_size, 2),
            'slider_coords': torch.zeros(batch_size, 2),
            'batch_size': batch_size
        }


class NPYDataPipeline:
    """
    NPY格式数据管道
    """
    
    def __init__(self, config: Dict):
        """
        初始化数据管道
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger('NPYDataPipeline')
        
        # 数据集和加载器
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # 数据统计
        self.num_train_samples = 0
        self.num_val_samples = 0
    
    def setup(self):
        """设置数据管道"""
        self.logger.info("初始化NPY数据管道...")
        
        # 获取数据配置
        data_config = self.config.get('data', {})
        data_root = data_config.get('processed_dir', 'data/processed')
        
        # 尝试创建训练数据集
        # 1. 首先检查是否有子目录结构（新格式）
        train_dir = Path(data_root) / "images" / "train"
        if train_dir.exists():
            self.logger.info("检测到新目录结构，使用目录扫描模式...")
            try:
                self.train_dataset = NPYBatchDataset(
                    data_root=data_root,
                    mode='train',
                    index_file=None  # 不使用索引文件，扫描目录
                )
                self.num_train_samples = self.train_dataset.total_samples
                
                # 创建训练数据加载器（批次已预处理，所以batch_size=1）
                num_workers = self.config.get('train', {}).get('num_workers', 8)
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=1,  # 每次加载一个预处理的批次
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True if num_workers > 0 else False,
                    prefetch_factor=4 if num_workers > 0 else None,
                    collate_fn=self._collate_batch
                )
                
                self.logger.info(f"训练样本数: {self.num_train_samples}")
                self.logger.info(f"训练批次数: {len(self.train_loader)}")
            except Exception as e:
                self.logger.warning(f"加载训练数据失败: {e}")
        
        # 2. 否则尝试使用索引文件（旧格式）
        else:
            train_index = "train_index.json"
            train_index_path = Path(data_root) / train_index
            # 也尝试在split_info目录查找
            if not train_index_path.exists():
                train_index_path = Path(data_root) / "split_info" / train_index
            
            if train_index_path.exists():
                self.logger.info("使用索引文件模式...")
                self.train_dataset = NPYBatchDataset(
                    data_root=data_root,
                    mode='train',
                    index_file=str(train_index_path.relative_to(data_root))
                )
                self.num_train_samples = self.train_dataset.total_samples
                
                num_workers = self.config.get('train', {}).get('num_workers', 8)
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True if num_workers > 0 else False,
                    prefetch_factor=4 if num_workers > 0 else None,
                    collate_fn=self._collate_batch
                )
                
                self.logger.info(f"训练样本数: {self.num_train_samples}")
                self.logger.info(f"训练批次数: {len(self.train_loader)}")
        
        # 创建验证数据集（类似逻辑）
        val_dir = Path(data_root) / "images" / "val"
        if val_dir.exists():
            try:
                self.val_dataset = NPYBatchDataset(
                    data_root=data_root,
                    mode='val',
                    index_file=None
                )
                self.num_val_samples = self.val_dataset.total_samples
                
                num_workers = self.config.get('train', {}).get('num_workers', 8)
                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True if num_workers > 0 else False,
                    prefetch_factor=4 if num_workers > 0 else None,
                    collate_fn=self._collate_batch
                )
                
                self.logger.info(f"验证样本数: {self.num_val_samples}")
                self.logger.info(f"验证批次数: {len(self.val_loader)}")
            except Exception as e:
                self.logger.warning(f"加载验证数据失败: {e}")
        else:
            val_index = "val_index.json"
            val_index_path = Path(data_root) / val_index
            if not val_index_path.exists():
                val_index_path = Path(data_root) / "split_info" / val_index
            
            if val_index_path.exists():
                self.val_dataset = NPYBatchDataset(
                    data_root=data_root,
                    mode='val',
                    index_file=str(val_index_path.relative_to(data_root))
                )
                self.num_val_samples = self.val_dataset.total_samples
                
                num_workers = self.config.get('train', {}).get('num_workers', 8)
                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True if num_workers > 0 else False,
                    prefetch_factor=4 if num_workers > 0 else None,
                    collate_fn=self._collate_batch
                )
                
                self.logger.info(f"验证样本数: {self.num_val_samples}")
                self.logger.info(f"验证批次数: {len(self.val_loader)}")
    
    def _collate_batch(self, batch_list):
        """
        整理批次数据
        
        由于每个item已经是一个批次，这里只需要解包第一个元素
        """
        if len(batch_list) != 1:
            raise ValueError(f"期望batch_list长度为1，但得到{len(batch_list)}")
        
        batch = batch_list[0]
        
        # 重新组织为训练需要的格式
        return {
            'image': batch['image'],
            'gap_coords': batch['gap_coords'],
            'slider_coords': batch['slider_coords'],
            'heatmap_gap': batch['heatmap_gap'],
            'heatmap_slider': batch['heatmap_slider'],
            'offset_gap': batch['offset_gap'],
            'offset_slider': batch['offset_slider'],
            'weight_gap': batch['weight_gap'],
            'weight_slider': batch['weight_slider']
        }
    
    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        if self.train_loader is None:
            self.setup()
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        if self.val_loader is None:
            self.setup()
        return self.val_loader
    
    def get_test_loader(self) -> Optional[DataLoader]:
        """
        获取测试数据加载器
        
        Returns:
            测试数据加载器或None
        """
        data_config = self.config.get('data', {})
        data_root = data_config.get('processed_dir', 'data/processed')
        
        # 检查测试数据目录
        test_dir = Path(data_root) / "images" / "test"
        if test_dir.exists():
            try:
                test_dataset = NPYBatchDataset(
                    data_root=data_root,
                    mode='test',
                    index_file=None
                )
                
                num_workers = self.config.get('train', {}).get('num_workers', 8)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True if num_workers > 0 else False,
                    prefetch_factor=4 if num_workers > 0 else None,
                    collate_fn=self._collate_batch
                )
                
                self.logger.info(f"测试样本数: {test_dataset.total_samples}")
                self.logger.info(f"测试批次数: {len(test_loader)}")
                return test_loader
            except Exception as e:
                self.logger.warning(f"加载测试数据失败: {e}")
                return None
        else:
            self.logger.info("未找到测试数据目录")
            return None
    
    def get_batch_info(self) -> Dict[str, int]:
        """
        获取批次信息
        
        Returns:
            批次信息字典
        """
        return {
            'train_batches': len(self.train_loader) if self.train_loader else 0,
            'val_batches': len(self.val_loader) if self.val_loader else 0,
            'batch_size': self.config['train']['batch_size'],
            'train_samples': self.num_train_samples,
            'val_samples': self.num_val_samples
        }


if __name__ == "__main__":
    # 测试NPY数据加载器
    config = {
        'data': {
            'processed_dir': 'data/processed'
        },
        'train': {
            'batch_size': 64,
            'num_workers': 2,
            'pin_memory': True
        }
    }
    
    pipeline = NPYDataPipeline(config)
    pipeline.setup()
    
    # 测试一个批次
    if pipeline.train_loader:
        for batch in pipeline.train_loader:
            print(f"批次形状:")
            print(f"  图像: {batch['image'].shape}")
            print(f"  缺口坐标: {batch['gap_coords'].shape}")
            print(f"  滑块坐标: {batch['slider_coords'].shape}")
            print(f"  缺口热力图: {batch['heatmap_gap'].shape}")
            print(f"  滑块热力图: {batch['heatmap_slider'].shape}")
            break