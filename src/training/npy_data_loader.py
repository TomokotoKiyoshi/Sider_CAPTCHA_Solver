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
import platform
import yaml


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
                # weights.npy不再需要，权重已集成在图像第4通道
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
        # weight_path已移除，权重从图像第4通道提取
        meta_path = self.data_root / "labels" / self.mode / f"{prefix}_meta.json"
        
        # 加载数据
        try:
            images = np.load(image_path)  # [B, 4, 256, 512] - 第4通道是padding mask
            heatmaps = np.load(heatmap_path)  # [B, 2, 64, 128]
            offsets = np.load(offset_path)  # [B, 4, 64, 128]
            
            # 从图像第4通道提取权重掩码并下采样到1/4分辨率
            # padding_mask: [B, 256, 512] -> weights: [B, 1, 64, 128]
            padding_mask = images[:, 3, :, :]  # 提取第4通道 [B, 256, 512]
            
            # 使用平均池化下采样（4x4窗口）
            batch_size = padding_mask.shape[0]
            h_out, w_out = 64, 128  # 目标尺寸
            weights = np.zeros((batch_size, 1, h_out, w_out), dtype=np.float32)
            
            for b in range(batch_size):
                for i in range(h_out):
                    for j in range(w_out):
                        # 计算4x4窗口的平均值
                        window = padding_mask[b, i*4:(i+1)*4, j*4:(j+1)*4]
                        weights[b, 0, i, j] = np.mean(window)
            
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
            
            # 提取混淆缺口信息（必须存在）
            if 'confusing_gaps' not in meta:
                raise ValueError(f"批次 {batch_id} 的元数据缺少 'confusing_gaps' 字段，无法计算硬负样本损失")
            confusing_gaps = meta['confusing_gaps']
            
            # 提取角度信息（必须存在）
            if 'gap_angles' not in meta:
                raise ValueError(f"批次 {batch_id} 的元数据缺少 'gap_angles' 字段，无法计算角度损失")
            gap_angles = meta['gap_angles']
            
            # 验证数据长度一致性
            if len(confusing_gaps) != batch_size:
                raise ValueError(f"批次 {batch_id}: confusing_gaps 长度 ({len(confusing_gaps)}) 与批次大小 ({batch_size}) 不匹配")
            if len(gap_angles) != batch_size:
                raise ValueError(f"批次 {batch_id}: gap_angles 长度 ({len(gap_angles)}) 与批次大小 ({batch_size}) 不匹配")
            
            # 将角度信息转换为张量
            gap_angles_tensor = torch.tensor(gap_angles, dtype=torch.float32)
            
            # 处理混淆缺口坐标（转换为特征图坐标）
            # confusing_gaps 是一个列表，每个元素是该样本的混淆缺口坐标列表
            # 需要将原图坐标转换为1/4分辨率的特征图坐标
            confusing_gaps_scaled = []
            for sample_gaps in confusing_gaps:
                scaled_gaps = []
                for gap in sample_gaps:
                    # gap 是 [x, y] 坐标，需要除以4转换到特征图尺度
                    scaled_gaps.append([gap[0] / 4.0, gap[1] / 4.0])
                confusing_gaps_scaled.append(scaled_gaps)
            
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
                'batch_size': batch_info['batch_size'],
                # 新增：混淆缺口和角度信息
                'confusing_gaps': confusing_gaps_scaled,  # 混淆缺口坐标（特征图尺度）
                'gap_angles': gap_angles_tensor,  # 缺口旋转角度（弧度）
                'angle': gap_angles_tensor  # 兼容训练引擎的命名
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
            'batch_size': batch_size,
            # 新增：混淆缺口和角度信息（空批次时使用默认值）
            'confusing_gaps': [[] for _ in range(batch_size)],  # 每个样本的空列表
            'gap_angles': torch.zeros(batch_size),  # 零角度
            'angle': torch.zeros(batch_size)  # 兼容训练引擎的命名
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
        
        # 缓存NPY批次大小
        self._npy_batch_size_cache = None
    
    def _get_npy_batch_size(self) -> int:
        """
        获取NPY文件的批次大小
        
        Returns:
            每个NPY文件包含的样本数
        """
        if self._npy_batch_size_cache is not None:
            return self._npy_batch_size_cache
            
        # 尝试从预处理配置文件读取
        try:
            preprocessing_config_path = Path('config/preprocessing_config.yaml')
            if preprocessing_config_path.exists():
                with open(preprocessing_config_path, 'r', encoding='utf-8') as f:
                    preprocessing_config = yaml.safe_load(f)
                    batch_size = preprocessing_config.get('dataset', {}).get('batch_size', 64)
                    self.logger.info(f"从预处理配置读取NPY批次大小: {batch_size}")
                    self._npy_batch_size_cache = batch_size
                    return batch_size
        except Exception as e:
            self.logger.warning(f"无法读取预处理配置: {e}")
        
        # 尝试从数据集检测
        if self.train_dataset and hasattr(self.train_dataset, 'batch_size'):
            batch_size = self.train_dataset.batch_size
            self._npy_batch_size_cache = batch_size
            self.logger.info(f"从数据集检测NPY批次大小: {batch_size}")
            return batch_size
        
        # 默认值
        default_batch_size = 64
        self.logger.warning(f"使用默认NPY批次大小: {default_batch_size}")
        self._npy_batch_size_cache = default_batch_size
        return default_batch_size
    
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
                
                # 创建训练数据加载器（优化：加载多个NPY批次）
                num_workers = self.config.get('train', {}).get('num_workers', 4)
                prefetch_factor = self.config.get('train', {}).get('prefetch_factor', 8)
                persistent_workers = self.config.get('train', {}).get('persistent_workers', True)
                
                # 动态调整batch_size基于NPY文件大小
                # 从预处理配置或数据集元数据中获取实际的NPY批次大小
                npy_batch_size = self._get_npy_batch_size()  # 每个NPY文件的实际样本数
                target_batch_size = self.config.get('train', {}).get('batch_size', 256)
                dataloader_batch_size = max(1, target_batch_size // npy_batch_size)  # 例如：256/64 = 4
                
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=dataloader_batch_size,  # 加载多个NPY文件以达到目标批次
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=persistent_workers if num_workers > 0 else False,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    collate_fn=self._collate_multiple_batches if dataloader_batch_size > 1 else self._collate_batch
                )
                
                self.logger.info(f"训练配置: NPY批次={npy_batch_size}, 目标批次={target_batch_size}, DataLoader批次={dataloader_batch_size}")
                self.logger.info(f"每次加载 {dataloader_batch_size} 个NPY文件，总共 {dataloader_batch_size * npy_batch_size} 个样本")
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
                
                num_workers = self.config.get('train', {}).get('num_workers', 4)
                prefetch_factor = self.config.get('train', {}).get('prefetch_factor', 8)
                persistent_workers = self.config.get('train', {}).get('persistent_workers', True)
                
                # 动态调整batch_size
                npy_batch_size = self._get_npy_batch_size()
                target_batch_size = self.config.get('train', {}).get('batch_size', 512)
                dataloader_batch_size = max(1, target_batch_size // npy_batch_size)
                
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=dataloader_batch_size,  # 加载多个NPY文件
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=persistent_workers if num_workers > 0 else False,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    collate_fn=self._collate_multiple_batches if dataloader_batch_size > 1 else self._collate_batch
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
                
                # Windows fix: 验证时使用num_workers=0避免pickle错误
                import platform
                if platform.system() == 'Windows':
                    num_workers = 0  # Windows验证时强制单进程
                    prefetch_factor = None
                    persistent_workers = False
                    self.logger.info("Windows检测：验证使用单进程加载 (num_workers=0)")
                else:
                    num_workers = self.config.get('train', {}).get('num_workers', 4)
                    prefetch_factor = self.config.get('train', {}).get('prefetch_factor', 8)
                    persistent_workers = self.config.get('train', {}).get('persistent_workers', True)
                
                # 动态调整batch_size
                npy_batch_size = self._get_npy_batch_size()
                target_batch_size = self.config.get('train', {}).get('batch_size', 512)
                dataloader_batch_size = max(1, target_batch_size // npy_batch_size)  # 验证也使用相同策略
                
                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=dataloader_batch_size,  # 加载多个NPY文件
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=persistent_workers if num_workers > 0 else False,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    collate_fn=self._collate_multiple_batches if dataloader_batch_size > 1 else self._collate_batch
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
                
                # Windows fix: 验证时使用num_workers=0避免pickle错误
                import platform
                if platform.system() == 'Windows':
                    num_workers = 0  # Windows验证时强制单进程
                    prefetch_factor = None
                    persistent_workers = False
                    self.logger.info("Windows检测：验证使用单进程加载 (num_workers=0)")
                else:
                    num_workers = self.config.get('train', {}).get('num_workers', 4)
                    prefetch_factor = self.config.get('train', {}).get('prefetch_factor', 8)
                    persistent_workers = self.config.get('train', {}).get('persistent_workers', True)
                
                # 动态调整batch_size
                npy_batch_size = self._get_npy_batch_size()
                target_batch_size = self.config.get('train', {}).get('batch_size', 512)
                dataloader_batch_size = max(1, target_batch_size // npy_batch_size)  # 验证也使用相同策略
                
                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=dataloader_batch_size,  # 加载多个NPY文件
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=persistent_workers if num_workers > 0 else False,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    collate_fn=self._collate_multiple_batches if dataloader_batch_size > 1 else self._collate_batch
                )
                
                self.logger.info(f"验证样本数: {self.num_val_samples}")
                self.logger.info(f"验证批次数: {len(self.val_loader)}")
    
    def _collate_batch(self, batch_list):
        """
        整理批次数据（旧版本，单批次加载）
        
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
            'weight_slider': batch['weight_slider'],
            # 添加混淆缺口和角度信息
            'confusing_gaps': batch.get('confusing_gaps', []),
            'gap_angles': batch.get('gap_angles'),
            'angle': batch.get('angle')
        }
    
    def _collate_multiple_batches(self, batch_list):
        """
        整理多个批次数据（优化版本，支持加载多个NPY文件）
        
        将多个NPY批次合并成一个大批次
        """
        if not batch_list:
            raise ValueError("batch_list不能为空")
        
        # 如果只有一个批次，直接返回
        if len(batch_list) == 1:
            return self._collate_batch(batch_list)
        
        # 合并多个批次
        merged_batch = {}
        
        # 需要拼接的张量字段
        tensor_fields = [
            'image', 'gap_coords', 'slider_coords',
            'heatmap_gap', 'heatmap_slider',
            'offset_gap', 'offset_slider',
            'weight_gap', 'weight_slider',
            'gap_angles', 'angle'
        ]
        
        # 拼接张量
        for field in tensor_fields:
            if field in batch_list[0]:
                tensors = [batch[field] for batch in batch_list if field in batch]
                if tensors:
                    merged_batch[field] = torch.cat(tensors, dim=0)
        
        # 处理混淆缺口列表（需要展平）
        confusing_gaps = []
        for batch in batch_list:
            if 'confusing_gaps' in batch:
                confusing_gaps.extend(batch['confusing_gaps'])
        merged_batch['confusing_gaps'] = confusing_gaps
        
        return merged_batch
    
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
                
                # Windows fix: 测试时使用num_workers=0避免pickle错误
                if platform.system() == 'Windows':
                    num_workers = 0  # Windows测试时强制单进程
                    prefetch_factor = None
                    persistent_workers = False
                else:
                    num_workers = self.config.get('train', {}).get('num_workers', 4)
                    prefetch_factor = self.config.get('train', {}).get('prefetch_factor', 8)
                    persistent_workers = self.config.get('train', {}).get('persistent_workers', True)
                
                # 动态调整batch_size
                npy_batch_size = self._get_npy_batch_size()
                target_batch_size = self.config.get('train', {}).get('batch_size', 512)
                dataloader_batch_size = max(1, target_batch_size // npy_batch_size)
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=dataloader_batch_size,  # 加载多个NPY文件
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=persistent_workers if num_workers > 0 else False,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    collate_fn=self._collate_multiple_batches if dataloader_batch_size > 1 else self._collate_batch
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