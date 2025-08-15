#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据管道系统 - 负责数据加载、预处理和批次管理
"""
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import cv2
import json
import logging
from PIL import Image
import random
import hashlib


class CaptchaDataset(Dataset):
    """
    滑块验证码数据集
    
    数据格式：
    - 图像文件命名: Pic{XXXX}_Bgx{X}Bgy{Y}_Sdx{X}Sdy{Y}_{hash}.png
    - 图像尺寸: 320x160 (背景) + 滑块
    - 坐标为中心点坐标
    """
    
    def __init__(self, 
                 data_dir: str, 
                 mode: str = 'train',
                 transform: Optional[Any] = None,
                 augment: bool = True):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            mode: 'train' 或 'val'
            transform: 图像变换
            augment: 是否进行数据增强
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.augment = augment and (mode == 'train')
        
        # 设置日志
        self.logger = logging.getLogger(f'CaptchaDataset.{mode}')
        
        # 加载数据样本
        self.samples = self._load_samples()
        
        self.logger.info(f"加载了 {len(self.samples)} 个{mode}样本")
    
    def _load_samples(self) -> List[Dict]:
        """
        加载数据样本
        
        Returns:
            样本列表，每个样本包含图像路径和标注信息
        """
        samples = []
        
        # 查找所有PNG图像
        image_paths = list(self.data_dir.glob("*.png"))
        
        for img_path in image_paths:
            # 解析文件名获取标注信息
            # 格式: Pic{XXXX}_Bgx{X}Bgy{Y}_Sdx{X}Sdy{Y}_{hash}.png
            filename = img_path.stem
            parts = filename.split('_')
            
            if len(parts) < 4:
                self.logger.warning(f"跳过格式错误的文件: {img_path}")
                continue
            
            try:
                # 解析缺口坐标
                bg_part = parts[1]  # Bgx{X}Bgy{Y}
                bgx_idx = bg_part.index('Bgx') + 3
                bgy_idx = bg_part.index('Bgy')
                bgx = float(bg_part[bgx_idx:bgy_idx])
                bgy = float(bg_part[bgy_idx+3:])
                
                # 解析滑块坐标
                sd_part = parts[2]  # Sdx{X}Sdy{Y}
                sdx_idx = sd_part.index('Sdx') + 3
                sdy_idx = sd_part.index('Sdy')
                sdx = float(sd_part[sdx_idx:sdy_idx])
                sdy = float(sd_part[sdy_idx+3:])
                
                sample = {
                    'image_path': str(img_path),
                    'gap_x': bgx,
                    'gap_y': bgy,
                    'slider_x': sdx,
                    'slider_y': sdy,
                    'filename': filename
                }
                
                samples.append(sample)
                
            except (ValueError, IndexError) as e:
                self.logger.warning(f"解析文件名失败: {img_path}, 错误: {e}")
                continue
        
        # 排序以确保一致性
        samples.sort(key=lambda x: x['filename'])
        
        return samples
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含图像和标注的字典
        """
        sample = self.samples[idx]
        
        # 加载图像
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise ValueError(f"无法加载图像: {sample['image_path']}")
        
        # BGR转RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 确保图像尺寸正确
        height, width = image.shape[:2]
        
        # 数据增强（训练时）
        if self.augment:
            image, gap_coords, slider_coords = self._augment(
                image,
                [sample['gap_x'], sample['gap_y']],
                [sample['slider_x'], sample['slider_y']]
            )
        else:
            gap_coords = [sample['gap_x'], sample['gap_y']]
            slider_coords = [sample['slider_x'], sample['slider_y']]
        
        # 转换为张量
        # 图像归一化到[0, 1]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # 添加padding mask通道（全1）
        padding_mask = torch.ones(1, height, width)
        image_tensor = torch.cat([image_tensor, padding_mask], dim=0)  # [4, H, W]
        
        # 坐标转换为张量
        gap_coords = torch.tensor(gap_coords, dtype=torch.float32)
        slider_coords = torch.tensor(slider_coords, dtype=torch.float32)
        
        return {
            'image': image_tensor,           # [4, 160, 320]
            'gap_coords': gap_coords,        # [2]
            'slider_coords': slider_coords,  # [2]
            'filename': sample['filename']
        }
    
    def _augment(self, 
                 image: np.ndarray, 
                 gap_coords: List[float], 
                 slider_coords: List[float]) -> Tuple[np.ndarray, List[float], List[float]]:
        """
        数据增强
        
        Args:
            image: 原始图像
            gap_coords: 缺口坐标
            slider_coords: 滑块坐标
            
        Returns:
            增强后的图像和坐标
        """
        height, width = image.shape[:2]
        
        # 1. 随机亮度调整
        if random.random() < 0.3:
            brightness = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # 2. 随机对比度调整
        if random.random() < 0.3:
            contrast = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        # 3. 随机噪声
        if random.random() < 0.2:
            noise = np.random.randn(*image.shape) * 5
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # 4. 随机水平翻转
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            gap_coords[0] = width - gap_coords[0]
            slider_coords[0] = width - slider_coords[0]
        
        # 5. 轻微的随机平移（不改变目标位置的标注）
        if random.random() < 0.3:
            max_shift = 5
            shift_x = random.randint(-max_shift, max_shift)
            shift_y = random.randint(-max_shift, max_shift)
            
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            image = cv2.warpAffine(image, M, (width, height), 
                                  borderMode=cv2.BORDER_REPLICATE)
            
            # 更新坐标
            gap_coords[0] += shift_x
            gap_coords[1] += shift_y
            slider_coords[0] += shift_x
            slider_coords[1] += shift_y
            
            # 确保坐标在有效范围内
            gap_coords[0] = np.clip(gap_coords[0], 0, width)
            gap_coords[1] = np.clip(gap_coords[1], 0, height)
            slider_coords[0] = np.clip(slider_coords[0], 0, width)
            slider_coords[1] = np.clip(slider_coords[1], 0, height)
        
        return image, gap_coords, slider_coords


class DataPipeline:
    """
    数据管道管理器
    
    负责：
    1. 创建训练和验证数据集
    2. 创建DataLoader
    3. 管理批次数据
    4. 提供数据统计信息
    """
    
    def __init__(self, config: Dict):
        """
        初始化数据管道
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger('DataPipeline')
        
        # 数据集和加载器
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # 数据统计
        self.num_train_samples = 0
        self.num_val_samples = 0
    
    def setup(self, train_dir: str = 'data/train', val_dir: str = 'data/val'):
        """
        设置数据管道
        
        Args:
            train_dir: 训练数据目录
            val_dir: 验证数据目录
        """
        self.logger.info("初始化数据管道...")
        
        # 创建训练数据集
        self.train_dataset = CaptchaDataset(
            data_dir=train_dir,
            mode='train',
            augment=True
        )
        self.num_train_samples = len(self.train_dataset)
        
        # 创建验证数据集
        self.val_dataset = CaptchaDataset(
            data_dir=val_dir,
            mode='val',
            augment=False
        )
        self.num_val_samples = len(self.val_dataset)
        
        # 创建训练数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=self.config['train'].get('num_workers', 4),
            pin_memory=self.config['train'].get('pin_memory', True),
            persistent_workers=True if self.config['train'].get('num_workers', 4) > 0 else False,
            prefetch_factor=2 if self.config['train'].get('num_workers', 4) > 0 else None,
            drop_last=True  # 丢弃不完整的批次
        )
        
        # 创建验证数据加载器
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False
        )
        
        self.logger.info(f"训练样本数: {self.num_train_samples}")
        self.logger.info(f"验证样本数: {self.num_val_samples}")
        self.logger.info(f"训练批次数: {len(self.train_loader)}")
        self.logger.info(f"验证批次数: {len(self.val_loader)}")
    
    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        if self.train_loader is None:
            raise RuntimeError("数据管道未初始化，请先调用setup()")
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        if self.val_loader is None:
            raise RuntimeError("数据管道未初始化，请先调用setup()")
        return self.val_loader
    
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
    
    def test_batch(self) -> Dict[str, torch.Tensor]:
        """
        获取一个测试批次用于验证
        
        Returns:
            测试批次数据
        """
        if self.train_loader is None:
            self.setup()
        
        # 获取一个批次
        for batch in self.train_loader:
            return batch
        
        raise RuntimeError("无法获取测试批次")


if __name__ == "__main__":
    # 测试数据管道
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    # 模拟配置
    config = {
        'train': {
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True
        }
    }
    
    # 创建数据管道
    pipeline = DataPipeline(config)
    
    # 检查数据目录是否存在
    train_dir = Path('data/train')
    val_dir = Path('data/val')
    
    if train_dir.exists() and val_dir.exists():
        # 设置数据管道
        pipeline.setup()
        
        # 获取批次信息
        batch_info = pipeline.get_batch_info()
        print("\n批次信息:")
        for key, value in batch_info.items():
            print(f"  {key}: {value}")
        
        # 测试一个批次
        test_batch = pipeline.test_batch()
        print("\n测试批次:")
        for key, value in test_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        print("\n数据管道测试通过！")
    else:
        print(f"警告: 数据目录不存在")
        print(f"  训练目录: {train_dir}")
        print(f"  验证目录: {val_dir}")
        print("请先生成训练数据")