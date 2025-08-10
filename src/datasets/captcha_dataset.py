# -*- coding: utf-8 -*-
"""
CAPTCHA数据集模块
负责数据加载、预处理和增强
"""
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class CaptchaDataset(Dataset):
    """滑块验证码数据集"""
    
    def __init__(
        self,
        annotations: List[Dict],
        data_dir: str = ".",
        transform_config: Optional[Dict] = None,
        is_train: bool = True
    ):
        """
        初始化数据集
        
        Args:
            annotations: 标注列表
            data_dir: 数据根目录
            transform_config: 数据增强配置
            is_train: 是否为训练模式
        """
        self.annotations = annotations
        self.data_dir = Path(data_dir)
        self.transform_config = transform_config or {}
        self.is_train = is_train
        
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        获取单个样本
        
        Returns:
            image: 图像张量 [3, H, W]
            target: 目标字典，包含坐标和元信息
        """
        ann = self.annotations[idx]
        
        # 加载图像
        image = self._load_image(ann['filename'])
        
        # 构建目标
        target = self._build_target(ann)
        
        # 数据增强（仅训练时）
        if self.is_train and self.transform_config:
            image = self._apply_augmentation(image)
        
        # 转换为张量
        image = self._to_tensor(image)
        
        return image, target
    
    def _load_image(self, filename: str) -> np.ndarray:
        """加载并预处理图像"""
        img_path = self.data_dir / "data" / "captchas" / filename
        
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Cannot load image: {img_path}")
        
        # BGR转RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _build_target(self, ann: Dict) -> Dict:
        """构建目标字典"""
        return {
            'gap_coord': torch.tensor([ann['bg_x'], ann['bg_y']], dtype=torch.float32),
            'slider_coord': torch.tensor([ann['sd_x'], ann['sd_y']], dtype=torch.float32),
            'shape': ann.get('shape', 'circle'),
            'size': ann.get('size', 40),
            'confusion_type': ann.get('confusion_type', 'none'),
            'filename': ann['filename']
        }
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """应用数据增强"""
        # 亮度调整
        if self.transform_config.get('random_brightness', 0) > 0:
            brightness = self.transform_config['random_brightness']
            delta = np.random.uniform(-brightness, brightness)
            image = np.clip(image.astype(np.float32) + delta * 255, 0, 255).astype(np.uint8)
        
        # 对比度调整
        if self.transform_config.get('random_contrast', 0) > 0:
            contrast = self.transform_config['random_contrast']
            alpha = 1.0 + np.random.uniform(-contrast, contrast)
            image = np.clip(alpha * image.astype(np.float32), 0, 255).astype(np.uint8)
        
        # 高斯噪声
        if self.transform_config.get('gaussian_noise', 0) > 0:
            noise_level = self.transform_config['gaussian_noise']
            noise = np.random.randn(*image.shape) * noise_level * 255
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """将numpy数组转换为torch张量"""
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        # 归一化到[0, 1]
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image)


def create_data_loaders(config: Dict):
    """
    创建数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    # 加载数据分割
    split_data = load_data_split(config)
    
    # 创建数据集
    train_dataset = CaptchaDataset(
        split_data['train'],
        data_dir=config['data']['data_dir'],
        transform_config=config['data'].get('augmentation'),
        is_train=True
    )
    
    val_dataset = CaptchaDataset(
        split_data['val'],
        data_dir=config['data']['data_dir'],
        transform_config=None,
        is_train=False
    )
    
    test_dataset = CaptchaDataset(
        split_data['test'],
        data_dir=config['data']['data_dir'],
        transform_config=None,
        is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['testing'].get('test_batch_size', config['data']['batch_size']),
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    print(f"Dataset Statistics:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")
    
    return train_loader, val_loader, test_loader


def load_data_split(config: Dict) -> Dict:
    """
    加载或生成数据分割
    
    Returns:
        包含train/val/test标注的字典
    """
    split_file = Path(config['data']['split_file'])
    
    # 如果分割文件存在，直接加载
    if split_file.exists():
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        return {
            'train': split_data['splits']['train']['annotations'],
            'val': split_data['splits']['val']['annotations'],
            'test': split_data['splits']['test']['annotations']
        }
    
    # 否则从标注文件创建分割
    print(f"Split file not found: {split_file}")
    print("Creating data split from annotations...")
    
    annotations_file = Path(config['data']['annotations_file'])
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    with open(annotations_file, 'r') as f:
        all_annotations = json.load(f)
    
    # 简单分割
    n = len(all_annotations)
    train_end = int(n * config['data']['train_ratio'])
    val_end = train_end + int(n * config['data']['val_ratio'])
    
    return {
        'train': all_annotations[:train_end],
        'val': all_annotations[train_end:val_end],
        'test': all_annotations[val_end:]
    }