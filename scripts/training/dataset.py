# -*- coding: utf-8 -*-
"""
数据集定义 - 独立模块以解决Windows多进程pickle问题
"""
import json
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset


class OptimizedCaptchaDataset(Dataset):
    """优化的滑块验证码数据集类，适用于Windows"""
    
    def __init__(self, data_dir, annotations_file, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # 加载标注数据
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations_list = json.load(f)
        
        # 处理数据
        self.samples = []
        for item in annotations_list:
            filename = item['filename']
            file_path = self.data_dir / split / filename
            if file_path.exists():
                self.samples.append({
                    'filename': filename,
                    'path': str(file_path),
                    'bg_center': item['bg_center'],
                    'slider_center': item['sd_center'],
                    'shape': item['shape'],
                    'size': item['size']
                })
        
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        try:
            image = cv2.imread(sample['path'])
            if image is None:
                raise ValueError(f"Failed to load image: {sample['path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {sample['path']}: {e}")
            # 返回一个随机样本
            return self.__getitem__(np.random.randint(0, len(self)))
        
        # 准备标签
        bg_center = np.array(sample['bg_center'], dtype=np.float32)
        slider_center = np.array(sample['slider_center'], dtype=np.float32)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'bg_center': bg_center,
            'slider_center': slider_center,
            'filename': sample['filename']
        }