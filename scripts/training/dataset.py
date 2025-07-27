# -*- coding: utf-8 -*-
"""
数据集定义 - 独立模块以解决Windows多进程pickle问题
"""
import re
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset


class OptimizedCaptchaDataset(Dataset):
    """优化的滑块验证码数据集类，适用于Windows"""
    
    def __init__(self, data_dir, annotations_file=None, split='train', transform=None):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_absolute():
            # 如果是相对路径，转换为绝对路径
            # 假设数据目录在项目根目录下
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            self.data_dir = project_root / self.data_dir
        
        self.split = split
        self.transform = transform
        
        # 直接从文件夹读取图片
        image_dir = self.data_dir / split
        print(f"Looking for images in: {image_dir}")
        self.image_files = list(image_dir.glob('*.png'))
        
        # 文件名解析正则表达式
        # 格式: Pic{编号}_Bgx{X}Bgy{Y}_Sdx{X}Sdy{Y}_{hash}.png
        self.pattern = re.compile(
            r'Pic(\d+)_Bgx(\d+)Bgy(\d+)_Sdx(\d+)Sdy(\d+)_([a-f0-9]+)\.png'
        )
        
        # 解析所有文件名
        self.samples = []
        for file_path in self.image_files:
            match = self.pattern.match(file_path.name)
            if match:
                pic_id, bg_x, bg_y, sd_x, sd_y, hash_val = match.groups()
                self.samples.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'bg_center': [int(bg_x), int(bg_y)],
                    'slider_center': [int(sd_x), int(sd_y)],
                    'pic_id': int(pic_id),
                    'hash': hash_val
                })
        
        print(f"Loaded {len(self.samples)} {split} samples from {image_dir}")
    
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