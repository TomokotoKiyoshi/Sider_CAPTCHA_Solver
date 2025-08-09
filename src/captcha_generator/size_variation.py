# -*- coding: utf-8 -*-
"""
尺寸变化模块 - 用于验证码生成
可被 generate_captchas_with_components.py 调用
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Optional


class SizeVariation:
    """验证码尺寸变化处理器"""
    
    def __init__(self, size_config):
        """
        初始化尺寸变化处理器
        
        Args:
            size_config: CaptchaSizeConfig实例
        """
        self.config = size_config
    
    def apply_size_variation(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict]:
        """
        应用尺寸变化到图像
        
        Args:
            image: 输入图像 (H, W, C)
            target_size: 目标尺寸 (width, height)，如果为None则随机生成
            
        Returns:
            resized_image: 调整尺寸后的图像
            size_info: 尺寸信息字典
        """
        # 生成或使用指定的目标尺寸
        if target_size is None:
            target_size = self.config.generate_random_size()
        
        # 调整图像尺寸
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 记录尺寸信息
        size_info = {
            'original_size': (image.shape[1], image.shape[0]),  # (width, height)
            'target_size': target_size,
            'width': target_size[0],
            'height': target_size[1],
            'aspect_ratio': target_size[0] / target_size[1]
        }
        
        return resized_image, size_info
    
    def scale_coordinates(self, coords: Tuple[float, float], 
                         original_size: Tuple[int, int],
                         new_size: Tuple[int, int]) -> Tuple[float, float]:
        """
        根据尺寸变化缩放坐标
        
        Args:
            coords: 原始坐标 (x, y)
            original_size: 原始尺寸 (width, height)
            new_size: 新尺寸 (width, height)
            
        Returns:
            缩放后的坐标 (x, y)
        """
        x, y = coords
        scale_x = new_size[0] / original_size[0]
        scale_y = new_size[1] / original_size[1]
        
        new_x = x * scale_x
        new_y = y * scale_y
        
        return (new_x, new_y)
    
    def scale_slider_size(self, slider_size: int,
                         original_size: Tuple[int, int],
                         new_size: Tuple[int, int]) -> int:
        """
        根据尺寸变化缩放滑块大小
        
        Args:
            slider_size: 原始滑块大小
            original_size: 原始尺寸 (width, height)
            new_size: 新尺寸 (width, height)
            
        Returns:
            缩放后的滑块大小
        """
        # 使用平均缩放比例
        scale = (new_size[0] / original_size[0] + new_size[1] / original_size[1]) / 2
        new_slider_size = int(slider_size * scale)
        
        # 确保滑块大小在合理范围内
        min_size = int(new_size[1] * 0.15)  # 最小为高度的15%
        max_size = int(new_size[1] * 0.35)  # 最大为高度的35%
        new_slider_size = max(min_size, min(max_size, new_slider_size))
        
        # 确保为偶数
        if new_slider_size % 2 != 0:
            new_slider_size += 1
            
        return new_slider_size
    
    def update_metadata_with_size(self, metadata: Dict, size_info: Dict) -> Dict:
        """
        更新元数据以包含尺寸信息
        
        Args:
            metadata: 原始元数据
            size_info: 尺寸信息
            
        Returns:
            更新后的元数据
        """
        metadata['size_info'] = size_info
        metadata['image_width'] = size_info['width']
        metadata['image_height'] = size_info['height']
        
        # 如果有坐标信息，添加归一化坐标（便于不同尺寸间比较）
        if 'gap_x' in metadata and 'gap_y' in metadata:
            metadata['gap_x_norm'] = metadata['gap_x'] / size_info['width']
            metadata['gap_y_norm'] = metadata['gap_y'] / size_info['height']
        
        if 'slider_x' in metadata and 'slider_y' in metadata:
            metadata['slider_x_norm'] = metadata['slider_x'] / size_info['width']
            metadata['slider_y_norm'] = metadata['slider_y'] / size_info['height']
        
        return metadata