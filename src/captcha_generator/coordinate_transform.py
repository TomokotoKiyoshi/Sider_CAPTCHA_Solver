# -*- coding: utf-8 -*-
"""
坐标变换系统
实现 padding → resize 的两步变换流程
确保训练和推理的坐标一致性
"""
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class TransformParams:
    """坐标变换参数"""
    # 原图尺寸
    original_size: Tuple[int, int]  # (W0, H0)
    
    # Padding参数
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    
    # Padding后的尺寸
    padded_size: Tuple[int, int]  # (W_pad, H_pad)
    
    # 缩放因子
    scale: float
    
    # 目标尺寸
    target_size: Tuple[int, int]  # (W_target, H_target)
    
    def to_dict(self) -> Dict:
        """转换为字典格式，便于保存"""
        return {
            'original_size': self.original_size,
            'padding': {
                'left': self.pad_left,
                'top': self.pad_top,
                'right': self.pad_right,
                'bottom': self.pad_bottom
            },
            'padded_size': self.padded_size,
            'scale': self.scale,
            'target_size': self.target_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TransformParams':
        """从字典恢复"""
        return cls(
            original_size=tuple(data['original_size']),
            pad_left=data['padding']['left'],
            pad_top=data['padding']['top'],
            pad_right=data['padding']['right'],
            pad_bottom=data['padding']['bottom'],
            padded_size=tuple(data['padded_size']),
            scale=data['scale'],
            target_size=tuple(data['target_size'])
        )


class CoordinateTransform:
    """
    坐标变换器
    负责处理 原图 → padding → resize 的坐标变换
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 256)):
        """
        Args:
            target_size: 目标尺寸 (width, height)
        """
        self.target_size = target_size
        self.target_aspect_ratio = target_size[0] / target_size[1]  # 2:1
    
    def calculate_padding(self, 
                         original_size: Tuple[int, int],
                         center: bool = True) -> Tuple[int, int, int, int]:
        """
        计算padding量，使图像符合目标长宽比
        
        Args:
            original_size: 原始尺寸 (W0, H0)
            center: 是否居中padding
            
        Returns:
            (pad_left, pad_top, pad_right, pad_bottom)
        """
        W0, H0 = original_size
        current_ratio = W0 / H0
        
        if current_ratio > self.target_aspect_ratio:
            # 图像太宽，需要上下padding
            W_pad = W0
            H_pad = int(W0 / self.target_aspect_ratio)
            
            total_pad_h = H_pad - H0
            if center:
                pad_top = total_pad_h // 2
                pad_bottom = total_pad_h - pad_top
            else:
                pad_top = 0
                pad_bottom = total_pad_h
            
            pad_left = pad_right = 0
            
        else:
            # 图像太高，需要左右padding
            H_pad = H0
            W_pad = int(H0 * self.target_aspect_ratio)
            
            total_pad_w = W_pad - W0
            if center:
                pad_left = total_pad_w // 2
                pad_right = total_pad_w - pad_left
            else:
                pad_left = 0
                pad_right = total_pad_w
            
            pad_top = pad_bottom = 0
        
        return pad_left, pad_top, pad_right, pad_bottom
    
    def get_transform_params(self, 
                            original_size: Tuple[int, int],
                            center_padding: bool = True) -> TransformParams:
        """
        获取完整的变换参数
        
        Args:
            original_size: 原始图像尺寸 (W0, H0)
            center_padding: 是否居中padding
            
        Returns:
            TransformParams对象
        """
        W0, H0 = original_size
        
        # 计算padding
        pad_left, pad_top, pad_right, pad_bottom = self.calculate_padding(
            original_size, center_padding
        )
        
        # Padding后的尺寸
        W_pad = W0 + pad_left + pad_right
        H_pad = H0 + pad_top + pad_bottom
        
        # 计算缩放因子（等比例缩放）
        scale_x = self.target_size[0] / W_pad
        scale_y = self.target_size[1] / H_pad
        
        # 确保是等比例缩放
        assert abs(scale_x - scale_y) < 1e-6, \
            f"Scale mismatch: scale_x={scale_x}, scale_y={scale_y}"
        
        scale = scale_x
        
        return TransformParams(
            original_size=original_size,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
            padded_size=(W_pad, H_pad),
            scale=scale,
            target_size=self.target_size
        )
    
    def transform_forward(self, 
                         coords: Tuple[float, float],
                         params: TransformParams) -> Tuple[float, float]:
        """
        正向变换：原图坐标 → 训练坐标
        
        Args:
            coords: 原图坐标 (x0, y0)
            params: 变换参数
            
        Returns:
            训练用的坐标 (x2, y2)
        """
        x0, y0 = coords
        
        # Step 1: Padding变换
        x1 = x0 + params.pad_left
        y1 = y0 + params.pad_top
        
        # Step 2: Resize变换
        x2 = x1 * params.scale
        y2 = y1 * params.scale
        
        return (x2, y2)
    
    def transform_backward(self,
                          coords: Tuple[float, float],
                          params: TransformParams) -> Tuple[float, float]:
        """
        反向变换：训练坐标 → 原图坐标
        
        Args:
            coords: 训练坐标 (x2, y2)
            params: 变换参数
            
        Returns:
            原图坐标 (x0, y0)
        """
        x2, y2 = coords
        
        # Step 1: 反向Resize变换
        x1 = x2 / params.scale
        y1 = y2 / params.scale
        
        # Step 2: 反向Padding变换
        x0 = x1 - params.pad_left
        y0 = y1 - params.pad_top
        
        return (x0, y0)
    
    def transform_bbox_forward(self,
                              bbox: Tuple[float, float, float, float],
                              params: TransformParams) -> Tuple[float, float, float, float]:
        """
        正向变换边界框
        
        Args:
            bbox: 原图边界框 (x1, y1, x2, y2)
            params: 变换参数
            
        Returns:
            训练用边界框
        """
        x1, y1, x2, y2 = bbox
        x1_new, y1_new = self.transform_forward((x1, y1), params)
        x2_new, y2_new = self.transform_forward((x2, y2), params)
        return (x1_new, y1_new, x2_new, y2_new)
    
    def transform_bbox_backward(self,
                               bbox: Tuple[float, float, float, float],
                               params: TransformParams) -> Tuple[float, float, float, float]:
        """
        反向变换边界框
        
        Args:
            bbox: 训练边界框 (x1, y1, x2, y2)
            params: 变换参数
            
        Returns:
            原图边界框
        """
        x1, y1, x2, y2 = bbox
        x1_new, y1_new = self.transform_backward((x1, y1), params)
        x2_new, y2_new = self.transform_backward((x2, y2), params)
        return (x1_new, y1_new, x2_new, y2_new)
    
    def validate_coordinates(self,
                            coords: Tuple[float, float],
                            image_size: Tuple[int, int],
                            margin: int = 0) -> bool:
        """
        验证坐标是否在图像范围内
        
        Args:
            coords: 坐标 (x, y)
            image_size: 图像尺寸 (width, height)
            margin: 安全边距
            
        Returns:
            是否有效
        """
        x, y = coords
        width, height = image_size
        
        return (margin <= x <= width - margin and 
                margin <= y <= height - margin)


def demonstrate_transform():
    """演示坐标变换的具体例子"""
    print("=" * 60)
    print("坐标变换演示")
    print("=" * 60)
    
    # 创建变换器
    transform = CoordinateTransform(target_size=(512, 256))
    
    # 示例1：原图 419×640（太高）
    print("\n示例1：原图 419×640")
    original_size = (419, 640)
    params = transform.get_transform_params(original_size)
    
    print(f"原图尺寸: {original_size}")
    print(f"Padding: left={params.pad_left}, top={params.pad_top}, "
          f"right={params.pad_right}, bottom={params.pad_bottom}")
    print(f"Padding后尺寸: {params.padded_size}")
    print(f"缩放因子: {params.scale:.4f}")
    print(f"目标尺寸: {params.target_size}")
    
    # 测试坐标变换
    test_coords = [(100, 200), (200, 400), (300, 500)]
    print("\n坐标变换测试:")
    for x0, y0 in test_coords:
        x2, y2 = transform.transform_forward((x0, y0), params)
        x0_back, y0_back = transform.transform_backward((x2, y2), params)
        print(f"原图({x0:3.0f}, {y0:3.0f}) → "
              f"训练({x2:3.1f}, {y2:3.1f}) → "
              f"还原({x0_back:3.0f}, {y0_back:3.0f})")
    
    # 示例2：原图 320×160（已经是2:1）
    print("\n示例2：原图 320×160")
    original_size = (320, 160)
    params = transform.get_transform_params(original_size)
    
    print(f"原图尺寸: {original_size}")
    print(f"Padding: left={params.pad_left}, top={params.pad_top}, "
          f"right={params.pad_right}, bottom={params.pad_bottom}")
    print(f"Padding后尺寸: {params.padded_size}")
    print(f"缩放因子: {params.scale:.4f}")
    
    # 测试坐标变换
    test_coords = [(50, 50), (160, 80), (270, 140)]
    print("\n坐标变换测试:")
    for x0, y0 in test_coords:
        x2, y2 = transform.transform_forward((x0, y0), params)
        x0_back, y0_back = transform.transform_backward((x2, y2), params)
        print(f"原图({x0:3.0f}, {y0:3.0f}) → "
              f"训练({x2:3.1f}, {y2:3.1f}) → "
              f"还原({x0_back:3.0f}, {y0_back:3.0f})")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demonstrate_transform()