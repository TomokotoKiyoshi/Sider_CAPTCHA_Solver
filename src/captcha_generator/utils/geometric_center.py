# -*- coding: utf-8 -*-
"""
几何中心计算工具
用于计算各种形状的真实几何中心（质心）
"""
import numpy as np
import cv2
from typing import Tuple, Optional, Union


class GeometricCenterCalculator:
    """计算各种形状的几何中心"""
    
    @staticmethod
    def calculate_shape_centroid(mask: np.ndarray, 
                                shape_type: Union[str, Tuple[str, ...]]) -> Tuple[int, int]:
        """
        计算形状的几何中心（质心）
        
        Args:
            mask: 形状的mask（alpha通道或二值图）
            shape_type: 形状类型 ('circle', 'triangle', 'square', 'hexagon', 'pentagon', 'normal')
                       或普通拼图的边缘组合 (top, right, bottom, left)
            
        Returns:
            (cx, cy): 相对于mask左上角的几何中心坐标
        """
        # 获取alpha通道或转换为二值图
        if len(mask.shape) == 3:
            if mask.shape[2] == 4:
                binary = mask[:, :, 3]
            else:
                # 如果是RGB，转换为灰度
                binary = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            binary = mask
        
        # 确保是二值图
        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)
        
        # 对所有形状都使用cv2.moments计算真实的几何中心（质心）
        # 这确保了一致性和准确性，无论是特殊形状还是普通拼图形状
        M = cv2.moments(binary)
        if M["m00"] != 0:
            # 计算质心坐标
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # 如果moments计算失败（例如全黑图像），降级到边界框中心
            cx = mask.shape[1] // 2
            cy = mask.shape[0] // 2
        
        return (cx, cy)
    
    @staticmethod
    def calculate_absolute_position(mask: np.ndarray,
                                   bbox_center: Tuple[int, int],
                                   shape_type: Union[str, Tuple[str, ...]]) -> Tuple[int, int]:
        """
        计算形状在图像中的绝对几何中心位置
        
        Args:
            mask: 形状的mask
            bbox_center: 边界框中心在图像中的位置
            shape_type: 形状类型
            
        Returns:
            (abs_cx, abs_cy): 形状在整个图像中的几何中心坐标
        """
        # 计算相对于mask的几何中心
        rel_cx, rel_cy = GeometricCenterCalculator.calculate_shape_centroid(mask, shape_type)
        
        # 计算mask的边界框中心
        mask_h, mask_w = mask.shape[:2]
        bbox_rel_cx = mask_w // 2
        bbox_rel_cy = mask_h // 2
        
        # 计算偏移量
        offset_x = rel_cx - bbox_rel_cx
        offset_y = rel_cy - bbox_rel_cy
        
        # 计算绝对位置
        abs_cx = bbox_center[0] + offset_x
        abs_cy = bbox_center[1] + offset_y
        
        return (abs_cx, abs_cy)
    
    @staticmethod
    def validate_centroid(mask: np.ndarray, 
                         centroid: Tuple[int, int],
                         shape_type: str) -> bool:
        """
        验证计算的质心是否合理
        
        Args:
            mask: 形状的mask
            centroid: 计算得到的质心
            shape_type: 形状类型
            
        Returns:
            是否合理
        """
        cx, cy = centroid
        h, w = mask.shape[:2]
        
        # 检查质心是否在mask范围内
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return False
        
        # 对于特殊形状，检查质心是否在形状内部
        if shape_type in ['circle', 'triangle', 'square', 'hexagon', 'pentagon']:
            if len(mask.shape) == 3 and mask.shape[2] == 4:
                # 检查质心位置的alpha值
                if mask[cy, cx, 3] == 0:
                    # 质心在透明区域，可能不正确
                    return False
        
        return True
    
    @staticmethod
    def get_shape_info(mask: np.ndarray) -> dict:
        """
        获取形状的详细信息
        
        Args:
            mask: 形状的mask
            
        Returns:
            包含各种几何信息的字典
        """
        # 获取二值图
        if len(mask.shape) == 3:
            if mask.shape[2] == 4:
                binary = mask[:, :, 3]
            else:
                binary = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            binary = mask.copy()
        
        # 计算moments
        M = cv2.moments(binary)
        
        info = {
            'bbox_size': (mask.shape[1], mask.shape[0]),  # (width, height)
            'bbox_center': (mask.shape[1] // 2, mask.shape[0] // 2),
        }
        
        if M["m00"] != 0:
            info['geometric_center'] = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            info['area'] = M["m00"]
        else:
            info['geometric_center'] = info['bbox_center']
            info['area'] = 0
        
        # 计算偏心度等高级特征（可选）
        if M["m00"] != 0 and M["mu20"] + M["mu02"] != 0:
            # 计算椭圆的长短轴
            mu20 = M["mu20"] / M["m00"]
            mu02 = M["mu02"] / M["m00"]
            mu11 = M["mu11"] / M["m00"]
            
            # 计算特征值
            lambda1 = (mu20 + mu02 + np.sqrt((mu20 - mu02)**2 + 4*mu11**2)) / 2
            lambda2 = (mu20 + mu02 - np.sqrt((mu20 - mu02)**2 + 4*mu11**2)) / 2
            
            if lambda1 > 0:
                info['eccentricity'] = np.sqrt(1 - lambda2/lambda1)
            else:
                info['eccentricity'] = 0
        
        return info


# 创建全局实例
geometric_calculator = GeometricCenterCalculator()


# 便捷函数
def calculate_geometric_center(mask: np.ndarray, 
                              shape_type: Union[str, Tuple[str, ...]]) -> Tuple[int, int]:
    """计算形状的几何中心"""
    return geometric_calculator.calculate_shape_centroid(mask, shape_type)


def calculate_absolute_geometric_center(mask: np.ndarray,
                                       bbox_center: Tuple[int, int],
                                       shape_type: Union[str, Tuple[str, ...]]) -> Tuple[int, int]:
    """计算形状在图像中的绝对几何中心"""
    return geometric_calculator.calculate_absolute_position(mask, bbox_center, shape_type)


if __name__ == "__main__":
    # 测试代码
    print("Testing GeometricCenterCalculator...")
    
    # 创建一个简单的三角形mask
    mask = np.zeros((100, 100, 4), dtype=np.uint8)
    
    # 绘制三角形
    pts = np.array([[50, 20], [20, 80], [80, 80]], np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255, 255))
    
    # 计算几何中心
    center = calculate_geometric_center(mask, 'triangle')
    print(f"Triangle geometric center: {center}")
    
    # 边界框中心
    bbox_center = (50, 50)
    print(f"Bounding box center: {bbox_center}")
    
    # 偏移量
    offset = (center[0] - bbox_center[0], center[1] - bbox_center[1])
    print(f"Offset from bbox center: {offset}")
    
    # 获取详细信息
    info = geometric_calculator.get_shape_info(mask)
    print(f"Shape info: {info}")