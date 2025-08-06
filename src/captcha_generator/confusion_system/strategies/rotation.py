# -*- coding: utf-8 -*-
"""
缺口旋转混淆策略 - 轻微旋转缺口造成视觉干扰
"""
import numpy as np
import cv2
from typing import Dict, Any
from ..base import ConfusionStrategy, GapImage


class RotationConfusion(ConfusionStrategy):
    """旋转混淆 - 轻微旋转缺口（0.5-1.8度），只旋转缺口不旋转滑块"""
    
    @property
    def name(self) -> str:
        return "rotation"
    
    @property
    def description(self) -> str:
        return "轻微旋转缺口造成视觉干扰，使对齐更困难"
    
    def validate_config(self):
        """验证并设置默认配置"""
        # 旋转角度（度）- 必须由外界传入
        self.rotation_angle = self.config.get('rotation_angle')
        
        # 验证参数
        assert self.rotation_angle is not None, "rotation_angle must be provided in config"
        assert isinstance(self.rotation_angle, (int, float)), "rotation_angle must be a number"
        assert -5 <= self.rotation_angle <= 5, f"rotation_angle must be between -5 and 5 degrees, got: {self.rotation_angle}"
    
    def apply_to_gap(self, gap_image: GapImage) -> GapImage:
        """
        在独立的gap图像上应用旋转效果
        
        Args:
            gap_image: 独立的gap图像（BGRA格式）
            
        Returns:
            GapImage: 应用旋转后的gap图像
        """
        # 获取图像尺寸
        h, w = gap_image.image.shape[:2]
        center = (w // 2, h // 2)
        
        # 计算旋转后的边界框大小（确保不裁剪）
        angle_rad = np.radians(abs(self.rotation_angle))
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # 计算旋转后需要的新尺寸
        new_w = int(h * sin_angle + w * cos_angle)
        new_h = int(h * cos_angle + w * sin_angle)
        
        # 创建旋转矩阵（不缩放内容，scale=1.0）
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
        
        # 调整旋转中心到新画布的中心
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        # 旋转BGRA图像 - 扩大画布以容纳完整旋转后的图像
        # 使用INTER_LINEAR插值，边界填充为透明
        rotated_image = cv2.warpAffine(
            gap_image.image,
            rotation_matrix,
            (new_w, new_h),  # 扩大后的尺寸
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)  # 透明背景
        )
        
        # 旋转原始掩码（用于保持形状信息）
        rotated_mask = cv2.warpAffine(
            gap_image.original_mask,
            rotation_matrix,
            (new_w, new_h),  # 扩大后的尺寸
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0) if gap_image.original_mask.shape[2] == 4 else 0
        )
        
        # 返回新的GapImage
        return GapImage(
            image=rotated_image,
            position=gap_image.position,  # 位置保持不变
            original_mask=rotated_mask
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取策略的元数据"""
        metadata = super().get_metadata()
        metadata.update({
            "rotation_angle": self.rotation_angle
        })
        return metadata