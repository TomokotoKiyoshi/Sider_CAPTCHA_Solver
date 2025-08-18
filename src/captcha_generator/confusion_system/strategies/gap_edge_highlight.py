# -*- coding: utf-8 -*-
"""
边缘高光混淆策略 - 为缺口边缘添加高光效果
"""
import numpy as np
import cv2
from typing import Dict, Any
from ..base import ConfusionStrategy, GapImage


class GapEdgeHighlightConfusion(ConfusionStrategy):
    """缺口边缘高光混淆 - 为缺口边缘添加高光效果，使边缘发光，增强3D凸起效果"""
    
    @property
    def name(self) -> str:
        return "gap_edge_highlight"
    
    @property
    def description(self) -> str:
        return "为缺口边缘添加白色高光效果，使边缘发光，创造3D凸起视觉效果"
    
    def validate_config(self):
        """验证并设置默认配置"""
        # 边缘高光强度（0-100）
        self.edge_lightness = self.config.get('edge_lightness', 50)
        # 边缘宽度（像素）
        self.edge_width = self.config.get('edge_width', 6)
        # 衰减系数
        self.decay_factor = self.config.get('decay_factor', 2.0)
        
        # 验证参数范围
        assert 0 < self.edge_lightness <= 100, "edge_lightness must be between 1 and 100"
        assert 1 <= self.edge_width <= 10, "edge_width must be between 1 and 10"
        assert 0.5 <= self.decay_factor <= 5.0, "decay_factor must be between 0.5 and 5.0"
    
    def apply_to_gap(self, gap_image: GapImage) -> GapImage:
        """
        GapEdgeHighlight策略不修改滑块本身，只作为标记让背景缺口应用边缘高光效果
        
        边缘高光效果会在背景生成时通过apply_gap_lighting函数的edge_lightness参数实现
        
        Args:
            gap_image: 独立的gap图像（BGRA格式）
            
        Returns:
            GapImage: 返回原始gap图像（不做修改）
        """
        # GapEdgeHighlight策略不修改滑块
        # 只是作为一个标记，让背景生成时知道要应用边缘高光效果
        # 实际的边缘高光效果会在generator.py的_create_gap_background中
        # 通过调用apply_gap_lighting并传递edge_lightness参数来实现
        return gap_image
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取策略元数据"""
        return {
            'edge_lightness': self.edge_lightness,
            'edge_width': self.edge_width,
            'decay_factor': self.decay_factor
        }