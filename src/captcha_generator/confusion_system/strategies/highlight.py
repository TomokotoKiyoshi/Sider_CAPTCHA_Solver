# -*- coding: utf-8 -*-
"""
高光混淆策略 - 将缺口阴影改为高光效果
"""
import numpy as np
import cv2
from typing import Dict, Any
from ..base import ConfusionStrategy, GapImage


class HighlightConfusion(ConfusionStrategy):
    """高光混淆 - 将缺口的阴影效果改为高光效果，造成视觉混淆"""
    
    @property
    def name(self) -> str:
        return "highlight"
    
    @property
    def description(self) -> str:
        return "将缺口的阴影效果改为高光效果，使其看起来凸起而非凹陷"
    
    def validate_config(self):
        """验证并设置默认配置"""
        self.base_lightness = self.config.get('base_lightness', 30)
        self.edge_lightness = self.config.get('edge_lightness', 45)
        self.directional_lightness = self.config.get('directional_lightness', 20)
        
        # 验证参数范围
        assert 0 <= self.base_lightness <= 100, "base_lightness must be between 0 and 100"
        assert 0 <= self.edge_lightness <= 100, "edge_lightness must be between 0 and 100"
        assert 0 <= self.directional_lightness <= 100, "directional_lightness must be between 0 and 100"
    
    def apply_to_gap(self, gap_image: GapImage) -> GapImage:
        """
        Highlight策略不修改滑块本身，只作为标记让背景缺口应用高光效果
        
        Args:
            gap_image: 独立的gap图像（BGRA格式）
            
        Returns:
            GapImage: 返回原始gap图像（不做修改）
        """
        # Highlight策略不应该修改滑块本身
        # 只是作为一个标记，让背景生成时知道要应用高光效果而不是阴影
        return gap_image