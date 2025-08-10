# -*- coding: utf-8 -*-
"""
混淆系统基础类和数据结构
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
import cv2


@dataclass
class GapImage:
    """独立的缺口图像"""
    image: np.ndarray              # BGRA格式，缺口内容+透明度
    position: Tuple[int, int]      # 在背景中的中心位置
    original_mask: np.ndarray      # 原始形状掩码
    metadata: Optional[Dict[str, Any]] = None  # 元数据（包含形状信息等）
    background_size: Optional[Tuple[int, int]] = None  # 背景图像尺寸 (width, height)
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """返回图像形状"""
        return self.image.shape
    
    @property
    def height(self) -> int:
        """返回图像高度"""
        return self.image.shape[0]
    
    @property
    def width(self) -> int:
        """返回图像宽度"""
        return self.image.shape[1]
    
    def copy(self) -> 'GapImage':
        """创建深拷贝"""
        return GapImage(
            image=self.image.copy(),
            position=self.position,
            original_mask=self.original_mask.copy(),
            metadata=self.metadata.copy() if self.metadata else None,
            background_size=self.background_size
        )


@dataclass
class CaptchaResult:
    """验证码生成结果"""
    background: np.ndarray          # 带缺口的背景图 (160, 320, 3)
    slider: np.ndarray              # 滑块图像 (h, w, 4) 带alpha通道
    gap_position: Tuple[int, int]   # 缺口中心位置
    slider_position: Tuple[int, int]# 滑块中心位置
    gap_mask: np.ndarray           # 缺口形状掩码
    
    # 混淆相关信息
    confusion_type: str             # 应用的混淆类型
    confusion_params: Dict[str, Any] # 混淆参数
    metadata: Dict[str, Any] = field(default_factory=dict)
    additional_gaps: Optional[List[Dict[str, Any]]] = None
    
    def save_background(self, filepath: str):
        """保存背景图（带缺口）"""
        cv2.imwrite(filepath, self.background)
    
    def save_slider(self, filepath: str):
        """保存滑块图"""
        cv2.imwrite(filepath, self.slider)
    
    def save_both(self, bg_path: str, slider_path: str):
        """分别保存背景和滑块"""
        self.save_background(bg_path)
        self.save_slider(slider_path)
    
    def to_dict(self) -> Dict:
        """转换为字典格式（用于保存元数据）"""
        return {
            'gap_position': self.gap_position,
            'slider_position': self.slider_position,
            'confusion_type': self.confusion_type,
            'confusion_params': self.confusion_params,
            'metadata': self.metadata,
            'additional_gaps': self.additional_gaps
        }


class ConfusionStrategy(ABC):
    """混淆策略基类 - 在独立的gap图像上应用混淆"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.additional_gaps = []  # 存储额外的gap信息
        self.validate_config()
    
    @abstractmethod
    def validate_config(self):
        """验证配置参数"""
        pass
    
    @abstractmethod
    def apply_to_gap(self, gap_image: GapImage) -> GapImage:
        """
        在独立的gap图像上应用混淆策略
        
        Args:
            gap_image: 独立的gap图像（BGRA格式）
            
        Returns:
            GapImage: 应用混淆后的gap图像
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """策略描述"""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取策略的元数据"""
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config
        }
    
    def get_additional_gaps(self) -> List[Dict[str, Any]]:
        """获取额外的gap信息（用于混淆缺口策略）"""
        return self.additional_gaps