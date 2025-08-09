# -*- coding: utf-8 -*-
"""
尺寸不统一混淆配置
"""
import random
from typing import Tuple, Dict


class CaptchaSizeConfig:
    """验证码尺寸混淆配置 - 简化版"""
    
    def __init__(self):
        # 尺寸范围
        self.min_size = (280, 140)  # 最小宽x高
        self.max_size = (480, 240)  # 最大宽x高
        
        # 补齐目标（神经网络输入）
        self.target_size = (512, 256)   # 目标尺寸（宽x高）
        
        # 常见尺寸（30%概率使用）
        self.common_sizes = [
            (320, 160),
            (360, 180), 
            (400, 200),
        ]
    
    def generate_random_size(self) -> Tuple[int, int]:
        """生成随机验证码尺寸"""
        # 30%概率使用常见尺寸
        if random.random() < 0.3:
            return random.choice(self.common_sizes)
        
        # 70%概率生成随机尺寸
        width = random.randint(self.min_size[0], self.max_size[0])
        height = random.randint(self.min_size[1], self.max_size[1])
        
        # 确保长宽比在合理范围(1.8~2.4)
        aspect_ratio = width / height
        if aspect_ratio < 1.8:
            width = int(height * 1.8)
        elif aspect_ratio > 2.4:
            height = int(width / 2.4)
        
        # 确保在范围内且为偶数
        width = max(self.min_size[0], min(self.max_size[0], width))
        height = max(self.min_size[1], min(self.max_size[1], height))
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1
        
        return (width, height)
    
    def get_letterbox_params(self, src_width: int, src_height: int, training: bool = False) -> Dict:
        """计算letterbox缩放和padding参数"""
        # 计算缩放比例（保持长宽比）
        scale = min(self.target_size[0] / src_width, self.target_size[1] / src_height)
        
        # 缩放后的尺寸
        scaled_w = int(src_width * scale)
        scaled_h = int(src_height * scale)
        
        # 计算padding
        pad_w = self.target_size[0] - scaled_w
        pad_h = self.target_size[1] - scaled_h
        
        # 训练时随机位置，推理时居中
        if training:
            pad_left = random.randint(0, pad_w)
            pad_top = random.randint(0, pad_h)
        else:
            pad_left = pad_w // 2
            pad_top = pad_h // 2
        
        return {
            'scale': scale,
            'scaled_size': (scaled_w, scaled_h),
            'pad_left': pad_left,
            'pad_top': pad_top,
            'pad_right': pad_w - pad_left,
            'pad_bottom': pad_h - pad_top
        }
    
    def transform_coords(self, x: float, y: float, params: Dict, inverse: bool = False) -> Tuple[float, float]:
        """坐标变换（原始<->补齐后）"""
        if inverse:
            # 补齐后 -> 原始
            x = (x - params['pad_left']) / params['scale']
            y = (y - params['pad_top']) / params['scale']
        else:
            # 原始 -> 补齐后
            x = x * params['scale'] + params['pad_left']
            y = y * params['scale'] + params['pad_top']
        
        return (x, y)


# 创建全局配置实例
size_config = CaptchaSizeConfig()