# -*- coding: utf-8 -*-
"""
尺寸混淆配置
用于生成不同尺寸的验证码并进行letterbox padding
"""
import random
import numpy as np
from typing import Tuple, Dict, Optional
import cv2
from .config_loader import ConfigLoader


class SizeConfusionConfig:
    """尺寸混淆配置 - 生成随机尺寸的验证码并padding到目标尺寸"""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """初始化配置"""
        if config_loader is None:
            config_loader = ConfigLoader()
        
        self.loader = config_loader
        self._load_config()
    
    def _load_config(self):
        """从YAML文件加载配置"""
        base_path = 'captcha_config.size_confusion'
        
        # ========== 生成尺寸范围 ==========
        # 验证码生成时的随机尺寸范围
        min_size_list = self.loader.get(f'{base_path}.generation.min_size')
        max_size_list = self.loader.get(f'{base_path}.generation.max_size')
        if min_size_list is None:
            raise ValueError(f"Must configure {base_path}.generation.min_size in captcha_config.yaml")
        if max_size_list is None:
            raise ValueError(f"Must configure {base_path}.generation.max_size in captcha_config.yaml")
        self.min_size = tuple(min_size_list)  # 最小宽x高
        self.max_size = tuple(max_size_list)  # 最大宽x高
        
        # ========== 目标尺寸（神经网络输入 - 从model_config读取） ==========
        # 最终padding到的目标尺寸
        target_size_list = self.loader.get('model_config.input.target_size')
        if target_size_list is None:
            raise ValueError("Must configure input.target_size in model_config.yaml")
        self.target_size = tuple(target_size_list)  # 目标宽x高
        
        # ========== 常见尺寸（提高这些尺寸的生成概率） ==========
        common_sizes_list = self.loader.get(f'{base_path}.generation.common_sizes')
        common_prob = self.loader.get(f'{base_path}.generation.common_size_probability')
        if common_sizes_list is None:
            raise ValueError(f"Must configure {base_path}.generation.common_sizes in captcha_config.yaml")
        if common_prob is None:
            raise ValueError(f"Must configure {base_path}.generation.common_size_probability in captcha_config.yaml")
        self.common_sizes = [tuple(size) for size in common_sizes_list]
        self.common_size_probability = common_prob
        
        # ========== Padding配置 (从model_config读取) ==========
        # 从model_config.yaml的preprocessing.padding部分读取
        padding_color_list = self.loader.get('model_config.preprocessing.padding.color')
        padding_mode = self.loader.get('model_config.preprocessing.padding.mode')
        if padding_color_list is None:
            raise ValueError("Must configure preprocessing.padding.color in model_config.yaml")
        if padding_mode is None:
            raise ValueError("Must configure preprocessing.padding.mode in model_config.yaml")
        self.padding_color = tuple(padding_color_list)  # 黑色padding
        self.padding_mode = padding_mode  # letterbox模式
    
    def generate_random_size(self) -> Tuple[int, int]:
        """
        生成随机验证码尺寸
        
        Returns:
            (width, height): 生成的尺寸
        """
        # 30%概率使用常见尺寸
        if random.random() < self.common_size_probability:
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
        
        # 确保在范围内且为偶数（便于处理）
        width = max(self.min_size[0], min(self.max_size[0], width))
        height = max(self.min_size[1], min(self.max_size[1], height))
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1
        
        return (width, height)
    
    def get_letterbox_params(self, src_width: int, src_height: int, 
                            training: bool = False) -> Dict:
        """
        计算letterbox缩放和padding参数
        
        Letterbox逻辑：
        1. 计算缩放比例，保持长宽比
        2. 缩放图像到不超过目标尺寸
        3. 计算需要的padding量
        4. 训练时随机放置，推理时居中
        
        Args:
            src_width: 原始宽度
            src_height: 原始高度
            training: 是否为训练模式（影响padding位置）
            
        Returns:
            包含缩放和padding参数的字典
        """
        target_w, target_h = self.target_size
        
        # 计算缩放比例（保持长宽比，确保不超过目标尺寸）
        scale = min(target_w / src_width, target_h / src_height)
        
        # 缩放后的尺寸
        scaled_w = int(src_width * scale)
        scaled_h = int(src_height * scale)
        
        # 计算总padding量
        pad_w = target_w - scaled_w
        pad_h = target_h - scaled_h
        
        # 训练时随机位置，推理时居中
        if training:
            # 随机决定padding分配
            pad_left = random.randint(0, pad_w)
            pad_top = random.randint(0, pad_h)
        else:
            # 居中padding
            pad_left = pad_w // 2
            pad_top = pad_h // 2
        
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        return {
            'scale': scale,
            'scaled_size': (scaled_w, scaled_h),
            'pad_left': pad_left,
            'pad_top': pad_top,
            'pad_right': pad_right,
            'pad_bottom': pad_bottom,
            'target_size': self.target_size
        }
    
    def apply_letterbox(self, image: np.ndarray, training: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        应用letterbox变换到图像
        
        Args:
            image: 输入图像 (H, W, C)
            training: 是否为训练模式
            
        Returns:
            padded_image: padding后的图像
            params: letterbox参数（用于坐标转换）
        """
        h, w = image.shape[:2]
        
        # 获取letterbox参数
        params = self.get_letterbox_params(w, h, training)
        
        # 缩放图像
        scaled_img = cv2.resize(image, params['scaled_size'], interpolation=cv2.INTER_LINEAR)
        
        # 创建目标尺寸的画布
        target_w, target_h = self.target_size
        if len(image.shape) == 3:
            padded_img = np.full((target_h, target_w, image.shape[2]), 
                                self.padding_color, dtype=image.dtype)
        else:
            padded_img = np.full((target_h, target_w), 
                                self.padding_color[0], dtype=image.dtype)
        
        # 将缩放后的图像放置到画布上
        y1 = params['pad_top']
        y2 = y1 + params['scaled_size'][1]
        x1 = params['pad_left']
        x2 = x1 + params['scaled_size'][0]
        
        padded_img[y1:y2, x1:x2] = scaled_img
        
        return padded_img, params
    
    def transform_coords(self, x: float, y: float, params: Dict, 
                        inverse: bool = False) -> Tuple[float, float]:
        """
        坐标变换（原始<->padding后）
        
        Args:
            x, y: 坐标
            params: letterbox参数
            inverse: 是否为逆变换（padding后->原始）
            
        Returns:
            变换后的坐标
        """
        if inverse:
            # padding后 -> 原始
            x = (x - params['pad_left']) / params['scale']
            y = (y - params['pad_top']) / params['scale']
        else:
            # 原始 -> padding后
            x = x * params['scale'] + params['pad_left']
            y = y * params['scale'] + params['pad_top']
        
        return (x, y)
    
    def transform_bbox(self, bbox: Tuple[float, float, float, float], 
                      params: Dict, inverse: bool = False) -> Tuple[float, float, float, float]:
        """
        边界框变换
        
        Args:
            bbox: (x1, y1, x2, y2) 格式的边界框
            params: letterbox参数
            inverse: 是否为逆变换
            
        Returns:
            变换后的边界框
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = self.transform_coords(x1, y1, params, inverse)
        x2, y2 = self.transform_coords(x2, y2, params, inverse)
        return (x1, y1, x2, y2)
    
    def validate_size(self, width: int, height: int) -> bool:
        """
        验证尺寸是否在合理范围内
        
        Args:
            width: 宽度
            height: 高度
            
        Returns:
            是否有效
        """
        # 检查是否在范围内
        if width < self.min_size[0] or width > self.max_size[0]:
            return False
        if height < self.min_size[1] or height > self.max_size[1]:
            return False
        
        # 检查长宽比
        aspect_ratio = width / height
        if aspect_ratio < 1.8 or aspect_ratio > 2.4:
            return False
        
        return True
    
    def print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("Size Confusion Configuration")
        print("=" * 60)
        
        print("\n[Generation Size Range]")
        print(f"  Min size (WxH): {self.min_size[0]}x{self.min_size[1]}")
        print(f"  Max size (WxH): {self.max_size[0]}x{self.max_size[1]}")
        
        print("\n[Target Size (Model Input)]")
        print(f"  Target size (WxH): {self.target_size[0]}x{self.target_size[1]}")
        
        print("\n[Common Sizes]")
        for size in self.common_sizes:
            print(f"  - {size[0]}x{size[1]}")
        print(f"  Probability: {self.common_size_probability*100}%")
        
        print("\n[Padding Configuration]")
        print(f"  Padding mode: {self.padding_mode}")
        print(f"  Padding color: RGB{self.padding_color}")
        
        print("=" * 60)


# 创建全局实例
size_config = None

def get_size_confusion_config() -> SizeConfusionConfig:
    """获取尺寸混淆配置单例"""
    global size_config
    if size_config is None:
        size_config = SizeConfusionConfig()
    return size_config


if __name__ == "__main__":
    # Test configuration
    config = get_size_confusion_config()
    config.print_config()
    
    print("\n[Test Random Size Generation]")
    for i in range(5):
        size = config.generate_random_size()
        print(f"  Generated size {i+1}: {size[0]}x{size[1]}")
    
    print("\n[Test Letterbox Parameters]")
    test_sizes = [(320, 160), (400, 200), (280, 140), (480, 240)]
    for size in test_sizes:
        params = config.get_letterbox_params(size[0], size[1], training=False)
        print(f"\n  Original: {size[0]}x{size[1]}")
        print(f"  Scale: {params['scale']:.3f}")
        print(f"  Scaled: {params['scaled_size'][0]}x{params['scaled_size'][1]}")
        print(f"  Padding: L={params['pad_left']}, T={params['pad_top']}, "
              f"R={params['pad_right']}, B={params['pad_bottom']}")
    
    print("\n[Test Coordinate Transformation]")
    # 测试坐标转换
    params = config.get_letterbox_params(320, 160, training=False)
    test_coords = [(100, 50), (200, 100), (300, 150)]
    print(f"  Letterbox params: scale={params['scale']:.3f}, "
          f"padding=({params['pad_left']}, {params['pad_top']})")
    for x, y in test_coords:
        x_new, y_new = config.transform_coords(x, y, params, inverse=False)
        x_back, y_back = config.transform_coords(x_new, y_new, params, inverse=True)
        print(f"  ({x}, {y}) -> ({x_new:.1f}, {y_new:.1f}) -> ({x_back:.1f}, {y_back:.1f})")