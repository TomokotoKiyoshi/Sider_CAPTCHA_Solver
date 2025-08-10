# -*- coding: utf-8 -*-
"""
数据集生成配置
从YAML文件加载配置并提供便捷访问接口
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from .config_loader import ConfigLoader


class DatasetConfig:
    """数据集生成配置"""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """初始化配置"""
        if config_loader is None:
            config_loader = ConfigLoader()
        
        self.loader = config_loader
        self._load_config()
    
    def _load_config(self):
        """从YAML文件加载配置"""
        # 路径配置
        self.INPUT_DIR = self.loader.get('captcha_config.dataset.paths.input_dir', 'data/raw')
        self.OUTPUT_DIR = self.loader.get('captcha_config.dataset.paths.output_dir', 'data')
        
        # 并行处理配置
        self.MAX_WORKERS = self.loader.get('captcha_config.dataset.processing.max_workers')
        self.MAX_IMAGES = self.loader.get('captcha_config.dataset.processing.max_images')
        self.RANDOM_SEED = self.loader.get('captcha_config.dataset.processing.random_seed')
        
        # 图像尺寸配置
        self.USE_RANDOM_SIZE = True
        self.SIZE_CONFIG_MODULE = 'src.config.size_confusion_config'
        
        # 数据集规模
        self.MIN_BACKGROUNDS = self.loader.get('captcha_config.dataset.scale.min_backgrounds', 2000)
        self.MAX_SAMPLES_PER_BG = self.loader.get('captcha_config.dataset.scale.max_samples_per_bg', 100)
        
        # 混淆策略数量控制
        self.CONFUSION_COUNTS = self.loader.get('captcha_config.dataset.confusion_counts', {
            'none': 30,
            'rotation': 10,
            'perlin_noise': 10,
            'highlight': 10,
            'confusing_gap': 10,
            'hollow_center': 10,
            'combined': 20
        })
        
        # 形状配置
        self.SPECIAL_SHAPES = self.loader.get('captcha_config.dataset.shapes.special', 
                                              ['circle', 'square', 'triangle', 'hexagon'])
        self.NORMAL_SHAPES_COUNT = self.loader.get('captcha_config.dataset.shapes.normal_count', 6)
        
        # Puzzle尺寸配置
        self.ALL_PUZZLE_SIZES = self.loader.get('captcha_config.dataset.puzzle_sizes.all_sizes',
                                                list(range(40, 62, 2)))
        self.SIZES_PER_IMAGE = self.loader.get('captcha_config.dataset.puzzle_sizes.sizes_per_image', 4)
        
        # 组件保存配置
        self.SAVE_FULL_IMAGE = self.loader.get('captcha_config.dataset.save.full_image', False)
        self.SAVE_COMPONENTS = self.loader.get('captcha_config.dataset.save.components', False)
        self.COMPONENT_FORMAT = self.loader.get('captcha_config.dataset.save.component_format', 'png')
        self.SAVE_MASKS = self.loader.get('captcha_config.dataset.save.masks', False)
        
        # Gap位置生成配置
        self.GAP_X_COUNT = self.loader.get('captcha_config.dataset.gap_generation.gap_x_count', 4)
        self.GAP_Y_COUNT = self.loader.get('captcha_config.dataset.gap_generation.gap_y_count', 3)
        self.SLIDER_X_COUNT = self.loader.get('captcha_config.dataset.gap_generation.slider_x_count', 4)
        
        # 旋转配置
        self.ROTATION_ENABLED = self.loader.get('captcha_config.dataset.rotation.enabled', True)
        self.MAX_ROTATION_ANGLE = self.loader.get('captcha_config.dataset.rotation.max_angle', 1.8)
    
    def validate(self) -> bool:
        """验证配置的合理性"""
        # 验证混淆策略数量总和
        total_confusion = sum(self.CONFUSION_COUNTS.values())
        if total_confusion != self.MAX_SAMPLES_PER_BG:
            print(f"Warning: Total confusion count ({total_confusion}) does not match samples per background ({self.MAX_SAMPLES_PER_BG})")
            print("Adjusting confusion counts proportionally...")
            # 自动调整混淆策略数量以匹配样本数
            self._adjust_confusion_counts()
            return True  # 继续执行，但已调整
        
        # 验证puzzle尺寸
        if not self.ALL_PUZZLE_SIZES:
            print("Error: No puzzle sizes defined")
            return False
        
        if self.SIZES_PER_IMAGE > len(self.ALL_PUZZLE_SIZES):
            print(f"Warning: Sizes per image ({self.SIZES_PER_IMAGE}) exceeds available sizes ({len(self.ALL_PUZZLE_SIZES)})")
            return False
        
        # 位置范围现在是动态计算的，不需要验证
        
        return True
    
    def _adjust_confusion_counts(self) -> None:
        """按比例调整混淆策略数量以匹配样本数"""
        original_total = sum(self.CONFUSION_COUNTS.values())
        if original_total == 0:
            # 如果没有定义混淆策略，使用默认分配
            self.CONFUSION_COUNTS = {
                'none': self.MAX_SAMPLES_PER_BG,
                'rotation': 0,
                'perlin_noise': 0,
                'highlight': 0,
                'confusing_gap': 0,
                'hollow_center': 0,
                'combined': 0
            }
            return
        
        # 按比例调整每种策略的数量
        ratio = self.MAX_SAMPLES_PER_BG / original_total
        adjusted_counts = {}
        total_assigned = 0
        
        for strategy, count in self.CONFUSION_COUNTS.items():
            adjusted = int(count * ratio)
            adjusted_counts[strategy] = adjusted
            total_assigned += adjusted
        
        # 处理舍入误差，将剩余的分配给'none'或第一个非零策略
        remaining = self.MAX_SAMPLES_PER_BG - total_assigned
        if remaining > 0:
            if 'none' in adjusted_counts:
                adjusted_counts['none'] += remaining
            else:
                # 找到第一个非零策略
                for strategy in adjusted_counts:
                    if adjusted_counts[strategy] > 0:
                        adjusted_counts[strategy] += remaining
                        break
                else:
                    # 如果都是0，分配给'none'
                    adjusted_counts['none'] = remaining
        
        self.CONFUSION_COUNTS = adjusted_counts
    
    def update_puzzle_sizes(self, sizes: List[int]) -> None:
        """更新puzzle尺寸列表"""
        self.ALL_PUZZLE_SIZES = sorted(sizes)
        print(f"Updated puzzle sizes: {self.ALL_PUZZLE_SIZES}")
    
    @staticmethod
    def parse_size_string(size_str: str) -> List[int]:
        """
        解析尺寸字符串，支持多种格式：
        - "30,40,50": 指定具体尺寸
        - "30-60": 范围（步长默认2）
        - "30-60:5": 范围和步长
        - "30": 单个尺寸
        
        Args:
            size_str: 尺寸配置字符串
            
        Returns:
            解析后的尺寸列表
        """
        size_str = size_str.strip()
        
        if '-' in size_str:
            # 范围格式
            if ':' in size_str:
                # 带步长的范围
                range_part, step = size_str.split(':')
                start, end = map(int, range_part.split('-'))
                step = int(step)
            else:
                # 默认步长为2
                start, end = map(int, size_str.split('-'))
                step = 2
            return list(range(start, end + 1, step))
        
        elif ',' in size_str:
            # 列表格式
            return sorted([int(s.strip()) for s in size_str.split(',')])
        
        else:
            # 单个值
            return [int(size_str)]
    
    def calculate_safe_gap_range(self, puzzle_size: int, img_width: int = 320, 
                                 img_height: int = 160, with_rotation: bool = None) -> Tuple[int, int]:
        """
        计算考虑旋转的安全gap范围
        
        Args:
            puzzle_size: 拼图大小
            img_width: 图像宽度
            img_height: 图像高度  
            with_rotation: 是否考虑旋转（None则使用配置）
            
        Returns:
            (gap_x_min, gap_x_max): 安全的gap x坐标范围
        """
        if with_rotation is None:
            with_rotation = self.ROTATION_ENABLED
            
        if with_rotation:
            # 考虑旋转后的尺寸增大
            angle_rad = np.radians(self.MAX_ROTATION_ANGLE)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            # 计算旋转后的边界框
            effective_size = int(puzzle_size * (cos_a + sin_a))
        else:
            effective_size = puzzle_size
            
        half_size = effective_size // 2
        
        # 计算滑块的最大位置
        slider_x_max = effective_size // 2 + 10
        
        # 缺口必须在滑块右边，并留有间隔
        gap_x_min = slider_x_max + effective_size + 10
        gap_x_max = img_width - half_size
        
        # 确保范围有效
        if gap_x_min > gap_x_max:
            # 对于大尺寸，可能没有有效范围
            return gap_x_min, gap_x_min  # 返回相同值表示无效
            
        return gap_x_min, gap_x_max
    
    def calculate_safe_slider_range(self, puzzle_size: int, img_height: int = 160,
                                   with_rotation: bool = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        计算考虑旋转的安全滑块范围
        
        Args:
            puzzle_size: 拼图大小
            img_height: 图像高度
            with_rotation: 是否考虑旋转
            
        Returns:
            ((x_min, x_max), (y_min, y_max)): 滑块的安全范围
        """
        if with_rotation is None:
            with_rotation = self.ROTATION_ENABLED
            
        if with_rotation:
            angle_rad = np.radians(self.MAX_ROTATION_ANGLE)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            effective_size = int(puzzle_size * (cos_a + sin_a))
        else:
            effective_size = puzzle_size
            
        half_size = effective_size // 2
        
        # X范围（根据CaptchaGenerator的验证逻辑）
        x_min = half_size
        x_max = half_size + 10
        
        # Y范围（确保滑块完全在图像内，留5像素安全边距）
        y_min = half_size + 5
        y_max = img_height - half_size - 5
        
        return (x_min, x_max), (y_min, y_max)
    
    def print_config(self) -> None:
        """打印当前配置（用于调试）"""
        print("=" * 60)
        print("Dataset Generation Configuration")
        print("=" * 60)
        
        print("\n[Position Sampling]")
        print(f"  Slider position: Calculated by calculate_safe_slider_range()")
        print(f"  Gap position: Calculated by calculate_safe_gap_range()")
        
        print("\n[Dataset Scale]")
        print(f"  Min backgrounds: {self.MIN_BACKGROUNDS}")
        print(f"  Samples per background: {self.MAX_SAMPLES_PER_BG}")
        
        print("\n[Confusion Strategy Distribution]")
        for strategy, count in self.CONFUSION_COUNTS.items():
            print(f"  {strategy}: {count} samples")
        print(f"  Total: {sum(self.CONFUSION_COUNTS.values())} samples")
        
        print("\n[Shape Configuration]")
        print(f"  Special shapes: {', '.join(self.SPECIAL_SHAPES)}")
        print(f"  Normal shapes count: {self.NORMAL_SHAPES_COUNT}")
        
        print("\n[Puzzle Sizes]")
        print(f"  Available sizes: {self.ALL_PUZZLE_SIZES}")
        print(f"  Sizes per image: {self.SIZES_PER_IMAGE}")
        
        print("\n[Save Configuration]")
        print(f"  Save full image: {self.SAVE_FULL_IMAGE}")
        print(f"  Save components: {self.SAVE_COMPONENTS}")
        print(f"  Component format: {self.COMPONENT_FORMAT}")
        print(f"  Save masks: {self.SAVE_MASKS}")
        
        print("=" * 60)


# 创建全局实例
dataset_config = None

def get_dataset_config() -> DatasetConfig:
    """获取数据集配置单例"""
    global dataset_config
    if dataset_config is None:
        dataset_config = DatasetConfig()
    return dataset_config


if __name__ == "__main__":
    # Test configuration
    config = get_dataset_config()
    config.print_config()
    
    # Validate configuration
    if config.validate():
        print("\nConfiguration validation passed")
    else:
        print("\nConfiguration validation failed")
    
    # Test size parsing
    print("\nTest size parsing:")
    test_cases = [
        "30,40,50",
        "30-60",
        "30-60:5",
        "45"
    ]
    for test in test_cases:
        result = DatasetConfig.parse_size_string(test)
        print(f"  '{test}' -> {result}")
    
    # Test dynamic position calculation
    print("\nTest dynamic position calculation:")
    test_sizes = [30, 40, 50, 60]
    for size in test_sizes:
        # Test slider range
        (x_min, x_max), (y_min, y_max) = config.calculate_safe_slider_range(size, img_height=160)
        print(f"  Puzzle size {size}:")
        print(f"    Slider X range: [{x_min}, {x_max}]")
        print(f"    Slider Y range: [{y_min}, {y_max}]")
        
        # Test gap range
        gap_x_min, gap_x_max = config.calculate_safe_gap_range(size, img_width=320, img_height=160)
        print(f"    Gap X range: [{gap_x_min}, {gap_x_max}]")