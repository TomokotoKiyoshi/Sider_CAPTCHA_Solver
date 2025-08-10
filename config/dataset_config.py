# -*- coding: utf-8 -*-
"""
数据集生成配置文件
集中管理验证码数据集生成的所有参数
"""
from typing import List, Dict, Tuple
import numpy as np

class DatasetConfig:
    """数据集生成配置"""
    
    # ========== 输入输出路径配置 ==========
    INPUT_DIR: str = 'data/raw'                     # 输入目录（背景图片）
    OUTPUT_DIR: str = 'data'                        # 输出目录
    
    # ========== 并行处理配置 ==========
    MAX_WORKERS: int = None                         # 工作进程数（None表示自动）
    MAX_IMAGES: int = None                          # 最大处理图片数（None表示全部）
    RANDOM_SEED: int = None                         # 随机种子（None表示随机）
    
    # ========== 图像尺寸配置 ==========
    USE_RANDOM_SIZE: bool = True                    # 是否使用随机尺寸
    SIZE_CONFIG_MODULE: str = 'config.size_confusion_config'  # 尺寸配置模块
    
    # ========== 位置采样范围 ==========
    SLIDER_X_RANGE: Tuple[int, int] = (20, 35)     # 滑块x范围（基于320x160）
    SLIDER_Y_RANGE: Tuple[int, int] = (30, 130)    # 滑块y范围（基于320x160）
    
    # ========== 数据集规模 ==========
    MIN_BACKGROUNDS: int = 2000        # 最少背景图数量
    MAX_SAMPLES_PER_BG: int = 100      # 每个背景最多样本数
    
    # ========== 混淆策略数量控制 ==========
    CONFUSION_COUNTS: Dict[str, int] = {
        'none': 30,           # 30个无混淆
        'rotation': 10,       # 10个旋转
        'perlin_noise': 10,   # 10个噪声
        'highlight': 10,      # 10个高光
        'confusing_gap': 10,  # 10个混淆缺口
        'hollow_center': 10,  # 10个空心中心
        'combined': 20        # 20个组合混淆（随机2-3种）
    }  # 总计100个
    
    # ========== 形状配置 ==========
    SPECIAL_SHAPES: List[str] = ['circle', 'square', 'triangle', 'hexagon']
    NORMAL_SHAPES_COUNT: int = 6  # 普通拼图形状数量
    
    # ========== Puzzle尺寸配置 ==========
    ALL_PUZZLE_SIZES: List[int] = list(range(40, 62, 2))  # 所有可能的拼图大小（30-60，步长2）
    SIZES_PER_IMAGE: int = 4  # 每张图片使用的拼图大小数量
    
    # ========== 组件保存配置 ==========
    SAVE_FULL_IMAGE: bool = False  # 不保存完整合成图片
    SAVE_COMPONENTS: bool = False  # 默认不保存单独的滑块和缺口组件
    COMPONENT_FORMAT: str = 'png'  # 组件图片格式
    SAVE_MASKS: bool = False       # 是否额外保存二值mask
    
    # ========== Gap位置生成配置 ==========
    GAP_X_COUNT: int = 4  # 每张图片的缺口x轴位置数量
    GAP_Y_COUNT: int = 3  # 每张图片的缺口y轴位置数量
    SLIDER_X_COUNT: int = 4  # 滑块x轴位置数量
    
    # ========== 旋转配置 ==========
    ROTATION_ENABLED: bool = True  # 是否启用旋转混淆
    MAX_ROTATION_ANGLE: float = 1.8  # 最大旋转角度（度）
    
    @classmethod
    def validate(cls) -> bool:
        """验证配置的合理性"""
        # 验证混淆策略数量总和
        total_confusion = sum(cls.CONFUSION_COUNTS.values())
        if total_confusion != cls.MAX_SAMPLES_PER_BG:
            print(f"Warning: Total confusion count ({total_confusion}) does not match samples per background ({cls.MAX_SAMPLES_PER_BG})")
            print("Adjusting confusion counts proportionally...")
            # 自动调整混淆策略数量以匹配样本数
            cls._adjust_confusion_counts()
            return True  # 继续执行，但已调整
        
        # 验证puzzle尺寸
        if not cls.ALL_PUZZLE_SIZES:
            print("Error: No puzzle sizes defined")
            return False
        
        if cls.SIZES_PER_IMAGE > len(cls.ALL_PUZZLE_SIZES):
            print(f"Warning: Sizes per image ({cls.SIZES_PER_IMAGE}) exceeds available sizes ({len(cls.ALL_PUZZLE_SIZES)})")
            return False
        
        # 验证位置范围
        if cls.SLIDER_X_RANGE[0] >= cls.SLIDER_X_RANGE[1]:
            print("Error: Invalid slider X range")
            return False
        
        if cls.SLIDER_Y_RANGE[0] >= cls.SLIDER_Y_RANGE[1]:
            print("Error: Invalid slider Y range")
            return False
        
        return True
    
    @classmethod
    def _adjust_confusion_counts(cls) -> None:
        """按比例调整混淆策略数量以匹配样本数"""
        original_total = sum(cls.CONFUSION_COUNTS.values())
        if original_total == 0:
            # 如果没有定义混淆策略，使用默认分配
            cls.CONFUSION_COUNTS = {
                'none': cls.MAX_SAMPLES_PER_BG,
                'rotation': 0,
                'perlin_noise': 0,
                'highlight': 0,
                'confusing_gap': 0,
                'hollow_center': 0,
                'combined': 0
            }
            return
        
        # 按比例调整每种策略的数量
        ratio = cls.MAX_SAMPLES_PER_BG / original_total
        adjusted_counts = {}
        total_assigned = 0
        
        for strategy, count in cls.CONFUSION_COUNTS.items():
            adjusted = int(count * ratio)
            adjusted_counts[strategy] = adjusted
            total_assigned += adjusted
        
        # 处理舍入误差，将剩余的分配给'none'或第一个非零策略
        remaining = cls.MAX_SAMPLES_PER_BG - total_assigned
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
        
        cls.CONFUSION_COUNTS = adjusted_counts
    
    @classmethod
    def update_puzzle_sizes(cls, sizes: List[int]) -> None:
        """更新puzzle尺寸列表"""
        cls.ALL_PUZZLE_SIZES = sorted(sizes)
        print(f"Updated puzzle sizes: {cls.ALL_PUZZLE_SIZES}")
    
    @classmethod
    def parse_size_string(cls, size_str: str) -> List[int]:
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
    
    @classmethod
    def calculate_safe_gap_range(cls, puzzle_size: int, img_width: int = 320, 
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
            with_rotation = cls.ROTATION_ENABLED
            
        if with_rotation:
            # 考虑旋转后的尺寸增大
            angle_rad = np.radians(cls.MAX_ROTATION_ANGLE)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            # 计算旋转后的边界框
            effective_size = int(puzzle_size * (cos_a + sin_a))
        else:
            effective_size = puzzle_size
            
        half_size = effective_size // 2
        
        # 计算滑块的最大位置
        slider_x_max = min(effective_size // 2 + 10, 40)
        
        # 缺口必须在滑块右边，并留有间隔
        gap_x_min = slider_x_max + effective_size + 10
        gap_x_max = img_width - half_size
        
        # 确保范围有效
        if gap_x_min > gap_x_max:
            # 对于大尺寸，可能没有有效范围
            return gap_x_min, gap_x_min  # 返回相同值表示无效
            
        return gap_x_min, gap_x_max
    
    @classmethod
    def calculate_safe_slider_range(cls, puzzle_size: int, img_height: int = 160,
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
            with_rotation = cls.ROTATION_ENABLED
            
        if with_rotation:
            angle_rad = np.radians(cls.MAX_ROTATION_ANGLE)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            effective_size = int(puzzle_size * (cos_a + sin_a))
        else:
            effective_size = puzzle_size
            
        half_size = effective_size // 2
        
        # X范围
        x_min = half_size
        x_max = min(half_size + 10, 40)
        
        # Y范围
        y_min = max(cls.SLIDER_Y_RANGE[0], half_size + 5)
        y_max = min(cls.SLIDER_Y_RANGE[1], img_height - half_size - 5)
        
        return (x_min, x_max), (y_min, y_max)
    
    @classmethod
    def print_config(cls) -> None:
        """打印当前配置（用于调试）"""
        print("=" * 60)
        print("数据集生成配置")
        print("=" * 60)
        
        print("\n[位置采样]")
        print(f"  滑块X范围: {cls.SLIDER_X_RANGE}")
        print(f"  滑块Y范围: {cls.SLIDER_Y_RANGE}")
        
        print("\n[数据集规模]")
        print(f"  最少背景图: {cls.MIN_BACKGROUNDS}")
        print(f"  每背景样本数: {cls.MAX_SAMPLES_PER_BG}")
        
        print("\n[混淆策略分布]")
        for strategy, count in cls.CONFUSION_COUNTS.items():
            print(f"  {strategy}: {count}个")
        print(f"  总计: {sum(cls.CONFUSION_COUNTS.values())}个")
        
        print("\n[形状配置]")
        print(f"  特殊形状: {', '.join(cls.SPECIAL_SHAPES)}")
        print(f"  普通形状数: {cls.NORMAL_SHAPES_COUNT}")
        
        print("\n[Puzzle尺寸]")
        print(f"  可用尺寸: {cls.ALL_PUZZLE_SIZES}")
        print(f"  每图使用: {cls.SIZES_PER_IMAGE}种尺寸")
        
        print("\n[保存配置]")
        print(f"  保存完整图片: {cls.SAVE_FULL_IMAGE}")
        print(f"  保存组件: {cls.SAVE_COMPONENTS}")
        print(f"  组件格式: {cls.COMPONENT_FORMAT}")
        print(f"  保存Mask: {cls.SAVE_MASKS}")
        
        print("=" * 60)


if __name__ == "__main__":
    # 测试配置
    DatasetConfig.print_config()
    
    # 验证配置
    if DatasetConfig.validate():
        print("\n配置验证通过 ✓")
    else:
        print("\n配置验证失败 ✗")
    
    # 测试尺寸解析
    print("\n测试尺寸解析:")
    test_cases = [
        "30,40,50",
        "30-60",
        "30-60:5",
        "45"
    ]
    for test in test_cases:
        result = DatasetConfig.parse_size_string(test)
        print(f"  '{test}' -> {result}")