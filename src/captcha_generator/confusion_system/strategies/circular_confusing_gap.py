# -*- coding: utf-8 -*-
"""
圆形混淆缺口策略 - 专门用于圆形形状的多缺口混淆
包含Y轴距离相关的智能缩放规则
"""
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
from ..base import ConfusionStrategy, GapImage


class CircularConfusingGapConfusion(ConfusionStrategy):
    """圆形混淆缺口 - 专门处理圆形形状的多缺口混淆策略"""
    
    # Y轴距离阈值（单位：像素）
    Y_DISTANCE_THRESHOLD = 7
    
    # 缩放比例范围（当Y轴距离小于阈值时使用）
    SCALE_RANGES = [
        (0.8, 0.9),   # 缩小范围
        (1.1, 1.2)    # 放大范围
    ]
    
    @property
    def name(self) -> str:
        return "circular_confusing_gap"
    
    @property
    def description(self) -> str:
        return "圆形形状专用的多缺口混淆策略，包含智能缩放规则"
    
    def validate_config(self):
        """验证并设置配置 - 必须从YAML文件读取"""
        # 基础配置 - 必须从配置文件读取
        self.num_confusing_gaps = self.config.get('num_confusing_gaps')
        self.confusing_type = self.config.get('confusing_type')
        
        # 验证基础配置已提供
        assert self.num_confusing_gaps is not None, \
            "num_confusing_gaps must be provided in config (from YAML file)"
        assert self.confusing_type is not None, \
            "confusing_type must be provided in config (from YAML file)"
        
        # 圆形特定配置 - 必须从配置文件读取
        self.enable_smart_scaling = self.config.get('enable_smart_scaling')
        self.y_threshold = self.config.get('y_distance_threshold')
        self.scale_ranges = self.config.get('scale_ranges')
        self.normal_scale_range = self.config.get('normal_scale_range')
        
        # 验证圆形特定配置已提供
        assert self.enable_smart_scaling is not None, \
            "enable_smart_scaling must be provided in config (from YAML file)"
        assert self.y_threshold is not None, \
            "y_distance_threshold must be provided in config (from YAML file)"
        assert self.scale_ranges is not None, \
            "scale_ranges must be provided in config (from YAML file)"
        assert self.normal_scale_range is not None, \
            "normal_scale_range must be provided in config (from YAML file)"
        
        # 转换scale_ranges为元组列表（如果需要）
        if isinstance(self.scale_ranges, list):
            self.scale_ranges = [tuple(r) if isinstance(r, list) else r for r in self.scale_ranges]
        
        # 转换normal_scale_range为元组（如果需要）
        if isinstance(self.normal_scale_range, (list, tuple)):
            self.normal_scale_range = tuple(self.normal_scale_range)
        
        # 验证参数值的合法性
        assert isinstance(self.num_confusing_gaps, int) and 1 <= self.num_confusing_gaps <= 3, \
            f"num_confusing_gaps must be 1-3, got: {self.num_confusing_gaps}"
        assert self.confusing_type in ['same_y', 'different_y', 'mixed'], \
            f"confusing_type must be 'same_y', 'different_y', or 'mixed', got: {self.confusing_type}"
        assert isinstance(self.enable_smart_scaling, bool), \
            f"enable_smart_scaling must be boolean, got: {type(self.enable_smart_scaling)}"
        assert isinstance(self.y_threshold, (int, float)) and self.y_threshold > 0, \
            f"y_distance_threshold must be positive number, got: {self.y_threshold}"
        assert isinstance(self.scale_ranges, list) and len(self.scale_ranges) >= 1, \
            f"scale_ranges must be a non-empty list, got: {self.scale_ranges}"
        
        # 验证scale_ranges的格式
        for i, scale_range in enumerate(self.scale_ranges):
            assert isinstance(scale_range, (list, tuple)) and len(scale_range) == 2, \
                f"scale_ranges[{i}] must be a [min, max] pair, got: {scale_range}"
            assert scale_range[0] < scale_range[1], \
                f"scale_ranges[{i}] must have min < max, got: {scale_range}"
            assert 0.5 <= scale_range[0] <= 1.5 and 0.5 <= scale_range[1] <= 1.5, \
                f"scale_ranges[{i}] values must be in [0.5, 1.5], got: {scale_range}"
        
        # 验证normal_scale_range的格式
        assert isinstance(self.normal_scale_range, (tuple, list)) and len(self.normal_scale_range) == 2, \
            f"normal_scale_range must be a [min, max] pair, got: {self.normal_scale_range}"
        assert self.normal_scale_range[0] < self.normal_scale_range[1], \
            f"normal_scale_range must have min < max, got: {self.normal_scale_range}"
        assert 0.5 <= self.normal_scale_range[0] <= 1.5 and 0.5 <= self.normal_scale_range[1] <= 1.5, \
            f"normal_scale_range values must be in [0.5, 1.5], got: {self.normal_scale_range}"
        
        # 验证scale_ranges必须包含指定的范围
        required_ranges = [(0.8, 0.9), (1.1, 1.2)]
        for req_range in required_ranges:
            found = False
            for scale_range in self.scale_ranges:
                if (abs(scale_range[0] - req_range[0]) < 0.01 and 
                    abs(scale_range[1] - req_range[1]) < 0.01):
                    found = True
                    break
            assert found, \
                f"scale_ranges must include required range {req_range}, current: {self.scale_ranges}"
        
        # 初始化随机数生成器
        self.rng = np.random.RandomState(self.config.get('random_seed', None))
    
    def apply_to_gap(self, gap_image: GapImage) -> GapImage:
        """
        应用圆形混淆缺口策略
        
        Args:
            gap_image: 独立的gap图像（BGRA格式）
            
        Returns:
            GapImage: 返回原始gap图像（不修改）
        """
        # 获取真实gap信息
        real_pos = gap_image.position
        gap_h, gap_w = gap_image.image.shape[:2]
        
        # 验证是否为圆形（假设圆形的宽高比接近1）
        aspect_ratio = gap_w / gap_h
        is_circular = 0.9 <= aspect_ratio <= 1.1
        
        if not is_circular:
            print(f"Warning: Gap is not circular (aspect ratio: {aspect_ratio:.2f})")
        
        # 背景尺寸
        assert gap_image.background_size is not None, "背景尺寸必须提供给GapImage"
        bg_width, bg_height = gap_image.background_size
        
        # 生成混淆gap配置
        confusing_configs = self._generate_circular_confusing_configs(
            real_pos, gap_w, gap_h, self.num_confusing_gaps, bg_width, bg_height
        )
        
        # 清空之前的additional_gaps
        self.additional_gaps = []
        
        # 处理每个混淆gap
        for gap_config in confusing_configs:
            # 创建圆形掩码并应用变换
            transformed = self._transform_circular_gap(gap_image.image, gap_config)
            
            # 提取alpha通道作为掩码
            if transformed.shape[2] == 4:
                alpha_mask = transformed[:, :, 3].copy()
            else:
                alpha_mask = np.ones(transformed.shape[:2], dtype=np.uint8) * 255
            
            # 保存混淆gap信息
            self.additional_gaps.append({
                'position': gap_config['position'],
                'mask': alpha_mask,
                'size': (transformed.shape[1], transformed.shape[0]),
                'type': gap_config['type'],
                'rotation': gap_config.get('rotation', 0),
                'scale': gap_config.get('scale', 1.0),
                'y_distance': gap_config.get('y_distance', 0),
                'smart_scaled': gap_config.get('smart_scaled', False)
            })
        
        return gap_image
    
    def _calculate_smart_scale(self, y_distance: float) -> tuple:
        """
        根据Y轴距离计算智能缩放比例
        
        Args:
            y_distance: 与真实gap的Y轴距离
            
        Returns:
            (缩放比例, 是否使用了智能缩放)
        """
        # 如果Y轴距离小于阈值，必须使用特定的缩放范围
        if abs(y_distance) <= self.y_threshold and self.enable_smart_scaling:
            # 随机选择缩小或放大
            scale_range = self.rng.choice(self.scale_ranges)
            scale = self.rng.uniform(scale_range[0], scale_range[1])
            
            # 验证缩放值确实在指定范围内
            valid_in_range = False
            for allowed_range in self.scale_ranges:
                if allowed_range[0] <= scale <= allowed_range[1]:
                    valid_in_range = True
                    break
            
            if not valid_in_range:
                raise ValueError(
                    f"Scale {scale:.2f} not in required ranges {self.scale_ranges} "
                    f"for Y-distance {abs(y_distance):.1f}px (threshold: {self.y_threshold}px)"
                )
            
            return scale, True
        else:
            # Y轴距离足够大，使用常规缩放范围（从配置文件读取）
            scale = self.rng.uniform(self.normal_scale_range[0], self.normal_scale_range[1])
            
            # 验证没有意外使用了特定范围
            for allowed_range in self.scale_ranges:
                if allowed_range[0] <= scale <= allowed_range[1]:
                    raise ValueError(
                        f"Scale {scale:.2f} should not be in smart scale ranges {self.scale_ranges} "
                        f"when Y-distance {abs(y_distance):.1f}px > threshold {self.y_threshold}px. "
                        f"Should use normal_scale_range {self.normal_scale_range}"
                    )
            
            return scale, False
    
    def _transform_circular_gap(self, gap_image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        变换圆形gap图像
        
        Args:
            gap_image: 原始gap图像（BGRA）
            config: 变换配置
            
        Returns:
            变换后的gap图像
        """
        h, w = gap_image.shape[:2]
        center = (w // 2, h // 2)
        
        # 获取变换参数
        scale = config.get('scale', 1.0)
        rotation = config.get('rotation', 0)
        
        # 对于圆形，旋转不会改变形状，但可以旋转内部纹理
        # 创建缩放矩阵
        if scale != 1.0:
            # 计算缩放后的尺寸
            new_size = int(w * scale)
            
            # 使用高质量的插值方法缩放
            if scale < 1.0:
                interpolation = cv2.INTER_AREA  # 缩小时使用AREA插值
            else:
                interpolation = cv2.INTER_CUBIC  # 放大时使用CUBIC插值
            
            # 缩放图像
            transformed = cv2.resize(gap_image, (new_size, new_size), interpolation=interpolation)
            
            # 如果需要旋转内部纹理
            if rotation != 0:
                rot_matrix = cv2.getRotationMatrix2D(
                    (new_size // 2, new_size // 2), 
                    rotation, 
                    1.0
                )
                transformed = cv2.warpAffine(
                    transformed,
                    rot_matrix,
                    (new_size, new_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)
                )
        else:
            transformed = gap_image.copy()
        
        return transformed
    
    def _generate_circular_confusing_configs(self, 
                                            real_pos: Tuple[int, int],
                                            gap_w: int, 
                                            gap_h: int,
                                            num_gaps: int,
                                            bg_width: int, 
                                            bg_height: int) -> List[Dict[str, Any]]:
        """
        生成圆形混淆gap的配置
        
        Args:
            real_pos: 真实gap的中心位置
            gap_w, gap_h: gap的宽高
            num_gaps: 要生成的混淆gap数量
            bg_width, bg_height: 背景图像的宽高
            
        Returns:
            混淆gap配置列表
        """
        configs = []
        real_x, real_y = real_pos
        min_distance = 15  # 圆形需要更大的最小间距
        radius = max(gap_w, gap_h) // 2  # 圆形半径
        
        for i in range(num_gaps):
            # 决定混淆gap的类型
            if self.confusing_type == 'same_y':
                gap_type = 'same_y'
            elif self.confusing_type == 'different_y':
                gap_type = 'different_y'
            else:  # mixed
                gap_type = 'same_y' if i % 2 == 0 else 'different_y'
            
            # 尝试生成有效位置
            attempts = 0
            max_attempts = 100
            
            while attempts < max_attempts:
                if gap_type == 'same_y':
                    # 保持相同Y坐标
                    y = real_y
                    y_distance = 0
                    
                    # 在X方向找合适位置
                    if i % 2 == 0:  # 左侧
                        x_min = radius + 10
                        x_max = real_x - 2 * radius - min_distance
                        if x_min < x_max:
                            x = self.rng.randint(x_min, x_max)
                        else:
                            attempts += 1
                            continue
                    else:  # 右侧
                        x_min = real_x + 2 * radius + min_distance
                        x_max = bg_width - radius - 10
                        if x_min < x_max:
                            x = self.rng.randint(x_min, x_max)
                        else:
                            attempts += 1
                            continue
                else:  # different_y
                    # 随机X位置
                    x = self.rng.randint(radius + 10, bg_width - radius - 10)
                    
                    # 计算Y位置（确保有足够偏移）
                    y_offset = self.rng.randint(10, 50)
                    if self.rng.random() < 0.5:
                        y = real_y - y_offset
                    else:
                        y = real_y + y_offset
                    
                    # 确保Y在边界内
                    y = np.clip(y, radius + 10, bg_height - radius - 10)
                    y_distance = abs(y - real_y)
                
                # 根据Y距离计算缩放比例
                scale, smart_scaled = self._calculate_smart_scale(y_distance)
                
                # 计算缩放后的半径
                scaled_radius = int(radius * scale)
                
                # 验证位置是否有效
                if (scaled_radius < x < bg_width - scaled_radius and
                    scaled_radius < y < bg_height - scaled_radius):
                    
                    # 检查与真实gap的距离
                    dist = np.sqrt((x - real_x)**2 + (y - real_y)**2)
                    min_safe_dist = radius + scaled_radius + min_distance
                    
                    if dist >= min_safe_dist:
                        # 检查与其他混淆gaps的距离
                        valid = True
                        for cfg in configs:
                            cx, cy = cfg['position']
                            other_scale = cfg.get('scale', 1.0)
                            other_radius = int(radius * other_scale)
                            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                            if dist < scaled_radius + other_radius + min_distance:
                                valid = False
                                break
                        
                        if valid:
                            configs.append({
                                'position': (x, y),
                                'type': gap_type,
                                'scale': scale,
                                'rotation': self.rng.uniform(-15, 15),  # 小幅旋转纹理
                                'y_distance': y_distance,
                                'smart_scaled': smart_scaled
                            })
                            break
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"Warning: Could not find valid position for confusing gap {i+1}")
                break
        
        return configs
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取策略的元数据"""
        metadata = super().get_metadata()
        metadata.update({
            "strategy_type": "circular_confusing_gap",
            "num_confusing_gaps": self.num_confusing_gaps,
            "confusing_type": self.confusing_type,
            "enable_smart_scaling": self.enable_smart_scaling,
            "y_distance_threshold": self.y_threshold,
            "scale_ranges": self.scale_ranges,
            "gaps_info": [
                {
                    "position": gap['position'],
                    "scale": gap['scale'],
                    "y_distance": gap.get('y_distance', 0),
                    "smart_scaled": gap.get('smart_scaled', False)
                }
                for gap in getattr(self, 'additional_gaps', [])
            ]
        })
        return metadata