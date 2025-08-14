# -*- coding: utf-8 -*-
"""
混淆缺口策略 - 复制并变换gap到其他位置作为假缺口
"""
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple
from ..base import ConfusionStrategy, GapImage
import sys
from pathlib import Path
# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from src.captcha_generator.utils.geometric_center import calculate_geometric_center


class ConfusingGapConfusion(ConfusionStrategy):
    """混淆缺口 - 复制gap并进行旋转/缩放，放置在其他位置作为干扰"""
    
    @property
    def name(self) -> str:
        return "confusing_gap"
    
    @property
    def description(self) -> str:
        return "复制并变换gap图像到其他位置，创建视觉干扰"
    
    def validate_config(self):
        """验证并设置默认配置"""
        # 简化的配置参数
        self.num_confusing_gaps = self.config.get('num_confusing_gaps')
        self.confusing_type = self.config.get('confusing_type')
        
        assert self.num_confusing_gaps is not None, "num_confusing_gaps must be provided in config"
        assert self.confusing_type is not None, "confusing_type must be provided in config"
        
        # 验证参数
        assert isinstance(self.num_confusing_gaps, int) and 1 <= self.num_confusing_gaps <= 3, \
            f"num_confusing_gaps must be 1-3, got: {self.num_confusing_gaps}"
        assert self.confusing_type in ['same_y', 'different_y', 'mixed'], \
            f"confusing_type must be 'same_y', 'different_y', or 'mixed', got: {self.confusing_type}"
        
        # 初始化随机数生成器（可选：使用固定种子以便复现）
        self.rng = np.random.RandomState(self.config.get('random_seed', None))
    
    def apply_to_gap(self, gap_image: GapImage) -> GapImage:
        """
        混淆缺口策略：生成额外的gap信息但不修改原始gap
        
        Args:
            gap_image: 独立的gap图像（BGRA格式）
            
        Returns:
            GapImage: 返回原始gap图像（不修改）
        """
        # 获取真实gap信息
        real_pos = gap_image.position
        gap_h, gap_w = gap_image.image.shape[:2]
        
        # 背景尺寸必须提供
        assert gap_image.background_size is not None, "背景尺寸必须提供给GapImage"
        bg_width, bg_height = gap_image.background_size
        
        # 生成混淆gap配置
        confusing_configs = self._generate_confusing_configs(
            real_pos, gap_w, gap_h, self.num_confusing_gaps, bg_width, bg_height
        )
        
        # 清空之前的additional_gaps
        self.additional_gaps = []
        
        # 处理每个混淆gap，生成额外的gap信息
        for gap_config in confusing_configs:
            # 复制并变换gap
            transformed = self._transform_gap(gap_image.image, gap_config)
            
            # 提取变换后的掩码（用于在背景上创建缺口）
            if transformed.shape[2] == 4:
                # 保留完整的alpha通道，包括hollow效果
                alpha_mask = transformed[:, :, 3].copy()
            else:
                # 如果没有alpha通道，创建一个全255的掩码
                alpha_mask = np.ones(transformed.shape[:2], dtype=np.uint8) * 255
            
            # 计算变换后形状的几何中心
            # 获取形状类型
            shape_type = gap_image.metadata.get('puzzle_shape', 'unknown') if gap_image.metadata else 'unknown'
            
            # 重要：对于旋转后的图像，我们需要计算旋转后的实际质心
            # 而不是假设质心相对位置不变
            mask_geometric_center = calculate_geometric_center(alpha_mask, shape_type)
            
            # 计算绝对几何中心位置
            # gap_config['position'] 是混淆缺口的预定放置位置（边界框中心）
            intended_center = gap_config['position']
            actual_h, actual_w = alpha_mask.shape[:2]
            
            # 计算从实际边界框中心到几何中心的偏移
            # 这个偏移是基于旋转后的实际mask计算的
            offset_x = mask_geometric_center[0] - actual_w // 2
            offset_y = mask_geometric_center[1] - actual_h // 2
            
            # 绝对几何中心 = 放置位置（边界框中心） + 偏移
            # 混淆缺口会被放置为其边界框中心在intended_center
            absolute_geometric_center = (
                intended_center[0] + offset_x,
                intended_center[1] + offset_y
            )
            
            # 保存额外的gap信息（使用完整的alpha通道而不是二值掩码）
            # 重要：size必须与mask的实际尺寸一致，用于正确放置
            self.additional_gaps.append({
                'position': gap_config['position'],  # 边界框中心位置（用于放置）
                'geometric_center': absolute_geometric_center,  # 几何中心位置（真实中心）
                'mask': alpha_mask,  # 完整的alpha通道（保留hollow效果）
                'size': (actual_w, actual_h),  # 实际的mask尺寸 (w, h)
                'actual_size': (actual_w, actual_h),  # 明确记录实际尺寸
                'type': gap_config['type'],
                'rotation': gap_config.get('rotation', 0),
                'scale': gap_config.get('scale', 1.0),
                'shape': shape_type  # 保存形状类型
            })
        
        # 返回原始gap图像（不修改）
        return gap_image
    
    def _calculate_transformed_bounds(self, w: int, h: int, angle: float, scale: float) -> Tuple[int, int]:
        """
        计算变换后的边界框大小
        
        Args:
            w, h: 原始宽高
            angle: 旋转角度（度）
            scale: 缩放比例
            
        Returns:
            (new_width, new_height): 变换后的边界框尺寸
        """
        angle_rad = np.radians(abs(angle))
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # 考虑缩放后的尺寸
        scaled_w = w * scale
        scaled_h = h * scale
        
        # 计算旋转后的边界框
        new_w = int(scaled_h * sin_angle + scaled_w * cos_angle)
        new_h = int(scaled_h * cos_angle + scaled_w * sin_angle)
        
        return new_w, new_h
    
    def _transform_gap(self, gap_image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        根据配置变换gap图像
        
        Args:
            gap_image: 原始gap图像（BGRA）
            config: 变换配置（rotation, scale）
            
        Returns:
            变换后的gap图像
        """
        h, w = gap_image.shape[:2]
        center = (w // 2, h // 2)
        
        # 获取变换参数
        angle = config.get('rotation', 0)
        scale = config.get('scale', 1.0)
        
        # 如果没有变换，直接返回副本
        if angle == 0 and scale == 1.0:
            return gap_image.copy()
        
        # 使用统一的函数计算变换后的边界框大小
        new_w, new_h = self._calculate_transformed_bounds(w, h, angle, scale)
        
        # 创建变换矩阵（旋转+缩放）
        # 注意：旋转是围绕原始图像中心进行的
        transform_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 调整平移量，使旋转后的图像居中在新的边界框中
        transform_matrix[0, 2] += (new_w - w) / 2
        transform_matrix[1, 2] += (new_h - h) / 2
        
        # 应用变换
        transformed = cv2.warpAffine(
            gap_image,
            transform_matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        return transformed
    
    def _generate_confusing_configs(self, real_pos: Tuple[int, int], 
                                   gap_w: int, gap_h: int, 
                                   num_gaps: int,
                                   bg_width: int, bg_height: int) -> List[Dict[str, Any]]:
        """
        自动生成混淆gap的配置
        
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
        min_distance = 10  # 最小间距
        
        # 根据类型决定生成策略
        for i in range(num_gaps):
            # 决定这个混淆gap的类型
            if self.confusing_type == 'same_y':
                gap_type = 'same_y'
            elif self.confusing_type == 'different_y':
                gap_type = 'different_y'
            else:  # mixed
                gap_type = 'same_y' if i % 2 == 0 else 'different_y'
            
            # 生成变换参数
            # 旋转角度：10-30度，随机选择正负
            rotation = self.rng.uniform(10, 30)
            if self.rng.random() < 0.5:
                rotation = -rotation  # 50%概率变为负值
            scale = self.rng.uniform(0.7, 1.3)
            
            # same_y类型已经通过生成10-30度旋转确保了有明显变换
            
            # 使用统一的函数计算变换后的边界框大小
            new_width, new_height = self._calculate_transformed_bounds(gap_w, gap_h, rotation, scale)
            
            # 使用最大维度作为安全边界（添加更大的安全边距）
            transformed_bound = max(new_width, new_height) + 10  # +10像素安全边距
            safe_radius = int(transformed_bound / 2)
            
            # 更保守的边界设置，确保混淆缺口完全在图像内
            # 左侧边界：确保混淆缺口左边缘不会超出图像左边界
            left_boundary = max(safe_radius + 5, gap_w + 10 + safe_radius)  # +5额外安全边距
            # 右侧边界：确保混淆缺口右边缘不会超出图像右边界
            right_boundary = min(bg_width - safe_radius - 5, bg_width - safe_radius)  # -5额外安全边距
            
            # Y轴的安全边界（这是关键！）
            top_boundary = safe_radius + 5  # 顶部边界
            bottom_boundary = bg_height - safe_radius - 5  # 底部边界
            
            # 生成位置
            attempts = 0
            while attempts < 100:
                if gap_type == 'same_y':
                    # 确保y坐标在安全边界内
                    y = np.clip(real_y, top_boundary, bottom_boundary)
                    # 在x方向上找位置，避开真实gap
                    if i % 2 == 0:  # 尝试左侧
                        # 从左边界到真实gap左侧
                        left_limit = left_boundary
                        right_limit = real_x - int(gap_w/2) - min_distance - safe_radius
                        if left_limit < right_limit:
                            x = self.rng.randint(left_limit, right_limit)
                        else:
                            # 左侧空间不足，尝试右侧
                            i += 1  # 切换到右侧逻辑
                            continue
                    else:  # 尝试右侧
                        # 从真实gap右侧到右边界
                        left_limit = real_x + int(gap_w/2) + min_distance + safe_radius
                        if left_limit < right_boundary:
                            x = self.rng.randint(left_limit, right_boundary)
                        else:
                            # 右侧空间也不足
                            attempts = 100
                            break
                else:  # different_y
                    # 先随机选择x位置
                    x = self.rng.randint(left_boundary, right_boundary)
                    
                    # 检查x方向是否与真实gap有重叠
                    x_overlap = abs(x - real_x) < (gap_w + transformed_bound) / 2
                    
                    if x_overlap:
                        # x有重叠，需要确保y有足够间距避免重叠
                        y_options = []
                        # 上方区域（使用top_boundary）
                        if real_y - int(gap_h/2) - min_distance - safe_radius > top_boundary:
                            y_options.extend(range(top_boundary, real_y - int(gap_h/2) - min_distance - safe_radius))
                        # 下方区域（使用bottom_boundary）
                        if real_y + int(gap_h/2) + min_distance + safe_radius < bottom_boundary:
                            y_options.extend(range(real_y + int(gap_h/2) + min_distance + safe_radius, bottom_boundary))
                        
                        if y_options:
                            y = self.rng.choice(y_options)
                        else:
                            # 如果垂直方向没有空间，重新选择x位置
                            attempts += 1
                            continue
                    else:
                        # x没有重叠，y只需偏移10px以上，确保不超出边界
                        # 使用已定义的边界
                        y_min = top_boundary  # 使用top_boundary
                        y_max = bottom_boundary  # 使用bottom_boundary
                        
                        # 向上和向下的可选范围（至少偏移10px）
                        # 向上偏移（y值变小）
                        y_up_max = real_y - min_distance  # 最接近真实gap的位置（至少偏移10px）
                        y_up_min = max(y_min, real_y - 50)  # 最远离真实gap的位置（最多偏移50px）
                        
                        # 向下偏移（y值变大）
                        y_down_min = real_y + min_distance  # 最接近真实gap的位置（至少偏移10px）
                        y_down_max = min(y_max, real_y + 50)  # 最远离真实gap的位置（最多偏移50px）
                        
                        # 收集所有可用的y值
                        y_candidates = []
                        if y_up_min <= y_up_max:
                            y_candidates.extend(range(y_up_min, y_up_max + 1))
                        if y_down_min <= y_down_max:
                            y_candidates.extend(range(y_down_min, y_down_max + 1))
                        
                        if y_candidates:
                            y = self.rng.choice(y_candidates)
                        else:
                            # 没有合适的位置，重试
                            attempts += 1
                            continue
                
                # 检查是否与其他gaps重叠
                valid = True
                # 检查与真实gap的距离（使用变换后的尺寸）
                dist = np.sqrt((x - real_x)**2 + (y - real_y)**2)
                # 真实gap保持原始尺寸，混淆gap使用变换后的尺寸
                min_safe_dist = (max(gap_w, gap_h) + transformed_bound) / 2 + min_distance
                if dist < min_safe_dist:
                    valid = False
                
                # 检查与其他混淆gaps的距离
                for cfg in configs:
                    cx, cy = cfg['position']
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < transformed_bound + min_distance:
                        valid = False
                        break
                
                if valid:
                    # 最终边界检查
                    if (x - safe_radius >= 0 and x + safe_radius <= bg_width and
                        y - safe_radius >= 0 and y + safe_radius <= bg_height):
                        configs.append({
                            'position': (x, y),
                            'type': gap_type,
                            'rotation': rotation,
                            'scale': scale
                        })
                        break
                    else:
                        # 位置超出边界，跳过
                        attempts += 1
                        continue
                
                attempts += 1
            
            # 如果找不到合适位置，减少混淆gap数量
            if attempts >= 100:
                break
        
        return configs
    
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取策略的元数据"""
        metadata = super().get_metadata()
        metadata.update({
            "num_confusing_gaps": self.num_confusing_gaps,
            "confusing_type": self.confusing_type
        })
        return metadata