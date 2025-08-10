# -*- coding: utf-8 -*-
"""
验证码生成器 - 从完整图片生成基础验证码
"""
import numpy as np
import cv2
from typing import Union, Optional, Tuple, List, Dict, Any, TYPE_CHECKING
import os
import sys

if TYPE_CHECKING:
    from .base import ConfusionStrategy

# 添加父目录到路径以导入现有模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ..puzzle_shapes_generator import create_common_puzzle_piece
from ..puzzle_shapes_generator import create_special_puzzle_piece
from ..lighting import apply_gap_lighting, apply_gap_highlighting, apply_slider_lighting
from .base import CaptchaResult, GapImage


class CaptchaGenerator:
    """验证码生成器 - 从完整图片生成基础验证码"""
    
    def generate(self,
                 image: Union[str, np.ndarray],
                 puzzle_shape: Union[str, Tuple[str, ...]],
                 puzzle_size: int,
                 gap_position: Tuple[int, int],
                 slider_position: Tuple[int, int],
                 confusion_strategies: Optional[List['ConfusionStrategy']] = None) -> CaptchaResult:
        """
        从完整图片生成基础验证码
        
        Args:
            image: 输入的完整图片路径或数组
            puzzle_shape: 拼图形状
            puzzle_size: 拼图大小
            gap_position: 缺口位置 (x, y)，x范围[65, 305]，y范围[30, 130]
            slider_position: 滑块位置 (x, y)，x范围[15, 35]，y必须等于gap_position的y
            confusion_strategies: 混淆策略列表（可选）
            
        Returns:
            CaptchaResult: 生成的验证码结果
        """
        # 1. 加载并调整图片
        img = self._load_and_resize_image(image)
        
        # 2. 创建拼图掩码
        gap_mask = self._create_puzzle_mask(puzzle_shape, puzzle_size)
        
        # 3. 验证位置合法性（传入实际图像尺寸）
        self._validate_positions(gap_position, slider_position, gap_mask.shape, img.shape)
        
        # 4. 提取原始区域作为基础图像（将成为滑块和缺口的来源）
        original_piece = self._extract_gap_image(img, gap_position, gap_mask, puzzle_shape)
        
        # 5. 应用混淆策略
        confusion_metadata = {}
        all_additional_gaps = []
        has_highlight = False
        highlight_params = None
        has_rotation = False
        rotated_piece_for_background = None
        
        # 滑块图像（会应用除旋转外的所有混淆）
        slider_image = original_piece
        # 背景缺口图像（用于需要同时影响滑块和背景的策略）
        background_piece = original_piece
        
        # 分两阶段处理策略，确保confusing_gap在hollow_center之后
        if confusion_strategies:
            # 第一阶段：处理除了confusing_gap之外的所有策略
            confusing_gap_strategy = None
            for strategy in confusion_strategies:
                if strategy.name == 'confusing_gap':
                    # 暂时保存，稍后处理
                    confusing_gap_strategy = strategy
                    continue
                    
                if strategy.name == 'rotation':
                    # 旋转策略：只用于背景缺口，不影响滑块
                    has_rotation = True
                    rotated_piece_for_background = strategy.apply_to_gap(background_piece)
                    confusion_metadata[strategy.name] = strategy.get_metadata()
                elif strategy.name == 'highlight':
                    # 高光策略：只是标记，不修改滑块
                    has_highlight = True
                    highlight_params = strategy.get_metadata()
                    confusion_metadata[strategy.name] = strategy.get_metadata()
                elif strategy.name == 'hollow_center':
                    # 空心策略：同时应用到滑块和背景缺口
                    slider_image = strategy.apply_to_gap(slider_image)
                    background_piece = strategy.apply_to_gap(background_piece)
                    # 如果存在旋转版本，也要应用hollow效果
                    if has_rotation and rotated_piece_for_background:
                        rotated_piece_for_background = strategy.apply_to_gap(rotated_piece_for_background)
                    confusion_metadata[strategy.name] = strategy.get_metadata()
                else:
                    # 检查是否是已知的策略
                    known_strategies = ['perlin_noise', 'rotation', 'highlight', 'confusing_gap', 'hollow_center']
                    if strategy.name not in known_strategies:
                        raise ValueError(f"Unknown confusion strategy: {strategy.name}. Known strategies are: {', '.join(known_strategies)}")
                    
                    # 其他策略（perlin_noise等）：只应用到滑块
                    slider_image = strategy.apply_to_gap(slider_image)
                    confusion_metadata[strategy.name] = strategy.get_metadata()
            
            # 第二阶段：处理confusing_gap策略（使用已应用其他效果的background_piece）
            if confusing_gap_strategy:
                # 使用已应用了其他效果（如空心）的background_piece
                confusing_gap_strategy.apply_to_gap(background_piece)
                confusion_metadata[confusing_gap_strategy.name] = confusing_gap_strategy.get_metadata()
                additional_gaps = confusing_gap_strategy.get_additional_gaps()
                if additional_gaps:
                    all_additional_gaps.extend(additional_gaps)
        
        # 6. 提取最终的滑块（已应用混淆效果，除了旋转）
        slider = self._extract_slider_from_gap(slider_image)
        
        # 6.5 应用滑块光照效果
        slider = apply_slider_lighting(slider)
        
        # 7. 提取用于背景的mask（如果有旋转，使用旋转版本；否则使用背景版本）
        # 注意：这里使用的是已经应用了hollow效果的background_piece
        if has_rotation and rotated_piece_for_background:
            background_mask = self._extract_mask_from_gap_image(rotated_piece_for_background)
        else:
            background_mask = self._extract_mask_from_gap_image(background_piece)
        
        # 8. 创建带缺口的背景（使用混淆后的mask）
        # 传递已应用hollow效果的piece（优先使用旋转版本）
        piece_for_background = rotated_piece_for_background if (has_rotation and rotated_piece_for_background) else background_piece
        background = self._create_gap_background(
            img, background_mask, gap_position, all_additional_gaps,
            has_highlight=has_highlight, highlight_params=highlight_params,
            rotated_piece_for_background=piece_for_background
        )
        
        # 确定混淆类型
        confusion_types = [s.name for s in confusion_strategies] if confusion_strategies else []
        confusion_type = ','.join(confusion_types) if confusion_types else 'none'
        
        return CaptchaResult(
            background=background,
            slider=slider,
            gap_position=gap_position,
            slider_position=slider_position,
            gap_mask=gap_mask,
            confusion_type=confusion_type,
            confusion_params=confusion_metadata,
            metadata={
                'puzzle_shape': str(puzzle_shape),
                'puzzle_size': puzzle_size,
                'original_image_shape': img.shape,
                'applied_confusions': confusion_types
            },
            additional_gaps=all_additional_gaps if all_additional_gaps else None
        )
    
    def _load_and_resize_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """加载图片（不再强制调整大小，支持动态尺寸）"""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Cannot load image: {image}")
        else:
            img = image.copy()
        
        # 不再强制调整到320x160，保留原始尺寸
        # 图片尺寸应该已经在外部处理好了
        
        return img
    
    def _create_puzzle_mask(self, 
                           puzzle_shape: Union[str, Tuple[str, ...]], 
                           puzzle_size: int) -> np.ndarray:
        """创建拼图掩码"""
        if isinstance(puzzle_shape, str):
            # 特殊形状（圆形、方形等）
            mask = create_special_puzzle_piece(puzzle_shape, puzzle_size)
        else:
            # 不规则拼图形状
            knob_radius_ratio = 0.2
            mask = create_common_puzzle_piece(
                piece_size=puzzle_size,
                knob_radius_ratio=knob_radius_ratio,
                edges=puzzle_shape
            )
        
        return mask
    
    def _validate_positions(self,
                           gap_position: Tuple[int, int],
                           slider_position: Tuple[int, int],
                           mask_shape: Tuple[int, int],
                           img_shape: Tuple[int, int, int]) -> None:
        """验证位置参数"""
        gap_x, gap_y = gap_position
        slider_x, slider_y = slider_position
        h, w = mask_shape[:2]
        
        # 使用实际图像尺寸而不是硬编码值
        bg_height, bg_width = img_shape[:2]
        
        # 验证滑块位置
        # 滑块中心x坐标必须在 [w/2, w/2 + 10] 范围内
        min_slider_x = w // 2
        max_slider_x = w // 2 + 10
        if not (min_slider_x <= slider_x <= max_slider_x):
            raise ValueError(f"Slider x must be in range [{min_slider_x}, {max_slider_x}], got: {slider_x}")
        
        # 验证滑块y坐标（确保滑块完全在背景内）
        if slider_y - h//2 < 0:
            raise ValueError(f"Slider position causes out of bounds (top edge), y={slider_y}")
        
        if slider_y + h//2 > bg_height:
            raise ValueError(f"Slider position causes out of bounds (bottom edge), y={slider_y}")
        
        # 缺口和滑块必须有相同的y坐标
        if gap_y != slider_y:
            raise ValueError(f"Gap y ({gap_y}) must equal slider y ({slider_y})")
        
        # 验证缺口和滑块不重叠
        # 计算滑块和缺口的边界
        slider_left = slider_x - w//2
        slider_right = slider_x + w//2
        gap_left = gap_x - w//2
        gap_right = gap_x + w//2
        
        # 检查水平方向上是否有重叠（因为y坐标相同，只需检查x方向）
        if not (gap_right <= slider_left or gap_left >= slider_right):
            raise ValueError(f"Gap and slider overlap! Gap x={gap_x}, Slider x={slider_x}, size={w}")
        
        # 验证缺口位置（确保缺口完全在背景内）
        if gap_x - w//2 < 0:
            raise ValueError(f"Gap position causes out of bounds (left edge), x={gap_x}")
        
        if gap_x + w//2 > bg_width:
            raise ValueError(f"Gap position causes out of bounds (right edge), x={gap_x}")
        
        if gap_y - h//2 < 0:
            raise ValueError(f"Gap position causes out of bounds (top edge), y={gap_y}")
        
        if gap_y + h//2 > bg_height:
            raise ValueError(f"Gap position causes out of bounds (bottom edge), y={gap_y}")
    
    def _extract_gap_image(self, 
                          full_image: np.ndarray,
                          gap_position: Tuple[int, int],
                          gap_mask: np.ndarray,
                          puzzle_shape: Union[str, Tuple[str, ...]]) -> GapImage:
        """提取gap区域为独立的BGRA图像"""
        x, y = gap_position
        h, w = gap_mask.shape[:2]
        
        # 计算左上角坐标
        x1 = x - w // 2
        y1 = y - h // 2
        
        # 验证坐标（根据CLAUDE.md约束，这里不应该小于0）
        assert x1 >= 0, f"x1 ({x1}) < 0, gap center x ({x}) too close to left edge"
        assert y1 >= 0, f"y1 ({y1}) < 0, gap center y ({y}) too close to top edge"
        
        x2 = x1 + w
        y2 = y1 + h
        
        # 创建BGRA图像
        gap_image = np.zeros((h, w, 4), dtype=np.uint8)
        
        # 复制RGB内容
        gap_image[:, :, :3] = full_image[y1:y2, x1:x2]
        
        # 设置alpha通道
        if gap_mask.shape[2] == 4:
            gap_image[:, :, 3] = gap_mask[:, :, 3]
        else:
            # 如果掩码没有alpha通道，创建一个（全不透明）
            gap_image[:, :, 3] = 255
        
        return GapImage(
            image=gap_image,
            position=gap_position,
            original_mask=gap_mask,
            metadata={'puzzle_shape': puzzle_shape},  # 传递形状信息
            background_size=(full_image.shape[1], full_image.shape[0])  # 传递背景尺寸 (width, height)
        )
    
    def _extract_slider_from_gap(self, gap_image: GapImage) -> np.ndarray:
        """从gap图像提取滑块（保持BGRA格式）"""
        # gap图像就是滑块，直接返回
        return gap_image.image.copy()
    
    def _extract_mask_from_gap_image(self, gap_image: GapImage) -> np.ndarray:
        """从gap图像中提取mask（包含混淆效果）"""
        # 创建一个与原始gap_mask相同形状的mask
        h, w = gap_image.image.shape[:2]
        
        # 如果原始mask有4个通道，创建BGRA格式的mask
        if gap_image.original_mask is not None and len(gap_image.original_mask.shape) > 2 and gap_image.original_mask.shape[2] == 4:
            mask = np.zeros((h, w, 4), dtype=np.uint8)
            # 复制alpha通道作为mask的所有通道
            alpha = gap_image.image[:, :, 3]
            mask[:, :, :3] = np.stack([alpha, alpha, alpha], axis=2)
            mask[:, :, 3] = alpha
        else:
            # 创建单通道mask
            mask = np.zeros((h, w, 4), dtype=np.uint8)
            alpha = gap_image.image[:, :, 3]
            mask[:, :, :3] = 255  # 白色背景
            mask[:, :, 3] = alpha
        
        return mask
    
    def _create_gap_background(self,
                              img: np.ndarray,
                              gap_mask: np.ndarray,
                              gap_position: Tuple[int, int],
                              additional_gaps: List[Dict[str, Any]] = None,
                              has_highlight: bool = False,
                              highlight_params: Dict[str, Any] = None,
                              rotated_piece_for_background: Optional['GapImage'] = None) -> np.ndarray:
        """创建带缺口的背景"""
        background = img.copy()
        
        # 1. 创建主缺口
        x, y = gap_position
        
        # 动态获取实际的掩码尺寸（可能是旋转后的非正方形）
        if rotated_piece_for_background is not None:
            actual_h, actual_w = rotated_piece_for_background.image.shape[:2]
            gap_mask_to_use = rotated_piece_for_background.original_mask
        else:
            actual_h, actual_w = gap_mask.shape[:2]
            gap_mask_to_use = gap_mask
        
        # 计算缺口位置（使用实际尺寸）
        x1 = x - actual_w // 2
        y1 = y - actual_h // 2
        x2 = x1 + actual_w
        y2 = y1 + actual_h
        
        # 边界检查断言 - 主缺口绝不能超出背景边界
        assert x1 >= 0, f"主缺口x1坐标不能为负: x1={x1}, x={x}, actual_w={actual_w}"
        assert y1 >= 0, f"主缺口y1坐标不能为负: y1={y1}, y={y}, actual_h={actual_h}"
        assert x2 <= img.shape[1], f"主缺口超出右边界: x2={x2}, img_w={img.shape[1]}, x={x}, actual_w={actual_w}"
        assert y2 <= img.shape[0], f"主缺口超出下边界: y2={y2}, img_h={img.shape[0]}, y={y}, actual_h={actual_h}"
        
        # 提取alpha通道（使用实际尺寸）
        # 优先使用rotated_piece_for_background的alpha通道（包含hollow效果）
        if rotated_piece_for_background is not None and hasattr(rotated_piece_for_background, 'image'):
            # 使用带hollow效果的piece的alpha通道
            if rotated_piece_for_background.image.shape[2] == 4:
                alpha_channel = rotated_piece_for_background.image[:actual_h, :actual_w, 3]
            else:
                alpha_channel = np.ones((actual_h, actual_w), dtype=np.uint8) * 255
        elif gap_mask_to_use.shape[2] == 4:
            alpha_channel = gap_mask_to_use[:actual_h, :actual_w, 3]
        else:
            alpha_channel = np.ones((actual_h, actual_w), dtype=np.uint8) * 255
        
        # 应用阴影或高光效果（使用实际尺寸）
        if has_highlight and highlight_params:
            # 应用高光效果（不使用外边缘效果）
            background = apply_gap_highlighting(
                background, x1, y1, 
                alpha_channel, actual_h, actual_w,
                base_lightness=highlight_params.get('base_lightness', 30),
                edge_lightness=highlight_params.get('edge_lightness', 45),
                directional_lightness=highlight_params.get('directional_lightness', 20),
                outer_edge_lightness=0  # 不使用外边缘效果
            )
        else:
            # 应用阴影效果
            background = apply_gap_lighting(
                background, x1, y1, 
                alpha_channel, actual_h, actual_w,
                base_darkness=40,
                edge_darkness=50,
                directional_darkness=20
            )
        
        # 2. 创建额外的混淆缺口
        if additional_gaps:
            for gap_info in additional_gaps:
                cx, cy = gap_info['position']
                mask = gap_info['mask']
                w, h = gap_info['size']
                
                # 计算位置
                x1 = cx - w // 2
                y1 = cy - h // 2
                x2 = x1 + w
                y2 = y1 + h
                
                # 边界检查断言 - 混淆缺口绝不能超出背景边界
                bg_h, bg_w = background.shape[:2]
                assert x1 >= 0, f"混淆缺口x1坐标不能为负: x1={x1}, cx={cx}, w={w}"
                assert y1 >= 0, f"混淆缺口y1坐标不能为负: y1={y1}, cy={cy}, h={h}"
                assert x2 <= bg_w, f"混淆缺口超出右边界: x2={x2}, bg_w={bg_w}, cx={cx}, w={w}"
                assert y2 <= bg_h, f"混淆缺口超出下边界: y2={y2}, bg_h={bg_h}, cy={cy}, h={h}"
                
                # 使用完整的alpha通道（已包含hollow效果）
                alpha_channel = mask[:h, :w].astype(np.uint8)
                
                # 应用阴影或高光效果（与主缺口相同）
                if has_highlight and highlight_params:
                    # 应用高光效果（不使用外边缘效果）
                    background = apply_gap_highlighting(
                        background, x1, y1, 
                        alpha_channel, h, w,
                        base_lightness=highlight_params.get('base_lightness', 30),
                        edge_lightness=highlight_params.get('edge_lightness', 45),
                        directional_lightness=highlight_params.get('directional_lightness', 20),
                        outer_edge_lightness=0  # 不使用外边缘效果
                    )
                else:
                    # 应用阴影效果
                    background = apply_gap_lighting(
                        background, x1, y1, 
                        alpha_channel, h, w,
                        base_darkness=40,
                        edge_darkness=50,
                        directional_darkness=20
                    )
        
        return background