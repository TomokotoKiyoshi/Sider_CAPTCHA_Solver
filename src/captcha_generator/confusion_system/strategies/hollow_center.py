# -*- coding: utf-8 -*-
"""
中心镂空策略 - 在gap中间开一个与外形相同但尺寸较小的透明孔
改进版：使用统一的多边形镂空算法
"""
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
from ..base import ConfusionStrategy, GapImage


class HollowCenterConfusion(ConfusionStrategy):
    """中心镂空 - 在gap中间创建一个形状相同但尺寸较小的透明孔"""
    
    @property
    def name(self) -> str:
        return "hollow_center"
    
    @property
    def description(self) -> str:
        return "在gap中心创建与外形相同但尺寸较小的透明孔"
    
    def validate_config(self):
        """验证并设置默认配置"""
        # 镂空比例（相对于原始大小）
        self.hollow_ratio = self.config.get('hollow_ratio')
        assert self.hollow_ratio is not None, "hollow_ratio must be provided in config"
        
        # 验证hollow_ratio的合理范围
        if not 0.1 <= self.hollow_ratio <= 0.9:
            self.hollow_ratio = max(0.1, min(0.9, self.hollow_ratio))
    
    def apply_to_gap(self, gap_image: GapImage) -> GapImage:
        """
        在gap中心创建镂空效果（统一的多边形处理方法）
        
        Args:
            gap_image: 独立的gap图像（BGRA格式）
            
        Returns:
            GapImage: 应用镂空效果后的gap图像
        """
        # 复制图像避免修改原始数据
        result_image = gap_image.image.copy()
        h, w = result_image.shape[:2]
        
        # 获取形状信息
        shape_info = self._get_shape_info(gap_image)
        
        # 获取alpha通道
        alpha = result_image[:, :, 3]
        
        # 创建镂空后的alpha通道
        new_alpha = self._create_hollow_alpha(alpha, shape_info)
        
        # 更新结果图像的alpha通道
        result_image[:, :, 3] = new_alpha
        
        # 返回新的GapImage对象
        return GapImage(
            image=result_image,
            position=gap_image.position,
            original_mask=gap_image.original_mask,
            metadata=gap_image.metadata,
            background_size=gap_image.background_size
        )
    
    def _get_shape_info(self, gap_image: GapImage) -> str:
        """
        获取形状信息
        
        Args:
            gap_image: gap图像对象
            
        Returns:
            形状类型字符串
        """
        # 优先从metadata获取
        if hasattr(gap_image, 'metadata') and gap_image.metadata:
            shape_info = gap_image.metadata.get('puzzle_shape', None)
            if shape_info:
                return shape_info
        
        # 否则通过轮廓检测
        if gap_image.image.shape[2] == 4:
            return self._detect_shape_from_mask(gap_image.image[:, :, 3])
        
        return 'unknown'
    
    def _create_hollow_alpha(self, alpha: np.ndarray, shape_type: str) -> np.ndarray:
        """
        根据形状类型创建镂空的alpha通道
        
        Args:
            alpha: 原始alpha通道
            shape_type: 形状类型
            
        Returns:
            带镂空的新alpha通道
        """
        # 多边形形状的顶点数映射
        polygon_vertices = {
            'triangle': 3,
            'square': 4,
            'rectangle': 4,
            'pentagon': 5,
            'hexagon': 6
        }
        
        # 如果是多边形，使用多边形镂空方法
        if shape_type in polygon_vertices:
            return self._create_polygon_hollow(alpha, polygon_vertices[shape_type])
        
        # 对于圆形或其他形状，使用通用的缩放方法
        return self._create_generic_hollow(alpha)
    
    def _create_polygon_hollow(self, alpha: np.ndarray, expected_vertices: int) -> np.ndarray:
        """
        创建多边形镂空（通用方法，适用于三角形、五边形、六边形等）
        
        核心算法：
        1. 获取多边形顶点
        2. 计算质心
        3. 相对质心缩放顶点
        4. 在缩放后的多边形区域设置alpha=0
        
        Args:
            alpha: 原始alpha通道
            expected_vertices: 期望的顶点数
            
        Returns:
            带镂空的新alpha通道
        """
        # 二值化
        _, binary = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return alpha
        
        # 获取最大轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 尝试获取多边形顶点
        vertices = self._get_polygon_vertices(contour, expected_vertices)
        
        if vertices is None:
            # 如果无法获取正确数量的顶点，回退到通用方法
            return self._create_generic_hollow(alpha)
        
        # 使用cv2.moments计算质心
        M = cv2.moments(vertices)
        if M["m00"] == 0:
            return self._create_generic_hollow(alpha)
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 创建内部多边形顶点（相对于质心缩放）
        inner_vertices = (vertices - [cx, cy]) * self.hollow_ratio + [cx, cy]
        inner_vertices = inner_vertices.astype(np.int32)
        
        # 创建新的alpha通道
        new_alpha = alpha.copy()
        
        # 在内部多边形区域设置alpha=0（创建镂空）
        cv2.fillPoly(new_alpha, [inner_vertices], 0)
        
        return new_alpha
    
    def _create_generic_hollow(self, alpha: np.ndarray) -> np.ndarray:
        """
        创建通用镂空（适用于圆形或无法识别的形状）
        使用简单的缩放方法
        
        Args:
            alpha: 原始alpha通道
            
        Returns:
            带镂空的新alpha通道
        """
        h, w = alpha.shape
        
        # 计算缩小后的尺寸
        hollow_w = max(1, int(w * self.hollow_ratio))
        hollow_h = max(1, int(h * self.hollow_ratio))
        
        # 缩小原始掩码
        hollow_mask = cv2.resize(alpha, (hollow_w, hollow_h), interpolation=cv2.INTER_LINEAR)
        
        # 二值化
        _, hollow_mask = cv2.threshold(hollow_mask, 128, 255, cv2.THRESH_BINARY)
        
        # 创建新的alpha通道
        new_alpha = alpha.copy()
        
        # 计算放置位置（居中）
        x_offset = (w - hollow_w) // 2
        y_offset = (h - hollow_h) // 2
        
        # 确保不越界
        x_end = min(x_offset + hollow_w, w)
        y_end = min(y_offset + hollow_h, h)
        
        # 在镂空区域设置alpha=0
        if x_end > x_offset and y_end > y_offset:
            hollow_region = hollow_mask[:y_end-y_offset, :x_end-x_offset] > 128
            new_alpha[y_offset:y_end, x_offset:x_end][hollow_region] = 0
        
        return new_alpha
    
    def _get_polygon_vertices(self, contour: np.ndarray, expected_vertices: int) -> Optional[np.ndarray]:
        """
        尝试获取多边形的顶点
        
        Args:
            contour: 轮廓点
            expected_vertices: 期望的顶点数
            
        Returns:
            多边形顶点数组，如果无法获取则返回None
        """
        perimeter = cv2.arcLength(contour, True)
        
        # 尝试不同的epsilon值
        epsilon_factors = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12]
        
        for epsilon_factor in epsilon_factors:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == expected_vertices:
                return approx.reshape(-1, 2)
        
        # 如果无法精确获取，尝试找最接近的
        best_approx = None
        min_diff = float('inf')
        
        for epsilon_factor in epsilon_factors:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            diff = abs(len(approx) - expected_vertices)
            
            if diff < min_diff and diff <= 1:  # 允许1个顶点的误差
                min_diff = diff
                best_approx = approx
        
        if best_approx is not None and len(best_approx) >= 3:
            return best_approx.reshape(-1, 2)
        
        return None
    
    def _detect_shape_from_mask(self, mask: np.ndarray) -> str:
        """
        通过分析掩码轮廓来检测形状类型
        
        Args:
            mask: Alpha通道掩码
            
        Returns:
            形状类型字符串
        """
        # 二值化
        _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'unknown'
        
        # 获取最大轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        
        # 尝试不同的epsilon值进行多边形近似
        best_vertices = 0
        for epsilon_factor in [0.02, 0.04, 0.06, 0.08]:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            num_vertices = len(approx)
            
            # 选择最清晰的形状识别结果
            if num_vertices == 3:
                return 'triangle'
            elif num_vertices == 4:
                # 检查是否为正方形
                vertices = approx.reshape(-1, 2)
                # 计算边长
                edges = []
                for i in range(4):
                    p1 = vertices[i]
                    p2 = vertices[(i + 1) % 4]
                    edge_length = np.linalg.norm(p2 - p1)
                    edges.append(edge_length)
                
                # 如果所有边长相近，则为正方形
                if max(edges) / min(edges) < 1.2:
                    return 'square'
                else:
                    return 'rectangle'
            elif num_vertices == 5:
                return 'pentagon'
            elif num_vertices == 6:
                return 'hexagon'
            
            best_vertices = max(best_vertices, num_vertices)
        
        # 如果顶点数大于6，可能是圆形
        if best_vertices > 6:
            return 'circle'
        
        return 'unknown'
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取策略的元数据"""
        metadata = super().get_metadata()
        metadata.update({
            "hollow_ratio": self.hollow_ratio,
            "algorithm": "polygon_scaling"  # 标记使用的算法
        })
        return metadata