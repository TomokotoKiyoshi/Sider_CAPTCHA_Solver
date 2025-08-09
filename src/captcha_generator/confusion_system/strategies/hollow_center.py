# -*- coding: utf-8 -*-
"""
中心镂空策略 - 在gap中间开一个与外形相同但尺寸较小的透明孔
"""
import numpy as np
import cv2
from typing import Dict, Any, Tuple
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
        self.hollow_ratio = self.config.get('hollow_ratio', 0.4)
        
        # 只验证参数存在性，不限制范围（由配置文件控制）
    
    def apply_to_gap(self, gap_image: GapImage) -> GapImage:
        """
        在gap中心创建镂空效果
        
        Args:
            gap_image: 独立的gap图像（BGRA格式）
            
        Returns:
            GapImage: 应用镂空效果后的gap图像
        """
        # 复制图像避免修改原始数据
        result_image = gap_image.image.copy()
        h, w = result_image.shape[:2]
        
        # 获取原始掩码（从gap_image或alpha通道）
        if gap_image.original_mask is not None and len(gap_image.original_mask.shape) > 2:
            # 如果有原始掩码，使用它
            if gap_image.original_mask.shape[2] == 4:
                original_alpha = gap_image.original_mask[:, :, 3]
            else:
                # 假设是BGR格式，转换为灰度作为掩码
                original_alpha = cv2.cvtColor(gap_image.original_mask[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            # 使用当前图像的alpha通道
            original_alpha = result_image[:, :, 3]
        
        # 创建缩小版的掩码（用于镂空）
        # 计算缩小后的尺寸
        hollow_w = int(w * self.hollow_ratio)
        hollow_h = int(h * self.hollow_ratio)
        
        # 缩小原始掩码
        hollow_mask = cv2.resize(original_alpha, (hollow_w, hollow_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建一个与原图相同大小的掩码，用于放置缩小的镂空
        center_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 根据形状类型计算质心位置
        x_offset, y_offset = self._calculate_centroid_offset(gap_image, w, h, hollow_w, hollow_h)
        
        # 将缩小的掩码放置在质心位置
        center_mask[y_offset:y_offset+hollow_h, x_offset:x_offset+hollow_w] = hollow_mask
        
        # 从原始alpha中减去中心镂空区域
        # 确保镂空区域完全透明
        new_alpha = result_image[:, :, 3].copy()
        hollow_region = center_mask > 128  # 二值化
        new_alpha[hollow_region] = 0  # 镂空区域设为完全透明
        
        # 更新结果图像的alpha通道
        result_image[:, :, 3] = new_alpha
        
        # 不要修改RGB通道！保持原始颜色，仅通过alpha=0实现透明
        # 这样可以避免黑色边缘泄露问题
        
        # 返回新的GapImage对象
        return GapImage(
            image=result_image,
            position=gap_image.position,
            original_mask=gap_image.original_mask
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取策略的元数据"""
        metadata = super().get_metadata()
        metadata.update({
            "hollow_ratio": self.hollow_ratio
        })
        return metadata
    
    def _calculate_centroid_offset(self, gap_image: GapImage, w: int, h: int, 
                                   hollow_w: int, hollow_h: int) -> tuple[int, int]:
        """
        根据形状类型计算质心偏移
        
        Args:
            gap_image: gap图像对象
            w, h: 原始尺寸
            hollow_w, hollow_h: 镂空尺寸
            
        Returns:
            (x_offset, y_offset): 镂空的偏移位置
        """
        # 尝试从metadata中获取形状信息
        shape_info = None
        if hasattr(gap_image, 'metadata') and gap_image.metadata:
            shape_info = gap_image.metadata.get('puzzle_shape', None)
        
        # 如果没有metadata，尝试分析alpha通道来判断形状
        if shape_info is None and gap_image.image.shape[2] == 4:
            shape_info = self._detect_shape_from_mask(gap_image.image[:, :, 3])
        
        # 根据形状类型计算质心
        if shape_info == 'triangle':
            # 对于三角形，使用重心计算（三个顶点坐标的平均值）
            alpha = gap_image.image[:, :, 3]
            centroid_x, centroid_y = self._calculate_triangle_centroid(alpha)
            
            # 计算镂空的偏移位置，使镂空中心与三角形重心对齐
            x_offset = int(centroid_x - hollow_w // 2)
            y_offset = int(centroid_y - hollow_h // 2)
            
            # 确保不超出边界
            x_offset = max(0, min(x_offset, w - hollow_w))
            y_offset = max(0, min(y_offset, h - hollow_h))
        
        elif shape_info == 'pentagon':
            # 五边形也使用精确的重心计算
            alpha = gap_image.image[:, :, 3]
            centroid_x, centroid_y = self._calculate_polygon_centroid(alpha, expected_vertices=5)
            
            x_offset = int(centroid_x - hollow_w // 2)
            y_offset = int(centroid_y - hollow_h // 2)
            
            x_offset = max(0, min(x_offset, w - hollow_w))
            y_offset = max(0, min(y_offset, h - hollow_h))
        
        else:
            # 其他形状（圆形、正方形、六边形等）使用边界框中心
            x_offset = (w - hollow_w) // 2
            y_offset = (h - hollow_h) // 2
        
        return x_offset, y_offset
    
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
        
        # 近似多边形
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 根据顶点数判断形状
        num_vertices = len(approx)
        
        if num_vertices == 3:
            return 'triangle'
        elif num_vertices == 4:
            # 可能是正方形或矩形
            return 'square'
        elif num_vertices == 5:
            return 'pentagon'
        elif num_vertices == 6:
            return 'hexagon'
        elif num_vertices > 6:
            # 圆形或其他
            return 'circle'
        else:
            return 'unknown'
    
    def _calculate_triangle_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """
        计算三角形的重心（三个顶点坐标的平均值）
        
        Args:
            mask: Alpha通道掩码
            
        Returns:
            (centroid_x, centroid_y): 重心坐标
        """
        # 二值化
        _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 如果找不到轮廓，返回中心点
            return mask.shape[1] / 2, mask.shape[0] / 2
        
        # 获取最大轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 近似三角形
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果不是三个顶点，尝试使用更大的epsilon
        if len(approx) != 3:
            epsilon = 0.08 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果还是不是三个顶点，使用轮廓的质心
        if len(approx) != 3:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return cx, cy
            else:
                return mask.shape[1] / 2, mask.shape[0] / 2
        
        # 计算三个顶点的重心
        vertices = approx.reshape(-1, 2)
        centroid_x = np.mean(vertices[:, 0])
        centroid_y = np.mean(vertices[:, 1])
        
        return centroid_x, centroid_y
    
    def _calculate_polygon_centroid(self, mask: np.ndarray, expected_vertices: int = None) -> Tuple[float, float]:
        """
        计算多边形的重心
        
        Args:
            mask: Alpha通道掩码
            expected_vertices: 期望的顶点数（用于多边形近似）
            
        Returns:
            (centroid_x, centroid_y): 重心坐标
        """
        # 二值化
        _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask.shape[1] / 2, mask.shape[0] / 2
        
        # 获取最大轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 如果有期望的顶点数，尝试近似
        if expected_vertices:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 调整epsilon直到得到正确的顶点数
            for factor in [0.04, 0.06, 0.08, 0.1]:
                if len(approx) == expected_vertices:
                    break
                epsilon = factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果成功得到期望的顶点数，使用顶点计算重心
            if len(approx) == expected_vertices:
                vertices = approx.reshape(-1, 2)
                centroid_x = np.mean(vertices[:, 0])
                centroid_y = np.mean(vertices[:, 1])
                return centroid_x, centroid_y
        
        # 如果无法得到期望的顶点数，使用轮廓矩计算质心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return cx, cy
        else:
            return mask.shape[1] / 2, mask.shape[0] / 2