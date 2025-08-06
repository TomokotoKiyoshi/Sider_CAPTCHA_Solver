# -*- coding: utf-8 -*-
"""
柏林噪声混淆策略 - 在滑块表面应用噪声纹理
"""
import numpy as np
import cv2
from typing import Dict, Any
from ..base import ConfusionStrategy, GapImage


class PerlinNoiseConfusion(ConfusionStrategy):
    """柏林噪声混淆 - 在滑块表面应用40-80%强度的噪声纹理"""
    
    @property
    def name(self) -> str:
        return "perlin_noise"
    
    @property
    def description(self) -> str:
        return "在滑块表面应用柏林噪声纹理，创造粗糙质感"
    
    def validate_config(self):
        """验证并设置默认配置"""
        # 噪声强度（0.0-1.0）
        self.noise_strength = self.config.get('noise_strength')
        
        # 噪声缩放（控制噪声的粗细）
        self.noise_scale = self.config.get('noise_scale', 0.1)
        
        # 验证参数
        assert self.noise_strength is not None, "noise_strength must be provided in config"
        assert isinstance(self.noise_strength, (int, float)), "noise_strength must be a number"
        assert 0.4 <= self.noise_strength <= 0.8, f"noise_strength must be between 0.4 and 0.8 (40-80%), got: {self.noise_strength}"
        assert 0.01 <= self.noise_scale <= 1.0, f"noise_scale must be between 0.01 and 1.0, got: {self.noise_scale}"
    
    def apply_to_gap(self, gap_image: GapImage) -> GapImage:
        """
        在独立的gap图像（滑块）上应用柏林噪声
        
        Args:
            gap_image: 独立的gap图像（BGRA格式）
            
        Returns:
            GapImage: 应用噪声后的gap图像
        """
        # 创建副本
        result_image = gap_image.image.copy()
        h, w = result_image.shape[:2]
        
        # 生成柏林噪声
        noise = self._generate_perlin_noise(h, w, self.noise_scale)
        
        # 将噪声归一化到 [-1, 1] 范围
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1
        
        # 提取RGB和Alpha通道
        rgb = result_image[:, :, :3].astype(np.float32)
        alpha = result_image[:, :, 3]
        
        # 创建掩码（只在不透明区域应用噪声）
        mask = alpha > 128
        
        # 应用噪声到RGB通道
        for c in range(3):
            # 计算噪声偏移量
            noise_offset = noise * self.noise_strength * 50  # 最大偏移50个亮度级别
            
            # 只在掩码区域应用噪声
            rgb[:, :, c][mask] += noise_offset[mask]
        
        # 确保值在0-255范围内
        rgb = np.clip(rgb, 0, 255)
        
        # 更新结果图像
        result_image[:, :, :3] = rgb.astype(np.uint8)
        
        # 返回新的GapImage
        return GapImage(
            image=result_image,
            position=gap_image.position,
            original_mask=gap_image.original_mask
        )
    
    def _generate_perlin_noise(self, height: int, width: int, scale: float) -> np.ndarray:
        """
        生成柏林噪声
        
        Args:
            height: 图像高度
            width: 图像宽度
            scale: 噪声缩放因子（越小噪声越细腻）
            
        Returns:
            噪声数组
        """
        # 创建随机梯度网格
        grid_height = int(height * scale) + 2
        grid_width = int(width * scale) + 2
        
        # 生成随机角度
        angles = np.random.rand(grid_height, grid_width) * 2 * np.pi
        gradients = np.stack([np.cos(angles), np.sin(angles)], axis=2)
        
        # 创建坐标网格
        y, x = np.mgrid[0:height, 0:width]
        y = y * scale
        x = x * scale
        
        # 初始化噪声数组
        noise = np.zeros((height, width))
        
        # 对每个像素计算噪声值
        for i in range(height):
            for j in range(width):
                # 获取网格坐标
                x0 = int(x[i, j])
                x1 = x0 + 1
                y0 = int(y[i, j])
                y1 = y0 + 1
                
                # 确保不超出边界
                x1 = min(x1, grid_width - 1)
                y1 = min(y1, grid_height - 1)
                
                # 计算插值权重
                sx = x[i, j] - x0
                sy = y[i, j] - y0
                
                # 计算四个角的梯度向量
                n00 = self._gradient_dot(gradients[y0, x0], x[i, j] - x0, y[i, j] - y0)
                n10 = self._gradient_dot(gradients[y0, x1], x[i, j] - x1, y[i, j] - y0)
                n01 = self._gradient_dot(gradients[y1, x0], x[i, j] - x0, y[i, j] - y1)
                n11 = self._gradient_dot(gradients[y1, x1], x[i, j] - x1, y[i, j] - y1)
                
                # 使用缓和曲线插值
                sx = self._smoothstep(sx)
                sy = self._smoothstep(sy)
                
                # 双线性插值
                nx0 = n00 * (1 - sx) + n10 * sx
                nx1 = n01 * (1 - sx) + n11 * sx
                noise[i, j] = nx0 * (1 - sy) + nx1 * sy
        
        return noise
    
    def _gradient_dot(self, gradient: np.ndarray, dx: float, dy: float) -> float:
        """计算梯度向量和距离向量的点积"""
        return gradient[0] * dx + gradient[1] * dy
    
    def _smoothstep(self, t: float) -> float:
        """平滑插值函数（3t² - 2t³）"""
        return t * t * (3 - 2 * t)
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取策略的元数据"""
        metadata = super().get_metadata()
        metadata.update({
            "noise_strength": self.noise_strength,
            "noise_scale": self.noise_scale
        })
        return metadata