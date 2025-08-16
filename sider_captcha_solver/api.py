"""简单的Python API接口 - 无需HTTP服务"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional
import cv2

from .predictor import CaptchaPredictor

# 全局预测器实例（懒加载）
_global_predictor = None


def _get_global_predictor():
    """获取全局预测器实例（单例模式）"""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = CaptchaPredictor(model_path='best', device='auto')
    return _global_predictor


def solve(image_path: Union[str, Path, np.ndarray], 
         detailed: bool = False) -> Union[float, Dict, None]:
    """最简单的API：输入图片，返回滑动距离
    
    Args:
        image_path: 图片路径或numpy数组
        detailed: 是否返回详细信息
    
    Returns:
        - 默认: 滑动距离（float）
        - detailed=True: 包含详细信息的字典
        - 失败: None
    
    Example:
        >>> distance = solve("captcha.png")
        >>> print(f"滑动 {distance} 像素")
        120.5
        
        >>> result = solve("captcha.png", detailed=True)
        >>> print(result['distance'], result['confidence'])
        120.5 0.94
    """
    try:
        predictor = _get_global_predictor()
        result = predictor.predict(image_path)
        
        if not result.get('success'):
            return None
        
        if detailed:
            return {
                'distance': result['sliding_distance'],
                'gap': (result['gap_x'], result['gap_y']),
                'slider': (result['slider_x'], result['slider_y']),
                'gap_confidence': result['gap_confidence'],
                'slider_confidence': result['slider_confidence'],
                'confidence': result['confidence'],
                'time_ms': result['processing_time_ms']
            }
        else:
            return result['sliding_distance']
    
    except Exception as e:
        print(f"预测失败: {e}")
        return None


def solve_batch(images: List[Union[str, Path]], 
               detailed: bool = False) -> List[Union[float, Dict, None]]:
    """批量处理接口
    
    Args:
        images: 图片路径列表
        detailed: 是否返回详细信息
    
    Returns:
        预测结果列表
    
    Example:
        >>> distances = solve_batch(["img1.png", "img2.png"])
        >>> print(distances)
        [120.5, 95.3]
    """
    results = []
    for image in images:
        result = solve(image, detailed=detailed)
        results.append(result)
    return results


def visualize(image_path: Union[str, Path, np.ndarray],
             result: Optional[Dict] = None,
             save_path: Optional[str] = None,
             show: bool = True) -> Optional[np.ndarray]:
    """可视化预测结果
    
    Args:
        image_path: 输入图片
        result: 预测结果（可选，不提供则自动预测）
        save_path: 保存路径（可选）
        show: 是否显示（默认True）
    
    Returns:
        可视化后的图像数组（如果save_path=None且show=False）
    
    Example:
        >>> visualize("captcha.png")  # 显示可视化
        >>> visualize("captcha.png", save_path="result.png")  # 保存
        >>> img = visualize("captcha.png", show=False)  # 返回图像数组
    """
    try:
        # 如果没有提供result，自动预测
        if result is None:
            result = solve(image_path, detailed=True)
            if result is None:
                print("预测失败，无法可视化")
                return None
        
        # 读取图像
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"无法读取图像: {image_path}")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path.copy()
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 创建可视化副本
        vis_image = image.copy()
        
        # 提取坐标
        gap_x, gap_y = result['gap']
        slider_x, slider_y = result['slider']
        distance = result['distance']
        confidence = result['confidence']
        
        # 绘制缺口位置（红色矩形）
        cv2.rectangle(vis_image, 
                     (int(gap_x - 20), int(gap_y - 20)),
                     (int(gap_x + 20), int(gap_y + 20)),
                     (255, 0, 0), 2)
        cv2.putText(vis_image, "Gap", 
                   (int(gap_x - 20), int(gap_y - 25)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 绘制滑块位置（绿色矩形）
        cv2.rectangle(vis_image,
                     (int(slider_x - 20), int(slider_y - 20)),
                     (int(slider_x + 20), int(slider_y + 20)),
                     (0, 255, 0), 2)
        cv2.putText(vis_image, "Slider",
                   (int(slider_x - 20), int(slider_y - 25)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 添加文字信息（顶部）
        info_text = f"Distance: {distance:.1f}px | Confidence: {confidence:.2%}"
        cv2.rectangle(vis_image, (0, 0), (len(info_text) * 9, 30), (255, 255, 255), -1)
        cv2.putText(vis_image, info_text,
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 保存图像
        if save_path:
            # 转换回BGR格式保存
            save_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), save_image)
            print(f"可视化已保存: {save_path}")
        
        # 显示图像
        if show:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.imshow(vis_image)
            plt.title(f"滑动距离: {distance:.1f}px | 置信度: {confidence:.2%}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # 返回图像数组
        if not save_path and not show:
            return vis_image
        
        return vis_image if not show else None
        
    except Exception as e:
        print(f"可视化失败: {e}")
        return None


class CaptchaSolver:
    """面向对象的API，适合需要多次调用的场景
    
    Example:
        >>> solver = CaptchaSolver()
        >>> distance = solver.solve("captcha.png")
        >>> solver.visualize("captcha.png", save_path="result.png")
    """
    
    def __init__(self, model: str = 'best', device: str = 'auto'):
        """初始化求解器
        
        Args:
            model: 模型路径或预设名称
            device: 运行设备 ('auto'/'cuda'/'cpu')
        """
        self._predictor = None
        self._model_path = model
        self._device = device
        self._last_result = None
    
    def _ensure_predictor(self):
        """确保预测器已初始化（懒加载）"""
        if self._predictor is None:
            self._predictor = CaptchaPredictor(
                model_path=self._model_path,
                device=self._device
            )
    
    def solve(self, image: Union[str, Path, np.ndarray]) -> float:
        """求解单张图片，返回滑动距离
        
        Args:
            image: 图片路径或numpy数组
        
        Returns:
            滑动距离（像素）
        """
        self._ensure_predictor()
        result = self._predictor.predict(image)
        
        if result.get('success'):
            self._last_result = {
                'distance': result['sliding_distance'],
                'gap': (result['gap_x'], result['gap_y']),
                'slider': (result['slider_x'], result['slider_y']),
                'gap_confidence': result['gap_confidence'],
                'slider_confidence': result['slider_confidence'],
                'confidence': result['confidence'],
                'time_ms': result['processing_time_ms']
            }
            return result['sliding_distance']
        else:
            self._last_result = None
            return None
    
    def solve_detailed(self, image: Union[str, Path, np.ndarray]) -> Dict:
        """获取详细结果
        
        Args:
            image: 图片路径或numpy数组
        
        Returns:
            包含详细信息的字典
        """
        distance = self.solve(image)
        return self._last_result if distance is not None else None
    
    def visualize(self, image: Union[str, Path, np.ndarray],
                 save_path: Optional[str] = None,
                 show: bool = True) -> Optional[np.ndarray]:
        """可视化最近的预测结果
        
        Args:
            image: 图片路径或numpy数组
            save_path: 保存路径（可选）
            show: 是否显示
        
        Returns:
            可视化后的图像数组
        """
        # 如果没有最近的结果，先预测
        if self._last_result is None:
            self.solve(image)
        
        if self._last_result is None:
            print("预测失败，无法可视化")
            return None
        
        return visualize(image, result=self._last_result, 
                        save_path=save_path, show=show)
    
    def batch_solve(self, images: List[Union[str, Path]], 
                   detailed: bool = False) -> List[Union[float, Dict, None]]:
        """批量求解
        
        Args:
            images: 图片路径列表
            detailed: 是否返回详细信息
        
        Returns:
            结果列表
        """
        results = []
        for image in images:
            if detailed:
                result = self.solve_detailed(image)
            else:
                result = self.solve(image)
            results.append(result)
        return results


# 便捷导出
__all__ = [
    'solve',
    'solve_batch',
    'visualize',
    'CaptchaSolver'
]