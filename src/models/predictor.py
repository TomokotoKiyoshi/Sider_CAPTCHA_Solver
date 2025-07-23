"""
滑块验证码预测接口
用于调用训练好的模型进行推理
"""

from pathlib import Path
import json

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 导入模型
from .captcha_solver import CaptchaSolver


class CaptchaPredictor:
    """验证码预测器"""
    
    def __init__(self, model_path='best', device='auto', hm_threshold=0.0):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重文件路径，'best'使用内置最佳模型
            device: 运行设备 ('auto', 'cuda', 'cpu')
            hm_threshold: 热力图阈值，用于过滤低置信度检测
        """
        # 自动选择设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.hm_threshold = hm_threshold
        print(f"Using device: {self.device}")
        
        # 处理模型路径
        if model_path == 'best':
            # 使用内置的最佳模型
            package_dir = Path(__file__).parent.parent
            model_path = package_dir / 'checkpoints' / 'best_model.pth'
            if not model_path.exists():
                raise FileNotFoundError(f"Built-in best model not found at {model_path}")
        
        # 加载模型
        self.model = CaptchaSolver()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
    
    def preprocess_image(self, image_path):
        """预处理图像"""
        # 读取图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image from: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # 如果输入已经是numpy数组
            image = image_path
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3 and isinstance(image_path, str):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 确保图像大小为320x160
        if image.shape[:2] != (160, 320):
            image = cv2.resize(image, (320, 160))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 标准化（ImageNet标准）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # 转换为张量
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image
    
    def decode_predictions(self, outputs):
        """解码模型预测，提取中心点坐标"""
        # 从CaptchaSolver的输出格式中提取（heatmap已经包含Sigmoid）
        gap_heatmap = outputs['gap_heatmap'].squeeze(0).squeeze(0)  # [H, W]
        gap_offset = outputs['gap_offset'].squeeze(0)  # [2, H, W]
        
        piece_heatmap = outputs['piece_heatmap'].squeeze(0).squeeze(0)  # [H, W]
        piece_offset = outputs['piece_offset'].squeeze(0)  # [2, H, W]
        
        # 背景缺口中心
        hm_bg = gap_heatmap.cpu().numpy()
        max_val_bg = hm_bg.max()
        
        if max_val_bg > self.hm_threshold:  # 使用实例属性阈值
            y, x = np.unravel_index(hm_bg.argmax(), hm_bg.shape)
            offset_x = gap_offset[0, y, x].cpu().numpy()
            offset_y = gap_offset[1, y, x].cpu().numpy()
            bg_center = np.array([x + offset_x, y + offset_y]) * 4  # 恢复到原始尺寸
        else:
            bg_center = None
        
        # 滑块中心
        hm_slider = piece_heatmap.cpu().numpy()
        max_val_slider = hm_slider.max()
        
        if max_val_slider > self.hm_threshold:  # 使用实例属性阈值
            y, x = np.unravel_index(hm_slider.argmax(), hm_slider.shape)
            offset_x = piece_offset[0, y, x].cpu().numpy()
            offset_y = piece_offset[1, y, x].cpu().numpy()
            slider_center = np.array([x + offset_x, y + offset_y]) * 4
        else:
            slider_center = None
        
        return {
            'bg_center': bg_center,
            'slider_center': slider_center,
            'bg_confidence': float(max_val_bg),
            'slider_confidence': float(max_val_slider)
        }
    
    def predict(self, image_path):
        """
        预测图像中的缺口和滑块位置
        
        Args:
            image_path: 图像路径或numpy数组
        
        Returns:
            dict: 包含预测结果的字典
                - gap_x: 缺口中心x坐标
                - gap_y: 缺口中心y坐标
                - slider_x: 滑块中心x坐标
                - slider_y: 滑块中心y坐标
                - gap_confidence: 缺口检测置信度
                - slider_confidence: 滑块检测置信度
        """
        # 预处理图像
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # 解码预测
        predictions = self.decode_predictions(outputs)
        
        # 提取x坐标（用于滑动距离计算）
        result = {
            'gap_x': predictions['bg_center'][0] if predictions['bg_center'] is not None else None,
            'gap_y': predictions['bg_center'][1] if predictions['bg_center'] is not None else None,
            'slider_x': predictions['slider_center'][0] if predictions['slider_center'] is not None else None,
            'slider_y': predictions['slider_center'][1] if predictions['slider_center'] is not None else None,
            'gap_confidence': predictions['bg_confidence'],
            'slider_confidence': predictions['slider_confidence']
        }
        
        return result
    
    def visualize_prediction(self, image_path, save_path=None, show=False):
        """
        可视化预测结果
        
        Args:
            image_path: 图像路径或numpy数组
            save_path: 保存路径（可选）
            show: 是否显示图像
        """
        # 预测
        result = self.predict(image_path)
        
        # 读取原图
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path.copy()
            if image.shape[2] == 3 and image.dtype == np.uint8:
                # 已经是RGB格式
                pass
            else:
                # 需要转换
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # 创建图像
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(image)
        
        # 绘制检测结果
        if result['gap_x'] is not None:
            # 绘制缺口中心（红色）
            circle = patches.Circle((result['gap_x'], result['gap_y']), 5, 
                                  linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(circle)
            ax.text(result['gap_x'], result['gap_y'] - 10, 
                   f"Gap ({result['gap_x']:.1f}, {result['gap_y']:.1f})\nConf: {result['gap_confidence']:.3f}",
                   color='red', fontsize=10, ha='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        if result['slider_x'] is not None:
            # 绘制滑块中心（蓝色）
            circle = patches.Circle((result['slider_x'], result['slider_y']), 5,
                                  linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(circle)
            ax.text(result['slider_x'], result['slider_y'] + 20,
                   f"Slider ({result['slider_x']:.1f}, {result['slider_y']:.1f})\nConf: {result['slider_confidence']:.3f}",
                   color='blue', fontsize=10, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 显示滑动距离
        if result['gap_x'] is not None and result['slider_x'] is not None:
            distance = result['gap_x'] - result['slider_x']
            ax.plot([result['slider_x'], result['gap_x']], 
                   [result['slider_y'], result['gap_y']], 
                   'g--', linewidth=2)
            ax.text((result['slider_x'] + result['gap_x']) / 2,
                   (result['slider_y'] + result['gap_y']) / 2 - 5,
                   f"Distance: {distance:.1f}px",
                   color='green', fontsize=12, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        ax.set_title('CAPTCHA Detection Result', fontsize=14)
        ax.axis('off')
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_heatmaps(self, image_path, save_path=None, show=False):
        """
        可视化热力图
        
        Args:
            image_path: 图像路径或numpy数组
            save_path: 保存路径（可选）
            show: 是否显示图像
        """
        # 预处理图像
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # 获取热力图（已经包含Sigmoid）
        gap_heatmap = outputs['gap_heatmap'].squeeze().cpu().numpy()
        piece_heatmap = outputs['piece_heatmap'].squeeze().cpu().numpy()
        
        # 读取原图
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path.copy()
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3 and image.dtype == np.uint8 and isinstance(image_path, np.ndarray):
                # 如果是numpy数组且是BGR格式，转换为RGB
                if image_path.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整热力图大小
        gap_heatmap_resized = cv2.resize(gap_heatmap, (320, 160))
        piece_heatmap_resized = cv2.resize(piece_heatmap, (320, 160))
        
        # 创建图像
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 原图
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # 缺口热力图
        im1 = axes[0, 1].imshow(gap_heatmap_resized, cmap='hot', alpha=0.8)
        axes[0, 1].imshow(image, alpha=0.3)
        axes[0, 1].set_title('Gap Heatmap', fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # 滑块热力图
        im2 = axes[1, 0].imshow(piece_heatmap_resized, cmap='hot', alpha=0.8)
        axes[1, 0].imshow(image, alpha=0.3)
        axes[1, 0].set_title('Piece Heatmap', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
        
        # 组合热力图
        combined_heatmap = np.maximum(gap_heatmap_resized, piece_heatmap_resized)
        im3 = axes[1, 1].imshow(combined_heatmap, cmap='hot', alpha=0.8)
        axes[1, 1].imshow(image, alpha=0.3)
        axes[1, 1].set_title('Combined Heatmap', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
        
        plt.suptitle('CAPTCHA Detection Heatmaps', fontsize=14)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Heatmap visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()