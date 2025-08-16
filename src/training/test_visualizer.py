#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试集预测可视化模块
用于每个epoch保存测试集的预测结果图片
"""
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm


class TestVisualizer:
    """测试集预测可视化器"""
    
    def __init__(self, output_dir: str = "test_output/results_visualization"):
        """
        初始化测试集可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger('TestVisualizer')
        
    def visualize_epoch_predictions(self, 
                                   model, 
                                   dataloader, 
                                   epoch: int,
                                   device: torch.device,
                                   max_samples: int = 1000,
                                   save_format: str = 'jpg'):
        """
        可视化并保存一个epoch的测试集预测结果
        
        Args:
            model: 模型
            dataloader: 测试数据加载器
            epoch: 当前epoch
            device: 设备
            max_samples: 最大保存样本数
            save_format: 图片格式
        """
        # 创建epoch输出目录
        epoch_dir = self.output_dir / f"epoch{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"开始保存Epoch {epoch} 的测试集预测结果到: {epoch_dir}")
        
        model.eval()
        saved_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} 预测可视化")):
                if saved_count >= max_samples:
                    break
                    
                # 数据传输到设备
                batch = self._batch_to_device(batch, device)
                
                # 前向传播
                outputs = model(batch['image'])
                
                # 解码预测（传入输入图像以使用padding mask）
                predictions = model.decode_predictions(outputs, input_images=batch['image'])
                
                # 可视化批次中的每个样本
                batch_size = batch['image'].size(0)
                for i in range(batch_size):
                    if saved_count >= max_samples:
                        break
                    
                    # 创建可视化图像
                    vis_img = self._create_visualization(
                        batch['image'][i],
                        predictions,
                        batch,
                        i,
                        outputs
                    )
                    
                    # 保存图像
                    filename = batch.get('filename', [f"sample_{batch_idx}_{i}"])[i] if 'filename' in batch else f"sample_{batch_idx:04d}_{i:02d}"
                    # 清理文件名，移除扩展名
                    if isinstance(filename, str):
                        filename = Path(filename).stem
                    
                    save_path = epoch_dir / f"{filename}_pred.{save_format}"
                    cv2.imwrite(str(save_path), vis_img)
                    
                    saved_count += 1
        
        self.logger.info(f"Epoch {epoch}: 保存了 {saved_count} 张预测结果图片")
        return saved_count
    
    def _batch_to_device(self, batch: Dict, device: torch.device) -> Dict:
        """将批次数据传输到设备"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _create_visualization(self,
                            image: torch.Tensor,
                            predictions: Dict,
                            batch: Dict,
                            idx: int,
                            outputs: Optional[Dict] = None) -> np.ndarray:
        """
        创建单个样本的可视化图像
        
        Args:
            image: 输入图像 [4, H, W]
            predictions: 预测结果
            batch: 批次数据
            idx: 样本索引
            outputs: 模型原始输出（包含热力图）
            
        Returns:
            可视化图像
        """
        # 获取RGB图像（去除padding通道）
        img = image[:3].cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        orig_h, orig_w = img.shape[:2]
        
        # 创建可视化画布（原图的2倍宽度）
        canvas_w = orig_w * 2
        canvas_h = orig_h
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        # 左侧：原图+预测标记
        left_img = img.copy()
        
        # 绘制预测坐标
        pred_gap = predictions['gap_coords'][idx].cpu().numpy()
        pred_slider = predictions['slider_coords'][idx].cpu().numpy()
        
        # 绘制缺口预测（红色十字）
        self._draw_crosshair(left_img, pred_gap, (0, 0, 255), "Gap_Pred")
        
        # 绘制滑块预测（蓝色方框）
        self._draw_box_marker(left_img, pred_slider, (255, 0, 0), "Slider_Pred")
        
        # 如果有真实标注，也绘制
        if 'gap_coords' in batch and 'slider_coords' in batch:
            gt_gap = batch['gap_coords'][idx].cpu().numpy()
            gt_slider = batch['slider_coords'][idx].cpu().numpy()
            
            # 绘制真实缺口（绿色十字）
            self._draw_crosshair(left_img, gt_gap, (0, 255, 0), "Gap_GT")
            
            # 绘制真实滑块（青色方框）
            self._draw_box_marker(left_img, gt_slider, (255, 255, 0), "Slider_GT")
            
            # 计算误差
            gap_error = np.linalg.norm(pred_gap - gt_gap)
            slider_error = np.linalg.norm(pred_slider - gt_slider)
            avg_error = (gap_error + slider_error) / 2.0
            
            # 在图像顶部添加误差信息
            error_text = f"Gap Err: {gap_error:.1f}px | Slider Err: {slider_error:.1f}px | Avg: {avg_error:.1f}px"
            cv2.putText(left_img, error_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                       cv2.LINE_AA)
            # 添加背景
            cv2.rectangle(left_img, (5, 5), (orig_w-5, 35), (0, 0, 0), -1)
            cv2.putText(left_img, error_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                       cv2.LINE_AA)
        
        # 放置左侧图像
        canvas[:, :orig_w] = left_img
        
        # 右侧：热力图叠加
        if outputs is not None:
            heatmap_img = self._create_heatmap_overlay(img, outputs, idx)
            canvas[:, orig_w:] = heatmap_img
        else:
            # 如果没有热力图，复制原图
            canvas[:, orig_w:] = img
        
        # 添加分割线
        cv2.line(canvas, (orig_w, 0), (orig_w, canvas_h), (128, 128, 128), 2)
        
        return canvas
    
    def _create_heatmap_overlay(self, img: np.ndarray, outputs: Dict, idx: int) -> np.ndarray:
        """
        创建热力图叠加效果
        
        Args:
            img: 原始图像 (H, W, 3)
            outputs: 模型输出
            idx: 样本索引
            
        Returns:
            叠加后的图像
        """
        orig_h, orig_w = img.shape[:2]
        
        # 获取热力图
        if 'heatmap_gap' in outputs and 'heatmap_slider' in outputs:
            gap_heatmap = outputs['heatmap_gap'][idx, 0].cpu().numpy()
            slider_heatmap = outputs['heatmap_slider'][idx, 0].cpu().numpy()
        else:
            return img
        
        # 上采样热力图到原图尺寸
        gap_heatmap_resized = cv2.resize(gap_heatmap, (orig_w, orig_h), 
                                        interpolation=cv2.INTER_CUBIC)
        slider_heatmap_resized = cv2.resize(slider_heatmap, (orig_w, orig_h), 
                                           interpolation=cv2.INTER_CUBIC)
        
        # 归一化
        gap_heatmap_resized = np.clip(gap_heatmap_resized, 0, 1)
        slider_heatmap_resized = np.clip(slider_heatmap_resized, 0, 1)
        
        # 应用不同的颜色映射
        gap_colored = cv2.applyColorMap((gap_heatmap_resized * 255).astype(np.uint8), 
                                       cv2.COLORMAP_JET)
        slider_colored = cv2.applyColorMap((slider_heatmap_resized * 255).astype(np.uint8), 
                                          cv2.COLORMAP_HOT)
        
        # 合并热力图（取最大值）
        combined_heatmap = np.maximum(gap_colored, slider_colored)
        
        # 叠加到原图
        alpha = 0.4
        overlay_img = cv2.addWeighted(img, 1-alpha, combined_heatmap, alpha, 0)
        
        # 添加标签
        cv2.putText(overlay_img, "Gap+Slider Heatmap", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                   cv2.LINE_AA)
        
        return overlay_img
    
    def _draw_crosshair(self, img: np.ndarray, pos: np.ndarray, 
                       color: tuple, label: str, size: int = 15):
        """绘制十字准星标记"""
        x, y = int(pos[0]), int(pos[1])
        
        # 绘制十字
        cv2.line(img, (x - size, y), (x + size, y), color, 2, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), color, 2, cv2.LINE_AA)
        
        # 绘制中心点
        cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)
        
        # 添加标签背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        label_pos = (x + size + 5, y - 5)
        cv2.rectangle(img, 
                     (label_pos[0] - 2, label_pos[1] - label_size[1] - 2),
                     (label_pos[0] + label_size[0] + 2, label_pos[1] + 2),
                     (0, 0, 0), -1)
        cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    def _draw_box_marker(self, img: np.ndarray, pos: np.ndarray, 
                        color: tuple, label: str, size: int = 12):
        """绘制方框标记"""
        x, y = int(pos[0]), int(pos[1])
        
        # 绘制方框
        cv2.rectangle(img, (x - size, y - size), (x + size, y + size), 
                     color, 2, cv2.LINE_AA)
        
        # 绘制中心点
        cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)
        
        # 添加标签背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        label_pos = (x + size + 5, y + size + 5)
        cv2.rectangle(img,
                     (label_pos[0] - 2, label_pos[1] - label_size[1] - 2),
                     (label_pos[0] + label_size[0] + 2, label_pos[1] + 2),
                     (0, 0, 0), -1)
        cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 255, 255), 1, cv2.LINE_AA)