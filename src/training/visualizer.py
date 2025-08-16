#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化监控系统 - 负责TensorBoard日志和可视化
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非GUI环境
import io
from PIL import Image


class Visualizer:
    """
    可视化监控系统
    
    功能：
    1. TensorBoard日志记录
    2. 训练/验证曲线可视化
    3. 学习率曲线
    4. 预测结果可视化
    5. 权重和梯度直方图
    6. 失败案例分析
    7. 混淆矩阵和热力图
    """
    
    def __init__(self, config: Dict):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger('Visualizer')
        
        # TensorBoard设置
        tb_dir = config['logging'].get('tensorboard_dir', 'logs/tensorboard')
        self.writer = SummaryWriter(tb_dir)
        self.logger.info(f"TensorBoard日志目录: {tb_dir}")
        
        # 可视化配置
        self.log_interval = config['logging'].get('log_interval', 10)
        self.vis_fail_k = config['eval'].get('vis_fail_k', 10)
        
        # 记录的项目
        self.tb_items = config['logging'].get('tb_items', [
            'loss/train', 'loss/val',
            'metrics/mae_px', 'metrics/hit_le_5px',
            'lr/current'
        ])
        
        # 全局步数
        self.global_step = 0
        
        # 时间追踪配置
        self.time_tracking_config = config['logging'].get('time_tracking', {})
        self.time_tracking_enabled = self.time_tracking_config.get('enabled', False)
        
        # 时间追踪变量
        import time
        self.training_start_time = time.time()
        self.epoch_times = []  # 存储每个epoch的训练时间
        self.total_epochs = config.get('sched', {}).get('epochs', 100)
        
    def log_scalars(self, 
                   metrics: Dict[str, float], 
                   step: int, 
                   prefix: str = ''):
        """
        记录标量指标
        
        Args:
            metrics: 指标字典
            step: 步数（epoch或global_step）
            prefix: 前缀（如'train'或'val'）
        """
        for key, value in metrics.items():
            if prefix:
                tag = f"{prefix}/{key}"
            else:
                tag = key
            
            # 检查是否在记录项目中
            if self._should_log(tag):
                self.writer.add_scalar(tag, value, step)
    
    def log_training_metrics(self, metrics: Dict[str, float], epoch: int):
        """
        记录训练指标
        
        Args:
            metrics: 训练指标
            epoch: 当前epoch
        """
        # 损失
        self.writer.add_scalar('loss/train', metrics['loss'], epoch)
        self.writer.add_scalar('loss/focal', metrics['focal_loss'], epoch)
        self.writer.add_scalar('loss/offset', metrics['offset_loss'], epoch)
        self.writer.add_scalar('loss/hard_negative', metrics.get('hard_negative_loss', 0.0), epoch)
        self.writer.add_scalar('loss/angle', metrics.get('angle_loss', 0.0), epoch)
        
        # MAE
        self.writer.add_scalar('train/gap_mae', metrics['gap_mae'], epoch)
        self.writer.add_scalar('train/slider_mae', metrics['slider_mae'], epoch)
        self.writer.add_scalar('train/mae_avg', 
                              (metrics['gap_mae'] + metrics['slider_mae']) / 2, epoch)
        
        # 学习率
        self.writer.add_scalar('lr/current', metrics['lr'], epoch)
        
        # 确保数据被写入
        self.writer.flush()
    
    def log_validation_metrics(self, metrics: Dict[str, float], epoch: int):
        """
        记录验证指标
        
        Args:
            metrics: 验证指标
            epoch: 当前epoch
        """
        # 只记录需要的指标
        # 分别的MAE
        self.writer.add_scalar('val/gap_mae', metrics['gap_mae'], epoch)
        self.writer.add_scalar('val/slider_mae', metrics['slider_mae'], epoch)
        
        # 命中率
        self.writer.add_scalar('val/hit_le_2px', metrics['hit_le_2px'], epoch)
        self.writer.add_scalar('val/hit_le_5px', metrics['hit_le_5px'], epoch)
        
        # 综合指标
        self.writer.add_scalar('val/mae_px', metrics['mae_px'], epoch)
        self.writer.add_scalar('val/rmse_px', metrics['rmse_px'], epoch)
        
        # 确保数据被写入
        self.writer.flush()
    
    def log_learning_rate(self, lr: float, step: int):
        """
        记录学习率
        
        Args:
            lr: 当前学习率
            step: 步数
        """
        self.writer.add_scalar('lr/current', lr, step)
    
    def log_histograms(self, model: nn.Module, epoch: int):
        """
        记录权重和梯度直方图
        
        Args:
            model: 模型
            epoch: 当前epoch
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 权重直方图
                self.writer.add_histogram(
                    f'weight/{name}', 
                    param.data.cpu(), 
                    epoch
                )
                
                # 梯度直方图（如果有梯度）
                if param.grad is not None:
                    self.writer.add_histogram(
                        f'grad/{name}', 
                        param.grad.cpu(), 
                        epoch
                    )
                    
                    # 梯度范数
                    grad_norm = param.grad.norm(2).item()
                    self.writer.add_scalar(
                        f'grad_norm/{name}', 
                        grad_norm, 
                        epoch
                    )
    
    def log_predictions(self, 
                       images: torch.Tensor,
                       predictions: Dict[str, torch.Tensor],
                       targets: Dict[str, torch.Tensor],
                       epoch: int,
                       num_samples: int = 4,
                       num_best: int = 2,
                       num_worst: int = 2):
        """
        可视化预测结果（热力图与原图叠加显示）
        
        Args:
            images: 输入图像 [B, 4, H, W]
            predictions: 预测结果（包含热力图和坐标）
            targets: 真实标注
            epoch: 当前epoch
            num_samples: 总可视化样本数（兼容旧代码）
            num_best: 显示最佳预测的数量
            num_worst: 显示最差预测的数量
        """
        batch_size = images.size(0)
        
        # 计算所有样本的误差
        errors = []
        for i in range(batch_size):
            pred_gap = predictions['gap_coords'][i].cpu().numpy()
            gt_gap = targets['gap_coords'][i].cpu().numpy()
            pred_slider = predictions['slider_coords'][i].cpu().numpy()
            gt_slider = targets['slider_coords'][i].cpu().numpy()
            
            gap_error = np.linalg.norm(pred_gap - gt_gap)
            slider_error = np.linalg.norm(pred_slider - gt_slider)
            avg_error = (gap_error + slider_error) / 2.0
            
            errors.append((i, avg_error, gap_error, slider_error))
        
        # 按误差排序
        errors.sort(key=lambda x: x[1])
        
        # 选择最佳和最差的样本
        best_indices = errors[:min(num_best, len(errors))]
        worst_indices = errors[-min(num_worst, len(errors)):][::-1]  # 反转使最差的在前
        
        # 合并并创建标签
        selected_samples = []
        for rank, (idx, avg_err, gap_err, slider_err) in enumerate(best_indices):
            selected_samples.append((idx, gap_err, slider_err, f"Best #{rank+1}", (0, 255, 0)))  # 绿色标记
        for rank, (idx, avg_err, gap_err, slider_err) in enumerate(worst_indices):
            selected_samples.append((idx, gap_err, slider_err, f"Worst #{rank+1}", (0, 0, 255)))  # 红色标记
        
        # 可视化选中的样本（热力图叠加模式）
        for vis_idx, (i, gap_error, slider_error, label, label_color) in enumerate(selected_samples):
            # 获取原始图像（去除padding通道）
            img = images[i, :3].cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            orig_h, orig_w = img.shape[:2]  # 256, 512
            
            # 获取并叠加热力图
            vis_img = self._create_heatmap_overlay(img, predictions, i)
            
            # 只保留质量标签（Best/Worst）
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (5, 5), (120, 30), label_color, -1)
            vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
            cv2.putText(vis_img, label, 
                       (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 不绘制任何位置标记，只显示热力图叠加
            # 获取误差值用于记录（但不显示在图像上）
            gt_gap = targets['gap_coords'][i].cpu().numpy()
            pred_gap = predictions['gap_coords'][i].cpu().numpy()
            gt_slider = targets['slider_coords'][i].cpu().numpy()
            pred_slider = predictions['slider_coords'][i].cpu().numpy()
            
            # 转换回RGB并添加到TensorBoard
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            
            # 使用更明确的命名，确保Best和Worst分别显示
            if "Best" in label:
                # Best_1, Best_2, etc.
                sample_num = label.split("#")[1] if "#" in label else str(vis_idx)
                tag = f'predictions/best/sample_{sample_num}'
            else:
                # Worst_1, Worst_2, etc.
                sample_num = label.split("#")[1] if "#" in label else str(vis_idx)
                tag = f'predictions/worst/sample_{sample_num}'
            
            self.writer.add_image(
                tag, 
                vis_img.transpose(2, 0, 1), 
                epoch
            )
        
        # 确保数据被写入
        self.writer.flush()
    
    def log_heatmaps(self,
                    heatmaps: Dict[str, torch.Tensor],
                    epoch: int,
                    num_samples: int = 2,
                    images: Optional[torch.Tensor] = None):
        """
        可视化热力图（可选择与原图叠加）
        
        Args:
            heatmaps: 热力图字典（包含gap和slider）
            epoch: 当前epoch
            num_samples: 可视化样本数
            images: 原始图像 [B, 4, H, W]（可选，用于叠加显示）
        """
        # 检查key名称并适配
        gap_key = 'heatmap_gap' if 'heatmap_gap' in heatmaps else 'gap_heatmap'
        slider_key = 'heatmap_slider' if 'heatmap_slider' in heatmaps else 'slider_heatmap'
        
        num_vis = min(num_samples, heatmaps[gap_key].size(0))
        
        for i in range(num_vis):
            # 获取原图（如果提供）
            if images is not None and i < images.size(0):
                # 获取RGB图像（去除padding通道）
                orig_img = images[i, :3].cpu().numpy().transpose(1, 2, 0)
                orig_img = (orig_img * 255).astype(np.uint8)
                orig_h, orig_w = orig_img.shape[:2]  # 256, 512
            else:
                orig_img = None
                orig_h, orig_w = 256, 512  # 默认尺寸
            
            # 处理缺口热力图
            gap_heatmap = heatmaps[gap_key][i, 0].cpu().numpy()
            
            if orig_img is not None:
                # 上采样热力图到原图尺寸
                gap_heatmap_resized = cv2.resize(gap_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                
                # 应用颜色映射
                gap_heatmap_colored = cv2.applyColorMap((gap_heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                # 叠加到原图上（半透明）
                alpha = 0.5  # 透明度
                gap_overlay = cv2.addWeighted(orig_img, 1-alpha, gap_heatmap_colored, alpha, 0)
                
                # 添加标题
                cv2.putText(gap_overlay, "Gap Heatmap", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                self.writer.add_image(
                    f'heatmap_overlay/gap_{i}',
                    gap_overlay.transpose(2, 0, 1),
                    epoch
                )
            
            # 也保存原始热力图（不叠加）
            gap_heatmap_raw = (gap_heatmap * 255).astype(np.uint8)
            gap_heatmap_raw_colored = cv2.applyColorMap(gap_heatmap_raw, cv2.COLORMAP_JET)
            self.writer.add_image(
                f'heatmap_raw/gap_{i}',
                gap_heatmap_raw_colored.transpose(2, 0, 1),
                epoch
            )
            
            # 处理滑块热力图
            slider_heatmap = heatmaps[slider_key][i, 0].cpu().numpy()
            
            if orig_img is not None:
                # 上采样热力图到原图尺寸
                slider_heatmap_resized = cv2.resize(slider_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                
                # 应用颜色映射
                slider_heatmap_colored = cv2.applyColorMap((slider_heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                # 叠加到原图上（半透明）
                slider_overlay = cv2.addWeighted(orig_img, 1-alpha, slider_heatmap_colored, alpha, 0)
                
                # 添加标题
                cv2.putText(slider_overlay, "Slider Heatmap", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                self.writer.add_image(
                    f'heatmap_overlay/slider_{i}',
                    slider_overlay.transpose(2, 0, 1),
                    epoch
                )
            
            # 也保存原始热力图（不叠加）
            slider_heatmap_raw = (slider_heatmap * 255).astype(np.uint8)
            slider_heatmap_raw_colored = cv2.applyColorMap(slider_heatmap_raw, cv2.COLORMAP_JET)
            self.writer.add_image(
                f'heatmap_raw/slider_{i}',
                slider_heatmap_raw_colored.transpose(2, 0, 1),
                epoch
            )
        
        # 确保数据被写入
        self.writer.flush()
    
    def log_failure_cases(self, 
                         failures: List[Dict], 
                         epoch: int):
        """
        记录失败案例
        
        Args:
            failures: 失败案例列表
            epoch: 当前epoch
        """
        for i, failure in enumerate(failures[:self.vis_fail_k]):
            # 创建文本描述
            text = (
                f"文件: {failure['filename']}\n"
                f"误差: {failure['error']:.2f}px\n"
                f"缺口预测: {failure['gap_pred']}\n"
                f"缺口真值: {failure['gap_gt']}\n"
                f"滑块预测: {failure['slider_pred']}\n"
                f"滑块真值: {failure['slider_gt']}"
            )
            
            self.writer.add_text(
                f'failures/case_{i}',
                text,
                epoch
            )
    
    def log_confusion_matrix(self,
                           predictions: np.ndarray,
                           targets: np.ndarray,
                           epoch: int,
                           class_names: Optional[List[str]] = None):
        """
        记录混淆矩阵（如果有分类任务）
        
        Args:
            predictions: 预测类别
            targets: 真实类别
            epoch: 当前epoch
            class_names: 类别名称
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # 计算混淆矩阵
        cm = confusion_matrix(targets, predictions)
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 转换为图像
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image = np.array(image)
        
        # 添加到TensorBoard
        self.writer.add_image('confusion_matrix', 
                             image.transpose(2, 0, 1), 
                             epoch)
        plt.close()
    
    def log_model_graph(self, model: nn.Module, input_sample: torch.Tensor):
        """
        记录模型结构图
        
        Args:
            model: 模型
            input_sample: 输入样本
        """
        try:
            self.writer.add_graph(model, input_sample)
            self.logger.info("模型结构图已添加到TensorBoard")
        except Exception as e:
            self.logger.warning(f"无法添加模型结构图: {e}")
    
    def log_hyperparameters(self, hparams: Dict, metrics: Dict):
        """
        记录超参数和对应的指标
        
        Args:
            hparams: 超参数字典
            metrics: 最终指标字典
        """
        self.writer.add_hparams(hparams, metrics)
    
    def _should_log(self, tag: str) -> bool:
        """
        检查是否应该记录该项
        
        Args:
            tag: 标签名
            
        Returns:
            是否记录
        """
        # 如果没有指定tb_items，记录所有
        if not self.tb_items:
            return True
        
        # 检查是否匹配任何模式
        for pattern in self.tb_items:
            if '*' in pattern:
                # 支持通配符
                pattern_prefix = pattern.replace('*', '')
                if tag.startswith(pattern_prefix):
                    return True
            elif tag == pattern:
                return True
        
        return False
    
    def log_time_metrics(self, epoch: int, epoch_time: float):
        """
        记录时间相关的指标
        
        Args:
            epoch: 当前epoch
            epoch_time: 本epoch训练时间（秒）
        """
        if not self.time_tracking_enabled:
            return
        
        import time
        
        # 记录本epoch时间
        self.epoch_times.append(epoch_time)
        
        # 计算各种时间指标
        current_time = time.time()
        total_elapsed = current_time - self.training_start_time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        
        # 计算预计剩余时间
        remaining_epochs = self.total_epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        
        # 计算预计总时间
        estimated_total_seconds = avg_epoch_time * self.total_epochs
        
        # 根据配置的时间格式转换
        time_format = self.time_tracking_config.get('time_format', 'hours')
        if time_format == 'hours':
            time_divisor = 3600
            time_suffix = 'h'
        elif time_format == 'minutes':
            time_divisor = 60
            time_suffix = 'min'
        else:  # seconds
            time_divisor = 1
            time_suffix = 's'
        
        # 转换时间单位
        epoch_time_display = epoch_time / time_divisor
        total_elapsed_display = total_elapsed / time_divisor
        avg_epoch_time_display = avg_epoch_time / time_divisor
        eta_display = eta_seconds / time_divisor
        estimated_total_display = estimated_total_seconds / time_divisor
        
        # 不再记录时间到TensorBoard（用户要求移除）
        # 注释掉时间记录相关代码
        # if self.time_tracking_config.get('log_to_tensorboard', True):
        #     self.writer.add_scalar(f'time/epoch_duration_{time_suffix}', epoch_time_display, epoch)
        #     self.writer.add_scalar(f'time/total_elapsed_{time_suffix}', total_elapsed_display, epoch)
        #     self.writer.add_scalar(f'time/avg_epoch_time_{time_suffix}', avg_epoch_time_display, epoch)
        #     self.writer.add_scalar(f'time/eta_{time_suffix}', eta_display, epoch)
        #     self.writer.add_scalar(f'time/estimated_total_{time_suffix}', estimated_total_display, epoch)
        #     
        #     # 记录进度百分比
        #     progress_percent = (epoch / self.total_epochs) * 100
        #     self.writer.add_scalar('time/progress_percent', progress_percent, epoch)
        
        # 打印到控制台
        if self.time_tracking_config.get('display_eta', True):
            self.logger.info(f"时间统计 - Epoch {epoch}/{self.total_epochs}:")
            self.logger.info(f"  本epoch用时: {epoch_time_display:.2f}{time_suffix}")
            self.logger.info(f"  平均每epoch: {avg_epoch_time_display:.2f}{time_suffix}")
            self.logger.info(f"  已用总时间: {total_elapsed_display:.2f}{time_suffix}")
            self.logger.info(f"  预计剩余: {eta_display:.2f}{time_suffix}")
            self.logger.info(f"  预计总时间: {estimated_total_display:.2f}{time_suffix}")
            self.logger.info(f"  训练进度: {progress_percent:.1f}%")
    
    def get_eta_string(self, epoch: int) -> str:
        """
        获取格式化的ETA字符串
        
        Args:
            epoch: 当前epoch
            
        Returns:
            格式化的ETA字符串
        """
        if not self.time_tracking_enabled or not self.epoch_times:
            return "N/A"
        
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        
        # 格式化为 HH:MM:SS
        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _create_heatmap_overlay(self, img: np.ndarray, predictions: Dict, idx: int) -> np.ndarray:
        """
        创建热力图叠加效果
        
        Args:
            img: 原始图像 (H, W, 3)
            predictions: 预测结果字典
            idx: 样本索引
            
        Returns:
            叠加后的图像
        """
        orig_h, orig_w = img.shape[:2]
        
        # 检查是否有热力图输出
        if 'heatmaps' in predictions and predictions['heatmaps'] is not None:
            # 从预测中获取热力图
            if 'gap' in predictions['heatmaps']:
                gap_heatmap = predictions['heatmaps']['gap'][idx, 0].cpu().numpy()
            elif 'heatmap_gap' in predictions:
                gap_heatmap = predictions['heatmap_gap'][idx, 0].cpu().numpy()
            else:
                return img
                
            if 'slider' in predictions['heatmaps']:
                slider_heatmap = predictions['heatmaps']['slider'][idx, 0].cpu().numpy()
            elif 'heatmap_slider' in predictions:
                slider_heatmap = predictions['heatmap_slider'][idx, 0].cpu().numpy()
            else:
                slider_heatmap = gap_heatmap  # 使用相同的热力图
        elif 'heatmap_gap' in predictions and 'heatmap_slider' in predictions:
            # 直接从predictions获取
            gap_heatmap = predictions['heatmap_gap'][idx, 0].cpu().numpy()
            slider_heatmap = predictions['heatmap_slider'][idx, 0].cpu().numpy()
        else:
            # 没有热力图，返回原图
            return img
        
        # 上采样热力图到原图尺寸
        gap_heatmap_resized = cv2.resize(gap_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        slider_heatmap_resized = cv2.resize(slider_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        # 应用不同的颜色映射
        gap_heatmap_colored = cv2.applyColorMap((gap_heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        slider_heatmap_colored = cv2.applyColorMap((slider_heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        
        # 合并两个热力图（取最大值）
        combined_heatmap = np.maximum(gap_heatmap_colored, slider_heatmap_colored)
        
        # 叠加到原图上（半透明）
        alpha = 0.3  # 透明度 - 30%确保热力图清晰可见
        overlay_img = cv2.addWeighted(img, 1-alpha, combined_heatmap, alpha, 0)
        
        return overlay_img
    
    def _draw_crosshair(self, img: np.ndarray, pos: np.ndarray, color: tuple, label: str, size: int = 10):
        """
        绘制十字准星标记
        
        Args:
            img: 图像
            pos: 位置坐标 [x, y]
            color: 颜色
            label: 标签文字
            size: 十字大小
        """
        x, y = int(pos[0]), int(pos[1])
        
        # 绘制十字
        cv2.line(img, (x - size, y), (x + size, y), color, 2)
        cv2.line(img, (x, y - size), (x, y + size), color, 2)
        
        # 绘制中心点
        cv2.circle(img, (x, y), 3, color, -1)
        
        # 添加标签
        label_pos = (x + size + 5, y - 5)
        cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_box_marker(self, img: np.ndarray, pos: np.ndarray, color: tuple, label: str, size: int = 8):
        """
        绘制方框标记
        
        Args:
            img: 图像
            pos: 位置坐标 [x, y]
            color: 颜色
            label: 标签文字
            size: 方框大小
        """
        x, y = int(pos[0]), int(pos[1])
        
        # 绘制方框
        cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, 2)
        
        # 绘制中心点
        cv2.circle(img, (x, y), 2, color, -1)
        
        # 添加标签
        label_pos = (x + size + 5, y + size + 5)
        cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def close(self):
        """关闭TensorBoard writer"""
        self.writer.close()
        self.logger.info("TensorBoard writer已关闭")
    
    def flush(self):
        """刷新TensorBoard writer"""
        self.writer.flush()


if __name__ == "__main__":
    # 测试可视化器
    print("可视化监控系统模块测试")
    
    # 模拟配置
    config = {
        'logging': {
            'tensorboard_dir': 'logs/test_tb',
            'log_interval': 10,
            'tb_items': ['loss/*', 'metrics/*', 'lr/*']
        },
        'eval': {
            'vis_fail_k': 10
        }
    }
    
    # 创建可视化器
    visualizer = Visualizer(config)
    
    # 测试记录标量
    test_metrics = {
        'loss': 0.5,
        'mae': 2.3,
        'hit_rate': 95.5
    }
    visualizer.log_scalars(test_metrics, step=1, prefix='test')
    
    print("可视化器测试完成，请打开TensorBoard查看结果")
    print(f"命令: tensorboard --logdir {config['logging']['tensorboard_dir']}")
    
    visualizer.close()