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
        
        # MAE
        self.writer.add_scalar('train/gap_mae', metrics['gap_mae'], epoch)
        self.writer.add_scalar('train/slider_mae', metrics['slider_mae'], epoch)
        self.writer.add_scalar('train/mae_avg', 
                              (metrics['gap_mae'] + metrics['slider_mae']) / 2, epoch)
        
        # 学习率
        self.writer.add_scalar('lr/current', metrics['lr'], epoch)
    
    def log_validation_metrics(self, metrics: Dict[str, float], epoch: int):
        """
        记录验证指标
        
        Args:
            metrics: 验证指标
            epoch: 当前epoch
        """
        # 主要指标
        self.writer.add_scalar('val/mae_px', metrics['mae_px'], epoch)
        self.writer.add_scalar('val/rmse_px', metrics['rmse_px'], epoch)
        
        # 分别的误差
        self.writer.add_scalar('val/gap_mae', metrics['gap_mae'], epoch)
        self.writer.add_scalar('val/slider_mae', metrics['slider_mae'], epoch)
        
        # 命中率
        self.writer.add_scalar('val/hit_le_1px', metrics['hit_le_1px'], epoch)
        self.writer.add_scalar('val/hit_le_2px', metrics['hit_le_2px'], epoch)
        self.writer.add_scalar('val/hit_le_5px', metrics['hit_le_5px'], epoch)
        
        # 统计信息
        self.writer.add_scalar('val/mae_std', metrics['mae_std'], epoch)
        self.writer.add_scalar('val/mae_max', metrics['mae_max'], epoch)
        self.writer.add_scalar('val/mae_median', metrics['mae_median'], epoch)
    
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
                       num_samples: int = 4):
        """
        可视化预测结果
        
        Args:
            images: 输入图像 [B, 4, H, W]
            predictions: 预测结果
            targets: 真实标注
            epoch: 当前epoch
            num_samples: 可视化样本数
        """
        # 选择前N个样本
        num_vis = min(num_samples, images.size(0))
        
        for i in range(num_vis):
            # 获取图像（去除padding通道）
            img = images[i, :3].cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 创建可视化图像
            vis_img = img.copy()
            
            # 真实缺口位置（绿色）
            gt_gap = targets['gap_coords'][i].cpu().numpy()
            cv2.circle(vis_img, tuple(gt_gap.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(vis_img, "GT_Gap", 
                       (int(gt_gap[0])+10, int(gt_gap[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 预测缺口位置（红色）
            pred_gap = predictions['gap_coords'][i].cpu().numpy()
            cv2.circle(vis_img, tuple(pred_gap.astype(int)), 5, (255, 0, 0), -1)
            cv2.putText(vis_img, "Pred_Gap", 
                       (int(pred_gap[0])+10, int(pred_gap[1])+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 真实滑块位置（青色）
            gt_slider = targets['slider_coords'][i].cpu().numpy()
            cv2.circle(vis_img, tuple(gt_slider.astype(int)), 5, (0, 255, 255), -1)
            cv2.putText(vis_img, "GT_Slider", 
                       (int(gt_slider[0])+10, int(gt_slider[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 预测滑块位置（品红色）
            pred_slider = predictions['slider_coords'][i].cpu().numpy()
            cv2.circle(vis_img, tuple(pred_slider.astype(int)), 5, (255, 0, 255), -1)
            cv2.putText(vis_img, "Pred_Slider", 
                       (int(pred_slider[0])+10, int(pred_slider[1])+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # 计算误差
            gap_error = np.linalg.norm(pred_gap - gt_gap)
            slider_error = np.linalg.norm(pred_slider - gt_slider)
            
            # 添加误差信息
            error_text = f"Gap Error: {gap_error:.2f}px, Slider Error: {slider_error:.2f}px"
            cv2.putText(vis_img, error_text, 
                       (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 转换回RGB并添加到TensorBoard
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            self.writer.add_image(
                f'prediction/sample_{i}', 
                vis_img.transpose(2, 0, 1), 
                epoch
            )
    
    def log_heatmaps(self,
                    heatmaps: Dict[str, torch.Tensor],
                    epoch: int,
                    num_samples: int = 2):
        """
        可视化热力图
        
        Args:
            heatmaps: 热力图字典（包含gap和slider）
            epoch: 当前epoch
            num_samples: 可视化样本数
        """
        num_vis = min(num_samples, heatmaps['heatmap_gap'].size(0))
        
        for i in range(num_vis):
            # 缺口热力图
            gap_heatmap = heatmaps['heatmap_gap'][i, 0].cpu().numpy()
            gap_heatmap = (gap_heatmap * 255).astype(np.uint8)
            gap_heatmap = cv2.applyColorMap(gap_heatmap, cv2.COLORMAP_JET)
            
            self.writer.add_image(
                f'heatmap/gap_{i}',
                gap_heatmap.transpose(2, 0, 1),
                epoch
            )
            
            # 滑块热力图
            slider_heatmap = heatmaps['heatmap_slider'][i, 0].cpu().numpy()
            slider_heatmap = (slider_heatmap * 255).astype(np.uint8)
            slider_heatmap = cv2.applyColorMap(slider_heatmap, cv2.COLORMAP_JET)
            
            self.writer.add_image(
                f'heatmap/slider_{i}',
                slider_heatmap.transpose(2, 0, 1),
                epoch
            )
    
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