#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证评估系统 - 负责模型验证、指标计算和早停管理
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from pathlib import Path


class Validator:
    """
    验证评估系统
    
    功能：
    1. 执行模型验证
    2. 计算评估指标（MAE、RMSE、命中率）
    3. 管理早停机制（双重防护）
    4. 跟踪最佳模型
    5. 保存验证结果
    """
    
    def __init__(self, config: Dict, device: torch.device):
        """
        初始化验证器
        
        Args:
            config: 配置字典
            device: 设备
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger('Validator')
        
        # 评估配置
        self.eval_config = config['eval']
        self.select_metric = self.eval_config['select_by']
        
        # 早停配置
        self.early_stopping_config = self.eval_config.get('early_stopping', {})
        self.min_epochs = self.early_stopping_config.get('min_epochs', 100)
        self.patience = self.early_stopping_config.get('patience', 18)
        
        # 主指标跟踪
        self.best_metric = -float('inf') if self._is_higher_better(self.select_metric) else float('inf')
        self.patience_counter = 0
        
        # 第二道防护指标
        self.second_guard_enabled = False
        if 'second_guard' in self.early_stopping_config:
            guard_cfg = self.early_stopping_config['second_guard']
            self.second_guard_metric = guard_cfg['metric']
            self.second_guard_mode = guard_cfg['mode']
            self.second_guard_min_delta = guard_cfg['min_delta']
            self.second_guard_best = float('inf') if guard_cfg['mode'] == 'min' else -float('inf')
            self.second_guard_enabled = True
            self.logger.info(f"启用第二道防护: {self.second_guard_metric} ({self.second_guard_mode})")
        
        # 指标历史
        self.metrics_history = []
        self.best_epoch = 0
        
        # 失败案例收集
        self.vis_fail_k = self.eval_config.get('vis_fail_k', 10)
        self.failure_cases = []
    
    def validate(self, 
                model: nn.Module, 
                dataloader, 
                epoch: int,
                use_ema: bool = False) -> Dict[str, float]:
        """
        执行验证
        
        Args:
            model: 模型
            dataloader: 验证数据加载器
            epoch: 当前epoch
            use_ema: 是否使用EMA模型
            
        Returns:
            验证指标字典
        """
        model.eval()
        
        # 初始化指标收集器
        all_gap_errors = []
        all_slider_errors = []
        all_combined_errors = []
        
        # 用于计算命中率
        total_samples = 0
        hit_counts = {
            1: 0,  # ≤1px
            2: 0,  # ≤2px
            5: 0   # ≤5px
        }
        
        # 失败案例
        epoch_failures = []
        
        self.logger.info(f"开始验证 Epoch {epoch}...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # 数据传输到设备
                batch = self._batch_to_device(batch)
                
                # 前向传播
                outputs = model(batch['image'])
                
                # 解码预测
                predictions = model.decode_predictions(outputs)
                
                # 计算误差
                gap_errors, slider_errors = self._calculate_errors(
                    predictions, 
                    batch
                )
                
                # 收集误差
                all_gap_errors.extend(gap_errors.cpu().numpy())
                all_slider_errors.extend(slider_errors.cpu().numpy())
                
                # 组合误差（缺口和滑块的平均）
                combined_errors = (gap_errors + slider_errors) / 2
                all_combined_errors.extend(combined_errors.cpu().numpy())
                
                # 计算命中率
                batch_size = batch['image'].size(0)
                for threshold in hit_counts.keys():
                    hit_counts[threshold] += (combined_errors <= threshold).sum().item()
                total_samples += batch_size
                
                # 收集失败案例
                if len(epoch_failures) < self.vis_fail_k:
                    self._collect_failures(
                        batch, predictions, combined_errors, 
                        epoch_failures
                    )
        
        # 计算汇总指标
        metrics = self._compute_metrics(
            all_gap_errors,
            all_slider_errors,
            all_combined_errors,
            hit_counts,
            total_samples
        )
        
        # 添加epoch信息
        metrics['epoch'] = epoch
        
        # 保存指标历史
        self.metrics_history.append(metrics)
        
        # 更新失败案例
        self.failure_cases = epoch_failures[:self.vis_fail_k]
        
        # 早停检查
        if self._check_early_stopping(metrics, epoch):
            metrics['early_stop'] = True
            self.logger.info(f"触发早停机制 at epoch {epoch}")
        else:
            metrics['early_stop'] = False
        
        # 检查是否为最佳模型
        if self._is_best_model(metrics):
            self.best_epoch = epoch
            metrics['is_best'] = True
            self.logger.info(f"新的最佳模型! {self.select_metric}: {metrics[self.select_metric]:.4f}")
        else:
            metrics['is_best'] = False
        
        # 打印验证结果
        self._print_metrics(metrics)
        
        return metrics
    
    def _batch_to_device(self, batch: Dict) -> Dict:
        """将批次数据传输到设备"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _calculate_errors(self, 
                         predictions: Dict[str, torch.Tensor], 
                         batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算预测误差
        
        Args:
            predictions: 模型预测
            batch: 批次数据
            
        Returns:
            缺口误差和滑块误差
        """
        # 计算L1误差（MAE）
        gap_errors = torch.abs(
            predictions['gap_coords'] - batch['gap_coords']
        ).mean(dim=1)  # [B]
        
        slider_errors = torch.abs(
            predictions['slider_coords'] - batch['slider_coords']
        ).mean(dim=1)  # [B]
        
        return gap_errors, slider_errors
    
    def _compute_metrics(self,
                        gap_errors: List[float],
                        slider_errors: List[float],
                        combined_errors: List[float],
                        hit_counts: Dict[int, int],
                        total_samples: int) -> Dict[str, float]:
        """
        计算汇总指标
        
        Args:
            gap_errors: 缺口误差列表
            slider_errors: 滑块误差列表
            combined_errors: 组合误差列表
            hit_counts: 命中计数
            total_samples: 总样本数
            
        Returns:
            指标字典
        """
        # 转换为numpy数组
        gap_errors = np.array(gap_errors)
        slider_errors = np.array(slider_errors)
        combined_errors = np.array(combined_errors)
        
        # 计算指标
        metrics = {
            # 主要指标（组合）
            'mae_px': np.mean(combined_errors),
            'rmse_px': np.sqrt(np.mean(combined_errors ** 2)),
            
            # 分别的MAE
            'gap_mae': np.mean(gap_errors),
            'slider_mae': np.mean(slider_errors),
            
            # 分别的RMSE
            'gap_rmse': np.sqrt(np.mean(gap_errors ** 2)),
            'slider_rmse': np.sqrt(np.mean(slider_errors ** 2)),
            
            # 命中率
            'hit_le_1px': (hit_counts[1] / total_samples) * 100,
            'hit_le_2px': (hit_counts[2] / total_samples) * 100,
            'hit_le_5px': (hit_counts[5] / total_samples) * 100,
            
            # 统计信息
            'mae_std': np.std(combined_errors),
            'mae_max': np.max(combined_errors),
            'mae_median': np.median(combined_errors),
            
            # 样本数
            'num_samples': total_samples
        }
        
        return metrics
    
    def _collect_failures(self,
                         batch: Dict,
                         predictions: Dict,
                         errors: torch.Tensor,
                         failure_list: List):
        """
        收集失败案例
        
        Args:
            batch: 批次数据
            predictions: 预测结果
            errors: 误差
            failure_list: 失败案例列表
        """
        # 找出误差最大的样本
        batch_size = errors.size(0)
        
        for i in range(batch_size):
            if len(failure_list) >= self.vis_fail_k:
                break
            
            # 只收集误差较大的案例
            if errors[i].item() > 5.0:  # 大于5像素的误差
                failure = {
                    'filename': batch['filename'][i] if 'filename' in batch else f"sample_{i}",
                    'error': errors[i].item(),
                    'gap_pred': predictions['gap_coords'][i].cpu().numpy(),
                    'gap_gt': batch['gap_coords'][i].cpu().numpy(),
                    'slider_pred': predictions['slider_coords'][i].cpu().numpy(),
                    'slider_gt': batch['slider_coords'][i].cpu().numpy(),
                }
                failure_list.append(failure)
    
    def _check_early_stopping(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        检查早停条件（双重防护）
        
        Args:
            metrics: 当前指标
            epoch: 当前epoch
            
        Returns:
            是否触发早停
        """
        # 未达到最小训练轮数
        if epoch < self.min_epochs:
            return False
        
        # 获取主指标
        current_metric = metrics[self.select_metric]
        
        # 检查主指标是否改善
        has_primary_improvement = False
        if self._is_better(current_metric, self.best_metric, self.select_metric):
            self.best_metric = current_metric
            self.patience_counter = 0
            has_primary_improvement = True
        else:
            self.patience_counter += 1
        
        # 第二道防护检查
        if self.second_guard_enabled and epoch >= self.min_epochs:
            current_guard_metric = metrics[self.second_guard_metric]
            
            if self.second_guard_mode == 'min':
                # 越小越好（如mae_px）
                improvement = self.second_guard_best - current_guard_metric
                if improvement > self.second_guard_min_delta:
                    self.second_guard_best = current_guard_metric
                    # 第二指标有显著改善，减少耐心计数
                    self.patience_counter = max(0, self.patience_counter - 3)
                    self.logger.info(
                        f"第二道防护: {self.second_guard_metric} 改善了 {improvement:.3f}px"
                    )
            else:
                # 越大越好
                improvement = current_guard_metric - self.second_guard_best
                if improvement > self.second_guard_min_delta:
                    self.second_guard_best = current_guard_metric
                    self.patience_counter = max(0, self.patience_counter - 3)
                    self.logger.info(
                        f"第二道防护: {self.second_guard_metric} 改善了 {improvement:.3f}"
                    )
        
        # 检查是否超过耐心值
        if self.patience_counter >= self.patience:
            self.logger.info(f"早停触发 at epoch {epoch}")
            self.logger.info(f"最佳 {self.select_metric}: {self.best_metric:.4f}")
            if self.second_guard_enabled:
                self.logger.info(f"最佳 {self.second_guard_metric}: {self.second_guard_best:.4f}")
            return True
        
        return False
    
    def _is_better(self, current: float, best: float, metric: str) -> bool:
        """判断当前值是否更好"""
        if self._is_higher_better(metric):
            return current > best
        else:
            return current < best
    
    def _is_higher_better(self, metric: str) -> bool:
        """判断指标是否越高越好"""
        higher_better_metrics = ['hit_le_1px', 'hit_le_2px', 'hit_le_5px']
        return metric in higher_better_metrics
    
    def _is_best_model(self, metrics: Dict[str, float]) -> bool:
        """判断是否为最佳模型"""
        current_metric = metrics[self.select_metric]
        return self._is_better(current_metric, self.best_metric, self.select_metric)
    
    def _print_metrics(self, metrics: Dict[str, float]):
        """打印验证指标"""
        self.logger.info(
            f"Epoch {metrics['epoch']} 验证结果: "
            f"MAE={metrics['mae_px']:.3f}px, "
            f"RMSE={metrics['rmse_px']:.3f}px, "
            f"Hit@1px={metrics['hit_le_1px']:.2f}%, "
            f"Hit@2px={metrics['hit_le_2px']:.2f}%, "
            f"Hit@5px={metrics['hit_le_5px']:.2f}%"
        )
        
        if self.second_guard_enabled:
            self.logger.info(
                f"  第二防护: {self.second_guard_metric}={metrics[self.second_guard_metric]:.3f}, "
                f"最佳={self.second_guard_best:.3f}"
            )
        
        self.logger.info(
            f"  耐心计数: {self.patience_counter}/{self.patience}"
        )
    
    def get_best_metrics(self) -> Optional[Dict[str, float]]:
        """获取最佳指标"""
        if not self.metrics_history:
            return None
        
        # 找到最佳epoch的指标
        for metrics in self.metrics_history:
            if metrics['epoch'] == self.best_epoch:
                return metrics
        
        return None
    
    def get_failure_cases(self) -> List[Dict]:
        """获取失败案例"""
        return self.failure_cases
    
    def save_metrics_history(self, path: str):
        """
        保存指标历史
        
        Args:
            path: 保存路径
        """
        import json
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"指标历史已保存到: {save_path}")


if __name__ == "__main__":
    # 测试验证器
    print("验证评估系统模块测试")
    print("注：需要配合模型和数据加载器使用")