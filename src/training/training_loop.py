#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练循环模块
包含训练和验证的循环逻辑
"""
import logging
import time
from typing import Dict, Optional, Any


def train_epoch(model, engine, dataloader, epoch: int, visualizer) -> Dict[str, float]:
    """
    训练一个epoch
    
    Args:
        model: 模型
        engine: 训练引擎
        dataloader: 数据加载器
        epoch: 当前epoch
        visualizer: 可视化器
    
    Returns:
        训练指标字典
    """
    logging.info(f"开始训练 Epoch {epoch}")
    
    # 执行训练
    train_metrics = engine.train_epoch(dataloader, epoch)
    
    # 记录到TensorBoard
    visualizer.log_training_metrics(train_metrics, epoch)
    
    # 记录权重直方图（每10个epoch）
    if epoch % 10 == 0:
        visualizer.log_histograms(model, epoch)
    
    return train_metrics


def validate_epoch(model, 
                  validator, 
                  dataloader, 
                  epoch: int, 
                  visualizer, 
                  global_step: Optional[int] = None) -> Dict[str, Any]:
    """
    验证一个epoch
    
    Args:
        model: 模型
        validator: 验证器
        dataloader: 数据加载器
        epoch: 当前epoch
        visualizer: 可视化器
        global_step: 全局步数（用于TensorBoard记录）
    
    Returns:
        验证指标字典
    """
    logging.info(f"开始验证 Epoch {epoch}")
    
    # 执行验证
    val_metrics = validator.validate(model, dataloader, epoch)
    
    # 记录到TensorBoard
    # 主验证指标使用global_step（如果提供）或epoch
    step_to_use = global_step if global_step is not None else epoch
    visualizer.log_validation_metrics(val_metrics, step_to_use)
    
    # 处理可视化数据
    if 'vis_data' in val_metrics:
        _handle_visualization_data(val_metrics, validator, visualizer, epoch)
    
    # 记录失败案例
    failures = validator.get_failure_cases()
    if failures:
        visualizer.log_failure_cases(failures, epoch)
    
    return val_metrics


def _handle_visualization_data(val_metrics: Dict, 
                               validator, 
                               visualizer, 
                               epoch: int) -> None:
    """
    处理验证过程中的可视化数据
    
    Args:
        val_metrics: 验证指标字典
        validator: 验证器
        visualizer: 可视化器
        epoch: 当前epoch
    """
    vis_data = val_metrics['vis_data']
    vis_config = validator.config.get('eval', {}).get('visualization', {})
    
    # 记录预测可视化（显示最佳和最差样本）
    if vis_config.get('save_predictions', True):
        num_best = vis_config.get('num_best_samples', 2)
        num_worst = vis_config.get('num_worst_samples', 2)
        num_pred_samples = num_best + num_worst
        visualizer.log_predictions(
            vis_data['images'],
            vis_data['predictions'],
            vis_data['targets'],
            epoch,
            num_samples=num_pred_samples,
            num_best=num_best,
            num_worst=num_worst
        )
    
    # 记录热力图（使用配置中的样本数）
    if vis_config.get('save_heatmaps', True):
        num_heatmap_samples = vis_config.get('num_heatmap_samples', 2)
        visualizer.log_heatmaps(
            vis_data['outputs'],
            epoch,
            num_samples=num_heatmap_samples,
            images=vis_data['images']  # 传入原图以启用叠加显示
        )
    
    # 清理可视化数据，避免保存到检查点
    del val_metrics['vis_data']


def run_training_loop(model,
                     engine,
                     validator,
                     data_pipeline,
                     visualizer,
                     checkpoint_manager,
                     config: Dict,
                     start_epoch: int = 1) -> None:
    """
    运行完整的训练循环
    
    Args:
        model: 模型
        engine: 训练引擎
        validator: 验证器
        data_pipeline: 数据管道
        visualizer: 可视化器
        checkpoint_manager: 检查点管理器
        config: 配置字典
        start_epoch: 起始epoch
    """
    total_epochs = config['sched']['epochs']
    
    # 检查是否显示进度条
    show_progress = config.get('logging', {}).get('time_tracking', {}).get('show_progress_bar', False)
    
    if show_progress:
        from tqdm import tqdm
        print()  # 在进度条之前添加空行
        epoch_iterator = tqdm(range(start_epoch, total_epochs + 1), 
                             desc="Training Progress", 
                             unit="epoch",
                             initial=start_epoch - 1,
                             total=total_epochs)
    else:
        epoch_iterator = range(start_epoch, total_epochs + 1)
    
    # 训练循环
    for epoch in epoch_iterator:
        epoch_start_time = time.time()
        
        # 训练
        train_loader = data_pipeline.get_train_loader()
        train_metrics = train_epoch(
            model, engine, train_loader, epoch, visualizer
        )
        
        # 验证
        val_loader = data_pipeline.get_val_loader()
        val_metrics = validate_epoch(
            model, validator, val_loader, epoch, visualizer,
            global_step=engine.global_step
        )
        
        # EMA已移除
        
        # 更新学习率
        engine.step_scheduler()
        
        # 保存检查点
        checkpoint_manager.save_checkpoint(
            model, engine, validator, epoch, config,
            is_best=val_metrics.get('is_best', False)
        )
        
        # 记录epoch时间
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch} 用时: {epoch_time:.2f}秒")
        
        # 检查早停
        if val_metrics.get('should_stop', False):
            logging.info(f"早停触发，在Epoch {epoch}停止训练")
            break
        
        # 刷新可视化
        visualizer.flush()


# EMA相关函数已移除