#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查点管理模块
负责模型检查点的保存、加载和管理
"""
import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Any


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_checkpoint(self, 
                       model, 
                       engine, 
                       validator, 
                       epoch: int, 
                       config: Dict,
                       is_best: bool = False) -> None:
        """
        保存检查点
        
        Args:
            model: 模型
            engine: 训练引擎
            validator: 验证器
            epoch: 当前epoch
            config: 配置字典
            is_best: 是否为最佳模型
        """
        # 获取检查点数据
        checkpoint = engine.save_checkpoint()
        
        # 添加额外信息
        checkpoint['epoch'] = epoch
        checkpoint['best_metric'] = validator.best_metric
        checkpoint['config'] = config
        
        # 保存当前epoch检查点
        save_interval = config.get('checkpoints', {}).get('save_interval', 1)
        if epoch % save_interval == 0:
            checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pth"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"保存检查点: {checkpoint_path}")
            
            # EMA已移除
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / "last.pth"
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            self._save_best_model(checkpoint, model, engine, config)
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model,
                       engine,
                       validator=None) -> int:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            model: 模型
            engine: 训练引擎
            validator: 验证器（可选）
        
        Returns:
            起始epoch
        """
        self.logger.info(f"加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型和优化器状态
        engine.load_checkpoint(checkpoint)
        
        # 加载验证器状态
        if validator and 'best_metric' in checkpoint:
            validator.best_metric = checkpoint['best_metric']
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        self.logger.info(f"从epoch {start_epoch}恢复训练")
        
        return start_epoch
    
    def _save_best_model(self, checkpoint: Dict, model, engine, config: Dict) -> None:
        """
        保存最佳模型
        
        Args:
            checkpoint: 检查点数据
            model: 模型
            engine: 训练引擎
            config: 配置字典
        """
        best_path = self.checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        self.logger.info(f"保存最佳模型: {best_path}")
        
        # 同时保存纯模型权重（用于推理）
        model_only_path = self.checkpoint_dir / "best_model_weights.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, model_only_path)
        
        # EMA已移除
    
    # EMA相关方法已移除
    
    def save_final_best_model(self, 
                             model, 
                             engine,
                             validator,
                             data_pipeline,
                             config: Dict,
                             epoch: int) -> None:
        """
        在训练结束时保存最终的最佳模型
        
        Args:
            model: 主模型
            engine: 训练引擎
            validator: 验证器
            data_pipeline: 数据管道
            config: 配置字典
            epoch: 最终epoch
        """
        self.logger.info("\n保存最终模型...")
        
        # 验证模型
        self.logger.info("验证模型...")
        final_metrics = validator.validate(
            model, 
            data_pipeline.get_val_loader(), 
            epoch
        )
        
        # 获取最终指标
        select_metric = config['eval'].get('select_by', 'hit_le_5px')
        final_score = final_metrics.get(select_metric, 0)
        
        self.logger.info(f"\n最终性能 ({select_metric}): {final_score:.4f}")
        
        # 保存最终模型
        final_path = self.checkpoint_dir / "final_best_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'final_score': final_score
        }, final_path)
        self.logger.info(f"最终模型保存至: {final_path}")