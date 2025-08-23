#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练工具函数模块
包含数据加载、参数解析、配置管理等辅助功能
"""
import argparse
import torch
import logging
import numpy as np
import random


class LimitedDataLoader:
    """
    限制批次数量的DataLoader包装器
    用于测试或部分数据训练
    """
    def __init__(self, loader, max_batches):
        """
        初始化限制数据加载器
        
        Args:
            loader: 原始数据加载器
            max_batches: 最大批次数量
        """
        self.loader = loader
        self.max_batches = max_batches
        
    def __iter__(self):
        """迭代器实现"""
        count = 0
        for batch in self.loader:
            if count >= self.max_batches:
                break
            yield batch
            count += 1
            
    def __len__(self):
        """返回限制后的批次数量"""
        return self.max_batches


def parse_training_args():
    """
    解析训练命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='训练Lite-HRNet-18+LiteFPN滑块验证码识别模型'
    )
    
    # 基础参数
    parser.add_argument('--config', type=str, 
                       default='config/training_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--eval-only', action='store_true',
                       help='仅执行评估')
    
    # 覆盖配置参数
    parser.add_argument('--batch-size', type=int, default=None,
                       help='覆盖配置中的批次大小')
    parser.add_argument('--lr', type=float, default=None,
                       help='覆盖配置中的学习率')
    parser.add_argument('--epochs', type=int, default=None,
                       help='覆盖配置中的训练轮数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--name', type=str, default=None,
                       help='实验名称')
    
    return parser.parse_args()


def set_global_seed(seed: int):
    """
    设置全局随机种子
    
    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Note: CuDNN settings are now handled by config_manager.apply_hardware_optimizations()


def override_config_from_args(config: dict, args):
    """
    根据命令行参数覆盖配置
    
    Args:
        config: 原始配置字典
        args: 命令行参数
    """
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
        logging.info(f"覆盖batch_size: {args.batch_size}")
    
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr
        logging.info(f"覆盖学习率: {args.lr}")
    
    if args.epochs is not None:
        config['sched']['epochs'] = args.epochs
        logging.info(f"覆盖训练轮数: {args.epochs}")
    
    if args.name is not None:
        # 使用name参数作为子目录后缀
        base_checkpoint_dir = config['checkpoints']['save_dir'].rsplit('/', 1)[0]
        base_log_dir = config['logging']['log_dir'].rsplit('-', 1)[0]
        base_tb_dir = config['logging']['tensorboard_dir'].rsplit('/', 1)[0]
        
        config['checkpoints']['save_dir'] = f"{base_checkpoint_dir}/{args.name}"
        config['logging']['log_dir'] = f"{base_log_dir}-{args.name}"
        config['logging']['tensorboard_dir'] = f"{base_tb_dir}/{args.name}"
        logging.info(f"实验名称: {args.name}")