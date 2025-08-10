#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的训练入口脚本
使用模块化组件训练PMN-R3-FP模型
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.pmn_r3_fp import PMN_R3_FP
from src.models.loss import PMN_R3_FP_Loss
from src.datasets.captcha_dataset import create_data_loaders
from src.training.trainer import Trainer
from src.training.utils import (
    load_config,
    set_seed,
    get_optimizer,
    get_scheduler,
    Logger,
    load_checkpoint
)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train PMN-R3-FP Model')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Path to training config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ids (e.g., "0,1,2")')
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(config['seed'], config['deterministic'])
    
    # 创建日志记录器
    logger = Logger(
        config['logging']['log_dir'],
        experiment_name=f"pmn_r3_fp_{config['model']['name']}"
    )
    logger.log("Configuration loaded successfully")
    logger.log(f"Device: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # 创建模型
    model = PMN_R3_FP(config['model']).to(device)
    
    # 多GPU支持
    if config['training']['multi_gpu'] and torch.cuda.device_count() > 1:
        logger.log(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    # 创建损失函数
    criterion = PMN_R3_FP_Loss(config['loss']['weights'])
    
    # 创建优化器和调度器
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        logger=logger,
        device=device
    )
    
    # 恢复训练（如果指定）
    if args.resume:
        checkpoint_info = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device
        )
        trainer.start_epoch = checkpoint_info['epoch']
        trainer.best_metric = checkpoint_info['best_metric']
        logger.log(f"Resumed from epoch {checkpoint_info['epoch']}")
    
    # 开始训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        checkpoint_dir=config['checkpoint']['save_dir'],
        test_loader=test_loader if config['testing']['run_test_after_training'] else None
    )
    
    logger.log("Training script completed!")


if __name__ == '__main__':
    main()