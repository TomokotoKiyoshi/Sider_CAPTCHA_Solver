#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lite-HRNet-18+LiteFPN 训练主脚本
滑块验证码识别模型训练入口
"""
import argparse
import torch
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入核心模块
from src.models import create_lite_hrnet_18_fpn
from src.training.config_manager import ConfigManager
from src.training.data_pipeline import DataPipeline
from src.training.training_engine import TrainingEngine
from src.training.validator import Validator
from src.training.visualizer import Visualizer

import logging
import time
from datetime import datetime
import json


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Lite-HRNet-18+LiteFPN滑块验证码识别模型')
    
    # 基础参数
    parser.add_argument('--config', type=str, 
                       default='config/training_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--eval-only', action='store_true',
                       help='仅执行评估')
    
    # 数据参数
    parser.add_argument('--train-dir', type=str, 
                       default=None,  # 将从配置文件读取
                       help='训练数据目录（覆盖配置文件中的值）')
    parser.add_argument('--val-dir', type=str, 
                       default=None,  # 将从配置文件读取
                       help='验证数据目录（覆盖配置文件中的值）')
    
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


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # 确保可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def override_config(config: dict, args):
    """根据命令行参数覆盖配置"""
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
        # 更新所有路径
        config['checkpoints']['save_dir'] = f"checkpoints/{args.name}"
        config['logging']['log_dir'] = f"logs/log-{args.name}"
        config['logging']['tensorboard_dir'] = f"logs/tensorboard/{args.name}"
        logging.info(f"实验名称: {args.name}")


def save_checkpoint(model, engine, validator, epoch, config, is_best=False):
    """
    保存检查点
    
    Args:
        model: 模型
        engine: 训练引擎
        validator: 验证器
        epoch: 当前epoch
        config: 配置
        is_best: 是否为最佳模型
    """
    checkpoint_dir = Path(config['checkpoints']['save_dir'])
    
    # 获取检查点数据
    checkpoint = engine.save_checkpoint()
    
    # 添加额外信息
    checkpoint['epoch'] = epoch
    checkpoint['best_metric'] = validator.best_metric
    checkpoint['config'] = config
    
    # 保存当前epoch检查点
    if epoch % config['checkpoints'].get('save_interval', 1) == 0:
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"保存检查点: {checkpoint_path}")
    
    # 保存最新检查点
    latest_path = checkpoint_dir / "last.pth"
    torch.save(checkpoint, latest_path)
    
    # 保存最佳模型
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        logging.info(f"保存最佳模型: {best_path}")
        
        # 同时保存纯模型权重（用于推理）
        model_only_path = checkpoint_dir / "best_model_weights.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, model_only_path)


def load_checkpoint(checkpoint_path, model, engine, validator=None):
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
    logging.info(f"加载检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型和优化器状态
    engine.load_checkpoint(checkpoint)
    
    # 加载验证器状态
    if validator and 'best_metric' in checkpoint:
        validator.best_metric = checkpoint['best_metric']
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    logging.info(f"从epoch {start_epoch}恢复训练")
    
    return start_epoch


def train_epoch(model, engine, dataloader, epoch, visualizer):
    """
    训练一个epoch
    
    Args:
        model: 模型
        engine: 训练引擎
        dataloader: 数据加载器
        epoch: 当前epoch
        visualizer: 可视化器
    
    Returns:
        训练指标
    """
    logging.info(f"\n开始训练 Epoch {epoch}")
    
    # 执行训练
    train_metrics = engine.train_epoch(dataloader, epoch)
    
    # 记录到TensorBoard
    visualizer.log_training_metrics(train_metrics, epoch)
    
    # 记录权重直方图（每10个epoch）
    if epoch % 10 == 0:
        visualizer.log_histograms(model, epoch)
    
    return train_metrics


def validate_epoch(model, validator, dataloader, epoch, visualizer, use_ema=False):
    """
    验证一个epoch
    
    Args:
        model: 模型（或EMA模型）
        validator: 验证器
        dataloader: 数据加载器
        epoch: 当前epoch
        visualizer: 可视化器
        use_ema: 是否使用EMA模型
    
    Returns:
        验证指标
    """
    logging.info(f"开始验证 Epoch {epoch}")
    
    # 执行验证
    val_metrics = validator.validate(model, dataloader, epoch, use_ema)
    
    # 记录到TensorBoard
    visualizer.log_validation_metrics(val_metrics, epoch)
    
    # 记录失败案例
    failures = validator.get_failure_cases()
    if failures:
        visualizer.log_failure_cases(failures, epoch)
    
    return val_metrics


def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化配置管理器
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # 覆盖配置（如果有命令行参数）
    override_config(config, args)
    
    # 保存最终配置
    config_manager.save_config()
    
    # 获取设备
    device = config_manager.get_device()
    
    # 创建模型
    logging.info("创建模型...")
    model = create_lite_hrnet_18_fpn(config['model'])
    
    # 如果只是评估
    if args.eval_only:
        if not args.resume:
            raise ValueError("评估模式需要指定--resume检查点路径")
        
        # 加载模型
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # 创建数据管道
        processed_dir_eval = config.get('data', {}).get('processed_dir', 'data/processed')
        npy_val_index = Path(processed_dir_eval) / "val_index.json"
        
        if npy_val_index.exists():
            from src.training.npy_data_loader import NPYDataPipeline
            data_pipeline = NPYDataPipeline(config)
            data_pipeline.setup()
        else:
            data_pipeline = DataPipeline(config)
            val_path = config.get('data', {}).get('val_dir', 'data/split_for_training/val')
            data_pipeline.setup(None, val_path)
        
        # 创建验证器
        validator = Validator(config, device)
        
        # 执行验证
        val_metrics = validator.validate(model, data_pipeline.get_val_loader(), 0)
        
        # 打印结果
        print("\n评估结果:")
        for key, value in val_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        return
    
    # 获取数据路径
    processed_dir = config.get('data', {}).get('processed_dir', 'data/processed')
    
    # 检测数据格式
    # 1. 首先检查是否有NPY批次数据（新目录结构）
    npy_train_dir = Path(processed_dir) / "images" / "train"
    npy_val_dir = Path(processed_dir) / "images" / "val"
    # 也检查索引文件（兼容旧格式）
    npy_train_index = Path(processed_dir) / "train_index.json"
    npy_val_index = Path(processed_dir) / "val_index.json"
    # 或在split_info目录
    if not npy_train_index.exists():
        npy_train_index = Path(processed_dir) / "split_info" / "train_index.json"
    if not npy_val_index.exists():
        npy_val_index = Path(processed_dir) / "split_info" / "val_index.json"
    
    if npy_train_dir.exists() or npy_val_dir.exists() or npy_train_index.exists() or npy_val_index.exists():
        # 使用NPY批次数据加载器
        logging.info("检测到NPY批次格式数据，使用NPY数据管道...")
        from src.training.npy_data_loader import NPYDataPipeline
        
        config['data']['processed_dir'] = processed_dir
        data_pipeline = NPYDataPipeline(config)
        data_pipeline.setup()
        
        logging.info(f"使用NPY数据目录: {processed_dir}")
        if npy_train_dir.exists():
            logging.info(f"训练数据目录: {npy_train_dir} (新目录结构)")
        elif npy_train_index.exists():
            logging.info(f"训练索引: {npy_train_index} (索引文件模式)")
        if npy_val_dir.exists():
            logging.info(f"验证数据目录: {npy_val_dir} (新目录结构)")
        elif npy_val_index.exists():
            logging.info(f"验证索引: {npy_val_index} (索引文件模式)")
        
    # 2. 使用传统目录数据加载器作为备用
    else:
        logging.warning("未找到NPY批次数据索引文件")
        logging.info("使用目录格式数据管道...")
        
        # 从配置文件获取目录路径
        train_path = config.get('data', {}).get('train_dir', 'data/split_for_training/train')
        val_path = config.get('data', {}).get('val_dir', 'data/split_for_training/val')
        
        data_pipeline = DataPipeline(config)
        data_pipeline.setup(train_path, val_path)
        
        logging.info(f"使用训练数据目录: {train_path}")
        logging.info(f"使用验证数据目录: {val_path}")
    
    # 获取批次信息
    batch_info = data_pipeline.get_batch_info()
    logging.info(f"训练批次数: {batch_info['train_batches']}")
    logging.info(f"验证批次数: {batch_info['val_batches']}")
    
    # 创建训练引擎
    logging.info("创建训练引擎...")
    engine = TrainingEngine(model, config, device)
    
    # 创建验证器
    logging.info("创建验证器...")
    validator = Validator(config, device)
    
    # 创建可视化器
    logging.info("创建可视化器...")
    visualizer = Visualizer(config)
    
    # 记录模型结构（可选）
    try:
        dummy_input = torch.randn(1, 4, 256, 512).to(device)
        visualizer.log_model_graph(model, dummy_input)
    except:
        pass
    
    # 恢复训练（如果需要）
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, engine, validator)
    
    # 训练循环
    logging.info(f"\n{'='*60}")
    logging.info("开始训练")
    logging.info(f"{'='*60}")
    
    total_epochs = config['sched']['epochs']
    
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{total_epochs}")
        print(f"学习率: {engine.get_lr():.6f}")
        print(f"{'='*50}")
        
        # 训练阶段
        train_metrics = train_epoch(
            model, engine, 
            data_pipeline.get_train_loader(),
            epoch, visualizer
        )
        
        # 验证阶段
        val_metrics = validate_epoch(
            model, validator,
            data_pipeline.get_val_loader(),
            epoch, visualizer,
            use_ema=False
        )
        
        # 如果有EMA模型，也验证EMA
        if engine.ema_model is not None:
            logging.info("验证EMA模型...")
            ema_metrics = validate_epoch(
                engine.ema_model, validator,
                data_pipeline.get_val_loader(),
                epoch, visualizer,
                use_ema=True
            )
            # 记录EMA指标
            for key, value in ema_metrics.items():
                visualizer.writer.add_scalar(f'ema/{key}', value, epoch)
        
        # 更新学习率
        engine.step_scheduler()
        
        # 保存检查点
        save_checkpoint(
            model, engine, validator, epoch, config,
            is_best=val_metrics.get('is_best', False)
        )
        
        # 记录epoch时间
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch} 用时: {epoch_time:.2f}秒")
        
        # 打印关键指标
        print(f"\n训练损失: {train_metrics['loss']:.4f}")
        print(f"验证MAE: {val_metrics['mae_px']:.3f}px")
        print(f"验证Hit@2px: {val_metrics['hit_le_2px']:.2f}%")
        print(f"验证Hit@5px: {val_metrics['hit_le_5px']:.2f}%")
        
        # 早停检查
        if val_metrics.get('early_stop', False):
            logging.info("触发早停，结束训练")
            break
        
        # 刷新可视化
        visualizer.flush()
    
    # 训练结束
    logging.info(f"\n{'='*60}")
    logging.info("训练完成！")
    logging.info(f"{'='*60}")
    
    # 获取最佳指标
    best_metrics = validator.get_best_metrics()
    if best_metrics:
        logging.info(f"\n最佳模型 (Epoch {validator.best_epoch}):")
        logging.info(f"  MAE: {best_metrics['mae_px']:.3f}px")
        logging.info(f"  RMSE: {best_metrics['rmse_px']:.3f}px")
        logging.info(f"  Hit@1px: {best_metrics['hit_le_1px']:.2f}%")
        logging.info(f"  Hit@2px: {best_metrics['hit_le_2px']:.2f}%")
        logging.info(f"  Hit@5px: {best_metrics['hit_le_5px']:.2f}%")
    
    # 保存指标历史
    metrics_history_path = Path(config['checkpoints']['save_dir']) / 'metrics_history.json'
    validator.save_metrics_history(metrics_history_path)
    
    # 记录超参数
    hparams = {
        'batch_size': config['train']['batch_size'],
        'lr': config['optimizer']['lr'],
        'weight_decay': config['optimizer']['weight_decay'],
        'epochs': epoch,
        'seed': args.seed
    }
    
    final_metrics = {
        'best_mae': best_metrics['mae_px'] if best_metrics else 0,
        'best_hit_5px': best_metrics['hit_le_5px'] if best_metrics else 0
    }
    
    visualizer.log_hyperparameters(hparams, final_metrics)
    
    # 在测试集上评估（如果有测试集）
    test_file = config.get('data', {}).get('test_file', 'data/split_for_training/test.json')
    if Path(test_file).exists():
        logging.info("\n在测试集上评估最佳模型...")
        from src.training.test_evaluator import TestEvaluator
        
        # 加载最佳模型
        best_checkpoint_path = Path(config['checkpoints']['save_dir']) / "best_model.pth"
        if best_checkpoint_path.exists():
            # 创建测试评估器
            test_evaluator = TestEvaluator(
                test_file=test_file,
                processed_dir=config.get('data', {}).get('processed_dir', 'data/processed')
            )
            
            # 加载最佳模型
            best_model = create_lite_hrnet_18_fpn(config['model'])
            checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
            best_model.load_state_dict(checkpoint['model_state_dict'])
            
            # 评估
            test_metrics = test_evaluator.evaluate(best_model, device)
            
            # 记录到TensorBoard
            for key, value in test_metrics.items():
                visualizer.writer.add_scalar(f'test/{key}', value, epoch)
        else:
            logging.warning("未找到最佳模型检查点，跳过测试集评估")
    else:
        logging.info(f"测试集文件不存在: {test_file}，跳过测试集评估")
    
    # 关闭可视化器
    visualizer.close()
    
    logging.info(f"\n检查点保存在: {config['checkpoints']['save_dir']}")
    logging.info(f"TensorBoard日志: {config['logging']['tensorboard_dir']}")
    logging.info(f"运行以下命令查看TensorBoard:")
    logging.info(f"  tensorboard --logdir {config['logging']['tensorboard_dir']}")


if __name__ == "__main__":
    main()