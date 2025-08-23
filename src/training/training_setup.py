#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练设置模块
包含模型创建、数据管道设置等训练准备工作
"""
import torch
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any


def setup_model(config: Dict, device: torch.device):
    """
    创建并设置模型
    
    Args:
        config: 配置字典
        device: 训练设备
    
    Returns:
        model: 配置好的模型
    """
    from src.models import create_lite_hrnet_18_fpn
    
    logging.info("创建模型...")
    model = create_lite_hrnet_18_fpn()
    
    # 启用torch.compile优化（PyTorch 2.0+）
    if config['train'].get('compile_model', False):
        try:
            compile_mode = config['train'].get('compile_mode', 'default')
            logging.info(f"正在编译模型，模式: {compile_mode}")
            model = torch.compile(model, mode=compile_mode)
            logging.info("模型编译成功，将显著提升训练速度")
        except Exception as e:
            logging.warning(f"模型编译失败: {e}，将使用未编译模型继续训练")
    
    return model


def setup_data_pipeline(config: Dict):
    """
    设置数据管道
    
    Args:
        config: 配置字典
    
    Returns:
        data_pipeline: 数据管道对象
    """
    from src.training.npy_data_loader import NPYDataPipeline
    
    processed_dir = config.get('data', {}).get('processed_dir', 'data/processed')
    
    # 使用NPY批次数据加载器（唯一支持的格式）
    logging.info("加载NPY批次格式数据...")
    
    # 检查数据目录是否存在
    npy_train_dir = Path(processed_dir) / "images" / "train"
    npy_val_dir = Path(processed_dir) / "images" / "val"
    
    # 检查索引文件
    npy_train_index = Path(processed_dir) / "train_index.json"
    npy_val_index = Path(processed_dir) / "val_index.json"
    
    # 或在split_info目录
    if not npy_train_index.exists():
        npy_train_index = Path(processed_dir) / "split_info" / "train_index.json"
    if not npy_val_index.exists():
        npy_val_index = Path(processed_dir) / "split_info" / "val_index.json"
    
    if not npy_train_dir.exists() or not npy_val_dir.exists():
        raise FileNotFoundError(
            f"NPY数据目录不存在!\n"
            f"请确保以下目录存在:\n"
            f"  - {npy_train_dir}\n"
            f"  - {npy_val_dir}\n"
            f"请先运行: python scripts/preprocessing/preprocess_captchas.py"
        )
    
    if not npy_train_index.exists() or not npy_val_index.exists():
        logging.warning(
            f"索引文件不存在:\n"
            f"  - {npy_train_index}: {'存在' if npy_train_index.exists() else '不存在'}\n"
            f"  - {npy_val_index}: {'存在' if npy_val_index.exists() else '不存在'}\n"
            f"将尝试直接加载NPY文件..."
        )
    
    # 创建数据管道
    data_pipeline = NPYDataPipeline(config)
    data_pipeline.setup()
    
    # 记录数据集信息
    logging.info(f"训练集批次数: {len(data_pipeline.get_train_loader())}")
    logging.info(f"验证集批次数: {len(data_pipeline.get_val_loader())}")
    
    return data_pipeline


def setup_training_components(model, config: Dict, device: torch.device):
    """
    设置训练组件（引擎、验证器、可视化器等）
    
    Args:
        model: 模型
        config: 配置字典
        device: 训练设备
    
    Returns:
        tuple: (engine, validator, visualizer, checkpoint_manager)
    """
    from src.training.training_engine import TrainingEngine
    from src.training.validator import Validator
    from src.training.visualizer import Visualizer
    from src.training.checkpoint_manager import CheckpointManager
    
    # 创建可视化器
    logging.info("创建可视化器...")
    visualizer = Visualizer(config)
    
    # 创建训练引擎（传入visualizer）
    logging.info("创建训练引擎...")
    engine = TrainingEngine(model, config, device, visualizer)
    
    # 创建验证器
    logging.info("创建验证器...")
    validator = Validator(config, device)
    
    # 创建检查点管理器
    logging.info("创建检查点管理器...")
    checkpoint_manager = CheckpointManager(config['checkpoints']['save_dir'])
    
    # 记录模型结构（可选）
    try:
        dummy_input = torch.randn(1, 4, 256, 512).to(device)
        visualizer.log_model_graph(model, dummy_input)
    except Exception as e:
        logging.debug(f"无法记录模型结构图: {e}")
    
    return engine, validator, visualizer, checkpoint_manager


def handle_eval_only(args, config: Dict, device: torch.device):
    """
    处理仅评估模式
    
    Args:
        args: 命令行参数
        config: 配置字典
        device: 训练设备
    """
    from src.training.validator import Validator
    from src.training.npy_data_loader import NPYDataPipeline
    from src.models import create_lite_hrnet_18_fpn
    
    if not args.resume:
        raise ValueError("评估模式需要指定--resume检查点路径")
    
    # 加载模型
    model = create_lite_hrnet_18_fpn()
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 创建数据管道
    data_pipeline = NPYDataPipeline(config)
    data_pipeline.setup()
    
    # 创建验证器
    validator = Validator(config, device)
    
    # 执行验证
    val_metrics = validator.validate(model, data_pipeline.get_val_loader(), 0)
    
    # 打印结果
    print("\n评估结果:")
    for key, value in val_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")


def print_training_summary(validator, config: Dict):
    """
    打印训练总结
    
    Args:
        validator: 验证器
        config: 配置字典
    """
    best_metrics = validator.get_best_metrics()
    if best_metrics:
        logging.info(f"\n最佳模型 (Epoch {validator.best_epoch}):")
        logging.info(f"  MAE: {best_metrics['mae_px']:.3f}px")
        logging.info(f"  RMSE: {best_metrics['rmse_px']:.3f}px")
        logging.info(f"  Hit@2px: {best_metrics['hit_le_2px']:.2f}%")
        logging.info(f"  Hit@5px: {best_metrics['hit_le_5px']:.2f}%")


def evaluate_on_test_set(model, config: Dict, device: torch.device, data_pipeline):
    """
    在测试集上评估模型
    
    Args:
        model: 模型
        config: 配置字典
        device: 训练设备
        data_pipeline: 数据管道
    """
    from src.training.validator import Validator
    
    # 检查是否有测试集
    test_loader = None
    if hasattr(data_pipeline, 'test_files') and data_pipeline.test_files:
        test_loader = data_pipeline.get_test_loader()
    
    if test_loader:
        logging.info("\n在测试集上评估...")
        test_validator = Validator(config, device)
        test_metrics = test_validator.validate(model, test_loader, 0)
        
        print("\n测试集结果:")
        print(f"  MAE: {test_metrics['mae_px']:.3f}px")
        print(f"  RMSE: {test_metrics['rmse_px']:.3f}px")
        print(f"  Hit@2px: {test_metrics['hit_le_2px']:.2f}%")
        print(f"  Hit@5px: {test_metrics['hit_le_5px']:.2f}%")
    else:
        logging.info("没有找到测试集NPY数据，跳过测试集评估")


def log_hyperparameters(validator, visualizer, config: Dict, args):
    """
    记录超参数到TensorBoard
    
    Args:
        validator: 验证器
        visualizer: 可视化器
        config: 配置字典
        args: 命令行参数
    """
    best_metrics = validator.get_best_metrics()
    
    hparams = {
        'batch_size': config['train']['batch_size'],
        'lr': config['optimizer']['lr'],
        'weight_decay': config['optimizer']['weight_decay'],
        'epochs': config['sched']['epochs'],
        'seed': args.seed
    }
    
    final_metrics = {
        'best_mae': best_metrics['mae_px'] if best_metrics else 0,
        'best_hit_5px': best_metrics['hit_le_5px'] if best_metrics else 0
    }
    
    visualizer.log_hyperparameters(hparams, final_metrics)