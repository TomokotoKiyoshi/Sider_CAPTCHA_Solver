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
from src.training.training_engine import TrainingEngine
from src.training.validator import Validator
from src.training.visualizer import Visualizer
import logging



# 限制批次数量的DataLoader包装器（用于测试或部分数据训练）
class LimitedDataLoader:
    """限制批次数量的DataLoader包装器"""
    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = max_batches
        
    def __iter__(self):
        count = 0
        for batch in self.loader:
            if count >= self.max_batches:
                break
            yield batch
            count += 1
            
    def __len__(self):
        return self.max_batches


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
    # Note: CuDNN settings are now handled by config_manager.apply_hardware_optimizations()


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
        # 使用name参数作为子目录后缀
        base_checkpoint_dir = config['checkpoints']['save_dir'].rsplit('/', 1)[0]
        base_log_dir = config['logging']['log_dir'].rsplit('-', 1)[0]
        base_tb_dir = config['logging']['tensorboard_dir'].rsplit('/', 1)[0]
        
        config['checkpoints']['save_dir'] = f"{base_checkpoint_dir}/{args.name}"
        config['logging']['log_dir'] = f"{base_log_dir}-{args.name}"
        config['logging']['tensorboard_dir'] = f"{base_tb_dir}/{args.name}"
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
    
    # 如果有可视化数据，记录预测和热力图
    if 'vis_data' in val_metrics:
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
    
    # 应用硬件优化（TF32、cuDNN等）
    if device.type == 'cuda':
        config_manager.apply_hardware_optimizations()
        logging.info("硬件优化已应用（TF32、cuDNN自动调优）")
    
    # 创建模型
    logging.info("创建模型...")
    model = create_lite_hrnet_18_fpn(config['model'])
    
    # 启用torch.compile优化（PyTorch 2.0+）
    if config['train'].get('compile_model', False):
        try:
            compile_mode = config['train'].get('compile_mode', 'default')
            logging.info(f"正在编译模型，模式: {compile_mode}")
            model = torch.compile(model, mode=compile_mode)
            logging.info("模型编译成功，将显著提升训练速度")
        except Exception as e:
            logging.warning(f"模型编译失败: {e}，将使用未编译模型继续训练")
    
    # 如果只是评估
    if args.eval_only:
        if not args.resume:
            raise ValueError("评估模式需要指定--resume检查点路径")
        
        # 加载模型
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # 创建数据管道 (仅支持NPY格式)
        from src.training.npy_data_loader import NPYDataPipeline
        processed_dir_eval = config.get('data', {}).get('processed_dir', 'data/processed')
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
        
        return
    
    # 获取数据路径
    processed_dir = config.get('data', {}).get('processed_dir', 'data/processed')
    
    # 使用NPY批次数据加载器（唯一支持的格式）
    logging.info("加载NPY批次格式数据...")
    from src.training.npy_data_loader import NPYDataPipeline
    
    # 检查数据目录是否存在
    npy_train_dir = Path(processed_dir) / "images" / "train"
    npy_val_dir = Path(processed_dir) / "images" / "val"
    # 也检查索引文件
    npy_train_index = Path(processed_dir) / "train_index.json"
    npy_val_index = Path(processed_dir) / "val_index.json"
    # 或在split_info目录
    if not npy_train_index.exists():
        npy_train_index = Path(processed_dir) / "split_info" / "train_index.json"
    if not npy_val_index.exists():
        npy_val_index = Path(processed_dir) / "split_info" / "val_index.json"
    
    # 验证数据存在
    if not (npy_train_dir.exists() or npy_train_index.exists()):
        raise FileNotFoundError(
            f"未找到训练数据！请先运行预处理脚本生成NPY格式数据。\n"
            f"期望路径: {npy_train_dir} 或 {npy_train_index}"
        )
    
    config['data']['processed_dir'] = processed_dir
    data_pipeline = NPYDataPipeline(config)
    data_pipeline.setup()
    
    logging.info(f"使用NPY数据目录: {processed_dir}")
    if npy_train_dir.exists():
        logging.info(f"训练数据目录: {npy_train_dir}")
    elif npy_train_index.exists():
        logging.info(f"训练索引: {npy_train_index}")
    if npy_val_dir.exists():
        logging.info(f"验证数据目录: {npy_val_dir}")
    elif npy_val_index.exists():
        logging.info(f"验证索引: {npy_val_index}")
    
    # 获取批次信息
    batch_info = data_pipeline.get_batch_info()
    logging.info(f"训练批次数: {batch_info['train_batches']}")
    logging.info(f"验证批次数: {batch_info['val_batches']}")
    
    # 创建可视化器（先创建，以便传递给训练引擎）
    logging.info("创建可视化器...")
    visualizer = Visualizer(config)
    
    # 创建训练引擎（传入visualizer）
    logging.info("创建训练引擎...")
    engine = TrainingEngine(model, config, device, visualizer)
    
    # 创建验证器
    logging.info("创建验证器...")
    validator = Validator(config, device)
    
    # 记录模型结构（可选）
    try:
        dummy_input = torch.randn(1, 4, 256, 512).to(device)
        visualizer.log_model_graph(model, dummy_input)
    except:
        pass
    
    # 自动启动TensorBoard（如果配置启用）
    if config.get('logging', {}).get('auto_launch_tensorboard', False):
        tensorboard_dir = config['logging']['tensorboard_dir']
        tensorboard_port = config['logging'].get('tensorboard_port', 6006)
        
        # 启动TensorBoard进程
        import subprocess
        import threading
        import socket
        
        def is_port_in_use(port):
            """检查端口是否已被占用"""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    return False
                except:
                    return True
        
        def launch_tensorboard():
            try:
                # Check if port is already in use
                if is_port_in_use(tensorboard_port):
                    logging.info(f"Port {tensorboard_port} is already in use, TensorBoard may be running")
                    print(f"\n{'='*60}")
                    print(f"INFO: TensorBoard may already be running!")
                    print(f"URL: http://localhost:{tensorboard_port}")
                    print(f"{'='*60}\n")
                else:
                    logging.info(f"Launching TensorBoard on port {tensorboard_port}...")
                    cmd = f"tensorboard --logdir {tensorboard_dir} --port {tensorboard_port} --bind_all"
                    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    logging.info(f"TensorBoard started! Access at: http://localhost:{tensorboard_port}")
                    print(f"\n{'='*60}")
                    print(f"TensorBoard auto-launched successfully!")
                    print(f"URL: http://localhost:{tensorboard_port}")
                    print(f"Note: TensorBoard will continue running after training stops")
                    print(f"{'='*60}\n")
            except Exception as e:
                logging.warning(f"Failed to auto-launch TensorBoard: {e}")
                logging.info(f"Please run manually: tensorboard --logdir {tensorboard_dir}")
        
        # 在后台线程中启动TensorBoard
        tb_thread = threading.Thread(target=launch_tensorboard, daemon=True)
        tb_thread.start()
        
        # 等待一下让TensorBoard启动
        import time
        time.sleep(2)
    
    # 恢复训练（如果需要）
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, engine, validator)
    
    # 训练循环
    logging.info(f"\n{'='*60}")
    logging.info("开始训练")
    logging.info(f"{'='*60}")
    
    total_epochs = config['sched']['epochs']
    
    # 检查是否显示进度条
    show_progress = config.get('logging', {}).get('time_tracking', {}).get('show_progress_bar', False)
    
    if show_progress:
        from tqdm import tqdm
        epoch_iterator = tqdm(range(start_epoch, total_epochs + 1), 
                             desc="Training Progress", 
                             unit="epoch",
                             ncols=100)
    else:
        epoch_iterator = range(start_epoch, total_epochs + 1)
    
    for epoch in epoch_iterator:
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
        
        # 记录时间指标到TensorBoard
        visualizer.log_time_metrics(epoch, epoch_time)
        
        # 获取ETA字符串
        eta_string = visualizer.get_eta_string(epoch)
        
        # 打印关键指标（包含ETA）
        print(f"\n训练损失: {train_metrics['loss']:.4f}")
        print(f"验证MAE: {val_metrics['mae_px']:.3f}px")
        print(f"验证Hit@2px: {val_metrics['hit_le_2px']:.2f}%")
        print(f"验证Hit@5px: {val_metrics['hit_le_5px']:.2f}%")
        print(f"预计剩余时间: {eta_string}")
        
        # 更新进度条描述（如果使用tqdm）
        if show_progress and hasattr(epoch_iterator, 'set_postfix'):
            epoch_iterator.set_postfix({
                'Loss': f"{train_metrics['loss']:.4f}",
                'MAE': f"{val_metrics['mae_px']:.2f}px",
                'ETA': eta_string
            })
        
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
    
    # 在测试集上评估（如果有测试集NPY数据）
    test_dir = Path(processed_dir) / "images" / "test"
    test_index = Path(processed_dir) / "test_index.json"
    if not test_index.exists():
        test_index = Path(processed_dir) / "split_info" / "test_index.json"
    
    if test_dir.exists() or test_index.exists():
        logging.info("\n在测试集上评估最佳模型...")
        
        # 加载最佳模型
        best_checkpoint_path = Path(config['checkpoints']['save_dir']) / "best_model.pth"
        if best_checkpoint_path.exists():
            # 创建测试数据加载器
            test_config = config.copy()
            test_pipeline = NPYDataPipeline(test_config)
            # 设置为测试模式
            test_loader = test_pipeline.get_test_loader() if hasattr(test_pipeline, 'get_test_loader') else None
            
            if test_loader:
                # 加载最佳模型
                best_model = create_lite_hrnet_18_fpn(config['model'])
                checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
                best_model.load_state_dict(checkpoint['model_state_dict'])
                best_model = best_model.to(device)
                
                # 评估
                test_validator = Validator(config, device)
                test_metrics = test_validator.validate(best_model, test_loader, 0)
                
                # 记录到TensorBoard
                for key, value in test_metrics.items():
                    if isinstance(value, (int, float)):
                        visualizer.writer.add_scalar(f'test/{key}', value, epoch)
                
                logging.info(f"测试集MAE: {test_metrics.get('mae_px', 0):.3f}px")
                logging.info(f"测试集Hit@2px: {test_metrics.get('hit_le_2px', 0):.2f}%")
            else:
                logging.info("测试数据加载器不可用")
        else:
            logging.warning("未找到最佳模型检查点，跳过测试集评估")
    else:
        logging.info("未找到测试集NPY数据，跳过测试集评佐")
    
    # 关闭可视化器
    visualizer.close()
    
    logging.info(f"\n检查点保存在: {config['checkpoints']['save_dir']}")
    logging.info(f"TensorBoard日志: {config['logging']['tensorboard_dir']}")
    logging.info(f"运行以下命令查看TensorBoard:")
    logging.info(f"  tensorboard --logdir {config['logging']['tensorboard_dir']}")


if __name__ == "__main__":
    main()