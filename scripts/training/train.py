#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 解决OpenMP冲突问题 - 必须在导入numpy/torch之前设置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
Lite-HRNet-18+LiteFPN 训练主脚本
滑块验证码识别模型训练入口
"""
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入核心模块
from src.training.config_manager import ConfigManager
from src.training.utils import (
    parse_training_args,
    set_global_seed,
    override_config_from_args
)
from src.training.training_loop import run_training_loop
from src.training.training_setup import (
    setup_model,
    setup_data_pipeline,
    setup_training_components,
    handle_eval_only,
    print_training_summary,
    evaluate_on_test_set,
    log_hyperparameters
)
import logging


def setup_environment(args):
    """
    设置训练环境
    
    Args:
        args: 命令行参数
    
    Returns:
        tuple: (config, device)
    """
    # 设置随机种子
    set_global_seed(args.seed)
    
    # 初始化配置管理器
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # 覆盖配置（如果有命令行参数）
    override_config_from_args(config, args)
    
    # 保存最终配置
    config_manager.save_config()
    
    # 获取设备
    device = config_manager.get_device()
    
    # 应用硬件优化（TF32、cuDNN等）
    if device.type == 'cuda':
        config_manager.apply_hardware_optimizations()
        logging.info("硬件优化已应用（TF32、cuDNN自动调优）")
    
    return config, device


def launch_tensorboard_if_configured(config):
    """
    启动TensorBoard（如果配置启用）
    
    Args:
        config: 配置字典
    """
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


def prepare_training(model, engine, validator, checkpoint_manager, args):
    """
    准备训练，处理恢复训练或新训练的初始化
    
    Args:
        model: 模型
        engine: 训练引擎
        validator: 验证器
        checkpoint_manager: 检查点管理器
        args: 命令行参数
    
    Returns:
        start_epoch: 起始epoch
    """
    start_epoch = 1
    if args.resume:
        start_epoch = checkpoint_manager.load_checkpoint(args.resume, model, engine, validator)
    # EMA已移除
    
    return start_epoch


def finalize_training(model, engine, validator, data_pipeline, 
                     checkpoint_manager, visualizer, config, args, device):
    """
    完成训练后的处理
    
    Args:
        model: 模型
        engine: 训练引擎
        validator: 验证器
        data_pipeline: 数据管道
        checkpoint_manager: 检查点管理器
        visualizer: 可视化器
        config: 配置字典
        args: 命令行参数
        device: 训练设备
    """
    # 训练结束
    logging.info("训练完成！")
    
    # 打印训练总结
    print_training_summary(validator, config)
    
    # 保存最终的最佳模型
    final_epoch = config['sched']['epochs']
    checkpoint_manager.save_final_best_model(
        model=model,
        engine=engine,
        validator=validator,
        data_pipeline=data_pipeline,
        config=config,
        epoch=final_epoch
    )
    
    # 记录超参数
    log_hyperparameters(validator, visualizer, config, args)
    
    # 在测试集上评估（如果有测试集）
    evaluate_on_test_set(model, config, device, data_pipeline)
    
    # 关闭可视化器
    visualizer.close()


def main():
    """主训练函数"""
    # 解析参数
    args = parse_training_args()
    
    # 设置环境
    config, device = setup_environment(args)
    
    # 如果只是评估模式
    if args.eval_only:
        handle_eval_only(args, config, device)
        return
    
    # 创建模型
    model = setup_model(config, device)
    
    # 设置数据管道
    data_pipeline = setup_data_pipeline(config)
    
    # 设置训练组件
    engine, validator, visualizer, checkpoint_manager = setup_training_components(
        model, config, device
    )
    
    # 自动启动TensorBoard（如果配置启用）
    launch_tensorboard_if_configured(config)
    
    # 准备训练（处理恢复训练或新训练）
    start_epoch = prepare_training(model, engine, validator, checkpoint_manager, args)
    
    # 运行训练循环
    logging.info("开始训练")
    run_training_loop(
        model=model,
        engine=engine,
        validator=validator,
        data_pipeline=data_pipeline,
        visualizer=visualizer,
        checkpoint_manager=checkpoint_manager,
        config=config,
        start_epoch=start_epoch
    )
    
    # 完成训练后的处理
    finalize_training(
        model=model,
        engine=engine,
        validator=validator,
        data_pipeline=data_pipeline,
        checkpoint_manager=checkpoint_manager,
        visualizer=visualizer,
        config=config,
        args=args,
        device=device
    )
    
    # 打印最终信息
    logging.info(f"\n检查点保存在: {config['checkpoints']['save_dir']}")
    logging.info(f"TensorBoard日志: {config['logging']['tensorboard_dir']}")
    logging.info(f"运行以下命令查看TensorBoard:")
    logging.info(f"  tensorboard --logdir {config['logging']['tensorboard_dir']}")


if __name__ == "__main__":
    main()