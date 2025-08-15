#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理器 - 负责加载、验证和管理所有配置参数
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import logging
from datetime import datetime

class ConfigManager:
    """
    配置管理器 - 负责加载、验证和管理所有配置参数
    
    功能：
    1. 加载YAML配置文件
    2. 验证配置完整性
    3. 创建必要的目录结构
    4. 管理设备配置
    5. 提供配置访问接口
    """
    
    def __init__(self, config_path: str):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 加载配置
        self.config = self._load_config()
        
        # 验证配置
        self._validate_config()
        
        # 设置路径
        self._setup_paths()
        
        # 设置日志
        self._setup_logging()
        
        # 打印配置信息
        self._print_config_info()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Returns:
            配置字典
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 添加时间戳
        config['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return config
    
    def _validate_config(self):
        """验证配置完整性"""
        # 必需的顶级配置项
        required_keys = ['model', 'optimizer', 'sched', 'train', 'eval', 
                        'checkpoints', 'logging', 'hardware']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"缺少必需的配置项: {key}")
        
        # 验证模型配置
        if 'model_path' not in self.config['model']:
            raise ValueError("缺少 model.model_path 配置")
        
        # 验证优化器配置
        opt_cfg = self.config['optimizer']
        if 'lr' not in opt_cfg:
            raise ValueError("缺少 optimizer.lr 配置")
        if 'betas' not in opt_cfg:
            opt_cfg['betas'] = [0.9, 0.999]
        if 'eps' not in opt_cfg:
            opt_cfg['eps'] = 1e-8
        
        # 验证调度器配置
        sched_cfg = self.config['sched']
        if 'epochs' not in sched_cfg:
            raise ValueError("缺少 sched.epochs 配置")
        
        # 验证训练配置
        train_cfg = self.config['train']
        if 'batch_size' not in train_cfg:
            raise ValueError("缺少 train.batch_size 配置")
        
        # 验证评估配置
        eval_cfg = self.config['eval']
        if 'select_by' not in eval_cfg:
            eval_cfg['select_by'] = 'hit_le_5px'
        
        # 验证早停配置
        if 'early_stopping' in eval_cfg:
            es_cfg = eval_cfg['early_stopping']
            if 'min_epochs' not in es_cfg:
                es_cfg['min_epochs'] = 100
            if 'patience' not in es_cfg:
                es_cfg['patience'] = 18
            
            # 验证第二道防护
            if 'second_guard' in es_cfg:
                sg_cfg = es_cfg['second_guard']
                if 'metric' not in sg_cfg:
                    sg_cfg['metric'] = 'mae_px'
                if 'mode' not in sg_cfg:
                    sg_cfg['mode'] = 'min'
                if 'min_delta' not in sg_cfg:
                    sg_cfg['min_delta'] = 0.05
    
    def _setup_paths(self):
        """创建必要的目录结构"""
        # 检查点目录
        checkpoint_dir = Path(self.config['checkpoints']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志目录
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard目录
        if self.config['logging'].get('tensorboard', False):
            tb_dir = Path(self.config['logging']['tensorboard_dir'])
            tb_dir.mkdir(parents=True, exist_ok=True)
        
        # 添加到配置中（转换为绝对路径）
        self.config['checkpoints']['save_dir'] = str(checkpoint_dir.absolute())
        self.config['logging']['log_dir'] = str(log_dir.absolute())
        if self.config['logging'].get('tensorboard', False):
            self.config['logging']['tensorboard_dir'] = str(tb_dir.absolute())
    
    def _setup_logging(self):
        """设置日志系统"""
        log_dir = Path(self.config['logging']['log_dir'])
        log_file = log_dir / f"training_{self.config['timestamp']}.log"
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('ConfigManager')
        self.logger.info(f"日志文件: {log_file}")
    
    def _print_config_info(self):
        """打印配置信息"""
        print("\n" + "=" * 60)
        print("训练配置信息")
        print("=" * 60)
        
        print(f"模型: {self.config['model']['name']}")
        print(f"批次大小: {self.config['train']['batch_size']}")
        print(f"学习率: {self.config['optimizer']['lr']}")
        print(f"训练轮数: {self.config['sched']['epochs']}")
        print(f"混合精度: {self.config['train'].get('amp', 'none')}")
        
        # 早停配置
        if 'early_stopping' in self.config['eval']:
            es_cfg = self.config['eval']['early_stopping']
            print(f"早停配置:")
            print(f"  - 最小轮数: {es_cfg['min_epochs']}")
            print(f"  - 耐心值: {es_cfg['patience']}")
            
            if 'second_guard' in es_cfg:
                sg = es_cfg['second_guard']
                print(f"  - 第二防护: {sg['metric']} ({sg['mode']}, Δ>{sg['min_delta']})")
        
        print(f"检查点目录: {self.config['checkpoints']['save_dir']}")
        print(f"日志目录: {self.config['logging']['log_dir']}")
        
        if self.config['logging'].get('tensorboard', False):
            print(f"TensorBoard: {self.config['logging']['tensorboard_dir']}")
        
        print("=" * 60 + "\n")
    
    def get_device(self) -> torch.device:
        """
        获取训练设备
        
        Returns:
            torch.device对象
        """
        device_str = self.config['hardware']['device']
        
        if device_str == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
                
                # 应用硬件优化设置
                self.apply_hardware_optimizations()
            else:
                device = torch.device('cpu')
                self.logger.warning("CUDA不可用，使用CPU训练")
        else:
            device = torch.device('cpu')
            self.logger.info("使用CPU训练")
        
        return device
    
    def apply_hardware_optimizations(self):
        """
        应用硬件优化设置
        """
        hardware_cfg = self.config['hardware']
        
        # CuDNN benchmark
        if hardware_cfg.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
            self.logger.info("CuDNN benchmark enabled")
        
        # CuDNN deterministic
        if not hardware_cfg.get('cudnn_deterministic', False):
            torch.backends.cudnn.deterministic = False
            self.logger.info("CuDNN non-deterministic mode enabled (faster)")
        else:
            torch.backends.cudnn.deterministic = True
            self.logger.info("CuDNN deterministic mode enabled (reproducible)")
        
        # TF32 acceleration (for Ampere GPUs and above)
        if hardware_cfg.get('allow_tf32', True):
            # Check if GPU supports TF32 (Ampere architecture or newer)
            device_capability = torch.cuda.get_device_capability()
            if device_capability[0] >= 8:  # Ampere is compute capability 8.x
                # Enable TF32 for matrix multiplication
                if hardware_cfg.get('matmul_allow_tf32', True):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    self.logger.info("TF32 enabled for matrix multiplication")
                
                # Enable TF32 for convolution
                if hardware_cfg.get('cudnn_allow_tf32', True):
                    torch.backends.cudnn.allow_tf32 = True
                    self.logger.info("TF32 enabled for convolution operations")
            else:
                self.logger.info(f"GPU compute capability {device_capability[0]}.{device_capability[1]} does not support TF32")
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """获取优化器配置"""
        return self.config['optimizer'].copy()
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """获取调度器配置"""
        return self.config['sched'].copy()
    
    def get_train_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config['train'].copy()
    
    def get_eval_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return self.config['eval'].copy()
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """获取检查点配置"""
        return self.config['checkpoints'].copy()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config['logging'].copy()
    
    def save_config(self, path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            path: 保存路径，如果为None则保存到检查点目录
        """
        if path is None:
            checkpoint_dir = Path(self.config['checkpoints']['save_dir'])
            path = checkpoint_dir / f"config_{self.config['timestamp']}.yaml"
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"配置已保存到: {path}")
    
    def __getitem__(self, key: str) -> Any:
        """允许通过索引访问配置"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """检查配置项是否存在"""
        return key in self.config


if __name__ == "__main__":
    # 测试配置管理器
    config_path = "config/training_config.yaml"
    
    try:
        config_manager = ConfigManager(config_path)
        
        # 获取设备
        device = config_manager.get_device()
        print(f"设备: {device}")
        
        # 获取各种配置
        opt_config = config_manager.get_optimizer_config()
        print(f"\n优化器配置: {opt_config}")
        
        # 保存配置副本
        config_manager.save_config()
        
        print("\n配置管理器测试通过！")
        
    except Exception as e:
        print(f"错误: {e}")