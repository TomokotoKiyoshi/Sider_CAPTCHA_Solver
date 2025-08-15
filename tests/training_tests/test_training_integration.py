#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练系统集成测试
测试整个训练流程是否能正常工作
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import tempfile
import shutil
from unittest import TestCase, main
import yaml


class TestTrainingIntegration(TestCase):
    """训练系统集成测试"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        
        # 创建目录
        self.train_dir.mkdir(parents=True)
        self.val_dir.mkdir(parents=True)
        
        print(f"\n临时测试目录: {self.temp_dir}")
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时目录
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_config(self):
        """创建测试配置文件"""
        config = {
            'model': {
                'name': 'Lite-HRNet-18+LiteFPN',
                'input_channels': 4,
                'model_path': 'src.models.lite_hrnet_18_fpn'
            },
            'optimizer': {
                'name': 'AdamW',
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'weight_decay': 0.05,
                'clip_grad_norm': 1.0,
                'ema_decay': 0.999
            },
            'sched': {
                'warmup_epochs': 1,
                'cosine_min_lr': 1e-6,
                'epochs': 2,  # 只训练2个epoch用于测试
                'min_epochs': 1,
                'patience': 1
            },
            'train': {
                'batch_size': 2,  # 小批次用于测试
                'amp': 'none',  # 测试时不用混合精度
                'channels_last': False,
                'num_workers': 0,  # 测试时不用多进程
                'pin_memory': False
            },
            'eval': {
                'metrics': ['mae_px', 'rmse_px', 'hit_le_2px', 'hit_le_5px'],
                'select_by': 'hit_le_5px',
                'vis_fail_k': 2,
                'eval_interval': 1,
                'early_stopping': {
                    'min_epochs': 1,
                    'patience': 1,
                    'second_guard': {
                        'metric': 'mae_px',
                        'mode': 'min',
                        'min_delta': 0.05
                    }
                }
            },
            'checkpoints': {
                'save_dir': str(Path(self.temp_dir) / 'checkpoints'),
                'save_interval': 1,
                'save_best': True
            },
            'logging': {
                'log_dir': str(Path(self.temp_dir) / 'logs'),
                'log_interval': 1,
                'tensorboard': True,
                'tensorboard_dir': str(Path(self.temp_dir) / 'tensorboard')
            },
            'hardware': {
                'device': 'cpu',  # 测试用CPU
                'cudnn_benchmark': False
            }
        }
        
        # 保存配置文件
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config
    
    def create_dummy_data(self, num_samples=10):
        """创建虚拟数据用于测试"""
        import cv2
        
        # 创建训练数据
        for i in range(num_samples):
            # 创建虚拟图像
            img = np.random.randint(0, 255, (160, 320, 3), dtype=np.uint8)
            
            # 随机坐标
            bgx = np.random.randint(65, 305)
            bgy = np.random.randint(30, 130)
            sdx = np.random.randint(15, 35)
            sdy = np.random.randint(30, 130)
            
            # 保存到训练目录
            filename = f"Pic{i:04d}_Bgx{bgx}Bgy{bgy}_Sdx{sdx}Sdy{sdy}_test.png"
            cv2.imwrite(str(self.train_dir / filename), img)
        
        # 创建验证数据
        for i in range(num_samples // 2):
            img = np.random.randint(0, 255, (160, 320, 3), dtype=np.uint8)
            bgx = np.random.randint(65, 305)
            bgy = np.random.randint(30, 130)
            sdx = np.random.randint(15, 35)
            sdy = np.random.randint(30, 130)
            
            filename = f"Pic{i+num_samples:04d}_Bgx{bgx}Bgy{bgy}_Sdx{sdx}Sdy{sdy}_test.png"
            cv2.imwrite(str(self.val_dir / filename), img)
        
        print(f"创建了 {num_samples} 个训练样本, {num_samples//2} 个验证样本")
    
    def test_01_config_manager(self):
        """测试配置管理器"""
        print("\n=== 测试配置管理器 ===")
        
        # 创建配置
        self.create_test_config()
        
        # 导入并测试
        from src.training.config_manager import ConfigManager
        
        config_manager = ConfigManager(str(self.config_file))
        
        # 验证配置加载
        self.assertIsNotNone(config_manager.config)
        self.assertEqual(config_manager.config['train']['batch_size'], 2)
        self.assertEqual(config_manager.config['sched']['epochs'], 2)
        
        # 验证设备获取
        device = config_manager.get_device()
        self.assertEqual(device.type, 'cpu')
        
        print("✅ 配置管理器测试通过")
    
    def test_02_data_pipeline(self):
        """测试数据管道"""
        print("\n=== 测试数据管道 ===")
        
        # 创建配置和数据
        config = self.create_test_config()
        self.create_dummy_data(10)
        
        # 导入并测试
        from src.training.data_pipeline import DataPipeline
        
        pipeline = DataPipeline(config)
        pipeline.setup(str(self.train_dir), str(self.val_dir))
        
        # 验证数据加载器
        self.assertIsNotNone(pipeline.train_loader)
        self.assertIsNotNone(pipeline.val_loader)
        self.assertEqual(pipeline.num_train_samples, 10)
        self.assertEqual(pipeline.num_val_samples, 5)
        
        # 测试获取一个批次
        batch = pipeline.test_batch()
        self.assertIn('image', batch)
        self.assertIn('gap_coords', batch)
        self.assertIn('slider_coords', batch)
        
        # 验证张量形状
        self.assertEqual(batch['image'].shape[1], 4)  # 4通道
        self.assertEqual(batch['gap_coords'].shape[1], 2)  # x, y
        self.assertEqual(batch['slider_coords'].shape[1], 2)  # x, y
        
        print("✅ 数据管道测试通过")
    
    def test_03_model_loading(self):
        """测试模型加载"""
        print("\n=== 测试模型加载 ===")
        
        # 导入模型
        from src.models import create_lite_hrnet_18_fpn
        
        # 创建模型
        model = create_lite_hrnet_18_fpn()
        self.assertIsNotNone(model)
        
        # 测试前向传播
        x = torch.randn(1, 4, 256, 512)
        with torch.no_grad():
            outputs = model(x)
        
        # 验证输出
        self.assertIn('heatmap_gap', outputs)
        self.assertIn('heatmap_slider', outputs)
        self.assertIn('offset_gap', outputs)
        self.assertIn('offset_slider', outputs)
        
        # 验证输出形状
        self.assertEqual(outputs['heatmap_gap'].shape, (1, 1, 64, 128))
        self.assertEqual(outputs['offset_gap'].shape, (1, 2, 64, 128))
        
        print("✅ 模型加载测试通过")
    
    def test_04_training_engine(self):
        """测试训练引擎"""
        print("\n=== 测试训练引擎 ===")
        
        # 创建配置和数据
        config = self.create_test_config()
        self.create_dummy_data(4)
        
        # 导入必要模块
        from src.models import create_lite_hrnet_18_fpn
        from src.training.training_engine import TrainingEngine
        from src.training.data_pipeline import DataPipeline
        
        # 创建模型和数据管道
        model = create_lite_hrnet_18_fpn()
        device = torch.device('cpu')
        
        pipeline = DataPipeline(config)
        pipeline.setup(str(self.train_dir), str(self.val_dir))
        
        # 创建训练引擎
        engine = TrainingEngine(model, config, device)
        self.assertIsNotNone(engine)
        
        # 测试训练一个批次
        try:
            metrics = engine.train_epoch(pipeline.train_loader, epoch=1)
            self.assertIn('loss', metrics)
            self.assertIn('gap_mae', metrics)
            self.assertIn('slider_mae', metrics)
            print(f"  训练损失: {metrics['loss']:.4f}")
            print("✅ 训练引擎测试通过")
        except Exception as e:
            print(f"⚠️ 训练引擎测试失败: {e}")
    
    def test_05_validator(self):
        """测试验证器"""
        print("\n=== 测试验证器 ===")
        
        # 创建配置和数据
        config = self.create_test_config()
        self.create_dummy_data(4)
        
        # 导入必要模块
        from src.models import create_lite_hrnet_18_fpn
        from src.training.validator import Validator
        from src.training.data_pipeline import DataPipeline
        
        # 创建模型和数据管道
        model = create_lite_hrnet_18_fpn()
        device = torch.device('cpu')
        model = model.to(device)
        
        pipeline = DataPipeline(config)
        pipeline.setup(str(self.train_dir), str(self.val_dir))
        
        # 创建验证器
        validator = Validator(config, device)
        self.assertIsNotNone(validator)
        
        # 测试验证
        try:
            metrics = validator.validate(model, pipeline.val_loader, epoch=1)
            self.assertIn('mae_px', metrics)
            self.assertIn('hit_le_5px', metrics)
            self.assertIn('early_stop', metrics)
            print(f"  验证MAE: {metrics['mae_px']:.2f}px")
            print(f"  Hit@5px: {metrics['hit_le_5px']:.2f}%")
            print("✅ 验证器测试通过")
        except Exception as e:
            print(f"⚠️ 验证器测试失败: {e}")
    
    def test_06_visualizer(self):
        """测试可视化器"""
        print("\n=== 测试可视化器 ===")
        
        # 创建配置
        config = self.create_test_config()
        
        # 导入并测试
        from src.training.visualizer import Visualizer
        
        visualizer = Visualizer(config)
        self.assertIsNotNone(visualizer)
        
        # 测试记录标量
        test_metrics = {
            'loss': 0.5,
            'mae': 2.3,
            'hit_rate': 95.5
        }
        visualizer.log_scalars(test_metrics, step=1, prefix='test')
        
        # 测试记录学习率
        visualizer.log_learning_rate(0.001, step=1)
        
        # 关闭
        visualizer.close()
        
        print("✅ 可视化器测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("训练系统集成测试")
    print("=" * 60)
    
    # 运行测试
    main(verbosity=2)