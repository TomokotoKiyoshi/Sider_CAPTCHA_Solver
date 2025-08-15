#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整训练管道测试
创建虚拟数据并测试整个训练流程
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import cv2
import tempfile
import shutil
import yaml
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class TrainingPipelineTest:
    """训练管道测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.setup_directories()
        logging.info(f"创建临时测试目录: {self.temp_dir}")
    
    def setup_directories(self):
        """设置测试目录结构"""
        self.data_dir = Path(self.temp_dir) / "data"
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        
        # 创建目录
        self.train_dir.mkdir(parents=True)
        self.val_dir.mkdir(parents=True)
    
    def cleanup(self):
        """清理临时文件"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logging.info("清理完成")
    
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
                'lr': 0.001,  # 较高的学习率用于快速测试
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'weight_decay': 0.01,
                'clip_grad_norm': 1.0,
                'ema_decay': 0  # 测试时不用EMA
            },
            'sched': {
                'warmup_epochs': 1,
                'cosine_min_lr': 1e-6,
                'epochs': 3,  # 只训练3个epoch
                'min_epochs': 2,
                'patience': 2
            },
            'train': {
                'batch_size': 4,  # 小批次
                'amp': 'none',  # 不用混合精度
                'channels_last': False,
                'num_workers': 0,  # 单进程
                'pin_memory': False
            },
            'eval': {
                'metrics': ['mae_px', 'rmse_px', 'hit_le_2px', 'hit_le_5px'],
                'select_by': 'hit_le_5px',
                'vis_fail_k': 3,
                'eval_interval': 1,
                'early_stopping': {
                    'min_epochs': 2,
                    'patience': 2,
                    'second_guard': {
                        'metric': 'mae_px',
                        'mode': 'min',
                        'min_delta': 0.1
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
                'tensorboard': False,  # 测试时不用TensorBoard
                'tensorboard_dir': str(Path(self.temp_dir) / 'tensorboard')
            },
            'hardware': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'cudnn_benchmark': False
            }
        }
        
        # 保存配置文件
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logging.info("测试配置文件已创建")
        return config
    
    def create_dummy_data(self, num_train=20, num_val=10):
        """
        创建虚拟训练数据
        
        Args:
            num_train: 训练样本数
            num_val: 验证样本数
        """
        logging.info(f"创建虚拟数据: {num_train}个训练样本, {num_val}个验证样本")
        
        # 创建训练数据
        for i in range(num_train):
            # 创建带有简单图案的图像
            img = np.ones((160, 320, 3), dtype=np.uint8) * 128
            
            # 添加一些随机矩形（模拟背景）
            for _ in range(5):
                x1 = np.random.randint(0, 280)
                y1 = np.random.randint(0, 140)
                x2 = x1 + np.random.randint(20, 40)
                y2 = y1 + np.random.randint(20, 40)
                color = tuple(np.random.randint(50, 200, 3).tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            # 随机坐标（真实范围内）
            bgx = np.random.randint(100, 250)
            bgy = np.random.randint(50, 110)
            sdx = np.random.randint(20, 30)
            sdy = bgy  # 滑块y坐标等于缺口y坐标
            
            # 在图像上标记位置（方便可视化）
            cv2.circle(img, (bgx, bgy), 5, (255, 0, 0), -1)  # 缺口位置（蓝色）
            cv2.circle(img, (sdx, sdy), 5, (0, 255, 0), -1)  # 滑块位置（绿色）
            
            # 保存图像
            filename = f"Pic{i:04d}_Bgx{bgx}Bgy{bgy}_Sdx{sdx}Sdy{sdy}_test.png"
            cv2.imwrite(str(self.train_dir / filename), img)
        
        # 创建验证数据
        for i in range(num_val):
            img = np.ones((160, 320, 3), dtype=np.uint8) * 128
            
            for _ in range(5):
                x1 = np.random.randint(0, 280)
                y1 = np.random.randint(0, 140)
                x2 = x1 + np.random.randint(20, 40)
                y2 = y1 + np.random.randint(20, 40)
                color = tuple(np.random.randint(50, 200, 3).tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            bgx = np.random.randint(100, 250)
            bgy = np.random.randint(50, 110)
            sdx = np.random.randint(20, 30)
            sdy = bgy
            
            cv2.circle(img, (bgx, bgy), 5, (255, 0, 0), -1)
            cv2.circle(img, (sdx, sdy), 5, (0, 255, 0), -1)
            
            filename = f"Pic{i+num_train:04d}_Bgx{bgx}Bgy{bgy}_Sdx{sdx}Sdy{sdy}_test.png"
            cv2.imwrite(str(self.val_dir / filename), img)
        
        logging.info("虚拟数据创建完成")
    
    def run_training(self):
        """运行训练测试"""
        logging.info("\n" + "="*60)
        logging.info("开始训练测试")
        logging.info("="*60)
        
        # 导入必要模块
        from src.models import create_lite_hrnet_18_fpn
        from src.training.config_manager import ConfigManager
        from src.training.data_pipeline import DataPipeline
        from src.training.training_engine import TrainingEngine
        from src.training.validator import Validator
        
        try:
            # 1. 加载配置
            config_manager = ConfigManager(str(self.config_file))
            config = config_manager.config
            device = config_manager.get_device()
            logging.info(f"使用设备: {device}")
            
            # 2. 创建模型
            model = create_lite_hrnet_18_fpn(config['model'])
            logging.info("模型创建成功")
            
            # 3. 创建数据管道
            data_pipeline = DataPipeline(config)
            data_pipeline.setup(str(self.train_dir), str(self.val_dir))
            logging.info(f"数据加载完成: {data_pipeline.num_train_samples}个训练样本, "
                        f"{data_pipeline.num_val_samples}个验证样本")
            
            # 4. 创建训练引擎
            engine = TrainingEngine(model, config, device)
            logging.info("训练引擎初始化成功")
            
            # 5. 创建验证器
            validator = Validator(config, device)
            logging.info("验证器初始化成功")
            
            # 6. 训练循环
            num_epochs = config['sched']['epochs']
            best_metric = 0
            
            for epoch in range(1, num_epochs + 1):
                logging.info(f"\n--- Epoch {epoch}/{num_epochs} ---")
                
                # 训练
                train_metrics = engine.train_epoch(
                    data_pipeline.get_train_loader(), 
                    epoch
                )
                logging.info(f"训练损失: {train_metrics['loss']:.4f}")
                
                # 验证
                val_metrics = validator.validate(
                    model,
                    data_pipeline.get_val_loader(),
                    epoch
                )
                logging.info(f"验证MAE: {val_metrics['mae_px']:.2f}px")
                logging.info(f"Hit@5px: {val_metrics['hit_le_5px']:.2f}%")
                
                # 更新最佳指标
                if val_metrics['hit_le_5px'] > best_metric:
                    best_metric = val_metrics['hit_le_5px']
                    logging.info(f"新的最佳模型! Hit@5px: {best_metric:.2f}%")
                
                # 早停检查
                if val_metrics.get('early_stop', False):
                    logging.info("触发早停，停止训练")
                    break
                
                # 更新学习率
                engine.step_scheduler()
            
            logging.info("\n" + "="*60)
            logging.info("训练测试完成！")
            logging.info(f"最佳Hit@5px: {best_metric:.2f}%")
            logging.info("="*60)
            
            return True
            
        except Exception as e:
            logging.error(f"训练测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_outputs(self):
        """验证输出文件"""
        logging.info("\n验证输出文件...")
        
        # 检查检查点
        checkpoint_dir = Path(self.temp_dir) / 'checkpoints'
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            logging.info(f"✅ 找到 {len(checkpoints)} 个检查点文件")
            for ckpt in checkpoints:
                logging.info(f"   - {ckpt.name}")
        else:
            logging.warning("❌ 未找到检查点目录")
        
        # 检查日志
        log_dir = Path(self.temp_dir) / 'logs'
        if log_dir.exists():
            logs = list(log_dir.glob("*.log"))
            logging.info(f"✅ 找到 {len(logs)} 个日志文件")
        else:
            logging.warning("❌ 未找到日志目录")
        
        return True


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("完整训练管道测试")
    print("="*60)
    
    tester = TrainingPipelineTest()
    
    try:
        # 1. 创建配置
        tester.create_test_config()
        
        # 2. 创建虚拟数据
        tester.create_dummy_data(num_train=20, num_val=10)
        
        # 3. 运行训练
        success = tester.run_training()
        
        # 4. 验证输出
        if success:
            tester.verify_outputs()
        
        if success:
            print("\n✅ 训练管道测试成功！所有组件正常工作。")
            print("\n下一步建议：")
            print("1. 生成真实训练数据")
            print("2. 调整超参数")
            print("3. 开始正式训练")
        else:
            print("\n❌ 训练管道测试失败，请检查错误信息。")
        
        return success
        
    finally:
        # 清理临时文件
        tester.cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)