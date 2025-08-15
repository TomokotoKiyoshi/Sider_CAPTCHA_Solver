#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试NPY数据训练流程
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import logging
from src.models import create_lite_hrnet_18_fpn
from src.training.npy_data_loader import NPYDataPipeline
from src.training.training_engine import TrainingEngine
from src.training.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO, format='%(message)s')


def test_npy_training():
    """测试使用NPY数据的训练流程"""
    print("\n" + "="*60)
    print("测试NPY数据训练流程")
    print("="*60)
    
    try:
        # 加载配置
        config_manager = ConfigManager('config/training_config.yaml')
        config = config_manager.config
        
        # 覆盖一些设置用于测试
        config['train']['batch_size'] = 64  # 使用较小批次进行测试
        config['sched']['epochs'] = 1
        
        # 创建模型
        print("\n创建模型...")
        model = create_lite_hrnet_18_fpn(config['model'])
        device = config_manager.get_device()
        
        # 创建NPY数据管道
        print("\n初始化NPY数据管道...")
        data_pipeline = NPYDataPipeline(config)
        data_pipeline.setup()
        
        # 获取数据加载器
        train_loader = data_pipeline.get_train_loader()
        
        if train_loader is None:
            print("❌ 无法获取训练数据加载器")
            return False
        
        print(f"✅ 训练批次数: {len(train_loader)}")
        print(f"✅ 训练样本数: {data_pipeline.num_train_samples}")
        
        # 测试加载一个批次
        print("\n测试加载一个批次...")
        for batch in train_loader:
            print(f"✅ 批次加载成功")
            print(f"  图像形状: {batch['image'].shape}")
            print(f"  缺口坐标形状: {batch['gap_coords'].shape}")
            print(f"  滑块坐标形状: {batch['slider_coords'].shape}")
            
            # 检查是否有热力图和偏移量
            if 'heatmap_gap' in batch:
                print(f"  缺口热力图形状: {batch['heatmap_gap'].shape}")
                print(f"  滑块热力图形状: {batch['heatmap_slider'].shape}")
                print(f"  缺口偏移量形状: {batch['offset_gap'].shape}")
                print(f"  滑块偏移量形状: {batch['offset_slider'].shape}")
            
            # 测试模型前向传播
            print("\n测试模型前向传播...")
            model = model.to(device)
            images = batch['image'].to(device)
            
            with torch.no_grad():
                outputs = model(images)
            
            print(f"✅ 前向传播成功")
            print(f"  输出键: {list(outputs.keys())}")
            
            # 测试损失计算（如果有标签）
            if 'heatmap_gap' in batch:
                print("\n测试损失计算...")
                from src.models.loss_calculation.total_loss import TotalLoss
                
                criterion = TotalLoss(config)
                
                # 准备标签
                labels = {
                    'heatmap_gap': batch['heatmap_gap'].to(device),
                    'heatmap_slider': batch['heatmap_slider'].to(device),
                    'offset_gap': batch['offset_gap'].to(device),
                    'offset_slider': batch['offset_slider'].to(device),
                    'weight_gap': batch['weight_gap'].to(device),
                    'weight_slider': batch['weight_slider'].to(device)
                }
                
                # 计算损失
                loss_dict = criterion(outputs, labels)
                total_loss = loss_dict['total']
                
                print(f"✅ 损失计算成功")
                print(f"  总损失: {total_loss.item():.4f}")
                print(f"  损失组件: {list(loss_dict.keys())}")
            
            break  # 只测试一个批次
        
        print("\n" + "="*60)
        print("✅ NPY数据训练流程测试通过！")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    success = test_npy_training()
    
    if success:
        print("\n下一步：")
        print("1. 开始训练: python scripts/training/train.py")
        print("2. 监控训练: tensorboard --logdir logs/tensorboard/1.1.0")
    else:
        print("\n请检查错误信息并修复问题")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())