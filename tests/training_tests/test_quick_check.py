#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速检查训练系统是否正常工作
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')


def test_imports():
    """测试所有模块是否能正常导入"""
    print("\n" + "="*60)
    print("1. 测试模块导入")
    print("="*60)
    
    try:
        # 导入模型
        from src.models import create_lite_hrnet_18_fpn
        print("✅ 模型模块导入成功")
        
        # 导入训练模块
        from src.training.config_manager import ConfigManager
        print("✅ 配置管理器导入成功")
        
        from src.training.data_pipeline import DataPipeline
        print("✅ 数据管道导入成功")
        
        from src.training.training_engine import TrainingEngine
        print("✅ 训练引擎导入成功")
        
        from src.training.validator import Validator
        print("✅ 验证器导入成功")
        
        from src.training.visualizer import Visualizer
        print("✅ 可视化器导入成功")
        
        # 导入主训练脚本
        import scripts.training.train as train_module
        print("✅ 主训练脚本导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_model_creation():
    """测试模型创建和前向传播"""
    print("\n" + "="*60)
    print("2. 测试模型创建")
    print("="*60)
    
    try:
        from src.models import create_lite_hrnet_18_fpn
        
        # 创建模型
        model = create_lite_hrnet_18_fpn()
        print("✅ 模型创建成功")
        
        # 获取参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   参数量: {total_params/1e6:.2f}M")
        
        # 测试前向传播
        x = torch.randn(1, 4, 256, 512)
        with torch.no_grad():
            outputs = model(x)
        
        print("✅ 前向传播成功")
        print(f"   输出键: {list(outputs.keys())}")
        
        # 测试解码
        decoded = model.decode_predictions(outputs)
        print("✅ 预测解码成功")
        print(f"   解码键: {list(decoded.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False


def test_config_loading():
    """测试配置文件加载"""
    print("\n" + "="*60)
    print("3. 测试配置加载")
    print("="*60)
    
    try:
        from src.training.config_manager import ConfigManager
        
        config_path = Path("config/training_config.yaml")
        
        if not config_path.exists():
            print(f"⚠️ 配置文件不存在: {config_path}")
            return False
        
        # 加载配置
        config_manager = ConfigManager(str(config_path))
        print("✅ 配置加载成功")
        
        # 检查关键配置
        config = config_manager.config
        print(f"   批次大小: {config['train']['batch_size']}")
        print(f"   学习率: {config['optimizer']['lr']}")
        print(f"   训练轮数: {config['sched']['epochs']}")
        print(f"   早停配置: min_epochs={config['eval']['early_stopping']['min_epochs']}, "
              f"patience={config['eval']['early_stopping']['patience']}")
        
        if 'second_guard' in config['eval']['early_stopping']:
            sg = config['eval']['early_stopping']['second_guard']
            print(f"   第二防护: {sg['metric']} ({sg['mode']}, Δ>{sg['min_delta']})")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


def test_device_availability():
    """测试设备可用性"""
    print("\n" + "="*60)
    print("4. 测试设备可用性")
    print("="*60)
    
    # 测试CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        
        # 测试GPU内存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU内存: {gpu_memory:.1f}GB")
    else:
        print("⚠️ CUDA不可用，将使用CPU训练")
    
    # 测试CPU
    print(f"✅ CPU核心数: {torch.get_num_threads()}")
    
    return True


def test_data_directory():
    """测试数据目录是否存在"""
    print("\n" + "="*60)
    print("5. 测试数据目录")
    print("="*60)
    
    # 检查数据文件（JSON格式）
    train_file = Path("data/split_for_training/train.json")
    val_file = Path("data/split_for_training/val.json")
    test_file = Path("data/split_for_training/test.json")
    processed_dir = Path("data/processed")
    
    results = []
    
    # 检查训练数据
    if train_file.exists():
        import json
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            if isinstance(train_data, dict) and 'filenames' in train_data:
                num_train = len(train_data['filenames'])
            else:
                num_train = len(train_data)
        print(f"✅ 训练文件存在: {train_file}")
        print(f"   训练样本数: {num_train}")
        results.append(True)
    else:
        print(f"❌ 训练文件不存在: {train_file}")
        results.append(False)
    
    # 检查验证数据
    if val_file.exists():
        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
            if isinstance(val_data, dict) and 'filenames' in val_data:
                num_val = len(val_data['filenames'])
            else:
                num_val = len(val_data)
        print(f"✅ 验证文件存在: {val_file}")
        print(f"   验证样本数: {num_val}")
        results.append(True)
    else:
        print(f"❌ 验证文件不存在: {val_file}")
        results.append(False)
    
    # 检查测试数据
    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            if isinstance(test_data, list):
                num_test = len(test_data)
            elif isinstance(test_data, dict) and 'filenames' in test_data:
                num_test = len(test_data['filenames'])
            else:
                num_test = 0
        print(f"✅ 测试文件存在: {test_file}")
        print(f"   测试样本数: {num_test}")
        results.append(True)
    else:
        print(f"❌ 测试文件不存在: {test_file}")
        results.append(False)
    
    # 检查处理后的图像目录
    if processed_dir.exists():
        num_images = len(list(processed_dir.glob("*.png")))
        print(f"✅ 图像目录存在: {processed_dir}")
        print(f"   图像文件数: {num_images}")
        results.append(True)
    else:
        print(f"❌ 图像目录不存在: {processed_dir}")
        results.append(False)
    
    return all(results)


def test_mini_training():
    """测试最小训练流程"""
    print("\n" + "="*60)
    print("6. 测试最小训练流程")
    print("="*60)
    
    try:
        from src.models import create_lite_hrnet_18_fpn
        import torch.nn as nn
        import torch.optim as optim
        
        # 创建模型
        model = create_lite_hrnet_18_fpn()
        device = torch.device('cpu')
        model = model.to(device)
        
        # 创建优化器
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        # 创建虚拟数据
        batch = {
            'image': torch.randn(2, 4, 256, 512).to(device),
            'gap_coords': torch.randn(2, 2).to(device) * 100 + 160,
            'slider_coords': torch.randn(2, 2).to(device) * 50 + 25
        }
        
        # 前向传播
        model.train()
        outputs = model(batch['image'])
        print("✅ 前向传播成功")
        
        # 简单损失（仅测试）
        loss = sum(output.mean() for output in outputs.values())
        print(f"✅ 损失计算成功: {loss.item():.4f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("✅ 反向传播成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 最小训练流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("训练系统快速检查")
    print("="*60)
    
    results = []
    
    # 运行各项测试
    results.append(("模块导入", test_imports()))
    results.append(("模型创建", test_model_creation()))
    results.append(("配置加载", test_config_loading()))
    results.append(("设备可用性", test_device_availability()))
    results.append(("数据目录", test_data_directory()))
    results.append(("最小训练流程", test_mini_training()))
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:15s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 所有测试通过！训练系统可以正常工作。")
        print("\n下一步:")
        print("1. 生成训练数据: python scripts/generate_captchas.py")
        print("2. 划分数据集: python scripts/data_generation/split_dataset.py")
        print("3. 开始训练: python scripts/training/train.py")
    else:
        print("\n⚠️ 部分测试失败，请检查并修复问题。")
        print("\n可能的解决方案:")
        print("1. 检查是否安装所有依赖: pip install -r requirements.txt")
        print("2. 检查数据目录是否存在并包含数据")
        print("3. 检查配置文件是否存在: config/training_config.yaml")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)