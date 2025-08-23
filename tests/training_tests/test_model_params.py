#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模型参数量
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent  # 上移两级到项目根目录
sys.path.insert(0, str(project_root))

import torch
from src.models import create_lite_hrnet_18_fpn

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def format_params(num_params):
    """格式化参数数量"""
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    else:
        return f"{num_params}"

def analyze_model_layers(model):
    """分析模型各层参数分布"""
    layer_params = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                # 获取主要层级名称
                main_name = name.split('.')[0] if '.' in name else name
                if main_name not in layer_params:
                    layer_params[main_name] = 0
                layer_params[main_name] += params
    
    return layer_params

def main():
    print("=" * 60)
    print("模型参数量测试")
    print("=" * 60)
    
    # 创建模型
    print("\n正在创建模型...")
    model = create_lite_hrnet_18_fpn()
    
    # 计算总参数量
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n📊 模型参数统计:")
    print(f"  总参数量: {format_params(total_params)} ({total_params:,})")
    print(f"  可训练参数: {format_params(trainable_params)} ({trainable_params:,})")
    print(f"  不可训练参数: {format_params(total_params - trainable_params)} ({total_params - trainable_params:,})")
    
    # 分析各层参数分布
    print(f"\n📈 各模块参数分布:")
    layer_params = analyze_model_layers(model)
    
    # 按参数量排序
    sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)
    
    total_counted = sum(layer_params.values())
    for layer_name, params in sorted_layers[:10]:  # 显示前10个最大的层
        percentage = (params / total_params) * 100
        print(f"  {layer_name:20s}: {format_params(params):>10s} ({percentage:5.1f}%)")
    
    # 模型大小估算
    print(f"\n💾 模型存储估算:")
    model_size_mb = (total_params * 4) / (1024 * 1024)  # 假设float32
    print(f"  FP32模型大小: ~{model_size_mb:.1f} MB")
    print(f"  FP16模型大小: ~{model_size_mb/2:.1f} MB")
    print(f"  INT8模型大小: ~{model_size_mb/4:.1f} MB")
    
    # 测试前向传播
    print(f"\n🔍 测试前向传播...")
    try:
        dummy_input = torch.randn(1, 2, 256, 512)  # [B, C, H, W]
        with torch.no_grad():
            output = model(dummy_input)
        
        if isinstance(output, dict):
            print(f"  ✅ 前向传播成功!")
            print(f"  输出键: {list(output.keys())}")
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: shape={value.shape}")
        else:
            print(f"  ✅ 前向传播成功! 输出shape: {output.shape}")
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
    
    # 与目标对比
    print(f"\n🎯 与目标对比:")
    print(f"  当前: {format_params(total_params)}")
    print(f"  目标上限: 50M")
    print(f"  使用率: {(total_params/50e6)*100:.1f}%")
    
    if total_params > 50e6:
        print(f"  ⚠️ 警告: 超出50M参数限制!")
    else:
        print(f"  ✅ 符合参数量限制 (剩余: {format_params(50e6 - total_params)})")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()