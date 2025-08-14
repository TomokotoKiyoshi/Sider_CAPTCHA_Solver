#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Lite-HRNet-18+LiteFPN模型参数量
验证模型大小是否符合预期（~3.5M参数）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from src.models import LiteHRNet18FPN, create_lite_hrnet_18_fpn


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        only_trainable: 是否只统计可训练参数
        
    Returns:
        参数数量
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def analyze_model_parameters():
    """分析模型参数分布"""
    print("=" * 80)
    print("Lite-HRNet-18+LiteFPN 模型参数分析")
    print("=" * 80)
    
    # 创建模型
    model = create_lite_hrnet_18_fpn()
    
    # 1. 总体参数统计
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, only_trainable=True)
    
    print("\n【总体统计】")
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"不可训练参数: {total_params - trainable_params:,}")
    print(f"模型大小 (FP32): {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"模型大小 (FP16): {total_params * 2 / 1024 / 1024:.2f} MB")
    print(f"模型大小 (INT8): {total_params / 1024 / 1024:.2f} MB")
    
    # 2. 各阶段参数分布
    print("\n【各阶段参数分布】")
    stage_params = {}
    
    # Stem阶段
    stem_params = count_parameters(model.stem)
    stage_params['Stem (Stage1)'] = stem_params
    
    # Stage2-4 (Lite-HRNet主干)
    stage2_params = count_parameters(model.stage2)
    stage_params['Stage2 (双分支)'] = stage2_params
    
    stage3_params = count_parameters(model.stage3)
    stage_params['Stage3 (三分支)'] = stage3_params
    
    stage4_params = count_parameters(model.stage4)
    stage_params['Stage4 (四分支)'] = stage4_params
    
    # LiteFPN
    fpn_params = count_parameters(model.lite_fpn)
    stage_params['LiteFPN (Stage5)'] = fpn_params
    
    # DualHead
    head_params = count_parameters(model.dual_head)
    stage_params['DualHead (Stage6)'] = head_params
    
    # 打印各阶段参数
    for stage_name, params in stage_params.items():
        percentage = params / total_params * 100
        print(f"  {stage_name:20s}: {params:10,} ({params/1e6:6.2f}M) [{percentage:5.1f}%]")
    
    # 验证总和
    sum_params = sum(stage_params.values())
    print(f"\n  {'总计':20s}: {sum_params:10,} ({sum_params/1e6:6.2f}M)")
    
    if sum_params != total_params:
        print(f"  警告: 各阶段参数总和({sum_params})与模型总参数({total_params})不一致!")
    
    # 3. 详细层级参数分析
    print("\n【详细层级参数分析】")
    
    # 分析Stem内部
    print("\n  Stem阶段内部:")
    for name, module in model.stem.named_children():
        params = count_parameters(module)
        print(f"    {name:15s}: {params:8,} ({params/1e3:.1f}K)")
    
    # 分析DualHead内部
    print("\n  DualHead内部:")
    for name, module in model.dual_head.named_children():
        params = count_parameters(module)
        print(f"    {name:15s}: {params:8,} ({params/1e3:.1f}K)")
    
    # 4. 参数类型分析
    print("\n【参数类型分析】")
    conv_params = 0
    bn_params = 0
    other_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name and 'norm' not in name:
            conv_params += param.numel()
        elif 'bn' in name or 'norm' in name:
            bn_params += param.numel()
        else:
            other_params += param.numel()
    
    print(f"  卷积层参数: {conv_params:,} ({conv_params/total_params*100:.1f}%)")
    print(f"  BN层参数: {bn_params:,} ({bn_params/total_params*100:.1f}%)")
    print(f"  其他参数: {other_params:,} ({other_params/total_params*100:.1f}%)")
    
    # 5. 内存占用估算
    print("\n【内存占用估算】")
    batch_size = 16
    input_shape = (batch_size, 4, 256, 512)
    
    # 计算输入内存
    input_memory = batch_size * 4 * 256 * 512 * 4 / 1024 / 1024  # MB
    
    # 估算中间特征内存（粗略估计为参数量的2-3倍）
    feature_memory = total_params * 4 * 2.5 / 1024 / 1024  # MB
    
    # 参数内存
    param_memory = total_params * 4 / 1024 / 1024  # MB
    
    # 梯度内存（训练时）
    grad_memory = trainable_params * 4 / 1024 / 1024  # MB
    
    print(f"  批次大小: {batch_size}")
    print(f"  输入内存: {input_memory:.2f} MB")
    print(f"  参数内存: {param_memory:.2f} MB")
    print(f"  特征内存(估算): {feature_memory:.2f} MB")
    print(f"  梯度内存(训练): {grad_memory:.2f} MB")
    print(f"  推理总内存(估算): {input_memory + param_memory + feature_memory:.2f} MB")
    print(f"  训练总内存(估算): {input_memory + param_memory + feature_memory + grad_memory:.2f} MB")
    
    # 6. 与目标对比
    print("\n【目标对比】")
    target_params = 3.5e6
    diff = total_params - target_params
    diff_percent = diff / target_params * 100
    
    print(f"  目标参数量: {target_params/1e6:.1f}M")
    print(f"  实际参数量: {total_params/1e6:.2f}M")
    print(f"  差异: {diff/1e6:+.2f}M ({diff_percent:+.1f}%)")
    
    if abs(diff_percent) < 10:
        print("  ✅ 参数量符合预期范围")
    else:
        print("  ⚠️ 参数量与预期相差较大")
    
    print("\n" + "=" * 80)
    
    return total_params


def test_model_forward():
    """测试模型前向传播"""
    print("\n【前向传播测试】")
    print("-" * 40)
    
    # 创建模型
    model = create_lite_hrnet_18_fpn()
    model.eval()
    
    # 创建随机输入
    batch_size = 2
    x = torch.randn(batch_size, 4, 256, 512)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x)
    
    # 打印输出形状
    print("\n输出形状:")
    for key, value in outputs.items():
        print(f"  {key:20s}: {value.shape}")
    
    # 测试解码
    decoded = model.decode_predictions(outputs)
    print("\n解码后的预测:")
    for key, value in decoded.items():
        print(f"  {key:20s}: {value.shape}")
    
    print("\n✅ 前向传播测试通过")
    print("-" * 40)


def test_model_device():
    """测试模型在不同设备上的运行"""
    print("\n【设备兼容性测试】")
    print("-" * 40)
    
    # 创建模型
    model = create_lite_hrnet_18_fpn()
    x = torch.randn(1, 4, 256, 512)
    
    # CPU测试
    print("CPU测试...")
    model.cpu()
    x_cpu = x.cpu()
    with torch.no_grad():
        output_cpu = model(x_cpu)
    print("  ✅ CPU运行正常")
    
    # GPU测试（如果可用）
    if torch.cuda.is_available():
        print("GPU测试...")
        model.cuda()
        x_gpu = x.cuda()
        with torch.no_grad():
            output_gpu = model(x_gpu)
        print(f"  ✅ GPU运行正常 (设备: {torch.cuda.get_device_name(0)})")
    else:
        print("  ⏭️ GPU不可用，跳过GPU测试")
    
    print("-" * 40)


if __name__ == "__main__":
    # 运行参数分析
    total_params = analyze_model_parameters()
    
    # 运行前向传播测试
    test_model_forward()
    
    # 运行设备兼容性测试
    test_model_device()
    
    # 最终总结
    print("\n" + "=" * 80)
    print("测试完成总结")
    print("=" * 80)
    print(f"✅ 模型参数量: {total_params/1e6:.2f}M")
    print("✅ 前向传播正常")
    print("✅ 设备兼容性正常")
    print("=" * 80)