# -*- coding: utf-8 -*-
"""
Stage3模块测试脚本
测试Stage3的配置加载、前向传播和跨分辨率融合
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from src.models.stage3 import create_stage3, CrossResolutionFusion3
from src.models.config_loader import get_stage3_config


def test_stage3():
    """测试Stage3模块"""
    print("=" * 60)
    print("测试Stage3模块 - 三分支结构")
    print("=" * 60)
    
    # 加载配置
    stage3_config = get_stage3_config()
    print("\n配置信息:")
    print(f"  分支通道: {stage3_config['channels']}")
    print(f"  块数量: {stage3_config['num_blocks']}")
    print(f"  扩张倍率: {stage3_config['expansion']}")
    
    # 创建模块
    stage3 = create_stage3(stage3_config)
    stage3.eval()
    
    # 准备测试输入 (来自Stage2的输出)
    batch_size = 2
    channels_1_4 = stage3_config['channels'][0]  # 32
    channels_1_8 = stage3_config['channels'][1]  # 64
    
    # Stage2的输出作为Stage3的输入
    y1 = torch.randn(batch_size, channels_1_4, 64, 128)  # 1/4分辨率
    y2 = torch.randn(batch_size, channels_1_8, 32, 64)   # 1/8分辨率
    inputs = [y1, y2]
    
    print(f"\n输入张量:")
    print(f"  Y1 (1/4分辨率): {y1.shape}")
    print(f"  Y2 (1/8分辨率): {y2.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = stage3(inputs)
    
    # 输出信息
    print(f"\n输出信息:")
    print(f"  输出数量: {len(outputs)}")
    print(f"  T1 (1/4分辨率): {outputs[0].shape}")
    print(f"  T2 (1/8分辨率): {outputs[1].shape}")
    print(f"  T3 (1/16分辨率): {outputs[2].shape}")
    
    # 验证输出形状
    expected_shapes = [
        (batch_size, stage3_config['channels'][0], 64, 128),   # 1/4
        (batch_size, stage3_config['channels'][1], 32, 64),    # 1/8
        (batch_size, stage3_config['channels'][2], 16, 32),    # 1/16
    ]
    
    for i, (output, expected) in enumerate(zip(outputs, expected_shapes)):
        assert output.shape == expected, f"输出{i}形状错误: {output.shape} != {expected}"
    
    print("\n✓ 输出形状验证通过")
    
    # 参数统计
    total_params = sum(p.numel() for p in stage3.parameters())
    trainable_params = sum(p.numel() for p in stage3.parameters() if p.requires_grad)
    
    print(f"\n参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 分模块参数统计
    print(f"\n分模块参数统计:")
    
    # Transition参数
    t3_params = sum(p.numel() for name, p in stage3.named_parameters() if 't3_' in name)
    print(f"  Transition3模块: {t3_params:,}")
    
    # 分支参数
    branch_1_4_params = sum(p.numel() for name, p in stage3.named_parameters() if 'branch_1_4' in name)
    branch_1_8_params = sum(p.numel() for name, p in stage3.named_parameters() if 'branch_1_8' in name)
    branch_1_16_params = sum(p.numel() for name, p in stage3.named_parameters() if 'branch_1_16' in name)
    print(f"  1/4分支 (3×LiteBlock): {branch_1_4_params:,}")
    print(f"  1/8分支 (3×LiteBlock): {branch_1_8_params:,}")
    print(f"  1/16分支 (3×LiteBlock): {branch_1_16_params:,}")
    
    # 融合模块参数
    fusion_params = sum(p.numel() for name, p in stage3.named_parameters() if 'fusion' in name)
    print(f"  CRF-3融合模块: {fusion_params:,}")
    
    print("\n✓ Stage3模块测试完成")
    print("=" * 60)


def test_cross_resolution_fusion3():
    """测试三向跨分辨率融合模块"""
    print("\n" + "=" * 60)
    print("测试跨分辨率融合模块 CRF-3")
    print("=" * 60)
    
    # 创建融合模块
    channels = [32, 64, 128]
    fusion = CrossResolutionFusion3(
        channels_1_4=channels[0],
        channels_1_8=channels[1],
        channels_1_16=channels[2]
    )
    fusion.eval()
    
    # 测试输入
    batch_size = 2
    x_1_4 = torch.randn(batch_size, channels[0], 64, 128)
    x_1_8 = torch.randn(batch_size, channels[1], 32, 64)
    x_1_16 = torch.randn(batch_size, channels[2], 16, 32)
    
    print(f"\n输入:")
    print(f"  1/4分辨率: {x_1_4.shape}")
    print(f"  1/8分辨率: {x_1_8.shape}")
    print(f"  1/16分辨率: {x_1_16.shape}")
    
    # 前向传播
    with torch.no_grad():
        t1, t2, t3 = fusion(x_1_4, x_1_8, x_1_16)
    
    print(f"\n输出:")
    print(f"  T1 (1/4分辨率): {t1.shape}")
    print(f"  T2 (1/8分辨率): {t2.shape}")
    print(f"  T3 (1/16分辨率): {t3.shape}")
    
    # 验证输出形状保持不变
    assert t1.shape == x_1_4.shape, "1/4分辨率形状不匹配"
    assert t2.shape == x_1_8.shape, "1/8分辨率形状不匹配"
    assert t3.shape == x_1_16.shape, "1/16分辨率形状不匹配"
    
    print("\n✓ CRF-3融合模块测试通过")
    
    # 参数统计
    fusion_params = sum(p.numel() for p in fusion.parameters())
    print(f"\nCRF-3融合模块参数量: {fusion_params:,}")
    
    # 详细统计各路径参数
    print("\n融合路径参数统计:")
    
    # 到1/4分支的路径
    to_1_4_params = sum(p.numel() for name, p in fusion.named_parameters() 
                        if 'to_1_4' in name or 'smooth_1_4' in name)
    print(f"  汇到1/4分支: {to_1_4_params:,}")
    
    # 到1/8分支的路径
    to_1_8_params = sum(p.numel() for name, p in fusion.named_parameters() 
                        if 'to_1_8' in name or 'smooth_1_8' in name)
    print(f"  汇到1/8分支: {to_1_8_params:,}")
    
    # 到1/16分支的路径
    to_1_16_params = sum(p.numel() for name, p in fusion.named_parameters() 
                         if 'to_1_16' in name or 'smooth_1_16' in name)
    print(f"  汇到1/16分支: {to_1_16_params:,}")
    
    print("=" * 60)


def test_multi_resolution_consistency():
    """测试多分辨率一致性"""
    print("\n" + "=" * 60)
    print("测试多分辨率一致性")
    print("=" * 60)
    
    # 创建Stage3
    stage3_config = get_stage3_config()
    stage3 = create_stage3(stage3_config)
    stage3.eval()
    
    # 创建具有特定模式的输入，以验证分辨率处理
    batch_size = 1
    
    # 创建具有渐变模式的输入
    y1 = torch.ones(batch_size, 32, 64, 128) * 0.5
    y2 = torch.ones(batch_size, 64, 32, 64) * 0.5
    
    with torch.no_grad():
        outputs = stage3([y1, y2])
    
    print("\n分辨率下采样验证:")
    print(f"  1/4 → 1/8: {outputs[0].shape} → {outputs[1].shape}")
    print(f"  1/8 → 1/16: {outputs[1].shape} → {outputs[2].shape}")
    
    # 验证通道数递增
    channels = [out.shape[1] for out in outputs]
    assert channels == stage3_config['channels'], f"通道数不匹配: {channels} != {stage3_config['channels']}"
    
    print("\n✓ 多分辨率一致性测试通过")
    print("=" * 60)


if __name__ == "__main__":
    # 测试Stage3主模块
    test_stage3()
    
    # 测试跨分辨率融合模块
    test_cross_resolution_fusion3()
    
    # 测试多分辨率一致性
    test_multi_resolution_consistency()
    
    print("\n" + "=" * 60)
    print("🎯 所有Stage3测试通过! ✓")
    print("=" * 60)