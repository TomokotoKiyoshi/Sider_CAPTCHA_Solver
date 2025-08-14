# -*- coding: utf-8 -*-
"""
Stage4模块测试脚本
测试Stage4的配置加载、前向传播和跨分辨率融合
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from src.models.stage4 import create_stage4, CrossResolutionFusion4
from src.models.config_loader import get_stage4_config


def test_stage4():
    """测试Stage4模块"""
    print("=" * 70)
    print("测试Stage4模块 - 四分支结构（主干最终输出）")
    print("=" * 70)
    
    # 加载配置
    stage4_config = get_stage4_config()
    print("\n配置信息:")
    print(f"  分支通道: {stage4_config['channels']}")
    print(f"  块数量: {stage4_config['num_blocks']}")
    print(f"  扩张倍率: {stage4_config['expansion']}")
    
    # 创建模块
    stage4 = create_stage4(stage4_config)
    stage4.eval()
    
    # 准备测试输入 (来自Stage3的输出)
    batch_size = 2
    channels_1_4 = stage4_config['channels'][0]   # 32
    channels_1_8 = stage4_config['channels'][1]   # 64
    channels_1_16 = stage4_config['channels'][2]  # 128
    
    # Stage3的输出作为Stage4的输入
    t1 = torch.randn(batch_size, channels_1_4, 64, 128)   # 1/4分辨率
    t2 = torch.randn(batch_size, channels_1_8, 32, 64)    # 1/8分辨率
    t3 = torch.randn(batch_size, channels_1_16, 16, 32)   # 1/16分辨率
    inputs = [t1, t2, t3]
    
    print(f"\n输入张量:")
    print(f"  T1 (1/4分辨率): {t1.shape}")
    print(f"  T2 (1/8分辨率): {t2.shape}")
    print(f"  T3 (1/16分辨率): {t3.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = stage4(inputs)
    
    # 输出信息
    print(f"\n输出信息（主干最终输出）:")
    print(f"  输出数量: {len(outputs)}")
    print(f"  B1 (1/4分辨率): {outputs[0].shape}")
    print(f"  B2 (1/8分辨率): {outputs[1].shape}")
    print(f"  B3 (1/16分辨率): {outputs[2].shape}")
    print(f"  B4 (1/32分辨率): {outputs[3].shape}")
    
    # 验证输出形状
    expected_shapes = [
        (batch_size, stage4_config['channels'][0], 64, 128),   # 1/4
        (batch_size, stage4_config['channels'][1], 32, 64),    # 1/8
        (batch_size, stage4_config['channels'][2], 16, 32),    # 1/16
        (batch_size, stage4_config['channels'][3], 8, 16),     # 1/32
    ]
    
    for i, (output, expected) in enumerate(zip(outputs, expected_shapes)):
        assert output.shape == expected, f"输出{i}形状错误: {output.shape} != {expected}"
    
    print("\n✓ 输出形状验证通过")
    
    # 参数统计
    total_params = sum(p.numel() for p in stage4.parameters())
    trainable_params = sum(p.numel() for p in stage4.parameters() if p.requires_grad)
    
    print(f"\n参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 分模块参数统计
    print(f"\n分模块参数统计:")
    
    # Transition参数
    t4_params = sum(p.numel() for name, p in stage4.named_parameters() if 't4_' in name)
    print(f"  Transition4模块: {t4_params:,}")
    
    # 分支参数
    branch_1_4_params = sum(p.numel() for name, p in stage4.named_parameters() if 'branch_1_4' in name)
    branch_1_8_params = sum(p.numel() for name, p in stage4.named_parameters() if 'branch_1_8' in name)
    branch_1_16_params = sum(p.numel() for name, p in stage4.named_parameters() if 'branch_1_16' in name)
    branch_1_32_params = sum(p.numel() for name, p in stage4.named_parameters() if 'branch_1_32' in name)
    print(f"  1/4分支 (2×LiteBlock): {branch_1_4_params:,}")
    print(f"  1/8分支 (2×LiteBlock): {branch_1_8_params:,}")
    print(f"  1/16分支 (2×LiteBlock): {branch_1_16_params:,}")
    print(f"  1/32分支 (2×LiteBlock): {branch_1_32_params:,}")
    
    # 融合模块参数
    fusion_params = sum(p.numel() for name, p in stage4.named_parameters() if 'fusion' in name)
    print(f"  CRF-4融合模块: {fusion_params:,}")
    
    print("\n✓ Stage4模块测试完成")
    print("=" * 70)


def test_cross_resolution_fusion4():
    """测试四向跨分辨率融合模块"""
    print("\n" + "=" * 70)
    print("测试跨分辨率融合模块 CRF-4")
    print("=" * 70)
    
    # 创建融合模块
    channels = [32, 64, 128, 256]
    fusion = CrossResolutionFusion4(
        channels_1_4=channels[0],
        channels_1_8=channels[1],
        channels_1_16=channels[2],
        channels_1_32=channels[3]
    )
    fusion.eval()
    
    # 测试输入
    batch_size = 2
    x_1_4 = torch.randn(batch_size, channels[0], 64, 128)
    x_1_8 = torch.randn(batch_size, channels[1], 32, 64)
    x_1_16 = torch.randn(batch_size, channels[2], 16, 32)
    x_1_32 = torch.randn(batch_size, channels[3], 8, 16)
    
    print(f"\n输入:")
    print(f"  1/4分辨率: {x_1_4.shape}")
    print(f"  1/8分辨率: {x_1_8.shape}")
    print(f"  1/16分辨率: {x_1_16.shape}")
    print(f"  1/32分辨率: {x_1_32.shape}")
    
    # 前向传播
    with torch.no_grad():
        b1, b2, b3, b4 = fusion(x_1_4, x_1_8, x_1_16, x_1_32)
    
    print(f"\n输出:")
    print(f"  B1 (1/4分辨率): {b1.shape}")
    print(f"  B2 (1/8分辨率): {b2.shape}")
    print(f"  B3 (1/16分辨率): {b3.shape}")
    print(f"  B4 (1/32分辨率): {b4.shape}")
    
    # 验证输出形状保持不变
    assert b1.shape == x_1_4.shape, "1/4分辨率形状不匹配"
    assert b2.shape == x_1_8.shape, "1/8分辨率形状不匹配"
    assert b3.shape == x_1_16.shape, "1/16分辨率形状不匹配"
    assert b4.shape == x_1_32.shape, "1/32分辨率形状不匹配"
    
    print("\n✓ CRF-4融合模块测试通过")
    
    # 参数统计
    fusion_params = sum(p.numel() for p in fusion.parameters())
    print(f"\nCRF-4融合模块参数量: {fusion_params:,}")
    
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
    
    # 到1/32分支的路径
    to_1_32_params = sum(p.numel() for name, p in fusion.named_parameters() 
                         if 'to_1_32' in name or 'smooth_1_32' in name)
    print(f"  汇到1/32分支: {to_1_32_params:,}")
    
    print("=" * 70)


def test_resolution_hierarchy():
    """测试分辨率层次结构"""
    print("\n" + "=" * 70)
    print("测试分辨率层次结构")
    print("=" * 70)
    
    # 创建Stage4
    stage4_config = get_stage4_config()
    stage4 = create_stage4(stage4_config)
    stage4.eval()
    
    # 创建具有渐变模式的输入
    batch_size = 1
    t1 = torch.ones(batch_size, 32, 64, 128) * 1.0
    t2 = torch.ones(batch_size, 64, 32, 64) * 0.75
    t3 = torch.ones(batch_size, 128, 16, 32) * 0.5
    
    with torch.no_grad():
        outputs = stage4([t1, t2, t3])
    
    print("\n分辨率层次验证:")
    resolutions = []
    for i, out in enumerate(outputs):
        h, w = out.shape[2:]
        scale = 2 ** (i + 2)  # 4, 8, 16, 32
        print(f"  分支{i+1}: 1/{scale}分辨率 = {h}×{w}")
        resolutions.append((h, w))
    
    # 验证分辨率递减
    for i in range(1, len(resolutions)):
        assert resolutions[i][0] == resolutions[i-1][0] // 2, f"高度递减错误"
        assert resolutions[i][1] == resolutions[i-1][1] // 2, f"宽度递减错误"
    
    # 验证通道数递增
    channels = [out.shape[1] for out in outputs]
    assert channels == stage4_config['channels'], f"通道数不匹配: {channels} != {stage4_config['channels']}"
    
    print("\n✓ 分辨率层次结构测试通过")
    print("=" * 70)


def test_complete_backbone():
    """测试完整的主干网络数据流"""
    print("\n" + "=" * 70)
    print("测试完整主干网络数据流（Stem → Stage2 → Stage3 → Stage4）")
    print("=" * 70)
    
    # 配置信息
    stage4_config = get_stage4_config()
    
    # 模拟完整数据流
    batch_size = 2
    
    # Stem输出 (1/4分辨率)
    stem_output = torch.randn(batch_size, 32, 64, 128)
    print(f"\nStem输出: {stem_output.shape}")
    
    # Stage2输入输出
    stage2_in = stem_output
    stage2_out = [
        torch.randn(batch_size, 32, 64, 128),  # Y1
        torch.randn(batch_size, 64, 32, 64)    # Y2
    ]
    print(f"Stage2输出: {[x.shape for x in stage2_out]}")
    
    # Stage3输入输出
    stage3_in = stage2_out
    stage3_out = [
        torch.randn(batch_size, 32, 64, 128),  # T1
        torch.randn(batch_size, 64, 32, 64),   # T2
        torch.randn(batch_size, 128, 16, 32)   # T3
    ]
    print(f"Stage3输出: {[x.shape for x in stage3_out]}")
    
    # Stage4输入输出
    stage4 = create_stage4(stage4_config)
    stage4.eval()
    
    with torch.no_grad():
        stage4_out = stage4(stage3_out)
    
    print(f"Stage4输出（主干最终）: {[x.shape for x in stage4_out]}")
    
    # 验证最终输出
    print("\n主干网络最终输出:")
    for i, out in enumerate(stage4_out):
        scale = 2 ** (i + 2)
        print(f"  B{i+1} (1/{scale}分辨率): {out.shape}")
    
    print("\n✓ 完整主干网络数据流测试通过")
    print("=" * 70)


if __name__ == "__main__":
    # 测试Stage4主模块
    test_stage4()
    
    # 测试跨分辨率融合模块
    test_cross_resolution_fusion4()
    
    # 测试分辨率层次结构
    test_resolution_hierarchy()
    
    # 测试完整主干网络数据流
    test_complete_backbone()
    
    print("\n" + "=" * 70)
    print("🎯 所有Stage4测试通过! ✓")
    print("主干网络（Lite-HRNet-18）构建完成!")
    print("=" * 70)