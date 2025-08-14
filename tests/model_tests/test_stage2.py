# -*- coding: utf-8 -*-
"""
Stage2模块测试脚本
测试Stage2的配置加载和前向传播
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from src.models.stage2 import create_stage2
from src.models.config_loader import get_stage2_config


def test_stage2():
    """测试Stage2模块"""
    print("=" * 50)
    print("测试Stage2模块")
    print("=" * 50)
    
    # 加载配置
    stage2_config = get_stage2_config()
    print("\n配置信息:")
    print(f"  输入通道: {stage2_config['in_channels']}")
    print(f"  分支通道: {stage2_config['channels']}")
    print(f"  块数量: {stage2_config['num_blocks']}")
    print(f"  扩张倍率: {stage2_config['expansion']}")
    
    # 创建模块
    stage2 = create_stage2(stage2_config)
    stage2.eval()
    
    # 测试输入 (来自Stage1的输出)
    batch_size = 2
    in_channels = stage2_config['in_channels']
    height, width = 64, 128  # 1/4分辨率
    x = torch.randn(batch_size, in_channels, height, width)
    
    print(f"\n输入张量形状: {x.shape}")
    print(f"  批次大小: {batch_size}")
    print(f"  通道数: {in_channels}")
    print(f"  高度: {height}")
    print(f"  宽度: {width}")
    
    # 前向传播
    with torch.no_grad():
        outputs = stage2(x)
    
    # 输出信息
    print(f"\n输出信息:")
    print(f"  输出数量: {len(outputs)}")
    print(f"  1/4分辨率输出: {outputs[0].shape}")
    print(f"  1/8分辨率输出: {outputs[1].shape}")
    
    # 验证输出形状
    expected_shape_1_4 = (batch_size, stage2_config['channels'][0], height, width)
    expected_shape_1_8 = (batch_size, stage2_config['channels'][1], height//2, width//2)
    
    assert outputs[0].shape == expected_shape_1_4, f"1/4分辨率输出形状错误: {outputs[0].shape} != {expected_shape_1_4}"
    assert outputs[1].shape == expected_shape_1_8, f"1/8分辨率输出形状错误: {outputs[1].shape} != {expected_shape_1_8}"
    
    print("\n✓ 输出形状验证通过")
    
    # 参数统计
    total_params = sum(p.numel() for p in stage2.parameters())
    trainable_params = sum(p.numel() for p in stage2.parameters() if p.requires_grad)
    
    print(f"\n参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 分模块参数统计
    print(f"\n分模块参数统计:")
    
    # Transition参数
    t2_params = sum(p.numel() for name, p in stage2.named_parameters() if 't2_' in name)
    print(f"  Transition模块: {t2_params:,}")
    
    # 分支参数
    branch_1_4_params = sum(p.numel() for name, p in stage2.named_parameters() if 'branch_1_4' in name)
    branch_1_8_params = sum(p.numel() for name, p in stage2.named_parameters() if 'branch_1_8' in name)
    print(f"  1/4分支: {branch_1_4_params:,}")
    print(f"  1/8分支: {branch_1_8_params:,}")
    
    # 融合模块参数
    fusion_params = sum(p.numel() for name, p in stage2.named_parameters() if 'fusion' in name)
    print(f"  融合模块: {fusion_params:,}")
    
    print("\n✓ Stage2模块测试完成")
    print("=" * 50)


def test_cross_resolution_fusion():
    """测试跨分辨率融合模块"""
    print("\n" + "=" * 50)
    print("测试跨分辨率融合模块")
    print("=" * 50)
    
    from src.models.stage2 import CrossResolutionFusion
    
    # 创建融合模块
    channels_1_4 = 32
    channels_1_8 = 64
    fusion = CrossResolutionFusion(channels_1_4, channels_1_8)
    fusion.eval()
    
    # 测试输入
    batch_size = 2
    x_1_4 = torch.randn(batch_size, channels_1_4, 64, 128)
    x_1_8 = torch.randn(batch_size, channels_1_8, 32, 64)
    
    print(f"\n输入:")
    print(f"  1/4分辨率: {x_1_4.shape}")
    print(f"  1/8分辨率: {x_1_8.shape}")
    
    # 前向传播
    with torch.no_grad():
        y_1_4, y_1_8 = fusion(x_1_4, x_1_8)
    
    print(f"\n输出:")
    print(f"  1/4分辨率: {y_1_4.shape}")
    print(f"  1/8分辨率: {y_1_8.shape}")
    
    # 验证输出形状
    assert y_1_4.shape == x_1_4.shape, f"1/4分辨率形状不匹配"
    assert y_1_8.shape == x_1_8.shape, f"1/8分辨率形状不匹配"
    
    print("\n✓ 融合模块测试通过")
    
    # 参数统计
    fusion_params = sum(p.numel() for p in fusion.parameters())
    print(f"\n融合模块参数量: {fusion_params:,}")
    print("=" * 50)


if __name__ == "__main__":
    # 测试Stage2主模块
    test_stage2()
    
    # 测试跨分辨率融合模块
    test_cross_resolution_fusion()
    
    print("\n所有测试通过! ✓")