# -*- coding: utf-8 -*-
"""
Stage5 LiteFPN模块测试脚本
测试LiteFPN的配置加载、前向传播和特征金字塔融合
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from src.models.stage5_lite_fpn import create_stage5_lite_fpn, LiteFPN, FusionModule
from src.models.config_loader import get_stage5_lite_fpn_config


def test_stage5_lite_fpn():
    """测试Stage5 LiteFPN模块"""
    print("=" * 70)
    print("测试Stage5 LiteFPN模块 - 轻量级特征金字塔网络")
    print("=" * 70)
    
    # 加载配置
    config = get_stage5_lite_fpn_config()
    print("\n配置信息:")
    print(f"  输入通道: {config['in_channels']}")
    print(f"  FPN通道: {config['fpn_channels']}")
    print(f"  融合类型: {config['fusion_type']}")
    print(f"  返回金字塔: {config['return_pyramid']}")
    
    # 创建模块
    lite_fpn = create_stage5_lite_fpn(config)
    lite_fpn.eval()
    
    # 创建测试输入（模拟Stage4的输出）
    batch_size = 2
    features = [
        torch.randn(batch_size, 32, 64, 128),   # B1 (1/4分辨率)
        torch.randn(batch_size, 64, 32, 64),    # B2 (1/8分辨率)
        torch.randn(batch_size, 128, 16, 32),   # B3 (1/16分辨率)
        torch.randn(batch_size, 256, 8, 16),    # B4 (1/32分辨率)
    ]
    
    print("\n输入张量（来自Stage4）:")
    for i, feat in enumerate(features, 1):
        print(f"  B{i} (1/{4*(2**(i-1))}分辨率): {feat.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = lite_fpn(features)
    
    print("\n输出信息:")
    if isinstance(output, tuple):
        hf, pyramid = output
        print(f"  主特征 Hf (1/4分辨率): {hf.shape}")
        print(f"  中间金字塔特征:")
        for i, feat in enumerate(pyramid):
            scale = 8 * (2**i)
            print(f"    P{i+3}_td (1/{scale}分辨率): {feat.shape}")
    else:
        print(f"  主特征 Hf (1/4分辨率): {output.shape}")
    
    # 验证输出形状
    expected_shape = (batch_size, config['fpn_channels'], 64, 128)
    actual_shape = output.shape if not isinstance(output, tuple) else output[0].shape
    assert actual_shape == expected_shape, \
        f"输出形状不匹配: 期望{expected_shape}, 实际{actual_shape}"
    
    print("\n✓ 输出形状验证通过")
    
    # 参数统计
    total_params = sum(p.numel() for p in lite_fpn.parameters())
    trainable_params = sum(p.numel() for p in lite_fpn.parameters() if p.requires_grad)
    
    print("\n参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 分模块参数统计
    lateral_params = sum(p.numel() for name, p in lite_fpn.named_parameters() if 'lateral' in name)
    smooth_params = sum(p.numel() for name, p in lite_fpn.named_parameters() if 'smooth' in name)
    fusion_params = sum(p.numel() for name, p in lite_fpn.named_parameters() if 'fuse' in name)
    
    print("\n分模块参数统计:")
    print(f"  侧连接模块: {lateral_params:,}")
    print(f"  平滑卷积: {smooth_params:,}")
    print(f"  融合模块: {fusion_params:,}")
    
    print("\n✓ Stage5 LiteFPN模块测试完成")
    print("=" * 70)


def test_fusion_modules():
    """测试不同的融合模块"""
    print("\n" + "=" * 70)
    print("测试融合模块")
    print("=" * 70)
    
    channels = 128
    batch_size = 2
    height, width = 32, 64
    
    # 创建测试输入
    a = torch.randn(batch_size, channels, height, width)
    b = torch.randn(batch_size, channels, height, width)
    
    # 测试不同融合类型
    fusion_types = ['add', 'weighted', 'attention']
    
    for fusion_type in fusion_types:
        print(f"\n测试 {fusion_type} 融合:")
        
        fusion_module = FusionModule(channels, fusion_type)
        fusion_module.eval()
        
        with torch.no_grad():
            output = fusion_module(a, b)
        
        assert output.shape == a.shape, \
            f"{fusion_type}融合输出形状不匹配"
        
        # 参数统计
        params = sum(p.numel() for p in fusion_module.parameters())
        print(f"  输出形状: {output.shape}")
        print(f"  参数量: {params:,}")
        
        if fusion_type == 'weighted':
            # 检查权重参数
            print(f"  权重a: {fusion_module.weight_a.item():.4f}")
            print(f"  权重b: {fusion_module.weight_b.item():.4f}")
        
        print(f"  ✓ {fusion_type}融合测试通过")
    
    print("\n" + "=" * 70)


def test_resolution_consistency():
    """测试分辨率一致性"""
    print("\n" + "=" * 70)
    print("测试分辨率一致性")
    print("=" * 70)
    
    # 加载配置
    config = get_stage5_lite_fpn_config()
    config['return_pyramid'] = True  # 返回中间金字塔
    
    # 创建模块
    lite_fpn = LiteFPN(**config)
    lite_fpn.eval()
    
    # 创建测试输入
    batch_size = 1
    features = [
        torch.randn(batch_size, 32, 64, 128),   # 1/4分辨率
        torch.randn(batch_size, 64, 32, 64),    # 1/8分辨率
        torch.randn(batch_size, 128, 16, 32),   # 1/16分辨率
        torch.randn(batch_size, 256, 8, 16),    # 1/32分辨率
    ]
    
    # 前向传播
    with torch.no_grad():
        hf, pyramid = lite_fpn(features)
    
    print("\n分辨率验证:")
    
    # 主特征
    print(f"  主特征 Hf: {hf.shape} (期望1/4分辨率: 64×128)")
    assert hf.shape[2:] == (64, 128), "主特征分辨率不正确"
    
    # 中间金字塔
    expected_resolutions = [(32, 64), (16, 32), (8, 16)]
    pyramid_names = ['P3_td', 'P4_td', 'P5']
    
    for name, feat, expected in zip(pyramid_names, pyramid, expected_resolutions):
        print(f"  {name}: {feat.shape} (期望: {expected})")
        assert feat.shape[2:] == expected, f"{name}分辨率不正确"
    
    print("\n✓ 分辨率一致性测试通过")
    print("=" * 70)


def test_complete_pipeline():
    """测试完整的数据流（Stage4 → Stage5）"""
    print("\n" + "=" * 70)
    print("测试完整数据流（Stage4 → Stage5 LiteFPN）")
    print("=" * 70)
    
    # 导入Stage4
    from src.models.stage4 import create_stage4
    from src.models.config_loader import get_stage4_config, get_stage3_config
    from src.models.stage3 import create_stage3
    
    # 创建Stage3输出（模拟）
    batch_size = 2
    stage3_output = [
        torch.randn(batch_size, 32, 64, 128),   # T1
        torch.randn(batch_size, 64, 32, 64),    # T2
        torch.randn(batch_size, 128, 16, 32),   # T3
    ]
    
    print("Stage3输出:")
    for i, feat in enumerate(stage3_output, 1):
        print(f"  T{i}: {feat.shape}")
    
    # Stage4处理
    stage4_config = get_stage4_config()
    stage4 = create_stage4(stage4_config)
    stage4.eval()
    
    with torch.no_grad():
        stage4_output = stage4(stage3_output)
    
    print("\nStage4输出（主干最终）:")
    for i, feat in enumerate(stage4_output, 1):
        print(f"  B{i}: {feat.shape}")
    
    # Stage5 LiteFPN处理
    stage5_config = get_stage5_lite_fpn_config()
    stage5 = create_stage5_lite_fpn(stage5_config)
    stage5.eval()
    
    with torch.no_grad():
        final_output = stage5(stage4_output)
    
    print("\nStage5 LiteFPN输出:")
    print(f"  主特征 Hf: {final_output.shape}")
    print(f"  通道数: {final_output.shape[1]} (统一到128)")
    print(f"  分辨率: 1/4 (64×128)")
    
    # 验证最终输出
    assert final_output.shape == (batch_size, 128, 64, 128), \
        "最终输出形状不正确"
    
    print("\n✓ 完整数据流测试通过")
    print("=" * 70)


if __name__ == "__main__":
    # 测试Stage5 LiteFPN主模块
    test_stage5_lite_fpn()
    
    # 测试融合模块
    test_fusion_modules()
    
    # 测试分辨率一致性
    test_resolution_consistency()
    
    # 测试完整管道
    test_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("🎯 所有Stage5 LiteFPN测试通过! ✓")
    print("特征金字塔网络构建完成!")
    print("=" * 70)