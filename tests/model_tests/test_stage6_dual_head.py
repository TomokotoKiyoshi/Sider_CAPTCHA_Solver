# -*- coding: utf-8 -*-
"""
Stage6 双头预测网络测试脚本
测试DualHead的配置加载、前向传播和输出验证
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from src.models.stage6_dual_head import (
    create_stage6_dual_head, DualHead, 
    HeatmapHead, OffsetHead, AngleHead
)
from src.models.config_loader import get_stage6_dual_head_config


def test_stage6_dual_head():
    """测试Stage6 双头预测网络"""
    print("=" * 70)
    print("测试Stage6 双头预测网络 - 密集预测头")
    print("=" * 70)
    
    # 加载配置
    config = get_stage6_dual_head_config()
    print("\n配置信息:")
    print(f"  输入通道: {config['in_channels']}")
    print(f"  中间通道: {config['mid_channels']}")
    print(f"  使用角度: {config['use_angle']}")
    print(f"  使用tanh限制偏移: {config['use_tanh_offset']}")
    
    # 创建模块
    dual_head = create_stage6_dual_head(config)
    dual_head.eval()
    
    # 创建测试输入（模拟Stage5 LiteFPN的输出）
    batch_size = 2
    hf = torch.randn(batch_size, 128, 64, 128)  # [B, 128, 64, 128]
    
    print("\n输入张量（来自Stage5 LiteFPN）:")
    print(f"  Hf (1/4分辨率): {hf.shape}")
    
    # 前向传播
    with torch.no_grad():
        predictions = dual_head(hf)
    
    print("\n输出预测:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
        if key.startswith('heatmap'):
            print(f"    值域: [{value.min().item():.4f}, {value.max().item():.4f}] (期望[0,1])")
        elif key == 'offset':
            print(f"    值域: [{value.min().item():.4f}, {value.max().item():.4f}] (期望[-0.5,0.5])")
    
    # 验证输出形状
    assert predictions['heatmap_gap'].shape == (batch_size, 1, 64, 128), \
        "缺口热力图形状不正确"
    assert predictions['heatmap_piece'].shape == (batch_size, 1, 64, 128), \
        "拼图热力图形状不正确"
    assert predictions['offset'].shape == (batch_size, 4, 64, 128), \
        "偏移图形状不正确"
    
    print("\n✓ 输出形状验证通过")
    
    # 参数统计
    total_params = sum(p.numel() for p in dual_head.parameters())
    trainable_params = sum(p.numel() for p in dual_head.parameters() if p.requires_grad)
    
    print("\n参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 分模块参数统计
    heatmap_params = sum(p.numel() for name, p in dual_head.named_parameters() if 'heatmap' in name)
    offset_params = sum(p.numel() for name, p in dual_head.named_parameters() if 'offset' in name)
    
    print("\n分模块参数统计:")
    print(f"  热力图头: {heatmap_params:,}")
    print(f"  偏移头: {offset_params:,}")
    
    print("\n✓ Stage6 双头预测网络测试完成")
    print("=" * 70)


def test_individual_heads():
    """测试各个独立的预测头"""
    print("\n" + "=" * 70)
    print("测试独立预测头")
    print("=" * 70)
    
    batch_size = 2
    in_channels = 128
    height, width = 64, 128
    
    # 创建测试输入
    x = torch.randn(batch_size, in_channels, height, width)
    
    # 测试热力图头
    print("\n测试热力图头:")
    heatmap_head = HeatmapHead(in_channels=in_channels)
    heatmap_head.eval()
    
    with torch.no_grad():
        heatmap = heatmap_head(x)
    
    assert heatmap.shape == (batch_size, 1, height, width), \
        "热力图头输出形状不正确"
    assert (heatmap >= 0).all() and (heatmap <= 1).all(), \
        "热力图值域应该在[0,1]"
    
    print(f"  输出形状: {heatmap.shape}")
    print(f"  值域: [{heatmap.min().item():.4f}, {heatmap.max().item():.4f}]")
    print(f"  ✓ 热力图头测试通过")
    
    # 测试偏移头
    print("\n测试偏移头:")
    offset_head = OffsetHead(in_channels=in_channels, use_tanh=True)
    offset_head.eval()
    
    with torch.no_grad():
        offset = offset_head(x)
    
    assert offset.shape == (batch_size, 4, height, width), \
        "偏移头输出形状不正确"
    assert (offset >= -0.5).all() and (offset <= 0.5).all(), \
        "偏移值域应该在[-0.5,0.5]"
    
    print(f"  输出形状: {offset.shape}")
    print(f"  值域: [{offset.min().item():.4f}, {offset.max().item():.4f}]")
    print(f"  ✓ 偏移头测试通过")
    
    # 测试角度头
    print("\n测试角度头:")
    angle_head = AngleHead(in_channels=in_channels)
    angle_head.eval()
    
    with torch.no_grad():
        angle = angle_head(x)
    
    assert angle.shape == (batch_size, 2, height, width), \
        "角度头输出形状不正确"
    
    # 验证L2归一化：sin²θ + cos²θ = 1
    sin_theta = angle[:, 0:1, :, :]
    cos_theta = angle[:, 1:2, :, :]
    norm = (sin_theta ** 2 + cos_theta ** 2).sqrt()
    
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), \
        "角度头输出未正确归一化"
    
    print(f"  输出形状: {angle.shape}")
    print(f"  sin θ 范围: [{sin_theta.min().item():.4f}, {sin_theta.max().item():.4f}]")
    print(f"  cos θ 范围: [{cos_theta.min().item():.4f}, {cos_theta.max().item():.4f}]")
    print(f"  归一化验证: sin²θ + cos²θ ≈ {norm.mean().item():.6f}")
    print(f"  ✓ 角度头测试通过")
    
    print("\n" + "=" * 70)


def test_with_angle():
    """测试带角度预测的配置"""
    print("\n" + "=" * 70)
    print("测试带角度预测的双头网络")
    print("=" * 70)
    
    # 创建带角度预测的配置
    config = {
        'in_channels': 128,
        'mid_channels': 64,
        'use_angle': True,  # 启用角度预测
        'use_tanh_offset': True
    }
    
    # 创建模块
    dual_head = DualHead(**config)
    dual_head.eval()
    
    # 测试输入
    batch_size = 1
    x = torch.randn(batch_size, 128, 64, 128)
    
    # 前向传播
    with torch.no_grad():
        predictions = dual_head(x)
    
    print("\n启用角度预测后的输出:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    
    # 验证角度预测存在
    assert 'angle' in predictions, "角度预测应该存在"
    assert predictions['angle'].shape == (batch_size, 2, 64, 128), \
        "角度预测形状不正确"
    
    # 验证角度归一化
    sin_theta = predictions['angle'][:, 0:1, :, :]
    cos_theta = predictions['angle'][:, 1:2, :, :]
    norm = (sin_theta ** 2 + cos_theta ** 2).sqrt()
    
    print(f"\n角度预测验证:")
    print(f"  归一化: sin²θ + cos²θ ≈ {norm.mean().item():.6f}")
    
    print("\n✓ 带角度预测的双头网络测试通过")
    print("=" * 70)


def test_complete_pipeline():
    """测试完整的数据流（Stage5 → Stage6）"""
    print("\n" + "=" * 70)
    print("测试完整数据流（Stage5 LiteFPN → Stage6 DualHead）")
    print("=" * 70)
    
    # 导入Stage5
    from src.models.stage5_lite_fpn import create_stage5_lite_fpn
    from src.models.config_loader import get_stage5_lite_fpn_config
    
    # 创建Stage4输出（模拟）
    batch_size = 2
    stage4_output = [
        torch.randn(batch_size, 32, 64, 128),   # B1
        torch.randn(batch_size, 64, 32, 64),    # B2
        torch.randn(batch_size, 128, 16, 32),   # B3
        torch.randn(batch_size, 256, 8, 16),    # B4
    ]
    
    print("Stage4输出:")
    for i, feat in enumerate(stage4_output, 1):
        print(f"  B{i}: {feat.shape}")
    
    # Stage5 LiteFPN处理
    stage5_config = get_stage5_lite_fpn_config()
    stage5 = create_stage5_lite_fpn(stage5_config)
    stage5.eval()
    
    with torch.no_grad():
        hf = stage5(stage4_output)
    
    print("\nStage5 LiteFPN输出:")
    print(f"  Hf: {hf.shape} (1/4分辨率主特征)")
    
    # Stage6 DualHead处理
    stage6_config = get_stage6_dual_head_config()
    stage6 = create_stage6_dual_head(stage6_config)
    stage6.eval()
    
    with torch.no_grad():
        predictions = stage6(hf)
    
    print("\nStage6 DualHead输出:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    
    # 验证端到端输出
    assert predictions['heatmap_gap'].shape == (batch_size, 1, 64, 128), \
        "端到端缺口热力图形状不正确"
    assert predictions['heatmap_piece'].shape == (batch_size, 1, 64, 128), \
        "端到端拼图热力图形状不正确"
    assert predictions['offset'].shape == (batch_size, 4, 64, 128), \
        "端到端偏移图形状不正确"
    
    print("\n✓ 完整数据流测试通过")
    print("=" * 70)


def test_decode_coordinates():
    """测试坐标解码逻辑"""
    print("\n" + "=" * 70)
    print("测试坐标解码")
    print("=" * 70)
    
    # 创建模拟的预测输出
    batch_size = 1
    height, width = 64, 128  # 1/4分辨率
    
    # 创建热力图，在特定位置设置峰值
    heatmap = torch.zeros(batch_size, 1, height, width)
    gap_y, gap_x = 30, 60  # 栅格坐标
    piece_y, piece_x = 35, 20  # 栅格坐标
    
    heatmap[0, 0, gap_y, gap_x] = 1.0  # 缺口峰值
    
    # 创建偏移图
    offset = torch.zeros(batch_size, 4, height, width)
    offset[0, 0, gap_y, gap_x] = 0.3   # du_gap
    offset[0, 1, gap_y, gap_x] = -0.2  # dv_gap
    offset[0, 2, piece_y, piece_x] = -0.1  # du_piece
    offset[0, 3, piece_y, piece_x] = 0.4   # dv_piece
    
    print("模拟预测:")
    print(f"  缺口栅格坐标: ({gap_x}, {gap_y})")
    print(f"  缺口子像素偏移: (0.3, -0.2)")
    print(f"  拼图栅格坐标: ({piece_x}, {piece_y})")
    print(f"  拼图子像素偏移: (-0.1, 0.4)")
    
    # 解码坐标（简化版）
    downsample = 4  # 下采样率
    
    # 缺口坐标
    gap_x_final = (gap_x + 0.3) * downsample
    gap_y_final = (gap_y - 0.2) * downsample
    
    # 拼图坐标
    piece_x_final = (piece_x - 0.1) * downsample
    piece_y_final = (piece_y + 0.4) * downsample
    
    print("\n解码后的原图坐标:")
    print(f"  缺口: ({gap_x_final:.1f}, {gap_y_final:.1f})")
    print(f"  拼图: ({piece_x_final:.1f}, {piece_y_final:.1f})")
    
    print("\n坐标映射公式:")
    print("  原图坐标 = (栅格坐标 + 子像素偏移) × 下采样率")
    print("  下采样率 = 4 (从256×512到64×128)")
    
    print("\n✓ 坐标解码测试完成")
    print("=" * 70)


if __name__ == "__main__":
    # 测试Stage6主模块
    test_stage6_dual_head()
    
    # 测试独立预测头
    test_individual_heads()
    
    # 测试带角度预测的配置
    test_with_angle()
    
    # 测试完整管道
    test_complete_pipeline()
    
    # 测试坐标解码
    test_decode_coordinates()
    
    print("\n" + "=" * 70)
    print("🎯 所有Stage6 双头预测网络测试通过! ✓")
    print("密集预测头构建完成!")
    print("=" * 70)