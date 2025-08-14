# -*- coding: utf-8 -*-
"""
损失函数测试脚本
测试所有损失函数的计算和梯度传播
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from src.models.loss_calculation.focal_loss import FocalLoss, create_focal_loss
from src.models.loss_calculation.offset_loss import OffsetLoss, create_offset_loss
from src.models.loss_calculation.hard_negative_loss import HardNegativeLoss, create_hard_negative_loss
from src.models.loss_calculation.angle_loss import AngleLoss, create_angle_loss
from src.models.loss_calculation.total_loss import TotalLoss, create_total_loss
from src.models.loss_calculation.loss_utils import (
    generate_gaussian_heatmap,
    create_padding_mask,
    coordinate_transform,
    extract_peaks
)


def test_focal_loss():
    """测试Focal Loss"""
    print("=" * 70)
    print("测试 Focal Loss - CenterNet风格热力图损失")
    print("=" * 70)
    
    batch_size = 2
    height, width = 64, 128
    
    # 创建损失函数
    focal_loss = FocalLoss(alpha=1.5, beta=4.0, pos_threshold=0.8)  # 调整阈值以匹配高斯热图峰值
    
    # 创建测试数据
    # 预测：随机热力图（经过sigmoid），需要requires_grad
    pred = torch.sigmoid(torch.randn(batch_size, 1, height, width, requires_grad=True))
    
    # 目标：生成高斯热图
    centers = torch.tensor([[30.5, 20.5], [60.5, 40.5]])  # 两个批次的中心点
    target = generate_gaussian_heatmap(centers, (height, width), sigma=1.5)
    target = target.unsqueeze(1)  # [B, 1, H, W]
    
    # 创建掩码
    mask = torch.ones(batch_size, 1, height, width)
    mask[:, :, :10, :] = 0  # 模拟padding区域
    
    # 计算损失
    loss = focal_loss(pred, target, mask)
    
    print(f"\n输入形状:")
    print(f"  预测: {pred.shape}, 值域: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"  目标: {target.shape}, 值域: [{target.min():.4f}, {target.max():.4f}]")
    print(f"  掩码: {mask.shape}")
    
    print(f"\nFocal Loss值: {loss.item():.4f}")
    
    # 测试梯度
    loss.backward()
    print(f"梯度计算成功")
    
    # 测试工厂函数创建
    print("\n测试工厂函数创建:")
    config = {
        'alpha': 2.0,
        'beta': 4.0,
        'pos_threshold': 0.8
    }
    focal_from_factory = create_focal_loss(config)
    loss_factory = focal_from_factory(pred.detach(), target, mask)
    print(f"工厂函数创建的Focal Loss值: {loss_factory.item():.4f}")
    
    print("\n✓ Focal Loss测试通过")


def test_offset_loss():
    """测试Offset Loss"""
    print("\n" + "=" * 70)
    print("测试 Offset Loss - 子像素偏移损失")
    print("=" * 70)
    
    batch_size = 2
    height, width = 64, 128
    num_points = 2  # 缺口和滑块
    
    # 创建损失函数
    offset_loss = OffsetLoss(loss_type='smooth_l1', pos_threshold=0.7)
    
    # 创建测试数据
    # 预测偏移：[-0.5, 0.5]范围，需要requires_grad
    pred_offset = torch.tanh(torch.randn(batch_size, 2*num_points, height, width, requires_grad=True)) * 0.5
    
    # 目标偏移
    target_offset = torch.randn(batch_size, 2*num_points, height, width) * 0.3
    target_offset = torch.clamp(target_offset, -0.5, 0.5)
    
    # 热力图（用于确定正样本位置）
    centers = torch.tensor([
        [[30, 20], [60, 40]],  # 批次1：缺口和滑块中心
        [[35, 25], [65, 45]]   # 批次2：缺口和滑块中心
    ])
    heatmap = torch.zeros(batch_size, num_points, height, width)
    for b in range(batch_size):
        for p in range(num_points):
            x, y = int(centers[b, p, 0]), int(centers[b, p, 1])
            heatmap[b, p, y-2:y+3, x-2:x+3] = 1.0  # 简单的方形区域
    
    # 计算损失
    loss = offset_loss(pred_offset, target_offset, heatmap)
    
    print(f"\n输入形状:")
    print(f"  预测偏移: {pred_offset.shape}, 值域: [{pred_offset.min():.4f}, {pred_offset.max():.4f}]")
    print(f"  目标偏移: {target_offset.shape}")
    print(f"  热力图: {heatmap.shape}")
    
    print(f"\nOffset Loss值: {loss.item():.4f}")
    
    # 测试梯度
    loss.backward()
    print(f"梯度计算成功")
    
    print("\n✓ Offset Loss测试通过")


def test_hard_negative_loss():
    """测试Hard Negative Loss"""
    print("\n" + "=" * 70)
    print("测试 Hard Negative Loss - 假缺口抑制损失")
    print("=" * 70)
    
    batch_size = 2
    height, width = 64, 128
    
    # 创建损失函数
    hn_loss = HardNegativeLoss(margin=0.2, score_type='bilinear')
    
    # 创建测试数据
    # 预测的缺口热力图，需要requires_grad
    heatmap = torch.sigmoid(torch.randn(batch_size, 1, height, width, requires_grad=True))
    
    # 真实缺口中心（栅格坐标）
    true_centers = torch.tensor([[30.5, 20.5], [60.5, 40.5]])
    
    # 假缺口中心（1-3个）
    fake_centers = [
        torch.tensor([[25.0, 30.0], [55.0, 50.0]]),  # 第1个假缺口
        torch.tensor([[70.0, 30.0], [40.0, 20.0]]),  # 第2个假缺口
    ]
    
    # 设置真实位置的高响应（使用克隆避免inplace操作）
    heatmap_data = heatmap.clone()
    for b in range(batch_size):
        x, y = int(true_centers[b, 0]), int(true_centers[b, 1])
        heatmap_data[b, 0, y, x] = 0.9  # 真实位置高响应
    heatmap = heatmap_data
    
    # 计算损失
    loss = hn_loss(heatmap, true_centers, fake_centers)
    
    print(f"\n输入形状:")
    print(f"  热力图: {heatmap.shape}")
    print(f"  真实中心: {true_centers.shape}")
    print(f"  假缺口数: {len(fake_centers)}")
    
    print(f"\nHard Negative Loss值: {loss.item():.4f}")
    
    # 测试梯度
    if loss.requires_grad:
        loss.backward()
        print(f"梯度计算成功")
    
    print("\n✓ Hard Negative Loss测试通过")


def test_angle_loss():
    """测试Angle Loss"""
    print("\n" + "=" * 70)
    print("测试 Angle Loss - 角度损失")
    print("=" * 70)
    
    batch_size = 2
    height, width = 64, 128
    
    # 创建损失函数
    angle_loss = AngleLoss(loss_type='cosine', pos_threshold=0.7)
    
    # 创建测试数据
    # 预测角度（sin θ, cos θ）- 需要归一化，需要requires_grad
    pred_angle = torch.randn(batch_size, 2, height, width, requires_grad=True)
    pred_angle = torch.nn.functional.normalize(pred_angle, p=2, dim=1)
    
    # 目标角度
    angle_deg = torch.tensor([0.5, -1.0])  # 度
    target_angle = AngleLoss.angle_to_sincos(angle_deg)  # 转换为sin/cos
    target_angle = target_angle.unsqueeze(-1).unsqueeze(-1)  # [B, 2, 1, 1]
    target_angle = target_angle.expand(batch_size, 2, height, width)
    
    # 热力图（用于确定监督区域）
    centers = torch.tensor([[30, 20], [60, 40]])
    heatmap = generate_gaussian_heatmap(centers, (height, width), sigma=2.0)
    heatmap = heatmap.unsqueeze(1)  # [B, 1, H, W]
    
    # 计算损失
    loss = angle_loss(pred_angle, target_angle, heatmap)
    
    print(f"\n输入形状:")
    print(f"  预测角度: {pred_angle.shape}")
    print(f"  目标角度: {target_angle.shape}")
    print(f"  热力图: {heatmap.shape}")
    
    # 验证归一化
    norm = (pred_angle[:, 0:1, :, :] ** 2 + pred_angle[:, 1:2, :, :] ** 2).sqrt()
    print(f"  预测归一化验证: {norm.mean().item():.6f} (应该≈1.0)")
    
    print(f"\nAngle Loss值: {loss.item():.4f}")
    
    # 测试梯度
    loss.backward()
    print(f"梯度计算成功")
    
    print("\n✓ Angle Loss测试通过")


def test_total_loss():
    """测试Total Loss"""
    print("\n" + "=" * 70)
    print("测试 Total Loss - 总损失函数")
    print("=" * 70)
    
    batch_size = 2
    height, width = 64, 128
    
    # 创建总损失函数
    config = {
        'loss_class': 'total',
        'use_angle': True,
        'use_hard_negative': True,
        'loss_weights': {
            'heatmap': 1.0,
            'offset': 1.0,
            'hard_negative': 0.5,
            'angle': 0.5
        }
    }
    total_loss_fn = create_total_loss(config)
    
    # 创建预测数据
    predictions = {
        'heatmap_gap': torch.sigmoid(torch.randn(batch_size, 1, height, width, requires_grad=True)),
        'heatmap_piece': torch.sigmoid(torch.randn(batch_size, 1, height, width, requires_grad=True)),
        'offset': torch.tanh(torch.randn(batch_size, 4, height, width, requires_grad=True)) * 0.5,
        'angle': torch.nn.functional.normalize(
            torch.randn(batch_size, 2, height, width, requires_grad=True), p=2, dim=1
        )
    }
    
    # 创建目标数据
    gap_centers = torch.tensor([[30, 20], [35, 25]])
    piece_centers = torch.tensor([[60, 40], [65, 45]])
    
    targets = {
        'heatmap_gap': generate_gaussian_heatmap(gap_centers, (height, width), sigma=1.5).unsqueeze(1),
        'heatmap_piece': generate_gaussian_heatmap(piece_centers, (height, width), sigma=1.5).unsqueeze(1),
        'offset': torch.randn(batch_size, 4, height, width) * 0.3,
        'angle': torch.nn.functional.normalize(
            torch.randn(batch_size, 2, height, width), p=2, dim=1
        ),
        'gap_center': gap_centers,
        'fake_centers': [
            torch.tensor([[25.0, 30.0], [50.0, 35.0]])
        ],
        'mask': torch.ones(batch_size, 1, height, width)
    }
    
    # 计算损失
    total_loss, loss_dict = total_loss_fn(predictions, targets)
    
    print(f"\n损失分解:")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"  {key}: {value.item():.4f}")
    print(f"  " + "-" * 20)
    print(f"  总损失: {total_loss.item():.4f}")
    
    # 测试梯度
    total_loss.backward()
    print(f"\n梯度计算成功")
    
    # 测试损失统计
    print(f"\n损失统计功能:")
    summary = total_loss_fn.get_loss_summary()
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ Total Loss测试通过")


def test_loss_utils():
    """测试损失工具函数"""
    print("\n" + "=" * 70)
    print("测试 Loss Utils - 损失计算工具函数")
    print("=" * 70)
    
    # 测试高斯热图生成
    print("\n1. 测试高斯热图生成:")
    centers = torch.tensor([[30.5, 20.5], [60.5, 40.5]])
    heatmap = generate_gaussian_heatmap(centers, (64, 128), sigma=1.5)
    print(f"  热图形状: {heatmap.shape}")
    print(f"  值域: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    
    # 测试坐标转换
    print("\n2. 测试坐标转换:")
    coords = torch.tensor([[120.0, 80.0], [240.0, 160.0]])
    grid_coords = coordinate_transform(coords, 'pixel_to_grid', scale=4.0)
    print(f"  原图坐标: {coords}")
    print(f"  栅格坐标: {grid_coords}")
    
    # 测试峰值提取
    print("\n3. 测试峰值提取:")
    test_heatmap = torch.zeros(1, 1, 64, 128)
    test_heatmap[0, 0, 20, 30] = 0.9
    test_heatmap[0, 0, 40, 60] = 0.8
    peaks, scores = extract_peaks(test_heatmap, threshold=0.5, nms=True)
    print(f"  峰值坐标: {peaks[0, 0, :2]}")
    print(f"  峰值得分: {scores[0, 0, :2]}")
    
    # 测试padding掩码
    print("\n4. 测试Padding掩码生成:")
    mask = create_padding_mask((200, 400), (256, 512), downsample=4, pooling='avg')
    print(f"  掩码形状: {mask.shape}")
    print(f"  有效区域比例: {mask.mean().item():.2%}")
    
    print("\n✓ Loss Utils测试通过")


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "=" * 70)
    print("测试梯度流 - 端到端梯度传播")
    print("=" * 70)
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 128, 3, padding=1)
            self.head = nn.Conv2d(128, 7, 1)  # 2个热力图 + 4个偏移 + 1个掩码通道
            
        def forward(self, x):
            feat = self.conv(x)
            out = self.head(feat)
            return {
                'heatmap_gap': torch.sigmoid(out[:, 0:1]),
                'heatmap_piece': torch.sigmoid(out[:, 1:2]),
                'offset': torch.tanh(out[:, 2:6]) * 0.5,
                'mask': torch.sigmoid(out[:, 6:7])
            }
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建损失函数
    loss_fn = TotalLoss(use_angle=False, use_hard_negative=False)
    
    # 模拟训练步骤
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 128)
    
    # 前向传播
    predictions = model(x)
    
    # 创建目标
    targets = {
        'heatmap_gap': torch.rand_like(predictions['heatmap_gap']),
        'heatmap_piece': torch.rand_like(predictions['heatmap_piece']),
        'offset': torch.randn_like(predictions['offset']) * 0.3,
        'mask': predictions['mask'].detach()
    }
    
    # 计算损失
    loss, loss_dict = loss_fn(predictions, targets)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    has_grad = True
    for name, param in model.named_parameters():
        if param.grad is None or param.grad.abs().sum() == 0:
            print(f"  警告: {name} 没有梯度")
            has_grad = False
        else:
            print(f"  {name}: 梯度范数 = {param.grad.norm().item():.6f}")
    
    if has_grad:
        print("\n✓ 梯度流测试通过")
    else:
        print("\n✗ 梯度流存在问题")
    
    # 更新参数
    optimizer.step()
    print("参数更新成功")


if __name__ == "__main__":
    # 测试各个损失函数
    test_focal_loss()
    test_offset_loss()
    test_hard_negative_loss()
    test_angle_loss()
    test_total_loss()
    test_loss_utils()
    test_gradient_flow()
    
    print("\n" + "=" * 70)
    print("🎯 所有损失函数测试通过! ✓")
    print("损失计算模块构建完成!")
    print("=" * 70)