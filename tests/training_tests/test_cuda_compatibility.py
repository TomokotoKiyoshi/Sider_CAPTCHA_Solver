# -*- coding: utf-8 -*-
"""
测试损失函数的CUDA兼容性
"""
import torch
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.loss_calculation.focal_loss import create_focal_loss
from src.models.loss_calculation.offset_loss import create_offset_loss
from src.models.loss_calculation.hard_negative_loss import create_hard_negative_loss
from src.models.loss_calculation.angle_loss import create_angle_loss
from src.models.loss_calculation.padding_bce_loss import create_padding_bce_loss
from src.models.loss_calculation.y_axis_loss import create_y_axis_loss
from src.models.loss_calculation.total_loss import create_total_loss
from src.models.loss_calculation.config_loader import ConfigLoader


def create_test_data(device='cuda'):
    """创建测试数据"""
    B, H, W = 2, 64, 128
    
    data = {
        # 预测
        'pred_heatmap_gap': torch.rand(B, 1, H, W, device=device),
        'pred_heatmap_piece': torch.rand(B, 1, H, W, device=device),
        'pred_offset': torch.randn(B, 4, H, W, device=device) * 0.5,
        'pred_angle': torch.nn.functional.normalize(
            torch.randn(B, 2, H, W, device=device), p=2, dim=1
        ),
        
        # 目标
        'target_heatmap_gap': torch.rand(B, 1, H, W, device=device),
        'target_heatmap_piece': torch.rand(B, 1, H, W, device=device),
        'target_offset': torch.randn(B, 4, H, W, device=device) * 0.5,
        'target_angle': torch.nn.functional.normalize(
            torch.randn(B, 2, H, W, device=device), p=2, dim=1
        ),
        
        # 掩码
        'mask': torch.ones(B, 1, H, W, device=device),
        'padding_mask': torch.randint(0, 2, (B, 1, H, W), device=device).float(),
        
        # 中心点
        'gap_center': torch.rand(B, 2, device=device) * 64,
        'piece_center': torch.rand(B, 2, device=device) * 64,
        'fake_centers': [torch.rand(B, 2, device=device) * 64 for _ in range(2)]
    }
    
    return data


def test_focal_loss(loss_config, data, device):
    """测试Focal Loss"""
    print("\n测试 Focal Loss...")
    focal_loss = create_focal_loss(loss_config['focal_loss'])
    focal_loss = focal_loss.to(device)
    
    loss_focal = focal_loss(
        data['pred_heatmap_gap'], 
        data['target_heatmap_gap'], 
        data['mask']
    )
    
    print(f"Focal Loss: {loss_focal.item():.6f}")
    assert loss_focal.device.type == device.type, \
        f"Focal Loss 不在正确的设备上: {loss_focal.device} vs {device}"
    
    return loss_focal


def test_offset_loss(loss_config, data, device):
    """测试Offset Loss"""
    print("\n测试 Offset Loss...")
    offset_loss = create_offset_loss(loss_config['offset_loss'])
    offset_loss = offset_loss.to(device)
    
    heatmap_combined = torch.cat([
        data['target_heatmap_gap'], 
        data['target_heatmap_piece']
    ], dim=1)
    
    loss_offset = offset_loss(
        data['pred_offset'], 
        data['target_offset'], 
        heatmap_combined, 
        data['mask']
    )
    
    print(f"Offset Loss: {loss_offset.item():.6f}")
    assert loss_offset.device.type == device.type, \
        f"Offset Loss 不在正确的设备上: {loss_offset.device} vs {device}"
    
    return loss_offset


def test_hard_negative_loss(loss_config, data, device):
    """测试Hard Negative Loss"""
    print("\n测试 Hard Negative Loss...")
    hn_loss = create_hard_negative_loss(loss_config['hard_negative_loss'])
    hn_loss = hn_loss.to(device)
    
    loss_hn = hn_loss(
        data['pred_heatmap_gap'], 
        data['gap_center'], 
        data['fake_centers'], 
        data['mask']
    )
    
    print(f"Hard Negative Loss: {loss_hn.item():.6f}")
    assert loss_hn.device.type == device.type, \
        f"Hard Negative Loss 不在正确的设备上: {loss_hn.device} vs {device}"
    
    return loss_hn


def test_angle_loss(loss_config, data, device):
    """测试Angle Loss"""
    print("\n测试 Angle Loss...")
    angle_loss = create_angle_loss(loss_config['angle_loss'])
    angle_loss = angle_loss.to(device)
    
    loss_angle = angle_loss(
        data['pred_angle'], 
        data['target_angle'], 
        data['target_heatmap_gap'], 
        data['mask']
    )
    
    print(f"Angle Loss: {loss_angle.item():.6f}")
    assert loss_angle.device.type == device.type, \
        f"Angle Loss 不在正确的设备上: {loss_angle.device} vs {device}"
    
    return loss_angle


def test_padding_bce_loss(loss_config, data, device):
    """测试Padding BCE Loss"""
    print("\n测试 Padding BCE Loss...")
    
    if not loss_config.get('padding_bce_loss', {}).get('enabled', False):
        print("Padding BCE Loss 未启用，跳过测试")
        return None
    
    padding_bce_loss = create_padding_bce_loss(loss_config['padding_bce_loss'])
    padding_bce_loss = padding_bce_loss.to(device)
    
    loss_padding = padding_bce_loss(
        data['pred_heatmap_gap'], 
        data['pred_heatmap_piece'],
        data['padding_mask']
    )
    
    print(f"Padding BCE Loss: {loss_padding.item():.6f}")
    assert loss_padding.device.type == device.type, \
        f"Padding BCE Loss 不在正确的设备上: {loss_padding.device} vs {device}"
    
    return loss_padding


def test_y_axis_loss(loss_config, data, device):
    """测试Y-Axis Loss"""
    print("\n测试 Y-Axis Loss...")
    
    if not loss_config.get('y_axis_loss', {}).get('enabled', False):
        print("Y-Axis Loss 未启用，跳过测试")
        return None
    
    y_axis_loss = create_y_axis_loss(loss_config['y_axis_loss'])
    y_axis_loss = y_axis_loss.to(device)
    
    # YAxisLoss只需要y坐标，从center中提取
    gap_y = data['gap_center'][:, 1]  # 提取y坐标 [B]
    piece_y = data['piece_center'][:, 1]  # 提取y坐标 [B]
    
    loss_y_axis, loss_dict = y_axis_loss(
        data['pred_heatmap_gap'], 
        data['pred_heatmap_piece'],
        gap_y * 4,  # 原始分辨率坐标（测试数据是1/4分辨率的）
        piece_y * 4   # 原始分辨率坐标
    )
    
    print(f"Y-Axis Loss: {loss_y_axis.item():.6f}")
    print("  Y-Axis子损失:")
    for key, value in loss_dict.items():
        if 'y_axis' in key:
            print(f"    {key}: {value.item():.6f}")
    assert loss_y_axis.device.type == device.type, \
        f"Y-Axis Loss 不在正确的设备上: {loss_y_axis.device} vs {device}"
    
    return loss_y_axis


def test_total_loss(loss_config, data, device):
    """测试Total Loss"""
    print("\n测试 Total Loss...")
    
    # 注意：create_total_loss需要完整的配置，不仅仅是total_loss部分
    total_loss = create_total_loss(loss_config)
    total_loss = total_loss.to(device)
    
    predictions = {
        'heatmap_gap': data['pred_heatmap_gap'],
        'heatmap_piece': data['pred_heatmap_piece'],
        'offset': data['pred_offset'],
        'angle': data['pred_angle']
    }
    
    targets = {
        'heatmap_gap': data['target_heatmap_gap'],
        'heatmap_piece': data['target_heatmap_piece'],
        'offset': data['target_offset'],
        'angle': data['target_angle'],
        'mask': data['mask'],
        'padding_mask': data['padding_mask'],
        'gap_center': data['gap_center'],
        'piece_center': data['piece_center'],
        'fake_centers': data['fake_centers']
    }
    
    loss_total, loss_dict = total_loss(predictions, targets)
    
    print(f"Total Loss: {loss_total.item():.6f}")
    print("子损失:")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"  {key}: {value.item():.6f}")
    
    assert loss_total.device.type == device.type, \
        f"Total Loss 不在正确的设备上: {loss_total.device} vs {device}"
    
    return loss_total, loss_dict


def test_backward_propagation(loss):
    """测试反向传播"""
    print("\n测试反向传播...")
    loss.backward()
    print("反向传播成功!")


def test_cuda_compatibility():
    """测试所有损失函数的CUDA兼容性"""
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    config_loader = ConfigLoader('config/loss.yaml')
    loss_config = config_loader.config
    
    # 创建测试数据
    data = create_test_data(device)
    
    # 测试各个损失函数
    test_focal_loss(loss_config, data, device)
    test_offset_loss(loss_config, data, device)
    test_hard_negative_loss(loss_config, data, device)
    test_angle_loss(loss_config, data, device)
    test_padding_bce_loss(loss_config, data, device)
    test_y_axis_loss(loss_config, data, device)
    
    # 测试总损失
    loss_total, _ = test_total_loss(loss_config, data, device)
    
    # 测试反向传播
    test_backward_propagation(loss_total)
    
    print("\n所有CUDA兼容性测试通过! ✅")


if __name__ == "__main__":
    test_cuda_compatibility()