# -*- coding: utf-8 -*-
"""
测试损失函数的CUDA兼容性
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.loss_calculation.focal_loss import create_focal_loss
from src.models.loss_calculation.offset_loss import create_offset_loss
from src.models.loss_calculation.hard_negative_loss import create_hard_negative_loss
from src.models.loss_calculation.angle_loss import create_angle_loss
from src.models.loss_calculation.total_loss import create_total_loss
from src.models.loss_calculation.config_loader import ConfigLoader


def test_cuda_compatibility():
    """测试所有损失函数的CUDA兼容性"""
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    B, H, W = 2, 64, 128
    
    # 预测
    pred_heatmap_gap = torch.rand(B, 1, H, W, device=device)
    pred_heatmap_piece = torch.rand(B, 1, H, W, device=device)
    pred_offset = torch.randn(B, 4, H, W, device=device) * 0.5
    pred_angle = torch.randn(B, 2, H, W, device=device)
    pred_angle = torch.nn.functional.normalize(pred_angle, p=2, dim=1)
    
    # 目标
    target_heatmap_gap = torch.rand(B, 1, H, W, device=device)
    target_heatmap_piece = torch.rand(B, 1, H, W, device=device)
    target_offset = torch.randn(B, 4, H, W, device=device) * 0.5
    target_angle = torch.randn(B, 2, H, W, device=device)
    target_angle = torch.nn.functional.normalize(target_angle, p=2, dim=1)
    
    # 掩码
    mask = torch.ones(B, 1, H, W, device=device)
    
    # 缺口中心和假缺口中心
    gap_center = torch.rand(B, 2, device=device) * 64
    fake_centers = [torch.rand(B, 2, device=device) * 64 for _ in range(2)]
    
    # 加载配置
    config_loader = ConfigLoader('config/loss.yaml')
    loss_config = config_loader.config
    
    # 1. 测试Focal Loss
    print("\n测试 Focal Loss...")
    focal_loss = create_focal_loss(loss_config['focal_loss'])
    focal_loss = focal_loss.to(device)
    loss_focal = focal_loss(pred_heatmap_gap, target_heatmap_gap, mask)
    print(f"Focal Loss: {loss_focal.item():.6f}")
    assert loss_focal.device.type == device.type, f"Focal Loss 不在正确的设备上: {loss_focal.device} vs {device}"
    
    # 2. 测试Offset Loss
    print("\n测试 Offset Loss...")
    offset_loss = create_offset_loss(loss_config['offset_loss'])
    offset_loss = offset_loss.to(device)
    heatmap_combined = torch.cat([target_heatmap_gap, target_heatmap_piece], dim=1)
    loss_offset = offset_loss(pred_offset, target_offset, heatmap_combined, mask)
    print(f"Offset Loss: {loss_offset.item():.6f}")
    assert loss_offset.device.type == device.type, f"Offset Loss 不在正确的设备上: {loss_offset.device} vs {device}"
    
    # 3. 测试Hard Negative Loss
    print("\n测试 Hard Negative Loss...")
    hn_loss = create_hard_negative_loss(loss_config['hard_negative_loss'])
    hn_loss = hn_loss.to(device)
    loss_hn = hn_loss(pred_heatmap_gap, gap_center, fake_centers, mask)
    print(f"Hard Negative Loss: {loss_hn.item():.6f}")
    assert loss_hn.device.type == device.type, f"Hard Negative Loss 不在正确的设备上: {loss_hn.device} vs {device}"
    
    # 4. 测试Angle Loss
    print("\n测试 Angle Loss...")
    angle_loss = create_angle_loss(loss_config['angle_loss'])
    angle_loss = angle_loss.to(device)
    loss_angle = angle_loss(pred_angle, target_angle, target_heatmap_gap, mask)
    print(f"Angle Loss: {loss_angle.item():.6f}")
    assert loss_angle.device.type == device.type, f"Angle Loss 不在正确的设备上: {loss_angle.device} vs {device}"
    
    # 5. 测试Total Loss
    print("\n测试 Total Loss...")
    total_loss = create_total_loss(loss_config['total_loss'])
    total_loss = total_loss.to(device)
    
    predictions = {
        'heatmap_gap': pred_heatmap_gap,
        'heatmap_piece': pred_heatmap_piece,
        'offset': pred_offset,
        'angle': pred_angle
    }
    
    targets = {
        'heatmap_gap': target_heatmap_gap,
        'heatmap_piece': target_heatmap_piece,
        'offset': target_offset,
        'angle': target_angle,
        'mask': mask,
        'gap_center': gap_center,
        'fake_centers': fake_centers
    }
    
    loss_total, loss_dict = total_loss(predictions, targets)
    print(f"Total Loss: {loss_total.item():.6f}")
    print("子损失:")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"  {key}: {value.item():.6f}")
    
    assert loss_total.device.type == device.type, f"Total Loss 不在正确的设备上: {loss_total.device} vs {device}"
    
    # 反向传播测试
    print("\n测试反向传播...")
    loss_total.backward()
    print("反向传播成功!")
    
    print("\n所有CUDA兼容性测试通过! ✅")


if __name__ == "__main__":
    test_cuda_compatibility()