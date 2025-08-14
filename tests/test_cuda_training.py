# -*- coding: utf-8 -*-
"""
CUDA训练流程测试
验证完整的训练流程在CUDA上正常工作
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import yaml
from src.models.loss_calculation.total_loss import create_total_loss

def test_cuda_training():
    """测试完整的CUDA训练流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if device.type == 'cpu':
        print("警告: CUDA不可用，使用CPU测试")
    
    # 创建简单模型（模拟Stage7架构）
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 下采样到1/4分辨率
            self.conv1 = nn.Conv2d(4, 64, 3, stride=2, padding=1)  # 4通道输入 -> 1/2
            self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # -> 1/4
            
            # 特征处理
            self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
            
            # 输出头（1/4分辨率）
            self.head_heatmap = nn.Conv2d(128, 2, 1)  # 2个热力图
            self.head_offset = nn.Conv2d(128, 4, 1)   # 4个偏移
            self.head_angle = nn.Conv2d(128, 2, 1)    # 2个角度（sin, cos）
            
        def forward(self, x):
            # 下采样
            feat = torch.relu(self.conv1(x))  # 1/2分辨率
            feat = torch.relu(self.conv2(feat))  # 1/4分辨率
            feat = torch.relu(self.conv3(feat))
            
            # 生成1/4分辨率输出
            heatmap = torch.sigmoid(self.head_heatmap(feat))
            offset = torch.tanh(self.head_offset(feat)) * 0.5
            angle = torch.nn.functional.normalize(self.head_angle(feat), p=2, dim=1)
            
            return {
                'heatmap_gap': heatmap[:, 0:1],
                'heatmap_piece': heatmap[:, 1:2],
                'offset': offset,
                'angle': angle
            }
    
    # 创建模型并移到CUDA
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 从YAML文件加载配置
    config_path = Path(__file__).parent.parent / 'config' / 'loss.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        loss_config = yaml.safe_load(f)
    
    # 使用stage7配置（启用了角度损失）
    if 'stage_configs' in loss_config and 'stage7' in loss_config['stage_configs']:
        # 合并stage7配置到主配置
        stage7_config = loss_config['stage_configs']['stage7']
        loss_config['focal_loss'].update(stage7_config.get('focal_loss', {}))
        loss_config['offset_loss'].update(stage7_config.get('offset_loss', {}))
        loss_config['hard_negative_loss'].update(stage7_config.get('hard_negative_loss', {}))
        loss_config['angle_loss'].update(stage7_config.get('angle_loss', {}))
        loss_config['total_loss'].update(stage7_config.get('total_loss', {}))
    
    # 创建损失函数
    criterion = create_total_loss(loss_config)
    
    # 准备测试数据
    batch_size = 2
    height, width = 160, 320
    h_feat, w_feat = height // 4, width // 4  # 1/4分辨率
    
    # 输入数据（4通道：RGB + mask）
    input_img = torch.randn(batch_size, 4, height, width).to(device)
    
    # 目标数据
    def generate_gaussian_heatmap(center, h, w, sigma=1.5):
        """生成高斯热图"""
        heatmap = torch.zeros(h, w)
        cx, cy = int(center[0]), int(center[1])
        if 0 <= cx < w and 0 <= cy < h:
            radius = int(3 * sigma)
            for i in range(max(0, cy - radius), min(h, cy + radius + 1)):
                for j in range(max(0, cx - radius), min(w, cx + radius + 1)):
                    dist_sq = (i - cy) ** 2 + (j - cx) ** 2
                    heatmap[i, j] = np.exp(-dist_sq / (2 * sigma ** 2))
        return heatmap
    
    # 生成目标热图
    gap_centers = torch.tensor([[20, 15], [25, 18]])  # 1/4分辨率坐标
    piece_centers = torch.tensor([[30, 20], [35, 25]])
    
    target_heat_gap = torch.stack([
        generate_gaussian_heatmap(gap_centers[i], h_feat, w_feat) 
        for i in range(batch_size)
    ]).unsqueeze(1).to(device)
    
    target_heat_piece = torch.stack([
        generate_gaussian_heatmap(piece_centers[i], h_feat, w_feat) 
        for i in range(batch_size)
    ]).unsqueeze(1).to(device)
    
    # 生成目标偏移
    target_offset = torch.randn(batch_size, 4, h_feat, w_feat).to(device) * 0.3
    target_offset = torch.clamp(target_offset, -0.5, 0.5)
    
    # 生成目标角度
    angles_deg = torch.tensor([0.8, -1.2]).to(device)  # 度
    angles_rad = angles_deg * (torch.pi / 180.0)
    target_angle = torch.stack([
        torch.sin(angles_rad),
        torch.cos(angles_rad)
    ], dim=1).unsqueeze(-1).unsqueeze(-1)
    target_angle = target_angle.expand(batch_size, 2, h_feat, w_feat)
    
    # 准备目标字典
    targets = {
        'heatmap_gap': target_heat_gap,
        'heatmap_piece': target_heat_piece,
        'offset': target_offset,
        'angle': target_angle,
        'mask': torch.ones(batch_size, 1, h_feat, w_feat).to(device),
        'gap_center': gap_centers.float().to(device),
        'fake_centers': [torch.tensor([[18.0, 25.0], [40.0, 30.0]]).to(device)]
    }
    
    print("\n开始训练测试...")
    
    # 训练几个步骤
    for epoch in range(3):
        # 前向传播
        predictions = model(input_img)
        
        # 计算损失
        total_loss, loss_dict = criterion(predictions, targets)
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Total Loss: {total_loss.item():.4f}")
        for key, value in loss_dict.items():
            if key != 'total':
                print(f"  {key}: {value.item():.4f}")
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 检查梯度
        has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters() if p.requires_grad)
        
        if not has_grad:
            print("错误: 部分参数没有梯度!")
            return False
        
        # 更新参数
        optimizer.step()
    
    print("\n✅ CUDA训练流程测试通过!")
    
    # 测试推理
    print("\n测试推理模式...")
    model.eval()
    with torch.no_grad():
        predictions = model(input_img)
        
        # 验证输出形状
        assert predictions['heatmap_gap'].shape == (batch_size, 1, h_feat, w_feat)
        assert predictions['heatmap_piece'].shape == (batch_size, 1, h_feat, w_feat)
        assert predictions['offset'].shape == (batch_size, 4, h_feat, w_feat)
        assert predictions['angle'].shape == (batch_size, 2, h_feat, w_feat)
        
        # 验证设备
        for key, tensor in predictions.items():
            assert tensor.device.type == device.type, f"{key} 不在正确的设备上"
        
        print("  输出形状验证通过 ✓")
        print("  设备验证通过 ✓")
    
    print("\n✅ 所有CUDA训练测试通过!")
    return True

if __name__ == "__main__":
    success = test_cuda_training()
    if not success:
        sys.exit(1)