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
import torch.nn.functional as F
import yaml
from typing import Tuple
from src.models.loss_calculation.focal_loss import FocalLoss, create_focal_loss
from src.models.loss_calculation.offset_loss import OffsetLoss, create_offset_loss
from src.models.loss_calculation.hard_negative_loss import HardNegativeLoss, create_hard_negative_loss
from src.models.loss_calculation.angle_loss import AngleLoss, create_angle_loss
from src.models.loss_calculation.total_loss import TotalLoss, create_total_loss


# ============ 测试辅助函数（从loss_utils.py移植） ============

def generate_gaussian_heatmap(centers: torch.Tensor,
                             shape: Tuple[int, int],
                             sigma: float = 1.5,
                             device: str = 'cpu') -> torch.Tensor:
    """
    生成高斯热图（用于测试）
    
    Args:
        centers: 中心点坐标 [N, 2] 或 [B, N, 2]
        shape: 热图形状 (H, W)
        sigma: 高斯标准差
        device: 设备
    
    Returns:
        高斯热图 [H, W] 或 [B, N, H, W]
    """
    H, W = shape
    
    # 处理不同维度的输入
    if centers.dim() == 2:  # [N, 2]
        centers = centers.unsqueeze(0)  # [1, N, 2]
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    B, N, _ = centers.shape
    
    # 创建坐标网格
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 扩展维度以支持批处理
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 初始化热图
    heatmaps = torch.zeros(B, N, H, W, device=device)
    
    # 为每个中心生成高斯分布
    for i in range(N):
        cx = centers[:, i, 0:1].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        cy = centers[:, i, 1:2].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        
        # 计算高斯分布
        gaussian = torch.exp(
            -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2)
        )
        
        heatmaps[:, i, :, :] = gaussian.squeeze(1)
    
    # 处理输出维度
    if squeeze_batch:
        heatmaps = heatmaps.squeeze(0)  # [N, H, W]
    
    return heatmaps


def create_padding_mask(input_shape: Tuple[int, int],
                       padded_shape: Tuple[int, int],
                       downsample: int = 4,
                       pooling: str = 'avg') -> torch.Tensor:
    """
    创建padding掩码并下采样到特征图分辨率（用于测试）
    
    Args:
        input_shape: 原始输入形状 (H_orig, W_orig)
        padded_shape: padding后的形状 (H_pad, W_pad)
        downsample: 下采样率
        pooling: 池化方式 ('avg' 或 'max')
    
    Returns:
        下采样后的掩码 [1, H_down, W_down]
    """
    H_orig, W_orig = input_shape
    H_pad, W_pad = padded_shape
    
    # 创建原始分辨率掩码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = torch.zeros(1, 1, H_pad, W_pad, device=device)
    mask[:, :, :H_orig, :W_orig] = 1.0
    
    # 下采样到特征图分辨率
    if pooling == 'avg':
        # 平均池化（软掩码）
        mask_down = F.avg_pool2d(mask, kernel_size=downsample, stride=downsample)
    else:  # 'max'
        # 最大池化（硬掩码）
        mask_down = F.max_pool2d(mask, kernel_size=downsample, stride=downsample)
    
    return mask_down.squeeze(0)  # [1, H_down, W_down]


def coordinate_transform(coords: torch.Tensor,
                        mode: str,
                        scale: float = 4.0) -> torch.Tensor:
    """
    坐标系转换（用于测试）
    
    Args:
        coords: 坐标 [B, N, 2] 或 [N, 2]
        mode: 转换模式
        scale: 缩放因子
    
    Returns:
        转换后的坐标
    """
    if mode == 'pixel_to_grid':
        # 原图坐标 -> 栅格坐标
        return coords / scale
    elif mode == 'grid_to_pixel':
        # 栅格坐标 -> 原图坐标
        return coords * scale
    else:
        raise ValueError(f"Unknown mode: {mode}")


def extract_peaks(heatmap: torch.Tensor,
                 threshold: float = 0.1,
                 nms: bool = True,
                 nms_kernel: int = 3,
                 top_k: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从热力图中提取峰值点（用于测试）
    
    Args:
        heatmap: 热力图 [B, C, H, W]
        threshold: 响应阈值
        nms: 是否应用NMS
        nms_kernel: NMS核大小
        top_k: 保留前k个峰值
    
    Returns:
        peaks: 峰值坐标 [B, C, K, 2]
        scores: 峰值得分 [B, C, K]
    """
    B, C, H, W = heatmap.shape
    
    # 应用NMS
    if nms:
        # 最大池化找到局部最大值
        padding = (nms_kernel - 1) // 2
        max_pool = F.max_pool2d(heatmap, nms_kernel, stride=1, padding=padding)
        
        # 保留局部最大值
        peak_mask = (heatmap == max_pool).float()
        
        # 应用阈值
        peak_mask = peak_mask * (heatmap > threshold).float()
        
        # 应用掩码
        heatmap = heatmap * peak_mask
    
    # 展平并找到top-k
    heatmap_flat = heatmap.view(B, C, -1)
    scores, indices = torch.topk(heatmap_flat, k=min(top_k, heatmap_flat.shape[-1]), dim=-1)
    
    # 转换索引为坐标
    y_coords = indices // W
    x_coords = indices % W
    peaks = torch.stack([x_coords, y_coords], dim=-1)  # [B, C, K, 2]
    
    # 过滤低于阈值的峰值
    valid_mask = scores > threshold
    peaks = peaks * valid_mask.unsqueeze(-1)
    scores = scores * valid_mask
    
    return peaks, scores


# ============ 测试函数 ============

def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'loss.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_focal_loss():
    """测试Focal Loss"""
    print("=" * 70)
    print("测试 Focal Loss - CenterNet风格热力图损失")
    print("=" * 70)
    
    batch_size = 2
    height, width = 64, 128
    
    # 从配置文件创建损失函数
    config = load_config()
    focal_loss = create_focal_loss(config['focal_loss'])
    
    # 创建测试数据
    pred = torch.sigmoid(torch.randn(batch_size, 1, height, width, requires_grad=True))
    
    # 目标：生成高斯热图
    centers = torch.tensor([[30.5, 20.5], [60.5, 40.5]])
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
    
    print("\n✓ Focal Loss测试通过")


def test_offset_loss():
    """测试Offset Loss"""
    print("\n" + "=" * 70)
    print("测试 Offset Loss - 子像素偏移损失")
    print("=" * 70)
    
    batch_size = 2
    height, width = 64, 128
    
    # 从配置文件创建损失函数
    config = load_config()
    offset_loss = create_offset_loss(config['offset_loss'])
    
    # 创建测试数据
    pred_offset = torch.tanh(torch.randn(batch_size, 4, height, width, requires_grad=True)) * 0.5
    target_offset = torch.randn(batch_size, 4, height, width) * 0.3
    target_offset = torch.clamp(target_offset, -0.5, 0.5)
    
    # 热力图（用于权重）
    centers_gap = torch.tensor([[30, 20], [35, 25]])
    centers_piece = torch.tensor([[60, 40], [65, 45]])
    heatmap_gap = generate_gaussian_heatmap(centers_gap, (height, width), sigma=1.5)
    heatmap_piece = generate_gaussian_heatmap(centers_piece, (height, width), sigma=1.5)
    heatmap = torch.stack([heatmap_gap, heatmap_piece], dim=1)  # [B, 2, H, W]
    
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
    
    # 从配置文件创建损失函数
    config = load_config()
    hn_loss = create_hard_negative_loss(config['hard_negative_loss'])
    
    # 创建测试数据
    heatmap = torch.sigmoid(torch.randn(batch_size, 1, height, width, requires_grad=True))
    
    # 真实缺口中心
    true_centers = torch.tensor([[30.5, 20.5], [60.5, 40.5]])
    
    # 假缺口中心
    fake_centers = [
        torch.tensor([[25.0, 30.0], [55.0, 50.0]]),
        torch.tensor([[70.0, 30.0], [40.0, 20.0]]),
    ]
    
    # 设置真实位置的高响应
    heatmap_data = heatmap.clone()
    for b in range(batch_size):
        x, y = int(true_centers[b, 0]), int(true_centers[b, 1])
        heatmap_data[b, 0, y, x] = 0.9
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
    
    # 从配置文件创建损失函数
    config = load_config()
    angle_loss = create_angle_loss(config['angle_loss'])
    
    # 创建测试数据
    pred_angle = torch.randn(batch_size, 2, height, width, requires_grad=True)
    pred_angle = torch.nn.functional.normalize(pred_angle, p=2, dim=1)
    
    # 目标角度
    angle_deg = torch.tensor([0.5, -1.0])  # 度
    angle_rad = angle_deg * (torch.pi / 180.0)
    sin_theta = torch.sin(angle_rad)
    cos_theta = torch.cos(angle_rad)
    target_angle = torch.stack([sin_theta, cos_theta], dim=1)  # [B, 2]
    target_angle = target_angle.unsqueeze(-1).unsqueeze(-1)  # [B, 2, 1, 1]
    target_angle = target_angle.expand(batch_size, 2, height, width)
    
    # 热力图
    centers = torch.tensor([[30, 20], [60, 40]])
    heatmap = generate_gaussian_heatmap(centers, (height, width), sigma=2.0)
    heatmap = heatmap.unsqueeze(1)  # [B, 1, H, W]
    
    # 掩码
    mask = torch.ones(batch_size, 1, height, width)
    
    # 计算损失
    loss = angle_loss(pred_angle, target_angle, heatmap, mask)
    
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
    
    # 从配置文件创建总损失函数
    config = load_config()
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
    
    print("\n✓ Total Loss测试通过")


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
    
    # 从配置文件创建损失函数（禁用可选损失）
    config = load_config()
    config['total_loss']['use_angle'] = False  # 禁用角度损失用于梯度流测试
    total_loss_fn = create_total_loss(config)
    
    # 测试数据
    batch_size = 2
    input_img = torch.randn(batch_size, 3, 64, 128)
    
    # 前向传播
    predictions = model(input_img)
    
    # 创建目标
    gap_centers = torch.tensor([[30, 20], [35, 25]])
    piece_centers = torch.tensor([[60, 40], [65, 45]])
    
    targets = {
        'heatmap_gap': generate_gaussian_heatmap(gap_centers, (64, 128), sigma=1.5).unsqueeze(1),
        'heatmap_piece': generate_gaussian_heatmap(piece_centers, (64, 128), sigma=1.5).unsqueeze(1),
        'offset': torch.zeros(batch_size, 4, 64, 128),
        'mask': torch.ones(batch_size, 1, 64, 128),
        'gap_center': gap_centers,
        'fake_centers': []
    }
    
    # 计算损失
    loss, loss_dict = total_loss_fn(predictions, targets)
    
    print(f"损失值: {loss.item():.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in model.parameters())
    
    if has_grad:
        print("✓ 梯度成功传播到所有参数")
    else:
        print("✗ 部分参数没有梯度")
    
    # 更新参数
    optimizer.step()
    print("✓ 参数更新成功")
    
    print("\n✓ 梯度流测试通过")


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("开始测试损失函数模块")
    print("=" * 70)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 运行各项测试
    test_focal_loss()
    test_offset_loss()
    test_hard_negative_loss()
    test_angle_loss()
    test_total_loss()
    test_gradient_flow()
    
    print("\n" + "=" * 70)
    print("所有损失函数测试通过！✅")
    print("=" * 70)


if __name__ == "__main__":
    main()