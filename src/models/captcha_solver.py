# -*- coding: utf-8 -*-
"""
CAPTCHA Solver Model
完整的滑块验证码识别模型 (ResNet18 Lite + CenterNet)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.resnet18_lite import ResNet18Lite
from src.models.centernet_heads import UpConvNeck, CenterNetHeads


class CaptchaSolver(nn.Module):
    """
    滑块验证码识别模型
    架构: ResNet18 Lite backbone + CenterNet detection heads
    """
    
    def __init__(self):
        super(CaptchaSolver, self).__init__()
        
        # Backbone: ResNet18 Lite
        self.backbone = ResNet18Lite()
        
        # Neck: 上采样模块
        self.neck = UpConvNeck(in_channels=256, out_channels=64)
        
        # Detection Heads
        self.heads = CenterNetHeads(in_channels=64)
        
        # 下采样倍数
        self.down_ratio = 4
    
    def forward(self, x):
        """
        前向传播
        输入: x shape [B, 3, 160, 320]
        输出: 字典包含检测结果
            - gap_heatmap: [B, 1, 40, 80] 缺口中心概率图
            - gap_offset: [B, 2, 40, 80] 缺口亚像素偏移
            - piece_heatmap: [B, 1, 40, 80] 滑块中心概率图
            - piece_offset: [B, 2, 40, 80] 滑块亚像素偏移
        """
        # Backbone特征提取
        features = self.backbone(x)  # [B, 256, 10, 20]
        
        # Neck上采样
        features = self.neck(features)  # [B, 64, 40, 80]
        
        # Detection heads
        outputs = self.heads(features)
        
        return outputs
    
    def decode_outputs(self, outputs, K=1):
        """
        解码模型输出为坐标
        
        Args:
            outputs: 模型输出字典
            K: 每个热力图取前K个峰值 (默认1)
        
        Returns:
            字典包含:
                - gap_coords: [B, K, 2] 缺口中心坐标 (x, y)
                - piece_coords: [B, K, 2] 滑块中心坐标 (x, y)
        """
        batch_size = outputs['gap_heatmap'].shape[0]
        
        # 解码缺口位置
        gap_coords = self._decode_heatmap(
            outputs['gap_heatmap'], 
            outputs['gap_offset'], 
            K=K
        )
        
        # 解码滑块位置
        piece_coords = self._decode_heatmap(
            outputs['piece_heatmap'], 
            outputs['piece_offset'], 
            K=K
        )
        
        return {
            'gap_coords': gap_coords,
            'piece_coords': piece_coords
        }
    
    def _decode_heatmap(self, heatmap, offset, K=1):
        """
        从热力图和偏移解码坐标
        
        Args:
            heatmap: [B, 1, H, W] 概率热力图
            offset: [B, 2, H, W] 亚像素偏移
            K: 取前K个峰值
        
        Returns:
            coords: [B, K, 2] 检测坐标
        """
        batch, _, height, width = heatmap.shape
        
        # 5×5 MaxPool增强峰值
        heatmap = F.max_pool2d(heatmap, kernel_size=5, stride=1, padding=2)
        
        # Flatten并找到top-K
        heatmap_flat = heatmap.view(batch, -1)
        topk_values, topk_indices = torch.topk(heatmap_flat, K, dim=1)
        
        # 转换为2D坐标
        topk_y = topk_indices // width
        topk_x = topk_indices % width
        
        # 获取对应的偏移值
        coords = []
        for b in range(batch):
            batch_coords = []
            for k in range(K):
                x_idx = topk_x[b, k]
                y_idx = topk_y[b, k]
                
                # 获取偏移
                dx = offset[b, 0, y_idx, x_idx]
                dy = offset[b, 1, y_idx, x_idx]
                
                # 计算最终坐标
                x = (x_idx.float() + dx) * self.down_ratio
                y = (y_idx.float() + dy) * self.down_ratio
                
                batch_coords.append([x.item(), y.item()])
            
            coords.append(batch_coords)
        
        return torch.tensor(coords, device=heatmap.device)
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 分模块统计
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        neck_params = sum(p.numel() for p in self.neck.parameters())
        heads_params = sum(p.numel() for p in self.heads.parameters())
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone_params': backbone_params,
            'neck_params': neck_params,
            'heads_params': heads_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }


if __name__ == "__main__":
    # 创建模型
    model = CaptchaSolver()
    model.eval()
    
    # 测试输入
    x = torch.randn(2, 3, 160, 320)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x)
    
    print("Model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # 解码坐标
    coords = model.decode_outputs(outputs, K=1)
    print("\nDecoded coordinates:")
    print(f"  Gap coords: {coords['gap_coords']}")
    print(f"  Piece coords: {coords['piece_coords']}")
    
    # 模型信息
    info = model.get_model_info()
    print("\nModel info:")
    print(f"  Total params: {info['total_params']:,}")
    print(f"  Backbone params: {info['backbone_params']:,}")
    print(f"  Neck params: {info['neck_params']:,}")
    print(f"  Heads params: {info['heads_params']:,}")
    print(f"  Model size (MB): {info['model_size_mb']:.2f}")