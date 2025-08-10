"""
PMN-R3-FP FPN+PAN Module
特征金字塔网络和路径聚合网络融合模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import sys
import os

# 添加配置路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.model_config import get_model_config

# 获取配置单例实例
model_config = get_model_config()


class FPN(nn.Module):
    """特征金字塔网络 (自顶向下)"""
    
    def __init__(self, in_channels_list=None, out_channels=None):
        """
        Args:
            in_channels_list: 各层输入通道数 (None则从配置读取)
            out_channels: 统一输出通道数 (None则从配置读取)
        """
        super().__init__()
        
        # 从配置文件读取参数
        if in_channels_list is None:
            if not hasattr(model_config, 'fpn_in_channels'):
                raise ValueError("fpn_in_channels not found in model_config.yaml")
            in_channels_list = model_config.fpn_in_channels
        
        if out_channels is None:
            if not hasattr(model_config, 'fpn_out_channels'):
                raise ValueError("fpn_out_channels not found in model_config.yaml")
            out_channels = model_config.fpn_out_channels
        
        # 横向连接 (1x1卷积调整通道数)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            )
        
        # 输出卷积 (3x3卷积平滑)
        self.smooth_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.smooth_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, features):
        """
        Args:
            features: Dict with keys 's1', 's2', 's3', 's4'
        Returns:
            fpn_features: List[Tensor] - FPN特征 [P1, P2, P3, P4]
        """
        # 获取输入特征
        s1, s2, s3, s4 = features['s1'], features['s2'], features['s3'], features['s4']
        inputs = [s1, s2, s3, s4]
        
        # 横向连接
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))
        
        # 自顶向下路径 (从P4开始)
        fpn_features = []
        
        # P4 (最高层，无需上采样)
        fpn_features.append(laterals[-1])
        
        # P3, P2, P1 (逐层上采样并融合)
        for i in range(len(laterals) - 2, -1, -1):
            # 上采样高层特征
            size = laterals[i].shape[-2:]
            upsampled = F.interpolate(
                fpn_features[0], 
                size=size, 
                mode='bilinear', 
                align_corners=False
            )
            
            # 元素相加
            fused = laterals[i] + upsampled
            
            # 插入到列表开头
            fpn_features.insert(0, fused)
        
        # 3x3卷积平滑
        smoothed_features = []
        for i, smooth_conv in enumerate(self.smooth_convs):
            smoothed_features.append(smooth_conv(fpn_features[i]))
        
        return smoothed_features


class PAN(nn.Module):
    """路径聚合网络 (自底向上)"""
    
    def __init__(self, in_channels=None, out_channels=None):
        """
        Args:
            in_channels: 输入通道数 (None则从配置读取)
            out_channels: 输出通道数 (None则从配置读取)
        """
        super().__init__()
        
        # 从配置文件读取参数
        if in_channels is None:
            if not hasattr(model_config, 'fpn_out_channels'):
                raise ValueError("fpn_out_channels not found in model_config.yaml")
            in_channels = model_config.fpn_out_channels
        
        if out_channels is None:
            if not hasattr(model_config, 'fpn_out_channels'):
                raise ValueError("fpn_out_channels not found in model_config.yaml")
            out_channels = model_config.fpn_out_channels
        
        # 下采样卷积
        self.downsample_convs = nn.ModuleList()
        for i in range(3):  # P1->P2, P2->P3, P3->P4
            self.downsample_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 融合卷积
        self.fusion_convs = nn.ModuleList()
        for i in range(4):  # N1, N2, N3, N4
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, fpn_features):
        """
        Args:
            fpn_features: List[Tensor] - FPN特征 [P1, P2, P3, P4]
        Returns:
            pan_features: List[Tensor] - PAN特征 [N1, N2, N3, N4]
        """
        p1, p2, p3, p4 = fpn_features
        
        # 自底向上路径
        # N1 = P1 (最低层，无需处理)
        n1 = p1
        
        # N2 = P2 + downsample(N1)
        n1_down = self.downsample_convs[0](n1)
        n2 = p2 + n1_down
        
        # N3 = P3 + downsample(N2)
        n2_down = self.downsample_convs[1](n2)
        n3 = p3 + n2_down
        
        # N4 = P4 + downsample(N3)
        n3_down = self.downsample_convs[2](n3)
        n4 = p4 + n3_down
        
        # 融合卷积
        pan_features = [n1, n2, n3, n4]
        for i in range(4):
            pan_features[i] = self.fusion_convs[i](pan_features[i])
        
        return pan_features


class FPN_PAN(nn.Module):
    """FPN+PAN双向特征金字塔"""
    
    def __init__(self, in_channels_list=None, out_channels=None):
        """
        Args:
            in_channels_list: 骨干网络输出通道数列表 (None则从配置读取)
            out_channels: 统一输出通道数 (None则从配置读取)
        """
        super().__init__()
        
        # 从配置文件读取参数
        if in_channels_list is None:
            if not hasattr(model_config, 'fpn_in_channels'):
                raise ValueError("fpn_in_channels not found in model_config.yaml")
            in_channels_list = model_config.fpn_in_channels
        
        if out_channels is None:
            if not hasattr(model_config, 'fpn_out_channels'):
                raise ValueError("fpn_out_channels not found in model_config.yaml")
            out_channels = model_config.fpn_out_channels
        
        # FPN (自顶向下)
        self.fpn = FPN(in_channels_list, out_channels)
        
        # PAN (自底向上)
        self.pan = PAN(out_channels, out_channels)
        
        # Region头部预测层
        self.region_heads = nn.ModuleDict()
        
        # Objectness预测 (是否包含缺口)
        self.region_heads['objectness'] = nn.ModuleList([
            nn.Conv2d(out_channels, 1, 1, 1, 0) for _ in range(4)
        ])
        
        # Centerness预测 (中心度质量)
        self.region_heads['centerness'] = nn.ModuleList([
            nn.Conv2d(out_channels, 1, 1, 1, 0) for _ in range(4)
        ])
        
        # Location预测 (中心点偏移)
        self.region_heads['location'] = nn.ModuleList([
            nn.Conv2d(out_channels, 2, 1, 1, 0) for _ in range(4)
        ])
        
        # Scale预测 (尺度信息)
        self.region_heads['scale'] = nn.ModuleList([
            nn.Conv2d(out_channels, 2, 1, 1, 0) for _ in range(4)
        ])
    
    def forward(self, features):
        """
        Args:
            features: Dict - 骨干网络输出特征
        Returns:
            Dict containing:
                - features: List[Tensor] - 多尺度特征
                - predictions: Dict - 各头部预测结果
        """
        # FPN处理
        fpn_features = self.fpn(features)
        
        # PAN处理
        pan_features = self.pan(fpn_features)
        
        # 生成预测
        predictions = {
            'objectness': [],
            'centerness': [],
            'location': [],
            'scale': []
        }
        
        for i, feat in enumerate(pan_features):
            # Objectness (是否包含缺口)
            obj = self.region_heads['objectness'][i](feat)
            predictions['objectness'].append(torch.sigmoid(obj))
            
            # Centerness (中心度质量)
            ctr = self.region_heads['centerness'][i](feat)
            predictions['centerness'].append(torch.sigmoid(ctr))
            
            # Location (中心点偏移)
            loc = self.region_heads['location'][i](feat)
            predictions['location'].append(loc)
            
            # Scale (尺度)
            scale = self.region_heads['scale'][i](feat)
            predictions['scale'].append(torch.exp(scale))
        
        return {
            'features': pan_features,
            'predictions': predictions
        }


class ProposalGenerator(nn.Module):
    """候选区域生成器"""
    
    def __init__(self, strides=None, top_k=None):
        """
        Args:
            strides: 各层特征图步长 (None则从配置读取)
            top_k: 保留的候选框数量 (None则从配置读取)
        """
        super().__init__()
        
        # 从配置文件读取参数
        if strides is None:
            if not hasattr(model_config, 'proposal_strides'):
                raise ValueError("proposal_strides not found in model_config.yaml")
            strides = model_config.proposal_strides
        
        if top_k is None:
            if not hasattr(model_config, 'proposal_top_k'):
                raise ValueError("proposal_top_k not found in model_config.yaml")
            top_k = model_config.proposal_top_k
        
        self.strides = strides
        self.top_k = top_k
    
    def forward(self, predictions):
        """
        Args:
            predictions: Dict - FPN_PAN的预测结果
        Returns:
            proposals: Tensor [B, top_k, 5] - (x, y, w, h, score)
        """
        batch_size = predictions['objectness'][0].shape[0]
        device = predictions['objectness'][0].device
        
        all_proposals = []
        
        for level_idx in range(len(self.strides)):
            stride = self.strides[level_idx]
            
            # 获取当前层预测
            objectness = predictions['objectness'][level_idx]  # [B, 1, H, W]
            centerness = predictions['centerness'][level_idx]  # [B, 1, H, W]
            location = predictions['location'][level_idx]      # [B, 2, H, W]
            scale = predictions['scale'][level_idx]            # [B, 2, H, W]
            
            B, _, H, W = objectness.shape
            
            # 生成网格坐标
            y_coords, x_coords = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            
            # 转换为原图坐标
            x_coords = (x_coords.float() + 0.5) * stride
            y_coords = (y_coords.float() + 0.5) * stride
            
            # 扩展维度
            x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
            y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
            
            # 应用偏移
            cx = x_coords + location[:, 0, :, :] * stride  # [B, H, W]
            cy = y_coords + location[:, 1, :, :] * stride  # [B, H, W]
            
            # 获取尺度
            w = scale[:, 0, :, :] * stride  # [B, H, W]
            h = scale[:, 1, :, :] * stride  # [B, H, W]
            
            # 计算得分
            scores = (objectness[:, 0, :, :] * centerness[:, 0, :, :]).sqrt()  # [B, H, W]
            
            # 重塑为列表
            cx = cx.reshape(B, -1)
            cy = cy.reshape(B, -1)
            w = w.reshape(B, -1)
            h = h.reshape(B, -1)
            scores = scores.reshape(B, -1)
            
            # 组合proposals
            level_proposals = torch.stack([cx, cy, w, h, scores], dim=-1)  # [B, H*W, 5]
            all_proposals.append(level_proposals)
        
        # 合并所有层的proposals
        all_proposals = torch.cat(all_proposals, dim=1)  # [B, total, 5]
        
        # 按得分排序并选择top_k
        scores = all_proposals[..., 4]  # [B, total]
        _, indices = torch.topk(scores, min(self.top_k, scores.shape[1]), dim=1)
        
        # 选择top_k proposals
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        proposals = all_proposals[batch_indices, indices]  # [B, top_k, 5]
        
        return proposals


if __name__ == "__main__":
    # 测试代码 - 使用配置文件的参数
    print("Loading configuration from model_config.yaml...")
    print(f"FPN in_channels: {model_config.fpn_in_channels}")
    print(f"FPN out_channels: {model_config.fpn_out_channels}")
    print(f"Proposal strides: {model_config.proposal_strides}")
    print(f"Proposal top_k: {model_config.proposal_top_k}")
    
    fpn_pan = FPN_PAN()  # 现在会自动使用配置文件的值
    
    # 模拟骨干网络输出
    features = {
        's1': torch.randn(2, 64, 64, 128),
        's2': torch.randn(2, 128, 32, 64),
        's3': torch.randn(2, 256, 16, 32),
        's4': torch.randn(2, 512, 8, 16)
    }
    
    output = fpn_pan(features)
    
    print("FPN+PAN Output:")
    print(f"  Features: {len(output['features'])} levels")
    for i, feat in enumerate(output['features']):
        print(f"    Level {i+1}: {feat.shape}")
    
    print(f"  Predictions:")
    for key, preds in output['predictions'].items():
        print(f"    {key}: {len(preds)} levels")
    
    # 测试Proposal生成器 - 使用配置文件的参数
    generator = ProposalGenerator()  # 现在会自动使用配置文件的值
    proposals = generator(output['predictions'])
    print(f"\nProposals shape: {proposals.shape}")
    print(f"Using top_k={generator.top_k}, strides={generator.strides}")