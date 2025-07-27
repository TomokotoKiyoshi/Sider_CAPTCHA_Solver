# -*- coding: utf-8 -*-
"""
CenterNet Detection Heads
用于检测缺口和滑块中心位置
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetHeads(nn.Module):
    """
    CenterNet检测头
    包含热力图和偏移预测
    """
    
    def __init__(self, in_channels=64):
        super(CenterNetHeads, self).__init__()
        
        # Gap Head (缺口检测)
        self.gap_heatmap = self._make_heatmap_head(in_channels)
        self.gap_offset = self._make_offset_head(in_channels)
        
        # Piece Head (滑块检测)
        self.piece_heatmap = self._make_heatmap_head(in_channels)
        self.piece_offset = self._make_offset_head(in_channels)
        
        # 初始化
        self._init_weights()
    
    def _make_heatmap_head(self, in_channels):
        """创建热力图检测头 - Conv3×3 → ReLU → Conv1×1 → Sigmoid"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()  # 根据README：激活 = Sigmoid
        )
    
    def _make_offset_head(self, in_channels):
        """创建偏移回归头 - Conv3×3 → ReLU → Conv1×1 (Linear)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1, stride=1, padding=0, bias=True)
            # 无激活函数，保持线性输出
        )
    
    def _init_weights(self):
        """初始化权重"""
        # 对所有模块进行Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 特殊处理：热力图的Conv1x1层（倒数第二层，因为最后是Sigmoid）
        # gap_heatmap的Conv1x1层
        nn.init.constant_(self.gap_heatmap[-2].weight, 0)
        nn.init.constant_(self.gap_heatmap[-2].bias, -2.19)  # -log((1-0.1)/0.1)
        
        # piece_heatmap的Conv1x1层
        nn.init.constant_(self.piece_heatmap[-2].weight, 0)
        nn.init.constant_(self.piece_heatmap[-2].bias, -2.19)
        
        # 偏移回归的最后一层使用较小的标准差
        nn.init.normal_(self.gap_offset[-1].weight, std=0.001)
        nn.init.constant_(self.gap_offset[-1].bias, 0)
        
        nn.init.normal_(self.piece_offset[-1].weight, std=0.001)
        nn.init.constant_(self.piece_offset[-1].bias, 0)
    
    def forward(self, x):
        """
        前向传播
        输入: x shape [B, 64, 40, 80]
        输出: 字典包含4个张量
            - gap_heatmap: [B, 1, 40, 80] (已经过Sigmoid激活)
            - gap_offset: [B, 2, 40, 80] (线性输出)
            - piece_heatmap: [B, 1, 40, 80] (已经过Sigmoid激活)
            - piece_offset: [B, 2, 40, 80] (线性输出)
        """
        # Gap检测
        gap_hm = self.gap_heatmap(x)  # 内部已包含Sigmoid
        gap_off = self.gap_offset(x)   # 线性输出
        
        # Piece检测
        piece_hm = self.piece_heatmap(x)  # 内部已包含Sigmoid
        piece_off = self.piece_offset(x)   # 线性输出
        
        return {
            'gap_heatmap': gap_hm,
            'gap_offset': gap_off,
            'piece_heatmap': piece_hm,
            'piece_offset': piece_off
        }


class UpConvNeck(nn.Module):
    """
    上采样Neck模块
    将backbone特征上采样到检测分辨率
    """
    
    def __init__(self, in_channels=256, out_channels=64):
        super(UpConvNeck, self).__init__()
        
        # Neck 1×1 Conv: 256×10×20 -> 128×10×20
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Up-Conv-1: 128×10×20 -> 64×20×40
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=3, stride=2, 
                              padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Up-Conv-2: 64×20×40 -> 64×40×80
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, 
                              padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming Normal初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        输入: x shape [B, 256, 10, 20]
        输出: features shape [B, 64, 40, 80]
        """
        x = self.neck(x)      # [B, 128, 10, 20]
        x = self.upconv1(x)   # [B, 64, 20, 40]
        x = self.upconv2(x)   # [B, 64, 40, 80]
        
        return x


if __name__ == "__main__":
    # 测试Neck和Heads
    neck = UpConvNeck(in_channels=256, out_channels=64)
    heads = CenterNetHeads(in_channels=64)
    
    # 模拟backbone输出
    backbone_features = torch.randn(2, 256, 10, 20)
    
    # Neck上采样
    neck_features = neck(backbone_features)
    print(f"Neck input shape: {backbone_features.shape}")
    print(f"Neck output shape: {neck_features.shape}")
    
    # Heads检测
    outputs = heads(neck_features)
    print("\nHeads output:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # 计算参数量
    neck_params = sum(p.numel() for p in neck.parameters())
    heads_params = sum(p.numel() for p in heads.parameters())
    
    print(f"\nNeck params: {neck_params:,}")
    print(f"Heads params: {heads_params:,}")
    print(f"Total params: {neck_params + heads_params:,}")