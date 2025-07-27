# -*- coding: utf-8 -*-
"""
ResNet18 Lite Backbone
轻量级ResNet18实现，用于滑块验证码检测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """残差块基本单元"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 主分支
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Skip-add
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18Lite(nn.Module):
    """
    轻量级ResNet18骨干网络
    输入: 3×160×320
    输出: 256×10×20 特征图
    """
    
    def __init__(self):
        super(ResNet18Lite, self).__init__()
        
        # Stem Conv: 3×160×320 -> 32×80×160
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stage-1: 32×80×160 -> 64×40×80 (ResBlock×2, channels 64, stride=2)
        self.stage1 = self._make_stage(32, 64, num_blocks=2, stride=2)
        
        # Stage-2: 64×40×80 -> 128×20×40 (ResBlock×2, channels 128, stride=2)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2)
        
        # Stage-3: 128×20×40 -> 256×10×20 (ResBlock×2, channels 256, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2)
        
        # 初始化权重
        self._init_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """构建一个stage"""
        layers = []
        
        # 第一个block可能需要降采样
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # 剩余的blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Kaiming Normal初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        输入: x shape [B, 3, 160, 320]
        输出: features shape [B, 256, 10, 20]
        """
        x = self.stem(x)      # [B, 32, 80, 160]
        x = self.stage1(x)    # [B, 64, 40, 80]
        x = self.stage2(x)    # [B, 128, 20, 40]
        x = self.stage3(x)    # [B, 256, 10, 20]
        
        return x


if __name__ == "__main__":
    # 测试backbone
    model = ResNet18Lite()
    
    # 输入张量
    x = torch.randn(2, 3, 160, 320)
    
    # 前向传播
    features = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {features.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小 (MB): {total_params * 4 / 1024 / 1024:.2f}")