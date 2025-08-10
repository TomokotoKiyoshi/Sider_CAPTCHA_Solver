"""
PMN-R3-FP Backbone Module
高分辨率多尺度骨干网络实现 (HRNet-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import sys
import os

# 添加配置路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.model_config import get_model_config

# 获取配置单例实例
model_config = get_model_config()


class BasicBlock(nn.Module):
    """基础残差块"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class HighResolutionModule(nn.Module):
    """高分辨率融合模块"""
    
    def __init__(self, num_branches, num_blocks, num_channels, multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        
        # 创建分支
        self.branches = self._make_branches(num_blocks, num_channels)
        
        # 创建融合层
        self.fuse_layers = self._make_fuse_layers(num_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def _make_branches(self, num_blocks, num_channels):
        """创建并行分支"""
        branches = []
        for i in range(self.num_branches):
            layers = []
            for j in range(num_blocks[i]):
                layers.append(BasicBlock(num_channels[i], num_channels[i]))
            branches.append(nn.Sequential(*layers))
        return nn.ModuleList(branches)
    
    def _make_fuse_layers(self, num_channels):
        """创建跨分辨率融合层"""
        if self.num_branches == 1:
            return None
        
        num_branches = self.num_branches
        fuse_layers = []
        
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # 上采样
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='bilinear', align_corners=False)
                    ))
                elif j == i:
                    # 同分辨率
                    fuse_layer.append(None)
                else:
                    # 下采样
                    conv_layers = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            conv_layers.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[i])
                            ))
                        else:
                            conv_layers.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv_layers))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        
        return nn.ModuleList(fuse_layers)
    
    def forward(self, x):
        """
        Args:
            x: List[Tensor] - 多分辨率输入
        Returns:
            List[Tensor] - 多分辨率输出
        """
        # 并行处理各分支
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        # 跨分辨率融合
        if self.fuse_layers is not None:
            x_fuse = []
            for i in range(len(self.fuse_layers)):
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                for j in range(1, self.num_branches):
                    if i == j:
                        y = y + x[j]
                    else:
                        y = y + self.fuse_layers[i][j](x[j])
                x_fuse.append(self.relu(y))
            x = x_fuse
        
        return x


class SharedStem(nn.Module):
    """共享Stem层"""
    
    def __init__(self, in_channels=None):
        super().__init__()
        
        # 从配置文件读取输入通道数
        if in_channels is None:
            # 输入通道数 = RGB(3) + 坐标编码(2) + Padding掩码(1) = 6
            if not hasattr(model_config, 'input_channels'):
                raise ValueError("input_channels not found in model_config.yaml")
            rgb_channels = model_config.input_channels
            coord_channels = model_config.config['input']['channels']['coord']
            padding_channels = model_config.config['input']['channels']['padding']
            in_channels = rgb_channels + coord_channels + padding_channels
        
        # 从配置读取第一个stage的通道数
        first_stage_channels = model_config.backbone_stages[0]['channels']
        
        # Stage 0: 7x7 Conv
        self.conv1 = nn.Conv2d(in_channels, first_stage_channels // 2, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(first_stage_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1: 两个3x3 Conv
        self.conv2 = nn.Conv2d(first_stage_channels // 2, first_stage_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(first_stage_channels)
        
        self.conv3 = nn.Conv2d(first_stage_channels, first_stage_channels, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(first_stage_channels)
    
    def forward(self, x):
        """
        Args:
            x: [B, 6, 256, 512]
        Returns:
            [B, 64, 64, 128]
        """
        x = self.conv1(x)  # [B, 32, 128, 256]
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)  # [B, 64, 128, 256]
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)  # [B, 64, 64, 128]
        x = self.bn3(x)
        x = self.relu(x)
        
        return x


class HRBackbone(nn.Module):
    """高分辨率骨干网络"""
    
    def __init__(self):
        super().__init__()
        
        # 从配置文件读取backbone stages配置
        if not hasattr(model_config, 'backbone_stages'):
            raise ValueError("backbone_stages not found in model_config.yaml")
        
        stages = model_config.backbone_stages
        
        # 配置参数 - 从配置文件读取
        self.stage2_cfg = {
            'num_branches': model_config.hrnet_stage2_num_branches,
            'num_blocks': model_config.hrnet_stage2_num_blocks,
            'num_channels': [stages[0]['channels'], stages[1]['channels']]
        }
        
        self.stage3_cfg = {
            'num_branches': model_config.hrnet_stage3_num_branches,
            'num_blocks': model_config.hrnet_stage3_num_blocks,
            'num_channels': [stages[0]['channels'], stages[1]['channels'], stages[2]['channels']]
        }
        
        self.stage4_cfg = {
            'num_branches': model_config.hrnet_stage4_num_branches,
            'num_blocks': model_config.hrnet_stage4_num_blocks,
            'num_channels': [stages[0]['channels'], stages[1]['channels'], 
                           stages[2]['channels'], stages[3]['channels']]
        }
        
        # 共享Stem - 让它自动从配置读取
        self.stem = SharedStem()
        
        # Stage 2: 引入第二分支
        self.stage2 = self._make_stage(self.stage2_cfg)
        self.transition1 = self._make_transition(
            [stages[0]['channels']], 
            [stages[0]['channels'], stages[1]['channels']]
        )
        
        # Stage 3: 引入第三分支
        self.stage3 = self._make_stage(self.stage3_cfg)
        self.transition2 = self._make_transition(
            [stages[0]['channels'], stages[1]['channels']], 
            [stages[0]['channels'], stages[1]['channels'], stages[2]['channels']]
        )
        
        # Stage 4: 引入第四分支
        self.stage4 = self._make_stage(self.stage4_cfg)
        # 添加transition3用于创建第4个分支
        self.transition3 = self._make_transition(
            [stages[0]['channels'], stages[1]['channels'], stages[2]['channels']], 
            [stages[0]['channels'], stages[1]['channels'], stages[2]['channels'], stages[3]['channels']]
        )
        
        # 初始化权重
        self._init_weights()
    
    def _make_stage(self, stage_cfg):
        """创建Stage"""
        return nn.Sequential(*[
            HighResolutionModule(
                stage_cfg['num_branches'],
                stage_cfg['num_blocks'],
                stage_cfg['num_channels']
            ) for _ in range(2)  # 每个stage包含2个HRModule
        ])
    
    def _make_transition(self, in_channels, out_channels):
        """创建过渡层"""
        transition_layers = []
        
        for i in range(len(out_channels)):
            if i < len(in_channels):
                if in_channels[i] != out_channels[i]:
                    # 通道变换
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(in_channels[i], out_channels[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(out_channels[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                # 新分支（下采样）
                conv_layers = []
                for j in range(i - len(in_channels) + 1):
                    in_ch = in_channels[-1] if j == 0 else out_channels[i]
                    conv_layers.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_channels[i], 3, 2, 1, bias=False),
                        nn.BatchNorm2d(out_channels[i]),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv_layers))
        
        return nn.ModuleList(transition_layers)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, 6, 256, 512]
        Returns:
            features: Dict with keys 's1', 's2', 's3', 's4'
                s1: [B, 64, 64, 128]
                s2: [B, 128, 32, 64]
                s3: [B, 256, 16, 32]
                s4: [B, 512, 8, 16]
        """
        # Stem
        x = self.stem(x)  # [B, 64, 64, 128]
        
        # Stage 2
        x_list = [x]
        for i, layer in enumerate(self.transition1):
            if layer is not None:
                if i < len(x_list):
                    x_list[i] = layer(x_list[i])
                else:
                    x_list.append(layer(x_list[-1]))
            else:
                x_list.append(x_list[i])
        
        x_list = self.stage2(x_list)
        
        # Stage 3
        x_list_new = []
        for i, layer in enumerate(self.transition2):
            if layer is not None:
                if i < len(x_list):
                    x_list_new.append(layer(x_list[i]))
                else:
                    x_list_new.append(layer(x_list[-1]))
            else:
                x_list_new.append(x_list[i])
        x_list = x_list_new
        
        x_list = self.stage3(x_list)
        
        # Stage 4 - 使用transition3添加第4个分支
        x_list_new = []
        for i, layer in enumerate(self.transition3):
            if layer is not None:
                if i < len(x_list):
                    x_list_new.append(layer(x_list[i]))
                else:
                    x_list_new.append(layer(x_list[-1]))
            else:
                x_list_new.append(x_list[i])
        x_list = x_list_new
        
        x_list = self.stage4(x_list)
        
        # 返回多尺度特征
        return {
            's1': x_list[0],  # [B, 64, 64, 128]
            's2': x_list[1],  # [B, 128, 32, 64]
            's3': x_list[2],  # [B, 256, 16, 32]
            's4': x_list[3]   # [B, 512, 8, 16]
        }


if __name__ == "__main__":
    # 测试代码
    print("Loading configuration from model_config.yaml...")
    print(f"Backbone type: {model_config.backbone_type}")
    print(f"Number of stages: {len(model_config.backbone_stages)}")
    for i, stage in enumerate(model_config.backbone_stages):
        print(f"  Stage {i+1}: channels={stage['channels']}, resolution={stage['resolution']}")
    
    model = HRBackbone()
    
    # 从配置获取输入形状
    batch_size = 2
    input_shape = model_config.get_input_shape(batch_size)
    # 但backbone期望6个通道（RGB + coord + padding）
    total_channels = (model_config.input_channels + 
                     model_config.config['input']['channels']['coord'] + 
                     model_config.config['input']['channels']['padding'])
    x = torch.randn(batch_size, total_channels, model_config.input_height, model_config.input_width)
    
    print(f"\nTesting with input shape: {x.shape}")
    features = model(x)
    
    print("HRBackbone Output Shapes:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")