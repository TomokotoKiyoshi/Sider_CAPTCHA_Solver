#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lite-HRNet-18 + LiteFPN 模型
专门用于滑块验证码识别的轻量级高分辨率网络

模型架构：
- Backbone: Lite-HRNet-18 (4个并行分支，保持多尺度特征)
- Neck: LiteFPN (轻量级特征金字塔网络)
- Head: DualHead (双头预测，热力图+偏移)

参数量：~3.5M
推理速度：CPU ~12-14ms, GPU <3ms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 导入各个阶段的模块
from .stem import create_stem
from .stage2 import create_stage2
from .stage3 import create_stage3
from .stage4 import create_stage4
from .stage5_lite_fpn import create_stage5_lite_fpn
from .stage6_dual_head import create_stage6_dual_head
from .utils import init_weights, count_parameters


class LiteHRNet18FPN(nn.Module):
    """
    Lite-HRNet-18 + LiteFPN 模型
    
    完整的6阶段架构：
    1. Stem: 特征提取，4×下采样 [256×512] → [64×128]
    2. Stage2: 双分支结构（1/4, 1/8分辨率）
    3. Stage3: 三分支结构（1/4, 1/8, 1/16分辨率）
    4. Stage4: 四分支结构（1/4, 1/8, 1/16, 1/32分辨率）
    5. LiteFPN: 轻量级特征金字塔，统一到128通道
    6. DualHead: 双头预测（热力图+偏移）
    
    输入: [B, 4, 256, 512] (RGB + padding mask)
    输出: 缺口和滑块的热力图及偏移量
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化Lite-HRNet-18+LiteFPN模型
        
        Args:
            config: 模型配置字典，如果为None则使用默认配置
        """
        super().__init__()
        
        # 模型名称
        self.model_name = "Lite-HRNet-18+LiteFPN"
        
        # 使用默认配置或用户配置
        self.config = self._get_default_config()
        if config is not None:
            self.config.update(config)
        
        # ========== 构建Lite-HRNet-18主干网络 ==========
        # Stage1: Stem特征提取
        self.stem = create_stem(self.config['stem'])
        
        # Stage2: 双分支 (Lite-HRNet开始)
        self.stage2 = create_stage2(self.config['stage2'])
        
        # Stage3: 三分支
        self.stage3 = create_stage3(self.config['stage3'])
        
        # Stage4: 四分支 (Lite-HRNet主干结束)
        self.stage4 = create_stage4(self.config['stage4'])
        
        # ========== 构建LiteFPN颈部网络 ==========
        # Stage5: LiteFPN (特征金字塔网络)
        self.lite_fpn = create_stage5_lite_fpn(self.config['lite_fpn'])
        
        # ========== 构建预测头 ==========
        # Stage6: DualHead (双头预测)
        self.dual_head = create_stage6_dual_head(self.config['dual_head'])
        
        # 初始化权重
        self.apply(init_weights)
        
        # 打印模型信息
        self._print_model_info()
    
    def _get_default_config(self) -> Dict:
        """
        获取Lite-HRNet-18+LiteFPN的默认配置
        
        Returns:
            默认配置字典
        """
        return {
            # ========== Lite-HRNet-18 Backbone配置 ==========
            # Stem配置 (Stage1)
            'stem': {
                'in_channels': 4,      # RGB(3) + padding mask(1)
                'out_channels': 32,    # Lite-HRNet-18标准通道数
                'expansion': 2         # LiteBlock扩张倍率
            },
            
            # Stage2配置（双分支）
            'stage2': {
                'in_channels': 32,
                'channels': [32, 64],      # [1/4分辨率, 1/8分辨率]
                'num_blocks': [2, 2],      # Lite-HRNet-18配置
                'expansion': 2
            },
            
            # Stage3配置（三分支）
            'stage3': {
                'channels': [32, 64, 128],    # [1/4, 1/8, 1/16分辨率]
                'num_blocks': [3, 3, 3],      # Lite-HRNet-18配置
                'expansion': 2
            },
            
            # Stage4配置（四分支）
            'stage4': {
                'channels': [32, 64, 128, 256],   # [1/4, 1/8, 1/16, 1/32分辨率]
                'num_blocks': [2, 2, 2, 2],       # Lite-HRNet-18配置
                'expansion': 2
            },
            
            # ========== LiteFPN Neck配置 ==========
            'lite_fpn': {
                'in_channels': [32, 64, 128, 256],  # 来自Lite-HRNet-18的4个分支
                'fpn_channels': 128,                # FPN统一通道数
                'fusion_type': 'weighted',          # 适合处理混淆缺口
                'return_pyramid': False             # 只返回主特征
            },
            
            # ========== DualHead配置 ==========
            'dual_head': {
                'in_channels': 128,          # 来自LiteFPN
                'mid_channels': 64,          # 中间层通道数
                'use_angle': True,           # 预测旋转角度（处理旋转缺口）
                'use_tanh_offset': True      # 限制偏移范围到[-0.5, 0.5]
            }
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 4, 256, 512] (RGB + padding mask)
            
        Returns:
            包含预测结果的字典：
            - 'heatmap_gap': [B, 1, 64, 128] 缺口中心热力图
            - 'heatmap_slider': [B, 1, 64, 128] 滑块中心热力图  
            - 'offset_gap': [B, 2, 64, 128] 缺口偏移 (du, dv)
            - 'offset_slider': [B, 2, 64, 128] 滑块偏移 (du, dv)
            - 'angle': [B, 2, 64, 128] 角度预测 (sin θ, cos θ)（可选）
        """
        # ========== Lite-HRNet-18 Backbone ==========
        # Stage1: Stem特征提取
        # [B, 4, 256, 512] → [B, 32, 64, 128]
        x = self.stem(x)
        
        # Stage2: 双分支
        # [B, 32, 64, 128] → [[B, 32, 64, 128], [B, 64, 32, 64]]
        features = self.stage2(x)
        
        # Stage3: 三分支
        # → [[B, 32, 64, 128], [B, 64, 32, 64], [B, 128, 16, 32]]
        features = self.stage3(features)
        
        # Stage4: 四分支
        # → [[B, 32, 64, 128], [B, 64, 32, 64], [B, 128, 16, 32], [B, 256, 8, 16]]
        features = self.stage4(features)
        
        # ========== LiteFPN Neck ==========
        # Stage5: 特征金字塔网络
        # → [B, 128, 64, 128]
        fpn_features = self.lite_fpn(features)
        
        # ========== DualHead ==========
        # Stage6: 双头预测
        # → {'heatmap_gap': ..., 'heatmap_piece': ..., 'offset': ..., 'angle': ...}
        predictions = self.dual_head(fpn_features)
        
        # 重新组织输出格式
        outputs = self._reorganize_outputs(predictions)
        
        return outputs
    
    def _reorganize_outputs(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        重新组织输出格式，使其更符合训练需求
        
        Args:
            predictions: DualHead的原始输出
            
        Returns:
            重新组织的输出字典
        """
        outputs = {}
        
        # 热力图
        outputs['heatmap_gap'] = predictions['heatmap_gap']        # [B, 1, 64, 128]
        outputs['heatmap_slider'] = predictions['heatmap_piece']   # [B, 1, 64, 128]
        
        # 分离偏移量
        offset = predictions['offset']  # [B, 4, 64, 128]
        outputs['offset_gap'] = offset[:, :2, :, :]      # [B, 2, 64, 128]
        outputs['offset_slider'] = offset[:, 2:4, :, :]  # [B, 2, 64, 128]
        
        # 角度（可选）
        if 'angle' in predictions:
            outputs['angle'] = predictions['angle']  # [B, 2, 64, 128]
        
        return outputs
    
    def decode_predictions(self, outputs: Dict[str, torch.Tensor], 
                          threshold: float = 0.1,
                          input_images: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        从模型输出解码坐标预测（用于推理）
        
        Args:
            outputs: 模型输出字典
            threshold: 热力图阈值
            input_images: 输入图像 [B, 4, H, W]，第4通道是padding mask
            
        Returns:
            解码后的坐标字典：
            - 'gap_coords': [B, 2] 缺口坐标 (x, y)
            - 'slider_coords': [B, 2] 滑块坐标 (x, y)
            - 'gap_score': [B] 缺口置信度
            - 'slider_score': [B] 滑块置信度
        """
        batch_size = outputs['heatmap_gap'].size(0)
        device = outputs['heatmap_gap'].device
        
        # 获取热力图
        gap_heatmap = outputs['heatmap_gap'].squeeze(1)      # [B, 64, 128]
        slider_heatmap = outputs['heatmap_slider'].squeeze(1) # [B, 64, 128]
        
        # 获取偏移量
        gap_offset = outputs['offset_gap']      # [B, 2, 64, 128]
        slider_offset = outputs['offset_slider'] # [B, 2, 64, 128]
        
        # 如果提供了输入图像，从第4通道提取padding mask并下采样到1/4分辨率
        mask_1_4 = None
        if input_images is not None:
            # 提取padding mask（第4通道）: padding=0, 有效=1（根据PREPROCESSING_OUTPUT.md）
            valid_mask = input_images[:, 3:4, :, :]  # [B, 1, 256, 512]
            # 下采样到1/4分辨率 (64, 128)
            valid_mask_1_4 = F.avg_pool2d(valid_mask, kernel_size=4, stride=4).squeeze(1)  # [B, 64, 128]
            # 转换为padding mask：padding=1, 有效=0（用于masked_fill）
            mask_1_4 = 1 - valid_mask_1_4
        
        # 解码缺口坐标
        gap_coords, gap_scores = self._decode_single_point(
            gap_heatmap, gap_offset, mask_1_4
        )
        
        # 解码滑块坐标
        slider_coords, slider_scores = self._decode_single_point(
            slider_heatmap, slider_offset, mask_1_4
        )
        
        return {
            'gap_coords': gap_coords,        # [B, 2]
            'slider_coords': slider_coords,   # [B, 2]
            'gap_score': gap_scores,         # [B]
            'slider_score': slider_scores    # [B]
        }
    
    def _decode_single_point(self, heatmap: torch.Tensor, offset: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解码单个点的坐标
        
        Args:
            heatmap: 热力图 [B, H, W]
            offset: 偏移量 [B, 2, H, W]
            mask: padding mask [B, H, W]，padding=1，有效=0
            
        Returns:
            coords: 坐标 [B, 2]
            scores: 置信度 [B]
        """
        batch_size = heatmap.size(0)
        height, width = heatmap.size(1), heatmap.size(2)
        device = heatmap.device
        
        # 如果有mask，将padding区域的热力图值设为-inf，确保不会被选为最大值
        if mask is not None:
            # padding区域（mask>0.5）设为-inf
            heatmap = heatmap.masked_fill(mask > 0.5, float('-inf'))
        
        # 应用阈值并找到峰值
        heatmap_flat = heatmap.view(batch_size, -1)
        scores, max_idx = torch.max(heatmap_flat, dim=1)  # [B], [B]
        
        # 转换为2D坐标
        y = max_idx // width  # [B]
        x = max_idx % width   # [B]
        
        # 获取对应位置的偏移量
        batch_idx = torch.arange(batch_size, device=device)
        offset_x = offset[batch_idx, 0, y, x]  # [B]
        offset_y = offset[batch_idx, 1, y, x]  # [B]
        
        # 计算最终坐标（上采样4倍）
        coords_x = (x.float() + 0.5 + offset_x) * 4.0
        coords_y = (y.float() + 0.5 + offset_y) * 4.0
        
        coords = torch.stack([coords_x, coords_y], dim=1)  # [B, 2]
        
        return coords, scores
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params, trainable_params = count_parameters(self)
        
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print("=" * 60)
        print(f"Architecture: Lite-HRNet-18 (Backbone) + LiteFPN (Neck) + DualHead")
        print(f"Input shape: [B, 4, 256, 512]")
        print(f"Output shape: [B, 2, 64, 128] (heatmaps) + [B, 4, 64, 128] (offsets)")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        print(f"Estimated FLOPs: ~0.45G")
        print("=" * 60)
    
    def save(self, path: str):
        """
        保存模型权重
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_name': self.model_name,
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to: {path}")
    
    def load(self, path: str, strict: bool = True):
        """
        加载模型权重
        
        Args:
            path: 权重文件路径
            strict: 是否严格匹配
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"Model loaded from: {path}")
        if 'model_name' in checkpoint:
            print(f"Model name: {checkpoint['model_name']}")


def create_lite_hrnet_18_fpn(config: Optional[Dict] = None, 
                             pretrained: Optional[str] = None) -> LiteHRNet18FPN:
    """
    创建Lite-HRNet-18+LiteFPN模型
    
    Args:
        config: 模型配置字典
        pretrained: 预训练权重路径
        
    Returns:
        LiteHRNet18FPN实例
    """
    model = LiteHRNet18FPN(config)
    
    if pretrained is not None:
        model.load(pretrained)
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("Testing Lite-HRNet-18+LiteFPN Model...")
    print("")
    
    # 创建模型
    model = create_lite_hrnet_18_fpn()
    
    # 测试前向传播
    x = torch.randn(2, 4, 256, 512)  # 批次大小2，4通道（RGB+mask），256×512分辨率
    outputs = model(x)
    
    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # 测试解码
    decoded = model.decode_predictions(outputs)
    print("\nDecoded predictions:")
    for key, value in decoded.items():
        print(f"  {key}: {value.shape}")
    
    print("\nModel test passed!")