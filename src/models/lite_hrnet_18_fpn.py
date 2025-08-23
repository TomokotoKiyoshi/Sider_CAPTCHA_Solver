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

# 导入配置加载器
from .config_loader import load_model_config

# 导入各个阶段的模块
from .stem import create_stem
from .stage2 import create_stage2
from .stage3 import create_stage3
from .stage4 import create_stage4
from .stage5_lite_fpn import create_stage5_lite_fpn
from .stage6_dual_head import create_stage6_dual_head
from .tamh import create_tamh
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
    
    输入: [B, C, 256, 512] (C=输入通道数，当前为2: 灰度图 + padding mask)
    输出: 缺口和滑块的热力图及偏移量
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化Lite-HRNet-18+LiteFPN模型
        
        Args:
            config: 模型配置字典，如果为None则从配置文件加载
        """
        super().__init__()
        
        # 模型名称
        self.model_name = "Lite-HRNet-18+LiteFPN"
        
        # 从配置文件加载或使用用户配置
        if config is None:
            self.config = self._load_config_from_file()
        else:
            self.config = config
        
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
        
        # ========== 构建TAMH模块（可选）==========
        # TAMH: Template-Aware Matching Head
        self.use_tamh = self.config.get('tamh', {}).get('enabled', False)
        if self.use_tamh:
            self.tamh = create_tamh(self.config.get('tamh', {}))
        
        # 初始化权重
        self.apply(init_weights)
        
        # 打印模型信息
        self._print_model_info()
    
    def _load_config_from_file(self) -> Dict:
        """
        从配置文件加载模型配置
        
        Returns:
            模型配置字典
        """
        # 加载完整配置
        full_config = load_model_config()
        
        # 提取backbone配置
        if 'backbone' not in full_config:
            raise ValueError("配置文件缺少backbone部分")
        
        backbone_config = full_config['backbone']
        if 'lite_hrnet' not in backbone_config:
            raise ValueError("配置文件缺少lite_hrnet部分")
        
        lite_hrnet = backbone_config['lite_hrnet']
        
        # 构建模型需要的配置格式
        model_config = {
            'stem': lite_hrnet['stem'],
            'stage2': lite_hrnet['stage2'],
            'stage3': lite_hrnet['stage3'],
            'stage4': lite_hrnet['stage4'],
            'lite_fpn': lite_hrnet['stage5_lite_fpn'],
            'dual_head': lite_hrnet['stage6_dual_head']
        }
        
        # 添加TAMH配置（如果存在）
        if 'tamh' in lite_hrnet:
            model_config['tamh'] = lite_hrnet['tamh']
        
        # 添加Stage2的输入通道（来自Stem的输出）
        model_config['stage2']['in_channels'] = model_config['stem']['out_channels']
        
        return model_config
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, 256, 512] (C=输入通道数)
            
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
        # [B, C, 256, 512] → [B, 32, 64, 128]
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
        
        # ========== TAMH增强（可选）==========
        if self.use_tamh:
            # 使用TAMH增强缺口热图预测
            tamh_outputs = self.tamh(
                Hf=fpn_features,
                H_piece=predictions['heatmap_piece'],
                H_gap=predictions['heatmap_gap']
            )
            # 用增强后的热图替换原始预测
            predictions['heatmap_gap'] = tamh_outputs['heatmap_gap_final']
            # 添加额外的输出（用于可视化和调试）
            predictions['heatmap_corr'] = tamh_outputs['heatmap_corr']
            predictions['template'] = tamh_outputs['template']
        
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
        
        # TAMH相关输出（可选）
        if 'heatmap_corr' in predictions:
            outputs['heatmap_corr'] = predictions['heatmap_corr']  # [B, 1, 64, 128]
        if 'template' in predictions:
            outputs['template'] = predictions['template']  # [B, 128, 16, 16]
        
        return outputs
    
    def decode_predictions(self, outputs: Dict[str, torch.Tensor], 
                          threshold: float = 0.1,
                          input_images: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        从模型输出解码坐标预测（用于推理）
        
        Args:
            outputs: 模型输出字典
            threshold: 热力图阈值
            input_images: 输入图像 [B, C, H, W]，最后一个通道是padding mask
            
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
        
        # 如果提供了输入图像，从最后一个通道提取padding mask并下采样到1/4分辨率
        mask_1_4 = None
        valid_mask_1_4 = None  # 保存有效mask供坐标裁剪使用
        if input_images is not None:
            # 提取valid mask（最后一个通道）: valid=1, padding=0
            valid_mask = input_images[:, -1:, :, :]  # [B, 1, 256, 512]
            
            # 使用最小池化下采样valid mask到1/4分辨率
            # min_pool确保任何包含padding(0)的网格都被标记为无效(0)
            # 由于PyTorch没有min_pool2d，使用-max_pool2d(-x)来实现
            valid_mask_1_4 = -F.max_pool2d(-valid_mask, kernel_size=4, stride=4).squeeze(1)  # [B, 64, 128]
            
            # 创建padding mask用于masked_fill (padding=1, valid=0)
            mask_1_4 = 1 - valid_mask_1_4
            
            # 直接在热力图上屏蔽padding区域！
            # 将padding区域的热力图值设为负无穷，确保不会被选为峰值
            gap_heatmap = gap_heatmap.masked_fill(mask_1_4 > 0.01, float('-inf'))
            slider_heatmap = slider_heatmap.masked_fill(mask_1_4 > 0.01, float('-inf'))
            
            # 同时将padding区域的偏移量归零
            valid_mask_expanded = valid_mask_1_4.unsqueeze(1)  # [B, 1, 64, 128]
            gap_offset = gap_offset * valid_mask_expanded
            slider_offset = slider_offset * valid_mask_expanded
        
        # 解码缺口坐标
        gap_coords, gap_scores = self._decode_single_point(
            gap_heatmap, gap_offset
        )
        
        # 解码滑块坐标
        slider_coords, slider_scores = self._decode_single_point(
            slider_heatmap, slider_offset
        )
        
        # Y轴对齐：在±4像素范围内扫描热力图最大值
        # 4像素 = 1个网格单位（因为下采样率是4）
        batch_size = gap_heatmap.size(0)
        height, width = gap_heatmap.size(1), gap_heatmap.size(2)
        
        for b in range(batch_size):
            if gap_scores[b] > slider_scores[b]:
                # 缺口置信度更高，用缺口Y作为参考，重新定位滑块
                # 将缺口Y转换为网格坐标
                ref_y_grid = torch.clamp(
                    torch.round((gap_coords[b, 1] / 4.0) - 0.5).long(),
                    min=0, max=height-1
                )
                
                # 在±1网格范围内扫描（对应±4像素）
                y_min = max(0, ref_y_grid - 1)
                y_max = min(height - 1, ref_y_grid + 1)
                
                # 提取Y范围内的热力图区域
                region = slider_heatmap[b, y_min:y_max+1, :]  # [3, 128] or less
                
                # 找到区域内的最大值位置
                region_flat = region.view(-1)
                max_val, max_idx = torch.max(region_flat, dim=0)
                
                # 转换为2D坐标
                local_y = max_idx // width
                local_x = max_idx % width
                
                # 转换为全局坐标
                global_y = y_min + local_y
                global_x = local_x
                
                # 获取偏移量
                offset_x = slider_offset[b, 0, global_y, global_x]
                offset_y = slider_offset[b, 1, global_y, global_x]
                
                # 计算新的滑块坐标
                new_slider_x = (global_x.float() + 0.5 + offset_x) * 4.0
                new_slider_y = (global_y.float() + 0.5 + offset_y) * 4.0
                
                # 更新滑块坐标
                slider_coords[b, 0] = torch.clamp(new_slider_x, min=0, max=512)
                slider_coords[b, 1] = torch.clamp(new_slider_y, min=0, max=256)
                
            else:
                # 滑块置信度更高，用滑块Y作为参考，重新定位缺口
                ref_y_grid = torch.clamp(
                    torch.round((slider_coords[b, 1] / 4.0) - 0.5).long(),
                    min=0, max=height-1
                )
                
                # 在±1网格范围内扫描
                y_min = max(0, ref_y_grid - 1)
                y_max = min(height - 1, ref_y_grid + 1)
                
                # 提取Y范围内的热力图区域
                region = gap_heatmap[b, y_min:y_max+1, :]
                
                # 找到区域内的最大值位置
                region_flat = region.view(-1)
                max_val, max_idx = torch.max(region_flat, dim=0)
                
                # 转换为2D坐标
                local_y = max_idx // width
                local_x = max_idx % width
                
                # 转换为全局坐标
                global_y = y_min + local_y
                global_x = local_x
                
                # 获取偏移量
                offset_x = gap_offset[b, 0, global_y, global_x]
                offset_y = gap_offset[b, 1, global_y, global_x]
                
                # 计算新的缺口坐标
                new_gap_x = (global_x.float() + 0.5 + offset_x) * 4.0
                new_gap_y = (global_y.float() + 0.5 + offset_y) * 4.0
                
                # 更新缺口坐标
                gap_coords[b, 0] = torch.clamp(new_gap_x, min=0, max=512)
                gap_coords[b, 1] = torch.clamp(new_gap_y, min=0, max=256)
        
        return {
            'gap_coords': gap_coords,        # [B, 2]
            'slider_coords': slider_coords,   # [B, 2]
            'gap_score': gap_scores,         # [B]
            'slider_score': slider_scores    # [B]
        }
    
    def _decode_single_point(self, heatmap: torch.Tensor, offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解码单个点的坐标
        注意：热力图和偏移量的mask处理已在decode_predictions中完成
        
        Args:
            heatmap: 热力图 [B, H, W]（已经过mask处理）
            offset: 偏移量 [B, 2, H, W]（已经过mask处理）
            
        Returns:
            coords: 坐标 [B, 2]
            scores: 置信度 [B]
        """
        batch_size = heatmap.size(0)
        height, width = heatmap.size(1), heatmap.size(2)
        device = heatmap.device
        
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
        
        # 改进3：坐标范围裁剪（确保在有效图像范围内）
        # 网络输入尺寸是512x256，有效坐标范围应该在[0, 512]和[0, 256]
        coords_x = torch.clamp(coords_x, min=0, max=512)
        coords_y = torch.clamp(coords_y, min=0, max=256)
        
        coords = torch.stack([coords_x, coords_y], dim=1)  # [B, 2]
        
        return coords, scores
    
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params, trainable_params = count_parameters(self)
        
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print("=" * 60)
        print(f"Architecture: Lite-HRNet-18 (Backbone) + LiteFPN (Neck) + DualHead")
        print(f"Input shape: [B, C, 256, 512] (C=input channels)")
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
    
    # 从模型配置获取输入通道数
    in_channels = model.stem.conv1.in_channels
    print(f"Model expects {in_channels} input channels")
    
    # 测试前向传播
    x = torch.randn(2, in_channels, 256, 512)  # 批次大小2，动态通道数，256×512分辨率
    outputs = model(x)
    
    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # 测试解码
    decoded = model.decode_predictions(outputs, input_images=x)
    print("\nDecoded predictions:")
    for key, value in decoded.items():
        print(f"  {key}: {value.shape}")
    
    print("\nModel test passed!")