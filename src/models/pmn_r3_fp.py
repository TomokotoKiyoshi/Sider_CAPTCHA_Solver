"""
PMN-R3-FP Complete Model
PuzzleMatchNet - Robust & Refined with FPN+PAN
完整的滑块验证码识别模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import sys
import os

# 添加配置路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.model_config import get_model_config

# 获取配置单例实例
model_config = get_model_config()

# 导入各个模块
from .backbone import HRBackbone, SharedStem
from .fpn_pan import FPN_PAN, ProposalGenerator
from .shape_sdf import ShapeBranch, ROIAlignExtractor, SDFDecoder, CoordConv2d
from .se2_transformer import SE2Transformer, GeometricRefinement


class InputPreprocessor(nn.Module):
    """输入预处理器"""
    
    def __init__(self):
        super().__init__()
        self.coord_conv = CoordConv2d()
    
    def forward(self, rgb_input):
        """
        Args:
            rgb_input: [B, C, H, W] - RGB输入，尺寸由配置文件决定
        Returns:
            X0: [B, 6, H, W] - 融合输入
        """
        B, C, H, W = rgb_input.shape
        device = rgb_input.device
        
        # 验证输入尺寸
        expected_h, expected_w = model_config.input_height, model_config.input_width
        if H != expected_h or W != expected_w:
            raise ValueError(f"Expected input size ({expected_h}, {expected_w}), got ({H}, {W})")
        
        # RGB通道
        X_rgb = rgb_input  # [B, C, H, W]
        
        # 坐标编码
        X_coord = self.coord_conv(X_rgb)  # [B, 2, H, W]
        
        # Padding掩码 (检测黑色边缘)
        gray = 0.299 * X_rgb[:, 0] + 0.587 * X_rgb[:, 1] + 0.114 * X_rgb[:, 2]
        X_pad = (gray < 0.1).float().unsqueeze(1)  # [B, 1, H, W]
        
        # 通道拼接
        X0 = torch.cat([X_rgb, X_coord, X_pad], dim=1)  # [B, 6, H, W]
        
        return X0


class PMN_R3_FP(nn.Module):
    """PMN-R3-FP完整模型"""
    
    def __init__(self, num_classes=2, top_k_proposals=None):
        """
        Args:
            num_classes: 分类数 (背景 + 缺口)
            top_k_proposals: 保留的候选框数量 (None则从配置读取)
        """
        super().__init__()
        
        # 从配置读取参数
        if top_k_proposals is None:
            top_k_proposals = model_config.proposal_top_k
        
        # 输入预处理
        self.input_preprocessor = InputPreprocessor()
        
        # 骨干网络
        self.backbone = HRBackbone()
        
        # Region支路 (FPN+PAN)
        self.fpn_pan = FPN_PAN(
            in_channels_list=model_config.fpn_in_channels,
            out_channels=model_config.fpn_out_channels
        )
        
        # Shape支路
        self.shape_branch = ShapeBranch()
        
        # Proposal生成器
        self.proposal_generator = ProposalGenerator(
            strides=model_config.proposal_strides,
            top_k=top_k_proposals
        )
        
        # ROI特征提取
        self.roi_align_region = ROIAlignExtractor(
            output_size=model_config.roi_region_size,
            spatial_scale=model_config.roi_region_scale
        )
        
        self.roi_align_shape = ROIAlignExtractor(
            output_size=model_config.roi_shape_size,
            spatial_scale=model_config.roi_shape_scale
        )
        
        # SE(2)变换器匹配
        self.se2_transformer = SE2Transformer(
            d_model=model_config.se2_d_model,
            n_heads=model_config.se2_n_heads,
            n_layers=model_config.se2_n_layers,
            dropout=model_config.se2_dropout
        )
        
        # 几何精修器
        self.geometric_refiner = GeometricRefinement(d_model=model_config.se2_d_model)
        
        # 排序判别器
        hidden_dims = model_config.ranking_hidden_dims
        dropout = model_config.ranking_dropout
        self.ranking_discriminator = nn.Sequential(
            nn.Linear(model_config.se2_d_model + 3, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[1], 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images, return_features=False):
        """
        Args:
            images: [B, C, H, W] - 输入图像，尺寸由配置文件决定
            return_features: 是否返回中间特征
        Returns:
            Dict containing:
                - gap_centers: [B, N, 2] - 缺口中心坐标
                - piece_centers: [B, N, 2] - 滑块中心坐标
                - match_scores: [B, N, N] - 匹配得分矩阵
                - geometry: [B, N, N, 3] - 几何变换参数
                - features: (可选) 中间特征
        """
        B = images.shape[0]
        device = images.device
        
        # 输入预处理
        X0 = self.input_preprocessor(images)  # [B, 6, H, W]
        
        # 骨干网络提取特征
        backbone_features = self.backbone(X0)
        # 特征图尺寸根据输入动态计算
        # s1: [B, 64, H/4, W/4]
        # s2: [B, 128, H/8, W/8]
        # s3: [B, 256, H/16, W/16]
        # s4: [B, 512, H/32, W/32]
        
        # Region支路 (FPN+PAN)
        region_output = self.fpn_pan(backbone_features)
        region_features = region_output['features']  # 多尺度特征
        region_predictions = region_output['predictions']  # 预测头输出
        
        # Shape支路
        shape_output = self.shape_branch(images, backbone_features['s1'])
        shape_mask = shape_output['shape_mask']  # [B, 1, 256, 512]
        sdf_map = shape_output['sdf_map']        # [B, 1, 256, 512]
        shape_features = shape_output['features']  # [B, 32, 256, 512]
        
        # 生成候选区域
        proposals = self.proposal_generator(region_predictions)  # [B, top_k, 5]
        
        # 提取ROI特征
        # 从Region特征提取
        region_roi_features = self.roi_align_region(
            region_features[0],  # 使用最高分辨率特征
            proposals
        )  # [B*top_k, 256, 64, 64]
        
        # 从Shape特征提取
        shape_roi_features = self.roi_align_shape(
            shape_features,
            proposals
        )  # [B*top_k, 32, 64, 64]
        
        # 重塑为批次格式
        region_roi_features = region_roi_features.view(B, -1, 256, 64, 64)
        shape_roi_features = shape_roi_features.view(B, -1, 32, 64, 64)
        
        # 分离滑块和缺口候选
        # 假设前半部分是缺口候选，后半部分是滑块候选
        n_proposals = proposals.shape[1]
        n_gap = n_proposals // 2
        n_piece = n_proposals - n_gap
        
        gap_features = region_roi_features[:, :n_gap]
        piece_features = region_roi_features[:, n_gap:]
        
        # SE(2)变换器匹配
        matching_output = self.se2_transformer(piece_features, gap_features)
        match_scores = matching_output['match_scores']  # [B, n_piece, n_gap]
        geometry = matching_output['geometry']          # [B, n_piece, n_gap, 3]
        
        # 提取最佳匹配
        best_matches = []
        best_geometries = []
        
        for b in range(B):
            # 找到最高匹配得分
            scores = match_scores[b]  # [n_piece, n_gap]
            max_score, max_idx = scores.view(-1).max(0)
            piece_idx = max_idx // n_gap
            gap_idx = max_idx % n_gap
            
            # 获取对应的中心坐标
            gap_center = proposals[b, gap_idx, :2]
            piece_center = proposals[b, n_gap + piece_idx, :2]
            
            # 获取几何参数
            geom = geometry[b, piece_idx, gap_idx]
            
            best_matches.append({
                'gap_center': gap_center,
                'piece_center': piece_center,
                'score': max_score,
                'geometry': geom
            })
        
        # 组织输出
        output = {
            'gap_centers': torch.stack([m['gap_center'] for m in best_matches]),
            'piece_centers': torch.stack([m['piece_center'] for m in best_matches]),
            'scores': torch.stack([m['score'] for m in best_matches]),
            'geometry': torch.stack([m['geometry'] for m in best_matches]),
            'match_matrix': match_scores
        }
        
        if return_features:
            output['features'] = {
                'backbone': backbone_features,
                'region': region_features,
                'shape': shape_features,
                'shape_mask': shape_mask,
                'sdf_map': sdf_map,
                'proposals': proposals
            }
        
        return output
    
    def predict(self, images):
        """
        推理接口
        Args:
            images: [B, C, H, W] - 输入图像，尺寸由配置文件决定
        Returns:
            Dict with gap and piece positions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(images)
            
            # 转换为像素坐标
            gap_x = output['gap_centers'][:, 0]
            gap_y = output['gap_centers'][:, 1]
            piece_x = output['piece_centers'][:, 0]
            piece_y = output['piece_centers'][:, 1]
            
            return {
                'gap_x': gap_x.cpu().numpy(),
                'gap_y': gap_y.cpu().numpy(),
                'piece_x': piece_x.cpu().numpy(),
                'piece_y': piece_y.cpu().numpy(),
                'scores': output['scores'].cpu().numpy()
            }


def create_model(pretrained=False, **kwargs):
    """
    创建PMN-R3-FP模型
    Args:
        pretrained: 是否加载预训练权重
        **kwargs: 其他模型参数
    Returns:
        model: PMN_R3_FP模型实例
    """
    model = PMN_R3_FP(**kwargs)
    
    if pretrained:
        # 加载预训练权重 (如果有的话)
        # checkpoint = torch.load('path/to/checkpoint.pth')
        # model.load_state_dict(checkpoint['state_dict'])
        pass
    
    return model


if __name__ == "__main__":
    # 测试代码
    model = create_model(pretrained=False)
    
    # 打印模型结构
    print("PMN-R3-FP Model Structure:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试前向传播
    # 从配置获取输入尺寸
    input_shape = model_config.get_input_shape(batch_size=2)
    print(f"\nTesting with input shape: {input_shape}")
    images = torch.randn(*input_shape)
    output = model(images, return_features=True)
    
    print("\nModel Output:")
    print(f"  Gap centers: {output['gap_centers'].shape}")
    print(f"  Piece centers: {output['piece_centers'].shape}")
    print(f"  Scores: {output['scores'].shape}")
    print(f"  Geometry: {output['geometry'].shape}")
    print(f"  Match matrix: {output['match_matrix'].shape}")
    
    # 测试推理接口
    predictions = model.predict(images)
    print("\nPredictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")