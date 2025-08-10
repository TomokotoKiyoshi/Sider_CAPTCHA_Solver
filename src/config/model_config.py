# -*- coding: utf-8 -*-
"""
模型配置加载器
从YAML文件加载模型配置
"""
import os
import yaml
from typing import Dict, Any, Optional
from .config_loader import ConfigLoader

class ModelConfig:
    """模型配置管理器"""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """
        初始化配置管理器
        
        Args:
            config_loader: ConfigLoader实例
        """
        if config_loader is None:
            config_loader = ConfigLoader()
        
        self.loader = config_loader
        self.config = self.loader.get_config('model_config')
        
        if self.config is None:
            raise FileNotFoundError("Model configuration not found in config/model_config.yaml")
        
        # 快速访问属性
        self._setup_attributes()
    
    def _setup_attributes(self):
        """设置快速访问属性"""
        # 输入配置
        self.target_size = tuple(self.config['input']['target_size'])  # (width, height)
        self.tensor_shape = tuple(self.config['input']['tensor_shape'])  # (C, H, W)
        
        # 从target_size推导tensor维度
        width, height = self.target_size
        self.input_height = height
        self.input_width = width
        self.input_channels = self.config['input']['channels']['rgb']
        
        # 骨干网络配置
        self.backbone_type = self.config['backbone']['type']
        self.backbone_stages = self.config['backbone']['stages']
        
        # FPN-PAN配置
        self.fpn_in_channels = self.config['fpn_pan']['in_channels']
        self.fpn_out_channels = self.config['fpn_pan']['out_channels']
        
        # Proposal配置
        self.proposal_strides = self.config['proposal']['strides']
        self.proposal_top_k = self.config['proposal']['top_k']
        
        # ROI配置
        self.roi_region_size = self.config['roi']['region']['output_size']
        self.roi_region_scale = self.config['roi']['region']['spatial_scale']
        self.roi_shape_size = self.config['roi']['shape']['output_size']
        self.roi_shape_scale = self.config['roi']['shape']['spatial_scale']
        
        # SE2变换器配置
        self.se2_d_model = self.config['se2_transformer']['d_model']
        self.se2_n_heads = self.config['se2_transformer']['n_heads']
        self.se2_n_layers = self.config['se2_transformer']['n_layers']
        self.se2_dropout = self.config['se2_transformer']['dropout']
        
        # Shape/SDF模块配置
        self.shape_channels_quarter = self.config['shape_sdf']['channels']['quarter']
        self.shape_channels_half = self.config['shape_sdf']['channels']['half']
        self.shape_channels_full = self.config['shape_sdf']['channels']['full']
        self.sdf_range = self.config['shape_sdf']['sdf_range']
        self.sdf_decoder_in_channels = self.config['shape_sdf']['decoder']['in_channels']
        self.sdf_decoder_hidden_dim = self.config['shape_sdf']['decoder']['hidden_dim']
        self.edge_rgb_weights = self.config['shape_sdf']['edge_detection']['rgb_weights']
        
        # 损失函数配置
        self.focal_alpha = self.config['loss']['focal']['alpha']
        self.focal_gamma = self.config['loss']['focal']['gamma']
        self.centernet_alpha = self.config['loss']['centernet']['alpha']
        self.centernet_beta = self.config['loss']['centernet']['beta']
        self.sdf_truncation = self.config['loss']['sdf']['truncation']
        self.gaussian_sigma = self.config['loss']['gaussian']['sigma']
        self.loss_weights = self.config['loss']['weights']
        
        # SE2变换器扩展配置
        self.se2_positional_max_len = self.config['se2_transformer']['positional_encoding']['max_len']
        self.se2_positional_base = self.config['se2_transformer']['positional_encoding']['base']
        self.se2_geometric_residual_scale = self.config['se2_transformer']['geometric_refinement']['residual_scale']
        
        # HRNet配置
        self.hrnet_stage2_num_branches = self.config['backbone']['hrnet']['stage2']['num_branches']
        self.hrnet_stage2_num_blocks = self.config['backbone']['hrnet']['stage2']['num_blocks']
        self.hrnet_stage3_num_branches = self.config['backbone']['hrnet']['stage3']['num_branches']
        self.hrnet_stage3_num_blocks = self.config['backbone']['hrnet']['stage3']['num_blocks']
        self.hrnet_stage4_num_branches = self.config['backbone']['hrnet']['stage4']['num_branches']
        self.hrnet_stage4_num_blocks = self.config['backbone']['hrnet']['stage4']['num_blocks']
        
        # 排序判别器配置
        self.ranking_hidden_dims = self.config['inference']['ranking_discriminator']['hidden_dims']
        self.ranking_dropout = self.config['inference']['ranking_discriminator']['dropout']
        
        # 推理配置
        self.coord_scale = self.config['inference']['coord_scale']
        self.confidence_threshold = self.config['inference']['confidence_threshold']
        self.nms_threshold = self.config['inference']['nms_threshold']
    
    def get_input_shape(self, batch_size: int = 1) -> tuple:
        """
        获取输入张量形状
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (B, C, H, W) 格式的张量形状
        """
        return (batch_size, self.input_channels, self.input_height, self.input_width)
    
    def get_letterbox_config(self) -> Dict[str, Any]:
        """获取letterbox配置"""
        return {
            'use_letterbox': self.config['augmentation']['use_letterbox'],
            'keep_aspect_ratio': self.config['augmentation']['keep_aspect_ratio'],
            'padding_color': self.config['augmentation']['padding_color']
        }
    
    def validate_compatibility(self) -> bool:
        """
        验证配置的兼容性
        
        Returns:
            是否通过验证
        """
        # 检查输入尺寸是否为32的倍数
        if self.input_width % 32 != 0 or self.input_height % 32 != 0:
            print(f"Warning: Input size ({self.input_width}, {self.input_height}) is not a multiple of 32")
            return False
        
        # 检查tensor_shape是否与target_size一致
        expected_shape = (self.input_channels, self.input_height, self.input_width)
        if self.tensor_shape != expected_shape:
            print(f"Warning: tensor_shape {self.tensor_shape} doesn't match expected {expected_shape}")
            return False
        
        return True
    
    def print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("Model Configuration")
        print("=" * 60)
        print(f"\n[Input]")
        print(f"  Target size (WxH): {self.target_size[0]}x{self.target_size[1]}")
        print(f"  Tensor shape (CxHxW): {self.tensor_shape}")
        print(f"  Input channels: {self.input_channels}")
        
        print(f"\n[Backbone]")
        print(f"  Type: {self.backbone_type}")
        print(f"  Stages: {len(self.backbone_stages)}")
        
        print(f"\n[FPN-PAN]")
        print(f"  In channels: {self.fpn_in_channels}")
        print(f"  Out channels: {self.fpn_out_channels}")
        
        print(f"\n[Proposal]")
        print(f"  Strides: {self.proposal_strides}")
        print(f"  Top-K: {self.proposal_top_k}")
        
        print(f"\n[SE2 Transformer]")
        print(f"  Model dimension: {self.se2_d_model}")
        print(f"  Heads: {self.se2_n_heads}")
        print(f"  Layers: {self.se2_n_layers}")
        
        print(f"\n[Inference]")
        print(f"  Coordinate scale: {self.coord_scale}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  NMS threshold: {self.nms_threshold}")
        print("=" * 60)
    
    def update(self, key: str, value: Any):
        """
        更新配置值
        
        Args:
            key: 配置键（支持嵌套，用.分隔）
            value: 新值
        """
        full_key = f'model_config.{key}'
        self.loader.update(full_key, value)
        
        # 重新加载配置
        self.config = self.loader.get_config('model_config')
        self._setup_attributes()
    
    def save(self):
        """
        保存配置到文件
        """
        self.loader.save_config('model_config', self.config)
        print(f"Model configuration saved")


# 创建全局实例
model_config = None

def get_model_config() -> ModelConfig:
    """获取模型配置单例"""
    global model_config
    if model_config is None:
        model_config = ModelConfig()
    return model_config


if __name__ == "__main__":
    # Test configuration
    config = get_model_config()
    
    # 打印配置
    config.print_config()
    
    # 验证兼容性
    if config.validate_compatibility():
        print("\nConfiguration validation passed")
    else:
        print("\nConfiguration validation failed")
    
    # 测试获取输入形状
    print(f"\nInput shape for batch_size=4: {config.get_input_shape(4)}")
    
    # 测试letterbox配置
    print(f"\nLetterbox config: {config.get_letterbox_config()}")