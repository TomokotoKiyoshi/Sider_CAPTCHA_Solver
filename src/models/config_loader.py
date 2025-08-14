# -*- coding: utf-8 -*-
"""
模型配置加载器
从YAML文件加载模型配置
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_model_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载模型配置
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
    
    Returns:
        模型配置字典
    
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置格式错误或缺少必要的键
    """
    if config_path is None:
        config_path = Path(__file__).parents[2] / 'config' / 'model_config.yaml'
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式错误: {e}")
    
    return config


def get_stem_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取Stem模块的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Stem配置字典
    
    Raises:
        ValueError: 配置缺少必要的Stem配置
    """
    config = load_model_config(config_path)
    
    # 检查backbone配置
    if 'backbone' not in config:
        raise ValueError("配置缺少backbone部分")
    
    backbone = config['backbone']
    if 'lite_hrnet' not in backbone:
        raise ValueError("配置缺少lite_hrnet部分")
    
    lite_hrnet = backbone['lite_hrnet']
    if 'stem' not in lite_hrnet:
        raise ValueError("配置缺少stem部分")
    
    stem_config = lite_hrnet['stem']
    
    # 验证必要的键
    required_keys = ['in_channels', 'out_channels', 'expansion']
    for key in required_keys:
        if key not in stem_config:
            raise ValueError(f"Stem配置缺少必要的键: {key}")
    
    return stem_config


def get_input_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取输入配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        输入配置字典
    """
    config = load_model_config(config_path)
    
    if 'input' not in config:
        raise ValueError("配置缺少input部分")
    
    return config['input']


def get_stage2_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取Stage2模块的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Stage2配置字典，包含:
        - in_channels: 输入通道数（从stem配置获取）
        - channels: 各分支通道数列表
        - num_blocks: 各分支的LiteBlock数量列表
        - expansion: 扩张倍率
    
    Raises:
        ValueError: 配置缺少必要的Stage2配置
    """
    config = load_model_config(config_path)
    
    # 检查backbone配置
    if 'backbone' not in config:
        raise ValueError("配置缺少backbone部分")
    
    backbone = config['backbone']
    if 'lite_hrnet' not in backbone:
        raise ValueError("配置缺少lite_hrnet部分")
    
    lite_hrnet = backbone['lite_hrnet']
    
    # 获取Stage2配置
    if 'stage2' not in lite_hrnet:
        raise ValueError("配置缺少stage2部分")
    
    stage2_config = lite_hrnet['stage2'].copy()
    
    # 验证必要的键
    required_keys = ['channels', 'num_blocks', 'expansion']
    for key in required_keys:
        if key not in stage2_config:
            raise ValueError(f"Stage2配置缺少必要的键: {key}")
    
    # 验证列表长度
    if len(stage2_config['channels']) != 2:
        raise ValueError("Stage2 channels必须包含2个元素 [1/4分辨率, 1/8分辨率]")
    
    if len(stage2_config['num_blocks']) != 2:
        raise ValueError("Stage2 num_blocks必须包含2个元素")
    
    # 从stem配置获取输入通道数
    if 'stem' not in lite_hrnet:
        raise ValueError("配置缺少stem部分")
    
    stem_config = lite_hrnet['stem']
    if 'out_channels' not in stem_config:
        raise ValueError("Stem配置缺少out_channels")
    
    # 添加输入通道数（来自stem的输出）
    stage2_config['in_channels'] = stem_config['out_channels']
    
    return stage2_config


def get_stage3_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取Stage3模块的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Stage3配置字典，包含:
        - channels: 各分支通道数列表 [1/4, 1/8, 1/16分辨率]
        - num_blocks: 各分支的LiteBlock数量列表
        - expansion: 扩张倍率
    
    Raises:
        ValueError: 配置缺少必要的Stage3配置
    """
    config = load_model_config(config_path)
    
    # 检查backbone配置
    if 'backbone' not in config:
        raise ValueError("配置缺少backbone部分")
    
    backbone = config['backbone']
    if 'lite_hrnet' not in backbone:
        raise ValueError("配置缺少lite_hrnet部分")
    
    lite_hrnet = backbone['lite_hrnet']
    
    # 获取Stage3配置
    if 'stage3' not in lite_hrnet:
        raise ValueError("配置缺少stage3部分")
    
    stage3_config = lite_hrnet['stage3'].copy()
    
    # 验证必要的键
    required_keys = ['channels', 'num_blocks', 'expansion']
    for key in required_keys:
        if key not in stage3_config:
            raise ValueError(f"Stage3配置缺少必要的键: {key}")
    
    # 验证列表长度
    if len(stage3_config['channels']) != 3:
        raise ValueError("Stage3 channels必须包含3个元素 [1/4分辨率, 1/8分辨率, 1/16分辨率]")
    
    if len(stage3_config['num_blocks']) != 3:
        raise ValueError("Stage3 num_blocks必须包含3个元素")
    
    return stage3_config


def get_stage4_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取Stage4模块的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Stage4配置字典，包含:
        - channels: 各分支通道数列表 [1/4, 1/8, 1/16, 1/32分辨率]
        - num_blocks: 各分支的LiteBlock数量列表
        - expansion: 扩张倍率
    
    Raises:
        ValueError: 配置缺少必要的Stage4配置
    """
    config = load_model_config(config_path)
    
    # 检查backbone配置
    if 'backbone' not in config:
        raise ValueError("配置缺少backbone部分")
    
    backbone = config['backbone']
    if 'lite_hrnet' not in backbone:
        raise ValueError("配置缺少lite_hrnet部分")
    
    lite_hrnet = backbone['lite_hrnet']
    
    # 获取Stage4配置
    if 'stage4' not in lite_hrnet:
        raise ValueError("配置缺少stage4部分")
    
    stage4_config = lite_hrnet['stage4'].copy()
    
    # 验证必要的键
    required_keys = ['channels', 'num_blocks', 'expansion']
    for key in required_keys:
        if key not in stage4_config:
            raise ValueError(f"Stage4配置缺少必要的键: {key}")
    
    # 验证列表长度
    if len(stage4_config['channels']) != 4:
        raise ValueError("Stage4 channels必须包含4个元素 [1/4, 1/8, 1/16, 1/32分辨率]")
    
    if len(stage4_config['num_blocks']) != 4:
        raise ValueError("Stage4 num_blocks必须包含4个元素")
    
    return stage4_config


def get_stage5_lite_fpn_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取Stage5 LiteFPN模块的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Stage5 LiteFPN配置字典，包含:
        - in_channels: 输入各分支的通道数列表 [1/4, 1/8, 1/16, 1/32分辨率]
        - fpn_channels: FPN统一通道数
        - fusion_type: 融合类型 ('add', 'weighted', 'attention')
        - return_pyramid: 是否返回中间金字塔特征
    
    Raises:
        ValueError: 配置缺少必要的Stage5配置
    """
    config = load_model_config(config_path)
    
    # 检查backbone配置
    if 'backbone' not in config:
        raise ValueError("配置缺少backbone部分")
    
    backbone = config['backbone']
    if 'lite_hrnet' not in backbone:
        raise ValueError("配置缺少lite_hrnet部分")
    
    lite_hrnet = backbone['lite_hrnet']
    
    # 获取Stage5 LiteFPN配置
    if 'stage5_lite_fpn' not in lite_hrnet:
        raise ValueError("配置缺少stage5_lite_fpn部分")
    
    stage5_config = lite_hrnet['stage5_lite_fpn'].copy()
    
    # 验证必要的键
    required_keys = ['in_channels', 'fpn_channels', 'fusion_type', 'return_pyramid']
    for key in required_keys:
        if key not in stage5_config:
            raise ValueError(f"Stage5 LiteFPN配置缺少必要的键: {key}")
    
    # 验证列表长度
    if len(stage5_config['in_channels']) != 4:
        raise ValueError("Stage5 in_channels必须包含4个元素 [1/4, 1/8, 1/16, 1/32分辨率]")
    
    # 验证融合类型
    valid_fusion_types = ['add', 'weighted', 'attention']
    if stage5_config['fusion_type'] not in valid_fusion_types:
        raise ValueError(f"不支持的融合类型: {stage5_config['fusion_type']}. "
                        f"支持的类型: {valid_fusion_types}")
    
    return stage5_config


def get_stage6_dual_head_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取Stage6 双头预测网络的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Stage6 双头配置字典，包含:
        - in_channels: 输入通道数（来自Stage5 LiteFPN）
        - mid_channels: 中间层通道数
        - use_angle: 是否使用角度预测头
        - use_tanh_offset: 是否使用tanh限制偏移范围
    
    Raises:
        ValueError: 配置缺少必要的Stage6配置
    """
    config = load_model_config(config_path)
    
    # 检查backbone配置
    if 'backbone' not in config:
        raise ValueError("配置缺少backbone部分")
    
    backbone = config['backbone']
    if 'lite_hrnet' not in backbone:
        raise ValueError("配置缺少lite_hrnet部分")
    
    lite_hrnet = backbone['lite_hrnet']
    
    # 获取Stage6 双头配置
    if 'stage6_dual_head' not in lite_hrnet:
        raise ValueError("配置缺少stage6_dual_head部分")
    
    stage6_config = lite_hrnet['stage6_dual_head'].copy()
    
    # 验证必要的键
    required_keys = ['in_channels', 'mid_channels', 'use_angle', 'use_tanh_offset']
    for key in required_keys:
        if key not in stage6_config:
            raise ValueError(f"Stage6 双头配置缺少必要的键: {key}")
    
    # 验证输入通道数应该与Stage5的FPN通道数一致
    if 'stage5_lite_fpn' in lite_hrnet:
        fpn_channels = lite_hrnet['stage5_lite_fpn'].get('fpn_channels', 128)
        if stage6_config['in_channels'] != fpn_channels:
            raise ValueError(f"Stage6输入通道数({stage6_config['in_channels']})应该与"
                           f"Stage5 FPN通道数({fpn_channels})一致")
    
    return stage6_config


# 使用示例
if __name__ == "__main__":
    # 加载完整配置
    config = load_model_config()
    print("模型配置加载成功")
    
    # 获取Stem配置
    stem_config = get_stem_config()
    print("\nStem配置:")
    print(f"  输入通道: {stem_config['in_channels']}")
    print(f"  输出通道: {stem_config['out_channels']}")
    print(f"  扩张倍率: {stem_config['expansion']}")
    
    # 获取输入配置
    input_config = get_input_config()
    print("\n输入配置:")
    print(f"  总通道数: {input_config['channels']['total']}")
    print(f"  张量形状: {input_config['tensor_shape']}")
    
    # 获取Stage2配置
    stage2_config = get_stage2_config()
    print("\nStage2配置:")
    print(f"  输入通道: {stage2_config['in_channels']}")
    print(f"  分支通道: {stage2_config['channels']}")
    print(f"  块数量: {stage2_config['num_blocks']}")
    print(f"  扩张倍率: {stage2_config['expansion']}")
    
    # 获取Stage3配置
    stage3_config = get_stage3_config()
    print("\nStage3配置:")
    print(f"  分支通道: {stage3_config['channels']}")
    print(f"  块数量: {stage3_config['num_blocks']}")
    print(f"  扩张倍率: {stage3_config['expansion']}")
    
    # 获取Stage4配置
    stage4_config = get_stage4_config()
    print("\nStage4配置:")
    print(f"  分支通道: {stage4_config['channels']}")
    print(f"  块数量: {stage4_config['num_blocks']}")
    print(f"  扩张倍率: {stage4_config['expansion']}")