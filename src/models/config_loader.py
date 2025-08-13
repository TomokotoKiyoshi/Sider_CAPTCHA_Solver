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