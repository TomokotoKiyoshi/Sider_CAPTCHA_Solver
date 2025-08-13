# -*- coding: utf-8 -*-
"""
配置加载器
从YAML文件加载配置并提供给预处理器
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
    
    Returns:
        配置字典
    
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式错误或缺少必要的键
    """
    # 默认配置文件路径
    if config_path is None:
        config_path = Path(__file__).parents[2] / 'config' / 'preprocessing_config.yaml'
    else:
        config_path = Path(config_path)
    
    # 检查文件是否存在
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # 加载YAML文件
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Config file format error: {e}")
    
    # 验证必要的配置键
    required_keys = {
        'preprocessing': {
            'letterbox': ['target_size', 'fill_value'],
            'coordinate': ['downsample'],
            'heatmap': ['sigma', 'threshold']
        }
    }
    
    # 递归验证配置键
    def validate_keys(config_dict: Dict, required: Dict, path: str = ""):
        for key, value in required.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in config_dict:
                raise ValueError(f"Config missing required key: {current_path}")
            
            if isinstance(value, dict):
                validate_keys(config_dict[key], value, current_path)
            elif isinstance(value, list):
                for subkey in value:
                    subpath = f"{current_path}.{subkey}"
                    if subkey not in config_dict[key]:
                        raise ValueError(f"Config missing required key: {subpath}")
    
    validate_keys(config, required_keys)
    
    return config


def get_preprocessing_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取预处理相关的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        预处理配置字典
    """
    config = load_config(config_path)
    return config.get('preprocessing', {})


def validate_preprocessing_config(config: Dict[str, Any]) -> bool:
    """
    验证预处理配置的合理性
    
    Args:
        config: 预处理配置字典
    
    Returns:
        配置是否有效
    
    Raises:
        ValueError: 配置参数不合理
    """
    # 验证letterbox配置
    letterbox = config.get('letterbox', {})
    target_size = letterbox.get('target_size')
    if not target_size or len(target_size) != 2:
        raise ValueError("target_size must be a list with two elements [width, height]")
    
    w, h = target_size
    if w <= 0 or h <= 0:
        raise ValueError(f"Target size must be positive: {target_size}")
    
    fill_value = letterbox.get('fill_value')
    if fill_value is None or not (0 <= fill_value <= 255):
        raise ValueError(f"fill_value must be in range 0-255: {fill_value}")
    
    # 验证coordinate配置
    coord = config.get('coordinate', {})
    downsample = coord.get('downsample')
    if not downsample or downsample <= 0:
        raise ValueError(f"downsample must be positive: {downsample}")
    
    # 验证下采样率与目标尺寸的兼容性
    if w % downsample != 0 or h % downsample != 0:
        raise ValueError(f"Target size {target_size} must be divisible by downsample rate {downsample}")
    
    # 验证heatmap配置
    heatmap = config.get('heatmap', {})
    sigma = heatmap.get('sigma')
    if not sigma or sigma <= 0:
        raise ValueError(f"sigma must be positive: {sigma}")
    
    threshold = heatmap.get('threshold')
    if not (0 < threshold < 1):
        raise ValueError(f"threshold must be in range (0,1): {threshold}")
    
    return True