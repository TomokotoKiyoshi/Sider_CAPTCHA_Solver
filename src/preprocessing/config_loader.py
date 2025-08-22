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
            'heatmap': ['sigma']  # 删除 'threshold'，预处理不使用它
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
    preprocessing_config = config.get('preprocessing', {})
    
    # 添加量化配置的默认值（如果没有配置）
    if 'quantization' not in preprocessing_config:
        preprocessing_config['quantization'] = {
            'enabled': False,
            'bits_to_keep': 8
        }
    
    return preprocessing_config


def get_dataset_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取数据集生成相关的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        数据集配置字典
    """
    config = load_config(config_path)
    return config.get('dataset', {})


def get_data_split_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取数据集划分相关的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        数据集划分配置字典，包含:
        - split_ratio: 训练集/验证集/测试集的比例
        - options: 划分选项（随机种子、是否按图片分层等）
    """
    config = load_config(config_path)
    data_split = config.get('data_split', {})
    
    # 提供默认值
    default_split = {
        'split_ratio': {
            'train': 0.8,
            'val': 0.1,
            'test': 0.1
        },
        'options': {
            'random_seed': 42,
            'stratify_by_image': True,
            'shuffle': True,
            'auto_split': False
        }
    }
    
    # 合并默认值和配置值
    if not data_split:
        return default_split
    
    # 确保所有必要的键都存在
    if 'split_ratio' not in data_split:
        data_split['split_ratio'] = default_split['split_ratio']
    if 'options' not in data_split:
        data_split['options'] = default_split['options']
    
    # 补充缺失的子键
    for key in default_split['split_ratio']:
        if key not in data_split['split_ratio']:
            data_split['split_ratio'][key] = default_split['split_ratio'][key]
    
    for key in default_split['options']:
        if key not in data_split['options']:
            data_split['options'][key] = default_split['options'][key]
    
    return data_split


def get_paths_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取路径相关的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        路径配置字典
    """
    config = load_config(config_path)
    return config.get('paths', {})


def get_output_structure_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取输出结构相关的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        输出结构配置字典
    """
    # 如果是部分配置文件，只加载输出结构部分
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    partial_config = yaml.safe_load(f)
                    # 如果只有output_structure，直接返回
                    if 'output_structure' in partial_config and 'preprocessing' not in partial_config:
                        return partial_config.get('output_structure', {})
            except:
                pass
    
    # 否则加载完整配置
    config = load_config(config_path)
    return config.get('output_structure', {})


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
    
    # threshold 已移除，预处理不使用它
    # 如果配置中有 threshold，忽略它即可
    
    return True


def validate_data_split_config(config: Dict[str, Any]) -> bool:
    """
    验证数据集划分配置的合理性
    
    Args:
        config: 数据集划分配置字典
    
    Returns:
        配置是否有效
    
    Raises:
        ValueError: 配置参数不合理
    """
    # 验证划分比例
    split_ratio = config.get('split_ratio', {})
    train_ratio = split_ratio.get('train', 0)
    val_ratio = split_ratio.get('val', 0)
    test_ratio = split_ratio.get('test', 0)
    
    # 检查比例和
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # 检查每个比例的范围
    for name, ratio in [('train', train_ratio), ('val', val_ratio), ('test', test_ratio)]:
        if ratio < 0 or ratio > 1:
            raise ValueError(f"{name} ratio must be between 0 and 1, got {ratio}")
    
    # 验证选项
    options = config.get('options', {})
    seed = options.get('random_seed')
    if seed is not None and not isinstance(seed, int):
        raise ValueError(f"random_seed must be an integer, got {type(seed).__name__}")
    
    return True


def get_full_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取完整的配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        完整配置字典
    """
    return load_config(config_path)