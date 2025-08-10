# -*- coding: utf-8 -*-
"""
Configuration Module
配置模块 - 提供统一的配置加载接口
"""

from .config_loader import ConfigLoader
from .dataset_config import DatasetConfig, get_dataset_config
from .size_confusion_config import SizeConfusionConfig, get_size_confusion_config
from .confusion_config import ConfusionConfig, get_confusion_config
from .model_config import ModelConfig, get_model_config

# 创建全局配置实例
config_loader = ConfigLoader()

# 导出的接口
__all__ = [
    'ConfigLoader',
    'DatasetConfig',
    'SizeConfusionConfig',
    'ConfusionConfig',
    'ModelConfig',
    'config_loader',
    'get_dataset_config',
    'get_size_confusion_config',
    'get_confusion_config',
    'get_model_config'
]