# -*- coding: utf-8 -*-
"""
验证码混淆系统
提供多种混淆策略来增强验证码的抗破解能力
"""

# 从配置文件读取版本号（组件版本与主版本保持一致）
import yaml
import os

_config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'version.yaml')
with open(_config_path, 'r') as f:
    _config = yaml.safe_load(f)
    __version__ = _config['version']

# 暂时留空，等各模块实现后再导入
# 这样可以避免循环依赖问题
__all__ = [
    # 基础类
    'ConfusionStrategy',
    'ConfusionResult',
    
    # 管理器和配置
    'ConfusionManager',
    'ConfusionConfig',
    'ConfusionBuilder',
    
    # API
    'CaptchaConfusionAPI',
    
    # 具体策略
    'HighlightConfusion',
    'RotationConfusion',
    'DualGapSameAxisConfusion',
    'DualGapDifferentAxisConfusion',
    'HollowCenterConfusion'
]