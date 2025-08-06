# -*- coding: utf-8 -*-
"""
验证码混淆系统
提供多种混淆策略来增强验证码的抗破解能力
"""

__version__ = "1.0.0"

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