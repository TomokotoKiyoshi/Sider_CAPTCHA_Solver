# -*- coding: utf-8 -*-
"""
Sider_CAPTCHA_Solver - Industrial-grade slider CAPTCHA recognition system

This package provides a high-precision slider CAPTCHA recognition solution
based on deep learning, utilizing an improved CenterNet architecture.
"""

# 从配置文件读取版本号
import yaml
import os

_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'version.yaml')
with open(_config_path, 'r') as f:
    _config = yaml.safe_load(f)
    __version__ = _config['version']
__author__ = "TomokotoKiyoshi"
__email__ = ""
__license__ = "MIT"

# Import main components for easier access
# Note: Model components will be imported when implemented
# try:
#     from .models.captcha_solver import CaptchaSolver
#     from .models.predictor import CaptchaPredictor
# except ImportError:
#     # For development environment
#     from models.captcha_solver import CaptchaSolver
#     from models.predictor import CaptchaPredictor

__all__ = [
    # "CaptchaSolver",
    # "CaptchaPredictor",
    "__version__",
    "__author__",
    "__license__",
]
