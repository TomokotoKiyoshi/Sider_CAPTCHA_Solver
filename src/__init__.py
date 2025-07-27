# -*- coding: utf-8 -*-
"""
Sider_CAPTCHA_Solver - Industrial-grade slider CAPTCHA recognition system

This package provides a high-precision slider CAPTCHA recognition solution
based on deep learning, utilizing an improved CenterNet architecture.
"""

__version__ = "1.0.3"
__author__ = "TomokotoKiyoshi"
__email__ = ""
__license__ = "MIT"

# Import main components for easier access
try:
    from .models.captcha_solver import CaptchaSolver
    from .models.predictor import CaptchaPredictor
except ImportError:
    # For development environment
    from models.captcha_solver import CaptchaSolver
    from models.predictor import CaptchaPredictor

__all__ = [
    "CaptchaSolver",
    "CaptchaPredictor",
    "__version__",
    "__author__",
    "__license__",
]
