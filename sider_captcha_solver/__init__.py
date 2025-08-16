"""Sider CAPTCHA Solver - 工业级滑块验证码识别系统

高精度CNN架构的滑块验证码识别解决方案
"""

import sys
from pathlib import Path

# 添加src到Python路径，使得可以导入src中的代码
package_dir = Path(__file__).parent.parent
src_path = package_dir / 'src'
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 从新的高层API导入
from .predictor import CaptchaPredictor
from .api import solve, solve_batch, visualize, CaptchaSolver
from .__version__ import __version__

# 暴露主要接口
__all__ = [
    # 简单API
    'solve',
    'solve_batch', 
    'visualize',
    # 类API
    'CaptchaSolver',
    'CaptchaPredictor',
    # 版本
    '__version__'
]

# 保留兼容性
def predict(image_path, model_path='best', device='auto'):
    """快速预测接口（已废弃，请使用solve）
    
    Args:
        image_path: 图片路径
        model_path: 模型路径或预设名称('best'/'v1.1.0')
        device: 运行设备('auto'/'cuda'/'cpu')
    
    Returns:
        Dict: 包含滑动距离、坐标、置信度等信息的字典
    """
    predictor = CaptchaPredictor(model_path, device)
    return predictor.predict(image_path)