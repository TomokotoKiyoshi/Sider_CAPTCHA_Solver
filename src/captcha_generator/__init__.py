# -*- coding: utf-8 -*-
"""
验证码生成器模块

模块结构：
- puzzle_shapes_generator: 拼图形状生成
  - create_common_puzzle_piece: 普通拼图（凸起/凹陷边缘）
  - create_special_puzzle_piece: 特殊形状（圆形、正方形、三角形、六边形）
- lighting: 光照效果
  - gap_lighting: 缺口光照效果（阴影/高光）
  - slider_lighting: 滑块光照效果（3D凸起）
- confusion_system: 混淆系统
  - strategies: 各种混淆策略
- label_generator: 训练标签生成
  - CaptchaLabelGenerator: 标签生成器类
  - create_label_from_captcha_result: 便捷标签生成函数
"""

# 导入标签生成器
from .label_generator import CaptchaLabelGenerator, create_label_from_captcha_result

__all__ = ['CaptchaLabelGenerator', 'create_label_from_captcha_result']
