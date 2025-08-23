"""Version information for sider-captcha-solver"""
import yaml
import os

# 从配置文件读取版本号
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'version.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    __version__ = config['version']

__author__ = 'Tomokotokiyoshi'
__description__ = 'Industrial-grade slider CAPTCHA solver using CNN architecture'