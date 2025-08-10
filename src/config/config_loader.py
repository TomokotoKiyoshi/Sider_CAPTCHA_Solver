# -*- coding: utf-8 -*-
"""
Unified Configuration Loader
统一配置加载器 - 从YAML文件加载所有配置
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """统一的配置加载器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的config文件夹
        """
        if config_dir is None:
            # 获取项目根目录
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            config_dir = project_root / 'config'
        else:
            config_dir = Path(config_dir)
        
        self.config_dir = config_dir
        self._configs = {}
        
        # 自动加载所有配置文件
        self._load_all_configs()
    
    def _load_all_configs(self):
        """加载配置目录下的所有YAML文件"""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
        
        # 加载所有.yaml文件
        for yaml_file in self.config_dir.glob('*.yaml'):
            config_name = yaml_file.stem
            self._configs[config_name] = self._load_yaml(yaml_file)
        
        # 加载所有.yml文件
        for yml_file in self.config_dir.glob('*.yml'):
            config_name = yml_file.stem
            self._configs[config_name] = self._load_yaml(yml_file)
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """
        加载单个YAML文件
        
        Args:
            file_path: YAML文件路径
            
        Returns:
            配置字典
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, config_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            config_path: 配置路径，格式为 "file.section.key"
            default: 默认值
            
        Returns:
            配置值
            
        Example:
            >>> loader.get('captcha_config.dataset.scale.min_backgrounds')
            2000
        """
        parts = config_path.split('.')
        
        if not parts:
            return default
        
        # 第一部分是配置文件名
        config_name = parts[0]
        if config_name not in self._configs:
            return default
        
        # 逐级获取配置
        value = self._configs[config_name]
        for part in parts[1:]:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        获取整个配置文件的内容
        
        Args:
            config_name: 配置文件名（不含扩展名）
            
        Returns:
            配置字典
        """
        return self._configs.get(config_name)
    
    def reload(self):
        """重新加载所有配置"""
        self._configs.clear()
        self._load_all_configs()
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]):
        """
        保存配置到文件
        
        Args:
            config_name: 配置文件名（不含扩展名）
            config_data: 配置数据
        """
        file_path = self.config_dir / f"{config_name}.yaml"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, 
                     allow_unicode=True, sort_keys=False)
        
        # 更新内存中的配置
        self._configs[config_name] = config_data
    
    def update(self, config_path: str, value: Any):
        """
        更新配置值
        
        Args:
            config_path: 配置路径，格式为 "file.section.key"
            value: 新值
        """
        parts = config_path.split('.')
        
        if not parts:
            return
        
        # 第一部分是配置文件名
        config_name = parts[0]
        if config_name not in self._configs:
            self._configs[config_name] = {}
        
        # 逐级设置配置
        config = self._configs[config_name]
        for part in parts[1:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        if parts[-1:]:
            config[parts[-1]] = value
        
        # 保存到文件
        self.save_config(config_name, self._configs[config_name])
    
    @property
    def available_configs(self) -> list:
        """获取所有可用的配置文件名"""
        return list(self._configs.keys())
    
    def print_config(self, config_name: Optional[str] = None):
        """
        打印配置信息
        
        Args:
            config_name: 配置文件名，None则打印所有
        """
        if config_name:
            if config_name in self._configs:
                print(f"\n=== Configuration: {config_name} ===")
                self._print_dict(self._configs[config_name])
            else:
                print(f"Configuration '{config_name}' not found")
        else:
            print(f"\n=== All Configurations ===")
            print(f"Config directory: {self.config_dir}")
            print(f"Available configs: {', '.join(self.available_configs)}")
            for name, config in self._configs.items():
                print(f"\n--- {name} ---")
                self._print_dict(config, indent=2)
    
    def _print_dict(self, d: dict, indent: int = 0):
        """递归打印字典"""
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{' ' * indent}{key}:")
                self._print_dict(value, indent + 2)
            elif isinstance(value, list):
                print(f"{' ' * indent}{key}: {value}")
            else:
                print(f"{' ' * indent}{key}: {value}")


# 便捷访问函数
def get_config(config_path: str, default: Any = None) -> Any:
    """
    全局配置访问函数
    
    Args:
        config_path: 配置路径
        default: 默认值
        
    Returns:
        配置值
    """
    from . import config_loader
    return config_loader.get(config_path, default)


if __name__ == "__main__":
    # 测试配置加载器
    loader = ConfigLoader()
    
    print("Available configurations:")
    for config in loader.available_configs:
        print(f"  - {config}")
    
    # 测试获取配置
    print("\nTest configuration access:")
    
    # 数据集配置
    min_bg = loader.get('captcha_config.dataset.scale.min_backgrounds')
    print(f"Min backgrounds: {min_bg}")
    
    # 目标尺寸 (从model_config读取)
    target_size = loader.get('model_config.input.target_size')
    print(f"Target size: {target_size}")
    
    # 模型配置
    model_input = loader.get('model_config.input.target_size')
    print(f"Model input size: {model_input}")
    
    # 打印完整配置
    print("\n" + "="*60)
    loader.print_config('captcha_config')