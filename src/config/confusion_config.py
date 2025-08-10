# -*- coding: utf-8 -*-
"""
混淆策略配置文件
从YAML文件加载混淆效果参数
"""
from typing import Optional, Dict, Any, List
from .config_loader import ConfigLoader


class ConfusionConfig:
    """混淆策略参数配置"""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """初始化配置"""
        if config_loader is None:
            config_loader = ConfigLoader()
        
        self.loader = config_loader
        self._load_config()
    
    def _load_config(self):
        """从YAML文件加载配置"""
        # 基础路径
        base_path = 'captcha_config.confusion_strategies'
        
        # ========== 柏林噪声 (PerlinNoise) ==========
        noise_min = self.loader.get(f'{base_path}.perlin_noise.noise_strength.min')
        noise_max = self.loader.get(f'{base_path}.perlin_noise.noise_strength.max')
        noise_scale = self.loader.get(f'{base_path}.perlin_noise.noise_scale')
        assert noise_min is not None, f"Must configure {base_path}.perlin_noise.noise_strength.min in captcha_config.yaml"
        assert noise_max is not None, f"Must configure {base_path}.perlin_noise.noise_strength.max in captcha_config.yaml"
        assert noise_scale is not None, f"Must configure {base_path}.perlin_noise.noise_scale in captcha_config.yaml"
        
        self.PERLIN_NOISE = {
            'noise_strength': {
                'min': noise_min,
                'max': noise_max,
                'description': '噪声强度，值越大噪声越明显'
            },
            'noise_scale': {
                'value': noise_scale,
                'description': '噪声缩放比例，影响噪声纹理的粗细'
            }
        }
        
        # ========== 旋转 (Rotation) ==========
        rot_min = self.loader.get(f'{base_path}.rotation.rotation_angle.min')
        rot_max = self.loader.get(f'{base_path}.rotation.rotation_angle.max')
        assert rot_min is not None, f"Must configure {base_path}.rotation.rotation_angle.min in captcha_config.yaml"
        assert rot_max is not None, f"Must configure {base_path}.rotation.rotation_angle.max in captcha_config.yaml"
        
        self.ROTATION = {
            'rotation_angle': {
                'min': rot_min,
                'max': rot_max,
                'description': '缺口旋转角度（度），只旋转缺口不旋转滑块'
            }
        }
        
        # ========== 高光 (Highlight) ==========
        hl_base_min = self.loader.get(f'{base_path}.highlight.base_lightness.min')
        hl_base_max = self.loader.get(f'{base_path}.highlight.base_lightness.max')
        hl_edge_min = self.loader.get(f'{base_path}.highlight.edge_lightness.min')
        hl_edge_max = self.loader.get(f'{base_path}.highlight.edge_lightness.max')
        hl_dir_min = self.loader.get(f'{base_path}.highlight.directional_lightness.min')
        hl_dir_max = self.loader.get(f'{base_path}.highlight.directional_lightness.max')
        hl_outer = self.loader.get(f'{base_path}.highlight.outer_edge_lightness')
        
        assert hl_base_min is not None, f"Must configure {base_path}.highlight.base_lightness.min in captcha_config.yaml"
        assert hl_base_max is not None, f"Must configure {base_path}.highlight.base_lightness.max in captcha_config.yaml"
        assert hl_edge_min is not None, f"Must configure {base_path}.highlight.edge_lightness.min in captcha_config.yaml"
        assert hl_edge_max is not None, f"Must configure {base_path}.highlight.edge_lightness.max in captcha_config.yaml"
        assert hl_dir_min is not None, f"Must configure {base_path}.highlight.directional_lightness.min in captcha_config.yaml"
        assert hl_dir_max is not None, f"Must configure {base_path}.highlight.directional_lightness.max in captcha_config.yaml"
        assert hl_outer is not None, f"Must configure {base_path}.highlight.outer_edge_lightness in captcha_config.yaml"
        
        self.HIGHLIGHT = {
            'base_lightness': {
                'min': hl_base_min,
                'max': hl_base_max,
                'description': '基础亮度增加值'
            },
            'edge_lightness': {
                'min': hl_edge_min,
                'max': hl_edge_max,
                'description': '边缘额外亮度'
            },
            'directional_lightness': {
                'min': hl_dir_min,
                'max': hl_dir_max,
                'description': '方向性高光强度'
            },
            'outer_edge_lightness': {
                'value': hl_outer,
                'description': '外边缘高光强度（设为0表示不使用）'
            }
        }
        
        # ========== 混淆缺口 (ConfusingGap) ==========
        cg_num = self.loader.get(f'{base_path}.confusing_gap.num_confusing_gaps')
        cg_type = self.loader.get(f'{base_path}.confusing_gap.confusing_type')
        cg_rot_min = self.loader.get(f'{base_path}.confusing_gap.rotation_range.min')
        cg_rot_max = self.loader.get(f'{base_path}.confusing_gap.rotation_range.max')
        cg_scale_min = self.loader.get(f'{base_path}.confusing_gap.scale_range.min')
        cg_scale_max = self.loader.get(f'{base_path}.confusing_gap.scale_range.max')
        
        assert cg_num is not None, f"Must configure {base_path}.confusing_gap.num_confusing_gaps in captcha_config.yaml"
        assert cg_type is not None, f"Must configure {base_path}.confusing_gap.confusing_type in captcha_config.yaml"
        assert cg_rot_min is not None, f"Must configure {base_path}.confusing_gap.rotation_range.min in captcha_config.yaml"
        assert cg_rot_max is not None, f"Must configure {base_path}.confusing_gap.rotation_range.max in captcha_config.yaml"
        assert cg_scale_min is not None, f"Must configure {base_path}.confusing_gap.scale_range.min in captcha_config.yaml"
        assert cg_scale_max is not None, f"Must configure {base_path}.confusing_gap.scale_range.max in captcha_config.yaml"
        
        self.CONFUSING_GAP = {
            'num_confusing_gaps': {
                'choices': cg_num,
                'description': '生成的混淆缺口数量'
            },
            'confusing_type': {
                'value': cg_type,
                'choices': ['same_y', 'different_y', 'mixed'],
                'description': '混淆缺口类型：same_y同行，different_y不同行，mixed混合'
            },
            'rotation_range': {
                'min': cg_rot_min,
                'max': cg_rot_max,
                'description': '混淆缺口的旋转角度范围（度）'
            },
            'scale_range': {
                'min': cg_scale_min,
                'max': cg_scale_max,
                'description': '混淆缺口的缩放比例范围'
            }
        }
        
        # ========== 空心中心 (HollowCenter) ==========
        hc_min = self.loader.get(f'{base_path}.hollow_center.scale.min')
        hc_max = self.loader.get(f'{base_path}.hollow_center.scale.max')
        assert hc_min is not None, f"Must configure {base_path}.hollow_center.scale.min in captcha_config.yaml"
        assert hc_max is not None, f"Must configure {base_path}.hollow_center.scale.max in captcha_config.yaml"
        
        self.HOLLOW_CENTER = {
            'scale': {
                'min': hc_min,
                'max': hc_max,
                'description': '空心部分占整个拼图的比例'
            }
        }
        
        # ========== 组合混淆 (Combined) ==========
        cb_num = self.loader.get(f'{base_path}.combined.num_strategies')
        cb_avail = self.loader.get(f'{base_path}.combined.available_strategies')
        assert cb_num is not None, f"Must configure {base_path}.combined.num_strategies in captcha_config.yaml"
        assert cb_avail is not None, f"Must configure {base_path}.combined.available_strategies in captcha_config.yaml"
        
        self.COMBINED = {
            'num_strategies': {
                'choices': cb_num,
                'description': '组合使用的混淆策略数量'
            },
            'available_strategies': {
                'value': cb_avail,
                'description': '可用于组合的混淆策略列表'
            }
        }
        
        # ========== 光照效果 (Lighting Effects) ==========
        lighting_path = 'captcha_config.lighting_effects'
        
        # Shadow params
        sh_base = self.loader.get(f'{lighting_path}.gap_shadow.base_darkness')
        sh_edge = self.loader.get(f'{lighting_path}.gap_shadow.edge_darkness')
        sh_dir = self.loader.get(f'{lighting_path}.gap_shadow.directional_darkness')
        assert sh_base is not None, f"Must configure {lighting_path}.gap_shadow.base_darkness in captcha_config.yaml"
        assert sh_edge is not None, f"Must configure {lighting_path}.gap_shadow.edge_darkness in captcha_config.yaml"
        assert sh_dir is not None, f"Must configure {lighting_path}.gap_shadow.directional_darkness in captcha_config.yaml"
        
        # Highlight params
        gh_base = self.loader.get(f'{lighting_path}.gap_highlight.base_lightness')
        gh_edge = self.loader.get(f'{lighting_path}.gap_highlight.edge_lightness')
        gh_dir = self.loader.get(f'{lighting_path}.gap_highlight.directional_lightness')
        gh_outer = self.loader.get(f'{lighting_path}.gap_highlight.outer_edge_lightness')
        assert gh_base is not None, f"Must configure {lighting_path}.gap_highlight.base_lightness in captcha_config.yaml"
        assert gh_edge is not None, f"Must configure {lighting_path}.gap_highlight.edge_lightness in captcha_config.yaml"
        assert gh_dir is not None, f"Must configure {lighting_path}.gap_highlight.directional_lightness in captcha_config.yaml"
        assert gh_outer is not None, f"Must configure {lighting_path}.gap_highlight.outer_edge_lightness in captcha_config.yaml"
        
        self.GAP_LIGHTING = {
            'shadow': {
                'base_darkness': {
                    'value': sh_base,
                    'description': '基础暗度（整个缺口区域的基础变暗程度）'
                },
                'edge_darkness': {
                    'value': sh_edge,
                    'description': '边缘暗度（边缘额外的变暗程度）'
                },
                'directional_darkness': {
                    'value': sh_dir,
                    'description': '方向性暗度（左上和右下的额外变暗程度）'
                }
            },
            'highlight': {
                'base_lightness': {
                    'value': gh_base,
                    'description': '基础亮度（整个缺口区域的基础变亮程度）'
                },
                'edge_lightness': {
                    'value': gh_edge,
                    'description': '边缘亮度（边羘额外的变亮程度）'
                },
                'directional_lightness': {
                    'value': gh_dir,
                    'description': '方向性亮度（右下和左上的额外变亮程度）'
                },
                'outer_edge_lightness': {
                    'value': gh_outer,
                    'description': '外边缘亮度（缺口外围的高光强度）'
                }
            }
        }
    
    def get_perlin_noise_params(self, rng):
        """获取柏林噪声参数"""
        return {
            'noise_strength': rng.uniform(
                self.PERLIN_NOISE['noise_strength']['min'],
                self.PERLIN_NOISE['noise_strength']['max']
            ),
            'noise_scale': self.PERLIN_NOISE['noise_scale']['value']
        }
    
    def get_rotation_params(self, rng):
        """获取旋转参数"""
        return {
            'rotation_angle': rng.uniform(
                self.ROTATION['rotation_angle']['min'],
                self.ROTATION['rotation_angle']['max']
            )
        }
    
    def get_highlight_params(self, rng):
        """获取高光参数"""
        return {
            'base_lightness': rng.randint(
                self.HIGHLIGHT['base_lightness']['min'],
                self.HIGHLIGHT['base_lightness']['max']
            ),
            'edge_lightness': rng.randint(
                self.HIGHLIGHT['edge_lightness']['min'],
                self.HIGHLIGHT['edge_lightness']['max']
            ),
            'directional_lightness': rng.randint(
                self.HIGHLIGHT['directional_lightness']['min'],
                self.HIGHLIGHT['directional_lightness']['max']
            ),
            'outer_edge_lightness': self.HIGHLIGHT['outer_edge_lightness']['value']
        }
    
    def get_confusing_gap_params(self, rng):
        """获取混淆缺口参数"""
        return {
            'num_confusing_gaps': int(rng.choice(self.CONFUSING_GAP['num_confusing_gaps']['choices'])),
            'confusing_type': self.CONFUSING_GAP['confusing_type']['value'],
            'rotation_range': (
                self.CONFUSING_GAP['rotation_range']['min'],
                self.CONFUSING_GAP['rotation_range']['max']
            ),
            'scale_range': (
                self.CONFUSING_GAP['scale_range']['min'],
                self.CONFUSING_GAP['scale_range']['max']
            )
        }
    
    def get_hollow_center_params(self, rng):
        """获取空心中心参数"""
        return {
            'hollow_ratio': rng.uniform(
                self.HOLLOW_CENTER['scale']['min'],
                self.HOLLOW_CENTER['scale']['max']
            )
        }
    
    def get_combined_strategies(self, rng):
        """获取组合策略的配置"""
        num_strategies = rng.choice(self.COMBINED['num_strategies']['choices'])
        available = self.COMBINED['available_strategies']['value']
        return rng.choice(available, num_strategies, replace=False)
    
    def print_config(self):
        """打印当前配置（用于调试）"""
        print("=" * 60)
        print("Confusion Strategy Configuration")
        print("=" * 60)
        
        print("\n[Perlin Noise]")
        print(f"  Noise strength: {self.PERLIN_NOISE['noise_strength']['min']} ~ {self.PERLIN_NOISE['noise_strength']['max']}")
        print(f"  Noise scale: {self.PERLIN_NOISE['noise_scale']['value']}")
        
        print("\n[Rotation]")
        print(f"  Rotation angle: {self.ROTATION['rotation_angle']['min']} ~ {self.ROTATION['rotation_angle']['max']} degrees")
        
        print("\n[Highlight]")
        print(f"  Base lightness: {self.HIGHLIGHT['base_lightness']['min']} ~ {self.HIGHLIGHT['base_lightness']['max']}")
        print(f"  Edge lightness: {self.HIGHLIGHT['edge_lightness']['min']} ~ {self.HIGHLIGHT['edge_lightness']['max']}")
        print(f"  Directional lightness: {self.HIGHLIGHT['directional_lightness']['min']} ~ {self.HIGHLIGHT['directional_lightness']['max']}")
        
        print("\n[Confusing Gap]")
        print(f"  Gap count: {self.CONFUSING_GAP['num_confusing_gaps']['choices']}")
        print(f"  Gap type: {self.CONFUSING_GAP['confusing_type']['value']}")
        print(f"  Rotation range: {self.CONFUSING_GAP['rotation_range']['min']} ~ {self.CONFUSING_GAP['rotation_range']['max']} degrees")
        print(f"  Scale range: {self.CONFUSING_GAP['scale_range']['min']} ~ {self.CONFUSING_GAP['scale_range']['max']}")
        
        print("\n[Hollow Center]")
        print(f"  Hollow ratio: {self.HOLLOW_CENTER['scale']['min']} ~ {self.HOLLOW_CENTER['scale']['max']}")
        
        print("\n[Combined]")
        print(f"  Strategy count: {self.COMBINED['num_strategies']['choices']}")
        print(f"  Available strategies: {', '.join(self.COMBINED['available_strategies']['value'])}")
        
        print("=" * 60)


# 创建全局实例
confusion_config = None

def get_confusion_config() -> ConfusionConfig:
    """获取混淆配置单例"""
    global confusion_config
    if confusion_config is None:
        confusion_config = ConfusionConfig()
    return confusion_config


if __name__ == "__main__":
    # Test configuration
    config = get_confusion_config()
    config.print_config()