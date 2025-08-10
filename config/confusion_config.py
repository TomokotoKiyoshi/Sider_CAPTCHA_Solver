# -*- coding: utf-8 -*-
"""
混淆策略配置文件
所有混淆效果的参数都在这里集中管理
"""

class ConfusionConfig:
    """混淆策略参数配置"""
    
    # ========== 柏林噪声 (PerlinNoise) ==========
    PERLIN_NOISE = {
        'noise_strength': {
            'min': 0.2,
            'max': 0.5,
            'description': '噪声强度，值越大噪声越明显'
        },
        'noise_scale': {
            'value': 0.1,
            'description': '噪声缩放比例，影响噪声纹理的粗细'
        }
    }
    
    # ========== 旋转 (Rotation) ==========
    ROTATION = {
        'rotation_angle': {
            'min': 0.5,
            'max': 1.5,
            'description': '缺口旋转角度（度），只旋转缺口不旋转滑块'
        }
    }
    
    # ========== 高光 (Highlight) ==========
    HIGHLIGHT = {
        'base_lightness': {
            'min': 20,
            'max': 40,
            'description': '基础亮度增加值'
        },
        'edge_lightness': {
            'min': 40,
            'max': 60,
            'description': '边缘额外亮度'
        },
        'directional_lightness': {
            'min': 15,
            'max': 30,
            'description': '方向性高光强度'
        },
        'outer_edge_lightness': {
            'value': 0,
            'description': '外边缘高光强度（设为0表示不使用）'
        }
    }
    
    # ========== 混淆缺口 (ConfusingGap) ==========
    CONFUSING_GAP = {
        'num_confusing_gaps': {
            'choices': [1, 2],
            'description': '生成的混淆缺口数量'
        },
        'confusing_type': {
            'value': 'mixed',
            'choices': ['same_y', 'different_y', 'mixed'],
            'description': '混淆缺口类型：same_y同行，different_y不同行，mixed混合'
        },
        'rotation_range': {
            'min': 10,
            'max': 30,
            'description': '混淆缺口的旋转角度范围（度）'
        },
        'scale_range': {
            'min': 0.8,
            'max': 1.2,
            'description': '混淆缺口的缩放比例范围'
        }
    }
    
    # ========== 空心中心 (HollowCenter) ==========
    HOLLOW_CENTER = {
        'scale': {
            'min': 0.3,
            'max': 0.4,
            'description': '空心部分占整个拼图的比例'
        }
    }
    
    # ========== 组合混淆 (Combined) ==========
    COMBINED = {
        'num_strategies': {
            'choices': [2, 3],
            'description': '组合使用的混淆策略数量'
        },
        'available_strategies': {
            'value': ['perlin_noise', 'rotation', 'highlight', 'hollow_center', 'confusing_gap'],
            'description': '可用于组合的混淆策略列表'
        }
    }
    
    # ========== 光照效果 (Lighting Effects) ==========
    GAP_LIGHTING = {
        'shadow': {
            'base_darkness': {
                'value': 40,
                'description': '基础暗度（整个缺口区域的基础变暗程度）'
            },
            'edge_darkness': {
                'value': 50,
                'description': '边缘暗度（边缘额外的变暗程度）'
            },
            'directional_darkness': {
                'value': 20,
                'description': '方向性暗度（左上和右下的额外变暗程度）'
            }
        },
        'highlight': {
            'base_lightness': {
                'value': 30,
                'description': '基础亮度（整个缺口区域的基础变亮程度）'
            },
            'edge_lightness': {
                'value': 45,
                'description': '边缘亮度（边缘额外的变亮程度）'
            },
            'directional_lightness': {
                'value': 20,
                'description': '方向性亮度（右下和左上的额外变亮程度）'
            },
            'outer_edge_lightness': {
                'value': 0,
                'description': '外边缘亮度（缺口外围的高光强度）'
            }
        }
    }
    
    @classmethod
    def get_perlin_noise_params(cls, rng):
        """获取柏林噪声参数"""
        return {
            'noise_strength': rng.uniform(
                cls.PERLIN_NOISE['noise_strength']['min'],
                cls.PERLIN_NOISE['noise_strength']['max']
            ),
            'noise_scale': cls.PERLIN_NOISE['noise_scale']['value']
        }
    
    @classmethod
    def get_rotation_params(cls, rng):
        """获取旋转参数"""
        return {
            'rotation_angle': rng.uniform(
                cls.ROTATION['rotation_angle']['min'],
                cls.ROTATION['rotation_angle']['max']
            )
        }
    
    @classmethod
    def get_highlight_params(cls, rng):
        """获取高光参数"""
        return {
            'base_lightness': rng.randint(
                cls.HIGHLIGHT['base_lightness']['min'],
                cls.HIGHLIGHT['base_lightness']['max']
            ),
            'edge_lightness': rng.randint(
                cls.HIGHLIGHT['edge_lightness']['min'],
                cls.HIGHLIGHT['edge_lightness']['max']
            ),
            'directional_lightness': rng.randint(
                cls.HIGHLIGHT['directional_lightness']['min'],
                cls.HIGHLIGHT['directional_lightness']['max']
            ),
            'outer_edge_lightness': cls.HIGHLIGHT['outer_edge_lightness']['value']
        }
    
    @classmethod
    def get_confusing_gap_params(cls, rng):
        """获取混淆缺口参数"""
        return {
            'num_confusing_gaps': int(rng.choice(cls.CONFUSING_GAP['num_confusing_gaps']['choices'])),
            'confusing_type': cls.CONFUSING_GAP['confusing_type']['value'],
            'rotation_range': (
                cls.CONFUSING_GAP['rotation_range']['min'],
                cls.CONFUSING_GAP['rotation_range']['max']
            ),
            'scale_range': (
                cls.CONFUSING_GAP['scale_range']['min'],
                cls.CONFUSING_GAP['scale_range']['max']
            )
        }
    
    @classmethod
    def get_hollow_center_params(cls, rng):
        """获取空心中心参数"""
        return {
            'hollow_ratio': rng.uniform(
                cls.HOLLOW_CENTER['scale']['min'],
                cls.HOLLOW_CENTER['scale']['max']
            )
        }
    
    @classmethod
    def get_combined_strategies(cls, rng):
        """获取组合策略的配置"""
        num_strategies = rng.choice(cls.COMBINED['num_strategies']['choices'])
        available = cls.COMBINED['available_strategies']['value']
        return rng.choice(available, num_strategies, replace=False)
    
    @classmethod
    def print_config(cls):
        """打印当前配置（用于调试）"""
        print("=" * 60)
        print("混淆策略配置参数")
        print("=" * 60)
        
        print("\n[柏林噪声 PerlinNoise]")
        print(f"  噪声强度: {cls.PERLIN_NOISE['noise_strength']['min']} ~ {cls.PERLIN_NOISE['noise_strength']['max']}")
        print(f"  噪声缩放: {cls.PERLIN_NOISE['noise_scale']['value']}")
        
        print("\n[旋转 Rotation]")
        print(f"  旋转角度: {cls.ROTATION['rotation_angle']['min']} ~ {cls.ROTATION['rotation_angle']['max']}度")
        
        print("\n[高光 Highlight]")
        print(f"  基础亮度: {cls.HIGHLIGHT['base_lightness']['min']} ~ {cls.HIGHLIGHT['base_lightness']['max']}")
        print(f"  边缘亮度: {cls.HIGHLIGHT['edge_lightness']['min']} ~ {cls.HIGHLIGHT['edge_lightness']['max']}")
        print(f"  方向亮度: {cls.HIGHLIGHT['directional_lightness']['min']} ~ {cls.HIGHLIGHT['directional_lightness']['max']}")
        
        print("\n[混淆缺口 ConfusingGap]")
        print(f"  缺口数量: {cls.CONFUSING_GAP['num_confusing_gaps']['choices']}")
        print(f"  缺口类型: {cls.CONFUSING_GAP['confusing_type']['value']}")
        print(f"  旋转范围: {cls.CONFUSING_GAP['rotation_range']['min']} ~ {cls.CONFUSING_GAP['rotation_range']['max']}度")
        print(f"  缩放范围: {cls.CONFUSING_GAP['scale_range']['min']} ~ {cls.CONFUSING_GAP['scale_range']['max']}")
        
        print("\n[空心中心 HollowCenter]")
        print(f"  空心比例: {cls.HOLLOW_CENTER['scale']['min']} ~ {cls.HOLLOW_CENTER['scale']['max']}")
        
        print("\n[组合混淆 Combined]")
        print(f"  组合数量: {cls.COMBINED['num_strategies']['choices']}")
        print(f"  可用策略: {', '.join(cls.COMBINED['available_strategies']['value'])}")
        
        print("=" * 60)


if __name__ == "__main__":
    # 测试配置
    ConfusionConfig.print_config()