# -*- coding: utf-8 -*-
"""
批量生成验证码数据集
"""
import cv2
import numpy as np
import random
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict
import json
from tqdm import tqdm
import itertools

from .simple_puzzle_generator import (
    create_puzzle_piece, get_all_puzzle_combinations,
    generate_special_shape
)
from .lighting_effects import apply_gap_lighting, add_puzzle_piece_shadow


class CaptchaGenerator:
    """验证码批量生成器"""
    
    def __init__(self, config_path: str = None):
        """初始化生成器"""
        # 所有81种普通拼图形状
        self.normal_shapes = get_all_puzzle_combinations()
        
        # 6种特殊形状
        self.special_shapes = ['circle', 'square', 'triangle', 'hexagon', 'pentagon', 'star']
        
        # 拼图大小配置（3种）
        self.puzzle_sizes = [
            {'size': 50, 'knob': 10},  # 小
            {'size': 60, 'knob': 12},  # 中
            {'size': 70, 'knob': 14},  # 大
        ]
        
        # 滑块宽度
        self.slider_width = 60
        
        # 目标图片尺寸
        self.target_width = 320
        self.target_height = 160
        
        # 加载配置
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """加载配置文件"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'captcha_generation' in config:
            gen_config = config['captcha_generation']
            if 'puzzle_sizes' in gen_config:
                self.puzzle_sizes = gen_config['puzzle_sizes']
            if 'target_size' in gen_config:
                self.target_width = gen_config['target_size']['width']
                self.target_height = gen_config['target_size']['height']
    
    def generate_captcha(self, image_path: Path, puzzle_config: Dict) -> Dict:
        """
        生成单个验证码
        
        Args:
            image_path: 原始图片路径
            puzzle_config: 拼图配置
                - shape_type: 'normal' 或 'special'
                - shape: 形状参数
                - size_config: 大小配置
                - position: (x, y) 位置
        
        Returns:
            验证码信息字典
        """
        # 读取并调整图片大小
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        img = cv2.resize(img, (self.target_width, self.target_height))
        
        # 获取拼图参数
        size_config = puzzle_config['size_config']
        puzzle_size = size_config['size']
        knob_radius = size_config['knob']
        
        # 生成拼图掩码
        if puzzle_config['shape_type'] == 'normal':
            puzzle_mask = create_puzzle_piece(
                piece_size=puzzle_size,
                knob_radius=knob_radius,
                edges=puzzle_config['shape']
            )
        else:
            # 特殊形状
            puzzle_mask = generate_special_shape(
                puzzle_config['shape'],
                size=puzzle_size
            )
        
        # 获取拼图位置
        x, y = puzzle_config['position']
        mask_h, mask_w = puzzle_mask.shape[:2]
        
        # 提取拼图块
        puzzle = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
        img_region = img[y:y+mask_h, x:x+mask_w]
        
        # 应用掩码
        alpha = puzzle_mask[:, :, 3]
        for c in range(3):  # BGR通道
            puzzle[:, :, c] = img_region[:, :, c]
        puzzle[:, :, 3] = alpha
        
        # 创建背景并应用光照效果
        background = apply_gap_lighting(img, x, y, alpha, mask_h, mask_w)
        
        # 为拼图块添加阴影效果
        puzzle = add_puzzle_piece_shadow(puzzle, shadow_offset=(3, 3), shadow_opacity=0.4)
        
        # 计算中心坐标
        bg_center_x = x + mask_w // 2
        bg_center_y = y + mask_h // 2
        
        # 滑块默认在左侧
        slider_x = self.slider_width // 2
        slider_y = bg_center_y
        
        return {
            'background': background,
            'puzzle': puzzle,
            'bg_center': (bg_center_x, bg_center_y),
            'slider_center': (slider_x, slider_y),
            'puzzle_size': puzzle_size,
            'shape_info': puzzle_config
        }
    
    def generate_positions(self, puzzle_size: int, count: int = 4) -> List[Tuple[int, int]]:
        """
        生成不重叠的拼图位置
        
        Args:
            puzzle_size: 拼图大小
            count: 生成位置数量
        
        Returns:
            位置列表 [(x, y), ...]
        """
        positions = []
        attempts = 0
        max_attempts = 1000
        
        # 拼图边界（考虑凹凸）
        padding = puzzle_size // 4
        min_x = self.slider_width + 10 + padding
        max_x = self.target_width - puzzle_size - padding
        min_y = padding
        max_y = self.target_height - puzzle_size - padding
        
        while len(positions) < count and attempts < max_attempts:
            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)
            
            # 检查是否与已有位置重叠
            valid = True
            for px, py in positions:
                if abs(x - px) < puzzle_size and abs(y - py) < puzzle_size:
                    valid = False
                    break
            
            if valid:
                positions.append((x, y))
            
            attempts += 1
        
        if len(positions) < count:
            print(f"Warning: Only generated {len(positions)} positions out of {count}")
        
        return positions
    
    def generate_batch(self, input_dir: Path, output_dir: Path, 
                      images_per_category: int = 200):
        """
        批量生成验证码数据集
        
        Args:
            input_dir: 输入图片目录
            output_dir: 输出目录
            images_per_category: 每个类别生成的图片数量
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建输出子目录
        backgrounds_dir = output_dir / 'backgrounds'
        puzzles_dir = output_dir / 'puzzles'
        backgrounds_dir.mkdir(exist_ok=True)
        puzzles_dir.mkdir(exist_ok=True)
        
        # 统计信息
        stats = {
            'total_generated': 0,
            'categories': {},
            'shape_usage': {shape: 0 for shape in self.special_shapes},
            'normal_shape_usage': {i: 0 for i in range(81)}
        }
        
        # 为每个类别生成验证码
        for category_dir in input_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            print(f"\nProcessing category: {category_name}")
            
            # 获取该类别的所有图片
            image_files = list(category_dir.glob("*.png"))
            if not image_files:
                print(f"No images found in {category_name}")
                continue
            
            category_stats = {
                'count': 0,
                'shapes_used': set()
            }
            
            # 为该类别生成指定数量的验证码
            for img_idx in tqdm(range(images_per_category), desc=category_name):
                # 随机选择一张原图
                source_image = random.choice(image_files)
                
                # 随机选择拼图配置
                # 5个普通形状 + 6个特殊形状 = 11个形状
                puzzle_configs = []
                
                # 随机选择5个不同的普通形状
                selected_normal = random.sample(range(81), 5)
                for idx in selected_normal:
                    shape = self.normal_shapes[idx]
                    size_config = random.choice(self.puzzle_sizes)
                    puzzle_configs.append({
                        'shape_type': 'normal',
                        'shape': shape,
                        'shape_index': idx,
                        'size_config': size_config
                    })
                    stats['normal_shape_usage'][idx] += 1
                
                # 添加所有6个特殊形状
                for special_shape in self.special_shapes:
                    size_config = random.choice(self.puzzle_sizes)
                    puzzle_configs.append({
                        'shape_type': 'special',
                        'shape': special_shape,
                        'size_config': size_config
                    })
                    stats['shape_usage'][special_shape] += 1
                
                # 为每个形状生成不同位置的验证码
                for shape_idx, puzzle_config in enumerate(puzzle_configs):
                    # 生成4个不重叠的位置
                    positions = self.generate_positions(
                        puzzle_config['size_config']['size'], 
                        count=4
                    )
                    
                    for pos_idx, position in enumerate(positions):
                        puzzle_config['position'] = position
                        
                        try:
                            # 生成验证码
                            result = self.generate_captcha(source_image, puzzle_config)
                            
                            # 生成文件名
                            pic_num = stats['total_generated'] + 1
                            bg_x, bg_y = result['bg_center']
                            sd_x, sd_y = result['slider_center']
                            
                            # 生成哈希值（基于图片内容）
                            hash_str = f"{source_image.name}_{shape_idx}_{pos_idx}"
                            hash_value = hashlib.md5(hash_str.encode()).hexdigest()[:8]
                            
                            # 文件名格式
                            filename = f"Pic{pic_num:04d}_Bgx{bg_x}Bgy{bg_y}_Sdx{sd_x}Sdy{sd_y}_{hash_value}.png"
                            
                            # 保存背景图
                            bg_path = backgrounds_dir / filename
                            cv2.imwrite(str(bg_path), result['background'])
                            
                            # 保存拼图块
                            puzzle_path = puzzles_dir / filename
                            cv2.imwrite(str(puzzle_path), result['puzzle'])
                            
                            # 更新统计
                            stats['total_generated'] += 1
                            category_stats['count'] += 1
                            
                            # 记录形状使用情况
                            if puzzle_config['shape_type'] == 'normal':
                                category_stats['shapes_used'].add(f"normal_{puzzle_config['shape_index']}")
                            else:
                                category_stats['shapes_used'].add(puzzle_config['shape'])
                            
                        except Exception as e:
                            print(f"Error generating captcha: {e}")
                            continue
            
            stats['categories'][category_name] = category_stats
        
        # 保存统计信息
        stats_path = output_dir / 'generation_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            # 转换set为list以便JSON序列化
            for cat_stats in stats['categories'].values():
                cat_stats['shapes_used'] = list(cat_stats['shapes_used'])
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nGeneration complete!")
        print(f"Total captchas generated: {stats['total_generated']}")
        print(f"Stats saved to: {stats_path}")
        
        # 验证81种普通形状都被使用
        unused_shapes = [i for i, count in stats['normal_shape_usage'].items() if count == 0]
        if unused_shapes:
            print(f"Warning: {len(unused_shapes)} normal shapes were not used: {unused_shapes}")
        else:
            print("All 81 normal shapes were used!")
        
        return stats


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent.parent
    
    # 配置路径
    config_path = project_root / "configs" / "config.yaml"
    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "captcha_dataset"
    
    # 检查输入目录
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print("Please run the image download script first.")
        return
    
    # 创建生成器
    generator = CaptchaGenerator(str(config_path))
    
    # 生成验证码数据集
    # 每个类别200张图，每张图11个形状，每个形状4个位置 = 每类别8800个验证码
    generator.generate_batch(input_dir, output_dir, images_per_category=200)


if __name__ == "__main__":
    main()