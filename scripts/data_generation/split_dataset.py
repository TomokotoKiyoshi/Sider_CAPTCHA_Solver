#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本 - 优化版
生成JSON索引文件，不复制图片，支持train/val/test三分划分
"""
import sys
import json
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigLoader


class DatasetSplitter:
    """数据集划分器"""
    
    def __init__(self):
        """初始化"""
        self.config_loader = ConfigLoader()
        self._load_config()
        
    def _load_config(self):
        """加载配置"""
        # 从data_split.yaml加载配置
        train_ratio = self.config_loader.get('data_split.split_ratio.train')
        val_ratio = self.config_loader.get('data_split.split_ratio.val')
        test_ratio = self.config_loader.get('data_split.split_ratio.test')
        assert train_ratio is not None, "Must configure split_ratio.train in data_split.yaml"
        assert val_ratio is not None, "Must configure split_ratio.val in data_split.yaml"
        assert test_ratio is not None, "Must configure split_ratio.test in data_split.yaml"
        
        self.split_ratio = {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        }
        
        # 验证比例和为1
        ratio_sum = sum(self.split_ratio.values())
        if abs(ratio_sum - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
        
        # 路径配置
        input_dir = self.config_loader.get('data_split.paths.input_dir')
        output_dir = self.config_loader.get('data_split.paths.output_dir')
        metadata_dir = self.config_loader.get('data_split.paths.metadata_dir')
        assert input_dir is not None, "Must configure paths.input_dir in data_split.yaml"
        assert output_dir is not None, "Must configure paths.output_dir in data_split.yaml"
        assert metadata_dir is not None, "Must configure paths.metadata_dir in data_split.yaml"
        
        self.input_dir = Path(project_root) / input_dir
        self.output_dir = Path(project_root) / output_dir
        self.metadata_dir = Path(project_root) / metadata_dir
        
        # 选项配置
        random_seed = self.config_loader.get('data_split.options.random_seed')
        stratify = self.config_loader.get('data_split.options.stratify_by_image')
        shuffle = self.config_loader.get('data_split.options.shuffle')
        assert random_seed is not None, "Must configure options.random_seed in data_split.yaml"
        assert stratify is not None, "Must configure options.stratify_by_image in data_split.yaml"
        assert shuffle is not None, "Must configure options.shuffle in data_split.yaml"
        
        self.random_seed = random_seed
        self.stratify_by_image = stratify
        self.shuffle = shuffle
        
    def load_annotations(self) -> List[Dict]:
        """加载标注文件"""
        annotations_file = self.metadata_dir / 'all_annotations.json'
        
        if not annotations_file.exists():
            # 尝试从generated目录下的metadata子目录查找
            alt_annotations_file = self.input_dir / 'metadata' / 'all_annotations.json'
            if alt_annotations_file.exists():
                annotations_file = alt_annotations_file
            else:
                raise FileNotFoundError(f"Annotations file not found at {annotations_file} or {alt_annotations_file}")
        
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
            
        print(f"Loaded {len(annotations)} annotations from {annotations_file}")
        return annotations
    
    def group_by_image(self, annotations: List[Dict]) -> Dict[str, List[Dict]]:
        """按图片ID分组"""
        image_groups = defaultdict(list)
        
        for ann in annotations:
            # 提取图片ID (如 Pic0001)
            filename = ann.get('filename', '')
            if filename:
                pic_id = filename.split('_')[0]
                image_groups[pic_id].append(ann)
        
        print(f"Grouped annotations into {len(image_groups)} unique images")
        return dict(image_groups)
    
    def split_image_ids(self, image_groups: Dict[str, List[Dict]]) -> Tuple[List[str], List[str], List[str]]:
        """划分图片ID到train/val/test"""
        unique_pic_ids = list(image_groups.keys())
        
        # 设置随机种子
        random.seed(self.random_seed)
        
        # 打乱顺序
        if self.shuffle:
            random.shuffle(unique_pic_ids)
        
        # 计算分割点
        n_pics = len(unique_pic_ids)
        train_end = int(n_pics * self.split_ratio['train'])
        val_end = train_end + int(n_pics * self.split_ratio['val'])
        
        # 划分
        train_ids = unique_pic_ids[:train_end]
        val_ids = unique_pic_ids[train_end:val_end]
        test_ids = unique_pic_ids[val_end:]
        
        return train_ids, val_ids, test_ids
    
    def create_dataset_json(self, 
                          dataset_type: str,
                          image_ids: List[str],
                          image_groups: Dict[str, List[Dict]]) -> Dict:
        """创建数据集JSON内容"""
        samples = []
        
        for pic_id in image_ids:
            for ann in image_groups[pic_id]:
                # 构建样本信息
                # 从bg_center和sd_center提取坐标
                bg_center = ann.get('bg_center', [0, 0])
                sd_center = ann.get('sd_center', [0, 0])
                
                # 解析confusion_type和confusion_details
                confusion_type = ann.get('confusion_type', 'none')
                confusion_details_str = ann.get('confusion_details', '')
                
                # 根据confusion_type推断策略
                confusion_strategies = {
                    'confusion_type': confusion_type,
                    'confusion_details': confusion_details_str,
                    'rotated_gap': 'rotated_gap' in confusion_type,
                    'perlin_noise': 'perlin_noise' in confusion_type,
                    'confusing_gap': 'confusing_gap' in confusion_type,
                    'gap_highlight': 'gap_highlight' in confusion_type
                }
                
                sample = {
                    'filename': ann['filename'],
                    'relative_path': f"../generated/{ann['filename']}",
                    'image_info': {
                        'width': ann.get('image_width', 320),
                        'height': ann.get('image_height', 160),
                        'original_pic_id': pic_id
                    },
                    'annotations': {
                        'gap_x': bg_center[0] if isinstance(bg_center, list) else bg_center,
                        'gap_y': bg_center[1] if isinstance(bg_center, list) else 0,
                        'slider_x': sd_center[0] if isinstance(sd_center, list) else sd_center,
                        'slider_y': sd_center[1] if isinstance(sd_center, list) else 0,
                        'puzzle_size': ann.get('size', 40),
                        'shape': ann.get('shape', 'unknown'),
                        'confusion_strategies': confusion_strategies
                    },
                    'metadata': {
                        'background_hash': ann.get('background_hash', ''),
                        'hash': ann.get('hash', ''),
                        'aspect_ratio': ann.get('aspect_ratio', 1.0),
                        'generation_metadata': ann.get('metadata', {})
                    }
                }
                samples.append(sample)
        
        # 构建完整的数据集JSON
        dataset_json = {
            'dataset_type': dataset_type,
            'num_samples': len(samples),
            'num_unique_images': len(image_ids),
            'samples': samples
        }
        
        return dataset_json
    
    def create_split_info(self,
                         train_ids: List[str],
                         val_ids: List[str],
                         test_ids: List[str],
                         image_groups: Dict[str, List[Dict]]) -> Dict:
        """创建划分信息JSON"""
        # 计算统计信息
        train_samples = sum(len(image_groups[pid]) for pid in train_ids)
        val_samples = sum(len(image_groups[pid]) for pid in val_ids)
        test_samples = sum(len(image_groups[pid]) for pid in test_ids)
        total_samples = train_samples + val_samples + test_samples
        
        split_info = {
            'split_configuration': {
                'ratio': self.split_ratio,
                'random_seed': self.random_seed,
                'stratify_by_image': self.stratify_by_image
            },
            'statistics': {
                'total': {
                    'samples': total_samples,
                    'unique_images': len(image_groups)
                },
                'train': {
                    'samples': train_samples,
                    'unique_images': len(train_ids),
                    'percentage': round(train_samples / total_samples * 100, 2)
                },
                'val': {
                    'samples': val_samples,
                    'unique_images': len(val_ids),
                    'percentage': round(val_samples / total_samples * 100, 2)
                },
                'test': {
                    'samples': test_samples,
                    'unique_images': len(test_ids),
                    'percentage': round(test_samples / total_samples * 100, 2)
                }
            },
            'image_distribution': {
                'train': sorted(train_ids),
                'val': sorted(val_ids),
                'test': sorted(test_ids)
            },
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'script_version': '2.0.0',
                'source_directory': str(self.input_dir.relative_to(project_root)),
                'metadata_source': str(self.metadata_dir.relative_to(project_root))
            }
        }
        
        return split_info
    
    def save_json(self, data: Dict, filename: str):
        """保存JSON文件"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {filename} ({len(data.get('samples', []))} samples)")
    
    def run(self):
        """执行数据集划分"""
        print("=" * 60)
        print("CAPTCHA Dataset Splitter v2.0")
        print("=" * 60)
        
        # 打印配置信息
        print(f"\nConfiguration:")
        print(f"  Split ratio: train={self.split_ratio['train']:.1%}, "
              f"val={self.split_ratio['val']:.1%}, test={self.split_ratio['test']:.1%}")
        print(f"  Input dir: {self.input_dir}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Random seed: {self.random_seed}")
        print(f"  Stratify by image: {self.stratify_by_image}")
        print("-" * 60)
        
        # 1. 加载标注
        try:
            annotations = self.load_annotations()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run generate_captchas_with_components.py first")
            return 1
        
        # 2. 按图片ID分组
        image_groups = self.group_by_image(annotations)
        
        # 3. 划分图片ID
        train_ids, val_ids, test_ids = self.split_image_ids(image_groups)
        
        # 4. 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated output directory: {self.output_dir}")
        
        # 5. 生成数据集JSON文件
        print("\nGenerating dataset JSON files...")
        
        # Train set
        train_json = self.create_dataset_json('train', train_ids, image_groups)
        self.save_json(train_json, 'train.json')
        
        # Validation set
        val_json = self.create_dataset_json('val', val_ids, image_groups)
        self.save_json(val_json, 'val.json')
        
        # Test set
        test_json = self.create_dataset_json('test', test_ids, image_groups)
        self.save_json(test_json, 'test.json')
        
        # 6. 生成划分信息文件
        print("\nGenerating split info...")
        split_info = self.create_split_info(train_ids, val_ids, test_ids, image_groups)
        self.save_json(split_info, 'split_info.json')
        
        # 7. 打印统计信息
        print("\n" + "=" * 60)
        print("Dataset Split Results:")
        print("-" * 60)
        print(f"Total: {split_info['statistics']['total']['unique_images']:4d} images, "
              f"{split_info['statistics']['total']['samples']:6d} samples")
        print(f"Train: {split_info['statistics']['train']['unique_images']:4d} images, "
              f"{split_info['statistics']['train']['samples']:6d} samples "
              f"({split_info['statistics']['train']['percentage']:.1f}%)")
        print(f"Val:   {split_info['statistics']['val']['unique_images']:4d} images, "
              f"{split_info['statistics']['val']['samples']:6d} samples "
              f"({split_info['statistics']['val']['percentage']:.1f}%)")
        print(f"Test:  {split_info['statistics']['test']['unique_images']:4d} images, "
              f"{split_info['statistics']['test']['samples']:6d} samples "
              f"({split_info['statistics']['test']['percentage']:.1f}%)")
        print("=" * 60)
        print("Dataset splitting completed successfully!")
        print(f"JSON files saved to: {self.output_dir}")
        print("=" * 60)
        
        return 0


def main():
    """主函数"""
    splitter = DatasetSplitter()
    return splitter.run()


if __name__ == "__main__":
    sys.exit(main())