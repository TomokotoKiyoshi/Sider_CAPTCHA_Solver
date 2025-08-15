#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集划分模块
基于图片ID (pic_id) 进行划分，确保同一张原始图片的所有验证码变体都在同一个数据集中
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


class DatasetSplitter:
    """
    数据集划分器
    确保同一张原始图片的所有验证码都在同一个数据集中
    """
    
    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, 
                 seed: int = 42, shuffle: bool = True):
        """
        初始化数据集划分器
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
            seed: 随机种子
            shuffle: 是否在划分前打乱数据
        """
        # 验证比例和为1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"比例之和必须为1，当前: {train_ratio + val_ratio + test_ratio}"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.shuffle = shuffle
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
    
    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'DatasetSplitter':
        """
        从配置文件创建DatasetSplitter实例
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
            
        Returns:
            DatasetSplitter实例
        """
        # 导入配置加载器
        from .config_loader import get_data_split_config
        
        # 加载配置
        config = get_data_split_config(config_path)
        
        # 提取划分比例
        split_ratio = config.get('split_ratio', {})
        train_ratio = split_ratio.get('train', 0.8)
        val_ratio = split_ratio.get('val', 0.1)
        test_ratio = split_ratio.get('test', 0.1)
        
        # 提取选项
        options = config.get('options', {})
        seed = options.get('random_seed', 42)
        shuffle = options.get('shuffle', True)  # 从配置文件读取shuffle选项，默认为True
        
        # 创建实例
        return cls(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            shuffle=shuffle
        )
    
    def split_by_pic_id(self, labels: List[Dict]) -> Dict[str, List[Dict]]:
        """
        基于pic_id划分数据集
        
        Args:
            labels: 标签列表，每个标签包含pic_id字段
            
        Returns:
            包含train/val/test三个列表的字典
        """
        # 按pic_id分组
        grouped_by_pic = defaultdict(list)
        for label in labels:
            pic_id = label.get('pic_id', None)
            if pic_id is None:
                # 如果没有pic_id，尝试从sample_id提取
                sample_id = label.get('sample_id', '')
                # sample_id格式: Pic0021_Bgx94Bgy34_Sdx24Sdy34_3b3f3fb5
                if sample_id.startswith('Pic'):
                    pic_id = sample_id.split('_')[0]  # 提取Pic0021
                else:
                    print(f"警告: 无法提取pic_id from {sample_id}")
                    continue
            grouped_by_pic[pic_id].append(label)
        
        # 获取所有不同的pic_id
        unique_pic_ids = list(grouped_by_pic.keys())
        print(f"总共有 {len(unique_pic_ids)} 张不同的原始图片")
        
        # 统计每张图片的验证码数量
        samples_per_pic = {pic_id: len(samples) for pic_id, samples in grouped_by_pic.items()}
        print(f"每张图片的验证码数量: 最小={min(samples_per_pic.values())}, "
              f"最大={max(samples_per_pic.values())}, "
              f"平均={np.mean(list(samples_per_pic.values())):.1f}")
        
        # 根据配置决定是否打乱pic_id顺序
        if self.shuffle:
            random.shuffle(unique_pic_ids)
            print(f"✅ 数据已打乱 (shuffle=True, seed={self.seed})")
        else:
            print(f"ℹ️ 数据未打乱 (shuffle=False)")
        
        # 计算各数据集的pic_id数量
        n_total = len(unique_pic_ids)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        # n_test使用剩余的，避免舍入误差
        
        # 划分pic_id
        train_pic_ids = unique_pic_ids[:n_train]
        val_pic_ids = unique_pic_ids[n_train:n_train + n_val]
        test_pic_ids = unique_pic_ids[n_train + n_val:]
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_pic_ids)} 张图片")
        print(f"  验证集: {len(val_pic_ids)} 张图片")
        print(f"  测试集: {len(test_pic_ids)} 张图片")
        
        # 收集各数据集的所有样本
        train_samples = []
        val_samples = []
        test_samples = []
        
        for pic_id in train_pic_ids:
            train_samples.extend(grouped_by_pic[pic_id])
        for pic_id in val_pic_ids:
            val_samples.extend(grouped_by_pic[pic_id])
        for pic_id in test_pic_ids:
            test_samples.extend(grouped_by_pic[pic_id])
        
        # 根据配置决定是否打乱样本顺序（但保持同一张图片的样本在同一个数据集）
        if self.shuffle:
            random.shuffle(train_samples)
            random.shuffle(val_samples)
            random.shuffle(test_samples)
        
        print(f"\n样本数量:")
        print(f"  训练集: {len(train_samples)} 个样本")
        print(f"  验证集: {len(val_samples)} 个样本")
        print(f"  测试集: {len(test_samples)} 个样本")
        print(f"  总计: {len(train_samples) + len(val_samples) + len(test_samples)} 个样本")
        
        # 验证没有重复
        self._validate_split(train_samples, val_samples, test_samples)
        
        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
    
    def _validate_split(self, train: List[Dict], val: List[Dict], test: List[Dict]):
        """
        验证数据集划分的正确性
        """
        # 提取pic_id
        def get_pic_ids(samples):
            pic_ids = set()
            for sample in samples:
                pic_id = sample.get('pic_id', None)
                if pic_id is None:
                    sample_id = sample.get('sample_id', '')
                    if sample_id.startswith('Pic'):
                        pic_id = sample_id.split('_')[0]
                if pic_id:
                    pic_ids.add(pic_id)
            return pic_ids
        
        train_pics = get_pic_ids(train)
        val_pics = get_pic_ids(val)
        test_pics = get_pic_ids(test)
        
        # 检查pic_id是否有重叠
        train_val_overlap = train_pics & val_pics
        train_test_overlap = train_pics & test_pics
        val_test_overlap = val_pics & test_pics
        
        if train_val_overlap:
            print(f"警告: 训练集和验证集有重叠的pic_id: {train_val_overlap}")
        if train_test_overlap:
            print(f"警告: 训练集和测试集有重叠的pic_id: {train_test_overlap}")
        if val_test_overlap:
            print(f"警告: 验证集和测试集有重叠的pic_id: {val_test_overlap}")
        
        if not (train_val_overlap or train_test_overlap or val_test_overlap):
            print("\n✅ 数据集划分验证通过: 没有pic_id重叠")
    
    def save_split_info(self, split_result: Dict[str, List[Dict]], output_dir: str):
        """
        保存划分信息到文件
        
        Args:
            split_result: 划分结果
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存划分信息
        split_info = {
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'seed': self.seed,
            'num_samples': {
                'train': len(split_result['train']),
                'val': len(split_result['val']),
                'test': len(split_result['test'])
            }
        }
        
        # 保存到JSON文件
        info_file = output_path / 'dataset_split_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n划分信息已保存到: {info_file}")
        
        # 为每个数据集保存样本ID列表
        for split_name, samples in split_result.items():
            sample_ids = [s.get('sample_id', '') for s in samples]
            id_file = output_path / f'{split_name}_sample_ids.json'
            with open(id_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ids, f, indent=2)
            print(f"{split_name} 样本ID已保存到: {id_file}")