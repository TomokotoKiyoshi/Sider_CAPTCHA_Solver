# -*- coding: utf-8 -*-
"""
流式数据集生成器
使用可复用批缓冲区，避免内存累积
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
import warnings
import gc

# 抑制警告
warnings.filterwarnings('ignore')

# 全局预处理器实例（每个进程一个）
_global_preprocessor = None

def init_worker(config: Dict):
    """
    初始化工作进程，创建预处理器实例
    每个进程只创建一次，避免重复初始化开销
    """
    global _global_preprocessor
    from .preprocessor import TrainingPreprocessor
    _global_preprocessor = TrainingPreprocessor(config)

def process_single_sample_optimized(label: Dict) -> Optional[Dict[str, Any]]:
    """
    处理单个样本的优化函数
    使用全局预处理器，避免重复创建
    """
    try:
        # 使用全局预处理器
        global _global_preprocessor
        if _global_preprocessor is None:
            return None
            
        # 构建图像路径（使用label中的data_root）
        data_root = Path(label['_data_root'])
        image_path = data_root / label['paths']['composite']
        
        if not image_path.exists():
            return None
        
        # 提取坐标信息
        gap_center = tuple(label['labels']['bg_gap_center'])
        slider_center = tuple(label['labels']['comp_piece_center'])
        gap_angle = label['labels']['gap_pose'].get('delta_theta_deg', 0.0)
        
        # 处理混淆缺口
        fake_gaps = []
        if 'fake_gaps' in label['labels']:
            for fake_gap in label['labels']['fake_gaps']:
                fake_gaps.append(tuple(fake_gap['center']))
        
        # 使用预处理器处理图像
        result = _global_preprocessor.preprocess(
            str(image_path),
            gap_center=gap_center,
            slider_center=slider_center,
            confusing_gaps=fake_gaps if fake_gaps else None,
            gap_angle=gap_angle
        )
        
        # 只返回必要的数据，减少序列化开销
        # 注意：result中的数据已经是NumPy数组，不需要再调用.numpy()
        return {
            'sample_id': label.get('sample_id', 'unknown'),
            'input': result['input'],  # 已经是numpy数组
            'heatmaps': result['heatmaps'],  # 已经是numpy数组
            'offsets': result['offsets'],  # 已经是numpy数组
            'weight_mask': result['weight_mask'],  # 已经是numpy数组
            'transform_params': result['transform_params'],
            'gap_grid': result['gap_grid'],
            'gap_offset': result['gap_offset'],
            'slider_grid': result['slider_grid'],
            'slider_offset': result['slider_offset'],
            'confusing_gaps': result.get('confusing_gaps', []),
            'gap_angle': result.get('gap_angle', 0.0)
        }
    except Exception as e:
        print(f"Failed to process sample {label.get('sample_id', 'unknown')}: {e}")
        return None


class StreamingDatasetGenerator:
    """
    流式数据集生成器
    核心优化：
    1. 使用可复用的批缓冲区，避免内存累积
    2. 批满即写盘，没有中间列表
    3. 分文件保存避免临时zip缓冲
    4. 内存使用稳定且可预测
    """
    
    def __init__(self,
                 data_root: str,
                 output_dir: str,
                 config_path: Optional[str] = None,
                 batch_size: int = None,
                 num_workers: Optional[int] = None):
        """
        初始化流式数据集生成器
        """
        from .config_loader import load_config
        from multiprocessing import cpu_count
        
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        
        # 加载配置
        self.config = load_config(config_path)
        preprocessing_config = self.config['preprocessing']
        
        # 从配置文件读取batch_size
        if batch_size is not None:
            self.batch_size = batch_size
        elif 'dataset' in self.config and 'batch_size' in self.config['dataset']:
            self.batch_size = self.config['dataset']['batch_size']
        else:
            self.batch_size = 64
        
        # 设置工作进程数
        if num_workers is not None:
            self.num_workers = num_workers
        elif 'dataset' in self.config and 'num_workers' in self.config['dataset']:
            self.num_workers = self.config['dataset']['num_workers']
            print(f"Using num_workers from config: {self.num_workers}")
        else:
            self.num_workers = max(1, cpu_count() - 1)
            print(f"Using default num_workers: {self.num_workers} (CPU cores - 1)")
        
        # 从配置读取输出结构
        self.output_structure = self.config.get('output_structure', {
            'image_subdir': 'images',
            'label_subdir': 'labels'
        })
        self.file_naming = self.config.get('file_naming', {
            'image_pattern': '{split}_{batch_id:04d}.npy',
            'heatmap_pattern': '{split}_{batch_id:04d}_heatmaps.npy',
            'offset_pattern': '{split}_{batch_id:04d}_offsets.npy',
            'weight_pattern': '{split}_{batch_id:04d}_weights.npy',
            'meta_pattern': '{split}_{batch_id:04d}_meta.json',
            'index_pattern': '{split}_index.json'
        })
        
        # 创建输出目录（使用配置文件中的split目录结构）
        self.output_dir.mkdir(parents=True, exist_ok=True)
        splits_config = self.output_structure.get('splits', {
            'train': 'train',
            'val': 'val', 
            'test': 'test'
        })
        
        for split_key, split_dir in splits_config.items():
            (self.output_dir / self.output_structure['image_subdir'] / split_dir).mkdir(parents=True, exist_ok=True)
            (self.output_dir / self.output_structure['label_subdir'] / split_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建split_info目录（如果配置中有）
        if 'split_info_subdir' in self.output_structure:
            (self.output_dir / self.output_structure['split_info_subdir']).mkdir(parents=True, exist_ok=True)
        
        # 记录配置信息
        self.target_size = preprocessing_config['letterbox']['target_size']
        self.downsample = preprocessing_config['coordinate']['downsample']
        
        # 计算输出维度
        self.input_shape = (4, self.target_size[1], self.target_size[0])
        self.grid_size = (
            self.target_size[1] // self.downsample,
            self.target_size[0] // self.downsample
        )
        
        # 初始化可复用的批缓冲区
        self._init_batch_buffers()
        
        print(f"Streaming dataset generator initialized")
        print(f"  Data root: {self.data_root}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Worker processes: {self.num_workers}")
        print(f"  Target size: {self.target_size}")
        print(f"  Grid size: {self.grid_size}")
        print(f"  ✅ Using streaming write with reusable buffers")
    
    def _init_batch_buffers(self):
        """初始化可复用的批缓冲区"""
        # 预分配批缓冲区，避免重复分配
        self._buf_images = np.empty((self.batch_size, *self.input_shape), dtype=np.float32)
        self._buf_heatmaps = np.empty((self.batch_size, 2, *self.grid_size), dtype=np.float32)
        self._buf_offsets = np.empty((self.batch_size, 4, *self.grid_size), dtype=np.float32)
        self._buf_weight_masks = np.empty((self.batch_size, *self.grid_size), dtype=np.float32)
        
        # 元数据缓冲（这些比较小，可以用列表）
        self._buf_metadata = {
            'sample_ids': [],
            'transform_params': [],
            'grid_coords': [],
            'offsets_meta': [],
            'confusing_gaps': [],
            'gap_angles': []
        }
        
        # 写入指针
        self._buf_ptr = 0
        self._batch_counter = 0
    
    def _reset_metadata_buffer(self):
        """重置元数据缓冲"""
        self._buf_metadata = {
            'sample_ids': [],
            'transform_params': [],
            'grid_coords': [],
            'offsets_meta': [],
            'confusing_gaps': [],
            'gap_angles': []
        }
    
    def _write_sample_to_buffer(self, sample: Dict[str, Any]) -> bool:
        """
        将样本写入缓冲区
        
        Returns:
            True if buffer is full and needs to be flushed
        """
        idx = self._buf_ptr
        
        # 直接写入预分配的数组缓冲区
        self._buf_images[idx] = sample['input']
        self._buf_heatmaps[idx] = sample['heatmaps']
        self._buf_offsets[idx] = sample['offsets']
        self._buf_weight_masks[idx] = sample['weight_mask']
        
        # 元数据添加到列表
        self._buf_metadata['sample_ids'].append(sample['sample_id'])
        self._buf_metadata['transform_params'].append(sample['transform_params'])
        self._buf_metadata['grid_coords'].append({
            'gap': sample['gap_grid'],
            'slider': sample['slider_grid']
        })
        self._buf_metadata['offsets_meta'].append({
            'gap': sample['gap_offset'],
            'slider': sample['slider_offset']
        })
        self._buf_metadata['confusing_gaps'].append(sample.get('confusing_gaps', []))
        self._buf_metadata['gap_angles'].append(sample.get('gap_angle', 0.0))
        
        # 移动指针
        self._buf_ptr += 1
        
        # 检查是否需要刷新缓冲区
        return self._buf_ptr >= self.batch_size
    
    def _flush_buffer_to_disk(self, split: str):
        """将当前缓冲区写入磁盘"""
        if self._buf_ptr == 0:
            return  # 空缓冲区，无需写入
        
        batch_size = self._buf_ptr
        batch_id = self._batch_counter
        
        # 使用配置的文件命名模式准备文件路径（使用配置中的split目录）
        splits_config = self.output_structure.get('splits', {
            'train': 'train',
            'val': 'val',
            'test': 'test'
        })
        split_dir = splits_config.get(split, split)  # 如果没有配置，使用原始split名称
        
        image_path = self.output_dir / self.output_structure['image_subdir'] / split_dir / self.file_naming['image_pattern'].format(split=split, batch_id=batch_id)
        heatmap_path = self.output_dir / self.output_structure['label_subdir'] / split_dir / self.file_naming['heatmap_pattern'].format(split=split, batch_id=batch_id)
        offset_path = self.output_dir / self.output_structure['label_subdir'] / split_dir / self.file_naming['offset_pattern'].format(split=split, batch_id=batch_id)
        weight_path = self.output_dir / self.output_structure['label_subdir'] / split_dir / self.file_naming['weight_pattern'].format(split=split, batch_id=batch_id)
        meta_path = self.output_dir / self.output_structure['label_subdir'] / split_dir / self.file_naming['meta_pattern'].format(split=split, batch_id=batch_id)
        
        # 分文件保存，避免np.savez的临时zip缓冲
        # 只保存实际使用的部分（0:batch_size）
        np.save(image_path, self._buf_images[:batch_size])
        np.save(heatmap_path, self._buf_heatmaps[:batch_size])
        np.save(offset_path, self._buf_offsets[:batch_size])
        np.save(weight_path, self._buf_weight_masks[:batch_size])
        
        # 保存元数据
        metadata = {
            'batch_id': batch_id,
            'batch_size': batch_size,
            **self._buf_metadata
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Batch {batch_id} saved: {batch_size} samples")
        
        # 重置缓冲区指针和元数据
        self._buf_ptr = 0
        self._reset_metadata_buffer()
        self._batch_counter += 1
    
    def load_labels(self, labels_path: str) -> List[Dict]:
        """加载标签文件"""
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} labels")
        
        # 在标签中添加data_root路径，避免序列化整个路径对象
        for label in labels:
            label['_data_root'] = str(self.data_root)
        
        return labels
    
    def generate_dataset(self, labels_path: str, split: str = 'train', max_samples: Optional[int] = None,
                        labels_subset: Optional[List[Dict]] = None):
        """
        使用流式写入生成数据集
        
        Args:
            labels_path: 标签文件路径
            split: 数据集划分 (train/val/test)
            max_samples: 最大样本数（用于测试）
            labels_subset: 预先划分好的标签子集（如果提供，则忽略labels_path）
        """
        print(f"\n{'='*60}")
        print(f"Generating {split} dataset (Streaming Version)")
        print(f"{'='*60}")
        print(f"Using {self.num_workers} parallel processes")
        
        # 加载标签
        if labels_subset is not None:
            labels = labels_subset
            print(f"Using provided subset: {len(labels)} labels")
            # 为subset模式下的标签也添加_data_root字段
            for label in labels:
                if '_data_root' not in label:
                    label['_data_root'] = str(self.data_root)
        else:
            labels = self.load_labels(labels_path)
        
        # 如果指定了最大样本数，截取标签
        if max_samples is not None and max_samples < len(labels):
            labels = labels[:max_samples]
            print(f"Limiting to {max_samples} samples for testing")
        
        # 重置缓冲区
        self._buf_ptr = 0
        self._batch_counter = 0
        self._reset_metadata_buffer()
        
        failed_count = 0
        
        # 从配置读取参数
        chunk_size_config = self.config.get('dataset', {}).get('chunk_size', 20)
        maxtasksperchild = self.config.get('dataset', {}).get('maxtasksperchild', 10)
        gc_interval = self.config.get('dataset', {}).get('gc_interval', 2)
        
        print(f"Configuration:")
        print(f"  Chunk size: {chunk_size_config}")
        print(f"  Max tasks per child: {maxtasksperchild}")
        print(f"  GC interval: every {gc_interval} batches")
        print(f"  💡 Streaming write enabled - no memory accumulation")
        
        # 创建进程池
        with Pool(
            processes=self.num_workers,
            initializer=init_worker,
            initargs=(self.config,),
            maxtasksperchild=maxtasksperchild
        ) as pool:
            
            with tqdm(total=len(labels), desc=f"Processing {split} samples") as pbar:
                # 使用imap_unordered流式处理
                for processed in pool.imap_unordered(
                    process_single_sample_optimized,
                    labels,
                    chunksize=chunk_size_config
                ):
                    if processed is not None:
                        # 直接写入缓冲区
                        if self._write_sample_to_buffer(processed):
                            # 缓冲区满，立即写盘
                            self._flush_buffer_to_disk(split)
                            
                            # 定期强制垃圾回收
                            if self._batch_counter % gc_interval == 0:
                                gc.collect(2)  # 完整垃圾回收
                    else:
                        failed_count += 1
                    
                    pbar.update(1)
        
        # 保存剩余的样本（未满一批的）
        if self._buf_ptr > 0:
            self._flush_buffer_to_disk(split)
        
        print(f"\n{'='*60}")
        print(f"Dataset generation completed:")
        print(f"  ✅ Success: {len(labels) - failed_count}")
        print(f"  ❌ Failed: {failed_count}")
        print(f"  📦 Total batches: {self._batch_counter}")
        print(f"{'='*60}")
        
        # 生成索引文件
        if len(labels) - failed_count > 0:
            self._generate_index(split)
    
    def _generate_index(self, split: str):
        """生成数据集索引文件"""
        image_files = sorted((self.output_dir / self.output_structure['image_subdir'] / split).glob(f'{split}_*.npy'))
        
        index = {
            'split': split,
            'num_batches': len(image_files),
            'batch_size': self.batch_size,
            'total_samples': 0,
            'input_shape': self.input_shape,
            'grid_size': self.grid_size,
            'batches': []
        }
        
        for img_file in image_files:
            batch_id = int(img_file.stem.split('_')[-1])
            meta_file = self.output_dir / self.output_structure['label_subdir'] / split / self.file_naming['meta_pattern'].format(split=split, batch_id=batch_id)
            
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            batch_info = {
                'batch_id': batch_id,
                'batch_size': meta['batch_size'],
                'image_file': img_file.name,
                'heatmap_file': self.file_naming['heatmap_pattern'].format(split=split, batch_id=batch_id),
                'offset_file': self.file_naming['offset_pattern'].format(split=split, batch_id=batch_id),
                'weight_file': self.file_naming['weight_pattern'].format(split=split, batch_id=batch_id),
                'meta_file': meta_file.name
            }
            index['batches'].append(batch_info)
            index['total_samples'] += meta['batch_size']
        
        # 保存索引（保存到对应的split文件夹）
        index_path = self.output_dir / self.output_structure['label_subdir'] / split / self.file_naming['index_pattern'].format(split=split)
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        
        print(f"\n📋 Generated index file: {index_path}")
        print(f"  Total batches: {len(image_files)}")
        print(f"  Total samples: {index['total_samples']}")