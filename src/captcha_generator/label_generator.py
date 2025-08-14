# -*- coding: utf-8 -*-
"""
训练标签生成器
独立的标签生成模块，可被验证码生成脚本调用
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict


class NumpyJSONEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class CaptchaLabelGenerator:
    """验证码训练标签生成器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化标签生成器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.labels = []
        self.labels_by_pic = defaultdict(list)
        
    def create_label(self,
                    pic_id: str,
                    sample_id: str,
                    gap_position: Tuple[int, int],
                    slider_position: Tuple[int, int],
                    puzzle_size: int,
                    component_paths: Dict[str, str],
                    confusion_metadata: Optional[Dict[str, Any]] = None,
                    fake_gaps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        创建单个样本的训练标签
        
        Args:
            pic_id: 原始图片ID (如 'Pic0001')
            sample_id: 样本唯一ID
            gap_position: 缺口中心坐标 (x, y)
            slider_position: 滑块中心坐标 (x, y)
            puzzle_size: 拼图尺寸
            component_paths: 组件文件路径字典 {'piece': ..., 'background': ..., 'composite': ...}
            confusion_metadata: 混淆策略元数据
            fake_gaps: 假缺口列表
            
        Returns:
            训练标签字典
        """
        # 基础标签（必需）
        label = {
            'sample_id': sample_id,
            'pic_id': pic_id,
            
            # 路径信息
            'paths': {
                'piece': component_paths.get('piece', ''),
                'background': component_paths.get('background', ''),
                'composite': component_paths.get('composite', '')
            },
            
            # 坐标标签
            'labels': {
                # 合成图中拼图的中心坐标
                'comp_piece_center': list(slider_position),  # (x_pc, y_pc)
                
                # 背景图中真实缺口的中心坐标
                'bg_gap_center': list(gap_position),  # (x_gc, y_gc)
                
                # 真实缺口相对拼图的微旋转与尺度
                'gap_pose': {
                    'delta_theta_deg': 0.0,  # 默认无旋转
                    'scale': 1.0  # 默认无缩放
                },
                
                # 拼图尺寸
                'puzzle_size': puzzle_size
            }
        }
        
        # 如果有旋转混淆，更新旋转角度
        if confusion_metadata and 'rotation' in confusion_metadata:
            rotation_params = confusion_metadata['rotation']
            if 'angle' in rotation_params:
                label['labels']['gap_pose']['delta_theta_deg'] = float(rotation_params['angle'])
        
        # 增强标签（可选但推荐）
        if fake_gaps:
            augmented_labels = {'fake_gaps': []}
            for gap in fake_gaps:
                fake_gap_label = {
                    'center': list(gap.get('position', [0, 0])),
                    'delta_theta_deg': float(gap.get('rotation_angle', 0.0)),
                    'scale': float(gap.get('scale', 1.0))
                }
                augmented_labels['fake_gaps'].append(fake_gap_label)
            
            label['augmented_labels'] = augmented_labels
        
        return label
    
    def add_label(self, label: Dict[str, Any]) -> None:
        """
        添加一个标签到集合中
        
        Args:
            label: 标签字典
        """
        self.labels.append(label)
        pic_id = label.get('pic_id', 'unknown')
        self.labels_by_pic[pic_id].append(label)
    
    def create_and_add_label(self, **kwargs) -> Dict[str, Any]:
        """
        创建并添加标签
        
        Returns:
            创建的标签字典
        """
        label = self.create_label(**kwargs)
        self.add_label(label)
        return label
    
    def save_labels(self, prefix: str = '') -> Dict[str, Path]:
        """
        保存所有标签到文件
        
        Args:
            prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        saved_files = {}
        
        # 保存所有标签
        all_labels_file = self.output_dir / f'{prefix}all_labels.json'
        with open(all_labels_file, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
        saved_files['all_labels'] = all_labels_file
        
        # 保存按pic_id分组的标签
        grouped_labels_file = self.output_dir / f'{prefix}labels_by_pic.json'
        with open(grouped_labels_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.labels_by_pic), f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
        saved_files['labels_by_pic'] = grouped_labels_file
        
        # 保存标签统计信息
        stats = self.get_statistics()
        stats_file = self.output_dir / f'{prefix}labels_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
        saved_files['stats'] = stats_file
        
        return saved_files
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取标签统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_labels': len(self.labels),
            'unique_pics': len(self.labels_by_pic),
            'samples_per_pic': {},
            'puzzle_sizes': defaultdict(int),
            'has_fake_gaps': 0,
            'has_rotation': 0,
            'avg_fake_gaps_per_sample': 0
        }
        
        # 统计每张图片的样本数
        for pic_id, labels in self.labels_by_pic.items():
            stats['samples_per_pic'][pic_id] = len(labels)
        
        # 统计其他信息
        total_fake_gaps = 0
        for label in self.labels:
            # 统计拼图尺寸
            puzzle_size = label['labels'].get('puzzle_size', 0)
            stats['puzzle_sizes'][str(puzzle_size)] += 1
            
            # 统计旋转
            delta_theta = label['labels']['gap_pose'].get('delta_theta_deg', 0)
            if abs(delta_theta) > 0.01:
                stats['has_rotation'] += 1
            
            # 统计假缺口
            if 'augmented_labels' in label and 'fake_gaps' in label['augmented_labels']:
                stats['has_fake_gaps'] += 1
                total_fake_gaps += len(label['augmented_labels']['fake_gaps'])
        
        # 计算平均假缺口数
        if stats['has_fake_gaps'] > 0:
            stats['avg_fake_gaps_per_sample'] = total_fake_gaps / stats['has_fake_gaps']
        
        # 转换defaultdict为普通dict
        stats['puzzle_sizes'] = dict(stats['puzzle_sizes'])
        
        return stats
    
    def validate_labels(self) -> List[str]:
        """
        验证标签的完整性和正确性
        
        Returns:
            错误信息列表
        """
        errors = []
        
        for i, label in enumerate(self.labels):
            # 检查必需字段
            required_fields = ['sample_id', 'pic_id', 'paths', 'labels']
            for field in required_fields:
                if field not in label:
                    errors.append(f"Label {i}: Missing required field '{field}'")
            
            # 检查路径
            if 'paths' in label:
                required_paths = ['piece', 'background', 'composite']
                for path_key in required_paths:
                    if path_key not in label['paths']:
                        errors.append(f"Label {i}: Missing path '{path_key}'")
            
            # 检查坐标
            if 'labels' in label:
                if 'comp_piece_center' in label['labels']:
                    center = label['labels']['comp_piece_center']
                    if not isinstance(center, list) or len(center) != 2:
                        errors.append(f"Label {i}: Invalid comp_piece_center format")
                    else:
                        x, y = center
                        # 验证坐标范围
                        if not (15 <= x <= 40):
                            errors.append(f"Label {i}: comp_piece_center x={x} out of range [15, 40]")
                        if not (30 <= y <= 130):
                            errors.append(f"Label {i}: comp_piece_center y={y} out of range [30, 130]")
                
                if 'bg_gap_center' in label['labels']:
                    center = label['labels']['bg_gap_center']
                    if not isinstance(center, list) or len(center) != 2:
                        errors.append(f"Label {i}: Invalid bg_gap_center format")
                    else:
                        x, y = center
                        # 验证坐标范围（缺口x范围更大）
                        if not (40 <= x <= 305):
                            errors.append(f"Label {i}: bg_gap_center x={x} out of range [40, 305]")
                        if not (30 <= y <= 130):
                            errors.append(f"Label {i}: bg_gap_center y={y} out of range [30, 130]")
                
                # 验证y坐标一致性
                if 'comp_piece_center' in label['labels'] and 'bg_gap_center' in label['labels']:
                    comp_y = label['labels']['comp_piece_center'][1]
                    gap_y = label['labels']['bg_gap_center'][1]
                    if abs(comp_y - gap_y) > 0.01:  # 允许浮点误差
                        errors.append(f"Label {i}: Y coordinates mismatch: comp_y={comp_y}, gap_y={gap_y}")
        
        return errors


def create_label_from_captcha_result(
    pic_index: int,
    sample_idx: int,
    gap_position: Tuple[int, int],
    slider_position: Tuple[int, int],
    puzzle_size: int,
    confusion_type: str,
    confusion_metadata: Optional[Dict[str, Any]] = None,
    additional_gaps: Optional[List[Dict[str, Any]]] = None,
    file_hash: str = '',
    base_filename: str = '',
    image_width: Optional[int] = None,
    image_height: Optional[int] = None
) -> Dict[str, Any]:
    """
    从验证码生成结果创建训练标签（便捷函数）
    
    Args:
        pic_index: 图片索引
        sample_idx: 样本索引
        gap_position: 缺口位置
        slider_position: 滑块位置
        puzzle_size: 拼图尺寸
        confusion_type: 混淆类型
        confusion_metadata: 混淆元数据
        additional_gaps: 额外的假缺口
        file_hash: 文件哈希
        base_filename: 基础文件名
        
    Returns:
        训练标签字典
    """
    pic_id = f"Pic{pic_index:04d}"
    
    if not base_filename:
        base_filename = f"{pic_id}_Bgx{gap_position[0]}Bgy{gap_position[1]}_Sdx{slider_position[0]}Sdy{slider_position[1]}_{file_hash}"
    
    # 组件路径（更新为正确的目录结构）
    component_paths = {
        'piece': f"components/sliders/{base_filename}_slider.png",
        'background': f"components/backgrounds/{base_filename}_gap.png",
        'composite': f"captchas/{base_filename}.png"
    }
    
    # 处理假缺口
    fake_gaps = None
    if additional_gaps:
        fake_gaps = []
        for gap in additional_gaps:
            # 使用geometric_center（质心）而不是position（边界框中心）
            geometric_center = gap.get('geometric_center')
            if geometric_center is None:
                # 如果没有geometric_center，尝试使用position作为备选
                geometric_center = gap.get('position', [0, 0])
            
            fake_gap = {
                'position': geometric_center,  # 使用质心坐标
                'rotation_angle': gap.get('rotation', 0.0),  # 注意：字段名是rotation而不是rotation_angle
                'scale': gap.get('scale', 1.0)
            }
            fake_gaps.append(fake_gap)
    
    # 创建标签生成器并生成标签
    generator = CaptchaLabelGenerator(Path('.'))  # 临时路径，仅用于创建标签
    label = generator.create_label(
        pic_id=pic_id,
        sample_id=base_filename,
        gap_position=gap_position,
        slider_position=slider_position,
        puzzle_size=puzzle_size,
        component_paths=component_paths,
        confusion_metadata=confusion_metadata,
        fake_gaps=fake_gaps
    )
    
    # 添加图片尺寸信息
    label['image_size'] = {
        'width': image_width,
        'height': image_height
    }
    
    return label


if __name__ == "__main__":
    # 测试代码
    import tempfile
    
    # 创建临时输出目录
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 创建标签生成器
        generator = CaptchaLabelGenerator(output_dir)
        
        # 生成测试标签
        for i in range(5):
            label = generator.create_and_add_label(
                pic_id=f"Pic{i+1:04d}",
                sample_id=f"sample_{i+1}",
                gap_position=(120 + i*10, 70),
                slider_position=(30, 70),
                puzzle_size=50,
                component_paths={
                    'piece': f"piece_{i+1}.png",
                    'background': f"bg_{i+1}.png",
                    'composite': f"comp_{i+1}.png"
                },
                confusion_metadata={'rotation': {'angle': 0.5}},
                fake_gaps=[
                    {'position': [150, 70], 'rotation_angle': 15.0, 'scale': 0.9}
                ]
            )
        
        # 保存标签
        saved_files = generator.save_labels(prefix='test_')
        
        # 打印统计信息
        stats = generator.get_statistics()
        print("Label Statistics:")
        print(json.dumps(stats, indent=2))
        
        # 验证标签
        errors = generator.validate_labels()
        if errors:
            print("\nValidation Errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("\nAll labels validated successfully!")
        
        print(f"\nSaved files:")
        for key, path in saved_files.items():
            print(f"  - {key}: {path}")