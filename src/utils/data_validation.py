# -*- coding: utf-8 -*-
"""
数据验证工具 - 验证生成的数据集质量和一致性
"""
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DataValidator:
    """数据集验证器"""
    
    def __init__(self, data_dir: str, label_file: str = None):
        """
        初始化验证器
        
        Args:
            data_dir: 数据目录路径
            label_file: 标签文件路径（如果为None，查找all_labels.json）
        """
        self.data_dir = Path(data_dir)
        
        if label_file:
            self.label_file = Path(label_file)
        else:
            self.label_file = self.data_dir / 'all_labels.json'
        
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate_all(self, sample_check_ratio: float = 0.1) -> Dict[str, Any]:
        """
        执行全面验证
        
        Args:
            sample_check_ratio: 抽样检查比例（图像验证）
            
        Returns:
            验证结果字典
        """
        print("Starting comprehensive validation...")
        
        # 1. 验证标签文件
        labels_valid = self.validate_labels()
        
        # 2. 验证文件存在性
        files_valid = self.validate_file_existence()
        
        # 3. 验证坐标范围
        coords_valid = self.validate_coordinates()
        
        # 4. 验证数据一致性
        consistency_valid = self.validate_consistency()
        
        # 5. 抽样验证图像
        images_valid = self.validate_images(sample_ratio=sample_check_ratio)
        
        # 6. 验证数据分布
        distribution_valid = self.validate_distribution()
        
        # 7. 生成验证报告
        report = self.generate_report()
        
        # 8. 保存验证结果
        self.save_validation_results()
        
        return report
    
    def validate_labels(self) -> bool:
        """验证标签文件格式和完整性"""
        print("Validating label file...")
        
        if not self.label_file.exists():
            self.errors.append(f"Label file not found: {self.label_file}")
            return False
        
        try:
            with open(self.label_file, 'r', encoding='utf-8') as f:
                self.labels = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in label file: {e}")
            return False
        
        if not isinstance(self.labels, list):
            self.errors.append("Labels should be a list")
            return False
        
        if len(self.labels) == 0:
            self.errors.append("No labels found in file")
            return False
        
        # 验证每个标签的结构
        required_fields = ['sample_id', 'pic_id', 'paths', 'labels']
        for i, label in enumerate(self.labels):
            for field in required_fields:
                if field not in label:
                    self.errors.append(f"Label {i}: Missing required field '{field}'")
        
        print(f"  Found {len(self.labels)} labels")
        return len(self.errors) == 0
    
    def validate_file_existence(self) -> bool:
        """验证所有引用的文件是否存在"""
        print("Validating file existence...")
        
        missing_files = []
        checked_files = 0
        
        for label in self.labels:
            paths = label.get('paths', {})
            for component_type, rel_path in paths.items():
                if rel_path:
                    file_path = self.data_dir / rel_path
                    checked_files += 1
                    if not file_path.exists():
                        missing_files.append(str(rel_path))
        
        if missing_files:
            self.errors.append(f"Missing {len(missing_files)} files")
            # 只记录前10个缺失文件
            for file in missing_files[:10]:
                self.errors.append(f"  - {file}")
            if len(missing_files) > 10:
                self.errors.append(f"  ... and {len(missing_files) - 10} more")
        
        print(f"  Checked {checked_files} files, {len(missing_files)} missing")
        return len(missing_files) == 0
    
    def validate_coordinates(self) -> bool:
        """验证坐标范围是否合法"""
        print("Validating coordinates...")
        
        coord_errors = []
        
        for i, label in enumerate(self.labels):
            labels_data = label.get('labels', {})
            
            # 验证合成图中拼图中心坐标
            comp_piece = labels_data.get('comp_piece_center', [])
            if len(comp_piece) == 2:
                x, y = comp_piece
                # 根据CLAUDE.md的约束
                if not (15 <= x <= 35):
                    coord_errors.append(f"Label {i}: comp_piece_center x={x} out of range [15, 35]")
                if not (30 <= y <= 130):
                    coord_errors.append(f"Label {i}: comp_piece_center y={y} out of range [30, 130]")
            
            # 验证背景图中缺口中心坐标
            bg_gap = labels_data.get('bg_gap_center', [])
            if len(bg_gap) == 2:
                x, y = bg_gap
                # 根据CLAUDE.md的约束
                if not (65 <= x <= 305):
                    coord_errors.append(f"Label {i}: bg_gap_center x={x} out of range [65, 305]")
                if not (30 <= y <= 130):
                    coord_errors.append(f"Label {i}: bg_gap_center y={y} out of range [30, 130]")
            
            # 验证y坐标一致性
            if len(comp_piece) == 2 and len(bg_gap) == 2:
                if abs(comp_piece[1] - bg_gap[1]) > 0.01:
                    coord_errors.append(
                        f"Label {i}: Y coordinate mismatch: "
                        f"comp_y={comp_piece[1]}, gap_y={bg_gap[1]}"
                    )
        
        if coord_errors:
            for error in coord_errors[:10]:
                self.errors.append(error)
            if len(coord_errors) > 10:
                self.errors.append(f"... and {len(coord_errors) - 10} more coordinate errors")
        
        print(f"  Found {len(coord_errors)} coordinate errors")
        return len(coord_errors) == 0
    
    def validate_consistency(self) -> bool:
        """验证数据一致性"""
        print("Validating data consistency...")
        
        # 按pic_id分组
        labels_by_pic = defaultdict(list)
        for label in self.labels:
            pic_id = label.get('pic_id', 'unknown')
            labels_by_pic[pic_id].append(label)
        
        inconsistencies = []
        
        # 检查同一pic_id的样本是否有一致的滑块位置
        for pic_id, pic_labels in labels_by_pic.items():
            slider_positions = set()
            for label in pic_labels:
                comp_piece = tuple(label.get('labels', {}).get('comp_piece_center', []))
                if len(comp_piece) == 2:
                    slider_positions.add(comp_piece)
            
            # 同一张图片的滑块位置应该相同（根据CLAUDE.md）
            if len(slider_positions) > 1:
                self.warnings.append(
                    f"Pic {pic_id}: Found {len(slider_positions)} different slider positions"
                )
        
        print(f"  Checked {len(labels_by_pic)} pictures")
        return True
    
    def validate_images(self, sample_ratio: float = 0.1) -> bool:
        """抽样验证图像"""
        print(f"Validating images (sampling {sample_ratio*100:.0f}%)...")
        
        n_samples = max(1, int(len(self.labels) * sample_ratio))
        sample_indices = np.random.choice(len(self.labels), n_samples, replace=False)
        
        image_errors = []
        
        for idx in sample_indices:
            label = self.labels[idx]
            paths = label.get('paths', {})
            
            # 验证拼图
            piece_path = self.data_dir / paths.get('piece', '')
            if piece_path.exists():
                piece = cv2.imread(str(piece_path), cv2.IMREAD_UNCHANGED)
                if piece is None:
                    image_errors.append(f"Cannot read piece: {piece_path}")
                elif piece.shape[2] != 4:
                    self.warnings.append(f"Piece missing alpha channel: {piece_path}")
            
            # 验证背景
            bg_path = self.data_dir / paths.get('background', '')
            if bg_path.exists():
                bg = cv2.imread(str(bg_path))
                if bg is None:
                    image_errors.append(f"Cannot read background: {bg_path}")
                elif bg.shape[:2] != (160, 320):
                    image_errors.append(f"Background wrong size {bg.shape[:2]}: {bg_path}")
            
            # 验证合成图
            comp_path = self.data_dir / paths.get('composite', '')
            if comp_path.exists():
                comp = cv2.imread(str(comp_path))
                if comp is None:
                    image_errors.append(f"Cannot read composite: {comp_path}")
                elif comp.shape[:2] != (160, 320):
                    image_errors.append(f"Composite wrong size {comp.shape[:2]}: {comp_path}")
        
        if image_errors:
            for error in image_errors[:5]:
                self.errors.append(error)
            if len(image_errors) > 5:
                self.errors.append(f"... and {len(image_errors) - 5} more image errors")
        
        print(f"  Sampled {n_samples} images, found {len(image_errors)} errors")
        return len(image_errors) == 0
    
    def validate_distribution(self) -> Dict[str, Any]:
        """验证数据分布"""
        print("Analyzing data distribution...")
        
        distribution = {
            'puzzle_sizes': defaultdict(int),
            'rotation_angles': [],
            'fake_gap_counts': [],
            'pics_with_samples': defaultdict(int)
        }
        
        for label in self.labels:
            # 拼图尺寸分布
            size = label.get('labels', {}).get('puzzle_size', 0)
            distribution['puzzle_sizes'][size] += 1
            
            # 旋转角度分布
            theta = label.get('labels', {}).get('gap_pose', {}).get('delta_theta_deg', 0)
            distribution['rotation_angles'].append(theta)
            
            # 假缺口数量
            fake_gaps = label.get('augmented_labels', {}).get('fake_gaps', [])
            distribution['fake_gap_counts'].append(len(fake_gaps))
            
            # 每张图片的样本数
            pic_id = label.get('pic_id', 'unknown')
            distribution['pics_with_samples'][pic_id] += 1
        
        # 计算统计信息
        stats = {
            'puzzle_size_distribution': dict(distribution['puzzle_sizes']),
            'rotation_ratio': sum(abs(a) > 0.01 for a in distribution['rotation_angles']) / len(self.labels),
            'avg_fake_gaps': np.mean(distribution['fake_gap_counts']),
            'samples_per_pic': {
                'min': min(distribution['pics_with_samples'].values()),
                'max': max(distribution['pics_with_samples'].values()),
                'mean': np.mean(list(distribution['pics_with_samples'].values()))
            }
        }
        
        self.validation_results['distribution'] = stats
        return stats
    
    def visualize_sample(self, sample_idx: int = None, save_path: str = None) -> None:
        """
        可视化一个样本
        
        Args:
            sample_idx: 样本索引（如果为None，随机选择）
            save_path: 保存路径
        """
        if sample_idx is None:
            sample_idx = np.random.randint(0, len(self.labels))
        
        label = self.labels[sample_idx]
        paths = label.get('paths', {})
        labels_data = label.get('labels', {})
        
        # 加载图像
        piece_path = self.data_dir / paths.get('piece', '')
        bg_path = self.data_dir / paths.get('background', '')
        comp_path = self.data_dir / paths.get('composite', '')
        
        piece = cv2.imread(str(piece_path), cv2.IMREAD_UNCHANGED) if piece_path.exists() else None
        bg = cv2.imread(str(bg_path)) if bg_path.exists() else None
        comp = cv2.imread(str(comp_path)) if comp_path.exists() else None
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示拼图
        if piece is not None:
            # BGR to RGB
            if piece.shape[2] == 4:
                piece_rgb = cv2.cvtColor(piece[:, :, :3], cv2.COLOR_BGR2RGB)
                axes[0].imshow(piece_rgb)
                axes[0].imshow(piece[:, :, 3], alpha=0.3, cmap='gray')
            else:
                axes[0].imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Piece (Size: {labels_data.get("puzzle_size", "?")})')
        axes[0].axis('off')
        
        # 显示背景
        if bg is not None:
            axes[1].imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
            # 标记缺口位置
            gap_x, gap_y = labels_data.get('bg_gap_center', [0, 0])
            size = labels_data.get('puzzle_size', 50)
            rect = patches.Rectangle((gap_x - size//2, gap_y - size//2), 
                                    size, size, 
                                    linewidth=2, edgecolor='r', facecolor='none')
            axes[1].add_patch(rect)
            axes[1].plot(gap_x, gap_y, 'r+', markersize=10)
        axes[1].set_title(f'Background (Gap: {gap_x:.0f}, {gap_y:.0f})')
        axes[1].axis('off')
        
        # 显示合成图
        if comp is not None:
            axes[2].imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
            # 标记滑块位置
            slider_x, slider_y = labels_data.get('comp_piece_center', [0, 0])
            size = labels_data.get('puzzle_size', 50)
            rect = patches.Rectangle((slider_x - size//2, slider_y - size//2), 
                                    size, size, 
                                    linewidth=2, edgecolor='g', facecolor='none')
            axes[2].add_patch(rect)
            axes[2].plot(slider_x, slider_y, 'g+', markersize=10)
        axes[2].set_title(f'Composite (Slider: {slider_x:.0f}, {slider_y:.0f})')
        axes[2].axis('off')
        
        plt.suptitle(f'Sample {sample_idx}: {label.get("sample_id", "unknown")}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        report = {
            'summary': {
                'total_labels': len(self.labels) if hasattr(self, 'labels') else 0,
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'validation_passed': len(self.errors) == 0
            },
            'errors': self.errors[:20],  # 只显示前20个错误
            'warnings': self.warnings[:20],  # 只显示前20个警告
            'distribution': self.validation_results.get('distribution', {}),
            'recommendations': []
        }
        
        # 生成建议
        if len(self.errors) > 0:
            report['recommendations'].append("Fix all errors before using the dataset")
        
        if len(self.warnings) > 0:
            report['recommendations'].append("Review warnings to improve data quality")
        
        # 特定建议
        if 'distribution' in self.validation_results:
            dist = self.validation_results['distribution']
            if dist.get('rotation_ratio', 0) < 0.05:
                report['recommendations'].append(
                    "Consider adding more rotation augmentation (currently <5%)"
                )
            if dist.get('avg_fake_gaps', 0) < 0.1:
                report['recommendations'].append(
                    "Consider adding more fake gaps for hard negative mining"
                )
        
        return report
    
    def save_validation_results(self, output_file: str = None) -> None:
        """保存验证结果"""
        if output_file is None:
            output_file = self.data_dir / 'validation_report.json'
        else:
            output_file = Path(output_file)
        
        report = self.generate_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\nValidation report saved to: {output_file}")
    
    def print_summary(self) -> None:
        """打印验证摘要"""
        report = self.generate_report()
        
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        
        summary = report['summary']
        print(f"Total labels: {summary['total_labels']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Warnings: {summary['total_warnings']}")
        
        if summary['validation_passed']:
            print("\n✅ VALIDATION PASSED")
        else:
            print("\n❌ VALIDATION FAILED")
        
        if report['errors']:
            print("\nFirst few errors:")
            for error in report['errors'][:5]:
                print(f"  - {error}")
        
        if report['warnings']:
            print("\nFirst few warnings:")
            for warning in report['warnings'][:5]:
                print(f"  - {warning}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")


def validate_dataset_cli():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate CAPTCHA dataset')
    parser.add_argument('data_dir', type=str, help='Path to data directory')
    parser.add_argument('--label-file', type=str, help='Path to label file (default: data_dir/all_labels.json)')
    parser.add_argument('--sample-ratio', type=float, default=0.1, help='Image sampling ratio (default: 0.1)')
    parser.add_argument('--visualize', type=int, help='Visualize sample with given index')
    parser.add_argument('--save-viz', type=str, help='Save visualization to file')
    parser.add_argument('--output', type=str, help='Output file for validation report')
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = DataValidator(
        data_dir=args.data_dir,
        label_file=args.label_file
    )
    
    # 执行验证
    report = validator.validate_all(sample_check_ratio=args.sample_ratio)
    
    # 打印摘要
    validator.print_summary()
    
    # 可视化（如果需要）
    if args.visualize is not None:
        validator.visualize_sample(
            sample_idx=args.visualize,
            save_path=args.save_viz
        )
    
    # 保存报告
    if args.output:
        validator.save_validation_results(args.output)


if __name__ == "__main__":
    # 测试代码
    print("Data validation tool ready for use")
    print("Use 'python data_validation.py --help' for usage information")