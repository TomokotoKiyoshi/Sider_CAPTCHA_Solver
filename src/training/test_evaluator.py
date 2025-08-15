#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试集评估器 - 用于评估模型在测试集上的性能
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import cv2
from PIL import Image


class TestEvaluator:
    """
    测试集评估器
    
    功能：
    1. 加载test.json文件
    2. 评估模型在测试集上的性能
    3. 生成详细的评估报告
    """
    
    def __init__(self, test_file: str, processed_dir: str = "data/processed"):
        """
        初始化测试评估器
        
        Args:
            test_file: test.json文件路径
            processed_dir: 处理后的图像目录
        """
        self.test_file = Path(test_file)
        self.processed_dir = Path(processed_dir)
        self.logger = logging.getLogger('TestEvaluator')
        
        # 加载测试数据
        self.test_data = self._load_test_data()
        self.logger.info(f"加载了 {len(self.test_data)} 个测试样本")
    
    def _load_test_data(self) -> List[Dict]:
        """
        加载test.json文件
        
        Returns:
            测试数据列表
        """
        if not self.test_file.exists():
            raise FileNotFoundError(f"测试文件不存在: {self.test_file}")
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据格式
        test_samples = []
        for item in data:
            # 期望格式: {"image": "PicXXXX_...", "gap_x": x, "gap_y": y, "slider_x": x, "slider_y": y}
            if 'image' in item:
                sample = {
                    'filename': item['image'],
                    'image_path': str(self.processed_dir / item['image']),
                    'gap_x': item.get('gap_x', item.get('bgx', 0)),
                    'gap_y': item.get('gap_y', item.get('bgy', 0)),
                    'slider_x': item.get('slider_x', item.get('sdx', 0)),
                    'slider_y': item.get('slider_y', item.get('sdy', 0))
                }
                test_samples.append(sample)
        
        return test_samples
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        加载并预处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            预处理后的图像张量 [4, H, W]
        """
        # 读取图像
        if Path(image_path).exists():
            image = Image.open(image_path).convert('RGBA')
        else:
            # 如果不存在，创建随机图像用于测试
            self.logger.warning(f"图像不存在: {image_path}，使用随机数据")
            image = Image.new('RGBA', (320, 160), (128, 128, 128, 255))
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 分离RGB和Alpha通道
        rgb = img_array[:, :, :3].astype(np.float32) / 255.0
        alpha = img_array[:, :, 3:4].astype(np.float32) / 255.0
        
        # 合并为4通道 [H, W, 4]
        combined = np.concatenate([rgb, alpha], axis=-1)
        
        # 转换为张量 [4, H, W]
        tensor = torch.from_numpy(combined).permute(2, 0, 1).float()
        
        # 调整大小到模型输入尺寸
        if tensor.shape[1:] != (160, 320):
            # 使用F.interpolate调整大小
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(160, 320),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return tensor
    
    def evaluate(self, model, device: torch.device, batch_size: int = 32) -> Dict:
        """
        在测试集上评估模型
        
        Args:
            model: 训练好的模型
            device: 设备
            batch_size: 批次大小
            
        Returns:
            评估结果字典
        """
        model.eval()
        model = model.to(device)
        
        all_predictions = []
        all_targets = []
        
        # 分批处理
        num_samples = len(self.test_data)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="评估测试集"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_samples = self.test_data[start_idx:end_idx]
                
                # 准备批次数据
                batch_images = []
                batch_gap_coords = []
                batch_slider_coords = []
                
                for sample in batch_samples:
                    # 加载图像
                    image = self.load_image(sample['image_path'])
                    batch_images.append(image)
                    
                    # 目标坐标
                    batch_gap_coords.append([sample['gap_x'], sample['gap_y']])
                    batch_slider_coords.append([sample['slider_x'], sample['slider_y']])
                
                # 转换为批次张量
                images = torch.stack(batch_images).to(device)
                gap_coords = torch.tensor(batch_gap_coords, dtype=torch.float32)
                slider_coords = torch.tensor(batch_slider_coords, dtype=torch.float32)
                
                # 模型预测
                outputs = model(images)
                predictions = model.decode_predictions(outputs)
                
                # 收集预测和目标
                pred_gap = predictions['gap_coords'].cpu().numpy()
                pred_slider = predictions['slider_coords'].cpu().numpy()
                
                for i in range(len(batch_samples)):
                    all_predictions.append({
                        'gap': pred_gap[i],
                        'slider': pred_slider[i]
                    })
                    all_targets.append({
                        'gap': gap_coords[i].numpy(),
                        'slider': slider_coords[i].numpy()
                    })
        
        # 计算指标
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # 生成报告
        self._generate_report(metrics, all_predictions, all_targets)
        
        return metrics
    
    def _calculate_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """
        计算评估指标
        
        Args:
            predictions: 预测列表
            targets: 目标列表
            
        Returns:
            指标字典
        """
        gap_errors = []
        slider_errors = []
        
        for pred, target in zip(predictions, targets):
            # 缺口误差
            gap_error = np.linalg.norm(pred['gap'] - target['gap'])
            gap_errors.append(gap_error)
            
            # 滑块误差
            slider_error = np.linalg.norm(pred['slider'] - target['slider'])
            slider_errors.append(slider_error)
        
        gap_errors = np.array(gap_errors)
        slider_errors = np.array(slider_errors)
        combined_errors = (gap_errors + slider_errors) / 2
        
        metrics = {
            'gap_mae': np.mean(gap_errors),
            'gap_rmse': np.sqrt(np.mean(gap_errors ** 2)),
            'slider_mae': np.mean(slider_errors),
            'slider_rmse': np.sqrt(np.mean(slider_errors ** 2)),
            'combined_mae': np.mean(combined_errors),
            'combined_rmse': np.sqrt(np.mean(combined_errors ** 2)),
            'hit_le_1px': np.mean(combined_errors <= 1.0) * 100,
            'hit_le_2px': np.mean(combined_errors <= 2.0) * 100,
            'hit_le_5px': np.mean(combined_errors <= 5.0) * 100,
            'hit_le_10px': np.mean(combined_errors <= 10.0) * 100,
        }
        
        return metrics
    
    def _generate_report(self, metrics: Dict, predictions: List[Dict], targets: List[Dict]):
        """
        生成评估报告
        
        Args:
            metrics: 指标字典
            predictions: 预测列表
            targets: 目标列表
        """
        print("\n" + "=" * 60)
        print("测试集评估报告")
        print("=" * 60)
        print(f"测试样本数: {len(self.test_data)}")
        print("\n性能指标:")
        print(f"  缺口 MAE: {metrics['gap_mae']:.2f}px")
        print(f"  缺口 RMSE: {metrics['gap_rmse']:.2f}px")
        print(f"  滑块 MAE: {metrics['slider_mae']:.2f}px")
        print(f"  滑块 RMSE: {metrics['slider_rmse']:.2f}px")
        print(f"  综合 MAE: {metrics['combined_mae']:.2f}px")
        print(f"  综合 RMSE: {metrics['combined_rmse']:.2f}px")
        print("\n命中率:")
        print(f"  ≤1px: {metrics['hit_le_1px']:.2f}%")
        print(f"  ≤2px: {metrics['hit_le_2px']:.2f}%")
        print(f"  ≤5px: {metrics['hit_le_5px']:.2f}%")
        print(f"  ≤10px: {metrics['hit_le_10px']:.2f}%")
        print("=" * 60)
        
        # 找出最差的预测
        errors = []
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            gap_error = np.linalg.norm(pred['gap'] - target['gap'])
            slider_error = np.linalg.norm(pred['slider'] - target['slider'])
            combined_error = (gap_error + slider_error) / 2
            errors.append((i, combined_error, self.test_data[i]['filename']))
        
        errors.sort(key=lambda x: x[1], reverse=True)
        
        print("\n最差的5个预测:")
        for i, (idx, error, filename) in enumerate(errors[:5]):
            print(f"  {i+1}. {filename}: 误差={error:.2f}px")
        
        # 保存详细结果
        report_path = Path("test_evaluation_report.json")
        report_data = {
            'metrics': metrics,
            'worst_cases': [
                {
                    'filename': self.test_data[idx]['filename'],
                    'error': float(error),
                    'prediction': {
                        'gap': predictions[idx]['gap'].tolist(),
                        'slider': predictions[idx]['slider'].tolist()
                    },
                    'target': {
                        'gap': targets[idx]['gap'].tolist(),
                        'slider': targets[idx]['slider'].tolist()
                    }
                }
                for idx, error, _ in errors[:10]
            ]
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细报告已保存到: {report_path}")


def evaluate_on_test_set(model_path: str, test_file: str, processed_dir: str = "data/processed"):
    """
    在测试集上评估模型的便捷函数
    
    Args:
        model_path: 模型权重文件路径
        test_file: test.json文件路径
        processed_dir: 处理后的图像目录
    """
    from src.models import create_lite_hrnet_18_fpn
    
    # 创建评估器
    evaluator = TestEvaluator(test_file, processed_dir)
    
    # 加载模型
    model = create_lite_hrnet_18_fpn()
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 评估
    metrics = evaluator.evaluate(model, device)
    
    return metrics


if __name__ == "__main__":
    # 测试评估器
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        test_file = "data/split_for_training/test.json"
        evaluate_on_test_set(model_path, test_file)
    else:
        print("用法: python test_evaluator.py <model_path>")