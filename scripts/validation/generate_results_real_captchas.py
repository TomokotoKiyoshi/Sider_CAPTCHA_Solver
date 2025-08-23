"""
处理真实验证码数据集（site1和site2）的评估脚本
生成可视化结果和评估指标
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import time
import yaml

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sider_captcha_solver.predictor import CaptchaPredictor


class RealCaptchaResultGenerator:
    """真实验证码结果生成器"""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化结果生成器
        
        Args:
            model_path: 模型路径 (默认从配置文件读取版本号)
            device: 推理设备 ('cuda', 'cpu', 或 None 自动选择)
        """
        # 从配置文件读取版本号
        config_path = Path('config/version.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            version_config = yaml.safe_load(f)
        self.version = version_config['version']
        
        # 如果没有指定模型路径，使用基于版本号的默认路径
        if model_path is None:
            model_path = f"src/checkpoints/{self.version}/best_model.pth"
        self.model_path = Path(model_path)
        
        # 自动选择设备 - 优先使用GPU
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"🚀 CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                self.device = 'cpu'
                print("⚠️ CUDA not available, using CPU")
        else:
            self.device = device
            if device == 'cuda' and not torch.cuda.is_available():
                print("⚠️ CUDA requested but not available, falling back to CPU")
                self.device = 'cpu'
        
        # 加载模型
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            # 强制使用指定的设备
            self.predictor = CaptchaPredictor(
                model_path=str(self.model_path),
                device=self.device  # 使用指定的设备而不是'auto'
            )
            
            # 验证模型确实在GPU上
            if self.device == 'cuda':
                # 获取模型的第一个参数来检查设备
                first_param = next(self.predictor.model.parameters())
                if first_param.is_cuda:
                    print(f"✅ Model loaded on GPU: {self.model_path}")
                    # 显示GPU内存使用情况
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"   GPU Memory - Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")
                else:
                    print(f"⚠️ Model loaded but not on GPU!")
            else:
                print(f"✅ Model loaded on CPU: {self.model_path}")
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def parse_filename(self, filename: str):
        """
        解析文件名获取GT坐标
        格式: PicXXXX_BgxXXXBgyYYY_SdxXXXSdyYYY_hash.png
        """
        pattern = r"Pic(\d+)_Bgx(\d+)Bgy(\d+)_Sdx(\d+)Sdy(\d+)_(\w+)\.png"
        match = re.match(pattern, filename)
        
        if match:
            pic_id, bgx, bgy, sdx, sdy, hash_val = match.groups()
            return {
                'pic_id': int(pic_id),
                'gap_x': int(bgx),
                'gap_y': int(bgy),
                'slider_x': int(sdx),
                'slider_y': int(sdy),
                'hash': hash_val,
                'sliding_distance': int(bgx) - int(sdx)
            }
        return None
    
    def evaluate_site(self, site_name: str, data_dir: Path, output_dir: Path, max_images: int = None, batch_size: int = 32):
        """
        评估单个站点的数据 - 使用批量GPU推理
        
        Args:
            site_name: 站点名称 (site1或site2)
            data_dir: 数据目录
            output_dir: 输出目录
            max_images: 最大处理图片数
            batch_size: 批处理大小
        """
        print(f"\n📊 Evaluating {site_name}...")
        print(f"   Data directory: {data_dir}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {self.device}")
        
        if not data_dir.exists():
            print(f"❌ Directory not found: {data_dir}")
            return None
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 获取所有图片
        image_files = sorted(data_dir.glob("*.png"))
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"   Found {len(image_files)} images")
        
        # 初始化结果
        results = {
            "site": site_name,
            "model_version": self.version,
            "timestamp": datetime.now().isoformat(),
            "total_images": len(image_files),
            "predictions": [],
            "metrics": {}
        }
        
        # 错误统计
        distance_errors = []
        gap_errors = []
        slider_errors = []
        gap_x_errors = []
        gap_y_errors = []
        slider_x_errors = []
        slider_y_errors = []
        processing_times = []
        
        # 解析所有GT信息
        gt_infos = []
        valid_files = []
        for img_path in image_files:
            gt_info = self.parse_filename(img_path.name)
            if gt_info:
                gt_infos.append(gt_info)
                valid_files.append(img_path)
            else:
                print(f"⚠️ Cannot parse filename: {img_path.name}")
        
        if not valid_files:
            print(f"❌ No valid files to process")
            return None
        
        # 批量推理 - 使用新的批量方法
        print(f"\n🚀 开始批量GPU推理 ({len(valid_files)} 张图片, 批大小={batch_size})...")
        start_time = time.time()
        
        # 使用批量推理方法
        batch_predictions = self.predictor.predict_batch(
            [str(f) for f in valid_files], 
            batch_size=batch_size
        )
        
        inference_time = time.time() - start_time
        print(f"⚡ 推理完成! 耗时: {inference_time:.2f}秒 ({len(valid_files)/inference_time:.1f} 张/秒)")
        
        # 处理结果
        for idx, (img_path, gt_info, pred) in enumerate(zip(valid_files, gt_infos, batch_predictions)):
            try:
                if not pred['success']:
                    print(f"❌ Prediction failed for {img_path.name}: {pred.get('error', 'Unknown')}")
                    continue
                
                # 计算误差
                gap_error = np.sqrt(
                    (pred['gap_x'] - gt_info['gap_x'])**2 + 
                    (pred['gap_y'] - gt_info['gap_y'])**2
                )
                slider_error = np.sqrt(
                    (pred['slider_x'] - gt_info['slider_x'])**2 + 
                    (pred['slider_y'] - gt_info['slider_y'])**2
                )
                distance_error = abs(pred['sliding_distance'] - gt_info['sliding_distance'])
                
                # 计算各坐标轴误差
                gap_x_error = abs(pred['gap_x'] - gt_info['gap_x'])
                gap_y_error = abs(pred['gap_y'] - gt_info['gap_y'])
                slider_x_error = abs(pred['slider_x'] - gt_info['slider_x'])
                slider_y_error = abs(pred['slider_y'] - gt_info['slider_y'])
                
                # 记录结果
                prediction = {
                    "image": img_path.name,
                    "ground_truth": {
                        "gap_x": gt_info['gap_x'],
                        "gap_y": gt_info['gap_y'],
                        "slider_x": gt_info['slider_x'],
                        "slider_y": gt_info['slider_y'],
                        "sliding_distance": gt_info['sliding_distance']
                    },
                    "prediction": {
                        "gap_x": pred['gap_x'],
                        "gap_y": pred['gap_y'],
                        "slider_x": pred['slider_x'],
                        "slider_y": pred['slider_y'],
                        "sliding_distance": pred['sliding_distance'],
                        "confidence": pred['confidence']
                    },
                    "errors": {
                        "gap_error": float(gap_error),
                        "slider_error": float(slider_error),
                        "distance_error": float(distance_error),
                        "gap_x_error": float(gap_x_error),
                        "gap_y_error": float(gap_y_error),
                        "slider_x_error": float(slider_x_error),
                        "slider_y_error": float(slider_y_error)
                    },
                    "processing_time_ms": pred.get('processing_time_ms', 0)  # 批量推理时可能没有单独的时间
                }
                
                results["predictions"].append(prediction)
                distance_errors.append(distance_error)
                gap_errors.append(gap_error)
                slider_errors.append(slider_error)
                gap_x_errors.append(gap_x_error)
                gap_y_errors.append(gap_y_error)
                slider_x_errors.append(slider_x_error)
                slider_y_errors.append(slider_y_error)
                processing_times.append(pred.get('processing_time_ms', inference_time * 1000 / len(valid_files)))
                
                # 生成可视化（可选，只对前几张）
                if idx < 10:  # 只可视化前10张
                    self._visualize_prediction(
                        img_path, pred, gt_info,
                        vis_dir / f"{img_path.stem}_result.png"
                    )
            
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
        
        # 计算统计指标
        if distance_errors:
            distance_errors = np.array(distance_errors)
            gap_errors = np.array(gap_errors)
            slider_errors = np.array(slider_errors)
            gap_x_errors = np.array(gap_x_errors)
            gap_y_errors = np.array(gap_y_errors)
            slider_x_errors = np.array(slider_x_errors)
            slider_y_errors = np.array(slider_y_errors)
            
            # 判断两者都在阈值内的精度
            both_within_5px = np.mean((gap_errors <= 5) & (slider_errors <= 5)) * 100
            both_within_7px = np.mean((gap_errors <= 7) & (slider_errors <= 7)) * 100
            
            results["metrics"] = {
                # 距离相关指标
                "distance_mae": float(np.mean(distance_errors)),
                "distance_within_5px": float(np.mean(distance_errors <= 5) * 100),
                "distance_within_7px": float(np.mean(distance_errors <= 7) * 100),
                
                # 滑块和缺口都在阈值内
                "both_within_5px": float(both_within_5px),
                "both_within_7px": float(both_within_7px),
                
                # 滑块指标
                "slider_mae": float(np.mean(slider_errors)),
                "slider_within_5px": float(np.mean(slider_errors <= 5) * 100),
                "slider_within_7px": float(np.mean(slider_errors <= 7) * 100),
                
                # 缺口指标
                "gap_mae": float(np.mean(gap_errors)),
                "gap_within_5px": float(np.mean(gap_errors <= 5) * 100),
                "gap_within_7px": float(np.mean(gap_errors <= 7) * 100),
                
                # 其他原有指标
                "median_distance_error": float(np.median(distance_errors)),
                "max_distance_error": float(np.max(distance_errors)),
                "hit_le_2px": float(np.mean(distance_errors <= 2) * 100),
                "hit_le_3px": float(np.mean(distance_errors <= 3) * 100),
                "hit_le_10px": float(np.mean(distance_errors <= 10) * 100),
                "avg_processing_time_ms": float(np.mean(processing_times)),
                "success_rate": float(len(results["predictions"]) / len(image_files) * 100)
            }
            
            # 生成统计图
            self._generate_statistics_plot(distance_errors, output_dir / "statistics.png", site_name)
        
        # 保存结果JSON
        json_path = output_dir / "evaluation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印结果
        print(f"\n✅ {site_name} Results:")
        print(f"   📊 Core Metrics:")
        print(f"      - Both within 5px: {results['metrics'].get('both_within_5px', 0):.1f}%")
        print(f"      - Both within 7px: {results['metrics'].get('both_within_7px', 0):.1f}%")
        print(f"   📏 Distance Metrics:")
        print(f"      - Distance MAE: {results['metrics'].get('distance_mae', 0):.2f}px")
        print(f"      - Distance within 5px: {results['metrics'].get('distance_within_5px', 0):.1f}%")
        print(f"      - Distance within 7px: {results['metrics'].get('distance_within_7px', 0):.1f}%")
        print(f"   🎯 Slider Metrics:")
        print(f"      - Slider MAE: {results['metrics'].get('slider_mae', 0):.2f}px")
        print(f"      - Slider within 5px: {results['metrics'].get('slider_within_5px', 0):.1f}%")
        print(f"      - Slider within 7px: {results['metrics'].get('slider_within_7px', 0):.1f}%")
        print(f"   🔍 Gap Metrics:")
        print(f"      - Gap MAE: {results['metrics'].get('gap_mae', 0):.2f}px")
        print(f"      - Gap within 5px: {results['metrics'].get('gap_within_5px', 0):.1f}%")
        print(f"      - Gap within 7px: {results['metrics'].get('gap_within_7px', 0):.1f}%")
        print(f"   💾 Results saved to: {output_dir}")
        
        return results
    
    def _visualize_prediction(self, img_path: Path, prediction: Dict, gt_info: Dict, output_path: Path):
        """生成单张图片的可视化结果"""
        try:
            # 读取图片
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 创建图形
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.imshow(image)
            
            # 坐标
            gt_gap_x = gt_info['gap_x']
            gt_gap_y = gt_info['gap_y']
            gt_slider_x = gt_info['slider_x']
            gt_slider_y = gt_info['slider_y']
            
            pred_gap_x = prediction['gap_x']
            pred_gap_y = prediction['gap_y']
            pred_slider_x = prediction['slider_x']
            pred_slider_y = prediction['slider_y']
            
            # 获取置信度
            confidence = prediction.get('confidence', 0.0)
            
            # 标注大小 - 根据置信度调整
            base_circle_size = 8
            pred_circle_size = base_circle_size * (0.7 + 0.3 * confidence)  # 置信度越高，圆圈越大
            
            # 根据置信度选择颜色
            if confidence >= 0.9:
                pred_gap_color = 'darkred'
                pred_slider_color = 'darkblue'
                conf_color = 'darkgreen'
            elif confidence >= 0.7:
                pred_gap_color = 'red'
                pred_slider_color = 'blue'
                conf_color = 'orange'
            else:
                pred_gap_color = 'lightcoral'
                pred_slider_color = 'lightblue'
                conf_color = 'red'
            
            # GT标注 - 绿色实心
            gt_gap_circle = patches.Circle((gt_gap_x, gt_gap_y), base_circle_size, 
                                          linewidth=2, edgecolor='green', 
                                          facecolor='green', alpha=0.3)
            ax.add_patch(gt_gap_circle)
            ax.text(gt_gap_x, gt_gap_y - 18, 'GT Gap', 
                   color='green', fontsize=10, fontweight='bold',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            gt_slider_circle = patches.Circle((gt_slider_x, gt_slider_y), base_circle_size,
                                             linewidth=2, edgecolor='green',
                                             facecolor='green', alpha=0.3)
            ax.add_patch(gt_slider_circle)
            ax.text(gt_slider_x, gt_slider_y - 18, 'GT Slider',
                   color='green', fontsize=10, fontweight='bold',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # 预测标注 - 根据置信度调整样式
            pred_gap_circle = patches.Circle((pred_gap_x, pred_gap_y), pred_circle_size, 
                                            linewidth=2 + confidence, edgecolor=pred_gap_color, 
                                            facecolor='none', linestyle='--',
                                            alpha=0.5 + 0.5 * confidence)
            ax.add_patch(pred_gap_circle)
            
            # Gap预测标签带置信度
            gap_label = f'Pred Gap\n(conf: {confidence:.2f})'
            ax.text(pred_gap_x, pred_gap_y + 18, gap_label,
                   color=pred_gap_color, fontsize=9, fontweight='bold',
                   ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            pred_slider_circle = patches.Circle((pred_slider_x, pred_slider_y), pred_circle_size,
                                               linewidth=2 + confidence, edgecolor=pred_slider_color,
                                               facecolor='none', linestyle='--',
                                               alpha=0.5 + 0.5 * confidence)
            ax.add_patch(pred_slider_circle)
            
            # Slider预测标签带置信度
            slider_label = f'Pred Slider\n(conf: {confidence:.2f})'
            ax.text(pred_slider_x, pred_slider_y + 18, slider_label,
                   color=pred_slider_color, fontsize=9, fontweight='bold',
                   ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # 添加连接线
            # GT连接线
            ax.plot([gt_slider_x, gt_gap_x], [gt_slider_y, gt_gap_y], 
                   'g-', linewidth=1.5, alpha=0.5, label='GT Distance')
            
            # 预测连接线
            ax.plot([pred_slider_x, pred_gap_x], [pred_slider_y, pred_gap_y], 
                   'r--', linewidth=1.5, alpha=0.5, label='Pred Distance')
            
            # 计算误差
            gt_distance = gt_gap_x - gt_slider_x
            pred_distance = prediction['sliding_distance']
            error = abs(pred_distance - gt_distance)
            
            # 计算坐标误差
            gap_error = np.sqrt((pred_gap_x - gt_gap_x)**2 + (pred_gap_y - gt_gap_y)**2)
            slider_error = np.sqrt((pred_slider_x - gt_slider_x)**2 + (pred_slider_y - gt_slider_y)**2)
            
            # 标题 - 包含更多信息
            title = (f"GT Distance: {gt_distance}px | Pred: {pred_distance:.1f}px | "
                    f"Error: {error:.1f}px | Confidence: {confidence:.3f}")
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # 添加详细信息框
            info_text = (f"Gap Error: {gap_error:.1f}px\n"
                        f"Slider Error: {slider_error:.1f}px\n"
                        f"Distance Error: {error:.1f}px\n"
                        f"Confidence: {confidence:.3f}")
            
            # 根据置信度选择信息框背景色
            if confidence >= 0.9:
                info_bg_color = '#e8f5e9'  # 浅绿色
            elif confidence >= 0.7:
                info_bg_color = '#fff3e0'  # 浅橙色
            else:
                info_bg_color = '#ffebee'  # 浅红色
            
            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor=info_bg_color, 
                            edgecolor=conf_color,
                            linewidth=2,
                            alpha=0.9))
            
            # 添加置信度条
            conf_bar_width = 100
            conf_bar_height = 10
            conf_bar_x = image.shape[1] - conf_bar_width - 20
            conf_bar_y = 20
            
            # 背景条
            bg_rect = patches.Rectangle((conf_bar_x, conf_bar_y), conf_bar_width, conf_bar_height,
                                       linewidth=1, edgecolor='black', facecolor='lightgray')
            ax.add_patch(bg_rect)
            
            # 置信度条
            conf_rect = patches.Rectangle((conf_bar_x, conf_bar_y), conf_bar_width * confidence, conf_bar_height,
                                        linewidth=0, facecolor=conf_color)
            ax.add_patch(conf_rect)
            
            # 置信度文字
            ax.text(conf_bar_x + conf_bar_width/2, conf_bar_y - 5, f'Confidence: {confidence:.1%}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.axis('off')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
            
            # 保存
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def _generate_statistics_plot(self, errors: np.ndarray, output_path: Path, site_name: str):
        """生成统计图表"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 误差分布直方图
        ax = axes[0]
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(errors), color='red', linestyle='--', label=f'MAE: {np.mean(errors):.2f}px')
        ax.axvline(x=5, color='green', linestyle='--', label='5px threshold')
        ax.axvline(x=7, color='orange', linestyle='--', label='7px threshold')
        ax.set_xlabel('Distance Error (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 精度柱状图
        ax = axes[1]
        thresholds = [2, 3, 5, 7, 10]
        accuracies = [np.mean(errors <= t) * 100 for t in thresholds]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA500', '#96CEB4']
        bars = ax.bar([f'≤{t}px' for t in thresholds], accuracies, color=colors)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', fontsize=10)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy at Different Thresholds')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. 累积分布
        ax = axes[2]
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax.plot(sorted_errors, cumulative, linewidth=2)
        ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95%')
        ax.axvline(x=5, color='green', linestyle='--', alpha=0.5, label='5px')
        ax.axvline(x=7, color='orange', linestyle='--', alpha=0.5, label='7px')
        ax.set_xlabel('Distance Error (pixels)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Cumulative Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(20, max(sorted_errors)))
        
        plt.suptitle(f'{site_name} - Statistical Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_charts(self, all_results: Dict, output_dir: Path):
        """生成多站点对比图表"""
        print("\n📊 Generating comparison charts...")
        
        # 准备数据
        sites = list(all_results.keys())
        metrics_data = {}
        
        # 要对比的指标
        comparison_metrics = [
            ('both_within_5px', 'Both within 5px'),
            ('both_within_7px', 'Both within 7px'),
            ('distance_within_5px', 'Distance within 5px'),
            ('distance_within_7px', 'Distance within 7px'),
            ('slider_within_5px', 'Slider within 5px'),
            ('slider_within_7px', 'Slider within 7px'),
            ('gap_within_5px', 'Gap within 5px'),
            ('gap_within_7px', 'Gap within 7px')
        ]
        
        mae_metrics = [
            ('distance_mae', 'Distance MAE'),
            ('slider_mae', 'Slider MAE'),
            ('gap_mae', 'Gap MAE')
        ]
        
        # 收集数据
        for metric_key, _ in comparison_metrics + mae_metrics:
            metrics_data[metric_key] = []
            for site in sites:
                value = all_results[site]['metrics'].get(metric_key, 0)
                metrics_data[metric_key].append(value)
        
        # 创建对比图
        fig = plt.figure(figsize=(18, 10))
        
        # 1. 精度对比柱状图
        ax1 = plt.subplot(2, 3, 1)
        x = np.arange(len(sites))
        width = 0.35
        
        # 5px精度对比
        vals_5px = metrics_data['both_within_5px']
        vals_7px = metrics_data['both_within_7px']
        
        bars1 = ax1.bar(x - width/2, vals_5px, width, label='Within 5px', color='#45B7D1')
        bars2 = ax1.bar(x + width/2, vals_7px, width, label='Within 7px', color='#96CEB4')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Site')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Comparison for Both Within Threshold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sites)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 距离精度对比
        ax2 = plt.subplot(2, 3, 2)
        bars1 = ax2.bar(x - width/2, metrics_data['distance_within_5px'], width, 
                       label='Within 5px', color='#FF6B6B')
        bars2 = ax2.bar(x + width/2, metrics_data['distance_within_7px'], width,
                       label='Within 7px', color='#FFA500')
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Site')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Sliding Distance Accuracy Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sites)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. MAE对比
        ax3 = plt.subplot(2, 3, 3)
        x2 = np.arange(len(mae_metrics))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, site in enumerate(sites):
            values = [metrics_data[m[0]][i] for m in mae_metrics]
            offset = (i - len(sites)/2 + 0.5) * width
            bars = ax3.bar(x2 + offset, values, width, label=site, alpha=0.8)
            
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('MAE (pixels)')
        ax3.set_title('MAE Comparison')
        ax3.set_xticks(x2)
        ax3.set_xticklabels([m[1] for m in mae_metrics])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 滑块精度对比
        ax4 = plt.subplot(2, 3, 4)
        bars1 = ax4.bar(x - width/2, metrics_data['slider_within_5px'], width,
                       label='Within 5px', color='#4ECDC4')
        bars2 = ax4.bar(x + width/2, metrics_data['slider_within_7px'], width,
                       label='Within 7px', color='#96CEB4')
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax4.set_xlabel('Site')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Slider Coordinate Accuracy Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(sites)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 缺口精度对比
        ax5 = plt.subplot(2, 3, 5)
        bars1 = ax5.bar(x - width/2, metrics_data['gap_within_5px'], width,
                       label='Within 5px', color='#45B7D1')
        bars2 = ax5.bar(x + width/2, metrics_data['gap_within_7px'], width,
                       label='Within 7px', color='#FFA500')
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax5.set_xlabel('Site')
        ax5.set_ylabel('Accuracy (%)')
        ax5.set_title('Gap Coordinate Accuracy Comparison')
        ax5.set_xticks(x)
        ax5.set_xticklabels(sites)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 综合指标雷达图
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        
        # 准备雷达图数据
        categories = ['Both 5px', 'Both 7px', 'Dist 5px', 'Dist 7px', 'Slider 5px', 'Gap 5px']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, site in enumerate(sites):
            values = [
                metrics_data['both_within_5px'][i],
                metrics_data['both_within_7px'][i],
                metrics_data['distance_within_5px'][i],
                metrics_data['distance_within_7px'][i],
                metrics_data['slider_within_5px'][i],
                metrics_data['gap_within_5px'][i]
            ]
            values += values[:1]
            
            ax6.plot(angles, values, 'o-', linewidth=2, label=site, alpha=0.7)
            ax6.fill(angles, values, alpha=0.25)
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories, size=8)
        ax6.set_ylim(0, 100)
        ax6.set_title('Comprehensive Performance Radar Chart', y=1.08)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax6.grid(True)
        
        plt.suptitle('Multi-Site Performance Comparison Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        comparison_path = output_dir / "comparison_charts.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Comparison charts saved to: {comparison_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Real CAPTCHA Dataset')
    parser.add_argument('--model', type=str, 
                       default=None,
                       help='Model path (default: use version from config)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu', None],
                       help='Device to use for inference (cuda/cpu, default: auto-detect)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for GPU inference (default: 32)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum images per site')
    parser.add_argument('--auto', action='store_true',
                       help='Automatically analyze all folders without prompting')
    parser.add_argument('--skip', nargs='+', default=[],
                       help='List of folder names to skip (e.g., --skip site1 test)')
    parser.add_argument('--only', nargs='+', default=[],
                       help='Only analyze specified folders (e.g., --only site1 site2)')
    
    args = parser.parse_args()
    
    print("🚀 Real CAPTCHA Evaluation")
    print("="*60)
    
    # 显示GPU信息
    if torch.cuda.is_available():
        print(f"📊 GPU Information:")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print("="*60)
    
    # 创建生成器，传入device参数
    generator = RealCaptchaResultGenerator(
        model_path=args.model,
        device=args.device  # 传入设备参数
    )
    
    # 获取所有annotated子文件夹
    annotated_dir = Path("data/real_captchas/annotated")
    all_results = {}
    
    if annotated_dir.exists():
        # Get all subdirectories
        all_site_dirs = [d for d in sorted(annotated_dir.iterdir()) if d.is_dir()]
        
        # Apply command-line filters
        if args.only:
            # Filter to only specified folders
            all_site_dirs = [d for d in all_site_dirs if d.name in args.only]
            if not all_site_dirs:
                print(f"\n⚠️ No folders found matching: {', '.join(args.only)}")
                return
        
        if args.skip:
            # Remove skipped folders
            all_site_dirs = [d for d in all_site_dirs if d.name not in args.skip]
        
        # Show available folders
        print("\n" + "="*60)
        print("Available folders for analysis:")
        print("="*60)
        for i, site_dir in enumerate(all_site_dirs, 1):
            # Count images in folder
            img_count = len(list(site_dir.glob("*.png"))) + len(list(site_dir.glob("*.jpg")))
            skip_note = " [SKIPPED]" if site_dir.name in args.skip else ""
            print(f"{i}. {site_dir.name} ({img_count} images){skip_note}")
        print("="*60)
        
        selected_dirs = []
        
        # Check if auto mode or command-line selection
        if args.auto or args.only:
            # Auto-select folders based on arguments
            selected_dirs = all_site_dirs
            if args.auto:
                print("\n✓ Auto mode: analyzing all folders")
            else:
                print(f"\n✓ Analyzing specified folders: {', '.join(args.only)}")
            
            for site_dir in selected_dirs:
                img_count = len(list(site_dir.glob("*.png"))) + len(list(site_dir.glob("*.jpg")))
                print(f"  • {site_dir.name} ({img_count} images)")
        else:
            # Interactive selection
            print("\nFolder selection options:")
            print("  A - Analyze all folders")
            print("  S - Select folders individually")
            print("  Q - Quit")
            
            batch_choice = input("\nYour choice [A/s/q]: ").strip().lower()
            
            if batch_choice in ['', 'a', 'all']:
                # Select all folders
                selected_dirs = all_site_dirs
                print("\n✓ Selected all folders for analysis")
                for site_dir in selected_dirs:
                    img_count = len(list(site_dir.glob("*.png"))) + len(list(site_dir.glob("*.jpg")))
                    print(f"  • {site_dir.name} ({img_count} images)")
            elif batch_choice in ['q', 'quit']:
                print("\nExiting...")
                selected_dirs = []
            else:
                # Ask user which folders to analyze individually
                print("\nSelecting folders individually...")
                for site_dir in all_site_dirs:
                    img_count = len(list(site_dir.glob("*.png"))) + len(list(site_dir.glob("*.jpg")))
                    
                    while True:
                        response = input(f"\nAnalyze folder '{site_dir.name}' ({img_count} images)? [Y/n/q]: ").strip().lower()
                        
                        if response in ['', 'y', 'yes']:
                            selected_dirs.append(site_dir)
                            print(f"  ✓ Will analyze: {site_dir.name}")
                            break
                        elif response in ['n', 'no']:
                            print(f"  ✗ Skipping: {site_dir.name}")
                            break
                        elif response in ['q', 'quit']:
                            print("\nStopping folder selection.")
                            break
                        else:
                            print("  Please enter Y (yes), N (no), or Q (quit)")
                    
                    if response in ['q', 'quit']:
                        break
        
        # Process selected folders
        if selected_dirs:
            print("\n" + "="*60)
            print(f"Processing {len(selected_dirs)} selected folder(s)...")
            print("="*60)
            
            for site_dir in selected_dirs:
                site_name = site_dir.name
                output_dir = Path(f"results/{generator.version}") / site_name
                # 使用批量推理
                result = generator.evaluate_site(
                    site_name, site_dir, output_dir, 
                    max_images=args.max_images,
                    batch_size=args.batch_size
                )
                if result:
                    all_results[site_name] = result
        else:
            print("\nNo folders selected for analysis.")
    
    # 生成对比图表
    if len(all_results) > 1:
        generator.generate_comparison_charts(all_results, Path(f"results/{generator.version}"))
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()