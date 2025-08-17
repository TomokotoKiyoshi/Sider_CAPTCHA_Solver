"""
生成v1.1.0版本的结果目录结构和评估报告
类似于v1.0.3的结构
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sider_captcha_solver.predictor import CaptchaPredictor


class ResultsGeneratorV110:
    """v1.1.0结果生成器"""
    
    def __init__(self, model_path: str = "src/checkpoints/1.1.0/best_model.pth"):
        """
        初始化结果生成器
        
        Args:
            model_path: 模型路径
        """
        self.model_path = Path(model_path)
        self.results_dir = Path("results/1.1.0")
        
        # 创建目录结构
        self.real_captchas_dir = self.results_dir / "real_captchas"
        self.test_dataset_dir = self.results_dir / "test_dataset"
        
        # 创建可视化子目录
        self.real_vis_dir = self.real_captchas_dir / "visualizations"
        self.test_vis_dir = self.test_dataset_dir / "visualizations"
        
        # 创建所有目录
        for dir_path in [self.real_vis_dir, self.test_vis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Created directory structure at: {self.results_dir}")
        
        # 加载模型
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            self.predictor = CaptchaPredictor(
                model_path=str(self.model_path),
                device='auto'
            )
            print(f"✅ Model loaded: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def evaluate_real_captchas(self):
        """评估真实验证码数据集"""
        print("\n📊 Evaluating real CAPTCHAs...")
        
        # 数据目录
        data_dir = Path("data/real_captchas/annotated")
        if not data_dir.exists():
            print(f"❌ Directory not found: {data_dir}")
            return
        
        # 获取所有图片
        image_files = sorted(data_dir.glob("*.png"))[:100]  # 限制前100张
        
        results = {
            "model_version": "1.1.0",
            "dataset": "real_captchas",
            "timestamp": datetime.now().isoformat(),
            "total_images": len(image_files),
            "predictions": [],
            "metrics": {}
        }
        
        errors = []
        processing_times = []
        
        # 评估每张图片
        for idx, img_path in enumerate(tqdm(image_files, desc="Processing")):
            try:
                # 解析文件名获取真实坐标
                # 格式: Pic0001_Bgx112Bgy97_Sdx32Sdy98_hash.png
                import re
                pattern = r"Pic(\d+)_Bgx(\d+)Bgy(\d+)_Sdx(\d+)Sdy(\d+)_(\w+)\.png"
                match = re.match(pattern, img_path.name)
                
                if not match:
                    continue
                
                _, gap_x_true, gap_y_true, slider_x_true, slider_y_true, _ = match.groups()
                gap_x_true = int(gap_x_true)
                gap_y_true = int(gap_y_true)
                slider_x_true = int(slider_x_true)
                slider_y_true = int(slider_y_true)
                
                # 预测
                pred = self.predictor.predict(str(img_path))
                
                if pred['success']:
                    # 计算误差
                    gap_error = np.sqrt((pred['gap_x'] - gap_x_true)**2 + 
                                      (pred['gap_y'] - gap_y_true)**2)
                    slider_error = np.sqrt((pred['slider_x'] - slider_x_true)**2 + 
                                         (pred['slider_y'] - slider_y_true)**2)
                    distance_error = abs(pred['sliding_distance'] - (gap_x_true - slider_x_true))
                    
                    # 记录结果
                    prediction = {
                        "image": img_path.name,
                        "ground_truth": {
                            "gap_x": gap_x_true,
                            "gap_y": gap_y_true,
                            "slider_x": slider_x_true,
                            "slider_y": slider_y_true,
                            "sliding_distance": gap_x_true - slider_x_true
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
                            "gap_error": gap_error,
                            "slider_error": slider_error,
                            "distance_error": distance_error
                        },
                        "processing_time_ms": pred['processing_time_ms']
                    }
                    
                    results["predictions"].append(prediction)
                    errors.append(distance_error)
                    processing_times.append(pred['processing_time_ms'])
                    
                    # 生成可视化（每张都保存，最多100张）
                    if idx < 100:  # 保存前100张的可视化
                        self._visualize_prediction(
                            img_path, pred, 
                            self.real_vis_dir / f"sample_{idx:04d}.png",
                            gt_gap_x=gap_x_true,
                            gt_gap_y=gap_y_true,
                            gt_slider_x=slider_x_true,
                            gt_slider_y=slider_y_true
                        )
            
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
        
        # 计算统计指标
        if errors:
            errors = np.array(errors)
            results["metrics"] = {
                "mae": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "max_error": float(np.max(errors)),
                "hit_le_2px": float(np.mean(errors <= 2) * 100),
                "hit_le_3px": float(np.mean(errors <= 3) * 100),
                "hit_le_5px": float(np.mean(errors <= 5) * 100),
                "hit_le_10px": float(np.mean(errors <= 10) * 100),
                "avg_processing_time_ms": float(np.mean(processing_times)),
                "success_rate": float(len(results["predictions"]) / len(image_files) * 100)
            }
        
        # 保存结果
        output_path = self.real_captchas_dir / "evaluation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Real CAPTCHA results saved to: {output_path}")
        print(f"   - MAE: {results['metrics'].get('mae', 0):.2f} px")
        print(f"   - 5px Accuracy: {results['metrics'].get('hit_le_5px', 0):.1f}%")
        
        return results
    
    def evaluate_test_dataset(self):
        """评估测试数据集"""
        print("\n📊 Evaluating test dataset...")
        
        # 数据目录
        data_dir = Path("data/captchas")
        if not data_dir.exists():
            print(f"❌ Directory not found: {data_dir}")
            return
        
        # 获取测试图片
        image_files = sorted(data_dir.glob("*.png"))[:1000]  # 限制前1000张
        
        results = {
            "model_version": "1.1.0",
            "dataset": "test_dataset",
            "timestamp": datetime.now().isoformat(),
            "total_images": len(image_files),
            "predictions": [],
            "metrics": {}
        }
        
        errors = []
        processing_times = []
        
        # 评估每张图片（处理前100张）
        for idx, img_path in enumerate(tqdm(image_files[:100], desc="Processing")):
            try:
                # 解析文件名
                import re
                pattern = r"Pic(\d+)_Bgx(\d+)Bgy(\d+)_Sdx(\d+)Sdy(\d+)_(\w+)\.png"
                match = re.match(pattern, img_path.name)
                
                if not match:
                    continue
                
                _, gap_x_true, gap_y_true, slider_x_true, slider_y_true, _ = match.groups()
                gap_x_true = int(gap_x_true)
                gap_y_true = int(gap_y_true)
                slider_x_true = int(slider_x_true)
                slider_y_true = int(slider_y_true)
                
                # 预测
                pred = self.predictor.predict(str(img_path))
                
                if pred['success']:
                    # 计算误差
                    distance_error = abs(pred['sliding_distance'] - (gap_x_true - slider_x_true))
                    
                    # 记录简化结果
                    prediction = {
                        "image": img_path.name,
                        "distance_error": distance_error,
                        "processing_time_ms": pred['processing_time_ms']
                    }
                    
                    results["predictions"].append(prediction)
                    errors.append(distance_error)
                    processing_times.append(pred['processing_time_ms'])
                    
                    # 生成可视化（每张都保存，最多100张）
                    if idx < 100:
                        self._visualize_prediction(
                            img_path, pred,
                            self.test_vis_dir / f"sample_{idx:04d}.png",
                            gt_gap_x=gap_x_true,
                            gt_gap_y=gap_y_true,
                            gt_slider_x=slider_x_true,
                            gt_slider_y=slider_y_true
                        )
            
            except Exception as e:
                continue
        
        # 计算统计指标
        if errors:
            errors = np.array(errors)
            results["metrics"] = {
                "mae": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "max_error": float(np.max(errors)),
                "hit_le_2px": float(np.mean(errors <= 2) * 100),
                "hit_le_3px": float(np.mean(errors <= 3) * 100),
                "hit_le_5px": float(np.mean(errors <= 5) * 100),
                "hit_le_10px": float(np.mean(errors <= 10) * 100),
                "avg_processing_time_ms": float(np.mean(processing_times))
            }
        
        # 保存结果
        output_path = self.test_dataset_dir / "evaluation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Test dataset results saved to: {output_path}")
        print(f"   - MAE: {results['metrics'].get('mae', 0):.2f} px")
        print(f"   - 5px Accuracy: {results['metrics'].get('hit_le_5px', 0):.1f}%")
        
        return results
    
    def _visualize_prediction(self, img_path: Path, prediction: Dict, output_path: Path, 
                              gt_gap_x: int = None, gt_gap_y: int = None,
                              gt_slider_x: int = None, gt_slider_y: int = None):
        """生成预测可视化 - 包含GT和Pred的对比"""
        try:
            # 读取图片
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 创建图形
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.imshow(image)
            
            # 提取预测坐标
            pred_gap_x = prediction['gap_x']
            pred_gap_y = prediction['gap_y']
            pred_slider_x = prediction['slider_x']
            pred_slider_y = prediction['slider_y']
            
            # 标注大小设置
            circle_size = 5  # 圆圈半径（再缩小一半）
            text_offset = 10  # 文字偏移
            
            # 如果提供了GT坐标，绘制GT标注（实心绿色）
            if gt_gap_x is not None and gt_gap_y is not None:
                # GT Gap - 实心绿色圆圈
                gt_gap_circle = patches.Circle((gt_gap_x, gt_gap_y), circle_size, 
                                              linewidth=2, edgecolor='green', 
                                              facecolor='green', alpha=0.3)
                ax.add_patch(gt_gap_circle)
                ax.text(gt_gap_x, gt_gap_y - text_offset, 'GT Gap', 
                       color='green', fontsize=10, fontweight='bold',
                       ha='center', va='bottom')
            
            if gt_slider_x is not None and gt_slider_y is not None:
                # GT Slider - 实心绿色圆圈
                gt_slider_circle = patches.Circle((gt_slider_x, gt_slider_y), circle_size,
                                                 linewidth=2, edgecolor='green',
                                                 facecolor='green', alpha=0.3)
                ax.add_patch(gt_slider_circle)
                ax.text(gt_slider_x, gt_slider_y - text_offset, 'GT Slider',
                       color='green', fontsize=10, fontweight='bold',
                       ha='center', va='bottom')
            
            # 绘制预测结果
            # Pred Gap - 红色虚线空心圆圈
            pred_gap_circle = patches.Circle((pred_gap_x, pred_gap_y), circle_size, 
                                            linewidth=2, edgecolor='red', 
                                            facecolor='none', linestyle='--')
            ax.add_patch(pred_gap_circle)
            ax.text(pred_gap_x, pred_gap_y + text_offset, 'Pred Gap',
                   color='red', fontsize=10, fontweight='bold',
                   ha='center', va='top')
            
            # Pred Slider - 蓝色虚线空心圆圈
            pred_slider_circle = patches.Circle((pred_slider_x, pred_slider_y), circle_size,
                                               linewidth=2, edgecolor='blue',
                                               facecolor='none', linestyle='--')
            ax.add_patch(pred_slider_circle)
            ax.text(pred_slider_x, pred_slider_y + text_offset, 'Pred Slider',
                   color='blue', fontsize=10, fontweight='bold',
                   ha='center', va='top')
            
            # 不绘制箭头
            
            # 计算误差
            distance = prediction['sliding_distance']
            confidence = prediction['confidence']
            
            # 如果有GT，计算误差
            if gt_gap_x is not None and gt_slider_x is not None:
                gt_distance = gt_gap_x - gt_slider_x
                error = abs(distance - gt_distance)
                title = f"Pred Distance: {distance:.1f}px | GT Distance: {gt_distance}px | Error: {error:.1f}px | Conf: {confidence:.3f}"
            else:
                title = f"Distance: {distance:.1f}px | Confidence: {confidence:.3f}"
            
            ax.set_title(title, fontsize=11)
            ax.axis('off')
            
            # 保存
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def generate_summary_report(self, real_results: Dict, test_results: Dict):
        """生成总结报告"""
        summary = {
            "model_version": "1.1.0",
            "timestamp": datetime.now().isoformat(),
            "real_captchas": {
                "total_evaluated": real_results.get("total_images", 0),
                "metrics": real_results.get("metrics", {})
            },
            "test_dataset": {
                "total_evaluated": test_results.get("total_images", 0),
                "metrics": test_results.get("metrics", {})
            },
            "comparison": {
                "real_vs_test_mae_diff": abs(
                    real_results.get("metrics", {}).get("mae", 0) - 
                    test_results.get("metrics", {}).get("mae", 0)
                ),
                "real_vs_test_5px_diff": abs(
                    real_results.get("metrics", {}).get("hit_le_5px", 0) - 
                    test_results.get("metrics", {}).get("hit_le_5px", 0)
                )
            },
            "model_info": {
                "checkpoint": str(self.model_path),
                "device": str(self.predictor.device) if self.predictor else "unknown"
            }
        }
        
        # 保存总结报告
        output_path = self.results_dir / "summary_report.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Summary report saved to: {output_path}")
        
        # 打印总结
        print("\n" + "="*60)
        print("EVALUATION SUMMARY - v1.1.0")
        print("="*60)
        print(f"Real CAPTCHAs:")
        print(f"  - MAE: {real_results.get('metrics', {}).get('mae', 0):.2f} px")
        print(f"  - 5px Accuracy: {real_results.get('metrics', {}).get('hit_le_5px', 0):.1f}%")
        print(f"\nTest Dataset:")
        print(f"  - MAE: {test_results.get('metrics', {}).get('mae', 0):.2f} px")
        print(f"  - 5px Accuracy: {test_results.get('metrics', {}).get('hit_le_5px', 0):.1f}%")
        print("="*60)
        
        return summary


def main():
    """主函数"""
    print("🚀 Generating v1.1.0 Results Structure")
    print("="*60)
    
    # 创建生成器
    generator = ResultsGeneratorV110()
    
    # 评估真实验证码
    real_results = generator.evaluate_real_captchas()
    
    # 评估测试数据集
    test_results = generator.evaluate_test_dataset()
    
    # 生成总结报告
    if real_results and test_results:
        generator.generate_summary_report(real_results, test_results)
    
    print("\n✅ All results generated successfully!")
    print(f"📁 Results location: results/1.1.0/")


if __name__ == "__main__":
    main()