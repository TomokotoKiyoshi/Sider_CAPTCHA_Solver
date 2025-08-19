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

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sider_captcha_solver.predictor import CaptchaPredictor


class RealCaptchaResultGenerator:
    """真实验证码结果生成器"""
    
    def __init__(self, model_path: str = "src/checkpoints/1.1.0/best_model.pth"):
        """
        初始化结果生成器
        
        Args:
            model_path: 模型路径
        """
        self.model_path = Path(model_path)
        
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
    
    def evaluate_site(self, site_name: str, data_dir: Path, output_dir: Path, max_images: int = None):
        """
        评估单个站点的数据
        
        Args:
            site_name: 站点名称 (site1或site2)
            data_dir: 数据目录
            output_dir: 输出目录
            max_images: 最大处理图片数
        """
        print(f"\n📊 Evaluating {site_name}...")
        print(f"   Data directory: {data_dir}")
        
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
            "model_version": "1.1.0",
            "timestamp": datetime.now().isoformat(),
            "total_images": len(image_files),
            "predictions": [],
            "metrics": {}
        }
        
        # 错误统计
        distance_errors = []
        gap_errors = []
        slider_errors = []
        processing_times = []
        
        # 处理每张图片
        for idx, img_path in enumerate(tqdm(image_files, desc=f"Processing {site_name}")):
            try:
                # 解析文件名获取GT
                gt_info = self.parse_filename(img_path.name)
                if not gt_info:
                    print(f"⚠️ Cannot parse filename: {img_path.name}")
                    continue
                
                # 预测
                pred = self.predictor.predict(str(img_path))
                
                if pred['success']:
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
                            "distance_error": float(distance_error)
                        },
                        "processing_time_ms": pred['processing_time_ms']
                    }
                    
                    results["predictions"].append(prediction)
                    distance_errors.append(distance_error)
                    gap_errors.append(gap_error)
                    slider_errors.append(slider_error)
                    processing_times.append(pred['processing_time_ms'])
                    
                    # 生成可视化
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
            
            results["metrics"] = {
                "distance_mae": float(np.mean(distance_errors)),
                "gap_mae": float(np.mean(gap_errors)),
                "slider_mae": float(np.mean(slider_errors)),
                "median_distance_error": float(np.median(distance_errors)),
                "max_distance_error": float(np.max(distance_errors)),
                "hit_le_2px": float(np.mean(distance_errors <= 2) * 100),
                "hit_le_3px": float(np.mean(distance_errors <= 3) * 100),
                "hit_le_5px": float(np.mean(distance_errors <= 5) * 100),
                "hit_le_7px": float(np.mean(distance_errors <= 7) * 100),
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
        print(f"   - MAE: {results['metrics'].get('distance_mae', 0):.2f} px")
        print(f"   - 5px Accuracy: {results['metrics'].get('hit_le_5px', 0):.1f}%")
        print(f"   - 7px Accuracy: {results['metrics'].get('hit_le_7px', 0):.1f}%")
        print(f"   - Results saved to: {output_dir}")
        
        return results
    
    def _visualize_prediction(self, img_path: Path, prediction: Dict, gt_info: Dict, output_path: Path):
        """生成单张图片的可视化结果"""
        try:
            # 读取图片
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 创建图形
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
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
            
            # 标注大小
            circle_size = 8
            
            # GT标注 - 绿色实心
            gt_gap_circle = patches.Circle((gt_gap_x, gt_gap_y), circle_size, 
                                          linewidth=2, edgecolor='green', 
                                          facecolor='green', alpha=0.3)
            ax.add_patch(gt_gap_circle)
            ax.text(gt_gap_x, gt_gap_y - 15, 'GT Gap', 
                   color='green', fontsize=10, fontweight='bold',
                   ha='center', va='bottom')
            
            gt_slider_circle = patches.Circle((gt_slider_x, gt_slider_y), circle_size,
                                             linewidth=2, edgecolor='green',
                                             facecolor='green', alpha=0.3)
            ax.add_patch(gt_slider_circle)
            ax.text(gt_slider_x, gt_slider_y - 15, 'GT Slider',
                   color='green', fontsize=10, fontweight='bold',
                   ha='center', va='bottom')
            
            # 预测标注 - 红色虚线
            pred_gap_circle = patches.Circle((pred_gap_x, pred_gap_y), circle_size, 
                                            linewidth=2, edgecolor='red', 
                                            facecolor='none', linestyle='--')
            ax.add_patch(pred_gap_circle)
            ax.text(pred_gap_x, pred_gap_y + 15, 'Pred Gap',
                   color='red', fontsize=10, fontweight='bold',
                   ha='center', va='top')
            
            pred_slider_circle = patches.Circle((pred_slider_x, pred_slider_y), circle_size,
                                               linewidth=2, edgecolor='blue',
                                               facecolor='none', linestyle='--')
            ax.add_patch(pred_slider_circle)
            ax.text(pred_slider_x, pred_slider_y + 15, 'Pred Slider',
                   color='blue', fontsize=10, fontweight='bold',
                   ha='center', va='top')
            
            # 计算误差
            gt_distance = gt_gap_x - gt_slider_x
            pred_distance = prediction['sliding_distance']
            error = abs(pred_distance - gt_distance)
            
            # 标题
            title = f"GT Distance: {gt_distance}px | Pred: {pred_distance:.1f}px | Error: {error:.1f}px"
            ax.set_title(title, fontsize=12)
            ax.axis('off')
            
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


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='评估真实验证码数据集')
    parser.add_argument('--site', type=str, choices=['site1', 'site2', 'both'], 
                       default='both', help='选择要评估的站点')
    parser.add_argument('--model', type=str, 
                       default='src/checkpoints/1.1.0/best_model.pth',
                       help='模型路径')
    parser.add_argument('--max-images', type=int, default=None,
                       help='每个站点最大处理图片数')
    
    args = parser.parse_args()
    
    print("🚀 Real CAPTCHA Evaluation")
    print("="*60)
    
    # 创建生成器
    generator = RealCaptchaResultGenerator(model_path=args.model)
    
    # 评估site1
    if args.site in ['site1', 'both']:
        site1_data = Path("data/real_captchas/annotated/site1")
        site1_output = Path("results/1.1.0/real_captchas/site1")
        generator.evaluate_site('site1', site1_data, site1_output, args.max_images)
    
    # 评估site2
    if args.site in ['site2', 'both']:
        site2_data = Path("data/real_captchas/annotated/site2")
        site2_output = Path("results/1.1.0/real_captchas/site2")
        generator.evaluate_site('site2', site2_data, site2_output, args.max_images)
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()