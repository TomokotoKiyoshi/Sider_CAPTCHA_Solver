"""
真实验证码数据集模型评估脚本
评估不同epoch模型在真实数据上的表现
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sider_captcha_solver.predictor import CaptchaPredictor


class RealCaptchaEvaluator:
    """真实验证码评估器"""
    
    def __init__(self, 
                 data_dir: str = "data/real_captchas/annotated",
                 checkpoint_dir: str = "src/checkpoints/1.1.0",
                 device: str = "auto"):
        """
        初始化评估器
        
        Args:
            data_dir: 真实验证码数据目录
            checkpoint_dir: 模型checkpoint目录
            device: 运行设备
        """
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        
        # 检查目录
        if not self.data_dir.exists():
            raise ValueError(f"数据目录不存在: {self.data_dir}")
        if not self.checkpoint_dir.exists():
            raise ValueError(f"模型目录不存在: {self.checkpoint_dir}")
        
        # 加载数据集
        self.image_files = self._load_dataset()
        print(f"✅ Loaded {len(self.image_files)} real CAPTCHA images")
        
        # 查找所有模型
        self.model_files = self._find_models()
        print(f"✅ Found {len(self.model_files)} model files")
    
    def _load_dataset(self) -> List[Dict]:
        """
        加载数据集并解析文件名中的坐标信息
        
        文件名格式: Pic0001_Bgx112Bgy97_Sdx32Sdy98_cb7cbd17.png
        """
        image_files = []
        
        # 定义文件名解析的正则表达式
        pattern = r"Pic(\d+)_Bgx(\d+)Bgy(\d+)_Sdx(\d+)Sdy(\d+)_(\w+)\.png"
        
        for file_path in sorted(self.data_dir.glob("*.png")):
            match = re.match(pattern, file_path.name)
            if match:
                pic_id, bg_x, bg_y, sd_x, sd_y, hash_val = match.groups()
                
                image_files.append({
                    'path': file_path,
                    'name': file_path.name,
                    'pic_id': int(pic_id),
                    'gap_x': int(bg_x),
                    'gap_y': int(bg_y),
                    'slider_x': int(sd_x),
                    'slider_y': int(sd_y),
                    'hash': hash_val,
                    'sliding_distance': int(bg_x) - int(sd_x)  # 真实滑动距离
                })
            else:
                print(f"⚠️ Cannot parse filename: {file_path.name}")
        
        return image_files
    
    def _find_models(self) -> List[Path]:
        """查找所有模型文件"""
        model_files = []
        
        # 查找所有.pth文件
        for pth_file in sorted(self.checkpoint_dir.glob("*.pth")):
            # 跳过优化器状态文件
            if "optimizer" not in pth_file.name.lower():
                model_files.append(pth_file)
        
        return model_files
    
    def evaluate_model(self, model_path: Path, threshold_px: int = 5) -> Dict:
        """
        评估单个模型
        
        Args:
            model_path: 模型文件路径
            threshold_px: 正确判定的像素阈值
        
        Returns:
            评估结果字典
        """
        print(f"\n📊 Evaluating model: {model_path.name}")
        
        # 加载模型
        try:
            predictor = CaptchaPredictor(
                model_path=str(model_path),
                device=self.device
            )
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
        
        # 评估指标
        results = {
            'model_name': model_path.name,
            'total_images': len(self.image_files),
            'predictions': [],
            'errors': [],
            'gap_errors': [],
            'slider_errors': [],
            'distance_errors': [],
            'processing_times': []
        }
        
        # 逐图评估
        for img_info in tqdm(self.image_files, desc="Evaluation Progress"):
            try:
                # 预测
                pred = predictor.predict(str(img_info['path']))
                
                if pred['success']:
                    # 计算误差
                    gap_error = np.sqrt(
                        (pred['gap_x'] - img_info['gap_x'])**2 + 
                        (pred['gap_y'] - img_info['gap_y'])**2
                    )
                    slider_error = np.sqrt(
                        (pred['slider_x'] - img_info['slider_x'])**2 + 
                        (pred['slider_y'] - img_info['slider_y'])**2
                    )
                    distance_error = abs(pred['sliding_distance'] - img_info['sliding_distance'])
                    
                    # 记录结果
                    results['predictions'].append({
                        'image': img_info['name'],
                        'gap_error': gap_error,
                        'slider_error': slider_error,
                        'distance_error': distance_error,
                        'gap_correct': gap_error <= threshold_px,
                        'slider_correct': slider_error <= threshold_px,
                        'distance_correct': distance_error <= threshold_px,
                        'confidence': pred['confidence'],
                        'processing_time': pred['processing_time_ms']
                    })
                    
                    results['gap_errors'].append(gap_error)
                    results['slider_errors'].append(slider_error)
                    results['distance_errors'].append(distance_error)
                    results['processing_times'].append(pred['processing_time_ms'])
                else:
                    results['errors'].append({
                        'image': img_info['name'],
                        'error': pred.get('error', 'Unknown error')
                    })
            
            except Exception as e:
                results['errors'].append({
                    'image': img_info['name'],
                    'error': str(e)
                })
        
        # 计算统计指标
        if results['gap_errors']:
            results['metrics'] = self._calculate_metrics(results, threshold_px)
        else:
            results['metrics'] = {'error': '所有预测失败'}
        
        return results
    
    def _calculate_metrics(self, results: Dict, threshold_px: int) -> Dict:
        """计算评估指标"""
        gap_errors = np.array(results['gap_errors'])
        slider_errors = np.array(results['slider_errors'])
        distance_errors = np.array(results['distance_errors'])
        
        metrics = {
            # 平均误差
            'gap_mae': np.mean(gap_errors),
            'slider_mae': np.mean(slider_errors),
            'distance_mae': np.mean(distance_errors),
            
            # 中位数误差
            'gap_median': np.median(gap_errors),
            'slider_median': np.median(slider_errors),
            'distance_median': np.median(distance_errors),
            
            # 最大误差
            'gap_max': np.max(gap_errors),
            'slider_max': np.max(slider_errors),
            'distance_max': np.max(distance_errors),
            
            # 正确率（阈值内）
            'gap_accuracy': np.mean(gap_errors <= threshold_px) * 100,
            'slider_accuracy': np.mean(slider_errors <= threshold_px) * 100,
            'distance_accuracy': np.mean(distance_errors <= threshold_px) * 100,
            
            # 不同阈值的正确率
            'hit_le_2px': np.mean(distance_errors <= 2) * 100,
            'hit_le_3px': np.mean(distance_errors <= 3) * 100,
            'hit_le_5px': np.mean(distance_errors <= 5) * 100,
            'hit_le_10px': np.mean(distance_errors <= 10) * 100,
            
            # 处理时间
            'avg_time_ms': np.mean(results['processing_times']),
            'median_time_ms': np.median(results['processing_times']),
            
            # 失败率
            'failure_rate': len(results['errors']) / results['total_images'] * 100
        }
        
        return metrics
    
    def evaluate_all_models(self, threshold_px: int = 5) -> pd.DataFrame:
        """
        评估所有模型并生成对比报告
        
        Args:
            threshold_px: 正确判定的像素阈值
        
        Returns:
            包含所有模型评估结果的DataFrame
        """
        all_results = []
        
        for model_path in self.model_files:
            result = self.evaluate_model(model_path, threshold_px)
            if result and 'metrics' in result:
                # 提取epoch编号
                epoch_match = re.search(r'epoch_(\d+)', model_path.name)
                epoch = int(epoch_match.group(1)) if epoch_match else -1
                
                # 构建结果行
                row = {
                    'model': model_path.name,
                    'epoch': epoch,
                    **result['metrics']
                }
                all_results.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(all_results)
        
        # 按epoch排序
        if 'epoch' in df.columns:
            df = df.sort_values('epoch')
        
        return df
    
    def generate_report(self, df: pd.DataFrame, output_dir: str = "outputs/real_captchas_report_1.1.0"):
        """
        生成评估报告
        
        Args:
            df: 评估结果DataFrame
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存CSV报告
        csv_path = output_dir / f"model_evaluation_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ CSV report saved: {csv_path}")
        
        # 2. 生成可视化报告
        self._generate_plots(df, output_dir, timestamp)
        
        # 3. 生成文本报告
        self._generate_text_report(df, output_dir, timestamp)
    
    def _generate_plots(self, df: pd.DataFrame, output_dir: Path, timestamp: str):
        """生成可视化图表"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 创建图表：正确率随epoch变化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 图1：5px正确率
        if 'epoch' in df.columns and df['epoch'].min() >= 0:
            ax = axes[0, 0]
            ax.plot(df['epoch'], df['hit_le_5px'], marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('5px Accuracy (%)', fontsize=12)
            ax.set_title('5px Accuracy over Training Epochs', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 标记最佳点
            best_idx = df['hit_le_5px'].idxmax()
            best_epoch = df.loc[best_idx, 'epoch']
            best_acc = df.loc[best_idx, 'hit_le_5px']
            ax.scatter(best_epoch, best_acc, color='red', s=200, zorder=5)
            ax.annotate(f'Best: {best_acc:.1f}%\nEpoch {best_epoch}',
                       xy=(best_epoch, best_acc),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 图2：不同阈值的正确率对比
        ax = axes[0, 1]
        thresholds = ['hit_le_2px', 'hit_le_3px', 'hit_le_5px', 'hit_le_10px']
        threshold_labels = ['≤2px', '≤3px', '≤5px', '≤10px']
        
        if all(col in df.columns for col in thresholds):
            # 选择最佳模型
            best_model_idx = df['hit_le_5px'].idxmax()
            best_model = df.loc[best_model_idx]
            
            values = [best_model[th] for th in thresholds]
            bars = ax.bar(threshold_labels, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title(f'Best Model Accuracy at Different Thresholds\n({best_model["model"]})', fontsize=14, fontweight='bold')
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}%', ha='center', fontsize=10)
        
        # 图3：平均误差对比
        ax = axes[1, 0]
        if 'distance_mae' in df.columns:
            ax.plot(df.index, df['distance_mae'], marker='s', linewidth=2, markersize=8, label='Distance MAE')
            ax.plot(df.index, df['gap_mae'], marker='^', linewidth=2, markersize=8, label='Gap MAE')
            ax.plot(df.index, df['slider_mae'], marker='v', linewidth=2, markersize=8, label='Slider MAE')
            ax.set_xlabel('Model Index', fontsize=12)
            ax.set_ylabel('MAE (pixels)', fontsize=12)
            ax.set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 图4：处理时间
        ax = axes[1, 1]
        if 'avg_time_ms' in df.columns:
            ax.bar(range(len(df)), df['avg_time_ms'], color='#95A5A6')
            ax.set_xlabel('Model Index', fontsize=12)
            ax.set_ylabel('Avg Time (ms)', fontsize=12)
            ax.set_title('Average Processing Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = output_dir / f"model_evaluation_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Visualization report saved: {plot_path}")
    
    def _generate_text_report(self, df: pd.DataFrame, output_dir: Path, timestamp: str):
        """生成文本格式的详细报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("真实验证码数据集模型评估报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据集路径: {self.data_dir}")
        report_lines.append(f"模型目录: {self.checkpoint_dir}")
        report_lines.append(f"评估图片数: {len(self.image_files)}")
        report_lines.append("")
        
        # 找出最佳模型
        if 'hit_le_5px' in df.columns:
            best_idx = df['hit_le_5px'].idxmax()
            best_model = df.loc[best_idx]
            
            report_lines.append("🏆 最佳模型（5px正确率）")
            report_lines.append("-" * 40)
            report_lines.append(f"模型: {best_model['model']}")
            report_lines.append(f"5px正确率: {best_model['hit_le_5px']:.2f}%")
            report_lines.append(f"滑动距离MAE: {best_model['distance_mae']:.2f} px")
            report_lines.append(f"平均处理时间: {best_model['avg_time_ms']:.2f} ms")
            report_lines.append("")
        
        # 所有模型的详细结果
        report_lines.append("📊 所有模型评估结果")
        report_lines.append("-" * 80)
        
        # 构建表格
        columns_to_show = ['model', 'hit_le_5px', 'distance_mae', 'gap_mae', 'slider_mae', 'avg_time_ms']
        available_columns = [col for col in columns_to_show if col in df.columns]
        
        # 格式化DataFrame为字符串
        df_display = df[available_columns].copy()
        
        # 格式化数值列
        for col in df_display.columns:
            if col != 'model':
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        report_lines.append(df_display.to_string(index=False))
        report_lines.append("")
        
        # 保存文本报告
        txt_path = output_dir / f"model_evaluation_report_{timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Text report saved: {txt_path}")
        
        # 同时打印到控制台
        print("\n" + "\n".join(report_lines))


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='评估模型在真实验证码数据集上的表现')
    parser.add_argument('--data-dir', type=str, 
                       default='data/real_captchas/annotated',
                       help='真实验证码数据目录')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='src/checkpoints/1.1.0',
                       help='模型checkpoint目录')
    parser.add_argument('--threshold', type=int, default=5,
                       help='正确判定的像素阈值')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='运行设备')
    parser.add_argument('--output-dir', type=str, default='outputs/real_captchas_report_1.1.0',
                       help='报告输出目录')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = RealCaptchaEvaluator(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )
    
    # 评估所有模型
    print("\n🚀 Starting evaluation of all models...")
    results_df = evaluator.evaluate_all_models(threshold_px=args.threshold)
    
    # 生成报告
    if not results_df.empty:
        print("\n📝 Generating evaluation report...")
        evaluator.generate_report(results_df, args.output_dir)
        print("\n✅ Evaluation completed!")
    else:
        print("\n❌ No models were successfully evaluated")


if __name__ == "__main__":
    main()