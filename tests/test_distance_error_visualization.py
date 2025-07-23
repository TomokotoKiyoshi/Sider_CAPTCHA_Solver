"""
测试脚本：可视化不同像素误差的效果
展示3px、4px、5px、6px、7px误差在滑块验证码上的视觉差异
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.inference.predict import CaptchaPredictor


def visualize_distance_errors(image_path, gt_gap, gt_slider, save_path):
    """可视化不同像素误差的效果"""
    # 读取图片
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 计算真实滑动距离
    true_distance = gt_gap[0] - gt_slider[0]
    
    # 定义要展示的误差值
    errors = [0, 3, 4, 5, 6, 7]  # 包含0误差作为参考
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, error_px in enumerate(errors):
        ax = axes[idx]
        ax.imshow(img)
        
        # 绘制真实位置（绿色实线）
        # Gap位置
        gap_circle = patches.Circle((gt_gap[0], gt_gap[1]), 8, 
                                  linewidth=3, edgecolor='green', 
                                  facecolor='none', label='True Gap')
        ax.add_patch(gap_circle)
        
        # Slider位置
        slider_circle = patches.Circle((gt_slider[0], gt_slider[1]), 8,
                                     linewidth=3, edgecolor='green', 
                                     facecolor='none', label='True Slider')
        ax.add_patch(slider_circle)
        
        # 绘制真实滑动线（绿色）
        ax.plot([gt_slider[0], gt_gap[0]], [gt_slider[1], gt_gap[1]], 
               'g-', linewidth=2, alpha=0.8)
        
        # 计算预测位置（基于误差）
        # 保持slider位置不变，调整gap位置来产生指定误差
        pred_distance = true_distance + error_px  # 添加误差
        pred_gap_x = gt_slider[0] + pred_distance
        pred_gap_y = gt_gap[1]  # y坐标保持不变
        
        if error_px > 0:  # 只在有误差时绘制预测位置
            # 预测的Gap位置（红色虚线）
            pred_gap_circle = patches.Circle((pred_gap_x, pred_gap_y), 8, 
                                           linewidth=3, edgecolor='red', 
                                           facecolor='none', linestyle='--',
                                           label='Predicted Gap')
            ax.add_patch(pred_gap_circle)
            
            # 预测的Slider位置（与真实相同）
            pred_slider_circle = patches.Circle((gt_slider[0], gt_slider[1]), 8,
                                              linewidth=3, edgecolor='red', 
                                              facecolor='none', linestyle='--',
                                              label='Predicted Slider')
            ax.add_patch(pred_slider_circle)
            
            # 绘制预测滑动线（红色）
            ax.plot([gt_slider[0], pred_gap_x], [gt_slider[1], pred_gap_y], 
                   'r--', linewidth=2, alpha=0.8)
            
            # 标注误差信息
            ax.text(img.shape[1]//2, 20, 
                   f'Distance Error: {error_px}px\nTrue: {true_distance:.1f}px, Pred: {pred_distance:.1f}px',
                   fontsize=12, ha='center', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
        else:
            # 完美预测的情况
            ax.text(img.shape[1]//2, 20, 
                   f'Perfect Prediction (0px error)\nDistance: {true_distance:.1f}px',
                   fontsize=12, ha='center', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # 绘制误差示意（放大显示）
        if error_px > 0:
            # 在图片底部绘制误差放大图
            error_y = img.shape[0] - 30
            center_x = img.shape[1] // 2
            
            # 真实距离线
            ax.plot([center_x - 50, center_x + 50], [error_y, error_y], 
                   'g-', linewidth=4, label='True Distance')
            
            # 预测距离线
            scale = 100 / true_distance  # 缩放因子
            error_scaled = error_px * scale
            ax.plot([center_x - 50, center_x + 50 + error_scaled], 
                   [error_y - 10, error_y - 10], 
                   'r--', linewidth=4, label='Predicted Distance')
            
            # 误差标注
            ax.annotate('', xy=(center_x + 50 + error_scaled, error_y - 10),
                       xytext=(center_x + 50, error_y),
                       arrowprops=dict(arrowstyle='<->', color='black', lw=2))
            ax.text(center_x + 50 + error_scaled/2, error_y - 20, 
                   f'{error_px}px', ha='center', fontsize=10, weight='bold')
        
        ax.set_title(f'Error: {error_px}px', fontsize=14, weight='bold')
        ax.axis('off')
        
        # 添加图例（只在第一个子图）
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    plt.suptitle('Slider CAPTCHA Distance Error Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")


def main():
    """主函数"""
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 使用captchas数据集
    data_dir = project_root / "data" / "captchas"
    annotations_path = data_dir / "annotations.json"
    
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 使用指定的图片
    target_filename = "Pic0001_Bgx116Bgy80_Sdx32Sdy80_9b469398.png"
    sample = None
    
    for ann in annotations:
        if ann['filename'] == target_filename:
            sample = ann
            break
    
    if sample is None:
        print(f"Target image {target_filename} not found in annotations!")
        return
    
    image_path = data_dir / sample['filename']
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
    
    print(f"Selected image: {sample['filename']}")
    print(f"Gap center: {sample['bg_center']}")
    print(f"Slider center: {sample['sd_center']}")
    print(f"True distance: {sample['bg_center'][0] - sample['sd_center'][0]:.1f}px")
    
    # 创建输出目录
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成可视化
    output_path = output_dir / "distance_error_visualization.png"
    visualize_distance_errors(
        image_path,
        np.array(sample['bg_center']),
        np.array(sample['sd_center']),
        output_path
    )
    
    # 额外：使用模型预测并展示实际预测结果
    print("\n" + "="*60)
    print("Testing with actual model prediction...")
    print("="*60)
    
    # 加载最佳模型
    model_path = project_root / "src" / "checkpoints" / "best_model.pth"
    if model_path.exists():
        predictor = CaptchaPredictor(
            model_path=str(model_path),
            device='auto',
            hm_threshold=0.1
        )
        
        # 预测
        result = predictor.predict(str(image_path))
        
        if result['gap_x'] is not None and result['slider_x'] is not None:
            pred_distance = result['gap_x'] - result['slider_x']
            true_distance = sample['bg_center'][0] - sample['sd_center'][0]
            distance_error = abs(pred_distance - true_distance)
            
            print(f"\nModel Prediction Results:")
            print(f"Predicted Gap: ({result['gap_x']:.1f}, {result['gap_y']:.1f})")
            print(f"Predicted Slider: ({result['slider_x']:.1f}, {result['slider_y']:.1f})")
            print(f"Predicted distance: {pred_distance:.1f}px")
            print(f"True distance: {true_distance:.1f}px")
            print(f"Distance error: {distance_error:.1f}px")
            
            # 创建实际预测结果的可视化
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # 读取图片
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            
            # 绘制真实位置（绿色）
            gt_gap = np.array(sample['bg_center'])
            gt_slider = np.array(sample['sd_center'])
            
            ax.add_patch(patches.Circle((gt_gap[0], gt_gap[1]), 8, 
                                      linewidth=3, edgecolor='green', facecolor='none'))
            ax.add_patch(patches.Circle((gt_slider[0], gt_slider[1]), 8,
                                      linewidth=3, edgecolor='green', facecolor='none'))
            ax.plot([gt_slider[0], gt_gap[0]], [gt_slider[1], gt_gap[1]], 
                   'g-', linewidth=2, label=f'True Distance: {true_distance:.1f}px')
            
            # 绘制预测位置（红色）
            ax.add_patch(patches.Circle((result['gap_x'], result['gap_y']), 8, 
                                      linewidth=3, edgecolor='red', facecolor='none', linestyle='--'))
            ax.add_patch(patches.Circle((result['slider_x'], result['slider_y']), 8,
                                      linewidth=3, edgecolor='red', facecolor='none', linestyle='--'))
            ax.plot([result['slider_x'], result['gap_x']], 
                   [result['slider_y'], result['gap_y']], 
                   'r--', linewidth=2, label=f'Pred Distance: {pred_distance:.1f}px')
            
            # 添加标题和图例
            ax.set_title(f'Actual Model Prediction - Distance Error: {distance_error:.1f}px', 
                        fontsize=14, weight='bold')
            ax.legend(loc='upper right')
            ax.axis('off')
            
            # 保存
            actual_output_path = output_dir / "actual_model_prediction.png"
            plt.tight_layout()
            plt.savefig(actual_output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\nActual prediction visualization saved to: {actual_output_path}")
        else:
            print("Model failed to detect gap or slider!")
    else:
        print(f"\nBest model not found at: {model_path}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()