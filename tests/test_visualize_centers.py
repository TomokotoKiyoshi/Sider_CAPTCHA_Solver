# -*- coding: utf-8 -*-
"""
可视化测试脚本 - 在验证码图片上标记滑块和缺口中心
测试100张图片，输出到test_outputs目录
"""
import cv2
import numpy as np
from pathlib import Path
import re
import random
from tqdm import tqdm
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def parse_filename(filename):
    """
    从文件名解析坐标信息
    文件名格式: Pic0001_Bgx120Bgy70_Sdx30Sdy70_{hash}.png
    
    Returns:
        dict: 包含pic_id, bg_x, bg_y, sd_x, sd_y的字典
    """
    # 使用正则表达式解析文件名
    pattern = r'Pic(\d+)_Bgx(\d+)Bgy(\d+)_Sdx(\d+)Sdy(\d+)_([a-f0-9]+)\.png'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'pic_id': int(match.group(1)),
            'bg_x': int(match.group(2)),  # 缺口中心x
            'bg_y': int(match.group(3)),  # 缺口中心y
            'sd_x': int(match.group(4)),  # 滑块中心x
            'sd_y': int(match.group(5)),  # 滑块中心y
            'hash': match.group(6)
        }
    return None

def draw_centers(image, bg_center, sd_center):
    """
    在图片上绘制滑块和缺口的中心点
    
    Args:
        image: 输入图像
        bg_center: 缺口中心坐标 (x, y)
        sd_center: 滑块中心坐标 (x, y)
    
    Returns:
        标记后的图像
    """
    result = image.copy()
    
    # 绘制缺口中心（红色）
    cv2.circle(result, bg_center, 8, (0, 0, 255), -1)  # 实心圆
    cv2.circle(result, bg_center, 12, (0, 0, 255), 2)  # 外圈
    cv2.putText(result, 'Gap', 
                (bg_center[0] - 15, bg_center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 绘制十字标记
    cv2.line(result, 
             (bg_center[0] - 15, bg_center[1]), 
             (bg_center[0] + 15, bg_center[1]), 
             (0, 0, 255), 2)
    cv2.line(result, 
             (bg_center[0], bg_center[1] - 15), 
             (bg_center[0], bg_center[1] + 15), 
             (0, 0, 255), 2)
    
    # 绘制滑块中心（绿色）
    cv2.circle(result, sd_center, 8, (0, 255, 0), -1)  # 实心圆
    cv2.circle(result, sd_center, 12, (0, 255, 0), 2)  # 外圈
    cv2.putText(result, 'Slider', 
                (sd_center[0] - 20, sd_center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 绘制十字标记
    cv2.line(result, 
             (sd_center[0] - 15, sd_center[1]), 
             (sd_center[0] + 15, sd_center[1]), 
             (0, 255, 0), 2)
    cv2.line(result, 
             (sd_center[0], sd_center[1] - 15), 
             (sd_center[0], sd_center[1] + 15), 
             (0, 255, 0), 2)
    
    # 绘制连接线（黄色虚线）
    # 计算连接线的点
    num_dots = 20
    for i in range(num_dots):
        if i % 2 == 0:  # 每隔一个点绘制，形成虚线
            t = i / num_dots
            x = int(sd_center[0] + t * (bg_center[0] - sd_center[0]))
            y = int(sd_center[1] + t * (bg_center[1] - sd_center[1]))
            cv2.circle(result, (x, y), 2, (0, 255, 255), -1)
    
    # 计算并显示距离
    distance = np.sqrt((bg_center[0] - sd_center[0])**2 + 
                      (bg_center[1] - sd_center[1])**2)
    mid_x = (sd_center[0] + bg_center[0]) // 2
    mid_y = (sd_center[1] + bg_center[1]) // 2
    cv2.putText(result, f'D={distance:.1f}px', 
                (mid_x - 30, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # 显示坐标信息
    info_text = f'Gap:({bg_center[0]},{bg_center[1]}) Slider:({sd_center[0]},{sd_center[1]})'
    cv2.putText(result, info_text, 
                (10, image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result

def create_grid_visualization(images_with_info, grid_size=(5, 4), target_size=(320, 160)):
    """
    创建网格可视化，将多张图片组合成一张大图
    
    Args:
        images_with_info: [(image, info), ...] 图片和信息的列表
        grid_size: (cols, rows) 网格大小
        target_size: (width, height) 统一的目标尺寸
    
    Returns:
        组合后的大图
    """
    cols, rows = grid_size
    if not images_with_info:
        return None
    
    # 使用统一的目标尺寸
    w, h = target_size
    
    # 创建大画布
    grid_h = h * rows
    grid_w = w * cols
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240  # 浅灰色背景
    
    # 填充网格
    for idx, (img, info) in enumerate(images_with_info[:cols*rows]):
        row = idx // cols
        col = idx % cols
        y1 = row * h
        y2 = y1 + h
        x1 = col * w
        x2 = x1 + w
        
        # 调整图片大小到目标尺寸
        img_resized = cv2.resize(img, (w, h))
        
        # 放置图片
        grid[y1:y2, x1:x2] = img_resized
        
        # 添加边框
        cv2.rectangle(grid, (x1, y1), (x2-1, y2-1), (100, 100, 100), 1)
        
        # 添加图片编号
        cv2.putText(grid, f'#{info["pic_id"]:04d}', 
                   (x1 + 5, y1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    
    return grid

def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    captchas_dir = project_root / "data" / "captchas"
    output_base = project_root / "test_output"
    output_dir = output_base / "visualize_centers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置测试数量
    NUM_TEST_SAMPLES = 100  # 只测试100张图片
    
    # 获取所有验证码图片
    image_files = list(captchas_dir.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {captchas_dir}")
        return
    
    print(f"Found {len(image_files)} images in {captchas_dir}")
    
    # 随机选择100张图片
    num_samples = min(NUM_TEST_SAMPLES, len(image_files))
    selected_files = random.sample(image_files, num_samples)
    
    print(f"Processing {num_samples} images for visualization test...")
    
    # 处理图片
    marked_images = []
    error_count = 0
    
    for img_path in tqdm(selected_files, desc="Processing images"):
        try:
            # 解析文件名
            info = parse_filename(img_path.name)
            if not info:
                print(f"Failed to parse filename: {img_path.name}")
                error_count += 1
                continue
            
            # 读取图片
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to read image: {img_path}")
                error_count += 1
                continue
            
            # 绘制标记
            bg_center = (info['bg_x'], info['bg_y'])
            sd_center = (info['sd_x'], info['sd_y'])
            marked_image = draw_centers(image, bg_center, sd_center)
            
            marked_images.append((marked_image, info))
            
            # 保存前20张标记图片作为样例
            if len(marked_images) <= 20:
                output_path = output_dir / f"marked_{img_path.name}"
                cv2.imwrite(str(output_path), marked_image)
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            error_count += 1
    
    print(f"\nProcessed {len(marked_images)} images successfully")
    print(f"Errors: {error_count}")
    
    # 创建网格可视化
    print("\nCreating grid visualizations...")
    
    # 创建多个网格图，每个包含20张图片（5x4）
    grid_size = (5, 4)
    images_per_grid = grid_size[0] * grid_size[1]
    num_grids = min(5, (len(marked_images) + images_per_grid - 1) // images_per_grid)  # 最多5个网格
    
    for i in range(num_grids):
        start_idx = i * images_per_grid
        end_idx = min(start_idx + images_per_grid, len(marked_images))
        grid_images = marked_images[start_idx:end_idx]
        
        grid = create_grid_visualization(grid_images, grid_size)
        if grid is not None:
            grid_path = output_dir / f"grid_{i+1:02d}.png"
            cv2.imwrite(str(grid_path), grid)
            print(f"Saved grid visualization: {grid_path}")
    
    # 创建统计报告
    print("\nGenerating statistics report...")
    
    # 计算统计信息
    bg_x_values = [info['bg_x'] for _, info in marked_images]
    bg_y_values = [info['bg_y'] for _, info in marked_images]
    sd_x_values = [info['sd_x'] for _, info in marked_images]
    sd_y_values = [info['sd_y'] for _, info in marked_images]
    
    distances = [np.sqrt((info['bg_x'] - info['sd_x'])**2 + 
                        (info['bg_y'] - info['sd_y'])**2) 
                for _, info in marked_images]
    
    stats = {
        'total_processed': len(marked_images),
        'errors': error_count,
        'bg_center': {
            'x_range': (min(bg_x_values), max(bg_x_values)),
            'y_range': (min(bg_y_values), max(bg_y_values)),
            'x_mean': np.mean(bg_x_values),
            'y_mean': np.mean(bg_y_values)
        },
        'sd_center': {
            'x_range': (min(sd_x_values), max(sd_x_values)),
            'y_range': (min(sd_y_values), max(sd_y_values)),
            'x_mean': np.mean(sd_x_values),
            'y_mean': np.mean(sd_y_values)
        },
        'distance': {
            'min': min(distances),
            'max': max(distances),
            'mean': np.mean(distances),
            'std': np.std(distances)
        }
    }
    
    # 保存统计报告到文件
    import json
    stats_path = output_dir / "statistics_report.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 打印统计报告
    print("\n" + "="*60)
    print("VISUALIZATION TEST STATISTICS REPORT")
    print("="*60)
    print(f"Test samples: {NUM_TEST_SAMPLES}")
    print(f"Successfully processed: {stats['total_processed']}")
    print(f"Errors encountered: {stats['errors']}")
    print(f"\nGap Center (Background):")
    print(f"  X range: {stats['bg_center']['x_range']}")
    print(f"  Y range: {stats['bg_center']['y_range']}")
    print(f"  Mean position: ({stats['bg_center']['x_mean']:.1f}, {stats['bg_center']['y_mean']:.1f})")
    print(f"\nSlider Center:")
    print(f"  X range: {stats['sd_center']['x_range']}")
    print(f"  Y range: {stats['sd_center']['y_range']}")
    print(f"  Mean position: ({stats['sd_center']['x_mean']:.1f}, {stats['sd_center']['y_mean']:.1f})")
    print(f"\nDistance between centers:")
    print(f"  Range: {stats['distance']['min']:.1f} - {stats['distance']['max']:.1f} pixels")
    print(f"  Mean: {stats['distance']['mean']:.1f} ± {stats['distance']['std']:.1f} pixels")
    print("="*60)
    
    print(f"\nTest outputs saved to: {output_dir}")
    print("  - Sample marked images: marked_*.png (first 20)")
    print("  - Grid visualizations: grid_*.png")
    print("  - Statistics report: statistics_report.json")

if __name__ == "__main__":
    main()