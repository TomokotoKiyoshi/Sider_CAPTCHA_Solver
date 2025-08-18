#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析真实验证码标注数据的坐标分布
统计滑块和缺口的x,y坐标分布情况
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import seaborn as sns
import pandas as pd

def parse_filename(filename):
    """
    解析文件名获取坐标信息
    格式: Pic0001_Bgx112Bgy97_Sdx32Sdy98_cb7cbd17.png
    """
    pattern = r"Pic(\d+)_Bgx(\d+)Bgy(\d+)_Sdx(\d+)Sdy(\d+)_(\w+)\.png"
    match = re.match(pattern, filename)
    if match:
        pic_id, bg_x, bg_y, sd_x, sd_y, hash_val = match.groups()
        return {
            'pic_id': int(pic_id),
            'gap_x': int(bg_x),
            'gap_y': int(bg_y),
            'slider_x': int(sd_x),
            'slider_y': int(sd_y),
            'distance': int(bg_x) - int(sd_x)
        }
    return None

def analyze_site_annotations(site_path, site_name):
    """分析单个站点的标注数据"""
    data = {
        'gap_x': [],
        'gap_y': [],
        'slider_x': [],
        'slider_y': [],
        'distance': []
    }
    
    # 遍历所有PNG文件
    png_files = list(Path(site_path).glob("*.png"))
    print(f"\n{site_name}: Found {len(png_files)} annotated images")
    
    for file_path in png_files:
        coords = parse_filename(file_path.name)
        if coords:
            data['gap_x'].append(coords['gap_x'])
            data['gap_y'].append(coords['gap_y'])
            data['slider_x'].append(coords['slider_x'])
            data['slider_y'].append(coords['slider_y'])
            data['distance'].append(coords['distance'])
    
    # 转换为numpy数组
    for key in data:
        data[key] = np.array(data[key])
    
    # 计算统计信息
    stats = {}
    for key in data:
        if len(data[key]) > 0:
            stats[key] = {
                'min': np.min(data[key]),
                'max': np.max(data[key]),
                'mean': np.mean(data[key]),
                'std': np.std(data[key]),
                'median': np.median(data[key])
            }
    
    return data, stats

def print_statistics(site_name, stats):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"{site_name} Statistics")
    print('='*60)
    
    # 创建表格
    headers = ['Metric', 'Min', 'Max', 'Mean', 'Std', 'Median']
    rows = []
    
    labels = {
        'gap_x': 'Gap X',
        'gap_y': 'Gap Y', 
        'slider_x': 'Slider X',
        'slider_y': 'Slider Y',
        'distance': 'Distance'
    }
    
    for key in ['gap_x', 'gap_y', 'slider_x', 'slider_y', 'distance']:
        if key in stats:
            s = stats[key]
            rows.append([
                labels[key],
                f"{s['min']:.0f}",
                f"{s['max']:.0f}",
                f"{s['mean']:.1f}",
                f"{s['std']:.1f}",
                f"{s['median']:.0f}"
            ])
    
    # 打印表格
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
    
    # 打印表头
    header_line = ''.join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print('-' * sum(col_widths))
    
    # 打印数据行
    for row in rows:
        row_line = ''.join(f"{val:<{w}}" for val, w in zip(row, col_widths))
        print(row_line)

def plot_distributions(site1_data, site2_data, site1_stats, site2_stats):
    """绘制分布对比图"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('CAPTCHA Annotation Distribution Comparison', fontsize=16, fontweight='bold')
    
    # 1. Gap X分布
    ax = axes[0, 0]
    if len(site1_data['gap_x']) > 0:
        ax.hist(site1_data['gap_x'], bins=20, alpha=0.6, label='Site1', color='blue', edgecolor='black')
    if len(site2_data['gap_x']) > 0:
        ax.hist(site2_data['gap_x'], bins=20, alpha=0.6, label='Site2', color='red', edgecolor='black')
    ax.set_xlabel('Gap X Position')
    ax.set_ylabel('Frequency')
    ax.set_title('Gap X Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Gap Y分布
    ax = axes[0, 1]
    if len(site1_data['gap_y']) > 0:
        ax.hist(site1_data['gap_y'], bins=20, alpha=0.6, label='Site1', color='blue', edgecolor='black')
    if len(site2_data['gap_y']) > 0:
        ax.hist(site2_data['gap_y'], bins=20, alpha=0.6, label='Site2', color='red', edgecolor='black')
    ax.set_xlabel('Gap Y Position')
    ax.set_ylabel('Frequency')
    ax.set_title('Gap Y Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Slider X分布
    ax = axes[1, 0]
    if len(site1_data['slider_x']) > 0:
        ax.hist(site1_data['slider_x'], bins=20, alpha=0.6, label='Site1', color='blue', edgecolor='black')
    if len(site2_data['slider_x']) > 0:
        ax.hist(site2_data['slider_x'], bins=20, alpha=0.6, label='Site2', color='red', edgecolor='black')
    ax.set_xlabel('Slider X Position')
    ax.set_ylabel('Frequency')
    ax.set_title('Slider X Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Slider Y分布
    ax = axes[1, 1]
    if len(site1_data['slider_y']) > 0:
        ax.hist(site1_data['slider_y'], bins=20, alpha=0.6, label='Site1', color='blue', edgecolor='black')
    if len(site2_data['slider_y']) > 0:
        ax.hist(site2_data['slider_y'], bins=20, alpha=0.6, label='Site2', color='red', edgecolor='black')
    ax.set_xlabel('Slider Y Position')
    ax.set_ylabel('Frequency')
    ax.set_title('Slider Y Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 滑动距离分布
    ax = axes[2, 0]
    if len(site1_data['distance']) > 0:
        ax.hist(site1_data['distance'], bins=20, alpha=0.6, label='Site1', color='blue', edgecolor='black')
    if len(site2_data['distance']) > 0:
        ax.hist(site2_data['distance'], bins=20, alpha=0.6, label='Site2', color='red', edgecolor='black')
    ax.set_xlabel('Sliding Distance (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Sliding Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 散点图 - Gap位置
    ax = axes[2, 1]
    if len(site1_data['gap_x']) > 0:
        ax.scatter(site1_data['gap_x'], site1_data['gap_y'], alpha=0.6, label='Site1 Gap', color='blue', s=30)
    if len(site2_data['gap_x']) > 0:
        ax.scatter(site2_data['gap_x'], site2_data['gap_y'], alpha=0.6, label='Site2 Gap', color='red', s=30)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Gap Position Distribution (2D)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path("outputs/annotation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "coordinate_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Distribution plot saved to: {output_path}")
    plt.show()

def main():
    """主函数"""
    # 定义路径
    site1_path = r"D:\Hacker\Sider_CAPTCHA_Solver\data\real_captchas\annotated\site1"
    site2_path = r"D:\Hacker\Sider_CAPTCHA_Solver\data\real_captchas\annotated\site2"
    
    print("="*60)
    print("CAPTCHA Annotation Distribution Analysis")
    print("="*60)
    
    # 分析Site1
    site1_data, site1_stats = analyze_site_annotations(site1_path, "Site1")
    print_statistics("Site1", site1_stats)
    
    # 分析Site2
    site2_data, site2_stats = analyze_site_annotations(site2_path, "Site2")
    print_statistics("Site2", site2_stats)
    
    # 对比分析
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print('='*60)
    
    # 计算范围差异
    if site1_stats and site2_stats:
        print("\nCoordinate Range Comparison:")
        print(f"  Site1 Gap X:    [{site1_stats['gap_x']['min']:.0f}, {site1_stats['gap_x']['max']:.0f}]")
        print(f"  Site2 Gap X:    [{site2_stats['gap_x']['min']:.0f}, {site2_stats['gap_x']['max']:.0f}]")
        print(f"  Site1 Gap Y:    [{site1_stats['gap_y']['min']:.0f}, {site1_stats['gap_y']['max']:.0f}]")
        print(f"  Site2 Gap Y:    [{site2_stats['gap_y']['min']:.0f}, {site2_stats['gap_y']['max']:.0f}]")
        print(f"  Site1 Slider X: [{site1_stats['slider_x']['min']:.0f}, {site1_stats['slider_x']['max']:.0f}]")
        print(f"  Site2 Slider X: [{site2_stats['slider_x']['min']:.0f}, {site2_stats['slider_x']['max']:.0f}]")
        print(f"  Site1 Distance: [{site1_stats['distance']['min']:.0f}, {site1_stats['distance']['max']:.0f}]")
        print(f"  Site2 Distance: [{site2_stats['distance']['min']:.0f}, {site2_stats['distance']['max']:.0f}]")
    
    # 绘制分布图
    plot_distributions(site1_data, site2_data, site1_stats, site2_stats)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()