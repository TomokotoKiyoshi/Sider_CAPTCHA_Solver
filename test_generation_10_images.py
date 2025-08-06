#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 使用10张图片生成验证码
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import subprocess
import json
import time
from datetime import datetime

def run_test():
    """运行测试生成"""
    print("="*50)
    print("CAPTCHA Generation Test - 10 Images")
    print("="*50)
    
    # 设置参数
    input_dir = "data/raw"
    output_dir = "test_output/captchas_10"
    max_images = 10
    workers = 4
    
    # 构建命令
    cmd = [
        sys.executable,
        "scripts/data_generation/generate_captchas.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--max-images", str(max_images),
        "--workers", str(workers),
        "--seed", "42"  # 固定种子以便重现
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max images: {max_images}")
    print(f"Workers: {workers}")
    print("-"*50)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行生成脚本
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0:
            print("[ERROR] Generation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
        print("[SUCCESS] Generation completed!")
        print("\nOutput:")
        print(result.stdout)
        
    except Exception as e:
        print(f"[ERROR] Failed to run generation: {e}")
        return False
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nTime taken: {elapsed_time:.2f} seconds")
    
    # 检查输出
    output_path = Path(output_dir)
    if not output_path.exists():
        print("[ERROR] Output directory not created!")
        return False
    
    # 统计生成的文件
    png_files = list(output_path.glob("*.png"))
    print(f"\nGenerated files: {len(png_files)} PNG images")
    
    # 读取报告
    report_path = output_path / "generation_report.json"
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        print("\nGeneration Report:")
        print(f"- Total backgrounds: {report.get('total_backgrounds', 0)}")
        print(f"- Total samples: {report.get('total_samples', 0)}")
        print(f"- Total errors: {report.get('total_errors', 0)}")
        print(f"- Error rate: {report.get('error_rate', 'N/A')}")
        print(f"- Generation time: {report.get('generation_time', 'N/A')}")
        
        # 显示统计信息
        if 'statistics' in report:
            print("\nStatistics:")
            stats = report['statistics']
            
            # 形状统计
            shape_stats = {k: v for k, v in stats.items() if k.startswith('shape_')}
            if shape_stats:
                print("  Shapes:")
                for shape, count in sorted(shape_stats.items()):
                    print(f"    {shape}: {count}")
            
            # 大小统计
            size_stats = {k: v for k, v in stats.items() if k.startswith('size_')}
            if size_stats:
                print("  Sizes:")
                for size, count in sorted(size_stats.items()):
                    print(f"    {size}: {count}")
            
            # 混淆类型统计
            confusion_stats = {k: v for k, v in stats.items() if k.startswith('confusion_')}
            if confusion_stats:
                print("  Confusion types:")
                for confusion, count in sorted(confusion_stats.items()):
                    print(f"    {confusion}: {count}")
    
    # 检查错误日志
    error_log_path = output_path / "generation_errors.json"
    if error_log_path.exists():
        with open(error_log_path, 'r', encoding='utf-8') as f:
            error_data = json.load(f)
        
        print(f"\n[WARNING] Found error log with {error_data['total_errors']} errors")
        print(f"Error rate: {error_data['error_rate']}")
        
        # 显示前5个错误
        if error_data['errors']:
            print("\nFirst 5 errors:")
            for i, error in enumerate(error_data['errors'][:5]):
                print(f"\n  Error {i+1}:")
                print(f"    Image: {error['img_path']}")
                print(f"    Shape: {error['shape']}")
                print(f"    Size: {error['size']}")
                print(f"    Error: {error['error_type']}: {error['error_message']}")
    
    # 检查标注文件
    annotations_path = output_path / "all_annotations.json"
    if annotations_path.exists():
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"\nAnnotations: {len(annotations)} entries")
        
        # 显示前3个标注示例
        if annotations:
            print("\nFirst 3 annotations:")
            for i, ann in enumerate(annotations[:3]):
                print(f"\n  Annotation {i+1}:")
                print(f"    Filename: {ann['filename']}")
                print(f"    Gap position: {ann['bg_center']}")
                print(f"    Slider position: {ann['sd_center']}")
                print(f"    Shape: {ann['shape']}")
                print(f"    Size: {ann['size']}")
                print(f"    Confusion: {ann['confusion_type']}")
    
    print("\n" + "="*50)
    print("Test completed successfully!")
    print("="*50)
    
    return True

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)