#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 测试组件保存功能
"""
import sys
from pathlib import Path
# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess
import json
import time
from datetime import datetime

def run_test():
    """运行测试生成"""
    print("="*50)
    print("Component Generation Test - 2 Images")
    print("="*50)
    
    # 设置参数（使用相对于项目根目录的路径）
    input_dir = str(project_root / "data" / "raw")
    output_dir = str(project_root / "test_output" / "captchas_components")
    max_images = 2
    workers = 2
    
    # 构建命令（使用相对于项目根目录的脚本路径）
    script_path = str(project_root / "scripts" / "data_generation" / "generate_captchas_with_components.py")
    cmd = [
        sys.executable,
        script_path,
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--max-images", str(max_images),
        "--workers", str(workers),
        "--seed", "42",  # 固定种子以便重现
        "--save-components"  # 启用组件保存
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max images: {max_images}")
    print(f"Workers: {workers}")
    print(f"Save components: ENABLED")
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
    print(f"\nGenerated CAPTCHA files: {len(png_files)} PNG images")
    
    # 检查组件目录
    components_dir = output_path / "components"
    if not components_dir.exists():
        print("[ERROR] Components directory not created!")
        return False
    
    sliders_dir = components_dir / "sliders"
    backgrounds_dir = components_dir / "backgrounds"
    
    if not sliders_dir.exists() or not backgrounds_dir.exists():
        print("[ERROR] Component subdirectories not created!")
        return False
    
    # 统计组件文件
    slider_files = list(sliders_dir.glob("*.png"))
    bg_files = list(backgrounds_dir.glob("*.png"))
    
    print(f"\nComponent files:")
    print(f"  - Sliders: {len(slider_files)} PNG images")
    print(f"  - Backgrounds: {len(bg_files)} PNG images")
    
    # 显示前3个组件文件名
    if slider_files:
        print("\nSample slider files:")
        for f in slider_files[:3]:
            print(f"  - {f.name}")
    
    if bg_files:
        print("\nSample background files:")
        for f in bg_files[:3]:
            print(f"  - {f.name}")
    
    # 读取报告
    report_path = output_path / "generation_report.json"
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        print("\nGeneration Report:")
        print(f"- Total backgrounds: {report.get('total_backgrounds', 0)}")
        print(f"- Total samples: {report.get('total_samples', 0)}")
        print(f"- Total errors: {report.get('total_errors', 0)}")
        print(f"- Generation time: {report.get('generation_time', 'N/A')}")
    
    # 检查标注文件
    annotations_path = output_path / "all_annotations.json"
    if annotations_path.exists():
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"\nAnnotations: {len(annotations)} entries")
        
        # 检查第一个标注是否包含组件文件路径
        if annotations:
            first_ann = annotations[0]
            print("\nFirst annotation:")
            print(f"  - Filename: {first_ann['filename']}")
            print(f"  - Gap position: {first_ann['bg_center']}")
            print(f"  - Slider position: {first_ann['sd_center']}")
            
            if 'slider_file' in first_ann:
                print(f"  - Slider file: {first_ann['slider_file']}")
            else:
                print("  - [WARNING] No slider_file field in annotation")
            
            if 'background_file' in first_ann:
                print(f"  - Background file: {first_ann['background_file']}")
            else:
                print("  - [WARNING] No background_file field in annotation")
    
    print("\n" + "="*50)
    print("Component generation test completed successfully!")
    print("="*50)
    
    return True

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)