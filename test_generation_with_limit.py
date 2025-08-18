#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试验证码生成脚本的限制功能
"""
import subprocess
import sys
from pathlib import Path

def test_generation_modes():
    """测试不同的生成模式"""
    script_path = Path("scripts/data_generation/generate_captchas_with_components.py")
    
    print("="*60)
    print("测试验证码生成脚本的限制功能")
    print("="*60)
    
    # 测试1：测试模式
    print("\n1. 测试 --test-mode 参数（默认10张图片）")
    print("-" * 40)
    cmd = [sys.executable, str(script_path), "--test-mode"]
    print(f"运行命令: {' '.join(cmd)}")
    print("期望: 处理10张图片，生成1000个验证码")
    print("\n按Enter继续...")
    input()
    
    # 测试2：自定义数量
    print("\n2. 测试 --max-images 参数（自定义5张图片）")
    print("-" * 40)
    cmd = [sys.executable, str(script_path), "--max-images", "5"]
    print(f"运行命令: {' '.join(cmd)}")
    print("期望: 处理5张图片，生成500个验证码")
    print("\n按Enter继续...")
    input()
    
    # 测试3：测试模式 + 覆盖数量
    print("\n3. 测试组合参数（测试模式但覆盖为3张图片）")
    print("-" * 40)
    cmd = [sys.executable, str(script_path), "--test-mode", "--max-images", "3"]
    print(f"运行命令: {' '.join(cmd)}")
    print("期望: 处理3张图片，生成300个验证码")
    print("\n按Enter继续...")
    input()
    
    print("\n" + "="*60)
    print("测试说明完成！")
    print("="*60)
    print("\n实际运行命令示例：")
    print("  # 测试模式（10张图片，1000个验证码）")
    print("  python scripts/data_generation/generate_captchas_with_components.py --test-mode")
    print()
    print("  # 自定义数量")
    print("  python scripts/data_generation/generate_captchas_with_components.py --max-images 20")
    print()
    print("  # 查看帮助")
    print("  python scripts/data_generation/generate_captchas_with_components.py --help")

if __name__ == "__main__":
    test_generation_modes()