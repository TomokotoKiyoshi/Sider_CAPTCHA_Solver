# -*- coding: utf-8 -*-
"""
测试批量生成验证码（小规模）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_captchas import generate_dataset_parallel


def test_small_batch():
    """测试生成少量验证码"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "test_captchas"
    
    # 创建测试目录
    test_input_dir = project_root / "data" / "test_input"
    test_input_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制几张测试图片
    import shutil
    image_count = 0
    for category_dir in input_dir.iterdir():
        if category_dir.is_dir():
            images = list(category_dir.glob("*.png"))[:2]  # 每个类别取2张
            for img in images:
                shutil.copy(img, test_input_dir / f"test_{image_count:03d}.png")
                image_count += 1
                if image_count >= 2:  # 只测试2张图片
                    break
        if image_count >= 2:
            break
    
    print(f"Prepared {image_count} test images")
    
    # 生成数据集（测试时使用单进程）
    generate_dataset_parallel(test_input_dir, output_dir, max_workers=1)
    
    # 清理测试输入目录
    shutil.rmtree(test_input_dir)
    
    print(f"\nTest completed! Check output at: {output_dir}")


if __name__ == "__main__":
    print("Running small batch test...")
    test_small_batch()