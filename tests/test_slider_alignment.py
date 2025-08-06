# -*- coding: utf-8 -*-
"""
测试滑块与缺口对齐的效果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.captcha_generator.confusion_system.generator import CaptchaGenerator
from src.captcha_generator.confusion_system.strategies import HighlightConfusion


def show_slider_aligned_with_gap():
    """展示滑块对齐到缺口位置的效果"""
    # 使用Abstract_Geometric_Art目录下的图片
    img_dir = Path(__file__).parent.parent / "data" / "raw" / "Abstract_Geometric_Art"
    img1 = cv2.imread(str(img_dir / "Pic0001.png"))
    img2 = cv2.imread(str(img_dir / "Pic0002.png"))
    
    if img1 is None or img2 is None:
        print("Error: Cannot load images")
        return
    
    generator = CaptchaGenerator()
    output_dir = Path(__file__).parent / "slider_alignment_output"
    output_dir.mkdir(exist_ok=True)
    
    # 测试配置
    configs = [
        {
            'name': 'normal_shadow',
            'image': img1,
            'gap_pos': (200, 80),
            'shape': 'circle',
            'size': 50,
            'strategies': []
        },
        {
            'name': 'highlight_effect',
            'image': img2,
            'gap_pos': (220, 90),
            'shape': 'square',
            'size': 60,
            'strategies': [HighlightConfusion({'base_lightness': 35, 'edge_lightness': 55})]
        }
    ]
    
    for config in configs:
        print(f"\nProcessing {config['name']}...")
        
        # 生成验证码（滑块初始位置在左侧）
        result = generator.generate(
            image=config['image'],
            puzzle_shape=config['shape'],
            puzzle_size=config['size'],
            gap_position=config['gap_pos'],
            slider_position=(25, config['gap_pos'][1]),  # y坐标与缺口相同
            apply_shadow=True,
            confusion_strategies=config['strategies']
        )
        
        # 创建对比图：左侧显示初始状态，右侧显示对齐状态
        h, w = result.background.shape[:2]
        comparison = np.ones((h, w * 2 + 20, 3), dtype=np.uint8) * 255
        
        # 左侧：初始状态（滑块在左侧）
        comparison[:, :w] = result.background
        # 在背景上叠加滑块
        slider_h, slider_w = result.slider.shape[:2]
        init_x = 25 - slider_w // 2
        init_y = config['gap_pos'][1] - slider_h // 2
        
        # 确保不超出边界
        x1 = max(0, init_x)
        y1 = max(0, init_y)
        x2 = min(w, init_x + slider_w)
        y2 = min(h, init_y + slider_h)
        
        # 叠加滑块（使用alpha混合）
        if result.slider.shape[2] == 4:
            for dy in range(y2 - y1):
                for dx in range(x2 - x1):
                    src_y = dy + max(0, -init_y)
                    src_x = dx + max(0, -init_x)
                    if src_y < slider_h and src_x < slider_w:
                        alpha = result.slider[src_y, src_x, 3] / 255.0
                        for c in range(3):
                            comparison[y1 + dy, x1 + dx, c] = int(
                                comparison[y1 + dy, x1 + dx, c] * (1 - alpha) +
                                result.slider[src_y, src_x, c] * alpha
                            )
        
        # 右侧：对齐状态（滑块移动到缺口位置）
        comparison[:, w + 20:] = result.background
        # 滑块对齐到缺口
        align_x = config['gap_pos'][0] - slider_w // 2
        align_y = config['gap_pos'][1] - slider_h // 2
        
        # 确保不超出边界
        x1 = max(0, align_x)
        y1 = max(0, align_y)
        x2 = min(w, align_x + slider_w)
        y2 = min(h, align_y + slider_h)
        
        # 叠加滑块
        if result.slider.shape[2] == 4:
            for dy in range(y2 - y1):
                for dx in range(x2 - x1):
                    src_y = dy + max(0, -align_y)
                    src_x = dx + max(0, -align_x)
                    if src_y < slider_h and src_x < slider_w:
                        alpha = result.slider[src_y, src_x, 3] / 255.0
                        for c in range(3):
                            comparison[y1 + dy, w + 20 + x1 + dx, c] = int(
                                comparison[y1 + dy, w + 20 + x1 + dx, c] * (1 - alpha) +
                                result.slider[src_y, src_x, c] * alpha
                            )
        
        # 添加分隔线
        cv2.line(comparison, (w + 10, 0), (w + 10, h), (200, 200, 200), 2)
        
        # 添加标签
        cv2.putText(comparison, "Initial Position", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(comparison, "Aligned Position", (w + 30, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 保存结果
        output_path = output_dir / f"{config['name']}_comparison.png"
        cv2.imwrite(str(output_path), comparison)
        print(f"Saved: {output_path}")
        
        # 额外保存一个只有对齐状态的图片
        aligned_only = result.background.copy()
        if result.slider.shape[2] == 4:
            for dy in range(y2 - y1):
                for dx in range(x2 - x1):
                    src_y = dy + max(0, -align_y)
                    src_x = dx + max(0, -align_x)
                    if src_y < slider_h and src_x < slider_w:
                        alpha = result.slider[src_y, src_x, 3] / 255.0
                        for c in range(3):
                            aligned_only[y1 + dy, x1 + dx, c] = int(
                                aligned_only[y1 + dy, x1 + dx, c] * (1 - alpha) +
                                result.slider[src_y, src_x, c] * alpha
                            )
        
        aligned_path = output_dir / f"{config['name']}_aligned.png"
        cv2.imwrite(str(aligned_path), aligned_only)
        print(f"Saved aligned version: {aligned_path}")
    
    print(f"\nAll results saved in: {output_dir}")


if __name__ == "__main__":
    print("Testing slider alignment with gap...")
    show_slider_aligned_with_gap()