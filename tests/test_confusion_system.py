# -*- coding: utf-8 -*-
"""
测试混淆系统 - 生成两张图的完整测试结果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import json
from datetime import datetime
import os

# 导入混淆系统
from src.captcha_generator.confusion_system.generator import CaptchaGenerator
from src.captcha_generator.confusion_system.strategies import (
    PerlinNoiseConfusion,
    RotationConfusion,
    HighlightConfusion,
    ConfusingGapConfusion,
    HollowCenterConfusion
)


def load_test_images() -> list:
    """
    加载实际的测试图片
    
    Returns:
        测试图片列表
    """
    # 使用实际的图片文件
    img_dir = Path(__file__).parent.parent / "data" / "raw" / "Geometric_art"
    
    # 选择两张图片
    img_files = [
        img_dir / "Pic0031.png",
        img_dir / "Pic0082.png"
    ]
    
    images = []
    for img_file in img_files:
        if img_file.exists():
            img = cv2.imread(str(img_file))
            if img is not None:
                # 调整到标准大小
                img = cv2.resize(img, (320, 160))
                images.append(img)
            else:
                print(f"Warning: Cannot read {img_file}")
                # 如果读取失败，创建默认图片
                images.append(create_default_image(len(images)))
        else:
            print(f"Warning: File not found {img_file}")
            images.append(create_default_image(len(images)))
    
    return images


def create_default_image(index: int) -> np.ndarray:
    """创建默认测试图片（当实际图片不可用时）"""
    img = np.ones((160, 320, 3), dtype=np.uint8) * 200
    cv2.putText(img, f"Default Image {index+1}", (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


def test_all_confusion_strategies(output_dir: Path):
    """
    测试所有混淆策略
    
    Args:
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建生成器
    generator = CaptchaGenerator()
    
    # 测试配置
    test_configs = [
        {
            'name': 'no_confusion',
            'image_index': 0,
            'shape': ('concave', 'flat', 'convex', 'flat'),
            'size': 50,
            'gap_pos': (200, 80),
            'slider_pos': (25, 80),
            'strategies': []
        },
        {
            'name': 'perlin_noise',
            'image_index': 0,
            'shape': 'circle',
            'size': 60,
            'gap_pos': (180, 90),
            'slider_pos': (30, 90),
            'strategies': [
                PerlinNoiseConfusion({'noise_strength': 0.6, 'noise_scale': 0.1})
            ]
        },
        {
            'name': 'rotation',
            'image_index': 0,
            'shape': 'square',
            'size': 55,
            'gap_pos': (220, 70),
            'slider_pos': (20, 70),
            'strategies': [
                RotationConfusion({'rotation_angle': 0.8})
            ]
        },
        {
            'name': 'highlight',
            'image_index': 1,
            'shape': 'triangle',
            'size': 50,
            'gap_pos': (190, 85),
            'slider_pos': (25, 85),
            'strategies': [
                HighlightConfusion({
                    'base_lightness': 35,
                    'edge_lightness': 55,
                    'directional_lightness': 25
                })
            ]
        },
        {
            'name': 'confusing_gap',
            'image_index': 1,
            'shape': 'hexagon',
            'size': 65,
            'gap_pos': (210, 75),
            'slider_pos': (30, 75),
            'strategies': [
                ConfusingGapConfusion({
                    'num_confusing_gaps': 2,
                    'confusing_type': 'mixed'
                })
            ]
        },
        {
            'name': 'hollow_center',
            'image_index': 1,
            'shape': ('convex', 'concave', 'flat', 'convex'),
            'size': 55,
            'gap_pos': (185, 95),
            'slider_pos': (25, 95),
            'strategies': [
                HollowCenterConfusion({'hollow_ratio': 0.5})
            ]
        },
        {
            'name': 'mixed_confusion',
            'image_index': 0,
            'shape': 'circle',
            'size': 60,
            'gap_pos': (200, 80),
            'slider_pos': (25, 80),
            'strategies': [
                PerlinNoiseConfusion({'noise_strength': 0.5}),
                RotationConfusion({'rotation_angle': 0.4}),
                HollowCenterConfusion({'hollow_ratio': 0.3})
            ]
        }
    ]
    
    # 加载测试图片
    test_images = load_test_images()
    
    # 保存原始测试图片
    cv2.imwrite(str(output_dir / "original_image_1.png"), test_images[0])
    cv2.imwrite(str(output_dir / "original_image_2.png"), test_images[1])
    
    results = []
    
    # 执行测试
    for i, config in enumerate(test_configs):
        print(f"\nTesting {config['name']}...")
        
        try:
            # 获取测试图片
            img = test_images[config['image_index']]
            
            # 生成验证码
            result = generator.generate(
                image=img,
                puzzle_shape=config['shape'],
                puzzle_size=config['size'],
                gap_position=config['gap_pos'],
                slider_position=config['slider_pos'],
                apply_shadow=True,
                confusion_strategies=config['strategies']
            )
            
            # 创建展示图像
            display_image = create_display_image(
                result.background,
                result.slider,
                config['slider_pos'],
                config['name']
            )
            
            # 保存结果
            output_path = output_dir / f"result_{i:02d}_{config['name']}.png"
            cv2.imwrite(str(output_path), display_image)
            
            # 记录结果
            test_result = {
                'test_name': config['name'],
                'image_index': config['image_index'],
                'shape': str(config['shape']),
                'size': config['size'],
                'gap_position': config['gap_pos'],
                'slider_position': config['slider_pos'],
                'confusion_type': result.confusion_type,
                'confusion_params': result.confusion_params,
                'output_file': output_path.name,
                'success': True
            }
            
            # 如果有额外的缺口信息
            if result.additional_gaps:
                # 转换 numpy 数组为列表
                additional_gaps_serializable = []
                for gap in result.additional_gaps:
                    gap_copy = gap.copy()
                    # 将 mask 转换为列表
                    if 'mask' in gap_copy and isinstance(gap_copy['mask'], np.ndarray):
                        gap_copy['mask'] = gap_copy['mask'].tolist()
                    additional_gaps_serializable.append(gap_copy)
                test_result['additional_gaps'] = additional_gaps_serializable
            
            results.append(test_result)
            print(f"[SUCCESS] {config['name']} completed successfully")
            
        except Exception as e:
            print(f"[FAILED] {config['name']} failed: {e}")
            results.append({
                'test_name': config['name'],
                'success': False,
                'error': str(e)
            })
    
    # 创建汇总图像
    create_summary_image(output_dir, results)
    
    # 保存测试结果
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_time': datetime.now().isoformat(),
            'total_tests': len(test_configs),
            'successful_tests': sum(1 for r in results if r.get('success', False)),
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nTest complete! Results saved in: {output_dir}")
    print(f"Check test_results.json for details")


def create_display_image(background: np.ndarray, slider: np.ndarray, 
                        slider_pos: tuple, title: str) -> np.ndarray:
    """
    创建展示图像（背景+滑块+标题）
    
    Args:
        background: 背景图
        slider: 滑块图
        slider_pos: 滑块位置
        title: 标题
        
    Returns:
        展示图像
    """
    # 创建更大的画布
    canvas = np.ones((200, 320, 3), dtype=np.uint8) * 255
    
    # 放置背景
    canvas[40:200, :] = background
    
    # 绘制滑块区域
    cv2.rectangle(canvas, (0, 40), (60, 200), (240, 240, 240), -1)
    cv2.line(canvas, (60, 40), (60, 200), (200, 200, 200), 2)
    
    # 放置滑块
    h, w = slider.shape[:2]
    x, y = slider_pos[0], slider_pos[1] + 40  # 调整y坐标
    
    x1 = max(0, x - w // 2)
    y1 = max(40, y - h // 2)
    x2 = min(60, x1 + w)
    y2 = min(200, y1 + h)
    
    actual_w = x2 - x1
    actual_h = y2 - y1
    
    if slider.shape[2] == 4:
        alpha = slider[:actual_h, :actual_w, 3] / 255.0
        for c in range(3):
            canvas[y1:y2, x1:x2, c] = (
                canvas[y1:y2, x1:x2, c] * (1 - alpha) + 
                slider[:actual_h, :actual_w, c] * alpha
            ).astype(np.uint8)
    
    # 添加标题
    cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 0), 2)
    
    return canvas


def create_summary_image(output_dir: Path, results: list):
    """
    创建汇总图像
    
    Args:
        output_dir: 输出目录
        results: 测试结果列表
    """
    # 计算成功的测试
    successful_results = [r for r in results if r.get('success', False)]
    
    # 创建两行的网格
    grid_cols = 4
    grid_rows = 2
    cell_width = 320
    cell_height = 200
    
    # 创建画布
    canvas_width = cell_width * grid_cols
    canvas_height = cell_height * grid_rows + 50  # 额外空间用于标题
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240
    
    # 添加总标题
    title = f"Confusion System Test Results - {len(successful_results)}/{len(results)} Tests Passed"
    cv2.putText(canvas, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, (0, 0, 0), 2)
    
    # 放置每个结果图像
    for i, result in enumerate(successful_results[:8]):  # 最多显示8个
        if 'output_file' in result:
            img_path = output_dir / result['output_file']
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    row = i // grid_cols
                    col = i % grid_cols
                    y1 = row * cell_height + 50
                    y2 = y1 + cell_height
                    x1 = col * cell_width
                    x2 = x1 + cell_width
                    
                    canvas[y1:y2, x1:x2] = img
    
    # 保存汇总图像
    cv2.imwrite(str(output_dir / "summary.png"), canvas)


def main():
    """主函数"""
    # 创建输出目录
    output_dir = Path(__file__).parent / "confusion_system_test_output"
    
    print("Starting confusion system test...")
    print(f"Output directory: {output_dir}")
    
    # 运行测试
    test_all_confusion_strategies(output_dir)
    
    print("\nTest completed!")
    print(f"View results: {output_dir}")
    print("Main files:")
    print("  - original_image_1.png, original_image_2.png: Original test images")
    print("  - result_*.png: Results of each confusion strategy")
    print("  - summary.png: Summary of all results")
    print("  - test_results.json: Detailed test data")


if __name__ == "__main__":
    main()