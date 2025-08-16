#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模型输出 - 分析当前模型能返回什么信息
"""
import torch
import sys
from pathlib import Path
import json
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import create_lite_hrnet_18_fpn

def test_model_output():
    """测试模型的输出结构"""
    
    print("=" * 80)
    print("模型输出分析")
    print("=" * 80)
    
    # 加载模型
    checkpoint_path = Path("src/checkpoints/1.1.0/best_model.pth")
    
    if not checkpoint_path.exists():
        print(f"错误：找不到模型文件 {checkpoint_path}")
        return
    
    # 创建模型
    model = create_lite_hrnet_18_fpn()
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功: {checkpoint_path}")
    print()
    
    # 创建测试输入
    batch_size = 2
    # 输入格式：[B, 4, 256, 512] - 4通道(RGB + padding mask)
    test_input = torch.randn(batch_size, 4, 256, 512)
    
    print("输入张量形状:")
    print(f"  Shape: {test_input.shape}")
    print(f"  说明: [batch_size, channels(RGB+mask), height, width]")
    print()
    
    # 前向传播
    with torch.no_grad():
        outputs = model(test_input)
    
    print("=" * 80)
    print("模型原始输出 (outputs字典):")
    print("=" * 80)
    for key, value in outputs.items():
        print(f"\n{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Dtype: {value.dtype}")
        print(f"  Range: [{value.min().item():.4f}, {value.max().item():.4f}]")
        
        # 解释每个输出的含义
        if key == 'heatmap_gap':
            print(f"  说明: 缺口中心点热力图 (概率分布)")
            print(f"  激活函数: Sigmoid")
            print(f"  分辨率: 1/4 原图 (64x128)")
        elif key == 'heatmap_slider':
            print(f"  说明: 滑块中心点热力图 (概率分布)")
            print(f"  激活函数: Sigmoid")
            print(f"  分辨率: 1/4 原图 (64x128)")
        elif key == 'offset_gap':
            print(f"  说明: 缺口亚像素偏移量 [dx, dy]")
            print(f"  激活函数: Linear")
            print(f"  作用: 精细化定位，提高精度")
        elif key == 'offset_slider':
            print(f"  说明: 滑块亚像素偏移量 [dx, dy]")
            print(f"  激活函数: Linear")
            print(f"  作用: 精细化定位，提高精度")
    
    # 解码预测结果
    print("\n" + "=" * 80)
    print("解码后的预测结果 (decode_predictions):")
    print("=" * 80)
    
    predictions = model.decode_predictions(outputs, input_images=test_input)
    
    for key, value in predictions.items():
        print(f"\n{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Dtype: {value.dtype}")
        
        if 'coords' in key:
            print(f"  说明: 原图分辨率下的坐标 [x, y]")
            print(f"  单位: 像素")
            for i in range(batch_size):
                print(f"  样本{i+1}: x={value[i, 0].item():.2f}, y={value[i, 1].item():.2f}")
        elif 'score' in key:
            print(f"  说明: 检测置信度分数")
            print(f"  范围: [0, 1]")
            for i in range(batch_size):
                print(f"  样本{i+1}: {value[i].item():.4f}")
    
    # 计算滑动距离
    print("\n" + "=" * 80)
    print("滑动距离计算:")
    print("=" * 80)
    
    gap_x = predictions['gap_coords'][:, 0]  # 缺口x坐标
    slider_x = predictions['slider_coords'][:, 0]  # 滑块x坐标
    sliding_distance = gap_x - slider_x
    
    for i in range(batch_size):
        print(f"\n样本 {i+1}:")
        print(f"  缺口位置: ({predictions['gap_coords'][i, 0].item():.2f}, "
              f"{predictions['gap_coords'][i, 1].item():.2f})")
        print(f"  滑块位置: ({predictions['slider_coords'][i, 0].item():.2f}, "
              f"{predictions['slider_coords'][i, 1].item():.2f})")
        print(f"  滑动距离: {sliding_distance[i].item():.2f} px")
        print(f"  缺口置信度: {predictions['gap_score'][i].item():.4f}")
        print(f"  滑块置信度: {predictions['slider_score'][i].item():.4f}")
        print(f"  平均置信度: {(predictions['gap_score'][i].item() + predictions['slider_score'][i].item()) / 2:.4f}")
    
    # 总结模型能力
    print("\n" + "=" * 80)
    print("模型能力总结:")
    print("=" * 80)
    print("""
当前模型能够返回的信息：

1. **坐标预测**:
   - gap_coords: 缺口中心坐标 [x, y] (像素)
   - slider_coords: 滑块中心坐标 [x, y] (像素)
   
2. **置信度分数**:
   - gap_score: 缺口检测置信度 (0-1)
   - slider_score: 滑块检测置信度 (0-1)
   
3. **热力图** (可选，用于可视化):
   - heatmap_gap: 缺口概率热力图 (64x128)
   - heatmap_slider: 滑块概率热力图 (64x128)
   
4. **偏移量** (内部使用):
   - offset_gap: 缺口亚像素偏移 [dx, dy]
   - offset_slider: 滑块亚像素偏移 [dx, dy]
   
5. **计算衍生信息**:
   - sliding_distance: 滑动距离 = gap_x - slider_x
   - combined_confidence: 平均置信度 = (gap_score + slider_score) / 2
   
6. **额外功能**:
   - 支持批量处理 (batch processing)
   - 自动处理padding mask
   - 亚像素精度定位
    """)
    
    # 保存示例输出
    example_output = {
        "model_outputs": {
            "heatmap_gap": f"Shape: {outputs['heatmap_gap'].shape}",
            "heatmap_slider": f"Shape: {outputs['heatmap_slider'].shape}",
            "offset_gap": f"Shape: {outputs['offset_gap'].shape}",
            "offset_slider": f"Shape: {outputs['offset_slider'].shape}"
        },
        "decoded_predictions": {
            "gap_coords": predictions['gap_coords'].tolist(),
            "slider_coords": predictions['slider_coords'].tolist(),
            "gap_score": predictions['gap_score'].tolist(),
            "slider_score": predictions['slider_score'].tolist()
        },
        "calculated_values": {
            "sliding_distance": sliding_distance.tolist(),
            "combined_confidence": ((predictions['gap_score'] + predictions['slider_score']) / 2).tolist()
        }
    }
    
    output_path = Path("scripts/model_output_example.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(example_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n示例输出已保存到: {output_path}")


if __name__ == "__main__":
    test_model_output()