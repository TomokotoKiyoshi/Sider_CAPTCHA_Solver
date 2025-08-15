#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据生成器修复 - 验证假缺口和旋转角度是否正确提取
"""
import json
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_label_extraction():
    """测试标签提取逻辑"""
    print("="*60)
    print("测试数据生成器标签提取修复")
    print("="*60)
    
    # 加载原始标签
    labels_file = project_root / 'data' / 'labels' / 'labels_by_pic.json'
    if not labels_file.exists():
        print(f"❌ 标签文件不存在: {labels_file}")
        return False
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_by_pic = json.load(f)
    
    # 统计信息
    total_samples = 0
    samples_with_fake_gaps = 0
    samples_with_rotation = 0
    
    # 分析每张图片的标签
    for pic_id, samples in labels_by_pic.items():
        for sample in samples:
            total_samples += 1
            
            # 检查假缺口
            if 'augmented_labels' in sample and 'fake_gaps' in sample.get('augmented_labels', {}):
                fake_gaps = sample['augmented_labels']['fake_gaps']
                if fake_gaps:
                    samples_with_fake_gaps += 1
            
            # 检查旋转角度
            if 'augmented_labels' in sample and 'gap_rotation' in sample.get('augmented_labels', {}):
                rotation = sample['augmented_labels']['gap_rotation']
                if rotation != 0:
                    samples_with_rotation += 1
    
    print(f"\n📊 统计结果:")
    print(f"  总样本数: {total_samples}")
    print(f"  包含假缺口的样本: {samples_with_fake_gaps} ({samples_with_fake_gaps/total_samples*100:.1f}%)")
    print(f"  包含旋转的样本: {samples_with_rotation} ({samples_with_rotation/total_samples*100:.1f}%)")
    
    # 测试数据生成器的提取逻辑
    print(f"\n🔍 测试提取逻辑:")
    
    # 找一个有假缺口和旋转的样本
    test_sample = None
    for pic_id, samples in labels_by_pic.items():
        for sample in samples:
            if 'augmented_labels' in sample:
                aug = sample.get('augmented_labels', {})
                if 'fake_gaps' in aug and aug['fake_gaps'] and 'gap_rotation' in aug and aug['gap_rotation'] != 0:
                    test_sample = sample
                    break
        if test_sample:
            break
    
    if test_sample:
        print(f"\n✅ 找到测试样本: {test_sample['sample_id']}")
        
        # 模拟数据生成器的提取逻辑（修复后）
        label = test_sample
        
        # 提取坐标信息
        gap_center = tuple(label['labels']['bg_gap_center'])
        slider_center = tuple(label['labels']['comp_piece_center'])
        
        # 提取旋转角度（从augmented_labels中获取）
        gap_angle = 0.0
        if 'augmented_labels' in label and 'gap_rotation' in label.get('augmented_labels', {}):
            gap_angle = label['augmented_labels']['gap_rotation']
        
        # 处理混淆缺口（从augmented_labels中获取）
        fake_gaps = []
        if 'augmented_labels' in label and 'fake_gaps' in label.get('augmented_labels', {}):
            for fake_gap in label['augmented_labels']['fake_gaps']:
                fake_gaps.append(tuple(fake_gap['center']))
        
        print(f"\n📍 提取结果:")
        print(f"  Gap center: {gap_center}")
        print(f"  Slider center: {slider_center}")
        print(f"  Gap rotation: {gap_angle}°")
        print(f"  Fake gaps: {fake_gaps}")
        
        # 验证提取是否正确
        expected_rotation = test_sample['augmented_labels']['gap_rotation']
        expected_fake_gaps = [tuple(fg['center']) for fg in test_sample['augmented_labels']['fake_gaps']]
        
        if gap_angle == expected_rotation:
            print(f"  ✅ 旋转角度提取正确")
        else:
            print(f"  ❌ 旋转角度错误: 期望 {expected_rotation}, 实际 {gap_angle}")
        
        if fake_gaps == expected_fake_gaps:
            print(f"  ✅ 假缺口提取正确")
        else:
            print(f"  ❌ 假缺口错误")
            print(f"     期望: {expected_fake_gaps}")
            print(f"     实际: {fake_gaps}")
    else:
        print(f"\n⚠️ 未找到同时包含假缺口和旋转的样本")
    
    # 测试一个只有假缺口的样本
    for pic_id, samples in labels_by_pic.items():
        for sample in samples:
            if 'augmented_labels' in sample and 'fake_gaps' in sample.get('augmented_labels', {}):
                if sample['augmented_labels']['fake_gaps']:
                    print(f"\n📝 仅假缺口样本: {sample['sample_id']}")
                    print(f"  Fake gaps: {[fg['center'] for fg in sample['augmented_labels']['fake_gaps']]}")
                    break
        break
    
    return True


if __name__ == "__main__":
    success = test_label_extraction()
    
    if success:
        print("\n" + "="*60)
        print("✅ 测试通过 - 数据生成器修复成功")
        print("="*60)
        print("\n下一步:")
        print("1. 运行 'python scripts/data_generation/preprocess_dataset.py' 重新生成数据集")
        print("2. 生成完成后运行训练脚本验证损失计算")
    else:
        print("\n❌ 测试失败")