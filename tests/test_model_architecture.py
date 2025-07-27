# -*- coding: utf-8 -*-
"""
测试模型架构
验证ResNet18 Lite + CenterNet实现是否符合设计要求
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import CaptchaSolver


def test_model_shapes():
    """测试模型输入输出形状"""
    print("=" * 60)
    print("Test Model Architecture")
    print("=" * 60)
    
    # 创建模型
    model = CaptchaSolver()
    model.eval()
    
    # 测试输入
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 160, 320)
    print(f"\nInput tensor shape: {input_tensor.shape}")
    print(f"Expected: [2, 3, 160, 320] [OK]")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print("\nOutput tensor shapes:")
    expected_shapes = {
        'gap_heatmap': (batch_size, 1, 40, 80),
        'gap_offset': (batch_size, 2, 40, 80),
        'piece_heatmap': (batch_size, 1, 40, 80),
        'piece_offset': (batch_size, 2, 40, 80)
    }
    
    all_correct = True
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(outputs[key].shape)
        is_correct = actual_shape == expected_shape
        all_correct &= is_correct
        
        status = "[OK]" if is_correct else "[FAIL]"
        print(f"  {key}: {actual_shape} (expected: {expected_shape}) {status}")
    
    print(f"\nShape test: {'PASSED' if all_correct else 'FAILED'}")
    
    return all_correct


def test_model_params():
    """测试模型参数量"""
    print("\n" + "=" * 60)
    print("Model Parameter Statistics")
    print("=" * 60)
    
    model = CaptchaSolver()
    info = model.get_model_info()
    
    print(f"\nParameters by module:")
    print(f"  Backbone (ResNet18 Lite): {info['backbone_params']:,}")
    print(f"  Neck (UpConv):           {info['neck_params']:,}")
    print(f"  Heads (Detection):       {info['heads_params']:,}")
    print(f"  Total parameters:        {info['total_params']:,}")
    
    print(f"\nModel size: {info['model_size_mb']:.2f} MB")
    
    # 检查是否符合设计要求 (约3.5M参数)
    expected_params = 3.5e6
    tolerance = 0.5e6  # 允许0.5M的误差
    
    is_within_range = abs(info['total_params'] - expected_params) < tolerance
    status = "[OK]" if is_within_range else "[FAIL]"
    
    print(f"\nParameter count check (expected: ~3.5M): {status}")
    
    return is_within_range


def test_decode_function():
    """测试解码功能"""
    print("\n" + "=" * 60)
    print("Test Coordinate Decoding")
    print("=" * 60)
    
    model = CaptchaSolver()
    model.eval()
    
    # 创建测试输入
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 160, 320)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        coords = model.decode_outputs(outputs, K=1)
    
    print(f"\nDecoding results:")
    print(f"  Gap coordinates shape: {coords['gap_coords'].shape}")
    print(f"  Piece coordinates shape: {coords['piece_coords'].shape}")
    
    # 验证坐标范围
    gap_x, gap_y = coords['gap_coords'][0, 0].tolist()
    piece_x, piece_y = coords['piece_coords'][0, 0].tolist()
    
    print(f"\nExample coordinates:")
    print(f"  Gap: ({gap_x:.1f}, {gap_y:.1f})")
    print(f"  Piece: ({piece_x:.1f}, {piece_y:.1f})")
    
    # 检查坐标是否在合理范围内
    x_range = (0, 320)
    y_range = (0, 160)
    
    gap_valid = x_range[0] <= gap_x <= x_range[1] and y_range[0] <= gap_y <= y_range[1]
    piece_valid = x_range[0] <= piece_x <= x_range[1] and y_range[0] <= piece_y <= y_range[1]
    
    print(f"\nCoordinate range check:")
    print(f"  Gap coordinates valid: {'[OK]' if gap_valid else '[FAIL]'}")
    print(f"  Piece coordinates valid: {'[OK]' if piece_valid else '[FAIL]'}")
    
    return gap_valid and piece_valid


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "=" * 60)
    print("Test Gradient Flow")
    print("=" * 60)
    
    model = CaptchaSolver()
    model.train()
    
    # 创建测试输入
    input_tensor = torch.randn(1, 3, 160, 320, requires_grad=True)
    
    # 前向传播
    outputs = model(input_tensor)
    
    # 模拟损失
    loss = 0
    for key, value in outputs.items():
        loss = loss + value.mean()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_gradient = input_tensor.grad is not None and torch.any(input_tensor.grad != 0)
    
    print(f"\nInput gradient exists: {'[OK]' if has_gradient else '[FAIL]'}")
    
    # 检查模型参数梯度
    param_count = 0
    grad_count = 0
    
    for name, param in model.named_parameters():
        param_count += 1
        if param.grad is not None and torch.any(param.grad != 0):
            grad_count += 1
    
    print(f"Parameters with gradients: {grad_count}/{param_count}")
    
    all_have_grad = grad_count == param_count
    print(f"Gradient flow test: {'PASSED' if all_have_grad else 'FAILED'}")
    
    return all_have_grad


def main():
    """主测试函数"""
    print("Testing CNN Model Architecture (ResNet18 Lite + CenterNet)")
    
    tests = [
        ("Shape Test", test_model_shapes),
        ("Parameter Test", test_model_params),
        ("Decode Test", test_decode_function),
        ("Gradient Flow Test", test_gradient_flow)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n{test_name} Error: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name}: {status}")
        all_passed &= passed
    
    print(f"\nOverall Result: {'ALL TESTS PASSED [OK]' if all_passed else 'SOME TESTS FAILED [FAIL]'}")


if __name__ == "__main__":
    main()