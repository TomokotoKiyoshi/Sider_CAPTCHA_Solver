import torch
import numpy as np
import cv2
from pathlib import Path
import time
import json
import sys

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.captcha_solver import CaptchaSolver



def benchmark_inference(device='cuda', num_runs=100, warmup_runs=10):
    """测试推理性能"""
    
    # 开始计时（整体时间）
    total_start_time = time.time()
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = torch.device('cpu')
    
    # 获取硬件信息
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        hardware_info = f"GPU: {gpu_name}"
    else:
        hardware_info = "CPU: AMD Ryzen 9 9950X"  # 硬编码CPU型号
    
    print(f"Running benchmark on: {device} ({hardware_info})")
    
    # 加载模型计时
    model_load_start = time.time()
    
    # 加载配置
    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / 'src' / 'checkpoints'
    # 使用正确的模型路径
    best_checkpoint_path = checkpoint_dir / 'best_model.pth'
    if not best_checkpoint_path.exists():
        print(f"Error: Model file not found at {best_checkpoint_path}")
        return
    
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    
    # 创建模型（使用CaptchaSolver）
    model = CaptchaSolver()
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model = model.to(device)
    
    model_load_time = time.time() - model_load_start
    
    # 准备测试图片
    image_prep_start = time.time()
    
    # 直接从data/captchas目录选择一张图片进行测试
    captchas_dir = project_root / 'data' / 'captchas'
    if not captchas_dir.exists():
        print(f"Captchas directory not found: {captchas_dir}")
        return
    
    # 获取第一张PNG图片
    captcha_files = list(captchas_dir.glob("*.png"))
    if not captcha_files:
        print("No captcha images found in data/captchas/")
        return
    
    # 使用第一张图片作为测试
    test_image_path = captcha_files[0]
    print(f"Using test image: {test_image_path.name}")
    
    # 加载和预处理图片
    image = cv2.imread(str(test_image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 160))
    image_tensor = torch.from_numpy(image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    image_prep_time = time.time() - image_prep_start
    
    # 预热运行
    print(f"Warming up with {warmup_runs} runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(image_tensor)
    
    # 同步CUDA（如果使用GPU）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 实际测试
    print(f"Running {num_runs} inference tests...")
    inference_times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            # 单次推理计时
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            # 推理
            outputs = model(image_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_runs} runs")
    
    # 后处理一次结果（解码预测）
    postprocess_start = time.time()
    
    # 解码预测结果（适配CaptchaSolver的输出格式）
    gap_heatmap = torch.sigmoid(outputs['gap_heatmap'][0, 0]).cpu().numpy()
    gap_offset = outputs['gap_offset'][0].cpu().numpy()
    piece_heatmap = torch.sigmoid(outputs['piece_heatmap'][0, 0]).cpu().numpy()
    piece_offset = outputs['piece_offset'][0].cpu().numpy()
    
    # 解码缺口位置
    if gap_heatmap.max() > 0.1:
        y, x = np.unravel_index(gap_heatmap.argmax(), gap_heatmap.shape)
        offset_x = gap_offset[0, y, x]
        offset_y = gap_offset[1, y, x]
        gap_center = ((x + offset_x) * 4, (y + offset_y) * 4)
    else:
        gap_center = None
    
    # 解码滑块位置
    if piece_heatmap.max() > 0.1:
        y, x = np.unravel_index(piece_heatmap.argmax(), piece_heatmap.shape)
        offset_x = piece_offset[0, y, x]
        offset_y = piece_offset[1, y, x]
        piece_center = ((x + offset_x) * 4, (y + offset_y) * 4)
    else:
        piece_center = None
    
    postprocess_time = time.time() - postprocess_start
    
    # 总时间
    total_time = time.time() - total_start_time
    
    # 统计结果
    inference_times = np.array(inference_times)
    
    print("\n" + "="*70)
    print(f"[*] Benchmark Results - {hardware_info}")
    print("="*70)
    
    print("\n[+] Startup Performance:")
    print(f"  - Model Loading Time: {model_load_time*1000:.2f} ms")
    print(f"  - Image Preprocessing Time: {image_prep_time*1000:.2f} ms")
    
    print(f"\n[+] Inference Performance (based on {num_runs} runs):")
    print(f"  - Mean Inference Time: {np.mean(inference_times):.2f} ms")
    print(f"  - Standard Deviation: {np.std(inference_times):.2f} ms")
    print(f"  - Minimum: {np.min(inference_times):.2f} ms")
    print(f"  - Maximum: {np.max(inference_times):.2f} ms")
    print(f"  - Median: {np.median(inference_times):.2f} ms")
    print(f"  - FPS (Frames Per Second): {1000/np.mean(inference_times):.2f}")
    
    print(f"\n[+] Post-processing Performance:")
    print(f"  - Decoding Time: {postprocess_time*1000:.2f} ms")
    
    print(f"\n[+] Overall Performance:")
    print(f"  - Total Pipeline Time: {total_time:.2f} seconds")
    print(f"  - Single Image End-to-End Time: {(model_load_time + image_prep_time + np.mean(inference_times)/1000 + postprocess_time)*1000:.2f} ms")
    
    # 验证预测准确性
    if test_image_path:
        filename = test_image_path.name
        # 从文件名提取真实坐标
        import re
        match = re.search(r'Bgx(\d+)Bgy(\d+)_Sdx(\d+)Sdy(\d+)', filename)
        if match:
            true_gap_x, true_gap_y = int(match.group(1)), int(match.group(2))
            true_slider_x, true_slider_y = int(match.group(3)), int(match.group(4))
            
            print(f"\n[+] Prediction Accuracy:")
            print(f"  - Test Image: {filename}")
            print(f"  - Ground Truth: Gap({true_gap_x}, {true_gap_y}), Slider({true_slider_x}, {true_slider_y})")
            if gap_center and piece_center:
                gap_error = np.sqrt((gap_center[0] - true_gap_x)**2 + (gap_center[1] - true_gap_y)**2)
                slider_error = np.sqrt((piece_center[0] - true_slider_x)**2 + (piece_center[1] - true_slider_y)**2)
                slide_distance_true = true_gap_x - true_slider_x
                slide_distance_pred = gap_center[0] - piece_center[0]
                slide_error = abs(slide_distance_pred - slide_distance_true)
                
                print(f"  - Predicted: Gap({gap_center[0]:.1f}, {gap_center[1]:.1f}), Slider({piece_center[0]:.1f}, {piece_center[1]:.1f})")
                print(f"  - Gap Localization Error: {gap_error:.2f} pixels")
                print(f"  - Slider Localization Error: {slider_error:.2f} pixels")
                print(f"  - Sliding Distance Error: {slide_error:.2f} pixels (GT:{slide_distance_true}, Pred:{slide_distance_pred:.1f})")
    
    print("\n" + "="*70)
    
    # 保存结果
    # 准备预测准确性数据
    accuracy_data = {}
    if test_image_path and gap_center and piece_center:
        filename = test_image_path.name
        import re
        match = re.search(r'Bgx(\d+)Bgy(\d+)_Sdx(\d+)Sdy(\d+)', filename)
        if match:
            true_gap_x, true_gap_y = int(match.group(1)), int(match.group(2))
            true_slider_x, true_slider_y = int(match.group(3)), int(match.group(4))
            
            gap_error = np.sqrt((gap_center[0] - true_gap_x)**2 + (gap_center[1] - true_gap_y)**2)
            slider_error = np.sqrt((piece_center[0] - true_slider_x)**2 + (piece_center[1] - true_slider_y)**2)
            slide_distance_true = true_gap_x - true_slider_x
            slide_distance_pred = gap_center[0] - piece_center[0]
            slide_error = abs(slide_distance_pred - slide_distance_true)
            
            accuracy_data = {
                'ground_truth': {
                    'gap': [true_gap_x, true_gap_y],
                    'slider': [true_slider_x, true_slider_y],
                    'slide_distance': slide_distance_true
                },
                'prediction': {
                    'gap': [float(gap_center[0]), float(gap_center[1])],
                    'slider': [float(piece_center[0]), float(piece_center[1])],
                    'slide_distance': float(slide_distance_pred)
                },
                'errors': {
                    'gap_error_pixels': float(gap_error),
                    'slider_error_pixels': float(slider_error),
                    'slide_distance_error_pixels': float(slide_error)
                }
            }
    
    results = {
        'hardware': hardware_info,
        'device': str(device),
        'num_runs': num_runs,
        'model_load_time_ms': model_load_time * 1000,
        'image_prep_time_ms': image_prep_time * 1000,
        'postprocess_time_ms': postprocess_time * 1000,
        'inference_mean_ms': float(np.mean(inference_times)),
        'inference_std_ms': float(np.std(inference_times)),
        'inference_min_ms': float(np.min(inference_times)),
        'inference_max_ms': float(np.max(inference_times)),
        'inference_median_ms': float(np.median(inference_times)),
        'fps': float(1000/np.mean(inference_times)),
        'total_time_seconds': total_time,
        'end_to_end_time_ms': (model_load_time + image_prep_time + np.mean(inference_times)/1000 + postprocess_time)*1000,
        'test_image': test_image_path.name if test_image_path else None,
        'accuracy': accuracy_data,
        'metric_explanations': {
            'model_load_time_ms': 'Time to load model weights from disk to memory/VRAM',
            'image_prep_time_ms': 'Time for image loading, color conversion, resizing, etc.',
            'inference_mean_ms': 'Average neural network forward pass time, the most important metric',
            'inference_std_ms': 'Variance in inference time, lower is more stable',
            'inference_min_ms': 'Fastest inference, close to hardware limit',
            'inference_max_ms': 'Slowest inference, possibly affected by system scheduling',
            'inference_median_ms': 'Typical inference time excluding outliers',
            'fps': 'Number of CAPTCHAs processable per second, higher is better',
            'postprocess_time_ms': 'Time to decode coordinates from model output',
            'end_to_end_time_ms': 'Complete processing time including all steps'
        }
    }
    
    # 确保logs目录存在
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    with open(logs_dir / f'benchmark_results_{device.type}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {logs_dir / f'benchmark_results_{device.type}.json'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark inference performance')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run benchmark on (default: cuda)')
    parser.add_argument('--num_runs', type=int, default=100,
                        help='Number of inference runs (default: 100)')
    parser.add_argument('--warmup_runs', type=int, default=10,
                        help='Number of warmup runs (default: 10)')
    
    args = parser.parse_args()
    
    benchmark_inference(device=args.device, num_runs=args.num_runs, warmup_runs=args.warmup_runs)