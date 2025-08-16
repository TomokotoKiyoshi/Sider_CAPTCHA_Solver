"""
滑块验证码识别 API 演示程序
===========================
展示所有API功能的简单交互式演示
"""

import sys
import os
from pathlib import Path

# 添加包路径
sys.path.insert(0, str(Path(__file__).parent))

from sider_captcha_solver import solve, solve_batch, visualize, CaptchaSolver

# ==================== 配置区域 ====================
# 图片路径配置 - 分为校准集(calibration)和测试集(test)

# 校准图片集 - 用于验证模型准确性的已知样本
CALIBRATION_IMAGES = {
    'cal1': r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0001_Bgx106Bgy197_Sdx33Sdy197_890b7fc7.png",
    'cal2': r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0002_Bgx105Bgy39_Sdx32Sdy39_8a945d34.png", 
    'cal3': r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0003_Bgx150Bgy79_Sdx25Sdy79_4ec8494d.png",
    'cal_batch': [
        r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0001_Bgx107Bgy119_Sdx29Sdy119_734cd58b.png",
        r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0002_Bgx189Bgy162_Sdx21Sdy162_506dae4d.png",
        r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0003_Bgx151Bgy128_Sdx25Sdy128_76da0e8e.png"
    ]
}

# 测试图片集 - 用于实际测试的样本
TEST_IMAGES = {
    # 主要测试图片
    'main': r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0004_Bgx102Bgy39_Sdx36Sdy39_15a3fbb4.png",
    
    # 各个演示使用的图片
    'demo1': r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0004_Bgx103Bgy118_Sdx27Sdy118_44c4737d.png",
    'demo2': r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0005_Bgx102Bgy86_Sdx31Sdy86_0295eb83.png",
    'demo3_batch': [
        r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0004_Bgx106Bgy119_Sdx28Sdy119_db38c9f1.png",
        r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0005_Bgx103Bgy131_Sdx32Sdy131_cccd7aa6.png",
        r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0006_Bgx100Bgy37_Sdx36Sdy37_00fb7854.png"
    ],
    'demo4': r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0007_Bgx102Bgy39_Sdx26Sdy39_ec7e2894.png",
    'demo5': r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0007_Bgx103Bgy111_Sdx27Sdy111_c6355a83.png",
    'demo5_batch': [
        r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0006_Bgx101Bgy119_Sdx27Sdy119_0cb8ae94.png",
        r"D:\Hacker\Sider_CAPTCHA_Solver\data\captchas\Pic0007_Bgx104Bgy117_Sdx33Sdy117_5a9d01ef.png"
    ]
}
# ===================================================

def select_mode():
    """选择运行模式"""
    print("\n" + "="*50)
    print("    Sider CAPTCHA Solver - 模式选择")
    print("="*50)
    print("1. 校准模式 (Calibration Mode)")
    print("   - 使用已知答案的数据集")
    print("   - 验证模型准确性")
    print()
    print("2. 测试模式 (Test Mode)")
    print("   - 使用测试数据集")
    print("   - 实际应用测试")
    print("-"*50)
    
    while True:
        choice = input("\n请选择模式 (1-2): ").strip()
        if choice == '1':
            return 'calibration', CALIBRATION_IMAGES
        elif choice == '2':
            return 'test', TEST_IMAGES
        else:
            print("❌ 无效选择，请输入 1 或 2")


def print_menu(mode_name):
    """打印功能菜单"""
    print("\n" + "="*50)
    if mode_name == 'calibration':
        print("    Sider CAPTCHA Solver - 校准模式")
    else:
        print("    Sider CAPTCHA Solver - 测试模式")
    print("="*50)
    print("1. 快速求解 - 获取滑动距离")
    print("2. 详细求解 - 获取完整信息")
    print("3. 批量求解 - 处理多张图片")
    print("4. 可视化结果 - 显示标注图片")
    print("5. 使用类API - CaptchaSolver演示")
    print("6. 精度统计 - 计算误差分析")
    print("9. 切换模式")
    print("0. 退出")
    print("-"*50)


def demo_quick_solve(current_images, mode_name):
    """演示1: 快速求解"""
    print("\n【快速求解演示】")
    
    # 根据模式选择对应的图片键
    if mode_name == 'calibration':
        image_key = 'cal1'
    else:
        image_key = 'demo1'
    
    image_path = current_images[image_key]
    
    print(f"图片: {Path(image_path).name}")
    print("调用: distance = solve(image_path)")
    
    distance = solve(image_path)
    
    if distance is not None:
        print(f"✅ 滑动距离: {distance:.1f} 像素")
        # 从文件名解析真实值
        filename = Path(image_path).stem
        parts = filename.split('_')
        bgx = int(parts[1][3:])  # Bgx112 -> 112
        sdx = int(parts[2][3:])  # Sdx32 -> 32
        real_distance = bgx - sdx
        print(f"📏 真实距离: {real_distance} 像素")
        print(f"📊 误差: {abs(distance - real_distance):.1f} 像素")
    else:
        print("❌ 预测失败")


def demo_detailed_solve(current_images, mode_name):
    """演示2: 详细求解"""
    print("\n【详细求解演示】")
    
    # 根据模式选择对应的图片键
    if mode_name == 'calibration':
        image_key = 'cal2'
    else:
        image_key = 'demo2'
    
    image_path = current_images[image_key]
    
    print(f"图片: {Path(image_path).name}")
    print("调用: result = solve(image_path, detailed=True)")
    
    result = solve(image_path, detailed=True)
    
    if result:
        print("\n返回结果:")
        print(f"  滑动距离: {result['distance']:.1f} px")
        print(f"  缺口位置: {result['gap']}")
        print(f"  滑块位置: {result['slider']}")
        print(f"  缺口置信度: {result['gap_confidence']:.3f}")
        print(f"  滑块置信度: {result['slider_confidence']:.3f}")
        print(f"  综合置信度: {result['confidence']:.3f}")
        print(f"  处理时间: {result['time_ms']:.1f} ms")
    else:
        print("❌ 预测失败")


def demo_batch_solve(current_images, mode_name):
    """演示3: 批量求解"""
    print("\n【批量求解演示】")
    
    # 使用配置的批量图片
    if mode_name == 'calibration':
        images = current_images['cal_batch']
    else:
        images = current_images['demo3_batch']
    
    print(f"批量处理 {len(images)} 张图片...")
    print("调用: distances = solve_batch(images)")
    
    distances = solve_batch(images)
    
    print("\n结果:")
    for i, (img_path, distance) in enumerate(zip(images, distances), 1):
        filename = Path(img_path).stem
        if distance is not None:
            print(f"  图片{i}: {distance:.1f} px ({Path(img_path).name[:8]}...)")
        else:
            print(f"  图片{i}: 失败 ({Path(img_path).name[:8]}...)")
    
    # 统计
    success = [d for d in distances if d is not None]
    print(f"\n成功率: {len(success)}/{len(distances)} ({len(success)*100/len(distances):.0f}%)")
    if success:
        print(f"平均距离: {sum(success)/len(success):.1f} px")


def demo_visualize(current_images, mode_name):
    """演示4: 可视化"""
    print("\n【可视化演示】")
    
    # 根据模式选择对应的图片键
    if mode_name == 'calibration':
        image_key = 'cal3'
    else:
        image_key = 'demo4'
    
    image_path = current_images[image_key]
    
    print(f"图片: {Path(image_path).name}")
    print("调用: visualize(image_path, save_path='demo_vis.png')")
    
    # 保存可视化结果
    output_path = "demo_visualization.png"
    visualize(image_path, save_path=output_path, show=False)
    
    if Path(output_path).exists():
        print(f"✅ 可视化已保存: {output_path}")
        print("   (红框=缺口, 绿框=滑块)")
    else:
        print("❌ 可视化失败")


def demo_class_api(current_images, mode_name):
    """演示5: 使用类API"""
    print("\n【CaptchaSolver 类API演示】")
    
    print("创建求解器实例...")
    print("solver = CaptchaSolver()")
    solver = CaptchaSolver()
    
    # 单张求解
    if mode_name == 'calibration':
        image_path = current_images['cal1']
        batch_images = current_images['cal_batch']
    else:
        image_path = current_images['demo5']
        batch_images = current_images['demo5_batch']
    print(f"\n求解单张: {Path(image_path).name}")
    print("distance = solver.solve(image_path)")
    
    distance = solver.solve(image_path)
    if distance:
        print(f"✅ 滑动距离: {distance:.1f} px")
    
    # 获取详细信息
    print("\n获取详细信息:")
    print("details = solver.solve_detailed(image_path)")
    details = solver.solve_detailed(image_path)
    if details:
        print(f"  置信度: {details['confidence']:.3f}")
        print(f"  处理时间: {details['time_ms']:.1f} ms")
    
    # 批量处理
    print("\n批量处理:")
    print("results = solver.batch_solve(images)")
    results = solver.batch_solve(batch_images)
    for i, dist in enumerate(results, 1):
        if dist:
            print(f"  图片{i}: {dist:.1f} px")


def demo_accuracy_stats(current_images, mode_name):
    """演示6: 精度统计"""
    print("\n【精度统计分析】")
    print(f"当前模式: {'校准模式' if mode_name == 'calibration' else '测试模式'}")
    print("-" * 40)
    
    # 选择要测试的图片
    if mode_name == 'calibration':
        test_keys = ['cal1', 'cal2', 'cal3']
    else:
        test_keys = ['demo1', 'demo2', 'demo4', 'demo5']
    
    results = []
    errors = []
    
    for key in test_keys:
        if key not in current_images:
            continue
            
        image_path = current_images[key]
        filename = Path(image_path).stem
        
        # 从文件名解析真实值
        parts = filename.split('_')
        if len(parts) >= 3:
            try:
                bgx = int(parts[1][3:])  # Bgx值
                sdx = int(parts[2][3:])  # Sdx值  
                real_distance = bgx - sdx
                
                # 预测
                predicted_distance = solve(image_path)
                
                if predicted_distance is not None:
                    error = abs(predicted_distance - real_distance)
                    errors.append(error)
                    results.append({
                        'name': Path(image_path).name[:20],
                        'real': real_distance,
                        'predicted': predicted_distance,
                        'error': error
                    })
                    
                    status = "✅" if error < 2 else "⚠️" if error < 5 else "❌"
                    print(f"{status} {key}: 真实={real_distance:3d}px, 预测={predicted_distance:6.1f}px, 误差={error:5.2f}px")
            except:
                continue
    
    # 统计分析
    if errors:
        print("-" * 40)
        print(f"📊 统计结果:")
        print(f"  成功率: {len(errors)}/{len(test_keys)} ({len(errors)*100/len(test_keys):.0f}%)")
        print(f"  平均误差: {sum(errors)/len(errors):.2f}px")
        print(f"  最大误差: {max(errors):.2f}px")
        print(f"  最小误差: {min(errors):.2f}px")
        
        # 精度评级
        avg_error = sum(errors) / len(errors)
        if avg_error < 2.0:
            print("🎯 精度评级: 优秀 (MAE < 2px)")
        elif avg_error < 5.0:
            print("✅ 精度评级: 良好 (MAE < 5px)")
        else:
            print("⚠️ 精度评级: 需优化 (MAE ≥ 5px)")


def main():
    """主程序"""
    print("\n" + "🚀"*25)
    print("  欢迎使用 Sider CAPTCHA Solver API 演示")
    print("  工业级滑块验证码识别系统")
    print("🚀"*25)
    
    # 初始模式选择
    mode_name, current_images = select_mode()
    
    while True:
        print_menu(mode_name)
        
        try:
            choice = input("\n请选择功能 (0-6,9): ").strip()
            
            if choice == '0':
                print("\n👋 再见！")
                break
            elif choice == '1':
                demo_quick_solve(current_images, mode_name)
            elif choice == '2':
                demo_detailed_solve(current_images, mode_name)
            elif choice == '3':
                demo_batch_solve(current_images, mode_name)
            elif choice == '4':
                demo_visualize(current_images, mode_name)
            elif choice == '5':
                demo_class_api(current_images, mode_name)
            elif choice == '6':
                demo_accuracy_stats(current_images, mode_name)
            elif choice == '9':
                # 切换模式
                mode_name, current_images = select_mode()
            else:
                print("❌ 无效选择，请输入正确的选项")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序已中断")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            print("请确保模型文件存在于 src/checkpoints/1.1.0/best_model.pth")
    
    print("\n演示结束")


if __name__ == "__main__":
    main()