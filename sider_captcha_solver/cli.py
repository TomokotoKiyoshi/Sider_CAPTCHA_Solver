"""命令行接口 - 滑块验证码识别工具"""

import click
import json
from pathlib import Path
from typing import Optional
import sys

# 确保可以导入predictor
from .predictor import CaptchaPredictor
from .__version__ import __version__


@click.group()
@click.version_option(version=__version__, prog_name='sider-captcha')
def cli():
    """Sider CAPTCHA Solver - 工业级滑块验证码识别系统
    
    高精度CNN架构的滑块验证码识别解决方案
    """
    pass


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model', '-m', default='best', 
              help='模型路径或预设名称 (best/1.1.0/路径)')
@click.option('--device', '-d', default='auto',
              type=click.Choice(['auto', 'cuda', 'cpu']),
              help='运行设备')
@click.option('--format', '-f', default='json',
              type=click.Choice(['json', 'simple', 'detailed']),
              help='输出格式')
@click.option('--visualize', '-v', is_flag=True,
              help='可视化预测结果')
@click.option('--save-viz', '-s', type=click.Path(),
              help='保存可视化结果到指定路径')
def predict(image_path: str, 
           model: str,
           device: str,
           format: str,
           visualize: bool,
           save_viz: Optional[str]):
    """预测单张验证码图片的滑动距离
    
    示例:
        sider-captcha predict captcha.png
        sider-captcha predict captcha.png --model 1.1.0 --device cuda
        sider-captcha predict captcha.png --format simple
        sider-captcha predict captcha.png --visualize --save-viz result.png
    """
    try:
        # 初始化预测器
        click.echo(f"正在加载模型...", err=True)
        predictor = CaptchaPredictor(model_path=model, device=device)
        
        # 预测
        click.echo(f"正在处理图片: {image_path}", err=True)
        result = predictor.predict(image_path)
        
        # 检查预测是否成功
        if not result['success']:
            click.echo(f"❌ 预测失败: {result.get('error', 'Unknown error')}", err=True)
            sys.exit(1)
        
        # 格式化输出
        if format == 'json':
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        elif format == 'simple':
            click.echo(f"{result['sliding_distance']:.1f}")
        else:  # detailed
            click.echo(f"滑动距离: {result['sliding_distance']:.1f} px")
            click.echo(f"缺口位置: ({result['gap_x']:.1f}, {result['gap_y']:.1f})")
            click.echo(f"滑块位置: ({result['slider_x']:.1f}, {result['slider_y']:.1f})")
            click.echo(f"缺口置信度: {result['gap_confidence']:.3f}")
            click.echo(f"滑块置信度: {result['slider_confidence']:.3f}")
            click.echo(f"综合置信度: {result['confidence']:.3f}")
            click.echo(f"处理时间: {result['processing_time_ms']:.1f} ms")
        
        # 可视化
        if visualize or save_viz:
            predictor.visualize_prediction(
                image_path,
                save_path=save_viz,
                show=visualize
            )
            if save_viz:
                click.echo(f"✅ 可视化已保存: {save_viz}", err=True)
        
    except Exception as e:
        click.echo(f"❌ 错误: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--output', '-o', type=click.Path(),
              help='输出JSON文件路径')
@click.option('--model', '-m', default='best',
              help='模型路径或预设名称')
@click.option('--device', '-d', default='auto',
              type=click.Choice(['auto', 'cuda', 'cpu']),
              help='运行设备')
@click.option('--pattern', '-p', default='*.png',
              help='图片文件匹配模式')
@click.option('--visualize-dir', '-v', type=click.Path(),
              help='保存可视化结果的目录')
def batch(input_dir: str,
         output: Optional[str],
         model: str,
         device: str,
         pattern: str,
         visualize_dir: Optional[str]):
    """批量预测目录中的所有验证码图片
    
    示例:
        sider-captcha batch ./captchas/
        sider-captcha batch ./captchas/ --output results.json
        sider-captcha batch ./captchas/ --pattern "*.jpg"
        sider-captcha batch ./captchas/ --visualize-dir ./viz_results/
    """
    try:
        from tqdm import tqdm
        
        # 初始化预测器
        click.echo(f"正在加载模型...", err=True)
        predictor = CaptchaPredictor(model_path=model, device=device)
        
        # 查找所有图片
        input_path = Path(input_dir)
        image_files = list(input_path.glob(pattern))
        
        if not image_files:
            click.echo(f"❌ 未找到匹配的图片文件: {pattern}", err=True)
            sys.exit(1)
        
        click.echo(f"找到 {len(image_files)} 个图片文件", err=True)
        
        # 创建可视化目录
        if visualize_dir:
            viz_path = Path(visualize_dir)
            viz_path.mkdir(parents=True, exist_ok=True)
        
        # 批量预测
        results = []
        success_count = 0
        
        for image_file in tqdm(image_files, desc="批量预测"):
            result = predictor.predict(str(image_file))
            result['filename'] = image_file.name
            results.append(result)
            
            if result['success']:
                success_count += 1
                
                # 可视化
                if visualize_dir:
                    viz_file = viz_path / f"{image_file.stem}_viz.png"
                    predictor.visualize_prediction(
                        str(image_file),
                        save_path=str(viz_file),
                        show=False
                    )
        
        # 统计
        click.echo(f"\n预测完成: {success_count}/{len(image_files)} 成功", err=True)
        
        # 计算平均指标
        if success_count > 0:
            avg_distance = sum(r['sliding_distance'] for r in results if r['success']) / success_count
            avg_confidence = sum(r['confidence'] for r in results if r['success']) / success_count
            avg_time = sum(r['processing_time_ms'] for r in results if r['success']) / success_count
            
            click.echo(f"平均滑动距离: {avg_distance:.1f} px", err=True)
            click.echo(f"平均置信度: {avg_confidence:.3f}", err=True)
            click.echo(f"平均处理时间: {avg_time:.1f} ms", err=True)
        
        # 保存结果
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            click.echo(f"✅ 结果已保存: {output}", err=True)
        else:
            # 输出到stdout
            click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        click.echo(f"❌ 错误: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model', '-m', default='best',
              help='模型路径或预设名称')
@click.option('--device', '-d', default='auto',
              type=click.Choice(['auto', 'cuda', 'cpu']),
              help='运行设备')
@click.option('--output', '-o', type=click.Path(),
              help='保存热力图可视化路径')
def heatmap(image_path: str,
           model: str,
           device: str,
           output: Optional[str]):
    """可视化模型的热力图输出
    
    显示模型如何定位缺口和滑块的位置
    
    示例:
        sider-captcha heatmap captcha.png
        sider-captcha heatmap captcha.png --output heatmap.png
    """
    try:
        # 初始化预测器
        click.echo(f"正在加载模型...", err=True)
        predictor = CaptchaPredictor(model_path=model, device=device)
        
        # 生成热力图
        click.echo(f"正在生成热力图: {image_path}", err=True)
        predictor.visualize_heatmaps(
            image_path,
            save_path=output,
            show=(output is None)
        )
        
        if output:
            click.echo(f"✅ 热力图已保存: {output}", err=True)
        
    except Exception as e:
        click.echo(f"❌ 错误: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', default='best',
              help='模型路径或预设名称')
@click.option('--device', '-d', default='auto',
              type=click.Choice(['auto', 'cuda', 'cpu']),
              help='运行设备')
def info(model: str, device: str):
    """显示模型信息和系统配置
    
    示例:
        sider-captcha info
        sider-captcha info --model 1.1.0
    """
    try:
        import torch
        import platform
        
        click.echo("=" * 50)
        click.echo("Sider CAPTCHA Solver 系统信息")
        click.echo("=" * 50)
        
        # 版本信息
        click.echo(f"软件版本: {__version__}")
        click.echo(f"Python版本: {sys.version.split()[0]}")
        click.echo(f"PyTorch版本: {torch.__version__}")
        click.echo(f"操作系统: {platform.system()} {platform.release()}")
        
        # GPU信息
        if torch.cuda.is_available():
            click.echo(f"CUDA可用: ✅")
            click.echo(f"CUDA版本: {torch.version.cuda}")
            click.echo(f"GPU设备: {torch.cuda.get_device_name(0)}")
            click.echo(f"GPU数量: {torch.cuda.device_count()}")
        else:
            click.echo(f"CUDA可用: ❌")
        
        click.echo("")
        
        # 加载模型信息
        click.echo("正在加载模型信息...")
        predictor = CaptchaPredictor(model_path=model, device=device)
        
        # 模型信息
        click.echo("=" * 50)
        click.echo("模型信息")
        click.echo("=" * 50)
        click.echo(f"模型路径: {predictor._get_model_path(model)}")
        click.echo(f"运行设备: {predictor.device}")
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in predictor.model.parameters())
        trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
        
        click.echo(f"总参数量: {total_params:,}")
        click.echo(f"可训练参数: {trainable_params:,}")
        click.echo(f"模型大小: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        
        # 测试推理速度
        click.echo("")
        click.echo("正在测试推理速度...")
        
        import time
        import numpy as np
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (160, 320, 3), dtype=np.uint8)
        
        # 预热
        for _ in range(3):
            predictor.predict(test_image)
        
        # 测试
        times = []
        for _ in range(10):
            start = time.time()
            predictor.predict(test_image)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        click.echo(f"平均推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
        click.echo(f"推理帧率: {1000/avg_time:.1f} FPS")
        
    except Exception as e:
        click.echo(f"❌ 错误: {str(e)}", err=True)
        sys.exit(1)


def main():
    """主入口函数"""
    cli()


if __name__ == '__main__':
    main()