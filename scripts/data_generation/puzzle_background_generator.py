# -*- coding: utf-8 -*-
"""
拼图背景生成器 - 生成白底复杂图案背景图
使用81种拼图形状和多种异形形状组合
包含光照渲染效果
"""
import numpy as np
import cv2
import random
from datetime import datetime
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.captcha_generator.simple_puzzle_generator import create_puzzle_piece, get_all_puzzle_combinations
from src.captcha_generator.lighting_effects import apply_gap_lighting
from src.captcha_generator.slider_effects import apply_slider_lighting

# 设置图像尺寸
WIDTH = 320
HEIGHT = 160
OUTPUT_DIR = project_root / "data" / "raw" / "Puzzle_Backgrounds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 特殊形状列表
SPECIAL_SHAPES = ['circle', 'square', 'triangle', 'hexagon', 'pentagon', 'star', 
                  'heart', 'diamond', 'cross', 'arrow', 'moon', 'cloud']


def generate_special_shape_extended(shape_type: str, size: int = 60) -> np.ndarray:
    """生成扩展的特殊形状"""
    # 创建透明画布
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    
    if shape_type == 'circle':
        cv2.circle(canvas, (center, center), int(size * 0.4), (255, 255, 255, 255), -1)
        
    elif shape_type == 'square':
        s = int(size * 0.6)
        top_left = (center - s // 2, center - s // 2)
        bottom_right = (center + s // 2, center + s // 2)
        cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255, 255), -1)
        
    elif shape_type == 'triangle':
        pts = np.array([
            [center, int(center - size * 0.35)],
            [int(center - size * 0.3), int(center + size * 0.25)],
            [int(center + size * 0.3), int(center + size * 0.25)]
        ], np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    elif shape_type == 'hexagon':
        # 六边形
        angles = np.linspace(0, 2 * np.pi, 7)
        radius = size * 0.4
        pts = np.array([[int(center + radius * np.cos(a)), 
                        int(center + radius * np.sin(a))] for a in angles], np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    elif shape_type == 'pentagon':
        # 五边形
        angles = np.linspace(-np.pi/2, 3*np.pi/2, 6)
        radius = size * 0.4
        pts = np.array([[int(center + radius * np.cos(a)), 
                        int(center + radius * np.sin(a))] for a in angles], np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    elif shape_type == 'star':
        # 五角星
        outer_radius = size * 0.4
        inner_radius = size * 0.2
        pts = []
        for i in range(10):
            angle = i * np.pi / 5 - np.pi / 2
            radius = outer_radius if i % 2 == 0 else inner_radius
            pts.append([int(center + radius * np.cos(angle)), 
                       int(center + radius * np.sin(angle))])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    elif shape_type == 'heart':
        # 心形
        t = np.linspace(0, 2 * np.pi, 100)
        x = 16 * np.sin(t)**3
        y = -13 * np.cos(t) + 5 * np.cos(2*t) + 2 * np.cos(3*t) + np.cos(4*t)
        # 缩放和平移
        x = (x / 20 * size * 0.35 + center).astype(int)
        y = (y / 20 * size * 0.35 + center).astype(int)
        pts = np.column_stack((x, y))
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    elif shape_type == 'diamond':
        # 菱形
        pts = np.array([
            [center, int(center - size * 0.4)],
            [int(center + size * 0.3), center],
            [center, int(center + size * 0.4)],
            [int(center - size * 0.3), center]
        ], np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    elif shape_type == 'cross':
        # 十字形
        thickness = int(size * 0.2)
        # 横条
        cv2.rectangle(canvas, 
                     (int(center - size * 0.35), center - thickness // 2),
                     (int(center + size * 0.35), center + thickness // 2),
                     (255, 255, 255, 255), -1)
        # 竖条
        cv2.rectangle(canvas, 
                     (center - thickness // 2, int(center - size * 0.35)),
                     (center + thickness // 2, int(center + size * 0.35)),
                     (255, 255, 255, 255), -1)
        
    elif shape_type == 'arrow':
        # 箭头
        pts = np.array([
            [center, int(center - size * 0.4)],
            [int(center + size * 0.2), int(center - size * 0.1)],
            [int(center + size * 0.1), int(center - size * 0.1)],
            [int(center + size * 0.1), int(center + size * 0.3)],
            [int(center - size * 0.1), int(center + size * 0.3)],
            [int(center - size * 0.1), int(center - size * 0.1)],
            [int(center - size * 0.2), int(center - size * 0.1)]
        ], np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255, 255))
        
    elif shape_type == 'moon':
        # 月牙形
        # 画大圆
        cv2.circle(canvas, (center, center), int(size * 0.4), (255, 255, 255, 255), -1)
        # 用小圆切掉一部分
        cv2.circle(canvas, (int(center + size * 0.15), center), 
                  int(size * 0.35), (0, 0, 0, 0), -1)
        
    elif shape_type == 'cloud':
        # 云朵形
        # 用多个圆组合
        circles = [
            (int(center - size * 0.15), int(center + size * 0.05), int(size * 0.2)),
            (center, int(center - size * 0.05), int(size * 0.25)),
            (int(center + size * 0.15), int(center + size * 0.05), int(size * 0.2)),
            (int(center - size * 0.25), int(center + size * 0.1), int(size * 0.15)),
            (int(center + size * 0.25), int(center + size * 0.1), int(size * 0.15)),
        ]
        for cx, cy, r in circles:
            cv2.circle(canvas, (cx, cy), r, (255, 255, 255, 255), -1)
    
    return canvas




def generate_puzzle_background(index, total_count=400):
    """生成单张拼图背景"""
    # 创建白色背景
    background = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    
    # 创建一个用于记录已经处理过的区域的掩码
    processed_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    
    # 获取所有拼图组合
    all_puzzle_shapes = get_all_puzzle_combinations()
    
    # 设置随机种子
    seed = int(time.time() * 1000 + index) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    
    # 计算需要填充的面积（90-95%）
    total_area = WIDTH * HEIGHT
    target_fill_area = total_area * random.uniform(0.90, 0.95)
    current_fill_area = 0
    
    # 生成拼图块并放置
    placed_pieces = []
    attempts_without_placement = 0
    max_attempts_without_placement = 100
    
    # 创建多种大小的形状，增加更多中等尺寸
    sizes = [70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15]
    
    # 首先放置一些大的形状作为基础
    initial_large_pieces = random.randint(12, 18)
    for _ in range(initial_large_pieces):
        size = random.choice(sizes[:3])  # 使用较大的尺寸
        
        # 随机选择形状
        if random.random() < 0.7:
            shape_type = 'puzzle'
            shape_data = random.choice(all_puzzle_shapes)
        else:
            shape_type = 'special'
            shape_data = random.choice(SPECIAL_SHAPES)
        
        # 生成形状
        if shape_type == 'puzzle':
            piece = create_puzzle_piece(
                piece_size=size,
                knob_radius=int(size * 0.2),
                edges=shape_data
            )
        else:
            piece = generate_special_shape_extended(shape_data, size)
        
        # 应用随机颜色（模拟从背景图片中提取）
        base_color = np.array([
            random.randint(50, 200),   # B
            random.randint(50, 200),   # G  
            random.randint(50, 200)    # R
        ])
        
        mask = piece[:, :, 3] > 0
        for c in range(3):
            piece[:, :, c] = np.where(mask, base_color[c], piece[:, :, c])
        
        # 所有形状都使用slider lighting
        piece = apply_slider_lighting(
            piece,
            edge_highlight=random.randint(60, 100),      # 边缘高光强度
            directional_highlight=random.randint(30, 50), # 方向性高光
            edge_width=random.randint(4, 8),             # 边缘宽度
            decay_factor=random.uniform(1.5, 2.5)        # 衰减系数
        )
        
        # 第一批大形状使用网格化放置策略
        placed = False
        for attempt in range(100):
            if len(placed_pieces) == 0:
                # 第一个形状完全随机
                x = random.randint(0, max(0, WIDTH - piece.shape[1]))
                y = random.randint(0, max(0, HEIGHT - piece.shape[0]))
            else:
                # 后续形状尝试靠近已放置的形状
                if random.random() < 0.7:
                    # 70%的概率尝试放在现有形状附近
                    ref_piece = random.choice(placed_pieces)
                    ref_x, ref_y, ref_w, ref_h = ref_piece
                    
                    # 在参考形状周围选择位置
                    offset = random.randint(-20, 20)
                    if random.random() < 0.5:
                        x = max(0, min(WIDTH - piece.shape[1], ref_x + ref_w + offset))
                        y = max(0, min(HEIGHT - piece.shape[0], ref_y + random.randint(-30, 30)))
                    else:
                        x = max(0, min(WIDTH - piece.shape[1], ref_x + random.randint(-30, 30)))
                        y = max(0, min(HEIGHT - piece.shape[0], ref_y + ref_h + offset))
                else:
                    # 30%的概率完全随机
                    x = random.randint(0, max(0, WIDTH - piece.shape[1]))
                    y = random.randint(0, max(0, HEIGHT - piece.shape[0]))
            
            # 检查重叠（第一批大形状也要检查）
            overlap = False
            for px, py, pw, ph in placed_pieces:
                if not (x + piece.shape[1] <= px or x >= px + pw or 
                       y + piece.shape[0] <= py or y >= py + ph):
                    overlap = True
                    break
            
            if not overlap:
                background = apply_puzzle_to_background(background, piece, x, y, apply_lighting=True)
                placed_pieces.append((x, y, piece.shape[1], piece.shape[0]))
                
                alpha_mask = piece[:, :, 3] > 128
                piece_fill_area = np.sum(alpha_mask)
                current_fill_area += piece_fill_area
                placed = True
                break
    
    # 在空缺的地方填充小形状
    filled_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for px, py, pw, ph in placed_pieces:
        filled_mask[py:py+ph, px:px+pw] = True
    
    # 继续填充直到达到目标
    while current_fill_area < target_fill_area and attempts_without_placement < max_attempts_without_placement:
        # 动态选择大小，根据填充率调整
        fill_rate = current_fill_area / target_fill_area
        if fill_rate < 0.6:
            # 前60%使用中大形状
            size = random.choice(sizes[1:5])
        elif fill_rate < 0.85:
            # 60-85%使用中小形状
            size = random.choice(sizes[4:8])
        else:
            # 最后15%使用小形状填补细节
            size = random.choice(sizes[7:])
        
        # 随机选择形状类型（70%普通拼图形状，30%特殊形状）
        if random.random() < 0.7:
            shape_type = 'puzzle'
            shape_data = random.choice(all_puzzle_shapes)
        else:
            shape_type = 'special'
            shape_data = random.choice(SPECIAL_SHAPES)
        
        # 生成形状
        if shape_type == 'puzzle':
            piece = create_puzzle_piece(
                piece_size=size,
                knob_radius=int(size * 0.2),
                edges=shape_data
            )
        else:
            piece = generate_special_shape_extended(shape_data, size)
        
        # 应用随机颜色（模拟从背景图片中提取）
        # 使用更真实的颜色范围
        base_color = np.array([
            random.randint(50, 200),   # B
            random.randint(50, 200),   # G  
            random.randint(50, 200)    # R
        ])
        
        # 为拼图块着色
        mask = piece[:, :, 3] > 0
        for c in range(3):
            piece[:, :, c] = np.where(mask, base_color[c], piece[:, :, c])
        
        # 不需要额外纹理，直接应用滑块光照效果
        # 所有形状都使用slider lighting
        piece = apply_slider_lighting(
            piece,
            edge_highlight=random.randint(60, 100),      # 边缘高光强度
            directional_highlight=random.randint(30, 50), # 方向性高光
            edge_width=random.randint(4, 8),             # 边缘宽度
            decay_factor=random.uniform(1.5, 2.5)        # 衰减系数
        )
        
        # 尝试放置拼图块（使用更智能的放置策略）
        max_placement_attempts = 100
        placed = False
        
        # 找出空缺区域并优先填充
        empty_regions = []
        scan_step = 10  # 扫描步长
        
        for y_scan in range(0, HEIGHT - size, scan_step):
            for x_scan in range(0, WIDTH - size, scan_step):
                # 检查这个区域是否有足够的空间
                if x_scan + size <= WIDTH and y_scan + size <= HEIGHT:
                    empty_area = np.sum(~filled_mask[y_scan:y_scan+size, x_scan:x_scan+size])
                    if empty_area > size * size * 0.8:  # 至少80%是空的
                        empty_regions.append((x_scan, y_scan, empty_area))
        
        # 按空缺面积排序，优先填充大的空缺
        empty_regions.sort(key=lambda x: x[2], reverse=True)
        
        # 如果没有找到合适的空缺，使用随机位置
        if not empty_regions:
            grid_positions = [(random.randint(0, WIDTH-size), random.randint(0, HEIGHT-size)) 
                            for _ in range(50)]
        else:
            # 优先在空缺区域周围尝试
            grid_positions = []
            for ex, ey, _ in empty_regions[:10]:  # 只看前10个最大的空缺
                for _ in range(5):
                    offset_x = random.randint(-10, 10)
                    offset_y = random.randint(-10, 10)
                    grid_positions.append((ex + offset_x, ey + offset_y))
        
        for base_x, base_y in grid_positions:
            if placed:
                break
            
            # 在每个位置周围尝试多次
            for _ in range(3):
                # 添加随机偏移
                x = base_x + random.randint(-5, 5)
                y = base_y + random.randint(-5, 5)
                
                # 确保在边界内
                if x < 0 or y < 0 or x + piece.shape[1] > WIDTH or y + piece.shape[0] > HEIGHT:
                    continue
                
                # 检查是否与已放置的拼图重叠（不允许重叠）
                overlap = False
                
                for px, py, pw, ph in placed_pieces:
                    # 检查是否有重叠（不允许任何重叠）
                    if not (x + piece.shape[1] <= px or x >= px + pw or 
                           y + piece.shape[0] <= py or y >= py + ph):
                        overlap = True
                        break
                
                if not overlap:
                    # 放置拼图块，应用光照效果
                    background = apply_puzzle_to_background(background, piece, x, y, apply_lighting=True)
                    placed_pieces.append((x, y, piece.shape[1], piece.shape[0]))
                    
                    # 更新填充遮罩
                    filled_mask[y:y+piece.shape[0], x:x+piece.shape[1]] = True
                    
                    # 计算实际填充的面积（考虑alpha通道）
                    alpha_mask = piece[:, :, 3] > 128
                    piece_fill_area = np.sum(alpha_mask)
                    current_fill_area += piece_fill_area
                    
                    placed = True
                    attempts_without_placement = 0
                    break
        
        if not placed:
            attempts_without_placement += 1
    
    # 添加背景纹理和效果
    background = add_background_effects(background)
    
    # 计算实际填充率
    actual_fill_percentage = (current_fill_area / total_area) * 100
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"puzzle_bg_{index:04d}_{timestamp}.png"
    
    # 打印填充率信息（调试用）
    if index % 10 == 0:
        print(f"Image {index}: Fill rate = {actual_fill_percentage:.1f}%, Pieces = {len(placed_pieces)}")
    
    return background, filename




def apply_puzzle_to_background(background, piece, x, y, apply_lighting=True):
    """将拼图块应用到背景上，使用正确的光照效果"""
    h, w = piece.shape[:2]
    alpha = piece[:, :, 3]
    
    if apply_lighting:
        # 应用缺口光照效果（类似apply_gap_lighting但适配白色背景）
        # 在白色背景上，缺口应该显示为有深度的凹陷
        background = apply_gap_lighting_on_white(background, x, y, alpha, h, w)
    else:
        # 直接将拼图块合成到背景上
        bg_region = background[y:y+h, x:x+w]
        piece_alpha = alpha / 255.0
        
        for c in range(3):
            bg_region[:, :, c] = (
                piece[:, :, c] * piece_alpha + 
                bg_region[:, :, c] * (1 - piece_alpha)
            ).astype(np.uint8)
    
    return background


def apply_gap_lighting_on_white(background, x, y, mask_alpha, mask_h, mask_w):
    """为白色背景上的拼图缺口应用光照效果"""
    # 在白色背景上，缺口应该表现为阴影和深度
    result = background.copy()
    
    # 使用apply_gap_lighting的逻辑，但调整参数以适应白色背景
    # 白色背景需要更强的阴影效果
    result = apply_gap_lighting(
        result, x, y, mask_alpha, mask_h, mask_w,
        base_darkness=30,      # 基础暗度（增加）
        edge_darkness=50,      # 边缘暗度（增加）
        directional_darkness=25,  # 方向性暗度（增加）
        outer_edge_darkness=0    # 外边缘暗度设置为0，按您的建议
    )
    
    return result


def add_background_effects(background):
    """添加背景效果"""
    h, w = background.shape[:2]
    
    # 1. 添加微妙的渐变
    gradient_y = np.linspace(0.95, 1.0, h)[:, np.newaxis]
    gradient_x = np.linspace(0.95, 1.0, w)[np.newaxis, :]
    gradient = gradient_y * gradient_x
    
    for c in range(3):
        background[:, :, c] = (background[:, :, c] * gradient).astype(np.uint8)
    
    # 2. 添加轻微的噪点
    noise = np.random.normal(0, 3, (h, w, 3))
    background = np.clip(background.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # 3. 添加轻微的模糊
    background = cv2.GaussianBlur(background, (3, 3), 0.5)
    
    return background


def generate_single_background(args):
    """生成单张背景（用于多进程）"""
    index, total_count, output_dir = args
    
    try:
        # 生成背景
        background, filename = generate_puzzle_background(index, total_count)
        
        # 保存图片
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), background)
        
        return (True, filename, None)
    except Exception as e:
        return (False, None, str(e))


def generate_batch(count=400, output_dir=None, num_workers=None):
    """批量生成拼图背景"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    print(f"Generating {count} puzzle backgrounds to: {output_dir}")
    print(f"Using {num_workers} worker processes")
    print("-" * 50)
    
    # 准备任务
    tasks = [(i, count, output_dir) for i in range(count)]
    
    success_count = 0
    errors = []
    start_time = time.time()
    
    # Windows平台使用顺序执行
    if os.name == 'nt':
        print("Note: Using sequential mode on Windows")
        for i, task in enumerate(tasks):
            success, filename, error = generate_single_background(task)
            
            if success:
                success_count += 1
            else:
                errors.append(f"Image {i}: {error}")
            
            # 每10张显示一次进度
            if (i + 1) % 10 == 0 or (i + 1) == count:
                progress = (i + 1) / count * 100
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * count - elapsed if i > 0 else 0
                
                print(f"Progress: {i+1}/{count} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")
    else:
        # 其他平台使用多进程
        with Pool(num_workers) as pool:
            results = pool.map(generate_single_background, tasks)
            
            for i, (success, filename, error) in enumerate(results):
                if success:
                    success_count += 1
                else:
                    errors.append(f"Image {i}: {error}")
                
                # 每10张显示一次进度
                if (i + 1) % 10 == 0 or (i + 1) == count:
                    progress = (i + 1) / count * 100
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + 1) * count - elapsed if i > 0 else 0
                    
                    print(f"Progress: {i+1}/{count} ({progress:.1f}%) - "
                          f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")
    
    # 完成统计
    total_time = time.time() - start_time
    print("-" * 50)
    print(f"Generation complete!")
    print(f"Success: {success_count}/{count}")
    if errors:
        print(f"Errors: {len(errors)}")
        if len(errors) <= 10:
            for error in errors:
                print(f"  - {error}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per image: {total_time/count:.2f}s")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Puzzle Background Generator')
    parser.add_argument('--count', type=int, default=400,
                        help='Number of backgrounds to generate (default: 400)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: data/raw/Puzzle_Backgrounds)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto)')
    parser.add_argument('--preview', action='store_true',
                        help='Generate and preview a single image')
    
    args = parser.parse_args()
    
    if args.preview:
        # 预览模式
        print("Generating preview...")
        background, filename = generate_puzzle_background(0)
        
        # 保存预览
        preview_path = OUTPUT_DIR / "preview.png"
        cv2.imwrite(str(preview_path), background)
        print(f"Preview saved to: {preview_path}")
        
        # 尝试显示
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title('Puzzle Background Preview')
            plt.tight_layout()
            plt.show()
        except:
            print("Could not display preview. Please check the saved file.")
    else:
        # 批量生成模式
        generate_batch(count=args.count, output_dir=args.output, num_workers=args.workers)