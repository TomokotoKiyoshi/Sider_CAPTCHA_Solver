import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import random
from datetime import datetime
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

# 设置图像尺寸
WIDTH = 1920
HEIGHT = 1080
DPI = 100

# 设置输出目录
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "Geometric_Generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 白色背景版本的输出目录
OUTPUT_DIR_WHITE = Path(__file__).parent.parent.parent / "data" / "raw" / "Geometric_Generated_white"
OUTPUT_DIR_WHITE.mkdir(parents=True, exist_ok=True)

# 生成随机颜色
def random_color(alpha=1.0):
    return (random.random(), random.random(), random.random(), alpha)

# 生成高斯噪声
def add_gaussian_noise(image_array, strength=20):
    """添加高斯噪声到图像数组"""
    noise = np.random.normal(0, strength/255, image_array.shape)
    noisy_image = image_array + noise
    return np.clip(noisy_image, 0, 1)

# 绘制圆形
def draw_circle(ax, x, y, radius, color, alpha=0.5):
    # 添加一些随机变形
    theta = np.linspace(0, 2*np.pi, 60)
    r = radius * (1 + 0.1 * np.sin(5*theta) + 0.1 * np.random.random(len(theta)))
    x_points = x + r * np.cos(theta)
    y_points = y + r * np.sin(theta)
    
    circle = plt.Polygon(list(zip(x_points, y_points)), 
                        facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_patch(circle)

# 绘制三角形
def draw_triangle(ax, x, y, size, rotation, color, alpha=0.5):
    angles = np.array([0, 120, 240]) + rotation
    points = [(x + size * np.cos(np.radians(angle)), 
               y + size * np.sin(np.radians(angle))) for angle in angles]
    triangle = plt.Polygon(points, facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_patch(triangle)

# 绘制正方形
def draw_square(ax, x, y, size, rotation, color, alpha=0.5):
    square = patches.Rectangle((x - size/2, y - size/2), size, size,
                              angle=rotation, facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_patch(square)

# 绘制五边形
def draw_pentagon(ax, x, y, size, rotation, color, alpha=0.5):
    angles = np.array([0, 72, 144, 216, 288]) + rotation
    points = [(x + size * np.cos(np.radians(angle)), 
               y + size * np.sin(np.radians(angle))) for angle in angles]
    pentagon = plt.Polygon(points, facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_patch(pentagon)

# 绘制六边形
def draw_hexagon(ax, x, y, size, rotation, color, alpha=0.5):
    angles = np.array([0, 60, 120, 180, 240, 300]) + rotation
    points = [(x + size * np.cos(np.radians(angle)), 
               y + size * np.sin(np.radians(angle))) for angle in angles]
    hexagon = plt.Polygon(points, facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_patch(hexagon)

# 绘制梯形
def draw_trapezoid(ax, x, y, size, rotation, color, alpha=0.5):
    # 创建梯形的四个顶点
    points = [(-size*0.7, -size*0.5), (size*0.7, -size*0.5), 
              (size*0.5, size*0.5), (-size*0.5, size*0.5)]
    
    # 旋转点
    cos_r = np.cos(np.radians(rotation))
    sin_r = np.sin(np.radians(rotation))
    rotated_points = []
    for px, py in points:
        rx = px * cos_r - py * sin_r + x
        ry = px * sin_r + py * cos_r + y
        rotated_points.append((rx, ry))
    
    trapezoid = plt.Polygon(rotated_points, facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_patch(trapezoid)

# 绘制螺旋线
def draw_spiral(ax, x, y, max_radius, color, alpha=0.6):
    turns = 5 + random.random() * 5
    t = np.linspace(0, 1, 200)
    angle = t * np.pi * 2 * turns
    radius = t * max_radius
    spiral_x = x + radius * np.cos(angle)
    spiral_y = y + radius * np.sin(angle)
    ax.plot(spiral_x, spiral_y, color=color[:3], alpha=alpha, linewidth=2)

# 绘制贝塞尔曲线网络
def draw_bezier_network(ax, points, color, alpha=0.3):
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            distance = np.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1])
            if distance < 600:  # 增加连接距离以适应1920x1080画布
                # 创建贝塞尔曲线的控制点
                t = np.linspace(0, 1, 100)
                cx1 = points[i][0] + (points[j][0] - points[i][0]) * 0.3
                cy1 = points[i][1] + random.uniform(-100, 100)
                cx2 = points[j][0] - (points[j][0] - points[i][0]) * 0.3
                cy2 = points[j][1] + random.uniform(-100, 100)
                
                # 贝塞尔曲线方程
                x = ((1-t)**3 * points[i][0] + 3*(1-t)**2*t * cx1 + 
                     3*(1-t)*t**2 * cx2 + t**3 * points[j][0])
                y = ((1-t)**3 * points[i][1] + 3*(1-t)**2*t * cy1 + 
                     3*(1-t)*t**2 * cy2 + t**3 * points[j][1])
                
                ax.plot(x, y, color=color[:3], alpha=alpha, linewidth=1)

# 生成适合白色背景的随机颜色（较深的颜色）
def random_color_for_white_bg(alpha=1.0):
    # 生成较深的颜色，确保在白色背景上清晰可见
    return (random.random() * 0.7, random.random() * 0.7, random.random() * 0.7, alpha)

# 主函数生成图案
def generate_pattern(noise_strength=20, output_dir=None):
    """
    生成密集的几何图案
    总计约1500+个图形元素：
    - 贝塞尔网络: 60个节点
    - 螺旋线: 35个
    - 大型背景图形: 55个
    - 网格分布中型图形: 144-240个
    - 小型图形: 300个
    - 微小图形: 350个
    - 装饰点: 800个
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal')
    
    # 设置深色背景
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    
    # 创建径向渐变背景（通过多个半透明圆形模拟）
    for i in range(20):
        radius = WIDTH * (1 - i/20)
        alpha = 0.05
        color = (0.08 + i*0.02/20, 0.08 + i*0.02/20, 0.2 + i*0.03/20)
        circle = plt.Circle((WIDTH/2, HEIGHT/2), radius, 
                          facecolor=color, alpha=alpha, edgecolor='none')
        ax.add_patch(circle)
    
    # 生成贝塞尔网络的随机点（增加密度）
    network_points = [(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)) 
                      for _ in range(35)]
    draw_bezier_network(ax, network_points, random_color(), 0.4)
    
    # 额外的贝塞尔网络层
    network_points2 = [(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)) 
                       for _ in range(25)]
    draw_bezier_network(ax, network_points2, random_color(), 0.3)
    
    # 绘制螺旋线（大幅增加数量和变化）
    # 大螺旋
    for _ in range(8):
        x = random.uniform(50, WIDTH-50)
        y = random.uniform(50, HEIGHT-50)
        max_radius = random.uniform(100, 250)
        draw_spiral(ax, x, y, max_radius, random_color(), 0.4)
    
    # 中等螺旋
    for _ in range(12):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        max_radius = random.uniform(50, 120)
        draw_spiral(ax, x, y, max_radius, random_color(), 0.5)
    
    # 小螺旋
    for _ in range(15):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        max_radius = random.uniform(20, 60)
        draw_spiral(ax, x, y, max_radius, random_color(), 0.6)
    
    # 创建网格来确保更均匀的分布
    grid_x = 8
    grid_y = 6
    cell_width = WIDTH / grid_x
    cell_height = HEIGHT / grid_y
    
    # 绘制大量几何图形 - 多层次策略
    
    # 第一层：大型背景图形
    # 大圆形
    for _ in range(30):
        x = random.uniform(-100, WIDTH + 100)
        y = random.uniform(-100, HEIGHT + 100)
        radius = random.uniform(80, 200)
        alpha = random.uniform(0.1, 0.3)
        draw_circle(ax, x, y, radius, random_color(), alpha)
    
    # 大六边形
    for _ in range(25):
        x = random.uniform(-50, WIDTH + 50)
        y = random.uniform(-50, HEIGHT + 50)
        size = random.uniform(80, 180)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.1, 0.3)
        draw_hexagon(ax, x, y, size, rotation, random_color(), alpha)
    
    # 第二层：中等大小图形，使用网格分布确保覆盖
    for i in range(grid_x):
        for j in range(grid_y):
            # 在每个网格单元中放置多个图形
            base_x = i * cell_width
            base_y = j * cell_height
            
            # 每个网格放置3-5个中等图形
            for _ in range(random.randint(3, 5)):
                x = base_x + random.uniform(0, cell_width)
                y = base_y + random.uniform(0, cell_height)
                
                shape_type = random.choice(['circle', 'triangle', 'square', 'pentagon', 'hexagon', 'trapezoid'])
                size = random.uniform(30, 80)
                alpha = random.uniform(0.3, 0.6)
                rotation = random.uniform(0, 360)
                
                if shape_type == 'circle':
                    draw_circle(ax, x, y, size, random_color(), alpha)
                elif shape_type == 'triangle':
                    draw_triangle(ax, x, y, size, rotation, random_color(), alpha)
                elif shape_type == 'square':
                    draw_square(ax, x, y, size, rotation, random_color(), alpha)
                elif shape_type == 'pentagon':
                    draw_pentagon(ax, x, y, size, rotation, random_color(), alpha)
                elif shape_type == 'hexagon':
                    draw_hexagon(ax, x, y, size, rotation, random_color(), alpha)
                else:  # trapezoid
                    draw_trapezoid(ax, x, y, size, rotation, random_color(), alpha)
    
    # 第三层：密集的小型图形
    # 三角形群
    for _ in range(80):
        x = random.uniform(-30, WIDTH + 30)
        y = random.uniform(-30, HEIGHT + 30)
        size = random.uniform(15, 40)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.4, 0.7)
        draw_triangle(ax, x, y, size, rotation, random_color(), alpha)
    
    # 正方形群
    for _ in range(60):
        x = random.uniform(-30, WIDTH + 30)
        y = random.uniform(-30, HEIGHT + 30)
        size = random.uniform(20, 50)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.4, 0.7)
        draw_square(ax, x, y, size, rotation, random_color(), alpha)
    
    # 五边形群
    for _ in range(50):
        x = random.uniform(-30, WIDTH + 30)
        y = random.uniform(-30, HEIGHT + 30)
        size = random.uniform(20, 45)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.4, 0.7)
        draw_pentagon(ax, x, y, size, rotation, random_color(), alpha)
    
    # 小圆形群
    for _ in range(70):
        x = random.uniform(-30, WIDTH + 30)
        y = random.uniform(-30, HEIGHT + 30)
        radius = random.uniform(15, 45)
        alpha = random.uniform(0.4, 0.7)
        draw_circle(ax, x, y, radius, random_color(), alpha)
    
    # 梯形群
    for _ in range(40):
        x = random.uniform(-30, WIDTH + 30)
        y = random.uniform(-30, HEIGHT + 30)
        size = random.uniform(25, 60)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.4, 0.7)
        draw_trapezoid(ax, x, y, size, rotation, random_color(), alpha)
    
    # 第四层：填充空隙的微小图形
    # 微小三角形
    for _ in range(150):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        size = random.uniform(8, 20)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.5, 0.8)
        draw_triangle(ax, x, y, size, rotation, random_color(), alpha)
    
    # 微小圆形
    for _ in range(200):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        radius = random.uniform(5, 15)
        alpha = random.uniform(0.5, 0.8)
        draw_circle(ax, x, y, radius, random_color(), alpha)
    
    # 添加装饰性小圆点（大幅增加）
    for _ in range(500):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        radius = random.uniform(1, 5)
        alpha = random.uniform(0.6, 0.9)
        circle = plt.Circle((x, y), radius, 
                          facecolor=random_color(), alpha=alpha, edgecolor='none')
        ax.add_patch(circle)
    
    # 超小装饰点
    for _ in range(300):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        radius = random.uniform(0.5, 2)
        alpha = random.uniform(0.7, 1.0)
        circle = plt.Circle((x, y), radius, 
                          facecolor=random_color(), alpha=alpha, edgecolor='none')
        ax.add_patch(circle)
    
    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 移除边距并保存临时图像
    plt.tight_layout(pad=0)
    
    # 先保存图像到临时文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_filename = output_dir / f"temp_geometric_pattern_{timestamp}.png"
    plt.savefig(temp_filename, bbox_inches='tight', pad_inches=0, 
                facecolor='#0a0a0a', dpi=DPI)
    
    # 读取刚保存的图像
    image_array = mpimg.imread(temp_filename)
    
    # 如果是RGBA，转换为RGB
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    
    # 确保数组是float类型
    if image_array.dtype == np.uint8:
        image_array = image_array.astype(np.float32) / 255.0
    
    # 添加高斯噪声
    noisy_image = add_gaussian_noise(image_array, noise_strength)
    
    # 关闭原图
    plt.close(fig)
    
    # 创建新图形显示带噪声的图像
    fig2, ax2 = plt.subplots(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI)
    ax2.imshow(noisy_image)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    
    # 保存最终图像
    final_filename = output_dir / f"geometric_pattern_{timestamp}.png"
    plt.savefig(final_filename, bbox_inches='tight', pad_inches=0, 
                facecolor='#0a0a0a', dpi=DPI)
    # 静默模式，不打印每个文件
    # print(f"Pattern saved to: {final_filename}")
    
    # 删除临时文件
    if temp_filename.exists():
        temp_filename.unlink()
    
    # 关闭图形释放内存
    plt.close('all')
    
    # 返回文件名供后续使用
    return str(final_filename)

# 简化版本（如果上面的版本有问题）
def generate_pattern_simple(noise_strength=20, output_dir=None):
    """简化版本，直接保存文件，不读取重新处理"""
    print("Using simplified version...")
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal')
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    
    # 添加渐变背景
    for i in range(20):
        radius = WIDTH * (1 - i/20)
        alpha = 0.05
        color = (0.08 + i*0.02/20, 0.08 + i*0.02/20, 0.2 + i*0.03/20)
        circle = plt.Circle((WIDTH/2, HEIGHT/2), radius, 
                          facecolor=color, alpha=alpha, edgecolor='none')
        ax.add_patch(circle)
    
    # 绘制大量几何图形（简化版，但数量依然很多）
    # 圆形
    for _ in range(100):
        x = random.uniform(-50, WIDTH + 50)
        y = random.uniform(-50, HEIGHT + 50)
        radius = random.uniform(20, 150)
        alpha = random.uniform(0.2, 0.7)
        circle = plt.Circle((x, y), radius, 
                          facecolor=random_color(), alpha=alpha, edgecolor='none')
        ax.add_patch(circle)
    
    # 矩形和多边形
    for _ in range(200):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        size = random.uniform(20, 100)
        sides = random.randint(3, 8)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.3, 0.7)
        
        angles = np.linspace(0, 2*np.pi, sides, endpoint=False) + np.radians(rotation)
        points = [(x + size * np.cos(a), y + size * np.sin(a)) for a in angles]
        polygon = plt.Polygon(points, facecolor=random_color(), alpha=alpha, edgecolor='none')
        ax.add_patch(polygon)
    
    # 模拟噪声效果（通过添加大量微小随机点）
    for _ in range(int(noise_strength * 100)):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        radius = random.uniform(0.5, 2)
        alpha = random.uniform(0.1, 0.3)
        color = (random.random(), random.random(), random.random())
        circle = plt.Circle((x, y), radius, facecolor=color, alpha=alpha, edgecolor='none')
        ax.add_patch(circle)
    
    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"geometric_pattern_simple_{timestamp}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, facecolor='#0a0a0a', dpi=DPI)
    plt.close()
    # print(f"Pattern saved to: {filename}")
    return str(filename)

# 白色背景版本的图案生成
def generate_pattern_white_bg(noise_strength=20, output_dir=None):
    """
    生成白色背景的密集几何图案
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR_WHITE
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal')
    
    # 设置白色背景
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    
    # 创建浅色径向渐变背景（通过多个半透明圆形模拟）
    for i in range(20):
        radius = WIDTH * (1 - i/20)
        alpha = 0.02  # 更低的透明度
        color = (0.95 - i*0.01/20, 0.95 - i*0.01/20, 0.95 - i*0.01/20)  # 浅灰色渐变
        circle = plt.Circle((WIDTH/2, HEIGHT/2), radius, 
                          facecolor=color, alpha=alpha, edgecolor='none')
        ax.add_patch(circle)
    
    # 生成贝塞尔网络的随机点（增加密度）
    network_points = [(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)) 
                      for _ in range(35)]
    draw_bezier_network(ax, network_points, random_color_for_white_bg(), 0.6)
    
    # 额外的贝塞尔网络层
    network_points2 = [(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)) 
                       for _ in range(25)]
    draw_bezier_network(ax, network_points2, random_color_for_white_bg(), 0.5)
    
    # 绘制螺旋线（大幅增加数量和变化）
    # 大螺旋
    for _ in range(8):
        x = random.uniform(50, WIDTH-50)
        y = random.uniform(50, HEIGHT-50)
        max_radius = random.uniform(100, 250)
        draw_spiral(ax, x, y, max_radius, random_color_for_white_bg(), 0.6)
    
    # 中等螺旋
    for _ in range(12):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        max_radius = random.uniform(50, 120)
        draw_spiral(ax, x, y, max_radius, random_color_for_white_bg(), 0.7)
    
    # 小螺旋
    for _ in range(15):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        max_radius = random.uniform(20, 60)
        draw_spiral(ax, x, y, max_radius, random_color_for_white_bg(), 0.8)
    
    # 创建网格来确保更均匀的分布
    grid_x = 8
    grid_y = 6
    cell_width = WIDTH / grid_x
    cell_height = HEIGHT / grid_y
    
    # 绘制大量几何图形 - 多层次策略
    
    # 第一层：大型背景图形
    # 大圆形
    for _ in range(30):
        x = random.uniform(-100, WIDTH + 100)
        y = random.uniform(-100, HEIGHT + 100)
        radius = random.uniform(80, 200)
        alpha = random.uniform(0.1, 0.3)
        draw_circle(ax, x, y, radius, random_color_for_white_bg(), alpha)
    
    # 大六边形
    for _ in range(25):
        x = random.uniform(-50, WIDTH + 50)
        y = random.uniform(-50, HEIGHT + 50)
        size = random.uniform(80, 180)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.1, 0.3)
        draw_hexagon(ax, x, y, size, rotation, random_color_for_white_bg(), alpha)
    
    # 第二层：中等大小图形，使用网格分布确保覆盖
    for i in range(grid_x):
        for j in range(grid_y):
            # 在每个网格单元中放置多个图形
            base_x = i * cell_width
            base_y = j * cell_height
            
            # 每个网格放置3-5个中等图形
            for _ in range(random.randint(3, 5)):
                x = base_x + random.uniform(0, cell_width)
                y = base_y + random.uniform(0, cell_height)
                
                shape_type = random.choice(['circle', 'triangle', 'square', 'pentagon', 'hexagon', 'trapezoid'])
                size = random.uniform(30, 80)
                alpha = random.uniform(0.3, 0.6)
                rotation = random.uniform(0, 360)
                
                if shape_type == 'circle':
                    draw_circle(ax, x, y, size, random_color_for_white_bg(), alpha)
                elif shape_type == 'triangle':
                    draw_triangle(ax, x, y, size, rotation, random_color_for_white_bg(), alpha)
                elif shape_type == 'square':
                    draw_square(ax, x, y, size, rotation, random_color_for_white_bg(), alpha)
                elif shape_type == 'pentagon':
                    draw_pentagon(ax, x, y, size, rotation, random_color_for_white_bg(), alpha)
                elif shape_type == 'hexagon':
                    draw_hexagon(ax, x, y, size, rotation, random_color_for_white_bg(), alpha)
                else:  # trapezoid
                    draw_trapezoid(ax, x, y, size, rotation, random_color_for_white_bg(), alpha)
    
    # 第三层：密集的小型图形
    # 三角形群
    for _ in range(80):
        x = random.uniform(-30, WIDTH + 30)
        y = random.uniform(-30, HEIGHT + 30)
        size = random.uniform(15, 40)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.4, 0.7)
        draw_triangle(ax, x, y, size, rotation, random_color_for_white_bg(), alpha)
    
    # 正方形群
    for _ in range(60):
        x = random.uniform(-30, WIDTH + 30)
        y = random.uniform(-30, HEIGHT + 30)
        size = random.uniform(20, 50)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.4, 0.7)
        draw_square(ax, x, y, size, rotation, random_color_for_white_bg(), alpha)
    
    # 五边形群
    for _ in range(50):
        x = random.uniform(-30, WIDTH + 30)
        y = random.uniform(-30, HEIGHT + 30)
        size = random.uniform(20, 45)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.4, 0.7)
        draw_pentagon(ax, x, y, size, rotation, random_color_for_white_bg(), alpha)
    
    # 小圆形群
    for _ in range(70):
        x = random.uniform(-30, WIDTH + 30)
        y = random.uniform(-30, HEIGHT + 30)
        radius = random.uniform(15, 45)
        alpha = random.uniform(0.4, 0.7)
        draw_circle(ax, x, y, radius, random_color_for_white_bg(), alpha)
    
    # 梯形群
    for _ in range(40):
        x = random.uniform(-30, WIDTH + 30)
        y = random.uniform(-30, HEIGHT + 30)
        size = random.uniform(25, 60)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.4, 0.7)
        draw_trapezoid(ax, x, y, size, rotation, random_color_for_white_bg(), alpha)
    
    # 第四层：填充空隙的微小图形
    # 微小三角形
    for _ in range(150):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        size = random.uniform(8, 20)
        rotation = random.uniform(0, 360)
        alpha = random.uniform(0.5, 0.8)
        draw_triangle(ax, x, y, size, rotation, random_color_for_white_bg(), alpha)
    
    # 微小圆形
    for _ in range(200):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        radius = random.uniform(5, 15)
        alpha = random.uniform(0.5, 0.8)
        draw_circle(ax, x, y, radius, random_color_for_white_bg(), alpha)
    
    # 添加装饰性小圆点（大幅增加）
    for _ in range(500):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        radius = random.uniform(1, 5)
        alpha = random.uniform(0.6, 0.9)
        circle = plt.Circle((x, y), radius, 
                          facecolor=random_color_for_white_bg(), alpha=alpha, edgecolor='none')
        ax.add_patch(circle)
    
    # 超小装饰点
    for _ in range(300):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        radius = random.uniform(0.5, 2)
        alpha = random.uniform(0.7, 1.0)
        circle = plt.Circle((x, y), radius, 
                          facecolor=random_color_for_white_bg(), alpha=alpha, edgecolor='none')
        ax.add_patch(circle)
    
    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 移除边距并保存临时图像
    plt.tight_layout(pad=0)
    
    # 先保存图像到临时文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_filename = output_dir / f"temp_geometric_pattern_white_{timestamp}.png"
    plt.savefig(temp_filename, bbox_inches='tight', pad_inches=0, 
                facecolor='#ffffff', dpi=DPI)
    
    # 读取刚保存的图像
    image_array = mpimg.imread(temp_filename)
    
    # 如果是RGBA，转换为RGB
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    
    # 确保数组是float类型
    if image_array.dtype == np.uint8:
        image_array = image_array.astype(np.float32) / 255.0
    
    # 添加高斯噪声
    noisy_image = add_gaussian_noise(image_array, noise_strength)
    
    # 关闭原图
    plt.close(fig)
    
    # 创建新图形显示带噪声的图像
    fig2, ax2 = plt.subplots(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI)
    ax2.imshow(noisy_image)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    
    # 保存最终图像
    final_filename = output_dir / f"geometric_pattern_white_{timestamp}.png"
    plt.savefig(final_filename, bbox_inches='tight', pad_inches=0, 
                facecolor='#ffffff', dpi=DPI)
    # print(f"Pattern saved to: {final_filename}")
    
    # 删除临时文件
    if temp_filename.exists():
        temp_filename.unlink()
    
    # 关闭图形释放内存
    plt.close('all')
    
    # 返回文件名供后续使用
    return str(final_filename)

def generate_single_image(args):
    """生成单张图片的工作函数（用于多进程）
    
    Args:
        args: (index, total_count, output_dir, white_bg) 元组
    
    Returns:
        (success, filename, error_msg) 元组
    """
    # 根据参数数量判断是否包含 white_bg 参数
    if len(args) == 4:
        index, total_count, output_dir, white_bg = args
    else:
        index, total_count, output_dir = args
        white_bg = False
    
    try:
        # 随机噪声强度
        noise_strength = random.uniform(15, 30)
        
        # 设置随机种子，确保每个进程生成不同的图片
        seed = (int(time.time() * 1000) + index) % (2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        
        # 根据背景颜色选择生成函数
        if white_bg:
            filename = generate_pattern_white_bg(noise_strength=noise_strength, output_dir=output_dir)
        else:
            filename = generate_pattern(noise_strength=noise_strength, output_dir=output_dir)
        
        return (True, filename, None)
    except Exception as e:
        return (False, None, str(e))

def generate_batch(count=200, output_dir=None, show_progress=True, num_workers=None, white_bg=False):
    """批量生成图片（使用多进程）
    
    Args:
        count: 要生成的图片数量
        output_dir: 输出目录，默认为 OUTPUT_DIR
        show_progress: 是否显示进度
        num_workers: 工作进程数，默认为 CPU 核心数
        white_bg: 是否生成白色背景
    
    Returns:
        生成的文件路径列表
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR_WHITE if white_bg else OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # 最多使用8个进程
    
    print(f"Generating {count} images to: {output_dir}")
    print(f"Background: {'White' if white_bg else 'Dark'}")
    print(f"Using {num_workers} worker processes")
    print("-" * 50)
    
    filenames = []
    errors = []
    start_time = time.time()
    
    # 准备任务参数
    tasks = [(i, count, output_dir, white_bg) for i in range(count)]
    
    # 使用进程池
    with Pool(num_workers) as pool:
        # 使用 imap_unordered 获取结果，可以实时显示进度
        results = pool.imap_unordered(generate_single_image, tasks)
        
        completed = 0
        for success, filename, error_msg in results:
            completed += 1
            
            if success:
                filenames.append(filename)
            else:
                errors.append(f"Image {completed}: {error_msg}")
            
            if show_progress:
                # 显示进度
                progress = completed / count * 100
                elapsed = time.time() - start_time
                eta = elapsed / completed * count - elapsed if completed > 0 else 0
                
                status = "OK" if success else "FAIL"
                file_info = Path(filename).name if success else error_msg[:30]
                
                print(f"Progress: {completed}/{count} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s - "
                      f"Status: {status} - {file_info}")
    
    # 完成统计
    total_time = time.time() - start_time
    print("-" * 50)
    print(f"Generation complete!")
    print(f"Success: {len(filenames)}/{count}")
    print(f"Errors: {len(errors)}")
    if errors and len(errors) <= 10:
        for error in errors:
            print(f"  - {error}")
    elif errors:
        print(f"  (showing first 10 errors)")
        for error in errors[:10]:
            print(f"  - {error}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per image: {total_time/count:.2f}s")
    print(f"Output directory: {output_dir}")
    
    return filenames

def generate_batch_sequential(count=200, output_dir=None, show_progress=True, white_bg=False):
    """批量生成图片（顺序执行版本，用于调试）
    
    Args:
        count: 要生成的图片数量
        output_dir: 输出目录，默认为 OUTPUT_DIR
        show_progress: 是否显示进度
        white_bg: 是否生成白色背景
    
    Returns:
        生成的文件路径列表
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR_WHITE if white_bg else OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {count} images to: {output_dir}")
    print(f"Background: {'White' if white_bg else 'Dark'}")
    print("Using sequential mode (single process)")
    print("-" * 50)
    
    filenames = []
    start_time = datetime.now()
    
    for i in range(count):
        # 随机噪声强度
        noise_strength = random.uniform(15, 30)
        
        # 生成图片
        try:
            if white_bg:
                filename = generate_pattern_white_bg(noise_strength=noise_strength, output_dir=output_dir)
            else:
                filename = generate_pattern(noise_strength=noise_strength, output_dir=output_dir)
            filenames.append(filename)
            
            if show_progress and ((i + 1) % 10 == 0 or (i + 1) == count):
                # 每10张显示一次进度
                progress = (i + 1) / count * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                eta = elapsed / (i + 1) * count - elapsed if i > 0 else 0
                avg_time = elapsed / (i + 1)
                
                print(f"Progress: {i+1}/{count} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s - "
                      f"Avg: {avg_time:.1f}s/img")
        except Exception as e:
            print(f"Error generating image {i+1}: {e}")
            continue
    
    # 完成统计
    total_time = (datetime.now() - start_time).total_seconds()
    print("-" * 50)
    print(f"Generation complete!")
    print(f"Total images: {len(filenames)}/{count}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per image: {total_time/len(filenames):.2f}s")
    print(f"Output directory: {output_dir}")
    
    return filenames

# 运行生成器
if __name__ == "__main__":
    # Windows 需要这个来支持多进程
    import platform
    if platform.system() == 'Windows':
        from multiprocessing import freeze_support
        freeze_support()
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Geometric Pattern Generator')
    parser.add_argument('--count', type=int, default=200,
                        help='Number of images to generate (default: 200)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: data/raw/Geometric_Generated)')
    parser.add_argument('--single', action='store_true',
                        help='Generate single image with preview')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto)')
    parser.add_argument('--sequential', action='store_true',
                        help='Use sequential mode instead of multiprocessing')
    parser.add_argument('--white-bg', action='store_true',
                        help='Generate with white background')
    
    args = parser.parse_args()
    
    if args.single:
        # 单张生成模式（带预览）
        print(f"Output directory: {OUTPUT_DIR}")
        filename = generate_pattern(noise_strength=20)
        
        # 尝试在不同环境中显示图像
        try:
            # 如果在Jupyter/IPython环境中
            from IPython.display import Image, display
            display(Image(filename))
            print("Image displayed in Jupyter")
        except:
            try:
                # 尝试使用系统默认图像查看器打开
                if platform.system() == 'Windows':
                    os.startfile(filename)
                elif platform.system() == 'Darwin':  # macOS
                    os.system(f'open {filename}')
                else:  # Linux
                    os.system(f'xdg-open {filename}')
                print("Image opened with system default viewer")
            except:
                print(f"Please manually open file: {filename}")
    else:
        # 批量生成模式
        if args.sequential:
            generate_batch_sequential(count=args.count, output_dir=args.output, white_bg=args.white_bg)
        else:
            # Windows 多进程可能有问题，提供选项
            if platform.system() == 'Windows' and args.workers is None:
                print("Note: On Windows, multiprocessing might have issues.")
                print("Use --sequential flag if you encounter problems.")
                print()
            generate_batch(count=args.count, output_dir=args.output, num_workers=args.workers, white_bg=args.white_bg)
