# -*- coding: utf-8 -*-
"""
测试混淆缺口效果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import random
from src.captcha_generator.simple_puzzle_generator import create_puzzle_piece
from src.captcha_generator.lighting_effects import apply_gap_lighting
from src.captcha_generator.slider_effects import apply_slider_lighting, create_slider_frame, composite_slider

def rotate_image(image, angle, center=None):
    """旋转图像"""
    h, w = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    if image.shape[2] == 4:  # BGRA
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    else:  # BGR
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
    
    return rotated

def check_overlap(x1, y1, w1, h1, x2, y2, w2, h2, min_distance=10):
    """
    检查两个矩形是否重叠或距离太近
    """
    # 扩展矩形边界以包含最小距离
    x1_min = x1 - min_distance
    y1_min = y1 - min_distance
    x1_max = x1 + w1 + min_distance
    y1_max = y1 + h1 + min_distance
    
    x2_min = x2
    y2_min = y2
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    # 检查是否重叠
    return not (x2_max < x1_min or x2_min > x1_max or 
                y2_max < y1_min or y2_min > y1_max)

def generate_confusing_gap_captchas(num_images=6):
    """生成带有混淆缺口的验证码"""
    # 创建输出目录
    output_dir = Path(__file__).parent.parent / "test_output" / "confusing_gaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取背景图片
    img_path = Path(__file__).parent.parent / "data" / "raw" / "nature" / "nature_001.jpg"
    if not img_path.exists():
        # 创建测试图像
        img = np.ones((160, 320, 3), dtype=np.uint8) * 100
        for _ in range(50):
            x = random.randint(0, 320)
            y = random.randint(0, 160)
            cv2.circle(img, (x, y), random.randint(5, 20), 
                      (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)), -1)
    else:
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (320, 160))
    
    # 创建预览画布 (2行3列)
    rows = 2
    cols = 3
    img_width = 320
    img_height = 160
    padding = 20  # 图片之间的间距
    
    canvas_width = cols * img_width + (cols + 1) * padding
    canvas_height = rows * img_height + (rows + 1) * padding + 40  # 额外空间给标题
    preview_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 添加标题
    cv2.putText(preview_canvas, "CAPTCHAs with Confusing Gaps", 
               (canvas_width // 2 - 150, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    print(f"Generating {num_images} CAPTCHAs with confusing gaps...")
    
    success_count = 0
    attempt_count = 0
    
    while success_count < num_images and attempt_count < num_images * 3:
        attempt_count += 1
        
        # 创建拼图形状
        size = 60
        puzzle_mask = create_puzzle_piece(
            piece_size=size,
            knob_radius=int(size * 0.2),
            edges=('convex', 'concave', 'convex', 'concave')
        )
        
        mask_h, mask_w = puzzle_mask.shape[:2]
        alpha = puzzle_mask[:, :, 3]
        
        # 真实缺口位置 - 确保不超出边界
        x = random.randint(80, min(200, 320 - mask_w))
        y = random.randint(20, min(80, 160 - mask_h))
        
        # 滑块位置
        slider_x = 20
        slider_width = 60
        
        # 创建背景（应用真实缺口）
        background = img.copy()
        background = apply_gap_lighting(background, x, y, alpha, mask_h, mask_w)
        
        # 尝试生成混淆缺口
        confusion_placed = False
        for _ in range(100):  # 最多尝试100次
            # 随机选择旋转角度（±10到±30度）
            angle_magnitude = random.uniform(10, 30)
            confusion_angle = angle_magnitude if random.random() < 0.5 else -angle_magnitude
            
            # 旋转拼图掩码
            confusion_mask = rotate_image(puzzle_mask, confusion_angle)
            confusion_h, confusion_w = confusion_mask.shape[:2]
            confusion_alpha = confusion_mask[:, :, 3]
            
            # 随机生成混淆缺口位置
            # 使用智能位置生成策略
            if random.random() < 0.7:
                # 70%概率在右侧区域（远离左侧滑块）
                conf_x = random.randint(max(100, slider_x + slider_width + 20), 
                                      max(100, min(320 - confusion_w, 250)))
            else:
                # 30%概率在其他区域
                conf_x = random.randint(0, max(0, 320 - confusion_w))
            conf_y = random.randint(0, max(0, 160 - confusion_h))
            
            # 检查是否与真实缺口重叠或太近
            if not check_overlap(x, y, mask_w, mask_h, 
                               conf_x, conf_y, confusion_w, confusion_h, 10):
                # 检查是否与滑块重叠或太近
                if not check_overlap(slider_x, y, slider_width, mask_h,
                                   conf_x, conf_y, confusion_w, confusion_h, 10):
                    # 检查是否超出边界
                    if conf_x >= 0 and conf_y >= 0 and conf_x + confusion_w <= 320 and conf_y + confusion_h <= 160:
                        # 应用混淆缺口
                        background = apply_gap_lighting(
                            background, conf_x, conf_y, confusion_alpha, confusion_h, confusion_w,
                            base_darkness=30,      # 稍微浅一点的混淆缺口
                            edge_darkness=40,
                            directional_darkness=20,
                            outer_edge_darkness=5
                        )
                        confusion_placed = True
                        break
        
        if not confusion_placed:
            print(f"  Failed to place confusion gap for attempt {attempt_count}")
            continue
        
        # 创建滑块
        puzzle = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
        img_region = img[y:y+mask_h, x:x+mask_w]
        
        for c in range(3):
            puzzle[:, :, c] = img_region[:, :, c]
        puzzle[:, :, 3] = alpha
        
        # 应用滑块光照
        puzzle = apply_slider_lighting(puzzle)
        
        # 创建滑块边框
        slider_frame = create_slider_frame(60, 160)
        
        # 合成最终图像
        sd_center_x = slider_x + 30
        sd_center_y = y + puzzle.shape[0] // 2
        
        final_image = composite_slider(background, puzzle, (sd_center_x, sd_center_y), slider_frame)
        
        # 在图片上标记真实缺口和混淆缺口的位置
        marked_image = final_image.copy()
        # 真实缺口（绿色框）
        cv2.rectangle(marked_image, (x, y), (x + mask_w, y + mask_h), (0, 255, 0), 2)
        cv2.putText(marked_image, "Real Gap", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # 混淆缺口（红色框）
        cv2.rectangle(marked_image, (conf_x, conf_y), (conf_x + confusion_w, conf_y + confusion_h), (0, 0, 255), 2)
        cv2.putText(marked_image, f"Confusion: {confusion_angle:.0f} deg", (conf_x, conf_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 保存图片
        filename = f"confusing_gap_{success_count + 1:02d}.png"
        cv2.imwrite(str(output_dir / filename), marked_image)
        
        # 添加到预览画布
        if success_count < 6:
            row = success_count // 3
            col = success_count % 3
            # 计算位置，包含间距
            y_offset = 40 + padding + row * (img_height + padding)
            x_offset = padding + col * (img_width + padding)
            preview_canvas[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = marked_image
        
        success_count += 1
        print(f"  Generated image {success_count}/{num_images}")
    
    # 保存预览图
    cv2.imwrite(str(output_dir / "confusing_gaps_preview.png"), preview_canvas)
    
    print(f"\nSuccessfully generated {success_count}/{num_images} images")
    print(f"Output directory: {output_dir}")
    print(f"Preview saved as: confusing_gaps_preview.png")

if __name__ == "__main__":
    generate_confusing_gap_captchas(6)