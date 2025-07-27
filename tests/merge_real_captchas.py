"""
合并真实验证码的bg和slider图片
"""
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import re


def merge_captcha_images(bg_path, slider_path):
    """合并背景和滑块图片 - 按照训练数据的格式"""
    # 读取图片
    bg_img = cv2.imread(str(bg_path))
    slider_img = cv2.imread(str(slider_path), cv2.IMREAD_UNCHANGED)
    
    if bg_img is None or slider_img is None:
        print(f"Failed to read images: {bg_path} or {slider_path}")
        return None
        
    # 确保背景图是320x160
    if bg_img.shape[:2] != (160, 320):
        bg_img = cv2.resize(bg_img, (320, 160))
    
    # 获取滑块尺寸
    slider_h, slider_w = slider_img.shape[:2]
    
    # 创建合并后的图片 - 复制背景图
    merged_img = bg_img.copy()
    
    # 滑块位置：在背景左侧，x坐标在[0, 10]范围内随机
    slider_x = np.random.randint(0, 11)  # 0到10像素
    # 滑块垂直居中
    slider_y = (160 - slider_h) // 2
    
    # 如果滑块有alpha通道，使用它进行混合
    if slider_img.shape[2] == 4:
        # 使用alpha通道混合滑块到背景
        for y in range(slider_h):
            for x in range(slider_w):
                if (slider_y + y < 160 and slider_y + y >= 0 and 
                    slider_x + x < 320 and slider_x + x >= 0):
                    alpha = slider_img[y, x, 3] / 255.0
                    if alpha > 0:
                        merged_img[slider_y + y, slider_x + x] = (
                            slider_img[y, x, :3] * alpha + 
                            merged_img[slider_y + y, slider_x + x] * (1 - alpha)
                        ).astype(np.uint8)
    else:
        # 没有alpha通道，直接复制（但这种情况不太可能，因为滑块应该有透明背景）
        y_end = min(slider_y + slider_h, 160)
        x_end = min(slider_x + slider_w, 320)
        y_start = max(slider_y, 0)
        x_start = max(slider_x, 0)
        
        if y_end > y_start and x_end > x_start:
            merged_img[y_start:y_end, x_start:x_end] = slider_img[
                y_start-slider_y:y_end-slider_y, 
                x_start-slider_x:x_end-slider_x
            ]
    
    return merged_img


def process_site_folder(site_folder, output_folder):
    """处理单个站点文件夹"""
    site_path = Path(site_folder)
    site_name = site_path.name
    
    # 创建输出子文件夹
    site_output = output_folder / site_name
    site_output.mkdir(exist_ok=True)
    
    # 收集所有bg文件
    bg_files = sorted(site_path.glob("*_bg.png"))
    
    print(f"\nProcessing {site_name}: found {len(bg_files)} image pairs")
    
    annotations = []
    success_count = 0
    
    for bg_file in tqdm(bg_files, desc=f"Merging {site_name}"):
        # 构造对应的slider文件名
        slider_file = site_path / bg_file.name.replace("_bg.png", "_slider.png")
        
        if not slider_file.exists():
            print(f"Missing slider file: {slider_file}")
            continue
        
        # 合并图片
        merged_img = merge_captcha_images(bg_file, slider_file)
        
        if merged_img is not None:
            # 保存合并后的图片
            output_filename = bg_file.name.replace("_bg.png", "_merged.png")
            output_path = site_output / output_filename
            cv2.imwrite(str(output_path), merged_img)
            
            # 提取信息用于标注（如果文件名包含坐标信息）
            # 这里假设真实数据没有坐标信息，所以留空
            annotations.append({
                'filename': output_filename,
                'original_bg': bg_file.name,
                'original_slider': slider_file.name,
                'site': site_name
            })
            
            success_count += 1
    
    print(f"{site_name}: Successfully merged {success_count}/{len(bg_files)} image pairs")
    
    # 保存标注文件
    annotations_file = site_output / f"{site_name}_annotations.json"
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    return success_count, len(bg_files)


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    real_captchas_dir = project_root / "data" / "real_captchas"
    merged_output_dir = real_captchas_dir / "merged"
    
    # 确保输出目录存在
    merged_output_dir.mkdir(exist_ok=True)
    
    # 处理site1
    total_success = 0
    total_files = 0
    
    for site_name in ['site1']:
        site_folder = real_captchas_dir / site_name
        if site_folder.exists():
            success, total = process_site_folder(site_folder, merged_output_dir)
            total_success += success
            total_files += total
        else:
            print(f"Warning: {site_folder} not found")
    
    print(f"\n{'='*50}")
    print(f"Total: Successfully merged {total_success}/{total_files} image pairs")
    print(f"Output saved to: {merged_output_dir}")


if __name__ == "__main__":
    main()