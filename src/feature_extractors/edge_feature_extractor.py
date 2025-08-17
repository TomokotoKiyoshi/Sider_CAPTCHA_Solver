# -*- coding: utf-8 -*-
"""
边缘特征提取器 - 用于提取验证码的边缘和形状特征
解决Domain Gap问题的关键组件
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


class EdgeFeatureExtractor:
    """边缘特征提取器"""
    
    def __init__(self, puzzle_size: int = 40):
        """
        初始化边缘特征提取器
        
        Args:
            puzzle_size: 拼图块的大小（像素）
        """
        self.puzzle_size = puzzle_size
        self.templates = None
        self._generate_puzzle_templates()
    
    def _generate_puzzle_templates(self):
        """生成拼图形状模板"""
        self.templates = []
        size = self.puzzle_size
        
        # 生成几种常见的拼图形状
        templates_specs = [
            # 标准拼图块（凸起在右侧）
            {'name': 'standard_right', 'type': 'convex_right'},
            # 标准拼图块（凸起在左侧）
            {'name': 'standard_left', 'type': 'convex_left'},
            # 圆形缺口
            {'name': 'circle', 'type': 'circle'},
            # 方形缺口
            {'name': 'square', 'type': 'square'},
        ]
        
        for spec in templates_specs:
            template = self._create_puzzle_template(size, spec['type'])
            self.templates.append({
                'name': spec['name'],
                'template': template,
                'edges': cv2.Canny(template, 50, 150)
            })
    
    def _create_puzzle_template(self, size: int, shape_type: str) -> np.ndarray:
        """
        创建单个拼图模板
        
        Args:
            size: 模板大小
            shape_type: 形状类型
            
        Returns:
            模板图像（二值图）
        """
        template = np.zeros((size, size), dtype=np.uint8)
        
        if shape_type == 'convex_right':
            # 创建右侧有凸起的拼图形状
            template[:, :size//2] = 255
            cv2.circle(template, (size//2, size//2), size//4, 255, -1)
            
        elif shape_type == 'convex_left':
            # 创建左侧有凸起的拼图形状
            template[:, size//2:] = 255
            cv2.circle(template, (size//2, size//2), size//4, 255, -1)
            
        elif shape_type == 'circle':
            # 圆形
            cv2.circle(template, (size//2, size//2), size//3, 255, -1)
            
        elif shape_type == 'square':
            # 方形
            margin = size // 5
            template[margin:-margin, margin:-margin] = 255
        
        return template
    
    def load_puzzle_templates(self) -> List[Dict]:
        """
        加载拼图模板
        
        Returns:
            模板列表
        """
        if self.templates is None:
            self._generate_puzzle_templates()
        return self.templates
    
    def match_templates(self, edge_image: np.ndarray, templates: List[Dict]) -> np.ndarray:
        """
        在边缘图像中匹配拼图模板
        
        Args:
            edge_image: 边缘检测后的图像
            templates: 模板列表
            
        Returns:
            模板匹配响应图
        """
        h, w = edge_image.shape
        response_map = np.zeros((h, w), dtype=np.float32)
        
        for template_info in templates:
            template = template_info['edges']
            
            # 使用模板匹配
            result = cv2.matchTemplate(edge_image, template, cv2.TM_CCOEFF_NORMED)
            
            # 将结果填充到原始大小
            th, tw = template.shape
            padded_result = np.zeros((h, w), dtype=np.float32)
            padded_result[:result.shape[0], :result.shape[1]] = np.abs(result)
            
            # 取最大响应
            response_map = np.maximum(response_map, padded_result)
        
        return response_map
    
    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取图像的边缘特征
        
        Args:
            image: 输入图像（BGR或RGB格式）
            
        Returns:
            特征字典，包含：
            - edges: Canny边缘
            - gradient: Sobel梯度幅值
            - morph: 形态学梯度
            - template: 模板匹配响应
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Canny边缘检测
        edges_canny = cv2.Canny(gray, 50, 150)
        
        # 2. Sobel梯度
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
        
        # 3. 形态学梯度
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # 4. 拼图形状模板匹配
        puzzle_templates = self.load_puzzle_templates()
        template_response = self.match_templates(edges_canny, puzzle_templates)
        template_response = (template_response * 255).astype(np.uint8)
        
        return {
            'edges': edges_canny,
            'gradient': gradient_magnitude,
            'morph': morph_gradient,
            'template': template_response
        }
    
    def visualize_features(self, image: np.ndarray, features: Dict[str, np.ndarray], 
                          save_path: Optional[str] = None, show: bool = False):
        """
        可视化提取的特征
        
        Args:
            image: 原始图像
            features: 特征字典
            save_path: 保存路径（可选）
            show: 是否显示图像
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Canny边缘
        axes[0, 1].imshow(features['edges'], cmap='gray')
        axes[0, 1].set_title('Canny Edges')
        axes[0, 1].axis('off')
        
        # Sobel梯度
        axes[0, 2].imshow(features['gradient'], cmap='gray')
        axes[0, 2].set_title('Sobel Gradient Magnitude')
        axes[0, 2].axis('off')
        
        # 形态学梯度
        axes[1, 0].imshow(features['morph'], cmap='gray')
        axes[1, 0].set_title('Morphological Gradient')
        axes[1, 0].axis('off')
        
        # 模板匹配响应
        axes[1, 1].imshow(features['template'], cmap='hot')
        axes[1, 1].set_title('Template Matching Response')
        axes[1, 1].axis('off')
        
        # 组合特征（加权平均）
        combined = (
            0.3 * features['edges'] + 
            0.3 * features['gradient'] + 
            0.2 * features['morph'] + 
            0.2 * features['template']
        ).astype(np.uint8)
        axes[1, 2].imshow(combined, cmap='gray')
        axes[1, 2].set_title('Combined Features')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


def batch_process():
    """批量处理验证码图像的边缘特征"""
    print("边缘特征批量提取")
    print("=" * 50)
    
    # 创建特征提取器
    extractor = EdgeFeatureExtractor(puzzle_size=40)
    
    # 输出根目录
    output_root = Path("D:/Hacker/Sider_CAPTCHA_Solver/extracted_features")
    
    # 为合成图和真实图片创建独立的输出目录
    synthetic_output = output_root / "synthetic_captchas"
    real_output = output_root / "real_captchas"
    comparison_output = output_root / "comparisons"
    
    # 创建所有必要的目录
    synthetic_output.mkdir(parents=True, exist_ok=True)
    real_output.mkdir(parents=True, exist_ok=True)
    comparison_output.mkdir(parents=True, exist_ok=True)
    
    # 收集图像文件
    all_images = []
    
    # 从data/captchas读取不同原始图片的验证码
    captchas_dir = Path("data/captchas")
    if captchas_dir.exists():
        # 获取所有验证码文件
        all_captcha_files = sorted(captchas_dir.glob("*.png"))
        
        # 提取不同的原始图片编号（PicXXXX）
        pic_groups = {}
        for f in all_captcha_files:
            # 从文件名提取Pic编号
            pic_id = f.stem.split('_')[0]  # 获取 PicXXXX 部分
            if pic_id not in pic_groups:
                pic_groups[pic_id] = []
            pic_groups[pic_id].append(f)
        
        # 从每个不同的Pic组中选择一个验证码，最多50个
        captcha_files = []
        for pic_id in sorted(pic_groups.keys())[:50]:
            # 从每个Pic组中选择第一个验证码
            captcha_files.append(pic_groups[pic_id][0])
        
        all_images.extend([(f, 'synthetic') for f in captcha_files])
        print(f"找到 {len(pic_groups)} 个不同的原始图片")
        print(f"选择了 {len(captcha_files)} 张合成验证码（每个原始图片选一张）")
    
    # 从data/real_captchas/annotated读取不同原始图片的真实验证码
    real_dir = Path("data/real_captchas/annotated")
    if real_dir.exists():
        # 获取所有真实验证码文件
        all_real_files = sorted(real_dir.glob("*.png"))
        
        # 提取不同的原始图片编号（PicXXXX）
        real_pic_groups = {}
        for f in all_real_files:
            # 从文件名提取Pic编号
            pic_id = f.stem.split('_')[0]  # 获取 PicXXXX 部分
            if pic_id not in real_pic_groups:
                real_pic_groups[pic_id] = []
            real_pic_groups[pic_id].append(f)
        
        # 从每个不同的Pic组中选择一个验证码，最多50个
        real_files = []
        for pic_id in sorted(real_pic_groups.keys())[:50]:
            # 从每个Pic组中选择第一个验证码
            real_files.append(real_pic_groups[pic_id][0])
        
        all_images.extend([(f, 'real') for f in real_files])
        print(f"找到 {len(real_pic_groups)} 个不同的真实原始图片")
        print(f"选择了 {len(real_files)} 张真实验证码（每个原始图片选一张）")
    
    if not all_images:
        print("未找到任何图像文件！")
        return
    
    print(f"\n总共处理 {len(all_images)} 张图像")
    print(f"输出根目录: {output_root}")
    print(f"  - 合成图片特征: {synthetic_output}")
    print(f"  - 真实图片特征: {real_output}")
    print(f"  - 对比分析: {comparison_output}")
    
    # 分别处理合成图和真实图片
    synthetic_count = 0
    real_count = 0
    
    for idx, (image_path, image_type) in enumerate(tqdm(all_images, desc="Processing")):
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"无法读取: {image_path.name}")
                continue
            
            # 提取特征
            features = extractor.extract_features(image)
            
            # 根据类型选择输出目录
            if image_type == 'synthetic':
                save_name = f"synthetic_{synthetic_count:04d}_{image_path.stem}_features.png"
                save_path = synthetic_output / save_name
                synthetic_count += 1
            else:  # real
                save_name = f"real_{real_count:04d}_{image_path.stem}_features.png"
                save_path = real_output / save_name
                real_count += 1
            
            # 保存可视化
            extractor.visualize_features(image, features, str(save_path), show=False)
            
        except Exception as e:
            print(f"处理 {image_path.name} 时出错: {e}")
            continue
    
    print(f"\n✅ 批量处理完成！")
    print(f"📁 合成图片: 处理了 {synthetic_count} 张")
    print(f"📁 真实图片: 处理了 {real_count} 张")
    
    # 生成对比示例（前5个合成 vs 前5个真实）
    print("\n生成对比示例...")
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    
    synthetic_samples = [img for img, t in all_images if t == 'synthetic'][:5]
    real_samples = [img for img, t in all_images if t == 'real'][:5]
    
    for i in range(min(5, len(synthetic_samples))):
        # 合成数据
        syn_img = cv2.imread(str(synthetic_samples[i]))
        syn_features = extractor.extract_features(syn_img)
        
        axes[i, 0].imshow(cv2.cvtColor(syn_img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Synthetic #{i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(syn_features['edges'], cmap='gray')
        axes[i, 1].set_title('Edges')
        axes[i, 1].axis('off')
    
    for i in range(min(5, len(real_samples))):
        # 真实数据
        real_img = cv2.imread(str(real_samples[i]))
        real_features = extractor.extract_features(real_img)
        
        axes[i, 2].imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title(f'Real #{i+1}')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(real_features['edges'], cmap='gray')
        axes[i, 3].set_title('Edges')
        axes[i, 3].axis('off')
    
    plt.suptitle('Synthetic vs Real CAPTCHA Edge Features', fontsize=16)
    plt.tight_layout()
    comparison_path = comparison_output / "synthetic_vs_real_edge_comparison.png"
    plt.savefig(comparison_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"📊 对比图保存在: {comparison_path}")
    
    # 生成统计报告
    print("\n生成特征统计报告...")
    generate_feature_statistics(synthetic_samples[:10], real_samples[:10], extractor, comparison_output)


def generate_feature_statistics(synthetic_samples, real_samples, extractor, output_dir):
    """生成特征统计报告"""
    import json
    
    # 计算合成图和真实图的边缘密度统计
    synthetic_edge_densities = []
    real_edge_densities = []
    
    for img_path in synthetic_samples:
        img = cv2.imread(str(img_path))
        if img is not None:
            features = extractor.extract_features(img)
            edge_density = np.sum(features['edges'] > 0) / features['edges'].size
            synthetic_edge_densities.append(edge_density)
    
    for img_path in real_samples:
        img = cv2.imread(str(img_path))
        if img is not None:
            features = extractor.extract_features(img)
            edge_density = np.sum(features['edges'] > 0) / features['edges'].size
            real_edge_densities.append(edge_density)
    
    # 生成统计信息
    stats = {
        "synthetic_captchas": {
            "samples_analyzed": len(synthetic_edge_densities),
            "edge_density": {
                "mean": float(np.mean(synthetic_edge_densities)),
                "std": float(np.std(synthetic_edge_densities)),
                "min": float(np.min(synthetic_edge_densities)),
                "max": float(np.max(synthetic_edge_densities))
            }
        },
        "real_captchas": {
            "samples_analyzed": len(real_edge_densities),
            "edge_density": {
                "mean": float(np.mean(real_edge_densities)),
                "std": float(np.std(real_edge_densities)),
                "min": float(np.min(real_edge_densities)),
                "max": float(np.max(real_edge_densities))
            }
        },
        "comparison": {
            "edge_density_diff": float(np.mean(real_edge_densities) - np.mean(synthetic_edge_densities)),
            "analysis": "真实验证码的边缘密度通常更高，表明纹理更复杂" if np.mean(real_edge_densities) > np.mean(synthetic_edge_densities) else "合成验证码的边缘密度更高"
        }
    }
    
    # 保存统计报告
    stats_path = output_dir / "feature_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"📈 统计报告保存在: {stats_path}")


if __name__ == "__main__":
    batch_process()