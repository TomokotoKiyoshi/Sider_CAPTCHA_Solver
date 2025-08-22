"""
2D FFT频谱比较测试脚本
比较真实验证码和合成验证码的频域特征差异
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random
import logging
from dataclasses import dataclass
from scipy import stats
import seaborn as sns

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SpectrumStats:
    """频谱统计信息"""
    mean_magnitude: float
    std_magnitude: float
    peak_frequency: Tuple[int, int]
    energy_distribution: np.ndarray
    high_freq_ratio: float  # 高频成分比例
    low_freq_ratio: float   # 低频成分比例
    
class FFTAnalyzer:
    """2D FFT频谱分析器"""
    
    def __init__(self, sample_size: int = 100):
        self.sample_size = sample_size
        self.real_dir = Path("data/real_captchas/merged/site3")
        self.synthetic_dir = Path("data/captchas")
        
    def load_random_images(self, directory: Path, count: int) -> List[np.ndarray]:
        """随机加载指定数量的图像"""
        image_files = list(directory.glob("*.png")) + list(directory.glob("*.jpg"))
        
        if len(image_files) < count:
            logger.warning(f"只找到 {len(image_files)} 张图片，少于请求的 {count} 张")
            count = len(image_files)
            
        selected_files = random.sample(image_files, count)
        images = []
        
        for file_path in selected_files:
            img = cv2.imread(str(file_path))
            if img is not None:
                # 转换为灰度图进行频谱分析
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(gray)
            else:
                logger.warning(f"无法读取图像: {file_path}")
                
        logger.info(f"从 {directory} 成功加载 {len(images)} 张图片")
        return images
    
    def compute_2d_fft(self, image: np.ndarray) -> np.ndarray:
        """计算2D FFT并返回幅度谱"""
        # 统一调整图像大小以保证一致性
        target_size = (256, 128)  # (width, height)
        if image.shape != target_size[::-1]:  # shape是(height, width)
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 计算2D FFT
        f_transform = np.fft.fft2(image)
        # 将零频率分量移到中心
        f_shift = np.fft.fftshift(f_transform)
        # 计算幅度谱（取对数以便可视化）
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        return magnitude_spectrum
    
    def extract_spectrum_features(self, magnitude_spectrum: np.ndarray) -> SpectrumStats:
        """提取频谱特征"""
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # 创建距离矩阵
        y, x = np.ogrid[:h, :w]
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 定义频率带
        max_radius = min(center_x, center_y)
        low_freq_mask = distance_from_center <= max_radius * 0.1  # 低频：中心10%
        high_freq_mask = distance_from_center >= max_radius * 0.7  # 高频：外围30%
        
        # 计算统计量
        total_energy = np.sum(magnitude_spectrum)
        low_freq_energy = np.sum(magnitude_spectrum[low_freq_mask])
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        
        # 找到峰值频率位置
        peak_idx = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
        
        # 计算径向能量分布
        num_bins = 50
        radial_bins = np.linspace(0, max_radius, num_bins)
        radial_energy = np.zeros(num_bins - 1)
        
        for i in range(len(radial_bins) - 1):
            mask = (distance_from_center >= radial_bins[i]) & (distance_from_center < radial_bins[i+1])
            radial_energy[i] = np.mean(magnitude_spectrum[mask]) if np.any(mask) else 0
        
        return SpectrumStats(
            mean_magnitude=np.mean(magnitude_spectrum),
            std_magnitude=np.std(magnitude_spectrum),
            peak_frequency=peak_idx,
            energy_distribution=radial_energy,
            high_freq_ratio=high_freq_energy / total_energy,
            low_freq_ratio=low_freq_energy / total_energy
        )
    
    def analyze_dataset(self, images: List[np.ndarray], dataset_name: str) -> Dict:
        """分析数据集的频谱特征"""
        logger.info(f"开始分析 {dataset_name} 数据集...")
        
        all_stats = []
        all_spectrums = []
        
        for img in images:
            spectrum = self.compute_2d_fft(img)
            stats = self.extract_spectrum_features(spectrum)
            all_stats.append(stats)
            all_spectrums.append(spectrum)
        
        # 汇总统计
        summary = {
            'name': dataset_name,
            'mean_magnitude': np.mean([s.mean_magnitude for s in all_stats]),
            'std_magnitude': np.mean([s.std_magnitude for s in all_stats]),
            'high_freq_ratio': np.mean([s.high_freq_ratio for s in all_stats]),
            'low_freq_ratio': np.mean([s.low_freq_ratio for s in all_stats]),
            'high_freq_std': np.std([s.high_freq_ratio for s in all_stats]),
            'low_freq_std': np.std([s.low_freq_ratio for s in all_stats]),
            'energy_distributions': np.array([s.energy_distribution for s in all_stats]),
            'sample_spectrums': all_spectrums[:5]  # 保存5个样本用于可视化
        }
        
        return summary
    
    def statistical_comparison(self, real_stats: Dict, synthetic_stats: Dict):
        """统计比较两个数据集的频谱特征"""
        logger.info("\n=== 统计比较结果 ===")
        
        # 高频成分比较
        print(f"\n高频成分比例:")
        print(f"  真实验证码: {real_stats['high_freq_ratio']:.4f} ± {real_stats['high_freq_std']:.4f}")
        print(f"  合成验证码: {synthetic_stats['high_freq_ratio']:.4f} ± {synthetic_stats['high_freq_std']:.4f}")
        
        # 低频成分比较
        print(f"\n低频成分比例:")
        print(f"  真实验证码: {real_stats['low_freq_ratio']:.4f} ± {real_stats['low_freq_std']:.4f}")
        print(f"  合成验证码: {synthetic_stats['low_freq_ratio']:.4f} ± {synthetic_stats['low_freq_std']:.4f}")
        
        # 平均幅度比较
        print(f"\n平均幅度:")
        print(f"  真实验证码: {real_stats['mean_magnitude']:.4f}")
        print(f"  合成验证码: {synthetic_stats['mean_magnitude']:.4f}")
        
        # T检验
        real_energy = real_stats['energy_distributions'].mean(axis=0)
        synthetic_energy = synthetic_stats['energy_distributions'].mean(axis=0)
        t_stat, p_value = stats.ttest_ind(real_energy, synthetic_energy)
        
        print(f"\n能量分布T检验:")
        print(f"  T统计量: {t_stat:.4f}")
        print(f"  P值: {p_value:.4f}")
        
        if p_value < 0.05:
            print("  结论: 两个数据集的频谱能量分布存在显著差异")
        else:
            print("  结论: 两个数据集的频谱能量分布无显著差异")
    
    def visualize_comparison(self, real_stats: Dict, synthetic_stats: Dict):
        """可视化比较结果"""
        # 创建输出目录
        output_dir = Path("test_output/fft_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 样本频谱对比 - 每个样本一张图
        for i in range(min(5, len(real_stats['sample_spectrums']))):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 真实验证码频谱
            im1 = axes[0].imshow(real_stats['sample_spectrums'][i], cmap='hot')
            axes[0].set_title(f'Real CAPTCHA {i+1} Spectrum', fontsize=14)
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            
            # 合成验证码频谱
            im2 = axes[1].imshow(synthetic_stats['sample_spectrums'][i], cmap='hot')
            axes[1].set_title(f'Synthetic CAPTCHA {i+1} Spectrum', fontsize=14)
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            
            plt.suptitle(f'Sample {i+1}: 2D FFT Spectrum Comparison', fontsize=16)
            plt.tight_layout()
            
            sample_path = output_dir / f"spectrum_sample_{i+1}.png"
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"样本 {i+1} 频谱对比已保存到: {sample_path}")
        
        # 2. 径向能量分布对比
        fig, ax = plt.subplots(figsize=(10, 6))
        real_energy_mean = real_stats['energy_distributions'].mean(axis=0)
        real_energy_std = real_stats['energy_distributions'].std(axis=0)
        synthetic_energy_mean = synthetic_stats['energy_distributions'].mean(axis=0)
        synthetic_energy_std = synthetic_stats['energy_distributions'].std(axis=0)
        
        x = np.arange(len(real_energy_mean))
        ax.plot(x, real_energy_mean, 'b-', label='Real CAPTCHA', linewidth=2.5)
        ax.fill_between(x, real_energy_mean - real_energy_std, 
                        real_energy_mean + real_energy_std, alpha=0.3, color='blue')
        ax.plot(x, synthetic_energy_mean, 'r-', label='Synthetic CAPTCHA', linewidth=2.5)
        ax.fill_between(x, synthetic_energy_mean - synthetic_energy_std,
                        synthetic_energy_mean + synthetic_energy_std, alpha=0.3, color='red')
        ax.set_xlabel('Frequency Band (Low → High)', fontsize=12)
        ax.set_ylabel('Average Energy', fontsize=12)
        ax.set_title('Radial Energy Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        energy_path = output_dir / "radial_energy_distribution.png"
        plt.savefig(energy_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"径向能量分布对比已保存到: {energy_path}")
        
        # 3. 高低频比例对比
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ['Low Frequency\n(< 10% radius)', 'High Frequency\n(> 70% radius)']
        real_values = [real_stats['low_freq_ratio'], real_stats['high_freq_ratio']]
        synthetic_values = [synthetic_stats['low_freq_ratio'], synthetic_stats['high_freq_ratio']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, real_values, width, label='Real CAPTCHA', 
                      color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, synthetic_values, width, label='Synthetic CAPTCHA', 
                      color='#A23B72', alpha=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Frequency Component', fontsize=12)
        ax.set_ylabel('Energy Ratio', fontsize=12)
        ax.set_title('Frequency Component Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        freq_path = output_dir / "frequency_component_distribution.png"
        plt.savefig(freq_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"频率成分分布对比已保存到: {freq_path}")
        
        # 4. 高频能量箱线图对比
        fig, ax = plt.subplots(figsize=(8, 6))
        all_real_high = [row[-10:].mean() for row in real_stats['energy_distributions']]
        all_synthetic_high = [row[-10:].mean() for row in synthetic_stats['energy_distributions']]
        
        bp = ax.boxplot([all_real_high, all_synthetic_high], 
                        labels=['Real CAPTCHA', 'Synthetic CAPTCHA'],
                        patch_artist=True, widths=0.6)
        
        # 设置颜色
        colors = ['#2E86AB', '#A23B72']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('High Frequency Energy (Top 10 bands)', fontsize=12)
        ax.set_title('High Frequency Energy Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        box_path = output_dir / "high_frequency_boxplot.png"
        plt.savefig(box_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"高频能量箱线图已保存到: {box_path}")
        
        # 5. 能量分布差异图
        fig, ax = plt.subplots(figsize=(10, 6))
        diff = real_energy_mean - synthetic_energy_mean
        
        ax.plot(diff, 'g-', linewidth=2.5, label='Real - Synthetic')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax.fill_between(np.arange(len(diff)), 0, diff, 
                       where=(diff >= 0), alpha=0.3, color='green', 
                       label='Real > Synthetic')
        ax.fill_between(np.arange(len(diff)), 0, diff, 
                       where=(diff < 0), alpha=0.3, color='red',
                       label='Real < Synthetic')
        
        ax.set_xlabel('Frequency Band (Low → High)', fontsize=12)
        ax.set_ylabel('Energy Difference', fontsize=12)
        ax.set_title('Energy Distribution Difference (Real - Synthetic)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        diff_path = output_dir / "energy_difference.png"
        plt.savefig(diff_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"能量差异图已保存到: {diff_path}")
        
        # 6. 创建汇总热力图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 真实验证码平均频谱
        avg_real_spectrum = np.mean([s for s in real_stats['sample_spectrums']], axis=0)
        im1 = axes[0].imshow(avg_real_spectrum, cmap='hot', aspect='auto')
        axes[0].set_title('Average Real CAPTCHA Spectrum', fontsize=14)
        axes[0].set_xlabel('Frequency X')
        axes[0].set_ylabel('Frequency Y')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 合成验证码平均频谱
        avg_synthetic_spectrum = np.mean([s for s in synthetic_stats['sample_spectrums']], axis=0)
        im2 = axes[1].imshow(avg_synthetic_spectrum, cmap='hot', aspect='auto')
        axes[1].set_title('Average Synthetic CAPTCHA Spectrum', fontsize=14)
        axes[1].set_xlabel('Frequency X')
        axes[1].set_ylabel('Frequency Y')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.suptitle('Average 2D FFT Spectrum Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        avg_path = output_dir / "average_spectrum_comparison.png"
        plt.savefig(avg_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"平均频谱对比已保存到: {avg_path}")
        
        logger.info(f"\n所有图表已保存到: {output_dir}")
    
    def run_comparison(self):
        """运行完整的比较分析"""
        logger.info("开始2D FFT频谱比较分析...")
        
        # 加载图像
        real_images = self.load_random_images(self.real_dir, self.sample_size)
        synthetic_images = self.load_random_images(self.synthetic_dir, self.sample_size)
        
        if not real_images or not synthetic_images:
            logger.error("无法加载足够的图像进行比较")
            return
        
        # 分析数据集
        real_stats = self.analyze_dataset(real_images, "真实验证码")
        synthetic_stats = self.analyze_dataset(synthetic_images, "合成验证码")
        
        # 统计比较
        self.statistical_comparison(real_stats, synthetic_stats)
        
        # 可视化
        self.visualize_comparison(real_stats, synthetic_stats)
        
        logger.info("分析完成！")
        
        return real_stats, synthetic_stats


def main():
    """主函数"""
    # 设置随机种子以保证可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 创建分析器并运行比较
    analyzer = FFTAnalyzer(sample_size=100)
    real_stats, synthetic_stats = analyzer.run_comparison()
    
    # 输出主要发现
    print("\n" + "="*50)
    print("主要发现:")
    print("="*50)
    
    high_freq_diff = abs(real_stats['high_freq_ratio'] - synthetic_stats['high_freq_ratio'])
    low_freq_diff = abs(real_stats['low_freq_ratio'] - synthetic_stats['low_freq_ratio'])
    
    if high_freq_diff > 0.05:
        print(f"• 高频成分差异显著 (差值: {high_freq_diff:.4f})")
        if real_stats['high_freq_ratio'] > synthetic_stats['high_freq_ratio']:
            print("  → 真实验证码包含更多高频细节（可能有更多噪声或纹理）")
        else:
            print("  → 合成验证码包含更多高频细节（可能过度锐化）")
    
    if low_freq_diff > 0.05:
        print(f"• 低频成分差异显著 (差值: {low_freq_diff:.4f})")
        if real_stats['low_freq_ratio'] > synthetic_stats['low_freq_ratio']:
            print("  → 真实验证码有更平滑的背景")
        else:
            print("  → 合成验证码有更平滑的背景")
    
    print("\n建议:")
    print("• 根据频谱差异调整合成算法以更接近真实数据")
    print("• 考虑在训练时混合使用两种数据源")
    print("• 可能需要对合成数据添加适当的噪声或纹理")


if __name__ == "__main__":
    main()