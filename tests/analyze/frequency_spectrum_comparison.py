#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Frequency spectrum comparison between site3 and test directories
Analyzes spectral characteristics and differences between two CAPTCHA datasets
"""

import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def compute_frequency_features(image_path):
    """
    Compute frequency domain features for an image
    """
    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Compute FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # Magnitude spectrum
    magnitude = np.abs(fshift)
    magnitude_log = np.log(magnitude + 1)
    
    # Phase spectrum
    phase = np.angle(fshift)
    
    # Compute radial profile
    center = (img.shape[0]//2, img.shape[1]//2)
    y, x = np.indices(img.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    # Radial average
    tbin = np.bincount(r.ravel(), magnitude.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    
    # Frequency bands analysis
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    # Define frequency bands (in pixels from center)
    low_freq_radius = min(rows, cols) // 8
    mid_freq_radius = min(rows, cols) // 4
    high_freq_radius = min(rows, cols) // 2
    
    # Create masks for frequency bands
    mask_low = np.zeros_like(magnitude)
    mask_mid = np.zeros_like(magnitude)
    mask_high = np.zeros_like(magnitude)
    
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if dist <= low_freq_radius:
                mask_low[i, j] = 1
            elif dist <= mid_freq_radius:
                mask_mid[i, j] = 1
            else:
                mask_high[i, j] = 1
    
    # Calculate energy in each band
    low_freq_energy = np.sum(magnitude * mask_low) / np.sum(mask_low)
    mid_freq_energy = np.sum(magnitude * mask_mid) / np.sum(mask_mid)
    high_freq_energy = np.sum(magnitude * mask_high) / np.sum(mask_high)
    
    # Normalize energies
    total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
    if total_energy > 0:
        low_freq_ratio = low_freq_energy / total_energy
        mid_freq_ratio = mid_freq_energy / total_energy
        high_freq_ratio = high_freq_energy / total_energy
    else:
        low_freq_ratio = mid_freq_ratio = high_freq_ratio = 0
    
    return {
        'magnitude': magnitude,
        'magnitude_log': magnitude_log,
        'phase': phase,
        'radial_profile': radial_profile,
        'low_freq_ratio': low_freq_ratio,
        'mid_freq_ratio': mid_freq_ratio,
        'high_freq_ratio': high_freq_ratio,
        'shape': img.shape
    }


def analyze_directory(directory_path, sample_size=50):
    """
    Analyze frequency characteristics of images in a directory
    """
    # Get all PNG files
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
    
    # Sample if too many
    if len(image_files) > sample_size:
        image_files = random.sample(image_files, sample_size)
    
    print(f"Analyzing {len(image_files)} images from {os.path.basename(directory_path)}...")
    
    features_list = []
    radial_profiles = []
    
    for img_file in image_files:
        img_path = os.path.join(directory_path, img_file)
        features = compute_frequency_features(img_path)
        
        if features is not None:
            features_list.append(features)
            # Normalize radial profile length
            profile = features['radial_profile'][:100]  # Take first 100 frequencies
            if len(profile) < 100:
                profile = np.pad(profile, (0, 100 - len(profile)), mode='constant')
            radial_profiles.append(profile)
    
    # Compute statistics
    if features_list:
        low_freq_ratios = [f['low_freq_ratio'] for f in features_list]
        mid_freq_ratios = [f['mid_freq_ratio'] for f in features_list]
        high_freq_ratios = [f['high_freq_ratio'] for f in features_list]
        
        stats_dict = {
            'low_freq_mean': np.mean(low_freq_ratios),
            'low_freq_std': np.std(low_freq_ratios),
            'mid_freq_mean': np.mean(mid_freq_ratios),
            'mid_freq_std': np.std(mid_freq_ratios),
            'high_freq_mean': np.mean(high_freq_ratios),
            'high_freq_std': np.std(high_freq_ratios),
            'radial_profiles': np.array(radial_profiles),
            'mean_radial_profile': np.mean(radial_profiles, axis=0),
            'std_radial_profile': np.std(radial_profiles, axis=0),
            'all_features': features_list
        }
        
        return stats_dict
    
    return None


def visualize_comparison(site3_stats, test_stats, output_dir):
    """
    Create comprehensive visualization comparing two datasets
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Sample images and their spectrums
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Get sample images
    site3_sample = site3_stats['all_features'][0]
    test_sample = test_stats['all_features'][0]
    
    # Plot magnitude spectrums
    ax1.imshow(site3_sample['magnitude_log'], cmap='hot')
    ax1.set_title('Site3 - Magnitude Spectrum', fontsize=10)
    ax1.axis('off')
    
    ax2.imshow(test_sample['magnitude_log'], cmap='hot')
    ax2.set_title('Test - Magnitude Spectrum', fontsize=10)
    ax2.axis('off')
    
    # Difference
    # Resize if necessary
    h1, w1 = site3_sample['magnitude_log'].shape
    h2, w2 = test_sample['magnitude_log'].shape
    min_h, min_w = min(h1, h2), min(w1, w2)
    
    diff = np.abs(site3_sample['magnitude_log'][:min_h, :min_w] - 
                  test_sample['magnitude_log'][:min_h, :min_w])
    
    ax3.imshow(diff, cmap='coolwarm')
    ax3.set_title('Spectrum Difference', fontsize=10)
    ax3.axis('off')
    
    # Phase comparison
    phase_diff = np.abs(site3_sample['phase'][:min_h, :min_w] - 
                       test_sample['phase'][:min_h, :min_w])
    ax4.imshow(phase_diff, cmap='twilight')
    ax4.set_title('Phase Difference', fontsize=10)
    ax4.axis('off')
    
    # 2. Radial profiles comparison
    ax5 = fig.add_subplot(gs[1, :2])
    
    # Plot mean radial profiles with confidence intervals
    frequencies = np.arange(len(site3_stats['mean_radial_profile']))
    
    ax5.plot(frequencies, site3_stats['mean_radial_profile'], 
            label='Site3', color='blue', linewidth=2)
    ax5.fill_between(frequencies,
                     site3_stats['mean_radial_profile'] - site3_stats['std_radial_profile'],
                     site3_stats['mean_radial_profile'] + site3_stats['std_radial_profile'],
                     alpha=0.3, color='blue')
    
    ax5.plot(frequencies, test_stats['mean_radial_profile'], 
            label='Test', color='red', linewidth=2)
    ax5.fill_between(frequencies,
                     test_stats['mean_radial_profile'] - test_stats['std_radial_profile'],
                     test_stats['mean_radial_profile'] + test_stats['std_radial_profile'],
                     alpha=0.3, color='red')
    
    ax5.set_xlabel('Frequency (pixels from center)')
    ax5.set_ylabel('Average Magnitude')
    ax5.set_title('Radial Frequency Profile Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 3. Frequency band energy comparison
    ax6 = fig.add_subplot(gs[1, 2:])
    
    categories = ['Low Freq\n(0-12.5%)', 'Mid Freq\n(12.5-25%)', 'High Freq\n(25-50%)']
    site3_means = [site3_stats['low_freq_mean'], 
                   site3_stats['mid_freq_mean'], 
                   site3_stats['high_freq_mean']]
    site3_stds = [site3_stats['low_freq_std'], 
                  site3_stats['mid_freq_std'], 
                  site3_stats['high_freq_std']]
    
    test_means = [test_stats['low_freq_mean'], 
                  test_stats['mid_freq_mean'], 
                  test_stats['high_freq_mean']]
    test_stds = [test_stats['low_freq_std'], 
                 test_stats['mid_freq_std'], 
                 test_stats['high_freq_std']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, site3_means, width, yerr=site3_stds, 
                    label='Site3', color='skyblue', capsize=5)
    bars2 = ax6.bar(x + width/2, test_means, width, yerr=test_stds, 
                    label='Test', color='lightcoral', capsize=5)
    
    ax6.set_xlabel('Frequency Band')
    ax6.set_ylabel('Energy Ratio')
    ax6.set_title('Frequency Band Energy Distribution')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax6.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax6.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 4. Statistical comparison heatmap
    ax7 = fig.add_subplot(gs[2, :2])
    
    # Create comparison matrix
    metrics = ['Low Freq', 'Mid Freq', 'High Freq']
    site3_data = np.array([site3_means]).T
    test_data = np.array([test_means]).T
    
    comparison_data = np.hstack([site3_data, test_data])
    
    im = ax7.imshow(comparison_data, cmap='YlOrRd', aspect='auto')
    ax7.set_xticks([0, 1])
    ax7.set_xticklabels(['Site3', 'Test'])
    ax7.set_yticks(range(len(metrics)))
    ax7.set_yticklabels(metrics)
    ax7.set_title('Energy Distribution Heatmap')
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(2):
            text = ax7.text(j, i, f'{comparison_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax7)
    
    # 5. Power spectrum density difference
    ax8 = fig.add_subplot(gs[2, 2:])
    
    # Calculate PSD difference
    psd_diff = site3_stats['mean_radial_profile'] - test_stats['mean_radial_profile']
    
    ax8.plot(frequencies, psd_diff, color='purple', linewidth=2)
    ax8.fill_between(frequencies, 0, psd_diff, 
                     where=(psd_diff > 0), color='blue', alpha=0.3, label='Site3 > Test')
    ax8.fill_between(frequencies, 0, psd_diff, 
                     where=(psd_diff < 0), color='red', alpha=0.3, label='Test > Site3')
    
    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Frequency (pixels from center)')
    ax8.set_ylabel('Power Difference')
    ax8.set_title('Power Spectrum Density Difference')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('Frequency Spectrum Analysis: Site3 vs Test Dataset', 
                 fontsize=16, fontweight='bold')
    
    # Save figure
    output_path = os.path.join(output_dir, 'frequency_spectrum_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison visualization to: {output_path}")
    
    return fig


def generate_detailed_report(site3_stats, test_stats, output_dir):
    """
    Generate detailed textual report of differences
    """
    report_lines = []
    
    report_lines.append("\n" + "="*70)
    report_lines.append("FREQUENCY SPECTRUM ANALYSIS REPORT")
    report_lines.append("="*70)
    
    report_lines.append("\n📊 DATASET OVERVIEW")
    report_lines.append("-"*40)
    report_lines.append(f"Site3 samples analyzed: {len(site3_stats['all_features'])}")
    report_lines.append(f"Test samples analyzed: {len(test_stats['all_features'])}")
    
    report_lines.append("\n🔍 FREQUENCY BAND ANALYSIS")
    report_lines.append("-"*40)
    
    # Low frequency comparison
    low_diff = abs(site3_stats['low_freq_mean'] - test_stats['low_freq_mean'])
    report_lines.append(f"\nLow Frequency (0-12.5% radius):")
    report_lines.append(f"  Site3: {site3_stats['low_freq_mean']:.4f} ± {site3_stats['low_freq_std']:.4f}")
    report_lines.append(f"  Test:  {test_stats['low_freq_mean']:.4f} ± {test_stats['low_freq_std']:.4f}")
    report_lines.append(f"  Difference: {low_diff:.4f} ({low_diff/site3_stats['low_freq_mean']*100:.1f}%)")
    
    # Mid frequency comparison
    mid_diff = abs(site3_stats['mid_freq_mean'] - test_stats['mid_freq_mean'])
    report_lines.append(f"\nMid Frequency (12.5-25% radius):")
    report_lines.append(f"  Site3: {site3_stats['mid_freq_mean']:.4f} ± {site3_stats['mid_freq_std']:.4f}")
    report_lines.append(f"  Test:  {test_stats['mid_freq_mean']:.4f} ± {test_stats['mid_freq_std']:.4f}")
    report_lines.append(f"  Difference: {mid_diff:.4f} ({mid_diff/site3_stats['mid_freq_mean']*100:.1f}%)")
    
    # High frequency comparison
    high_diff = abs(site3_stats['high_freq_mean'] - test_stats['high_freq_mean'])
    report_lines.append(f"\nHigh Frequency (25-50% radius):")
    report_lines.append(f"  Site3: {site3_stats['high_freq_mean']:.4f} ± {site3_stats['high_freq_std']:.4f}")
    report_lines.append(f"  Test:  {test_stats['high_freq_mean']:.4f} ± {test_stats['high_freq_std']:.4f}")
    report_lines.append(f"  Difference: {high_diff:.4f} ({high_diff/site3_stats['high_freq_mean']*100:.1f}%)")
    
    report_lines.append("\n📈 KEY FINDINGS")
    report_lines.append("-"*40)
    
    # Determine which has more high frequency content
    if site3_stats['high_freq_mean'] > test_stats['high_freq_mean']:
        report_lines.append("• Site3 images contain MORE high-frequency content")
        report_lines.append(f"  → {(site3_stats['high_freq_mean']/test_stats['high_freq_mean'] - 1)*100:.1f}% more high-freq energy")
        report_lines.append("  → Likely more detailed edges, textures, or noise")
    else:
        report_lines.append("• Test images contain MORE high-frequency content")
        report_lines.append(f"  → {(test_stats['high_freq_mean']/site3_stats['high_freq_mean'] - 1)*100:.1f}% more high-freq energy")
        report_lines.append("  → Likely more detailed edges, textures, or noise")
    
    # Determine which has more low frequency content
    if site3_stats['low_freq_mean'] > test_stats['low_freq_mean']:
        report_lines.append("\n• Site3 images have STRONGER low-frequency components")
        report_lines.append("  → More dominant large-scale structures")
        report_lines.append("  → Potentially smoother backgrounds")
    else:
        report_lines.append("\n• Test images have STRONGER low-frequency components")
        report_lines.append("  → More dominant large-scale structures")
        report_lines.append("  → Potentially smoother backgrounds")
    
    # Statistical significance test
    report_lines.append("\n📊 STATISTICAL SIGNIFICANCE")
    report_lines.append("-"*40)
    
    # Perform t-test on radial profiles
    from scipy import stats as scipy_stats
    
    site3_profiles = site3_stats['radial_profiles']
    test_profiles = test_stats['radial_profiles']
    
    # Compare mean profiles
    t_stat, p_value = scipy_stats.ttest_ind(
        site3_profiles.mean(axis=1),
        test_profiles.mean(axis=1)
    )
    
    report_lines.append(f"T-test on radial profiles:")
    report_lines.append(f"  t-statistic: {t_stat:.4f}")
    report_lines.append(f"  p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        report_lines.append("  ✅ Statistically significant difference (p < 0.05)")
    else:
        report_lines.append("  ❌ No statistically significant difference (p >= 0.05)")
    
    report_lines.append("\n🎯 PRACTICAL IMPLICATIONS")
    report_lines.append("-"*40)
    
    if high_diff > 0.01:
        report_lines.append("• High frequency difference suggests:")
        report_lines.append("  - Different image processing or compression")
        report_lines.append("  - Different noise characteristics")
        report_lines.append("  - Different edge sharpness or detail levels")
    
    if mid_diff > 0.01:
        report_lines.append("\n• Mid frequency difference indicates:")
        report_lines.append("  - Different texture patterns")
        report_lines.append("  - Different object sizes or scales")
    
    if low_diff > 0.01:
        report_lines.append("\n• Low frequency difference implies:")
        report_lines.append("  - Different background characteristics")
        report_lines.append("  - Different overall brightness patterns")
        report_lines.append("  - Different image composition styles")
    
    report_lines.append("\n" + "="*70)
    
    # Print to console
    for line in report_lines:
        print(line)
    
    # Save to file
    report_path = os.path.join(output_dir, 'frequency_spectrum_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✅ Saved detailed report to: {report_path}")


def main():
    """
    Main function to run the frequency spectrum comparison
    """
    # Create output directory
    output_dir = r"D:\Hacker\Sider_CAPTCHA_Solver\tests\analyze\analyze_result\frequency_spectrum"
    os.makedirs(output_dir, exist_ok=True)
    
    site3_dir = r"D:\Hacker\Sider_CAPTCHA_Solver\data\real_captchas\annotated\site3"
    test_dir = r"D:\Hacker\Sider_CAPTCHA_Solver\data\real_captchas\annotated\test"
    
    print("="*70)
    print("FREQUENCY SPECTRUM COMPARISON ANALYSIS")
    print("Site3 vs Test Dataset")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    
    # Analyze both directories
    print("\n📁 Analyzing Site3 directory...")
    site3_stats = analyze_directory(site3_dir, sample_size=50)
    
    print("\n📁 Analyzing Test directory...")
    test_stats = analyze_directory(test_dir, sample_size=50)
    
    if site3_stats and test_stats:
        # Create visualizations
        print("\n📊 Creating visualizations...")
        visualize_comparison(site3_stats, test_stats, output_dir)
        
        # Generate report
        generate_detailed_report(site3_stats, test_stats, output_dir)
        
        print("\n✅ Analysis complete!")
        print(f"Generated files in: {output_dir}")
    else:
        print("\n❌ Error: Could not analyze one or both directories")


if __name__ == "__main__":
    main()