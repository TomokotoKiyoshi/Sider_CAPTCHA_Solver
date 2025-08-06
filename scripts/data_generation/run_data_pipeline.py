#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generation pipeline main script
Coordinates the entire dataset generation process
"""
import sys
from pathlib import Path
import subprocess
import json
import argparse
from datetime import datetime
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_raw_data(raw_dir: Path) -> dict:
    """Check if raw data is ready"""
    print("\n=== Checking Raw Data ===")
    
    if not raw_dir.exists():
        return {'status': False, 'message': f'Raw data directory does not exist: {raw_dir}'}
    
    categories = list(raw_dir.glob('*/'))
    if not categories:
        return {'status': False, 'message': 'Raw data directory is empty'}
    
    # Count images in each category
    category_stats = {}
    total_images = 0
    
    for category in categories:
        if category.is_dir():
            images = list(category.glob('*.png')) + list(category.glob('*.jpg'))
            category_stats[category.name] = len(images)
            total_images += len(images)
    
    print(f"Found {len(category_stats)} categories with {total_images} images total")
    for cat, count in category_stats.items():
        print(f"  - {cat}: {count} images")
    
    if total_images < 100:
        return {'status': False, 'message': f'Too few images: {total_images}'}
    
    return {
        'status': True, 
        'categories': len(category_stats),
        'total_images': total_images,
        'stats': category_stats
    }


def download_images(force: bool = False) -> bool:
    """Download images if needed"""
    print("\n=== Downloading Images ===")
    
    raw_dir = project_root / 'data' / 'raw'
    check_result = check_raw_data(raw_dir)
    
    if check_result['status'] and not force:
        print("Raw data already exists, skipping download")
        return True
    
    print("Starting image download...")
    download_script = project_root / 'scripts' / 'data_generation' / 'download_images.py'
    
    try:
        result = subprocess.run(
            [sys.executable, str(download_script)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def generate_captchas(workers: int = None, max_images: int = None) -> bool:
    """Generate CAPTCHA dataset"""
    print("\n=== Generating CAPTCHAs ===")
    
    generate_script = project_root / 'scripts' / 'data_generation' / 'generate_captchas.py'
    
    # Build command
    cmd = [sys.executable, str(generate_script)]
    if workers:
        cmd.extend(['--workers', str(workers)])
    if max_images:
        cmd.extend(['--max-images', str(max_images)])
    
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Generation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def split_dataset() -> bool:
    """Split dataset into train and test sets"""
    print("\n=== Splitting Dataset ===")
    
    split_script = project_root / 'scripts' / 'data_generation' / 'split_dataset.py'
    
    if not split_script.exists():
        print(f"Warning: Dataset split script does not exist: {split_script}")
        print("Skipping dataset split step")
        return True
    
    try:
        result = subprocess.run(
            [sys.executable, str(split_script)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Split failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def verify_output(output_dir: Path) -> dict:
    """Verify output results"""
    print("\n=== Verifying Output ===")
    
    if not output_dir.exists():
        return {'status': False, 'message': 'Output directory does not exist'}
    
    # Check key files
    annotations_file = output_dir / 'all_annotations.json'
    report_file = output_dir / 'generation_report.json'
    
    if not annotations_file.exists():
        return {'status': False, 'message': 'Annotation file does not exist'}
    
    if not report_file.exists():
        return {'status': False, 'message': 'Report file does not exist'}
    
    # Read statistics
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    total_samples = report.get('total_samples', 0)
    total_backgrounds = report.get('total_backgrounds', 0)
    
    print(f"Generation completed:")
    print(f"  - Total samples: {total_samples}")
    print(f"  - Background images: {total_backgrounds}")
    print(f"  - Generation time: {report.get('generation_time', 'N/A')}")
    
    # Count image files
    image_files = list(output_dir.glob('*.png'))
    print(f"  - Image files: {len(image_files)}")
    
    return {
        'status': True,
        'total_samples': total_samples,
        'total_backgrounds': total_backgrounds,
        'image_files': len(image_files)
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Data generation pipeline - Complete process from download to CAPTCHA generation'
    )
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download step (if raw data already exists)')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download images')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes for CAPTCHA generation')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of background images to use')
    parser.add_argument('--skip-split', action='store_true',
                        help='Skip dataset split step')
    parser.add_argument('--clean', action='store_true',
                        help='Clean output directory before regenerating')
    
    args = parser.parse_args()
    
    print("=== Data Generation Pipeline Started ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define paths
    raw_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'data' / 'captchas'
    
    # Clean output directory if needed
    if args.clean and output_dir.exists():
        print(f"\nCleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Step 1: Download images
    if not args.skip_download:
        if not download_images(force=args.force_download):
            print("\n[ERROR] Image download failed")
            return 1
    
    # Step 2: Check raw data
    check_result = check_raw_data(raw_dir)
    if not check_result['status']:
        print(f"\n[ERROR] Raw data check failed: {check_result.get('message', '')}")
        return 1
    
    # Step 3: Generate CAPTCHAs
    if not generate_captchas(workers=args.workers, max_images=args.max_images):
        print("\n[ERROR] CAPTCHA generation failed")
        return 1
    
    # Step 4: Verify output
    verify_result = verify_output(output_dir)
    if not verify_result['status']:
        print(f"\n[ERROR] Output verification failed: {verify_result.get('message', '')}")
        return 1
    
    # Step 5: Split dataset (optional)
    if not args.skip_split:
        if not split_dataset():
            print("\n[WARNING] Dataset split failed, but generation process completed")
    
    print("\n[SUCCESS] Data generation pipeline completed!")
    print(f"Output directory: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())