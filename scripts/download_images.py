#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行数据采集的脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.pixabay_downloader import PixabayDownloader


def main():
    """主函数"""
    print("Starting Pixabay image download...")
    print("=" * 50)
    
    # 创建下载器并开始下载
    downloader = PixabayDownloader()
    downloader.download_all_categories()
    
    print("\nDownload completed!")
    print("Check the data/raw directory for downloaded images.")


if __name__ == "__main__":
    main()