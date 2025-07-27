# -*- coding: utf-8 -*-
"""
Pixabay图片下载器
从Pixabay API下载指定类别的图片
"""
import os
import time
import yaml
import requests
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from tqdm import tqdm


class PixabayDownloader:
    """Pixabay图片下载器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化下载器
        
        Args:
            config_path: 配置文件路径
        """
        # 如果没有指定配置文件路径，使用默认路径
        if config_path is None:
            # 获取项目根目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            config_path = project_root / "configs" / "config.yaml"
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.api_key = config['pixabay']['api_key']
        self.base_url = config['pixabay']['base_url']
        self.categories = config['data_collection']['categories']
        self.images_per_category = config['data_collection']['images_per_category']
        self.output_dir = Path(config['data_collection']['output_dir'])
        self.timeout = config['data_collection']['download_timeout']
        self.max_retries = config['data_collection']['max_retries']
        self.concurrent_downloads = config['data_collection']['concurrent_downloads']
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logger.add(
            "logs/data_collection.log",
            rotation="500 MB",
            encoding="utf-8",
            enqueue=True
        )
        
    def search_images(self, category: str, per_page: int = 200) -> List[Dict]:
        """
        搜索指定类别的图片
        
        Args:
            category: 搜索类别
            per_page: 每页返回数量
            
        Returns:
            图片信息列表
        """
        all_images = []
        page = 1
        
        # 多页搜索以获取足够的图片
        while len(all_images) < self.images_per_category:
            params = {
                'key': self.api_key,
                'q': category,
                'image_type': 'photo',
                'per_page': min(per_page, 200),  # API最多返回200
                'page': page,
                'safesearch': 'true',
                'order': 'popular'
            }
            
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                if 'hits' in data and len(data['hits']) > 0:
                    all_images.extend(data['hits'])
                    logger.info(f"Page {page}: Found {len(data['hits'])} images for '{category}' (Total: {len(all_images)})")
                    
                    # 如果这一页返回的图片少于请求的数量，说明没有更多图片了
                    if len(data['hits']) < per_page:
                        logger.warning(f"No more images available for '{category}'. Total found: {len(all_images)}")
                        break
                    
                    page += 1
                else:
                    logger.warning(f"No more images found for '{category}' on page {page}")
                    break
                    
            except Exception as e:
                logger.error(f"Error searching images for '{category}' on page {page}: {str(e)}")
                break
        
        # 返回所需数量的图片
        result = all_images[:self.images_per_category]
        logger.info(f"Returning {len(result)} images for '{category}' (requested: {self.images_per_category})")
        return result
    
    async def download_image(
        self,
        session: aiohttp.ClientSession,
        url: str,
        save_path: Path,
        retries: int = 0
    ) -> bool:
        """
        异步下载单张图片
        
        Args:
            session: aiohttp会话
            url: 图片URL
            save_path: 保存路径
            retries: 当前重试次数
            
        Returns:
            是否下载成功
        """
        try:
            async with session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    content = await response.read()
                    save_path.write_bytes(content)
                    return True
                elif response.status == 429:
                    # 速率限制，需要等待更长时间
                    wait_time = 10 * (retries + 1)  # 递增等待时间
                    logger.warning(f"Rate limited (HTTP 429). Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    if retries < self.max_retries:
                        return await self.download_image(session, url, save_path, retries + 1)
                    return False
                else:
                    logger.warning(f"Failed to download {url}: HTTP {response.status}")
                    return False
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading {url}")
            if retries < self.max_retries:
                await asyncio.sleep(2 ** retries)  # 指数退避
                return await self.download_image(session, url, save_path, retries + 1)
            return False
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            if retries < self.max_retries:
                await asyncio.sleep(2 ** retries)
                return await self.download_image(session, url, save_path, retries + 1)
            return False
    
    async def download_category_async(self, category: str, image_infos: List[Dict]):
        """
        异步下载一个类别的所有图片
        
        Args:
            category: 类别名称
            image_infos: 图片信息列表
        """
        # 创建类别目录
        category_dir = self.output_dir / category.replace(' ', '_')
        category_dir.mkdir(exist_ok=True)
        
        # 创建下载任务
        connector = aiohttp.TCPConnector(limit=self.concurrent_downloads)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            
            for idx, info in enumerate(image_infos):
                # 使用webformatURL（中等尺寸）而不是largeImageURL（大尺寸）
                image_url = info.get('webformatURL', info.get('largeImageURL'))
                if not image_url:
                    continue
                
                # 生成文件名：Pic{4位数字}.png
                filename = f"Pic{idx+1:04d}.png"
                save_path = category_dir / filename
                
                # 跳过已存在的文件
                if save_path.exists():
                    logger.info(f"Skip existing file: {save_path}")
                    continue
                
                # 添加延迟以避免触发速率限制
                if len(tasks) > 0 and len(tasks) % 10 == 0:
                    await asyncio.sleep(1)  # 每10个请求暂停1秒
                
                task = self.download_image(session, image_url, save_path)
                tasks.append(task)
            
            # 使用进度条显示下载进度
            if tasks:
                results = []
                with tqdm(total=len(tasks), desc=f"Downloading {category}") as pbar:
                    for task in asyncio.as_completed(tasks):
                        result = await task
                        results.append(result)
                        pbar.update(1)
                
                success_count = sum(results)
                logger.info(f"Downloaded {success_count}/{len(tasks)} images for {category}")
    
    def download_all_categories(self):
        """下载所有类别的图片"""
        logger.info(f"Starting download for {len(self.categories)} categories")
        
        for category in self.categories:
            logger.info(f"Processing category: {category}")
            
            # 搜索图片
            image_infos = self.search_images(category)
            if not image_infos:
                logger.warning(f"No images found for {category}, skipping...")
                continue
            
            # 异步下载
            asyncio.run(self.download_category_async(category, image_infos))
            
            # 避免请求过快
            time.sleep(1)
        
        logger.info("Download completed!")
        
        # 统计下载结果
        self._print_statistics()
    
    def _print_statistics(self):
        """打印下载统计信息"""
        total_images = 0
        category_stats = []
        
        for category_dir in self.output_dir.iterdir():
            if category_dir.is_dir():
                image_count = len(list(category_dir.glob("*.png")))
                total_images += image_count
                category_stats.append((category_dir.name, image_count))
        
        logger.info("=" * 50)
        logger.info("Download Statistics:")
        logger.info("=" * 50)
        for category, count in category_stats:
            logger.info(f"{category}: {count} images")
        logger.info(f"Total: {total_images} images")
        logger.info("=" * 50)


def main():
    """主函数"""
    downloader = PixabayDownloader()
    downloader.download_all_categories()


if __name__ == "__main__":
    main()