# -*- coding: utf-8 -*-
"""
日志配置工具
"""
import os
from loguru import logger
from pathlib import Path


def setup_logger(
    log_file: str = "logs/app.log",
    level: str = "INFO",
    rotation: str = "500 MB",
    retention: str = "10 days",
    encoding: str = "utf-8"
):
    """
    配置日志
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
        rotation: 日志轮转大小
        retention: 日志保留时间
        encoding: 编码格式
    """
    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 移除默认的处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=level
    )
    
    # 添加文件输出
    logger.add(
        sink=log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        encoding=encoding,
        enqueue=True  # 异步写入
    )
    
    return logger