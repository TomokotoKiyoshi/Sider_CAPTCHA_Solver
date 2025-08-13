# -*- coding: utf-8 -*-
"""
模型工具函数模块
包含权重初始化、参数统计等通用功能
"""
import torch
import torch.nn as nn


def init_weights(module):
    """
    权重初始化函数
    使用Kaiming Normal初始化卷积层，适用于ReLU/SiLU激活函数
    
    Args:
        module: nn.Module对象
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight, mode='fan_out', nonlinearity='relu'
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(model):
    """
    统计模型参数量
    
    Args:
        model: nn.Module对象
    
    Returns:
        tuple: (总参数量, 可训练参数量)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size(model):
    """
    获取模型大小（MB）
    
    Args:
        model: nn.Module对象
    
    Returns:
        float: 模型大小（MB）
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb