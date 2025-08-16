# PyPI最小化包结构设计（复用现有代码）

## 🎯 核心原则：最大化复用现有 src 代码

### 现状分析

您已经有完整的 `src/` 目录结构：
- `src/models/` - 模型架构（lite_hrnet_18_fpn.py等）✅
- `src/training/` - 训练相关（但推理不需要）
- `src/captcha_generator/` - 生成验证码（推理不需要）
- `src/preprocessing/` - 预处理代码
- `src/utils/` - 工具函数
- `src/checkpoints/` - 模型权重文件

## 📦 最小化PyPI包结构

### 方案A：直接使用src作为包（推荐）

```
sider-captcha-solver/
│
├── sider_captcha_solver/         # 简单的包装层
│   ├── __init__.py               # 从src导入并暴露核心API
│   ├── __version__.py            # 版本信息
│   ├── predictor.py              # 高层封装类（调用src代码）
│   └── cli.py                    # CLI接口
│
├── src/                          # 保持现有结构不变！
│   ├── models/                   # 直接使用现有模型代码
│   │   ├── __init__.py
│   │   ├── lite_hrnet_18_fpn.py # 现有模型架构
│   │   └── ...
│   ├── utils/                    # 直接使用现有工具
│   └── ...                       # 其他现有代码
│
├── setup.py                      # 配置包含src目录
├── pyproject.toml
├── MANIFEST.in
└── README.md
```

### 方案B：仅打包推理所需的最小子集

```
sider-captcha-solver/
│
├── sider_captcha_solver/         
│   ├── __init__.py               
│   ├── __version__.py            
│   ├── predictor.py              # 推理封装
│   ├── cli.py                    
│   │
│   ├── models/                   # 仅复制推理所需的模型文件
│   │   ├── __init__.py          # 从src.models复制
│   │   ├── lite_hrnet_18_fpn.py # 从src.models复制
│   │   └── modules/              # 从src.models.modules复制必要部分
│   │
│   └── utils/                    # 仅复制必要的工具
│       ├── __init__.py
│       └── download.py           # 新增：模型下载
│
├── setup.py
└── ...
```

## 🚀 推荐方案：直接包含src（方案A）

### setup.py 配置

```python
from setuptools import setup, find_packages

setup(
    name="sider-captcha-solver",
    version="1.1.0",
    packages=find_packages() + ['src', 'src.models', 'src.utils'],  # 包含src
    package_dir={
        'sider_captcha_solver': 'sider_captcha_solver',
        'src': 'src',  # 直接映射src目录
    },
    package_data={
        'sider_captcha_solver': ['data/*.json'],
        # 不包含大的pth文件，使用延迟下载
    },
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python>=4.5.0',
        'Pillow>=9.0.0',
        'numpy>=1.20.0',
        'requests>=2.25.0',
        'tqdm>=4.60.0',
    ],
    entry_points={
        'console_scripts': [
            'sider-captcha=sider_captcha_solver.cli:main',
            'captcha-solver=sider_captcha_solver.cli:main',
        ],
    },
)
```

### 核心包装器 (sider_captcha_solver/__init__.py)

```python
"""Sider CAPTCHA Solver - 工业级滑块验证码识别系统"""

import sys
from pathlib import Path

# 添加src到Python路径（如果需要）
package_dir = Path(__file__).parent.parent
src_path = package_dir / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

# 导入现有的模型代码
from src.models import create_lite_hrnet_18_fpn
from src.models.lite_hrnet_18_fpn import LiteHRNet18FPN

# 导入新的高层API
from .predictor import CaptchaPredictor
from .__version__ import __version__

__all__ = [
    'CaptchaPredictor',
    'create_lite_hrnet_18_fpn',  # 暴露原始模型创建函数
    '__version__'
]

# 便捷函数
def predict(image_path, model_path='best', device='auto'):
    """快速预测接口"""
    predictor = CaptchaPredictor(model_path, device)
    return predictor.predict(image_path)
```

### 预测器封装 (sider_captcha_solver/predictor.py)

```python
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import requests
from tqdm import tqdm

# 使用现有的src代码
from src.models import create_lite_hrnet_18_fpn

class CaptchaPredictor:
    """高层预测器API，封装src中的模型"""
    
    MODEL_URLS = {
        'best': 'https://github.com/.../best_model.pth',
        'v1.1.0': 'https://github.com/.../v1.1.0.pth',
    }
    
    def __init__(self, model_path='best', device='auto', hm_threshold=0.1):
        self.device = self._setup_device(device)
        self.threshold = hm_threshold
        
        # 加载模型（使用src中的现有代码）
        self.model = create_lite_hrnet_18_fpn()
        
        # 加载权重
        weights_path = self._get_model_path(model_path)
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image_path: Union[str, Path, np.ndarray]) -> Dict:
        """预测单张图片"""
        # 预处理
        image_tensor = self._preprocess(image_path)
        
        # 推理（使用src.models中的代码）
        with torch.no_grad():
            outputs = self.model(image_tensor)
            predictions = self.model.decode_predictions(outputs, image_tensor)
        
        # 后处理并返回结果
        return self._format_result(predictions)
    
    def _get_model_path(self, model_path):
        """获取模型路径（本地或下载）"""
        if Path(model_path).exists():
            return model_path
        
        # 如果是内置模型名称，下载到缓存
        if model_path in self.MODEL_URLS:
            cache_dir = Path.home() / '.cache' / 'sider_captcha_solver'
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            local_path = cache_dir / f"{model_path}.pth"
            if not local_path.exists():
                self._download_model(self.MODEL_URLS[model_path], local_path)
            
            return local_path
        
        raise ValueError(f"Model not found: {model_path}")
    
    def _download_model(self, url, save_path):
        """下载模型文件"""
        print(f"Downloading model to {save_path}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def _preprocess(self, image):
        """图像预处理"""
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        
        # Resize到模型输入尺寸
        image = cv2.resize(image, (512, 256))
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        
        # 添加padding mask通道（全0）
        padding_mask = torch.zeros(1, 256, 512)
        image_tensor = torch.cat([image_tensor, padding_mask], dim=0)
        
        # 添加batch维度
        return image_tensor.unsqueeze(0).to(self.device)
    
    def _format_result(self, predictions):
        """格式化输出结果"""
        gap_x = predictions['gap_coords'][0, 0].item()
        gap_y = predictions['gap_coords'][0, 1].item()
        slider_x = predictions['slider_coords'][0, 0].item()
        slider_y = predictions['slider_coords'][0, 1].item()
        
        return {
            'success': True,
            'sliding_distance': gap_x - slider_x,
            'gap_x': gap_x,
            'gap_y': gap_y,
            'slider_x': slider_x,
            'slider_y': slider_y,
            'gap_confidence': predictions['gap_score'][0].item(),
            'slider_confidence': predictions['slider_score'][0].item(),
            'confidence': (predictions['gap_score'][0].item() + 
                          predictions['slider_score'][0].item()) / 2
        }
```

## 📊 优势对比

### 方案A优势（推荐）
✅ **零冗余**：完全复用现有src代码
✅ **维护简单**：只需维护一份代码
✅ **开发快速**：仅需添加薄封装层
✅ **功能完整**：用户可访问所有原始功能

### 方案B优势
✅ **包更小**：仅包含推理必需代码
✅ **更干净**：不包含训练相关代码
❌ **需要复制代码**：产生冗余
❌ **维护困难**：需要同步更新两份代码

## 🎯 用户安装后的使用

```python
# 1. 高层API（推荐）
from sider_captcha_solver import CaptchaPredictor
predictor = CaptchaPredictor()
result = predictor.predict('captcha.png')

# 2. 也可以直接访问底层src代码（高级用户）
from src.models import create_lite_hrnet_18_fpn
model = create_lite_hrnet_18_fpn()
```

## 📦 发布时排除不必要的文件

### MANIFEST.in
```
include README.md
include LICENSE
include requirements.txt

# 包含src中的Python文件
recursive-include src *.py

# 排除不需要的文件
recursive-exclude src/checkpoints *.pth
recursive-exclude src/captcha_generator *
recursive-exclude src/training *
recursive-exclude src/data_collection *
recursive-exclude * __pycache__
recursive-exclude * *.pyc
```

## 🚀 实施步骤

1. **创建薄封装层** (1天)
   - 创建 `sider_captcha_solver` 目录
   - 编写 `predictor.py` 高层API
   - 编写 `cli.py` 命令行接口

2. **配置打包** (半天)
   - 创建 `setup.py` 和 `pyproject.toml`
   - 配置 `MANIFEST.in` 排除大文件
   - 测试打包

3. **模型托管** (半天)
   - 上传模型到 GitHub Release
   - 实现自动下载功能
   - 测试下载流程

4. **发布** (半天)
   - 本地测试安装
   - 发布到TestPyPI
   - 发布到PyPI

## ✅ 最终效果

- **包大小**: <1MB（不含模型）
- **首次使用**: 自动下载36MB模型
- **代码复用**: 100%复用现有src代码
- **维护成本**: 最低，只需维护一份代码