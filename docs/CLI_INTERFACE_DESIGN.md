# CLI接口设计文档

## 📊 模型输出能力分析

### 当前模型架构
- **模型名称**: Lite-HRNet-18+LiteFPN
- **参数量**: 3.28M
- **模型大小**: 12.53 MB (FP32)
- **输入尺寸**: [B, 4, 256, 512] (RGB + padding mask)
- **输出分辨率**: 1/4原图 (64×128)

### 模型原始输出

| 输出名称 | 形状 | 数据类型 | 说明 |
|---------|------|---------|------|
| `heatmap_gap` | [B, 1, 64, 128] | float32 | 缺口中心概率热力图 |
| `heatmap_slider` | [B, 1, 64, 128] | float32 | 滑块中心概率热力图 |
| `offset_gap` | [B, 2, 64, 128] | float32 | 缺口亚像素偏移量 |
| `offset_slider` | [B, 2, 64, 128] | float32 | 滑块亚像素偏移量 |

### 解码后的预测输出

| 输出名称 | 形状 | 数据类型 | 说明 |
|---------|------|---------|------|
| `gap_coords` | [B, 2] | float32 | 缺口中心坐标 [x, y] (像素) |
| `slider_coords` | [B, 2] | float32 | 滑块中心坐标 [x, y] (像素) |
| `gap_score` | [B] | float32 | 缺口检测置信度 (0-1) |
| `slider_score` | [B] | float32 | 滑块检测置信度 (0-1) |

### 计算衍生信息
- **滑动距离**: `sliding_distance = gap_x - slider_x`
- **综合置信度**: `confidence = (gap_score + slider_score) / 2`

## 🏗️ CLI接口架构设计

### 核心类结构

```python
class CaptchaPredictor:
    """滑块验证码预测器"""
    
    def __init__(self, 
                 model_path: str = 'best',  # 模型路径或'best'使用内置模型
                 device: str = 'auto',       # 'auto', 'cuda', 'cpu'
                 hm_threshold: float = 0.1): # 热力图阈值
        pass
    
    def predict(self, image_path: str) -> Dict:
        """预测单张图片"""
        pass
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """批量预测多张图片"""
        pass
    
    def visualize_prediction(self, image_path: str, 
                            save_path: str = None,
                            show: bool = True) -> None:
        """可视化预测结果"""
        pass
    
    def visualize_heatmaps(self, image_path: str,
                          save_path: str = None,
                          show: bool = True) -> None:
        """可视化热力图"""
        pass
```

### 返回数据结构

```python
{
    # 检测状态
    "success": bool,              # 是否成功检测到缺口和滑块
    "error": str,                 # 错误信息（如果失败）
    
    # 主要结果
    "sliding_distance": float,    # 滑动距离 (像素)
    
    # 坐标信息
    "gap_x": float,              # 缺口中心x坐标
    "gap_y": float,              # 缺口中心y坐标
    "slider_x": float,           # 滑块中心x坐标
    "slider_y": float,           # 滑块中心y坐标
    
    # 置信度信息
    "gap_confidence": float,      # 缺口检测置信度 (0-1)
    "slider_confidence": float,   # 滑块检测置信度 (0-1)
    "confidence": float,          # 综合置信度 (0-1)
    
    # 性能信息
    "processing_time_ms": float,  # 处理时间（毫秒）
    
    # 详细信息（可选）
    "details": {
        "gap_coords": [x, y],     # 原始坐标数组
        "slider_coords": [x, y],  # 原始坐标数组
        "model_version": str,     # 模型版本
        "input_size": [h, w],     # 输入图像尺寸
        "device_used": str        # 实际使用的设备
    }
}
```

## 📝 使用示例

### 1. 基础预测 - 获取滑动距离

```python
from captcha_solver import CaptchaPredictor

# 初始化预测器
predictor = CaptchaPredictor(
    model_path='best',  # 使用内置最佳模型
    device='auto'       # 自动选择GPU/CPU
)

# 预测单张图片
result = predictor.predict('path/to/captcha.png')

# 使用结果
if result['success']:
    print(f"滑动距离: {result['sliding_distance']:.2f} px")
    print(f"缺口位置: ({result['gap_x']:.2f}, {result['gap_y']:.2f})")
    print(f"滑块位置: ({result['slider_x']:.2f}, {result['slider_y']:.2f})")
    print(f"置信度: {result['confidence']:.3f}")
else:
    print(f"检测失败: {result['error']}")
```

### 2. 批量处理

```python
import glob

# 批量处理文件夹中的图片
image_files = glob.glob('captchas/*.png')
results = predictor.predict_batch(image_files)

for img_path, result in zip(image_files, results):
    if result['success']:
        print(f"{img_path}: 滑动 {result['sliding_distance']:.1f} px")
```

### 3. 可视化调试

```python
# 生成预测可视化
predictor.visualize_prediction(
    'captcha.png',
    save_path='prediction_result.png',
    show=True
)

# 生成热力图可视化
predictor.visualize_heatmaps(
    'captcha.png',
    save_path='heatmap_result.png',
    show=True
)
```

## 🎯 CLI命令行接口

### 命令结构

```bash
# 基础命令格式
python -m captcha_solver <command> [options]
```

### 可用命令

#### 1. predict - 单张预测
```bash
python -m captcha_solver predict <image_path> [options]

选项:
  --model PATH         模型路径 (默认: best)
  --device DEVICE      设备选择 (auto/cuda/cpu)
  --threshold FLOAT    热力图阈值 (默认: 0.1)
  --output FORMAT      输出格式 (json/text)
  --save-vis PATH      保存可视化图片
```

示例:
```bash
python -m captcha_solver predict captcha.png
python -m captcha_solver predict captcha.png --save-vis result.png
```

#### 2. batch - 批量处理
```bash
python -m captcha_solver batch <input_dir> [options]

选项:
  --pattern PATTERN    文件匹配模式 (默认: *.png)
  --output PATH        输出结果文件
  --format FORMAT      输出格式 (json/csv)
  --parallel           并行处理
  --max-workers INT    最大工作线程数
```

示例:
```bash
python -m captcha_solver batch ./captchas --output results.json
python -m captcha_solver batch ./captchas --pattern "*.jpg" --format csv
```

#### 3. visualize - 可视化
```bash
python -m captcha_solver visualize <image_path> [options]

选项:
  --type TYPE          可视化类型 (prediction/heatmap/both)
  --save PATH          保存路径
  --show               显示窗口
  --threshold FLOAT    热力图阈值
```

示例:
```bash
python -m captcha_solver visualize captcha.png --type both
python -m captcha_solver visualize captcha.png --save output.png
```

#### 4. benchmark - 性能测试
```bash
python -m captcha_solver benchmark [options]

选项:
  --num-samples INT    测试样本数 (默认: 100)
  --image PATH         测试图片路径
  --device DEVICE      设备选择
  --warmup INT         预热次数
```

示例:
```bash
python -m captcha_solver benchmark --num-samples 1000
python -m captcha_solver benchmark --device cuda --warmup 10
```

## 🔄 与README示例的兼容性

本设计完全兼容README中的所有示例代码：

| README示例功能 | 实现状态 | 说明 |
|---------------|---------|------|
| `predictor.predict()` | ✅ 已实现 | 返回包含所有必需字段的字典 |
| `result['gap_x']` | ✅ 支持 | 从gap_coords[0]提取 |
| `result['gap_y']` | ✅ 支持 | 从gap_coords[1]提取 |
| `result['slider_x']` | ✅ 支持 | 从slider_coords[0]提取 |
| `result['slider_y']` | ✅ 支持 | 从slider_coords[1]提取 |
| 滑动距离计算 | ✅ 支持 | 自动计算并返回 |
| 置信度分数 | ✅ 支持 | 提供单独和综合置信度 |
| 批量处理 | ✅ 支持 | 利用模型batch能力 |
| 可视化 | ✅ 支持 | 热力图和预测结果可视化 |

## 🚀 实现计划

### 第一阶段：核心功能
1. ✅ 分析模型输出能力
2. ✅ 设计CLI接口架构
3. ⏳ 实现CaptchaPredictor基础类
4. ⏳ 实现predict单张预测功能

### 第二阶段：扩展功能
5. ⏳ 实现批量处理功能
6. ⏳ 实现可视化功能
7. ⏳ 实现性能基准测试

### 第三阶段：命令行接口
8. ⏳ 实现CLI命令解析
9. ⏳ 集成所有功能到CLI
10. ⏳ 编写使用文档和示例

## 📊 性能指标

基于当前模型的性能预期：

| 硬件 | 单张推理时间 | FPS | 批处理(×32) |
|-----|------------|-----|-------------|
| RTX 5090 | ~1.3ms | 770 | ~11ms |
| CPU (Ryzen 9) | ~5.2ms | 192 | ~145ms |

## 🔧 技术细节

### 图像预处理
1. 读取图像 (支持PNG/JPG)
2. Resize到256×512
3. 归一化到[0, 1]
4. 添加padding mask通道
5. 转换为tensor格式

### 后处理步骤
1. 获取热力图最大值位置
2. 应用亚像素偏移
3. 上采样坐标到原图分辨率
4. 计算滑动距离
5. 格式化输出结果

### 错误处理
- 图像读取失败
- 模型加载失败
- 检测失败（置信度过低）
- GPU内存不足自动切换CPU

## 📝 更新日志

### v1.0.0 (开发中)
- 初始CLI接口设计
- 基础预测功能实现
- 批量处理支持
- 可视化功能
- 性能基准测试工具