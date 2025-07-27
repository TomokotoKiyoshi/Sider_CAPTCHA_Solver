# 测试脚本使用指南

本目录包含了各种测试和验证脚本，用于测试模型性能、可视化效果和数据生成功能。

## 📑 目录

1. [性能测试](#性能测试)
   - [benchmark_inference.py](#benchmark_inferencepy)

2. [可视化测试](#可视化测试)
   - [test_distance_error_visualization.py](#test_distance_error_visualizationpy)
   - [test_darkness_levels.py](#test_darkness_levelspy)
   - [test_slider_effects.py](#test_slider_effectspy)
   - [test_gap_highlighting.py](#test_gap_highlightingpy)

3. [数据生成测试](#数据生成测试)
   - [test_captcha_generation.py](#test_captcha_generationpy)
   - [test_generate_captchas.py](#test_generate_captchaspy)
   - [test_all_puzzle_shapes.py](#test_all_puzzle_shapespy)

4. [抗混淆能力增强特性测试](#抗混淆能力增强特性测试)
   - [test_rotated_gap.py](#test_rotated_gappy)
   - [test_slider_perlin_noise.py](#test_slider_perlin_noisepy)
   - [test_confusing_gap.py](#test_confusing_gappy)

5. [模型测试](#模型测试)
   - [test_model_architecture.py](#test_model_architecturepy)
   - [test_real_captchas.py](#test_real_captchaspy)

6. [数据处理工具](#数据处理工具)
   - [merge_real_captchas.py](#merge_real_captchaspy)

---

## 性能测试

### benchmark_inference.py
**功能**：全面测试模型的推理性能，包括模型加载、图像预处理、推理速度、后处理解码等各环节耗时，并验证预测准确性。

**核心特性**：
- 自动检测并显示硬件信息（CPU/GPU型号）
- 分阶段性能测试（加载→预处理→推理→后处理）
- 预热机制确保测试准确性（GPU缓存预热）
- 自动计算预测准确性（从文件名提取真实坐标）
- 结果保存为JSON格式，包含详细的指标解释

**使用示例**：
```bash
# 基础用法 - GPU测试100次
python tests/benchmark_inference.py

# CPU测试50次
python tests/benchmark_inference.py --device cpu --num_runs 50

# GPU测试1000次，预热20次（更精确的基准测试）
python tests/benchmark_inference.py --device cuda --num_runs 1000 --warmup_runs 20

# 快速测试（10次）
python tests/benchmark_inference.py --num_runs 10 --warmup_runs 5
```

**输出说明**：
- **Model Loading Time**: 模型权重加载时间
- **Mean Inference Time**: 平均推理时间（最重要指标）
- **FPS**: 每秒可处理验证码数量
- **Sliding Distance Error**: 滑动距离预测误差（像素）

**结果文件**：
- `logs/benchmark_results_cpu.json` - CPU测试结果
- `logs/benchmark_results_cuda.json` - GPU测试结果

JSON文件包含完整的性能数据和每个指标的详细解释。

---

## 可视化测试

### test_distance_error_visualization.py
**功能**：可视化展示不同像素误差（0px、3px、4px、5px、6px、7px）对滑块验证码识别的影响，并使用实际模型进行预测对比。

**核心特性**：
- 生成6个子图展示不同误差级别的视觉效果
- 绿色圆圈和实线表示真实位置，红色虚线圆圈表示预测位置
- 底部包含误差放大示意图，直观展示误差大小
- 自动使用最佳模型进行实际预测并生成对比图
- 固定使用特定测试图片确保结果可重现

**使用示例**：
```bash
# 直接运行（使用默认图片）
python tests/test_distance_error_visualization.py
```

**输出文件**：
- `outputs/distance_error_visualization.png` - 6种误差级别的可视化对比图
- `outputs/actual_model_prediction.png` - 模型实际预测结果可视化

**输出信息**：
- 显示选定图片的真实Gap和Slider坐标
- 计算并显示真实滑动距离
- 如果有模型，还会显示实际预测结果和误差

**注意**：脚本固定使用 `Pic0001_Bgx116Bgy80_Sdx32Sdy80_9b469398.png` 作为测试图片，确保结果的可重现性。

### test_darkness_levels.py
**功能**：测试验证码缺口处的不同暗度渲染效果，包括基础暗度、边缘暗度、方向性光照和外边缘阴影的参数调试。

**核心特性**：
- 展示4种预设暗度配置（浅、默认、深、带外边缘）
- 可视化不同暗度参数组合的效果
- 支持自定义参数测试，精细调整缺口渲染效果
- 使用真实拼图形状测试光照效果
- 生成对比图展示参数影响

**使用示例**：
```bash
# 运行完整测试（生成4种配置对比图 + 自定义参数演示）
python tests/test_darkness_levels.py
```

**输出文件**：
- `outputs/darkness_levels_comparison.png` - 4种预设配置的对比图

**参数说明**：
- **base_darkness**: 缺口内部的基础暗度（0-50）
- **edge_darkness**: 缺口边缘的额外暗度，产生深度效果
- **directional_darkness**: 方向性光照强度，模拟光源方向
- **outer_edge_darkness**: 缺口外围的柔和阴影宽度

**注意**：需要先运行 `scripts/download_images.py` 下载测试图片。

### test_slider_effects.py
**功能**：测试滑块的3D光照渲染效果，包括边缘高光、方向性光照、衰减因子等参数，以及滑块框架的合成效果。

**核心特性**：
- 展示6种不同场景的滑块渲染效果
- 对比4种不同强度的光照参数配置
- 演示完整的滑块框架合成过程
- 支持自定义光照参数调试
- 使用灰度拼图更清晰地展示光照效果

**使用示例**：
```bash
# 运行完整测试（生成6格效果图 + 4种参数对比）
python tests/test_slider_effects.py
```

**输出文件**：
- `outputs/slider_lighting_test.png` - 6种场景的滑块效果展示
- `outputs/slider_lighting_comparison.png` - 4种光照强度对比

**光照参数说明**：
- **edge_highlight**: 边缘高光强度（0-100）
- **directional_highlight**: 方向性光照强度，模拟3D效果
- **edge_width**: 高光边缘宽度（像素）
- **decay_factor**: 光照衰减因子，控制过渡柔和度

**注意**：需要先运行 `scripts/download_images.py` 下载测试图片。

### test_gap_highlighting.py
**功能**：测试验证码缺口的高光效果（变亮），与传统的阴影效果（变暗）形成对比，用于增强验证码的视觉复杂度。

**核心特性**：
- 对比展示变暗效果（传统）vs 变亮效果（新增）
- 在不同背景颜色上测试效果（深色、中等、浅色）
- 可调整高光强度参数
- 支持真实拼图形状测试
- 生成并排对比图便于效果评估

**使用示例**：
```bash
# 运行测试生成对比图
python tests/test_gap_highlighting.py
```

**输出文件**：
- `test_output/gap_highlighting/gap_effect_comparison.png` - 暗/亮效果对比图

**参数说明**：
- **base_lightness**: 缺口内部的基础亮度增加（0-60）
- **edge_lightness**: 缺口边缘的额外亮度，产生高光效果
- **directional_lightness**: 方向性光照强度
- **outer_edge_lightness**: 缺口外围的柔和高光宽度

**注意**：此功能已集成到 `generate_captchas.py` 中，30%概率生成高光缺口。

---

## 数据生成测试

### test_captcha_generation.py
**功能**：测试从真实图片生成滑块验证码的完整流程，包括拼图提取、缺口渲染、光照效果和多种拼图形状的对比。

**核心特性**：
- 从真实图片中随机提取拼图块
- 应用缺口凹陷光照效果（保留内容但变暗）
- 为拼图块添加阴影增强立体感
- 测试6种不同的拼图边缘组合
- 自动确保拼图位置合理（避免与滑块重叠）

**使用示例**：
```bash
# 运行完整测试（单张随机 + 6种形状对比）
python tests/test_captcha_generation.py
```

**输出文件**：
- `outputs/captcha_generation_test.png` - 单张图片的验证码生成效果
- `outputs/multiple_shapes_test.png` - 同一图片的6种拼图形状对比

**拼图形状类型**：
- **flat**: 平直边缘
- **convex**: 凸出边缘
- **concave**: 凹陷边缘

**注意**：需要先运行 `scripts/download_images.py` 下载测试图片。

### test_generate_captchas.py
**功能**：测试批量验证码生成脚本的功能，使用小规模数据验证生成流程的正确性。

**核心特性**：
- 从原始图片中选取2张进行测试
- 每张图片生成120个验证码变体（10种形状×3种大小×4个位置）
- 测试标注文件的生成和格式正确性
- 使用单进程模式便于调试
- 自动清理临时测试文件

**使用示例**：
```bash
# 运行小批量测试（2张图片，共240个验证码）
python tests/test_generate_captchas.py
```

**输出目录**：
- `data/test_captchas/` - 测试生成的验证码和标注文件

**生成规则**：
- 10种拼图形状：5种普通形状 + 5种异形形状
- 3种拼图大小：48px、60px、72px
- 4个随机位置：x轴需大于(滑块宽度+10px)

**注意**：需要先运行 `scripts/download_images.py` 下载测试图片。

### test_all_puzzle_shapes.py
**功能**：生成并展示所有87种拼图形状（81种普通组合 + 6种特殊形状），用于验证拼图生成算法的完整性。

**核心特性**：
- 展示81种四边组合形状（3³ = 81种）
- 展示6种特殊形状（圆形、正方形、三角形、六边形、五边形、五角星）
- 使用蓝色拼图配浅灰色背景便于观察
- 生成额外的示例展示图便于理解
- 所有形状带编号或名称标注

**使用示例**：
```bash
# 生成所有形状展示图
python tests/test_all_puzzle_shapes.py
```

**输出文件**：
- `outputs/all_puzzle_shapes_2d.png` - 87种形状的完整展示（10×9网格）
- `outputs/example_puzzle_pieces.png` - 6种典型形状的详细示例

**形状组合规则**：
- 每条边可以是：flat（平）、convex（凸）、concave（凹）
- 顺序：上、右、下、左
- 共计 3×3×3×3 = 81 种组合

**特殊形状**：用于增加验证码的多样性和复杂度。

---

## 抗混淆能力增强特性测试

### test_rotated_gap.py
**功能**：测试验证码缺口的微小旋转效果，生成旋转0.5-2度的缺口，增加机器识别难度。

**核心特性**：
- 生成6张不同旋转角度的测试图片
- 旋转角度：0°、0.5°、1.0°、1.5°、2.0°、-1.0°
- 只旋转缺口，滑块保持原始角度
- 生成2×3网格的预览图，便于对比
- 每张图片标注具体旋转角度

**使用示例**：
```bash
# 生成旋转缺口测试图
python tests/test_rotated_gap.py
```

**输出文件**：
- `test_output/rotated_gaps/rotated_gaps_preview.png` - 6种旋转角度的预览图
- `test_output/rotated_gaps/rotated_gap_*.png` - 单独的测试图片

**技术说明**：
- 使用OpenCV的仿射变换实现旋转
- 自动调整旋转后的边界框
- 确保旋转后的缺口不超出图像边界

**注意**：此功能已集成到 `generate_captchas.py` 中，50%概率生成0.5-1.8度旋转。

### test_slider_perlin_noise.py
**功能**：测试在滑块表面应用柏林噪声的效果，模拟真实世界的纹理干扰。

**核心特性**：
- 生成6个不同噪声强度的测试样本
- 噪声强度：30%、40%、50%、70%、85%、100%
- 使用多层octave生成自然的噪声纹理
- 只在滑块有效区域（alpha>0）应用噪声
- 生成2×3网格的渐变对比图

**使用示例**：
```bash
# 生成柏林噪声渐变测试图
python tests/test_slider_perlin_noise.py
```

**输出文件**：
- `test_output/slider_noise/noise_gradient_preview.png` - 噪声强度渐变图
- `test_output/slider_noise/slider_noise_*.png` - 单独的测试图片

**噪声参数**：
- **scale**: 噪声的缩放因子（默认20）
- **octaves**: 噪声层数（默认3）
- **persistence**: 振幅衰减（默认0.6）
- **base_gray**: 基础灰度值（180-220随机）

**注意**：此功能已集成到 `generate_captchas.py` 中，50%概率应用40-80%强度噪声。

### test_confusing_gap.py
**功能**：测试混淆缺口的生成，在真实缺口附近生成相似但角度不同的干扰缺口。

**核心特性**：
- 生成6张带有混淆缺口的测试图片
- 混淆缺口旋转角度：±10-30度
- 确保混淆缺口与真实缺口至少相隔10像素
- 确保混淆缺口不与滑块重叠
- 使用绿框标记真实缺口，红框标记混淆缺口

**使用示例**：
```bash
# 生成混淆缺口测试图
python tests/test_confusing_gap.py
```

**输出文件**：
- `test_output/confusing_gaps/confusing_gaps_preview.png` - 6个样本的预览图
- `test_output/confusing_gaps/confusing_gap_*.png` - 单独的测试图片

**放置策略**：
- 优先放置在右侧区域（远离滑块）
- 最多尝试100次寻找合适位置
- 如果找不到合适位置则不生成混淆缺口

**注意**：此功能已集成到 `generate_captchas.py` 中，60%概率生成混淆缺口。

---

## 模型测试

### test_model_architecture.py
**功能**：全面验证CaptchaSolver模型架构（ResNet18 Lite + CenterNet）的正确性，包括输入输出形状、参数数量、解码功能和梯度流。

**核心特性**：
- 验证输入输出张量形状是否符合设计规范
- 统计模型参数量（目标：~3.5M参数）
- 测试坐标解码功能的正确性
- 检查梯度反向传播是否正常
- 输出详细的模块参数统计

**使用示例**：
```bash
# 运行完整的架构测试
python tests/test_model_architecture.py
```

**测试项目**：
1. **形状测试**：验证输入(3×160×320) → 输出(4个40×80的特征图)
2. **参数测试**：检查总参数量是否约为3.5M（允许±0.5M误差）
3. **解码测试**：验证坐标解码是否在合理范围(0-320, 0-160)
4. **梯度测试**：确保所有参数都能接收梯度

**输出信息**：
- 各模块参数量统计（Backbone、Neck、Heads）
- 模型大小（MB）
- 各项测试的通过/失败状态

### test_real_captchas.py
**功能**：在真实验证码数据集上测试旧版CaptchaDetector模型的检测性能，并生成可视化结果。

**核心特性**：
- 从真实验证码数据集中随机抽取100张测试
- 统计Gap和Piece的检测率
- 生成5张可视化结果图（每张4×5=20个样本）
- 使用红框标记Gap位置，绿框标记Piece位置
- 输出JSON格式的检测统计数据

**使用示例**：
```bash
# 测试真实验证码（需要先准备merged数据）
python tests/test_real_captchas.py
```

**输出文件**：
- `results/real_captcha_results/site1_detection_results_*.png` - 可视化结果（5张）
- `results/real_captcha_results/site1_detection_stats.json` - 检测统计数据

**检测标记**：
- 红色圆点+红框：检测到的Gap位置
- 绿色圆点+绿框：检测到的Piece位置
- 标题显示：✓（检测成功）或 ✗（检测失败）

**注意**：需要先运行 `merge_real_captchas.py` 准备数据。

---

## 数据处理工具

### merge_real_captchas.py
**功能**：合并真实验证码的背景图（bg）和滑块图（slider）为单张图片，便于模型测试和可视化。

**核心特性**：
- 自动匹配*_bg.png和*_slider.png文件对
- 将滑块正确合成到背景图上（考虑透明通道）
- 滑块随机放置在左侧0-10像素范围内
- 确保输出尺寸统一为320×160像素
- 生成JSON格式的标注文件记录处理信息

**使用示例**：
```bash
# 合并site1文件夹中的所有验证码
python tests/merge_real_captchas.py
```

**输入目录结构**：
```
data/real_captchas/site1/
  ├── captcha_001_bg.png
  ├── captcha_001_slider.png
  ├── captcha_002_bg.png
  └── captcha_002_slider.png
```

**输出文件**：
- `data/real_captchas/merged/site1/*_merged.png` - 合并后的验证码图片
- `data/real_captchas/merged/site1/site1_annotations.json` - 处理记录

**合成规则**：
- 背景图调整为320×160
- 滑块保持原始大小
- 滑块x坐标：0-10px随机
- 滑块y坐标：垂直居中
- 使用alpha通道混合实现透明效果

---

## 🔧 通用说明

1. 所有脚本都应该在项目根目录下运行
2. 确保已安装所有依赖：`pip install -r requirements.txt`
3. 大部分测试脚本会将结果输出到 `outputs/` 目录
4. 性能测试结果会保存到 `logs/` 目录