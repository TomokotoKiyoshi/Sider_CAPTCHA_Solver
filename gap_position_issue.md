# Gap坐标随机性问题分析与解决方案

## 问题描述

当前验证码生成系统中，每张原图（Pic）生成的gap位置是完全随机的，导致以下问题：

1. **位置分布不可控**：同一张图片生成的100个验证码，gap位置分布过于随机
2. **测试集不稳定**：无法控制测试集的位置分布，难以评估模型对特定位置的预测能力
3. **数据重现性差**：相同的原图每次生成的验证码位置都不同

## 当前实现分析

### 位置生成函数
文件：`scripts/data_generation/generate_captchas.py`

```python
def generate_continuous_positions(puzzle_size: int, config: DatasetConfig):
    """生成连续的位置（当前为完全随机）"""
    half_size = puzzle_size // 2
    
    # 滑块x坐标：在 [half_size, half_size+10] 范围内随机
    slider_x = np.random.randint(half_size, half_size + 11)
    
    # 滑块y坐标：在 [30, 130] 范围内随机（考虑边界）
    slider_y = np.random.randint(
        max(30, half_size), 
        min(130, 160 - half_size) + 1
    )
    
    # gap x坐标：完全随机（避开滑块）
    gap_x_min = slider_x + half_size + 10
    gap_x_max = 320 - half_size
    gap_x = np.random.randint(gap_x_min, gap_x_max + 1)
    
    # gap y坐标：与滑块相同
    gap_y = slider_y
    
    return (slider_x, slider_y), (gap_x, gap_y)
```

### 问题根源

在生成循环中（第226-232行），每个样本都重新调用位置生成函数：

```python
for sample_idx in range(num_samples):
    # 每次都生成新的随机位置
    slider_pos, gap_pos = generate_continuous_positions(size, dataset_config)
```

## 期望的行为

### 需求
1. 每张原图的gap位置应该是**确定性的网格分布**
2. X轴：4个固定位置
3. Y轴：3个固定位置
4. 总共：4×3 = 12个固定位置组合
5. 原图之间可以随机，但单张图片的验证码位置需要固定

### 优势
- 位置分布可控
- 便于测试模型对不同位置的泛化能力
- 可以分析模型在特定位置的表现
- 提高数据的可重现性

## 建议的解决方案

### 方案1：固定网格位置

```python
def generate_fixed_grid_positions(puzzle_size: int, pic_index: int, sample_idx: int):
    """
    为每张图片生成固定的位置网格
    pic_index: 图片索引（用于确定该图的固定位置集）
    sample_idx: 样本索引（0-11，对应12个位置组合）
    """
    half_size = puzzle_size // 2
    
    # 滑块位置固定
    slider_x = half_size + 5  # 固定在中间位置
    
    # Gap的X轴4个固定位置（均匀分布）
    gap_x_positions = [80, 140, 200, 260]
    
    # Gap的Y轴3个固定位置
    gap_y_positions = [50, 80, 110]
    
    # 根据sample_idx选择位置（12个组合）
    x_idx = sample_idx % 4
    y_idx = (sample_idx // 4) % 3
    
    gap_x = gap_x_positions[x_idx]
    gap_y = gap_y_positions[y_idx]
    slider_y = gap_y  # 保持一致
    
    return (slider_x, slider_y), (gap_x, gap_y)
```

### 方案2：伪随机位置（确定性随机）

```python
def generate_pseudo_random_positions(puzzle_size: int, pic_index: int, num_positions: int = 100):
    """
    使用图片索引作为种子生成确定性的随机位置
    """
    # 使用图片索引作为随机种子
    rng = np.random.RandomState(pic_index)
    
    positions = []
    for i in range(num_positions):
        # 使用确定性的随机数生成器
        slider_x = rng.randint(half_size, half_size + 11)
        slider_y = rng.randint(30, 130)
        
        gap_x = rng.randint(gap_x_min, gap_x_max + 1)
        gap_y = slider_y
        
        positions.append(((slider_x, slider_y), (gap_x, gap_y)))
    
    return positions
```

### 方案3：混合方案（推荐）

结合固定网格和伪随机，提供更好的覆盖：

```python
def generate_hybrid_positions(puzzle_size: int, pic_index: int, sample_idx: int, total_samples: int):
    """
    混合方案：前12个使用固定网格，之后使用伪随机
    """
    if sample_idx < 12:
        # 前12个样本使用固定网格
        return generate_fixed_grid_positions(puzzle_size, pic_index, sample_idx)
    else:
        # 之后的样本使用伪随机（带种子）
        rng = np.random.RandomState(pic_index * 1000 + sample_idx)
        # ... 生成伪随机位置
```

## 实施步骤

1. **修改位置生成函数**
   - 添加 `pic_index` 参数
   - 实现固定网格或伪随机逻辑

2. **更新调用代码**
   - 在 `generate_samples_for_background` 函数中传递 `pic_index`
   - 确保位置生成的确定性

3. **测试验证**
   - 验证同一张图片多次生成的位置一致性
   - 检查位置分布的合理性

4. **文档更新**
   - 更新 CLAUDE.md 中的位置生成规则说明
   - 添加位置分布的可视化

## 影响评估

### 正面影响
- ✅ 提高数据集的可重现性
- ✅ 便于模型性能分析
- ✅ 位置分布更均匀
- ✅ 测试集更稳定

### 潜在风险
- ⚠️ 需要重新生成整个数据集
- ⚠️ 可能影响已训练模型的兼容性
- ⚠️ 固定位置可能降低数据多样性

## 结论

建议采用**混合方案**，既保证基础位置的覆盖（通过固定网格），又保持一定的随机性（通过伪随机）。这样可以在可控性和多样性之间取得平衡。

---
*创建时间：2024-12-23*
*状态：待实施*