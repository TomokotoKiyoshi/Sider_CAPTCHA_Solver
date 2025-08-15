# GPU优化分析报告

## 问题描述
RTX 5090 GPU在训练时功耗仅100W左右，远低于575W峰值功耗，GPU利用率低。

## 分析时间
2025-01-15

## 发现的问题

### 1. 训练时重复计算已预处理的目标数据

**位置**: `src/training/training_engine.py:266-275`

**问题描述**: 
- 数据集中已经预计算并保存了热力图(heatmap)和偏移量(offset)
- 训练循环中每个step都重新生成这些目标，使用Python嵌套循环
- `_generate_targets()`函数进行逐像素计算，产生大量小kernel调用

**影响**: Python循环成为主要瓶颈，GPU大部分时间在等待CPU计算完成

### 2. DataLoader硬编码配置，忽略配置文件

**位置**: `src/training/npy_data_loader.py:281-289, 313-321, 337-345, 365-373`

**问题描述**:
- DataLoader硬编码`num_workers=2`，`batch_size=1`
- 配置文件中设置的`num_workers=24`被忽略
- 没有设置`prefetch_factor`参数

**影响**: 数据加载并行度低，I/O成为瓶颈

### 3. CuDNN自动调优被禁用

**位置**: `scripts/training/train.py:79-80`

**问题描述**:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
- 配置文件中`cudnn_benchmark: true`被代码覆盖
- 强制使用确定性算法，禁用了kernel自动选择

**影响**: 无法选择最优kernel，卷积操作性能降低

### 4. 缺少梯度累积机制

**位置**: `src/training/training_engine.py:419-433`

**问题描述**:
- 每个批次立即执行`optimizer.step()`
- 由于使用预处理的NPY批次，无法直接增大batch_size
- 没有梯度累积来模拟更大的有效批次

**影响**: 有效批次大小受限于预处理时的设置(256)

### 5. 数据传输未使用异步模式

**位置**: `src/training/training_engine.py:201-206`

**问题描述**:
- 虽然启用了`pin_memory=True`
- 但`.to(device)`调用没有使用`non_blocking=True`
- H2D传输是同步的

**影响**: 数据传输和计算无法重叠，增加了每次迭代的等待时间

### 6. 模型规模过小

**事实数据**:
- 模型参数: 3.5M
- 内存占用: ~14MB
- FLOPs: 0.45G
- RTX 5090能力: 1321 TFLOPS, 77GB VRAM

**问题**: 模型计算量对于RTX 5090来说微不足道

### 7. 未启用TF32加速

**问题描述**:
- 没有设置`torch.backends.cuda.matmul.allow_tf32 = True`
- 没有设置`torch.backends.cudnn.allow_tf32 = True`
- 虽然使用了BF16，但部分FP32操作未优化

### 8. 未使用torch.compile

**问题描述**:
- 没有使用PyTorch 2.0+的编译优化功能
- 小模型的kernel启动开销占比高
- 没有使用CUDA Graphs减少启动开销

## 配置与实际不符

| 配置项 | 配置文件值 | 实际使用值 |
|-------|----------|----------|
| num_workers | 24 | 2 |
| cudnn_benchmark | true | false |
| batch_size (DataLoader) | N/A | 1 (硬编码) |

## 资源使用情况

| 指标 | 当前值 | GPU能力 | 利用率 |
|-----|-------|--------|-------|
| 功耗 | ~100W | 575W | 17% |
| 显存 | <5GB | 77GB | 6% |
| 计算 | 0.45 GFLOPs | 1321 TFLOPs | <0.01% |

## 建议优化措施

### 立即可实施（低难度）

1. **删除重复的目标生成**
   - 使用数据集中预计算的标签
   - 删除`_generate_targets()`函数调用

2. **修复DataLoader配置**
   - 从配置文件读取`num_workers`
   - 添加`prefetch_factor=4`

3. **启用性能优化**
   ```python
   torch.backends.cudnn.benchmark = True
   torch.backends.cudnn.deterministic = False
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```

### 中期实施

4. **实现梯度累积**
   - 添加`gradient_accumulation_steps`配置项
   - 修改训练循环支持累积

5. **启用异步传输**
   - 所有`.to(device)`添加`non_blocking=True`

6. **添加torch.compile**
   ```python
   model = torch.compile(model, mode="max-autotune")
   ```

### 长期考虑

7. **模型架构升级**
   - 考虑使用更大的模型变体
   - 或实现模型集成

## 预期效果

如果实施上述优化，预计：
- GPU功耗: 100W → 400-500W
- 显存使用: 5GB → 50-60GB  
- 训练速度: 提升5-10倍