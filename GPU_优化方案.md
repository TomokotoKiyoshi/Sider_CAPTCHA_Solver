# GPU利用率优化方案 - CAPTCHA Solver

## 问题描述
- **当前状况**：RTX 5090/4090 GPU只跑到300W（约52%利用率）
- **峰值功耗**：575W
- **目标**：提升到450-500W（80-90%利用率）

---

## 🚨 关键问题诊断

### 1. 混合精度配置错误【严重】
**问题定位**：
- 配置文件：`amp: true`（布尔值）
- 代码期望：`amp: 'bf16'`（字符串）
- 文件位置：`training_engine.py` L71-80

**影响**：
- 损失20-40%性能
- Tensor Core未被利用
- 实际运行在FP32模式

### 2. torch.compile未启用【严重】
**问题定位**：
- 配置存在但未调用
- `compile_model: false`
- `compile_mode: "default"`

**影响**：
- 错失10-30%性能提升
- PyTorch 2.0优化未生效

### 3. 硬件优化未应用【高】
**问题定位**：
- `apply_hardware_optimizations()`从未被调用
- TF32/cuDNN优化可能未生效

### 4. 模型和批量大小问题【高】
**现状**：
- 模型：3.5M参数，0.45G FLOPs
- batch_size：256
- 输入：320×160像素
- num_workers：28（过多）

**影响**：
- GPU计算量不足
- CPU瓶颈
- 内存带宽浪费

### 5. 训练循环频繁GPU同步【中】
**问题**：
- 每个batch都调用`.item()`
- 不必要的CPU-GPU同步

---

## ✅ 立即实施方案

### 步骤1：修复混合精度（5分钟）

修改 `src/training/training_engine.py` L71：
```python
# 原代码
self.use_amp = config['train'].get('amp', 'none') == 'bf16'

# 修改为
amp_config = config['train'].get('amp', 'none')
if amp_config == True or amp_config == 'true':
    self.use_amp = True
    self.amp_dtype = torch.bfloat16  # RTX 40系列使用BF16
elif amp_config == 'bf16':
    self.use_amp = True
    self.amp_dtype = torch.bfloat16
else:
    self.use_amp = False
```

### 步骤2：启用torch.compile（5分钟）

在 `scripts/training/train.py` 创建模型后添加：
```python
# 在 L401 后添加
if config['train'].get('compile_model', False):
    model = torch.compile(model, mode=config['train']['compile_mode'])
    logging.info(f"模型已编译，模式：{config['train']['compile_mode']}")
```

### 步骤3：应用硬件优化（2分钟）

在 `scripts/training/train.py` L301后添加：
```python
# 应用硬件优化
config_manager.apply_hardware_optimizations()
logging.info("硬件优化已应用")
```

### 步骤4：优化配置文件（2分钟）

修改 `config/training_config.yaml`：
```yaml
train:
  batch_size: 512          # 从256增加到512，后续可增到1024
  amp: true                # 保持不变
  channels_last: true      # 保持不变
  num_workers: 8           # 从28减少到8
  pin_memory: true         # 保持不变
  non_blocking: true       # 保持不变
  gradient_accumulation_steps: 2  # 从1增加到2，有效batch=1024
  compile_model: true      # 从false改为true
  compile_mode: "max-autotune"  # 从default改为max-autotune
  prefetch_factor: 4       # 添加此项

optimizer:
  lr: 6.0e-4              # 从3.0e-4增加到6.0e-4（因为batch变大）
```

### 步骤5：减少GPU同步（10分钟）

修改 `src/training/training_engine.py`：
```python
# 修改 L366-373，只在需要记录时调用.item()
if batch_idx % self.log_interval == 0:
    # 记录损失（转换为标量）
    loss_val = loss.item()
    gap_error_val = gap_error.item()
    slider_error_val = slider_error.item()
    self.batch_metrics['loss'].append(loss_val)
    # ... 其他记录
else:
    # 不记录时不调用.item()
    pass
```

### 步骤6：使用fused优化器（2分钟）

修改 `src/training/training_engine.py` L158：
```python
# 添加 fused=True 参数
optimizer = AdamW(
    self.model.parameters(),
    lr=lr,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay,
    fused=True  # 添加此行
)
```

---

## 📊 预期效果

| 优化项 | 性能提升 | 功耗变化 | 优先级 |
|-------|---------|---------|--------|
| 修复混合精度 | +20-40% | 300W→380W | 最高 |
| torch.compile | +10-30% | +50W | 高 |
| 批量大小×2-4 | +30-50% | +80W | 高 |
| 减少同步 | +5-10% | +20W | 中 |
| fused优化器 | +3-5% | +10W | 低 |
| **总计** | **+68-135%** | **300W→450-500W** | - |

---

## 🔧 监控脚本

创建 `scripts/monitor_gpu.py`：
```python
#!/usr/bin/env python3
"""GPU性能监控脚本"""

import nvidia_ml_py as nvml
import time
import sys

def monitor_gpu():
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    
    print("GPU监控中... (Ctrl+C退出)")
    print("-" * 60)
    
    try:
        while True:
            # 获取功耗
            power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            
            # 获取利用率
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            
            # 获取显存
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_gb = mem.used / 1024 / 1024 / 1024
            mem_total_gb = mem.total / 1024 / 1024 / 1024
            
            # 获取温度
            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            
            # 获取时钟频率
            sm_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
            mem_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
            
            # 打印信息
            sys.stdout.write(f"\r功耗: {power:.1f}W/{power_limit:.0f}W ({power/power_limit*100:.1f}%) | "
                           f"利用率: {util.gpu}% | "
                           f"显存: {mem_used_gb:.1f}GB/{mem_total_gb:.1f}GB | "
                           f"温度: {temp}°C | "
                           f"核心: {sm_clock}MHz | "
                           f"显存: {mem_clock}MHz")
            sys.stdout.flush()
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n监控已停止")
        nvml.nvmlShutdown()

if __name__ == "__main__":
    monitor_gpu()
```

---

## 🚀 进阶优化方案（如基础优化不够）

### 1. 使用更大的模型
```python
# 考虑使用 HRNet-W32 或 HRNet-W48
# 参数量：~30M 或 ~65M
# FLOPs：~7G 或 ~15G
```

### 2. 多GPU训练（DDP）
```python
# 使用 torch.distributed 进行分布式训练
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### 3. NVIDIA DALI数据管道
```python
# 使用DALI替代PyTorch DataLoader
# GPU端数据预处理，减少CPU瓶颈
pip install nvidia-dali-cuda110
```

### 4. 增加输入分辨率
```yaml
# 考虑使用更大的输入
input_size: [512, 256]  # 从 [320, 160] 增加
```

---

## ⚡ 实施顺序建议

### 第一阶段（立即，30分钟）
1. ✅ 修复混合精度配置
2. ✅ 启用torch.compile
3. ✅ 调整batch_size和num_workers
4. ✅ 应用硬件优化

### 第二阶段（测试后）
5. 📈 监控GPU性能，逐步增加batch_size
6. 📈 如果显存允许，继续增加到1024或2048
7. 📈 调整gradient_accumulation_steps

### 第三阶段（可选）
8. 🔧 实施进阶优化方案
9. 🔧 考虑模型架构调整

---

## 📝 测试验证

### 运行前基准测试
```bash
# 记录当前性能
python scripts/monitor_gpu.py &
python scripts/training/train.py --epochs 1
```

### 应用优化后测试
```bash
# 应用所有优化后
python scripts/monitor_gpu.py &
python scripts/training/train.py --epochs 1
```

### 预期结果
- **优化前**：~300W，52% GPU利用率，~100 samples/s
- **优化后**：450-500W，80-90% GPU利用率，~200-250 samples/s

---

## ⚠️ 注意事项

1. **逐步增加batch_size**：避免OOM错误
2. **监控温度**：确保GPU温度在安全范围内（<85°C）
3. **验证精度**：确保优化不影响模型收敛
4. **保存配置备份**：在修改前备份原配置文件

---

## 📞 故障排除

### 如果OOM（显存不足）
- 减小batch_size
- 增加gradient_accumulation_steps
- 使用gradient checkpointing

### 如果torch.compile报错
- 设置`compile_model: false`
- 检查PyTorch版本（需要>=2.0）
- 尝试不同的compile_mode

### 如果性能提升不明显
- 确认混合精度是否真正启用
- 检查是否有其他瓶颈（如磁盘IO）
- 考虑使用更大的模型

---

*生成时间：2025-01-16*
*针对项目：Sider_CAPTCHA_Solver v1.1.0*