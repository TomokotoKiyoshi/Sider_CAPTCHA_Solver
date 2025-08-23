# TAMH 逐步架构（详细 I/O）

## Step 1. 提取滑块模板 R_piece

**输入**：
- `Hf = [B,128,64,128]`
- 辅助：已有 `H_piece = [B,1,64,128]` 热图

**操作**：
- argmax 找到滑块中心 `(u_p,v_p)`
- 从 Hf 裁 16×16 patch：
  ```
  R_piece = Crop(Hf, center=(u_p,v_p), size=16×16)
  ```

**输出**：`R_piece = [B,128,16,16]`

## Step 2. 模板向量/卷积核生成

**输入**：`R_piece = [B,128,16,16]`

**动态卷积**：
```
g = GAP(R_piece)                              # [B,128,1,1] → 展平 [B,128]
W = FC(g, out=128*3*3)                        # [B,1152]
W = reshape(W, [B,128,3,3])                   # 每通道一张 3×3 核
W = L2_norm_per_channel(W)                    # 稳定对比尺度
```

**输出**：`W = [B,128,3,3]`

## Step 3. Cross-Correlation 相关搜索

**输入**：
- 模板 `W = [B,128,3,3]`
- 整图 `Hf = [B,128,64,128]`

**操作**：
2D 相关：
```
H_corr = Conv(Hf, weight=W, groups=128)
```
（等价于 depthwise conv，每个通道做相关）

**输出**：`H_corr = [B,1,64,128]` （相关热图）

## Step 4. 融合最终 gap 热图

**输入**：
- `H_gap = [B,1,64,128]`（卷积预测的 gap 热图）
- `H_corr = [B,1,64,128]`

**操作**：
- 拼接：[B,2,64,128]
- 卷积融合：
  ```
  Fused = Conv3×3(2→1, BN, SiLU)
  ```

**输出**：`H_gap_final = [B,1,64,128]`

## 最终输出

`H_gap_final`：比单纯卷积预测的更稳健 → 提供缺口坐标监督
