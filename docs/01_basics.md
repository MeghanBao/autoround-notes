# 01 · 基础概念

> 读后续文档之前需要掌握的概念。有 ML 基础可跳过。

---

## 目录

- [1. 为什么需要量化](#1-为什么需要量化)
- [2. RTN：最简单的量化](#2-rtn最简单的量化)
- [3. Forward 和 Backward](#3-forward-和-backward)
- [4. 梯度下降](#4-梯度下降)
- [5. Zero-Point 零点](#5-zero-point-零点)
- [6. Fake Quantization 假量化](#6-fake-quantization-假量化)
- [7. imatrix 重要性矩阵](#7-imatrix-重要性矩阵)
- [8. Transformer Block 结构](#8-transformer-block-结构)
- [名词速查表](#名词速查表)

---

## 1. 为什么需要量化

一个 7B 参数的 LLM，用 FP16（每个数占 2 字节）存储：

```
7,000,000,000 × 2 字节 = 14 GB
```

普通消费级显卡装不下，量化把每个数从 16 bit 压缩到 4 bit：

```
14 GB → 3.5 GB  ✅ RTX 3080 能跑了
```

| 格式 | 大小（7B 模型）| 可用硬件 |
|------|-------------|---------|
| FP16（原始）| 14 GB | A100 / H100 |
| INT8 | 7 GB | 24GB 显卡 |
| **INT4** | **3.5 GB** | **RTX 3080 / 4070** |
| INT2 | 1.8 GB | 普通消费显卡 |

---

## 2. RTN：最简单的量化

RTN = Round-To-Nearest，就是**四舍五入**。

```python
# 把浮点权重映射到 4-bit 整数（共 16 个格子）
scale = (max(W) - min(W)) / 15   # 格子间距
W_int = round(W / scale)          # 四舍五入到最近格子
W_dq  = W_int × scale             # 反量化（还原为浮点）
```

**RTN 的问题：** 每个权重独立取整，不考虑整体效果。

```
例：3 个权重都是 0.74，scale = 0.5
  RTN：全部取整到 1（即 0.5）
  总误差：3 × (0.74 - 0.5) = 0.72

  如果其中一个取整到 2（即 1.0）：
  总误差：2×(0.74-0.5) + 1×(0.74-1.0) = 0.22  ← 更小！
```

AutoRound 的核心想法：**允许某些权重不走最近的格子，只要整体输出误差更小。**

---

## 3. Forward 和 Backward

神经网络每次训练包含两个阶段：

```
Forward（前向）= 「做题」
  输入 x → 经过网络各层 → 得到输出 y → 算误差 loss

Backward（反向）= 「对答案，找错在哪」
  从 loss 出发 → 自动计算每个参数的梯度
  梯度 = 「这个参数增大一点，loss 会怎么变」
```

**在 AutoRound 里：**

```python
# Forward：用量化权重算输出
y_quant = block(x)                     # WrapperLinear 自动用 fake-quant

# 算误差
loss = MSE(y_quant, y_original)        # 量化输出 vs 浮点参考输出

# Backward：算梯度
loss.backward()                        # 自动算 ∂loss/∂V, ∂loss/∂α, ∂loss/∂β
```

**STE（Straight-Through Estimator）：** `round()` 函数不可微（导数=0），STE 的解决方案是前向用 `round()`，反向假装没有 `round()`，梯度直接穿透。

---

## 4. 梯度下降

梯度下降 = 沿着「误差下降最快的方向」走一小步，反复执行，找到误差最小的点。

```
类比：闭眼在山上找山谷
  感受脚下哪边更陡 → 往那边走一小步 → 重复 → 到达山谷
```

**AutoRound 用 SignSGD（符号梯度下降）：**

```python
# 普通梯度下降：步长和梯度大小有关
V = V - lr × 0.0032      # 梯度很小，步很小

# SignSGD：只用梯度正负号，步长固定
V = V - lr × sign(grad)  # sign(0.0032) = +1，步长固定 = lr
```

为什么用 SignSGD？$V \in [-0.5, 0.5]$，空间很小，固定步长 200 步就能覆盖整个范围。

**学习率线性衰减：**

```
第  1 步：lr = 5e-3       ← 大步探索
第100步：lr = 2.5e-3     ← 中步
第200步：lr → 0          ← 小步精调

平均步长 ≈ 0.0025
总位移 ≈ 200 × 0.0025 = 0.5 = V 的半径
```

---

## 5. Zero-Point 零点

量化时把浮点数映射到整数格子，zero-point 是「原始浮点的 0 对应哪个整数格子」。

```
对称量化（sym=True）：
  权重范围 [-1.0, +1.0]
  整数格子 [-8, -7, ..., 0, ..., 7]
  零点 zp = 0（0 在格子正中间）

非对称量化（sym=False）：
  权重范围 [-0.3, +1.5]（不对称）
  整数格子 [0, 1, 2, ..., 15]
  零点 zp = 2（原始 0.0 落在格子 2 的位置）

公式：
  W_int = round(W / scale + zero_point)
  W_dq  = (W_int - zero_point) × scale
```

非对称量化精度更好（格子更贴合实际分布），但推理时需要处理 zp 偏移。

---

## 6. Fake Quantization 假量化

量化时需要梯度来更新 $V$，但整数不可微。解决方案：用浮点模拟量化效果。

```python
# 真正量化（推理时）：
W_int = round(W / scale)      # 整数，不可微！

# Fake Quantization（训练时）：
W_dq  = round(W / scale) × scale  # 浮点，包含量化误差，可微
```

WrapperLinear 就是实现 Fake Quantization 的组件：
- `forward()` 用含误差的浮点权重，误差体现在输出里
- `backward()` 梯度通过 STE 穿透 `round()`，到达 V/α/β

---

## 7. imatrix 重要性矩阵

不同的输入维度对输出影响不同：

```python
y = W₁×x₁ + W₂×x₂
# x₁ = 100  → W₁ 的量化误差被放大 100 倍 → 影响很大！
# x₂ = 0.001 → W₂ 的误差几乎没影响
```

imatrix 记录每个输入维度的「重要程度」：

```python
# 收集方式：对校准数据的激活做平方和
imatrix[j] = Σ_样本 Σ_token  x_j²

# 使用：scale 搜索时，重要维度的误差权重更大
weighted_error = (imatrix × quant_error²).sum()
```

imatrix 越大 → 这个维度越重要 → scale 搜索时越重视这里的精度。

> **Fisher 信息近似：** imatrix 是 Fisher 信息矩阵对角线的近似。  
> 完整 Fisher 矩阵是 n×n，一层就需要 500TB 存不下。  
> 用激活平方和近似（只有 n 个数），一次前向就搞定。

---

## 8. Transformer Block 结构

LLM 由多个相同结构的 Block 堆叠（7B 模型有 32 个）：

```
输入 x
  ↓
LayerNorm           ← 归一化，让数值稳定在 [-2, +2] 附近
  ↓
Self-Attention      ← 4 个 Linear 层（Q/K/V/O projection）
  ↓
残差连接 x = x + attn_output
  ↓
LayerNorm
  ↓
FFN（前馈网络）     ← 3 个 Linear 层（gate / up / down proj）
  ↓
残差连接
  ↓
输出 x
```

AutoRound 量化的对象就是这些 Linear 层的权重矩阵，**一次处理一个 Block**。

---

## 名词速查表

| 术语 | 含义 |
|------|------|
| FP16 / BF16 | 16 位浮点，模型原始存储格式 |
| INT4 / INT2 | 4/2 位整数，量化目标格式 |
| scale | 量化格子间距，$s = (\max W - \min W) / (2^b - 1)$ |
| zero_point (zp) | 零点偏移，非对称量化才有 |
| RTN | Round-To-Nearest，直接四舍五入 |
| QDQ | Quantize-Dequantize，先量化再反量化 |
| STE | Straight-Through Estimator，梯度穿透 round() |
| group_size | 每多少个权重共享一个 scale（默认 128）|
| iters | SignSGD 优化步数（默认 200）|
| nsamples | 校准数据条数（默认 128）|
| PAD | 填充符号，让一批文本对齐到相同长度 |
| attention_mask | 标记真实内容（1）和 PAD（0）|
| hidden_dim | 每个 token 的向量维度（LLaMA-7B = 4096）|
| AMP | Automatic Mixed Precision，自动混合精度 |
| MoE | Mixture of Experts，混合专家模型 |

---

➡️ 下一章：[02 · 算法原理](./02_algorithm.md)
