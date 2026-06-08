# MXFP4 和 NVFP4 格式详解

---

## MXFP4（MicroScaling FP4）

微软 2023 年提出，用 4-bit 浮点（不是整数）表示权重。

**和 INT4 的根本区别：**

```
INT4 格子：均匀分布
  -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7

MXFP4 格子：靠近 0 密集，远离 0 稀疏（浮点分布）
  0, ±0.125, ±0.25, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
```

神经网络权重通常集中在 0 附近，MXFP4 的格子分布更贴合。

**共享指数（shared_exp）：**

每 group（如 32 个数）共享一个 8-bit 指数：

```
group 里最大绝对值 = 0.75
shared_exp = floor(log2(0.75)) = floor(-0.41) = -1
scale = 2^(-1) = 0.5

每个数先除以 scale，再用 4-bit 浮点表示归一化后的值
```

**为什么用 floor_ste 而不是 floor：**

量化训练需要梯度，`floor()` 不可微（导数=0），用 `floor_ste` 让梯度穿透。

**适用场景：** H100/B200 以上 GPU，需要 LLM-Compressor 或 vLLM。

---

## NVFP4

NVIDIA 定义的 FP4 格式，每 16 个权重共享一个 FP8 类型的 per-group scale，加上整个 tensor 的 global_scale。

**三层结构：**

```
W_nvfp4 (4-bit)          ← 最小，每个权重
group_scale (FP8)         ← 每 16 个权重一个
global_scale (FP32)       ← 整个 tensor 一个

反量化：W_dq = W_nvfp4 × group_scale × global_scale
```

**为什么需要 global_scale：**

FP8 的表示范围有限。如果权重值很大，FP8 格式的 group_scale 会溢出。  
global_scale 先把整个 tensor 缩放到合理范围：

```python
global_scale = FP8_MAX × FP4_MAX / max(|W|)
# 让最大权重恰好填满 FP4 的表示范围
```

**适用场景：** H100/B200，需要 TensorRT-LLM 或专用 NVFP4 kernel。

---

[返回文档地图](../../README.md)
