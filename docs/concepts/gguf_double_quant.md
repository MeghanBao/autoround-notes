# GGUF 双重量化详解

---

## 什么是双重量化

普通量化：权重（FP16）→ INT4 + scale（FP16）

scale 本身占多少空间？

```
group_size = 32
32 个权重：32 × 4-bit = 16 字节
1 个 scale：2 字节
额外开销：2/16 = 12.5%
```

scale 不小！双重量化把 scale 本身再量化一次：

```
scale（FP16）→ scale_int（INT6）+ d_scale（FP16）
scale 从 16-bit → 6-bit，再节省约 0.5-1 bit/权重
```

---

## 存储对比

| 项目 | 普通量化 | 双重量化 |
|------|---------|---------|
| 权重 | INT4 | INT4 |
| scale | FP16 | INT6 |
| scale 的 scale | 无 | FP16（很少）|
| zp | INT4 | INT6 |
| zp 的 scale | 无 | FP16（很少）|

---

## 代码里的变量名

```python
prev_scale   = 权重的量化 scale
prev_d_scale = scale 的量化 scale（d = double）
prev_wmin    = 权重的量化最小值（非对称用）
prev_d_wmin  = wmin 的量化 scale
```

---

## 为什么每 10 步重搜

GGUF 的 scale 搜索（search_gguf_scale_min）有循环，用 `@inference_mode` 装饰，不可微。

```
每步都搜：太慢
从不重搜：V 更新了，scale 跟不上，精度差

折中：每 10 步重搜一次
```

中间步用 `.detach()` 复用缓存：

```python
params = {"scale": self.prev_scale.detach()}
# detach() = 切断计算图
# 推理模式产生的张量不能参与反向传播
# detach 后变成普通张量，backward 到这里停住
```

---

## 适用场景

GGUF 是 llama.cpp 使用的格式，主要用于：
- 在消费级硬件（8GB 内存笔记本）本地运行大模型
- 不需要 CUDA，CPU 也能推理（慢但能用）

AutoRound 的 GGUF 格式导出即可被 llama.cpp / Ollama / LM Studio 直接使用。

---

[返回文档地图](../../README.md)
