# GPTQ CUDA Kernel 详解

> 量化后模型如何在 GPU 上高效推理。

---

## 问题：为什么需要专门的 Kernel

```
FP16 矩阵乘法：GPU 有 Tensor Core，非常快
INT4 矩阵乘法：没有原生指令，需要：
  1. 解包 INT4（从 INT32 里用位操作取出）
  2. 反量化 -> FP16
  3. 做 FP16 矩阵乘法
```

---

## INT4 的存储格式

GPU 最小存储单位是 8-bit。两个 INT4 打包进一个 INT32（共装 8 个）：

```
w0 = 5  = 0b0101   (bit 0-3)
w1 = 10 = 0b1010   (bit 4-7)
...
packed = (w7<<28)|(w6<<24)|...|(w1<<4)|w0

# 解包：
w0 = (packed >> 0) & 0xF
w1 = (packed >> 4) & 0xF
```

存储形状：`[out, in]` -> `[out, in//8]`，每个 INT32 含 8 个 INT4。

---

## Marlin Kernel（最快）

**核心设计三点：**

**① 利用 Tensor Core**

把计算组织成 16×16 的 tile，让 Tensor Core 的 WMMA 指令处理：
```
每个 warp（32线程）处理一个 tile
解包 + 反量化在寄存器里完成，不写 global memory
```

**② 权重预排（Reorder）**

量化时就把权重重排成 Tensor Core 的访问模式，推理时每个线程取自己负责的部分，没有线程间通信。

**③ Double Buffering（流水线）**

```
第 k 轮：Tensor Core 计算第 k 块
         同时预取第 k+1 块权重
两件事并行，GPU 利用率接近峰值
```

---

## ExLlamaV2 Kernel

针对 batch_size=1（单 token 生成）优化。

**核心原因：** 单 token 时矩阵乘法退化为向量×矩阵，瓶颈是**内存带宽**（把权重从显存读出来），不是计算。

优化：用 128-bit 加载指令一次读 4 个 INT32（=32 个 INT4），最大化带宽利用率。

---

## AutoRound 推理 Backend 选择

```python
# 按优先级探测已安装的库
BACKENDS = [
    ("auto_round_kernel", priority=6),  # Intel 自研
    ("Marlin",            priority=6),  # CUDA，对称 INT4
    ("ExLlamaV2",         priority=5),  # CUDA，GPTQ 格式
    ("IPEX",              priority=5),  # Intel Extension
    ("Triton",            priority=2),  # 通用 GPU，Python 写
    ("PyTorch",           priority=0),  # 纯 Python 兜底
]
```

**Marlin 要求：** sym=True + group_size=128 + INT4

---

## 性能对比（7B 模型，A100，batch=1）

| Kernel | tokens/s | 显存 |
|--------|---------|------|
| FP16 原始 | 100 | 14GB |
| PyTorch INT4 | 80 | 4GB |
| Triton | 150 | 4GB |
| ExLlamaV2 | 250 | 4GB |
| Marlin | 350+ | 4GB |

Marlin 比 FP16 还快，因为权重读取量减少 4 倍，内存带宽压力大幅降低。

---

## Triton 简介

OpenAI 开发的 GPU 编程语言，比 CUDA 高级，接近写 NumPy：

```python
@triton.jit
def int4_matmul_kernel(x_ptr, w_ptr, out_ptr, ...):
    # 只需描述：处理哪个 tile，加载什么数据，做什么计算
    w_packed = tl.load(w_ptr + ...)
    w0 = (w_packed >> 0) & 0xF        # 解包
    w0_dq = (w0.to(float16) - zp) * scale  # 反量化
    acc += tl.dot(x_block, w_dq)      # 矩阵乘法
    tl.store(out_ptr + ..., acc)
    # Triton 自动处理 thread 管理、shared memory、bank conflict
```

---

[返回文档地图](../../README.md)
