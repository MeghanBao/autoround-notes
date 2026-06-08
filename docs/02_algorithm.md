# 02 · 算法原理

> SignRound V1（EMNLP 2024）和 V2（arXiv 2025）完整推导。

---

## V1：SignRound 核心算法

### 换一个优化目标

RTN 最小化权重误差：$\min \|W - \text{round}(W)\|^2$

AutoRound 最小化**层输出误差**：

$$\min_{V, \alpha, \beta} \| \text{Block}(X; W) - \text{Block}(X; W_q(V, \alpha, \beta)) \|_F^2$$

### 三个可学习参数

**① $V$（Rounding Value）**

$$W_q = \text{round}(W/s + V), \quad V \in [-0.5, 0.5]$$

- $V = 0$：等价于 RTN
- $V > 0$：倾向上取整；$V < 0$：倾向下取整
- 逐元素：一个 [4096, 4096] 权重矩阵有 16M 个独立的 $V$

**② $\alpha$（max_scale）和 ③ $\beta$（min_scale）**

$$s = \frac{\alpha \cdot \max(W) - \beta \cdot \min(W)}{2^b - 1}, \quad \alpha, \beta \in [0,1]$$

初始 1.0（不裁剪），通过学习缩小量化范围，减少异常值影响。

### SignSGD 更新规则

$$\theta_{t+1} = \theta_t - lr \cdot \text{sign}(\nabla L)$$

| 理由 | 说明 |
|------|------|
| 解空间有界 | $V \in [-0.5, 0.5]$，$\alpha/\beta \in [0,1]$ |
| 超参简单 | $lr=5\times10^{-3}$，200步，总位移 $200 \times 0.0025 = 0.5$ 恰好覆盖解空间 |
| 省显存 | 不需要 Adam 的动量和方差缓冲区 |

### V1 伪代码

```python
for block in model.blocks:
    wrapper_block(block)              # Linear → WrapperLinear，注入 V/α/β

    for step in range(200):
        y_quant = block(x)            # Forward（fake-quant）
        loss    = MSE(y_quant, y_fp)  # 量化输出 vs 浮点参考
        loss.backward()               # STE 穿透 round()
        V.grad  = sign(V.grad)        # 梯度 → ±1
        optimizer.step()              # V -= lr × (±1)
        lr 线性衰减

    unwrapper_block(block, best_params)   # 写回权重，还原 nn.Linear
```

---

## V2：三项改进

V1 在 4-bit 下优秀，但 2-bit 时精度差距明显。V2 加了三项改进。

### 改进① Scale 搜索（预调优初始化）

**问题：** V1 初始 $\alpha=1, V=0$，在 2-bit 下可能陷入差的局部最优。

**方法：** 在 SignSGD 开始前，先搜索最优初始 scale：

$$s_{init} = \arg\min_{s \in S} \sum_j |\bar{A}_j| \cdot (W_j - \text{QDQ}(W_j, s))^2$$

- $S$：候选 scale 集合（枚举约 100 个候选，步长 0.01）
- $\bar{A}_j$：通道 $j$ 的平均激活值（即 imatrix，体现重要性）

找到 $s_{init}$ 后，V2 把 $\alpha$ 的范围扩展到 $[0, 2]$（可缩可扩）：

$$s = s_{init} \times \alpha$$

### 改进② Loss Filtering（损失过滤）

**问题：** 2-bit 时某些 token 产生异常大的输出误差，梯度破坏已收敛参数。

**方法：** 排除差异最大的 top-0.1% 位置：

$$L_{filtered} = \text{mean}\left(L \setminus \text{TopK}(|Y_q - Y_{fp}|,\ 0.001 \times |L|)\right)$$

步骤：
1. 算 $|Y_q - Y_{fp}|$（绝对差异）
2. 找差异最大的 top-0.1% 位置
3. 排除这些位置，对剩余 99.9% 的误差平方求均值

### 改进③ DeltaLoss 敏感度（用于 AutoScheme 比特分配）

不同层量化敏感度不同，DeltaLoss 度量第 $i$ 层的敏感度：

$$\Delta L_i = \| g_{W_q} \odot (W_f - W_q) \|_1$$

- $g_{W_q}$：量化模型的梯度（哪些权重改变会影响 loss）
- $W_f - W_q$：量化误差（实际差了多少）

用动态规划（DP）在平均 bit 预算约束下，最小化总 DeltaLoss，求最优比特分配。

### V2 完整流程

```
阶段 0：收集 imatrix
  → 注册 forward hook，喂 128 条校准数据
  → 每层 Linear 收集激活平方和

阶段 1：逐 Block 量化
  → WrapperLinearV2.__init__() 调 search_scales(weight, imatrix)
  → 得到 self.init_scale（更好的起点）

  → SignSGD 200 步：
      Forward：scale = init_scale × α  ← 起点更好
      Loss：_get_loss_ext()  ← 排除 top-0.1% 异常
      Backward + SignSGD

  → unwrap，写回权重
```

---

## V1 vs V2 对比

| 特性 | V1 | V2 |
|------|----|----|
| $V$ 初始值 | 0 | 0 |
| $\alpha/\beta$ 范围 | $[0, 1]$ | $[0, 2]$ |
| scale 初始化 | 统计最大值 | search_scales 搜索 |
| loss 函数 | MSE | MSE + Loss Filtering |
| imatrix | 无 | 有 |
| 适合场景 | W4 / W3 | W2 / MXFP4 / NVFP4 |

---

## 支持的量化格式

| 格式 | 注册名 | 特点 |
|------|--------|------|
| INT4 非对称 | `int_asym` | 默认，精度最好 |
| INT4 对称 | `int_sym` | 推理更快，精度略低 |
| GPTQ 对称 | `int_sym_gptq` | 兼容 Marlin/ExLlamaV2 kernel |
| MXFP4 | `mx_fp4` | 微软浮点格式 |
| NVFP4 | `nv_fp4` | NVIDIA，需要 global_scale |
| GGUF | `int_*_dq` | llama.cpp，使用双重量化 |

---

➡️ 下一章：[03 · 项目架构](./03_architecture.md)
