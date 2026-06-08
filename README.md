# AutoRound 源码精读笔记

> **Intel 开源 LLM 量化工具包 AutoRound 的完整学习文档**
>
> 源码：[intel/auto-round](https://github.com/intel/auto-round) · Commit `09bc7437`  
> 适合人群：**本科生 / 研究生**，从零开始理解 LLM 量化的算法与工程实现

---

## 这个项目是什么

AutoRound 实现了论文 **SignRound**（EMNLP 2024）和 **SignRoundV2**（arXiv 2025），是目前工业界最完整的 LLM 量化工具之一。

这份笔记的目标：**把论文公式和真实代码一一对应起来**，不假设任何预备知识，从「量化是什么」讲到「每行代码为什么这样写」。

---

## 快速导航

```
📖 第一次阅读？从这里开始：
   docs/01_basics.md  →  docs/02_algorithm.md  →  docs/03_architecture.md

🔍 想看具体代码？
   docs/code/  →  选择对应的源文件

💡 想理解设计决策？
   docs/engineering/architecture_design.md

❓ 有具体概念不懂？
   docs/concepts/  →  找对应的概念文件
```

---

## 文档地图

```
autoround-notes/
│
├── 📄 README.md                        ← 你在这里
│
├── 📁 docs/
│   ├── 01_basics.md                    基础概念（RTN / 梯度下降 / Forward-Backward 等 8 个）
│   ├── 02_algorithm.md                 算法原理（SignRound V1 + V2 三项改进）
│   ├── 03_architecture.md              项目架构（5层设计 + 调用链 + 文件布局原因）
│   │
│   ├── 📁 code/                        核心源码逐行注释
│   │   ├── int_py.md                   data_type/int.py（量化数学公式）
│   │   ├── wrapper_py.md               wrapper.py（WrapperLinear + Fake Quant）
│   │   ├── sign_sgd_py.md              sign_round/sign_sgd.py（SignSGD 优化器）
│   │   ├── quantizer_py.md             sign_round/quantizer.py（SignSGD 主循环）
│   │   ├── alg_ext_py.md               alg_ext.py（V2 算法扩展，完整逐行）
│   │   └── autoround_py.md             autoround.py（入口 Factory）
│   │
│   ├── 📁 concepts/                    专题概念深度讲解
│   │   ├── gptq_cuda_kernel.md         GPTQ CUDA Kernel（Marlin / ExLlamaV2）
│   │   ├── mxfp4_nvfp4.md              MXFP4 和 NVFP4 格式详解
│   │   └── gguf_double_quant.md        GGUF 双重量化详解
│   │
│   └── 📁 engineering/                 工程设计与启发
│       ├── architecture_design.md      为什么这样设计架构
│       └── research_insights.md        对研究生的启发与建议
│
└── 📁 assets/
    └── call_chain.md                   完整调用链文字图
```

---

## 核心概念速查

### 论文符号 ↔ 代码变量

| 论文符号 | 代码变量 | 文件 | 含义 |
|----------|---------|------|------|
| $V$ | `self.value` | `wrapper.py` | 可学习取整偏移，每个权重独立 |
| $\alpha$ | `self.max_scale` | `wrapper.py` | 上界缩放因子 |
| $\beta$ | `self.min_scale` | `wrapper.py` | 下界缩放因子 |
| $s$ | `scale` | `data_type/int.py` | 量化格子间距 |
| $zp$ | `zp` | `data_type/int.py` | 零点（非对称量化用）|
| $\hat{W}$ | `qdq_result` | `data_type/int.py` | 反量化浮点权重（Fake Quant 结果）|
| $Y_{orig}$ | `reference_output` | `quantizer.py` | 浮点参考输出（优化目标）|
| $Y_{quant}$ | `output_q` | `quantizer.py` | 量化后输出 |
| $\text{sign}(\nabla)$ | `torch.sign(p.grad)` | `quantizer.py` | 梯度符号（SignSGD 核心）|
| $s_{init}$ | `self.init_scale` | `alg_ext.py` | Scale 搜索结果（V2）|
| imatrix | `self.imatrix` | `alg_ext.py` | 激活重要性矩阵（V2）|

### 术语速查

| 术语 | 一句话解释 |
|------|----------|
| RTN | Round-To-Nearest，直接四舍五入，最简单的量化 |
| QDQ | Quantize-Dequantize，先量化再反量化，用于模拟量化误差 |
| Fake Quant | 训练时保持浮点但模拟量化效果，让梯度能传播 |
| STE | Straight-Through Estimator，让梯度穿透不可微的 round() |
| group_size | 每多少个权重共享一个 scale（默认 128）|
| imatrix | 激活平方和，衡量每个输入维度的重要性 |
| init_scale | V2 Scale 搜索的结果，比默认 scale 更优 |
| Block-wise | 一次优化一个 Transformer Block，而不是整个模型 |
| PAD | 填充符号，让一批文本对齐到相同长度 |
| AMP | Automatic Mixed Precision，自动混合精度 |
| MoE | Mixture of Experts，混合专家模型（如 Mixtral）|
| MXFP4 | 微软定义的 FP4 浮点格式，格子分布不均匀 |
| NVFP4 | NVIDIA 定义的 FP4 格式，有额外 global_scale |
| GGUF | llama.cpp 的量化格式，支持双重量化 |

---

## 五分钟理解 AutoRound

**问题：** 把 LLM 权重从 FP16 压缩到 INT4 时，直接四舍五入（RTN）精度损失大。

**原因：** RTN 每个权重独立取整，不考虑「取整后整层输出误差有多大」。

**AutoRound 的解法：** 给每个权重加一个可学习偏移 $V \in [-0.5, 0.5]$，用校准数据优化 $V$，让量化后的输出尽量接近原始输出。

```
RTN：        W_q = round(W / s)
AutoRound：  W_q = round(W / s + V)   ← V 可学习
```

**优化目标：**

$$\min_{V, \alpha, \beta} \| \text{Block}(X; W) - \text{Block}(X; W_q(V, \alpha, \beta)) \|_F^2$$

**优化方法：** SignSGD，只用梯度方向（±1）更新参数，200 步，线性衰减学习率。

---

## 快速上手（Mac / 无 GPU）

```bash
pip install auto-round transformers

# 方法一：直接用已量化好的模型（推荐）
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    'OPEA/Qwen3-0.6B-int4-sym-inc', device_map='cpu')
tokenizer = AutoTokenizer.from_pretrained('OPEA/Qwen3-0.6B-int4-sym-inc')
inputs = tokenizer('你好', return_tensors='pt')
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
"

# 方法二：Google Colab 上自己量化（免费 T4 GPU）
# !pip install auto-round
# !auto-round --model Qwen/Qwen3-0.6B --scheme W4A16 --iters 0   # RTN，最快
# !auto-round --model Qwen/Qwen3-0.6B --scheme W4A16             # 完整 SignRound
```

---

## 推荐阅读顺序

### 路线一：理解算法（适合想做量化研究的同学）

1. `docs/01_basics.md` — 搞清楚 RTN、梯度下降、zero-point 等基础
2. `docs/02_algorithm.md` — 理解 SignRound 的核心公式和 V2 的三项改进
3. `docs/code/quantizer_py.md` — 看算法主循环怎么实现的
4. `docs/code/alg_ext_py.md` — 理解 V2 改进在代码里怎么插进去的

### 路线二：理解工程（适合想做 MLSys 的同学）

1. `docs/03_architecture.md` — 理解 5 层架构和调用链
2. `docs/code/wrapper_py.md` — WrapperLinear 是怎么实现 Fake Quant 的
3. `docs/engineering/architecture_design.md` — 每个设计决策背后的工程逻辑
4. `docs/concepts/gptq_cuda_kernel.md` — CUDA Kernel 的底层优化

### 路线三：快速入门（适合只需要了解大概的同学）

1. 本文档的「五分钟理解」部分
2. `docs/01_basics.md` 的名词速查表
3. `docs/03_architecture.md` 的调用链图

---

## 参考资料

- [AutoRound 论文（SignRound，EMNLP 2024）](https://arxiv.org/abs/2309.05516)
- [AutoRound 源码](https://github.com/intel/auto-round)
- [GPTQ 论文](https://arxiv.org/abs/2210.17323)
- [Marlin Kernel](https://github.com/IST-DASLab/marlin)
- [MXFP4 规格（微软）](https://arxiv.org/abs/2310.10537)

---

*文档基于 commit `09bc7437`，Apache 2.0 协议。*
