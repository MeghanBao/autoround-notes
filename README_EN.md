# AutoRound Source Code Study Notes

> **Comprehensive learning documentation for Intel's open-source LLM quantization toolkit AutoRound**
>
> Source: [intel/auto-round](https://github.com/intel/auto-round)
> Audience: **Undergraduates / Graduate students** — understanding LLM quantization from scratch, both algorithm and engineering

---

## What Is This Project

AutoRound implements the **SignRound** paper (EMNLP 2024) and **SignRoundV2** (arXiv 2025), and is one of the most complete LLM quantization toolkits in industry today.

The goal of these notes: **map every paper formula to the actual code**, assuming no prior knowledge — from "what is quantization" all the way to "why is each line written this way".

---

## Quick Navigation

```
📖 First time reading? Start here:
   docs/01_basics.md  →  docs/02_algorithm.md  →  docs/03_architecture.md

🔍 Want to read specific code?
   docs/code/  →  pick the corresponding source file

💡 Want to understand design decisions?
   docs/engineering/architecture_design.md

❓ Unclear on a specific concept?
   docs/concepts/  →  find the relevant concept file
```

---

## Document Map

```
autoround-notes/
│
├── 📄 README.md                        ← Chinese version
├── 📄 README_EN.md                     ← You are here
│
├── 📁 docs/
│   ├── 01_basics.md                    Fundamentals (RTN / gradient descent / Forward-Backward, 8 concepts)
│   ├── 02_algorithm.md                 Algorithm (SignRound V1 + V2 three improvements)
│   ├── 03_architecture.md              Architecture (5-layer design + call chain + file layout rationale)
│   │
│   ├── 📁 code/                        Core source code with line-by-line annotations
│   │   ├── int_py.md                   data_type/int.py (quantization math formulas)
│   │   ├── wrapper_py.md               wrapper.py (WrapperLinear + Fake Quant)
│   │   ├── quantizer_py.md             sign_round/quantizer.py (SignSGD main loop)
│   │   └── alg_ext_py.md               alg_ext.py (V2 algorithm extension, full line-by-line)
│   │
│   ├── 📁 concepts/                    In-depth explanations of specific topics
│   │   ├── gptq_cuda_kernel.md         GPTQ CUDA Kernel (Marlin / ExLlamaV2)
│   │   ├── mxfp4_nvfp4.md              MXFP4 and NVFP4 format explained
│   │   └── gguf_double_quant.md        GGUF double quantization explained
│   │
│   └── 📁 engineering/                 Engineering design and insights
│       ├── architecture_design.md      Why the architecture is designed this way
│       └── research_insights.md        Takeaways and advice for graduate researchers
│
└── 📁 assets/
    └── call_chain.md                   Full call chain diagram (text)
```

---

## Core Concept Reference

### Paper Notation ↔ Code Variables

| Paper Symbol | Code Variable | File | Meaning |
|---|---|---|---|
| $V$ | `self.value` | `wrapper.py` | Learnable rounding offset, one per weight |
| $\alpha$ | `self.max_scale` | `wrapper.py` | Upper bound scaling factor |
| $\beta$ | `self.min_scale` | `wrapper.py` | Lower bound scaling factor |
| $s$ | `scale` | `data_type/int.py` | Quantization grid spacing |
| $zp$ | `zp` | `data_type/int.py` | Zero point (used in asymmetric quantization) |
| $\hat{W}$ | `qdq_result` | `data_type/int.py` | Dequantized float weight (Fake Quant result) |
| $Y_{orig}$ | `reference_output` | `quantizer.py` | Float reference output (optimization target) |
| $Y_{quant}$ | `output_q` | `quantizer.py` | Quantized output |
| $\text{sign}(\nabla)$ | `torch.sign(p.grad)` | `quantizer.py` | Gradient sign (core of SignSGD) |
| $s_{init}$ | `self.init_scale` | `alg_ext.py` | Scale search result (V2) |
| imatrix | `self.imatrix` | `alg_ext.py` | Activation importance matrix (V2) |

### Glossary

| Term | One-line Explanation |
|---|---|
| RTN | Round-To-Nearest — simplest quantization, round each weight directly |
| QDQ | Quantize-Dequantize — quantize then dequantize to simulate quantization error |
| Fake Quant | Stay in float during training but simulate quantization, so gradients can flow |
| STE | Straight-Through Estimator — lets gradients pass through the non-differentiable `round()` |
| group_size | How many weights share one scale (default 128) |
| imatrix | Sum of squared activations — measures importance of each input dimension |
| init_scale | Result of V2 scale search, better than the default scale |
| Block-wise | Optimize one Transformer Block at a time, not the whole model |
| PAD | Padding token to align a batch of texts to the same length |
| AMP | Automatic Mixed Precision |
| MoE | Mixture of Experts (e.g. Mixtral) |
| MXFP4 | Microsoft's FP4 floating-point format, non-uniform grid spacing |
| NVFP4 | NVIDIA's FP4 format, includes an extra global_scale |
| GGUF | llama.cpp quantization format, supports double quantization |

---

## AutoRound in 5 Minutes

**Problem:** When compressing LLM weights from FP16 to INT4, naive round-to-nearest (RTN) causes large accuracy loss.

**Why:** RTN rounds each weight independently, ignoring how much rounding error accumulates across the whole layer's output.

**AutoRound's solution:** Add a learnable offset $V \in [-0.5, 0.5]$ to each weight. Use calibration data to optimize $V$ so that the quantized output stays close to the original output.

```
RTN:         W_q = round(W / s)
AutoRound:   W_q = round(W / s + V)   ← V is learnable
```

**Optimization objective:**

$$\min_{V, \alpha, \beta} \| \text{Block}(X; W) - \text{Block}(X; W_q(V, \alpha, \beta)) \|_F^2$$

**Optimizer:** SignSGD — only the sign (±1) of the gradient is used to update parameters. 200 steps with linear learning rate decay.

---

## Quick Start (Mac / No GPU)

```bash
pip install auto-round transformers

# Option 1: Use a pre-quantized model (recommended)
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    'OPEA/Qwen3-0.6B-int4-sym-inc', device_map='cpu')
tokenizer = AutoTokenizer.from_pretrained('OPEA/Qwen3-0.6B-int4-sym-inc')
inputs = tokenizer('Hello', return_tensors='pt')
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
"

# Option 2: Quantize yourself on Google Colab (free T4 GPU)
# !pip install auto-round
# !auto-round --model Qwen/Qwen3-0.6B --scheme W4A16 --iters 0   # RTN, fastest
# !auto-round --model Qwen/Qwen3-0.6B --scheme W4A16             # Full SignRound
```

---

## Recommended Reading Order

### Path 1: Understand the Algorithm (for those interested in quantization research)

1. `docs/01_basics.md` — Get clear on RTN, gradient descent, zero-point, and other fundamentals
2. `docs/02_algorithm.md` — Understand SignRound's core formulas and V2's three improvements
3. `docs/code/quantizer_py.md` — See how the algorithm's main loop is implemented
4. `docs/code/alg_ext_py.md` — Understand how V2 improvements are wired into the code

### Path 2: Understand the Engineering (for those interested in MLSys)

1. `docs/03_architecture.md` — Understand the 5-layer architecture and call chain
2. `docs/code/wrapper_py.md` — How WrapperLinear implements Fake Quant
3. `docs/engineering/architecture_design.md` — The engineering logic behind each design decision
4. `docs/concepts/gptq_cuda_kernel.md` — Low-level optimizations in CUDA kernels

### Path 3: Quick Overview (for those who just need the big picture)

1. The "5 Minutes" section in this document
2. The glossary table in `docs/01_basics.md`
3. The call chain diagram in `docs/03_architecture.md`

---

## References

- [AutoRound paper (SignRound, EMNLP 2024)](https://arxiv.org/abs/2309.05516)
- [AutoRound source code](https://github.com/intel/auto-round)
- [GPTQ paper](https://arxiv.org/abs/2210.17323)
- [Marlin Kernel](https://github.com/IST-DASLab/marlin)
- [MXFP4 specification (Microsoft)](https://arxiv.org/abs/2310.10537)

---

*Apache 2.0 License.*
