# 03 · 项目架构

> 文件布局、模块职责、完整调用链、以及为什么这样设计。

---

## 目录结构

```
auto_round/
├── __init__.py              必须放顶层：让 auto_round/ 成为合法的 Python 包
├── __main__.py              支持 python -m auto_round 命令行调用
├── autoround.py           ★ 用户入口（Factory Pattern）
├── schemes.py               量化方案字符串解析
├── wrapper.py             ★ WrapperLinear（Fake Quant 核心）
├── alg_ext.py             ★ V2 算法扩展
├── calib_dataset.py         校准数据集加载
├── formats.py               导出格式管理
│
├── compressors/           ★ 编排层（谁来执行量化）
│   ├── entry.py             模型类型检测 + 动态类组合
│   ├── base.py              BaseCompressor（基础设施）
│   └── data_driven.py     ★ 主循环（quantize + _quantize_blocks）
│
├── algorithms/            ★ 算法层（怎么量化）
│   └── quantization/sign_round/
│       ├── quantizer.py   ★ SignSGD 主循环（quantize_block）
│       └── sign_sgd.py    ★ SignSGD 优化器
│
├── auto_scheme/             自动混合精度（AutoScheme + DP 比特分配）
├── data_type/             ★ 量化数学（公式实现）
│   ├── int.py             ★ INT 量化
│   ├── mxfp.py              MXFP4 量化
│   └── nvfp.py              NVFP4 量化
│
├── calibration/             校准 Hook 系统
├── formats/                 导出格式（GPTQ/AWQ/GGUF）
├── modeling/                推理时加载量化模型
└── utils/                   工具函数

auto_round_extension/        C++/CUDA 加速 kernel（仅推理用）
```

> ⚠️ **常见误区：** 以下文件在实际源码中**不存在**：  
> `algorithms/rtn.py`、`auto_scheme/dp_solver.py`、`calibration/data_sampler.py`、`modeling/patcher.py`

---

## 五层架构

```
① 入口层   autoround.py
           Factory，根据模型类型返回不同 Compressor
           ↓
② 编排层   compressors/data_driven.py
           谁来执行：调度校准、Block 循环、导出
           ↓
③ 算法层   sign_round/quantizer.py
           怎么量化：SignSGD 200 步优化循环
           ↓
④ 数学层   data_type/int.py
           量化公式：quant_tensor_asym/sym
           ↓
⑤ 导出层   formats/
           存成什么格式：GPTQ/AWQ/GGUF/auto_round
```

**最重要的设计原则：编排层和算法层完全解耦。**
- 换算法：不需要改 Compressor
- 支持新模型：不需要改算法
- 加新导出格式：不需要改任何 Compressor 或算法代码

---

## 入口：Factory Pattern

`AutoRound.__new__` 不返回 `AutoRound` 实例：

```python
ar = AutoRound("Qwen3-0.6B", scheme="W4A16")
print(type(ar))   # <class 'DataDrivenCompressor'>
```

路由逻辑：

```
AutoRound.__new__()
  ↓
detect_model_type(model)
  ├── 'diffusion' → DiffusionMixin + DataDrivenCompressor
  ├── 'mllm'     → MLLMMixin + DataDrivenCompressor
  └── 'llm'      → DataDrivenCompressor（默认）

iters == 0?
  ├── True  → ZeroShotCompressor
  └── False → DataDrivenCompressor
```

---

## 完整调用链

```
用户：ar = AutoRound("Qwen3-0.6B", scheme="W4A16")
      ar.quantize_and_save("./output")
      │
      ▼
autoround.py  AutoRound.__new__()
  解析 scheme → bits=4, group_size=128, sym=False
  检测模型类型 → 'llm'，iters=200
  返回 DataDrivenCompressor 实例
      │
      ▼
compressors/data_driven.py  quantize()
  │
  ├─ [V2] alg_ext.py  _register_act_max_hook_ext()
  │       注册 imatrix hook → 喂校准数据 → 每层收集激活平方和
  │
  ├─ calib_dataset.py  get_dataloader()
  │       下载 128 条文本，tokenize → DataLoader
  │
  └─ for each Transformer Block:
       _quantize_blocks()
         │
         ├─ 浮点前向：收集 reference_output（优化目标）
         │
         ├─ sign_round/quantizer.py  quantize_block()
         │    │
         │    ├─ [V1] wrapper.py  wrapper_block()
         │    │        Linear → WrapperLinear（V=0, α=β=1）
         │    │
         │    ├─ [V2] alg_ext.py  WrapperLinearV2.__init__()
         │    │        search_scales(weight, imatrix) → self.init_scale
         │    │
         │    ├─ for step in range(200):
         │    │    ① Forward: WrapperLinear.forward(x)
         │    │         wrapper.py  _qdq_weight()
         │    │           data_type/int.py  quant_tensor_asym()
         │    │             W_int = round(W/scale + V + zp)  [STE]
         │    │             W_dq  = (W_int - zp) × scale     [fake-quant]
         │    │         y = x @ W_dq.T + bias
         │    │    ② Loss:
         │    │         [V1] MSE(y, reference_output)
         │    │         [V2] alg_ext.py _get_loss_ext()
         │    │              排除差异 top-0.1% 后算 MSE
         │    │    ③ Backward: STE 穿透 round()
         │    │    ④ SignSGD:
         │    │         sign_sgd.py  p.grad = sign(p.grad) → ±1
         │    │         p -= lr × sign(∇)，lr 线性衰减
         │    │    ⑤ 保存 loss 最低那步的参数
         │    │
         │    └─ wrapper.py  unwrapper_block(best_params)
         │             写回整数权重，WrapperLinear → nn.Linear
         │
         └─ [可选] 立即 pack 或立即写盘
      │
      ▼
formats/  pack_weight + 写 config.json + safetensors
```

---

## alg_ext.py 与算法的关系

`alg_ext.py` 通过动态绑定把 V2 方法挂载到量化器实例上，V1 代码一行不改：

```python
def wrapper_autoround(cls):          # cls = SignRoundQuantizer 实例
    # ① 替换 loss 函数 → Loss Filtering
    cls._get_loss = types.MethodType(_get_loss_ext, cls)

    # ② 替换 wrapper → WrapperLinearV2（内含 scale 搜索）
    setattr(cls, "wrapper_block", wrapper_block_v2)

    # ③ 替换 hook → 额外收集 imatrix
    cls._register_act_max_hook = types.MethodType(_register_act_max_hook_ext, cls)
```

| 插入点 | 替换了什么 | 对应论文 |
|--------|-----------|---------|
| `_get_loss_ext` | Loss Filtering | §V2 改进② |
| `wrapper_block_v2` + `WrapperLinearV2` | Scale 搜索初始化 | §V2 改进① |
| `_register_act_max_hook_ext` | imatrix 收集 | §V2 改进③ |

---

## 为什么 `__init__.py` 和 `__main__.py` 放顶层

- `__init__.py`：Python 强制规定，必须和它代表的包在同一层，无法移到子目录
- `__main__.py`：支持 `python -m auto_round`，Python 固定查找当前包根目录

---

➡️ 下一章：[04 · 代码注释（选择对应文件）](./code/)
