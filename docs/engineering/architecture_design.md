# 为什么 AutoRound 的架构是这样的

---

## 每个设计决策背后的原因

### 1. Factory Pattern：为什么用 `__new__`

用户是 ML 研究员，不想记住「文本模型用 LLMCompressor，图文模型用 MLLMCompressor」。

`__new__` 让用户永远只写一行：

```python
ar = AutoRound(model, scheme="W4A16")
```

背后自动路由，对用户透明。这叫**最小惊讶原则**。

### 2. 为什么 iters=0 是独立的 ZeroShotCompressor

iters=0（RTN）和 iters>0（SignSGD）的资源需求完全不同：
- RTN：不需要校准数据、不需要梯度、可以 CPU 运行
- SignSGD：需要 GPU、DataLoader、优化器...

如果写在一起用 if/else，两条路径互相污染——RTN 路径也要初始化 DataLoader 等完全用不到的东西。

### 3. 为什么 alg_ext.py 用动态绑定而不是继承

V1 发布后大量用户在生产环境使用，V2 必须保证不破坏 V1 用户。

继承需要用户把代码里的 `SignRoundQuantizer` 改成 `V2Quantizer`，有破坏性。

动态绑定：用户只加一个参数 `enable_alg_ext=True`，V1 代码一行不改。

**代价：** 读代码更难，方法来源不明显。  
这是工程里常见的 tradeoff：用户体验简单 vs 代码理解复杂。

### 4. 为什么权重先放 CPU 再移到 GPU

```
7B 模型有 32 个 Block，全放 GPU 需要 14GB
每次只量化 1 个 Block，其余放 CPU，只需要 ~1GB

用完立刻移回 CPU：
mv_module_from_gpu(block)
torch.cuda.empty_cache()
```

这是**延迟加载**原则：用到时才准备，用完立刻释放。

### 5. 为什么 loss 最低点不是最后一步

SignSGD 步长固定，200步后参数在最优点附近振荡，不保证停在最优。

保存所有步中 loss 最低的参数 = 承认「不知道哪步最好，把最好的保存下来」。

这是工程务实态度：不假设算法收敛。

---

## 五层架构的本质

AutoRound 的架构本质上是一个**编译器**的结构：

```
前端（解析）：schemes.py 解析 "W4A16" -> 配置对象
IR 变换：    wrapper_block() 把 Linear -> WrapperLinear
优化：       SignSGD 200步优化 V/alpha/beta
后端（生成）：formats/ 把量化权重打包成 GPTQ/GGUF
```

不同层对应不同的变化原因：
- 数学层：论文改变时才变（量化公式更新）
- 工程层：PyTorch API 变化时才变
- 算法层：优化策略变化时才变
- 基础设施层：硬件平台变化时才变

四个不同的变化原因，就是要分四层的理由。

---

## 给重构者的建议

| 建议 | 说明 |
|------|------|
| 入口只做路由 | `__new__` 只检测和转发，不做业务 |
| 配置对象 > kwargs 堆叠 | 超过 5 个相关参数打包成 dataclass |
| 注册表模式 | 用装饰器注册，避免 if/elif 链 |
| 延迟导入 | 用到时才 import，减少循环依赖 |
| 基础设施和算法分离 | 设备管理不进算法，算法不知道设备 |
| 在变化的地方留插槽 | loss 函数、wrapper、优化器三个插槽 |

---

[返回文档地图](../../README.md)
