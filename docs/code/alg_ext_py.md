# alg_ext.py 逐行注释

> V2 算法扩展完整解析。commit `09bc7437`，共 872 行。

---

## 文件设计思路

alg_ext.py 不修改 V1 的任何代码。  
通过 `wrapper_autoround()` 把 V2 的方法动态挂载到量化器实例上。

```python
__all__ = ['wrapper_autoround']   # 对外只暴露这一个函数
```

---

## wrapper_autoround()（L39–56）—— V2 注入入口

```python
def wrapper_autoround(cls):
    # cls = SignRoundQuantizer 实例（参数名叫 cls 是历史遗留）

    # 无论是否开启 V2，都替换 hook
    cls._register_act_max_hook = types.MethodType(_register_act_max_hook_ext, cls)
    # types.MethodType(函数, 实例)：绑定后调用时 cls 自动作为 self 传入

    if cls.sym and cls.enable_alg_ext and ...:
        cls._get_loss = types.MethodType(_get_loss_ext, cls)   # 替换 loss
        setattr(cls, "wrapper_block", wrapper_block_v2)         # 替换 wrapper

    if cls.data_type.endswith("dq"):
        setattr(cls, "wrapper_block", dq_wrapper_block)  # GGUF 双重量化
```

---

## get_abs_top_percent_mask()（L59–75）

```python
def get_abs_top_percent_mask(x, percent=1.0):
    flat = x.view(-1)                        # 展平：全局排名需要一维
    k    = max(1, int(flat.numel() * percent / 1000))
    # k = 总数 × 0.1%（percent=1 → 0.001），至少 1 个

    _, idx = torch.topk(torch.abs(flat), k)  # 找绝对值最大的 k 个位置
    mask   = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True                          # 最大的 k 个位置标为 True
    return mask.view_as(x), ~mask             # (top_mask, inv_mask)
    # inv_mask=True 表示保留，inv_mask=False 表示排除（异常位置）
```

---

## _get_loss_ext()（L78–107）—— Loss Filtering

```python
def _get_loss_ext(self, output_q, current_output, indices, mse_loss, device):
    _, mask = get_abs_top_percent_mask(torch.abs(output_q - current_output))
    # _ 忽略 top_mask，只要 inv_mask（保留位置）
    # mask=True：正常位置，参与计算
    # mask=False：差异极大，乘 0 清零，不影响 loss

    with autocast_ctx:
        if self.attention_mask:
            loss = torch.mean(
                (|output_q - current_output| * attention_mask * mask) ** 2
            )
            # attention_mask：排除 PAD 位（填充符号，不是真实内容）
            # mask：排除差异 top-0.1%
        else:
            loss = torch.mean(
                (|output_q - current_output| * mask) ** 2
            )
    return loss
```

---

## WrapperLinearV2._init_tuning_params()

```python
def _init_tuning_params_and_quant_func(self):
    super()._init_tuning_params_and_quant_func()  # V1 初始化（V=0, alpha=beta=1）

    # 整理 imatrix 形状
    if hasattr(self.orig_layer, "imatrix"):
        imatrix = ...  # reshape/expand 成和 weight_reshape 相同的形状
    else:
        imatrix = 1.0  # MoE 某些层没有 imatrix，均等权重兜底

    # 根据数据类型选搜索方式
    if data_type.startswith("int"):
        self.init_scale = search_scales(weight_reshape, bits, imatrix)
    elif data_type.startswith("mx"):
        self.init_scale = mx_init(weight_reshape, bits, imatrix)
    elif data_type.startswith("nv"):
        self.init_scale = nv_init(weight_reshape, bits, imatrix)

    delattr(self.orig_layer, "imatrix")  # 搜索完彻底删除，释放内存

    # 替换量化函数（假设 init_scale 已算好，用简化版）
    self.weight_quant_func = quant_tensor_sym   # 本文件内的版本
```

---

## DQWrapperLinear —— 双重量化，每 10 步重搜

```python
def _qdq_weight(self, value, min_scale, max_scale, iter=None):
    min_scale.data.clamp_(0.5, 1.5)  # DQ 范围更窄，双重量化更敏感
    max_scale.data.clamp_(0.5, 1.5)

    if self._is_dq_path:
        need_search = (iter % 10 == 0) or (self.prev_scale is None)
        # 每 10 步重搜一次（搜索有循环、不可微，用 @inference_mode）

        if need_search:
            params = self._run_search(weight, value)  # 慢但准
            self.prev_scale = params["scale"]          # 缓存
        else:
            params = {"scale": self.prev_scale.detach(), ...}
            # detach()：推理模式产生的张量不能参与反向传播图
            # 复用缓存时必须 detach，否则 backward() 报错
            # 目的：只优化 V，不让梯度传回缓存的 scale
```

---

[返回文档地图](../../README.md)
