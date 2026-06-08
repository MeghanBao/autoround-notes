# wrapper.py 逐行注释

> WrapperLinear：量化期间替换 nn.Linear 的核心组件。共 856 行。

---

## WrapperLinear 的作用

```
量化期间：
  nn.Linear  →  WrapperLinear
  （原始权重）   （V/alpha/beta + Fake Quant）

量化完成：
  WrapperLinear  →  nn.Linear
  （删除 V 等参数）  （权重已写入量化值）
```

---

## _init_tuning_params_and_quant_func（L140–193）

初始化所有可学习参数：

```python
def _init_tuning_params_and_quant_func(self):
    self.params = {}          # 收集所有可学习参数，供优化器使用
    p_dtype = torch.float32  # 所有参数用 FP32（梯度更稳定）

    orig_weight = getattr(self.orig_layer, "get_weight",
                          lambda: self.orig_layer.weight)()
    # 优先用 get_weight()：支持 meta 设备（只有形状没有数据）
    # 没有 get_weight() 才用 .weight

    # 初始化 V（rounding offset）
    self._init_params("value", p_dtype, weight_shape, 0,
                      enable_round_tuning)
    # 初始值 0 = 等价于 RTN；形状和权重相同（逐元素）

    # 初始化 alpha/beta（clip 边界）
    self._init_params("min_scale", p_dtype, scale_shape, 1.0, ...)
    self._init_params("max_scale", p_dtype, scale_shape, 1.0, ...)
    # 初始值 1.0 = 不裁剪
```

---

## _qdq_weight（L212–257）

```python
def _qdq_weight(self, value, min_scale, max_scale):
    if self.orig_layer.bits >= 16:
        return self.orig_layer.weight, None, None  # 不量化

    min_scale.data.clamp_(0.0, 1.0)   # 强制 beta in [0,1]，in-place
    max_scale.data.clamp_(0.0, 1.0)   # 强制 alpha in [0,1]
    # V2 的 WrapperLinearV2 把范围扩展到 [0, 2.0]

    weight = self.orig_layer.weight
    if weight.device.type == "meta":
        weight = self.orig_layer.get_weight().to(self.device)
    # meta 设备：只有形状没有数据，需要 get_weight() 真正加载

    weight_q, scale, zp = self.weight_quant_func(
        weight, v=value, min_scale=min_scale, max_scale=max_scale,
        imatrix=self.orig_layer.imatrix if hasattr(...) else None,
        init_scale=getattr(self, "init_scale", None),  # V2 专用
        ...
    )
    return weight_q, scale, zp     # fake-quant 权重（含误差的浮点）
```

---

## forward（L429–450）

```python
def forward(self, x):
    x = x.to(self.device)
    weight_q, *_ = self._qdq_weight(self.value, self.min_scale, self.max_scale)
    # self.value 随 SignSGD 更新，每步不同

    output = self.orig_forward(x, weight_q, bias).to(self.output_device)
    # y = x @ weight_q.T + bias
    # weight_q 含量化误差 -> 输出含误差 -> MSE loss 可以度量并优化

    return output
```

---

## unwrapper（L306–410）

```python
def unwrapper(self, best_params):
    # best_params = loss 最低那步的参数快照
    v         = best_params.get("value", 0.0)
    min_scale = best_params.get("min_scale", 1.0)
    max_scale = best_params.get("max_scale", 1.0)

    qdq_weight, scale, zp = self._qdq_weight(v, min_scale, max_scale)
    # 用最优参数做最后一次量化

    self.orig_layer.weight.data.copy_(qdq_weight)
    # in-place 覆盖权重（不创建新张量）

    self.orig_layer.scale = scale.to("cpu")
    self.orig_layer.zp    = zp.to("cpu")
    # 存储 scale/zp（导出时需要）

    return self.orig_layer   # 返回已量化的 nn.Linear
```

---

[返回文档地图](../../README.md)
