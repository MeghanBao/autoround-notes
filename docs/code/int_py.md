# data_type/int.py 逐行注释

> 量化数学公式的代码实现。commit `09bc7437`

## 函数总览

| 函数 | 行号 | 场景 |
|------|------|------|
| `search_scales` | L24–88 | V2 Scale 搜索 |
| `quant_tensor_asym` | L228–275 | 默认路径（sym=False）|
| `quant_tensor_sym` | L160–225 | 对称量化 |
| `quant_tensor_rtn_sym` | L125–157 | iters=0 纯 RTN |
| `quant_tensor_asym_wo_round` | L278–320 | bias/norm 专用 |
| `quant_tensor_sym_gptq` | L322–365 | GPTQ 格式 |

---

## quant_tensor_asym（L228–275）

```python
@register_dtype("int_asym")
def quant_tensor_asym(tensor, bits=4, group_size=-1,
                      v=0, min_scale=1.0, max_scale=1.0,
                      tensor_min=None, tensor_max=None,
                      q_scale_thresh=1e-5, **kwargs):

    # 步骤1：按 group 重排 [out,in] -> [n_groups, group_size]
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = int(2.0**bits) - 1    # 4-bit -> 15，无符号范围 [0,15]

    # 步骤2：每 group 的 min/max（有预计算值就用，省时间）
    if tensor_min is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp, wmax_tmp = tensor_min, tensor_max

    # 步骤3：应用可学习 clip 边界
    if isinstance(min_scale, torch.Tensor):   # SignSGD 训练时是 nn.Parameter
        wmin = wmin_tmp * min_scale            # beta × min(W)
        wmax = wmax_tmp * max_scale            # alpha × max(W)
    else:
        wmin, wmax = wmin_tmp, wmax_tmp        # iters=0 不学习

    # 步骤4：计算 scale 和 zero_point
    scale = ((wmax - wmin) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)  # 防 scale=0->NaN
    zp    = round_ste(-wmin / scale)
    scale = scale.unsqueeze(dim=-1)
    zp    = zp.unsqueeze(dim=-1)

    # 步骤5：量化（浮点->整数）
    int_w = round_ste(tensor / scale + v)   # V 改变取整方向，STE 穿透
    q     = torch.clamp(int_w + zp, 0, maxq)

    # 步骤6：反量化（整数->浮点）= Fake Quant
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape, pad_len)
    return qdq_result, scale, zp
```

**三个关键点：**
- `round_ste`：前向取整，反向梯度穿透（STE）
- `q_scale_thresh`：scale 不能为 0，防止 NaN
- `tensor_min/max`：初始化时预计算，训练中不重复统计

---

## quant_tensor_asym_wo_round（L278–320）

和上面完全相同，唯一区别：

```python
# 有 round_ste（普通量化）：
int_w = round_ste(tensor / scale + v)

# 无 round（bias/norm 专用）：
int_w = tensor / scale + v    # 连续可微，梯度直接流，不需要 STE
```

---

## quant_tensor_sym_gptq（L322–365）

和普通对称量化唯一区别：zp 固定为中点 maxq，而不是 0。

```python
zp = torch.full_like(scale, (maxq + 1) / 2)
# 4-bit：zp = 8（固定）
# 把有符号 [-8,7] 平移为无符号 [0,15]
# 数学等价，兼容 Marlin/ExLlamaV2 kernel 的存储格式
```

---

## search_scales（L24–88）

```python
def search_scales(data, bits, qw=None):
    # 用绝对值最大元素算初始 scale
    group_max = ...
    scales    = get_reciprocal(iscales)   # get_reciprocal = 1/x，比除法快

    # 算初始加权误差（基线）
    best_loss = ((scales * L - data)**2 * qw).sum(dim=-1)

    # 枚举候选 scale，更好就替换
    for _is in range(-search_min, search_min + 1):
        if _is == 0: continue            # 跳过初始值
        # 在初始 scale 附近微调：_is 是调整步数，step 是调整幅度
        loss = ((tmp_scales * tmp_L - data)**2 * qw).sum(dim=-1)
        replace_id = loss < best_loss
        if replace_id.any():
            scales[replace_id]    = tmp_scales[replace_id]
            best_loss[replace_id] = loss[replace_id]

    return scales   # 每行激活加权误差最小的 scale
```

---

[返回文档地图](../../README.md)
