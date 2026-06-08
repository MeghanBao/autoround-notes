# sign_round/quantizer.py 逐行注释

> SignSGD 主循环实现。commit `09bc7437`，共 573 行。

---

## quantize_block()（L200–400 约）

```python
def quantize_block(self, block, input_ids, input_others,
                   reference_output, loss_device, ...):

    # 阶段1：注入可学习参数
    wrapper_block(block, enable_minmax_tuning=..., iters=self.iters)
    # Linear -> WrapperLinear，注入 V=0, alpha=beta=1

    # 阶段2：[V2] Scale 搜索
    if self.enable_alg_ext:
        from auto_round.alg_ext import tune_v2_init
        tune_v2_init(block, ...)
        # WrapperLinearV2 初始化时调 search_scales，设 self.init_scale

    # 阶段3：收集参数，创建优化器
    round_params  = [m.params["value"] for m in WrapperLinear 层]
    minmax_params = [m.params["min_scale"], m.params["max_scale"] for ...]
    optimizer = SignSGD([
        {"params": round_params,  "lr": self.lr},        # V 组
        {"params": minmax_params, "lr": self.minmax_lr}  # alpha/beta 组
    ])
    scheduler = LinearLR(optimizer,
        start_factor=1.0, end_factor=0.0, total_iters=self.iters)
    # lr 从 5e-3 线性衰减到 0
    # 平均步长 0.0025，总位移 200×0.0025=0.5 = V 的半径

    # 阶段4：SignSGD 主循环
    best_loss, best_params = float("inf"), {}
    for step in range(self.iters):      # 默认 200 步
        total_loss = 0.0
        for indices in index_sampler:   # 每次采样一批校准样本

            output_q = block_forward(block, input_ids, input_others, indices)
            # WrapperLinear.forward() 自动用 fake-quant 权重

            loss = self._get_loss(output_q, reference_output, ...)
            # [V1] MSE(output_q, reference_output)
            # [V2] _get_loss_ext()：排除差异 top-0.1% 后的 MSE

            loss.backward()   # STE 穿透 round()，梯度到达 V/alpha/beta
            total_loss += loss.item()

        # SignSGD 核心：梯度 -> ±1
        for p in all_params:
            if p.grad is not None:
                p.grad = torch.sign(p.grad)
                # [0.003, -15.2, 0.0001] -> [+1, -1, +1]
                # 不管梯度大小，每步固定走 ±lr

        optimizer.step()      # V -= lr × sign(∇V)
        scheduler.step()      # lr 线性衰减
        optimizer.zero_grad() # 清零！不清零下一步梯度会累加

        # 保存 loss 最低那步（不一定是最后步）
        if total_loss < best_loss * (1 + self.dynamic_max_gap):
            best_loss   = total_loss
            best_params = collect_best_params(block)  # 深拷贝

    # 阶段5：写回权重，还原 nn.Linear
    unwrapper_block(block, best_params)
```

---

## SignSGD（sign_sgd.py）

```python
class SignSGD(torch.optim.Optimizer):
    def step(self, closure=None):
        for group in self.param_groups:
            lr       = group["lr"]       # 当前 lr（scheduler 已更新）
            momentum = group["momentum"] # 默认 0，一般不用

            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.data        # 此时已是 ±1（外部 sign() 处理过）

                if momentum > 0:
                    buf = ... * momentum + grad  # 历史梯度加权平均
                    grad = buf

                p.data.add_(grad, alpha=-lr)   # p = p - lr × grad
                # grad=±1，所以每步固定走 ±lr
```

> **注意：** `sign()` 操作在外部循环里执行，不在 `SignSGD.step()` 里。  
> 这样设计是为了让 `SignSGD` 保持通用性，可以和其他使用方式兼容。

---

[返回文档地图](../../README.md)
