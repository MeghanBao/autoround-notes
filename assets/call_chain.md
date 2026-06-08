# 完整调用链

```
用户：
  ar = AutoRound("Qwen3-0.6B", scheme="W4A16")
  ar.quantize_and_save("./output")
         │
         ▼
autoround.py  AutoRound.__new__()
  解析 scheme：bits=4, group_size=128, sym=False
  detect_model_type -> 'llm'，iters=200
  返回 DataDrivenCompressor 实例（不是 AutoRound！）
         │
         ▼
compressors/data_driven.py  quantize()
  │
  ├─ [V2] alg_ext.py  _register_act_max_hook_ext()
  │       注册 imatrix hook -> 喂 128 条数据 -> 收集激活平方和
  │
  ├─ calib_dataset.py  get_dataloader()
  │       下载 128 条文本，tokenize -> DataLoader
  │
  └─ for each Transformer Block（共 32 个）:
       _quantize_blocks()
         │
         ├─ 浮点前向 -> 收集 reference_output（优化目标）
         │
         ├─ sign_round/quantizer.py  quantize_block()
         │    │
         │    ├─ [V1] wrapper.py  wrapper_block()
         │    │        Linear -> WrapperLinear（V=0, alpha=beta=1）
         │    │
         │    ├─ [V2] alg_ext.py  WrapperLinearV2.__init__()
         │    │        search_scales(weight, imatrix) -> self.init_scale
         │    │
         │    ├─ for step in range(200):   SignSGD 主循环
         │    │    ① Forward: WrapperLinear.forward(x)
         │    │         wrapper.py  _qdq_weight()
         │    │           data_type/int.py  quant_tensor_asym()
         │    │             W_int = round(W/scale + V + zp)  [STE]
         │    │             W_dq  = (W_int - zp) × scale     [fake-quant]
         │    │         y = x @ W_dq.T + bias
         │    │    ② Loss:
         │    │         [V1] MSE(y, reference_output)
         │    │         [V2] _get_loss_ext()：排除差异 top-0.1%
         │    │    ③ Backward: loss.backward()  STE 穿透 round()
         │    │    ④ SignSGD:
         │    │         p.grad = sign(p.grad) -> ±1
         │    │         p -= lr × sign(∇)，lr 线性衰减
         │    │    ⑤ 保存 loss 最低那步
         │    │
         │    └─ wrapper.py  unwrapper_block(best_params)
         │             写回整数权重，WrapperLinear -> nn.Linear
         │
         └─ 移回 CPU，清理显存
         │
         ▼
formats/  pack_weight（2个INT4->1个INT32）+ 写 config.json + safetensors
```
