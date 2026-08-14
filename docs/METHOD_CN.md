# QuantVLA-OPQD 方法说明

## 1. 研究对象

本项目比较同一组 GR00T N1.5 suite checkpoint 的三种运行方式：

| 方法名 | 权重/激活 | 是否训练 | 作用 |
|---|---|---:|---|
| `fp16` | FP16 | 否 | 全精度参考线 |
| `quantvla` | W4A8 fake quant | 否 | QuantVLA 量化参考线 |
| `quantvla-opqd` | W4A8 fake quant + action-head LoRA | 是 | 用 OPQD 恢复关键状态的量化动作能力 |

三个方法使用相同的基础 checkpoint、suite、denoising steps、任务列表、初始状态和 policy seed。`quantvla-opqd` 是论文与输出中的规范名称；代码中的 `gap_opqd` 是该实现的历史模块名。

## 2. QuantVLA W4A8 路径

入口 `run_quantvla.sh` 通过环境配置加载 `gr00t/quantization/duquant_layers.py` 与 `duquant_preprocess.py`。当前默认配置为：

- 权重 4 bit、激活 8 bit；
- block size 64；
- 激活标定 percentile 99.9，标定 32 steps；
- language model 的 Q/K/V/O 与 MLP linear，以及 DiT 的 FFN linear 进入 W4A8；
- vision、RADIO、norm、embedding、LM head 和 DiT `attn1` 保持浮点；
- suite 专属 ATM/OHB 系数来自 `atm_alpha_beta_{spatial,goal,object,long}.json`；
- suite 专属 DuQuant pack 位于 `model/quantvla/groot-n1.5/<suite>/duquant_pack/`。

DuQuant 对 linear 权重执行 block rotation/重排准备，前向时对变换后的权重和激活做对称 fake quant。ATM 和 OHB 对 DiT 内部激活/attention head 做预先标定的尺度补偿。`run_quantvla.sh` 明确固定目标层和排除层，避免实验之间因环境变量残留改变范围。

这里的 W4A8 是浮点张量上的量化误差模拟，可用于成功率与动作误差比较；进程显存不能当作真实 INT4 kernel 的部署显存。

## 3. OPQD 训练数据流

训练入口为 `train_quantvla_opqd.sh`，核心实现位于：

```text
scripts/train_gap_opqd.py
gr00t/experiment/gap_opqd.py
```

一次训练包含两个冻结策略：

- teacher：未量化 FP16 GR00T；
- student：QuantVLA W4A8 GR00T，仅 action head 中注入的 LoRA 可训练。

LIBERO 与训练环境依赖分离：

1. `libero_plus_env_service.py` 在 `libero_test` 环境提供 OOD calibration rollout；
2. `libero_iid_env_service.py` 在 `libero_test` 环境提供 clean LIBERO anchor；
3. trainer 在 `groot_test` 环境加载 teacher/student，通过 localhost ZMQ 与两个环境服务交互；
4. simulator rollout 不反传梯度，trainer 缓存冻结 backbone feature、action input、初始噪声和 teacher action，仅重新前向 action-head LoRA。

默认 OOD 数据为 `first_24_per_category.json`：每个 suite/category 取 task index 最小的 6 条，四个 suite 合计每类 24 条、七类共 168 条。

## 4. OPQD 目标

对 rollout 中状态 `t`，student 与 teacher 产生动作块。环境实际执行动作块的第 0 个动作，只比较前 7 个 LIBERO 动作维度。瞬时量化分歧为：

```text
q_t = MSE(a_student[t, 0, :7], a_teacher[t, 0, :7])
```

长度为 `H`、折扣为 `γ` 的未来风险为：

```text
r_t = Σ(j=0..H) γ^j q_(t+j) / Σ(j=0..H) γ^j
```

轨迹末端窗口按实际剩余长度截断并重新归一化，避免末端状态系统性低估。令 `q̄_t`、`r̄_t` 分别为除以轨迹均值后的非负量，则关键状态权重为：

```text
w_t = clip(1 + α q̄_t + β r̄_t, w_min, w_max)
```

`q_t`、`r_t` 和 `w_t` 全部 detach，不通过权重构造反传。OOD 蒸馏损失按权重归一化：

```text
L_opqd = Σ_t w_t L_distill(t) / Σ_t w_t
```

同时从 clean LIBERO replay 中抽取状态，加入 IID 保持项：

```text
L = L_opqd + λ_anchor L_anchor
```

默认参数为 `H=4`、`γ=0.9`、`α=1`、`β=1`、`w∈[1,5]`、`λ_anchor=0.1`、LoRA rank 16/alpha 32。

## 5. Checkpoint 与推理

训练 checkpoint 结构为：

```text
checkpoint-000100/
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── trainer_state.pt
```

推理时先按 QuantVLA 路径构造 W4A8 student，再加载 action-head LoRA adapter。`eval_quantvla_opqd.sh --checkpoint` 同时接受 checkpoint 目录和其中的 `adapter/` 目录，不修改也不删除训练 checkpoint。

## 6. 结论边界

- 当前 OPQD 训练和默认 LIBERO-Plus eval 使用同一 first-24 task 集，属于 calibration-set evaluation，可证明量化性能恢复，不能单独证明完全未见任务泛化。
- 未见泛化结论应另建不重叠 manifest，或在标准 LIBERO/更大 LIBERO-Plus holdout 上评测。
- 三方法比较必须保持 task IDs、初始状态、policy seed、基础 checkpoint 和 denoising steps 一致。
- W4A8 为 fake quant；速度和显存结论需要真实量化 kernel 的独立部署实验。
