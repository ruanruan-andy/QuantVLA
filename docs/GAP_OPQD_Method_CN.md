# OOD-calibrated GAP-OPQD

## 1. 目标与数据协议

给定冻结的全精度 VLA teacher $\pi_T$，QuantVLA 产生 W4A8 student：

$$
\pi_Q^{(0)}=Q(\pi_T).
$$

当前版本直接在 LIBERO-Plus OOD calibration trajectories 上蒸馏，目标是快速检验少量目标域数据能否恢复量化模型性能。七个分类合计各取前 24 条，四个 suite 均匀分布：

| Suite | 每类数量 | 七类总数 |
|---|---:|---:|
| `libero_spatial` | 6 | 42 |
| `libero_goal` | 6 | 42 |
| `libero_object` | 6 | 42 |
| `libero_10` | 6 | 42 |
| 合计 | 24 | 168 |

训练和 eval 必须共同读取 `configs/libero_plus/first_24_per_category.json`，因此 task IDs 完全一致。这是 calibration-set evaluation，不是未见 OOD 泛化评测。

## 2. OOD student-controlled rollout

量化 student 在选定的 LIBERO-Plus task 上控制环境：

$$
a_t^Q=\pi_Q(o_t^{OOD},l),\qquad
o_{t+1}^{OOD}=\mathcal E_{OOD}(o_t^{OOD},a_t^Q).
$$

冻结 teacher 只在 student 已经到达的同一 OOD 状态上提供目标：

$$
a_t^T=\pi_T(o_t^{OOD},l).
$$

teacher action 不进入 `env.step`。teacher 与 student 使用相同 observation、语言和 flow 初始噪声，从而避免把随机 action sampling 方差误认为量化误差。

## 3. 即时量化分歧

GR00T 输出 action chunk，LIBERO 实际执行第 0 个动作。训练在 normalized action space 比较第 0 个连续 7D action：

$$
q_t=\frac{1}{7}\sum_{i=1}^{7}(a_{t,i}^T-a_{t,i}^Q)^2.
$$

不可导的 gripper 二值化只用于环境执行，不进入蒸馏 loss。

## 4. Temporal quantization risk

定义固定未来窗口内的折扣平均分歧：

$$
r_t^{(H)}=
\frac{\sum_{j=0}^{m_t}\gamma^j q_{t+j}}
{\sum_{j=0}^{m_t}\gamma^j},\qquad
m_t=\min(H,T-t).
$$

$q_t$ 描述当前动作分歧；$r_t$ 描述当前附近是否持续出现分歧。分母消除 rollout 尾部窗口较短造成的系统性低估。$r_t$ 是 future-disagreement proxy，不是严格的因果 return。

## 5. Critical-state weighted OPQD

在每条 rollout 内用 detached mean 归一化：

$$
\hat q_t=\frac{q_t}{\operatorname{mean}(q)+\epsilon},\qquad
\hat r_t=\frac{r_t}{\operatorname{mean}(r)+\epsilon}.
$$

状态权重为：

$$
w_t=\operatorname{clip}
\left(1+\alpha\hat q_t+\beta\hat r_t,w_{\min},w_{\max}\right).
$$

$w_t$ 停止梯度。固定 rollout 长度 $K$ 上的主损失为：

$$
\mathcal L_{OOD\text{-}OPQD}=
\frac{\sum_{t=1}^{K}w_t\lVert a_t^Q-a_t^T\rVert_2^2}
{\sum_{t=1}^{K}w_t}.
$$

当前实现不使用 adaptive horizon。

## 6. Clean IID anchor

为降低目标域过拟合和 clean 能力退化，另一个标准 clean LIBERO 环境服务产生短 student rollouts，并维护有限 IID replay。从中均匀采样状态进行无权重蒸馏：

$$
\mathcal L_{anchor}=\mathbb E_{o\sim D_{clean\ replay}}
\lVert\pi_Q(o)-\pi_T(o)\rVert_2^2.
$$

最终目标为：

$$
\mathcal L=
\mathcal L_{OOD\text{-}OPQD}
+\lambda_{anchor}\mathcal L_{anchor}.
$$

设置 $\lambda_{anchor}=0$ 可以关闭 clean anchor。

## 7. 可训练参数与部署

- FP teacher 完全冻结；
- W4A8 quantization pack、VLM backbone 和 action-head 原参数冻结；
- 仅训练 action-head/DiT attention 的 Q/K/V LoRA；
- rollout 阶段无梯度并缓存冻结 student backbone features；
- 反向阶段只重跑 action head；
- 部署只需要 W4A8 QuantVLA 和 LoRA adapter，不需要 teacher。

因此该方法属于 **target-OOD-calibrated, PTQ-initialized parameter-efficient post-training adaptation**，而不是纯 PTQ。

## 8. 必要对照与结论边界

| 方法 | OOD 数据 | $\alpha$ | $\beta$ | Clean anchor |
|---|---:|---:|---:|---:|
| 原始 QuantVLA | 否 | — | — | — |
| Uniform OOD distillation | 是 | 0 | 0 | 0 |
| OOD critical OPQD | 是 | 1 | 0 | 0 |
| OOD temporal OPQD | 是 | 1 | 1 | 0 |
| 完整 OOD GAP-OPQD | 是 | 1 | 1 | 0.1 |

当前训练与 eval 使用同一批 168 个 task，因此 success rate 用于快速验证适配和组件机制。它能够支持“少量已见目标域数据恢复量化性能”的结论，不能支持“完全未见 OOD 泛化提升”的结论。
