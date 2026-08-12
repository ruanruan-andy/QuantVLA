# ShiftGuard-VLA 实验设计

## 1. 实验目标

实验必须分别回答四个问题，不能只给最终平均成功率：

1. QuantVLA 的量化误差是否会在七类 LIBERO-Plus shift 下呈现不同于 ID 的泛化退化？
2. 稀疏闭环 trajectory mean 是否比单初始帧、weight/activation MSE、tangent/Jacobian 和 CVaR 更能预测 held-out block 风险？
3. 风险局部化的正交补偿是否在 uniform W4A8、相同模型大小下稳定优于 QuantVLA？
4. 离线 action preservation 的改善能否转化为配对 closed-loop success，而非仅是 proxy 改善？

## 2. 数据划分与防泄漏

七类 shift 为 Camera Viewpoints、Robot Initial States、Language Instructions、Light Conditions、Background Textures、Sensor Noise 和 Objects Layout。

每个 suite、每个类别建立互不重叠的四级 manifest：

- `calibration`：只计算 block 风险与候选 shortlist；
- `trajectory_validation`：只验证指标迁移和候选接受门槛；
- `rollout_gate`：只决定冻结候选是否允许进入正式测试；
- `test`：只报告最终结果，禁止调参或修改门槛。

当前 `libero_spatial` pilot 使用：

| split | Camera | Robot | Language | Light | Background | Noise | Layout |
|---|---:|---:|---:|---:|---:|---:|---:|
| calibration | 649 | 299 | 1025 | 2151 | 41 | 1415 | 1766 |
| trajectory validation | 650 | 300 | 1026 | 2152 | 42 | 1416 | 1767 |
| rollout gate | 651 | 301 | 1027 | 2153 | 43 | 1417 | 1768 |

正式实验中，用户设定的“每类前 100 条 rollout”应固定为 test manifest；calibration/validation/gate 必须使用这 100 条之外的任务，或先预注册不重叠划分。所有方法共享 task、initial state、环境 seed 与 policy seed。

## 3. 当前 pilot 协议

### 3.1 轨迹采集

- 每类每个 split 1 条闭环轨迹；
- 每 15 个 policy steps 保存一次；
- 额外保留 gripper switch 与最后 5 个状态；
- 每条最多 24 个状态；
- calibration 共 116 states，validation 共 114 states；
- 8 个 denoising steps，随机种子 20260811。

### 3.2 Block probe

- 分别探测 16 个 DiT blocks；
- probe 时仅目标 block 为 W4A8，其他被测量化层为 FP；
- FP/quant 使用完全相同 diffusion noise；
- 主指标为 normalized trajectory mean；
- calibration→validation 用 Spearman 和 5,000 次 permutation test。

### 3.3 候选搜索

- top blocks：0、15、3、2；
- scope：input/output/both；
- angle：−0.05/+0.05；
- 24 个单 block 候选；
- calibration 安全过滤后只验证 top-4；
- validation 门槛在运行前固定，禁止事后放宽。

### 3.4 小规模 rollout gate

- 每类 1 个全新 task，共 7 个；
- QuantVLA 与冻结候选严格配对；
- pilot gate：候选成功数必须严格高于 QuantVLA；
- 若相同或更差，部署仍回退 QuantVLA；
- 该结果只决定继续/停止，不作为正式论文主表。

## 4. 正式论文主实验

### 4.1 模型与 benchmark

最低可接受配置：

- GR00T N1.5：LIBERO-Spatial、Object、Goal、Long 四个 suites；
- 第二个结构不同的 VLA，例如 \(\pi_{0.5}\) 或 OpenVLA-OFT；
- 每个 suite 的七类 shift 各 100 条 test rollout；
- 至少 3 个 policy/environment seeds；
- 若算力受限，先完成四 suites × 一个模型，再将第二模型作为外部泛化验证。

只在 `libero_spatial` 上截取少量任务可以用于 method screening，不能作为 CCF-A 主结果。

### 4.2 对比方法

主表至少包含：

- FP16/BF16；
- 原始 QuantVLA W4A8；
- QuantVLA + 随机 block、等规模 Givens 搜索；
- QuantVLA + initial-observation risk；
- QuantVLA + trajectory risk（ShiftGuard）；
- 可复现时加入 Ω-QVLA、ActQuant；若位宽不同，分别报告同位宽质量与 success–memory Pareto，不能直接混为同一列优劣。

所有 W4A8 方法必须使用相同量化布局、group size、ATM/OHB、calibration 数量和推理 kernel。

## 5. 消融实验

### A. 风险指标

- raw MSE vs action-energy normalized MSE；
- initial state vs uniform trajectory mean；
- max、tail CVaR、critical-state CVaR；
- tangent/Jacobian、activation MSE；
- DiT vs LLM branch。

主指标选择必须由 calibration→validation 排序迁移决定，而不是由最终 test 表现决定。

### B. 轨迹采样成本

- 每类轨迹数：1/2/5/10；
- stride：5/15/30；
- 是否加入 gripper/tail states；
- 状态总数与排序 \(\rho\)、候选收益、GPU 小时的关系。

### C. 搜索设计

- top-\(K\)：1/2/4/8；
- block：风险 top-k vs bottom-k vs random-k；
- scope：input/output/both；
- angle：±0.02/±0.05/±0.10，但每一轮新增搜索都必须有新的 validation split；
- 单 block vs 多 block 组合；
- 无 safety gate、无 rollout gate 的反事实结果。

### D. 被否定路线

- v1 initial-observation candidate；
- tangent risk；
- trajectory CVaR；
- 风险 top-k mixed precision W6/W8。

这些负结果应保留，以证明最终设计来自闭环验证，而非只展示成功配置。

## 6. 评价指标与统计

### 6.1 主指标

- 每类 success rate；
- 七类 macro average；
- worst-category success；
- 相对 ID 的 generalization gap；
- 四 suites 平均与逐 suite 结果。

### 6.2 辅助指标

- normalized trajectory action discrepancy；
- trajectory robust mean；
- maximum-risk mean；
- calibration→validation Spearman；
- 模型静态大小、实际峰值显存、端到端 latency、吞吐与能耗。

当前代码使用 fake quantization，不能用其 PyTorch 显存或 latency 声称真实 INT4 加速；部署效率必须在相同真实低比特 kernel 上测量。

### 6.3 显著性

- 同 task 配对 success：McNemar exact test；
- success 差值：task-level paired bootstrap 95% CI；
- action discrepancy：trajectory-level paired bootstrap 与 exact sign-flip；
- 多类别检验：Holm 校正；
- 同时报告绝对百分点、相对变化、CI 和 \(p\)，不只报告最好 seed。

建议正式 superiority 主张要求：宏平均 success 的 paired 95% CI 下界大于 0；若目标是“不降性能”，则提前给定 non-inferiority margin，例如 −1 pp，并按功效分析确定样本量。

## 7. 固定接受规则

候选进入 test 前必须依次满足：

1. block 排序在独立 split 上达到 \(\rho>0.5,p<0.05\)；
2. held-out trajectory mean 与 robust mean 均优于 QuantVLA；
3. 最大风险不回退超过 2%；
4. 至少 5/7 shift 改善，任一类别回退不超过 5%；
5. 独立 rollout gate 达到预注册的 superiority 或 non-inferiority 条件；
6. 无 NaN、服务错误、task 丢失或样本不配对。

任一失败即回退，不允许从 test 中挑选角度、block 或 checkpoint。

## 8. 当前已完成结果与下一阶段

已完成：

- 14 条稀疏轨迹、230 个状态；
- 16 个 DiT blocks 的 calibration/validation 风险验证；
- 24 个 uniform-W4 正交候选；
- 6 个 mixed-precision 反事实候选；
- 冻结候选的 held-out trajectory gate。

关键结果：

- normalized trajectory mean：\(\rho=0.732,p=0.0018\)；
- 冻结候选 `b2_input_0p05`：validation mean −2.13%，robust mean −1.75%，max risk −2.66%，5/7 类改善；
- paired sign-flip \(p=0.03125\)，bootstrap 95% CI 为 \([-7.70\times10^{-4},-1.37\times10^{-4}]\)；
- mixed precision 全部未过 gate，已从主方法删除。

rollout gate 已在结论不可逆时提前停止：QuantVLA 已完成 6 条、成功 5 条；候选已完成 4 条、成功 1 条。四条配对任务中 baseline-only=2、candidate-only=0；候选即使剩余三条全成功也无法超过 baseline，因此候选被拒绝并回退。

下一阶段不应直接运行每类 100 条 test，也不应继续在同一 validation 上搜索角度。应先分析为何 action preservation 与 success 错位，并引入能够反映动作方向、gripper decision、阶段进度或环境状态转移的闭环选择信号；只有新的候选在全新 validation 与 rollout gate 上通过后，才进入正式 test。随后再扩展到其余三个 suites 和第二个 VLA backbone。

## 9. CCF-A 投稿判断标准

仅凭当前 pilot 不够 CCF-A。接近 CCF-A 的最低证据应同时具备：

- 清楚区别于 QuantVLA、ActQuant 和 Ω-QVLA 的 shift-trajectory 问题定义；
- 两个以上 VLA backbones 或充分的跨结构泛化证据；
- 四个 suites、七类 shift、完整 paired rollouts；
- 相同预算强基线、真实 kernel 效率和完整消融；
- success 改善具有统计显著性，而不是只改善 action discrepancy；
- 失败边界、回退率和 calibration 成本透明可复现。

若最终只在 GR00T-Spatial 的少量任务上成立，应定位为有价值的 workshop/短文或泛化诊断工作，而不是强行包装成 CCF-A 完整方法论文。
