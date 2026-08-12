# ShiftGuard-VLA 闭环验证报告

## 1. 最终结论

本轮优化得到了一个更可靠的离线指标，也找到一个在独立 trajectories 上显著降低动作偏差的 uniform-W4A8 候选，但该候选在全新任务的真实 rollout gate 上明显失败。因此：

- 轨迹平均风险是有效的 **block 定位与离线筛选诊断**；
- 它尚不是可靠的 **closed-loop success surrogate**；
- `b2_input_0p05` 被拒绝；
- 当前部署继续使用原始 QuantVLA W4A8；
- 不能声称 ShiftGuard-VLA 已提高 LIBERO-Plus success rate，也不能据此宣称达到 CCF-A 水平。

这是一次完整的负向闭环，而不是无效运行：它明确筛掉了 tangent、CVaR、mixed precision 和当前正交候选，并定位了下一版真正需要解决的问题——离线动作保持与环境状态转移/任务成功之间的错位。

## 2. 实验环境

- Repository：`/lumos-vePFS/suda/ruan/QuantVLA`
- Backbone：GR00T N1.5 LIBERO-Spatial post-train
- Baseline：QuantVLA W4A8 + ATM + OHB
- Shift benchmark：LIBERO-Plus 七类分布偏移
- Denoising steps：8
- Probe seed：20260811
- 计算节点：suzhou-C、D、H、I

## 3. 稀疏轨迹数据

Calibration tasks：649、299、1025、2151、41、1415、1766。

Validation tasks：650、300、1026、2152、42、1416、1767。

每条轨迹保存 stride-15 状态、gripper switch 和尾部 5 个状态，每条最多 24 个：

- calibration：7 trajectories，116 states，4 success / 3 failure；
- validation：7 trajectories，114 states，5 success / 2 failure；
- 合计：14 trajectories，230 states。

## 4. 风险指标验证

在 16 个 DiT blocks 上分别计算 calibration 与 validation 排序：

| 指标 | Spearman rho | permutation p | 判断 |
|---|---:|---:|---|
| normalized trajectory mean | **0.732** | **0.0018** | 通过，主指标 |
| raw trajectory mean | 0.668 | 0.0058 | 通过，但受动作尺度影响 |
| normalized tail CVaR | 0.356 | 0.170 | 不通过 |
| normalized initial risk | 0.106 | 0.688 | 不通过 |
| raw max risk | 0.062 | 0.825 | 不通过 |
| normalized critical CVaR | −0.056 | 0.829 | 不通过 |
| 原复合 trajectory risk | 0.288 | 0.274 | 不通过 |

此前直接 action-risk 的结果为 DiT rho=0.594、p=0.0159，LLM rho=−0.070、p=0.833；tangent/activation/Jacobian proxy 的最佳相关性约 0.2 且不显著。新版稀疏轨迹均值将 DiT 排序相关性提高到 0.732，但 CVaR 假设未成立。

全 W4A8 风险对 14 条轨迹 success/failure 的预测仍较弱：最佳 normalized max-risk AUC=0.733，但 point-biserial p=0.158。该结果已经预示它无法可靠替代闭环 success。

## 5. uniform-W4 正交搜索

按 calibration normalized trajectory mean 选择 blocks 0、15、3、2，构造 input/output/both × ±0.05，共 24 个候选。

Calibration top-4 进入 held-out validation。最终只有 `b2_input_0p05` 通过全部固定门槛：

| 指标 | QuantVLA | 候选 | 相对变化 |
|---|---:|---:|---:|
| validation trajectory mean | 0.0216075 | 0.0211465 | **−2.13%** |
| validation robust mean | 0.0226487 | 0.0222526 | **−1.75%** |
| validation max-risk mean | 0.0401660 | 0.0390965 | **−2.66%** |
| 改善类别数 | — | 5/7 | 通过 |
| 最坏类别回退 | — | +0.87% | 通过 |

七类相对变化：

- Language Instructions：−5.52%；
- Sensor Noise：−3.54%；
- Camera Viewpoints：−2.62%；
- Robot Initial States：−2.90%；
- Light Conditions：−2.31%；
- Objects Layout：+0.87%；
- Background Textures：+0.65%。

轨迹级配对统计：

- mean difference：−0.0004609；
- relative difference：−2.13%；
- 5/7 trajectories 改善；
- exact one-sided sign-flip p=0.03125；
- paired bootstrap 95% CI=[−0.0007698, −0.0001372]。

所以该候选的离线改善不是单个 observation 偶然值，但这仍只证明动作输出更接近 FP。

## 6. mixed-precision 反事实

逐层位宽能力被用于验证一个反事实：若 block 风险是可加的 bit utility，则相同预算下 risk top-k 应优于 low-k。结果不支持该假设：

| 候选 | 额外理论 weight storage | validation mean 变化 | p | 通过 |
|---|---:|---:|---:|---|
| risk top-1 W8 | 9 MiB | +0.112% | 0.820 | 否 |
| risk top-4 W6 | 18 MiB | +0.089% | 0.852 | 否 |
| risk top-2 W8 | 18 MiB | +0.149% | 0.898 | 否 |
| low-4 W8 | 36 MiB | −0.127% | 0.266 | 否 |
| risk top-4 W8 | 36 MiB | +0.115% | 0.891 | 否 |
| risk top-8 W6 | 36 MiB | +0.100% | 0.648 | 否 |

全量化模型存在层间误差抵消和非线性交互；单 block 高风险不等于“提高该层位宽后全模型一定改善”。因此 mixed precision 被从方法中删除，也避免与 ActQuant 的 action-guided bit allocation 重叠。

## 7. disjoint rollout gate

全新任务为 651、301、1027、2153、43、1417、1768，每类一个。评测在结论不可逆时按预设规则提前停止。

QuantVLA 已完成 6 条：

- Camera 651：失败；
- Robot 301：成功；
- Language 1027：成功；
- Light 2153：成功；
- Background 43：成功；
- Noise 1417：成功；
- 合计 5/6。

候选已完成 4 条：

- Camera 651：失败；
- Robot 301：失败；
- Language 1027：成功；
- Light 2153：失败；
- 合计 1/4。

在四条配对任务上：

- QuantVLA：3/4；
- candidate：1/4；
- candidate-only success：0；
- baseline-only success：2。

候选剩余三条即使全部成功，最多为 4/7；QuantVLA 已经取得 5 次成功。因此候选不可能满足“成功数严格高于 baseline”的 gate，提前停止不会改变接受结论。

最终：`accepted=false`，`selected_model=groot-quantvla-w4a8`。

## 8. 为什么离线改善却 rollout 失败

当前 risk 仍有四个结构性缺口：

1. 对全部 action dimensions 做均匀 MSE，但夹爪开关、末端位姿方向和接触阶段的重要性不同；
2. 比较的是同一 observation 上的动作，不度量该动作导致的下一个环境状态；
3. 对轨迹状态做平均会稀释少数决定成功/失败的因果动作；CVaR 在当前样本上又不稳定；
4. 保存的 trajectories 来自 baseline，候选 rollout 发生状态分布漂移后，离线数据不再覆盖其访问状态。

因此下一版若继续，应把目标从“更接近 FP action”升级为“更接近 FP 导致的状态转移或任务进度”。可检验方向包括：

- action dimensions 的物理量纲与阶段加权；
- gripper decision disagreement；
- one-step environment transition discrepancy；
- 使用 candidate-induced states 做 DAgger-style risk recollection；
- 用 value/progress estimator 评估动作对任务完成度的影响。

这些方向需要新的 calibration 与 validation，不能在当前 split 上继续调参。

## 9. 对 CCF-A 的客观判断

当前版本不能投 CCF-A 主会作为“有效方法”，原因不是文档不完整，而是核心候选未通过真实闭环 gate，且目前只有一个 backbone、一个 suite 和少量 rollout。

值得保留的研究结果是：

- 一个显著、可迁移的稀疏 trajectory block-risk 诊断；
- 对 tangent、CVaR、mixed precision 和局部旋转的系统闭环否证；
- 一套防止 proxy 改善被误报为 success 提升的选择协议。

若将论文方向改为“VLA 量化在分布偏移下的 proxy–control gap 与可靠评测”，这些负结果可能形成有价值的分析工作；若坚持方法论文，则必须先解决状态转移级风险，再重新完成独立闭环验证。

## 10. 关键产物

- Trajectories：`output/shiftguard_v2/trajectories/`
- 16-block probes：`output/shiftguard_v2/block_probes/`
- 指标分析：`output/shiftguard_v2/trajectory_risk_validation_v2.json`
- 24-candidate search：`output/shiftguard_v2/rotation_search_v3/`
- 冻结候选统计：`output/shiftguard_v2/rotation_search_v3/selected_stats.json`
- Mixed precision：`output/shiftguard_v2/mixed_precision/analysis.json`
- Rollout logs：`output/shiftguard_v2/logs/rollout_gate/`
- Rollout gate decision：`output/shiftguard_v2/rollout_gate/partial_gate_result.json`
- Final deployment selection：`output/shiftguard_v2/deployment_selection.json`
