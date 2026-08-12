# ShiftGuard-VLA：面向分布偏移的轨迹风险引导 VLA 量化补偿

## 1. 核心思想

QuantVLA 在常规 LIBERO 上有效，并不保证其量化误差在相机、机器人初态、语言、光照、纹理、传感器噪声和物体布局变化下仍然稳定。ShiftGuard-VLA 不重新训练策略，也不增加在线模块，而是用少量无标签、闭环采集的 shifted trajectories 回答两个问题：

1. 哪些 DiT blocks 的量化误差在不同 shift 和不同轨迹间具有可迁移的高风险排序？
2. 能否只在这些 blocks 的既有正交量化坐标附近做小幅补偿，同时保持 uniform W4A8 和原推理图？

方法最终是一个带回退机制的离线模型选择流程：只有候选依次通过 calibration、独立 trajectory validation 和全新任务 rollout gate，才允许部署；任何一级失败都回退到原始 QuantVLA。

## 2. 问题定义

给定浮点策略 \(\pi_{\mathrm{FP}}\)、QuantVLA W4A8 策略 \(\pi_Q\)，以及七类分布偏移 \(d\in\mathcal D\)，目标是在不训练模型、不改变位宽、不增加在线算子的约束下，降低 shifted closed-loop states 上的动作失真，并最终保持或提高任务成功率。

对轨迹 \(\tau_i=\{o_{i,t}\}_{t=1}^{T_i}\)，固定相同的 diffusion noise \(\epsilon_{i,t}\)，定义成对归一化动作风险：

\[
r(o_{i,t};Q)=
\frac{
\operatorname{MSE}\!\left(
a_Q(o_{i,t},\epsilon_{i,t}),
a_{\mathrm{FP}}(o_{i,t},\epsilon_{i,t})
\right)
}{
\operatorname{mean}\!\left(a_{\mathrm{FP}}(o_{i,t},\epsilon_{i,t})^2\right)+\varepsilon
}.
\]

归一化消除了不同任务动作幅值的尺度混杂；共享 noise 消除了 diffusion 随机性。该指标只用于低成本定位和选择，不能替代最终 closed-loop success。

## 3. 稀疏闭环轨迹校准

每类 shift 采集少量完整 rollout，但只保存稀疏 observations：固定每 15 个 policy steps 保存一次，同时保留 gripper switch 和末尾 5 个状态，每条轨迹最多 24 个状态。这样能覆盖早期感知、接近、抓取、搬运和放置阶段，而不需要保存全部视频帧。

对任意量化模型 \(Q\)，轨迹风险采用简单平均：

\[
R_i(Q)=\frac{1}{|\mathcal S_i|}\sum_{t\in\mathcal S_i}r(o_{i,t};Q).
\]

当前数据明确否定了“尾部 CVaR 必然更适合闭环”的先验：在 16 个 DiT blocks 上，calibration 到 validation 的排序相关性为：

- 归一化 trajectory mean：Spearman \(\rho=0.732\)，permutation \(p=0.0018\)；
- initial-state risk：\(\rho=0.106\)，\(p=0.688\)；
- tail CVaR：\(\rho=0.356\)，\(p=0.170\)；
- critical-state CVaR：\(\rho=-0.056\)，\(p=0.829\)。

因此主指标只使用 trajectory mean；最大轨迹风险仅作为安全约束，不与均值任意加权成新的复合分数。

## 4. 可迁移的 DiT block 风险定位

对每个 DiT block \(b\)，构造单 block 探针 \(Q_b\)：只激活该 block 的 W4A8，其他被测量化层保持 FP。其 calibration 风险为：

\[
S_b^{\mathrm{cal}}=\frac{1}{N_{\mathrm{cal}}}
\sum_{i=1}^{N_{\mathrm{cal}}}R_i(Q_b).
\]

在互不重叠的 validation trajectories 上计算 \(S_b^{\mathrm{val}}\)，并先验证完整 block 排序是否可迁移。只有当预注册的排序检验通过，例如 \(\rho>0.5\) 且 permutation \(p<0.05\)，该分支才允许进入后续优化。

当前结果是：DiT 的归一化 trajectory mean 通过；LLM block、tangent/Jacobian proxy、initial-state risk 和 CVaR 均未通过。因此最终方法只优化 DiT，不声称这些失败指标有效。

## 5. 风险局部化的正交补偿

QuantVLA 已为每个 64-channel group 提供正交输入/输出变换。ShiftGuard 选取 calibration 风险最高的 top-\(K\) DiT blocks，并在原坐标附近构造小型 Givens rotation bank \(G(\theta;\mathcal M)\)：

\[
R'_{\mathrm{in}}=R_{\mathrm{in}}G(\theta;\mathcal M),
\qquad
R'_{\mathrm{out}}=G(\theta;\mathcal M)R_{\mathrm{out}}.
\]

其中 \(\mathcal M\) 是固定随机种子生成的不相交通道配对，\(\theta\) 为小角度。由于 \(G^\top G=I\)，更新保持正交；最终矩阵预先写入 pack，不增加运行时矩阵乘法。

当前冻结搜索空间为：

- blocks：按 calibration trajectory mean 排名前四的 \(\{0,15,3,2\}\)；
- scope：input、output、both；
- angle：\(-0.05,+0.05\)；
- 每次只修改一个 block，共 24 个候选。

不允许在看过 held-out validation 后继续加角度或组合多个 blocks；若要做 fine search，必须新增独立 validation split。

## 6. 三阶段选择与回退

### 6.1 Calibration shortlist

对候选 \(Q\) 计算：

\[
J_{\mathrm{cal}}(Q)=
\operatorname{mean}_i R_i(Q)+
\kappa\operatorname{std}_i R_i(Q),\qquad \kappa=0.5.
\]

同时要求 calibration 的平均最大轨迹风险不超过 baseline 的 1.02 倍。在安全候选中取 \(J_{\mathrm{cal}}\) 最低的 top-4；validation 在此之前不可参与排序。

### 6.2 Held-out trajectory gate

shortlist 候选必须同时满足：

1. validation mean 低于 QuantVLA；
2. validation robust mean 低于 QuantVLA；
3. 平均最大轨迹风险不超过 QuantVLA 的 1.02 倍；
4. 七类 shift 中至少五类改善；
5. 任一类别相对回退不超过 5%。

若多个候选通过，选择 validation robust mean 最低者；若无人通过，输出 QuantVLA。

本次冻结候选为 `block 2 / input / +0.05`。相对 QuantVLA，其 held-out trajectory mean 下降 2.13%，robust mean 下降 1.75%，最大风险下降 2.66%，七类中五类改善，最坏类别回退 0.87%；七条轨迹的 exact one-sided sign-flip \(p=0.03125\)，paired bootstrap 95% CI 不跨零。

### 6.3 Disjoint rollout gate

离线通过仍不等于策略成功率提高。冻结候选必须在未参与上述任何选择的任务上，与 QuantVLA 使用相同 task、initial state 和随机种子做配对 rollout。pilot 采用严格门槛：候选成功数必须高于 QuantVLA，且无运行错误；否则仍回退 QuantVLA。正式论文应扩大样本并预注册 non-inferiority 或 superiority margin，不能依据 test 结果修改门槛。

本次 gate 在结论已不可逆时提前停止：QuantVLA 已完成 6 条并成功 5 条；候选已完成 4 条并成功 1 条。在四条已配对任务上，QuantVLA 为 3/4、候选为 1/4，baseline-only success=2、candidate-only success=0。即使候选剩余三条全部成功，其最高也只有 4/7，仍低于 QuantVLA 已取得的 5/7，因此候选被拒绝，最终部署保持 QuantVLA。

## 7. 算法

```text
Input: FP policy, QuantVLA W4A8 pack,
       disjoint shifted calibration / trajectory-validation / rollout-gate splits

1. Collect sparse states along a few closed-loop trajectories per shift.
2. For every candidate block b:
     enable only block b as W4A8;
     compute same-noise normalized action risk on every sparse state;
     aggregate by trajectory mean.
3. Test calibration-to-validation rank transfer.
4. If the branch/metric fails, remove it; never optimize with an unstable score.
5. Rank the valid DiT blocks and freeze top-K.
6. Generate single-block local Givens candidates at fixed scopes and angles.
7. Use calibration robust mean plus max-risk constraint to form a top-M shortlist.
8. Apply held-out mean / robust / domain / maximum-risk gates.
9. Freeze the best eligible pack; do not tune again on this validation set.
10. Run paired rollouts on a new gate split.
11. Accept only if the rollout rule passes; otherwise deploy QuantVLA.

Output: a uniform W4A8 pack with no extra online operator, or safe rollback.
```

## 8. 与现有工作的边界

- 与 QuantVLA 相比：位宽、量化布局、ATM/OHB 和推理结构不变；新增的是 shifted trajectory 风险定位、局部正交补偿和严格回退选择。
- 与 ActQuant 相比：本方法不做 mixed-precision bit allocation，不需要 ground-truth actions，目标是 W4A8 在分布偏移下的闭环泛化，而不是 sub-4-bit 压缩。
- 与 Ω-QVLA 相比：本方法不提出全局 SVD-Hadamard 或 per-denoising-step scaling；只在现有 QuantVLA pack 的少数、经跨域验证的风险 blocks 周围做局部补偿。

混合精度消融也支持这一边界：top-k W6/W8 在 held-out 上均未改善，说明单 block 风险不是可直接相加的 bit utility；因此 mixed precision 被删除，而不是包装成贡献。

## 9. 当前可主张与不可主张

当前证据支持：

- 稀疏闭环 trajectory mean 比 initial observation、tangent 和 CVaR 更能迁移 DiT block 风险排序；
- 该排序能将局部正交搜索集中到有效区域；
- 冻结候选在当前 held-out trajectories 上获得小但一致的动作保持改善；
- 方法保持 uniform W4A8，无额外在线结构和位宽开销。

当前证据尚不支持：

- 已显著提高最终 LIBERO-Plus success rate；
- 对所有 VLA、suite 和量化配置都有效；
- 仅凭一个 GR00T checkpoint、一个 suite 和少量 rollout 已达到 CCF-A 证据强度。

方法流程已经形成闭环，但当前优化候选仍未跨越 success proxy 与真实闭环行为之间的鸿沟。它现在应被视为一个有统计信号但最终候选失败的研究原型，而不是有效的新量化方法。CCF-A 级论文必须先提出并验证能预测闭环失败的选择信号，再完成四个 LIBERO suites、至少第二个 VLA backbone、完整七类测试、强基线和多随机种子验证。
