# QuantVLA-OPQD 方法

## 1. 研究问题与对照

目标是在不改动 GR00T N1.5 主体结构的前提下，修复 W4A8 QuantVLA 在 LIBERO-Plus 分布偏移状态上的动作误差。

| 方法 | 权重 | 训练 | 作用 |
|---|---|---|---|
| FP16 | 全精度 | 否 | 精度参考线 |
| QuantVLA | W4A8 + ATM + OHB | 否 | 原始量化方法 |
| QuantVLA-OPQD | QuantVLA + action-head LoRA | 是 | 本文方法 |

OPQD 不使用离线 expert action 标签，也没有 Uniform-KD 分支。教师监督来自 FP16 模型在学生实际访问状态上的在线预测。

## 2. Shared-560 目标域协议

`scripts/build_libero_plus_shared_manifest.py` 在每个 suite、七种 OOD category 内按 task index 排序并取前 20 条：

| Manifest | 每 suite/category | 每 suite | 四 suite |
|---|---:|---:|---:|
| Shared-560 first-20 | 20 | 140 | 560 |

manifest 保存显式 task IDs、category、difficulty 与 `train_eval_relation=same_task_ids`。OPQD train 和三种方法 eval 使用完全相同的 560 个 task ID 和 initial state 0；train seed 为 0/1/2，三种方法的 eval seed 固定为 2026。

因此本文研究的是目标域 same-task transductive adaptation：基础模型从 Standard LIBERO 迁移到 LIBERO-Plus OOD 分类，OPQD 可以在目标任务上采集无标签 rollout。它不代表对未见 LIBERO-Plus task ID 的 held-out-task generalization。

## 3. 完整 student-on-policy rollout

每个训练 episode：

1. 在 LIBERO-Plus Shared-560 重置一个 task；
2. 学生 QuantVLA 执行动作并决定后续访问状态；
3. 在每个访问状态上，FP16 教师与学生使用同一份 diffusion initial noise；
4. 记录教师/学生动作，直到成功或 suite horizon。

默认 horizon 为 spatial 220、object 280、goal 300、libero_10 520。`episode_success` 是环境真实终止结果，不再用短 rollout 是否终止来冒充成功率。

## 4. 状态评分

对 timestep (t)，只比较实际执行的 action chunk 第一个动作、前七个连续维度：

\[
q_t=\frac{1}{7}\lVert a_t^S-a_t^T\rVert_2^2.
\]

再计算未来局部风险，默认 (H=4,\gamma=0.9)：

\[
r_t=\frac{\sum_{j=0}^{H}\gamma^j q_{t+j}}
{\sum_{j=0}^{H}\gamma^j},
\]

轨迹尾部按实际剩余长度截断并重新归一化。(q_t) 表示当前量化动作偏差，(r_t) 表示短期内持续出现偏差的程度。

## 5. 四阶段稀疏选择

将实际轨迹等比例分为 Early、Mid-1、Mid-2、Late 四段。在每段内部对 (q,r) 做带平均 tie rank 的 percentile ranking：

\[
s_t=\alpha\,rank(q_t)+\beta\,rank(r_t),\qquad \alpha=\beta=1.
\]

每阶段选择：

- 2 个最高 (s_t) 的 priority states；
- 2 个可复现随机 states；
- 合计 4 个，四阶段固定为 16 个。

`min_temporal_gap=4` 是目标间隔：选中 10 后，下一候选至少为 14，避免相邻帧重复监督。selector 在每个阶段内依次尝试 gap 4、3、2、1，直到该阶段恰好补齐 2 个 priority 和 2 个 random；候选绝不跨阶段补位、不重复。轨迹少于 16 个状态时直接报错，不能产生伪完整样本。输出记录每阶段实际 gap、数量和 `selection_valid`，因此固定配额和约束放宽都可审计。

这种设计不是原始稠密 OPD 的复刻，而是本文的计算受限稀疏蒸馏假设：阶段覆盖负责轨迹多样性，高分状态负责困难性，随机状态降低纯 top-k 的选择偏差。

## 6. 蒸馏目标与更新

入选状态的权重为：

\[
w_t=clip\left(\frac{s_t}{mean(s)},0.5,2.0\right),
\]

随后归一化使权重和为 1。主损失仍比较共享 noise 下、实际执行动作位置的七维教师/学生 MSE。每个 episode 对同一批 16 个状态做 5 次 optimizer update。

只有 action-head attention 的 Q/K/V linear layers 注入 LoRA；backbone、量化参数和教师冻结。默认 LoRA rank 16、alpha 32、dropout 0.05，AdamW 学习率 (5\times10^{-5})、weight decay 0.01、50-step warmup 后 cosine decay、gradient clip 1.0。

## 7. Clean anchor

为限制 OOD 适配破坏 Standard LIBERO，每轮从 clean LIBERO 采集最多 4 个教师/学生状态，放入容量 256 的 replay；每次更新随机取 4 个，以 `lambda_anchor=0.1` 加入损失。它是训练正则项，不是 validation，也不参与 checkpoint 选择。

## 8. 可复现性与输出

训练 seed 同时控制 task schedule、随机状态选择、initial-state 抽样和 diffusion noise。checkpoint 保存 LoRA、optimizer、scheduler、Python/NumPy/Torch RNG 与 task schedule。`run.json` 记录方法、代码版本、host、GPU、端口、manifest 路径/哈希与完整配置；train/eval 的 manifest SHA256 必须一致。`metrics.jsonl` 默认只记录入选状态的 q/r、索引、phase/reason、配额校验、success、loss、梯度与耗时，只有诊断时才显式保存全轨迹 q/r。checkpoint 统一放在 `checkpoints/`，默认仅保留最近 2 个完整 checkpoint。

## 9. 可行性与局限

可行性依据是：量化误差能直接在共享噪声下测量；LoRA 只调整 action head，训练参数和遗忘风险较小；student-on-policy 完整轨迹能覆盖接触、搬运、放置和 late states；显式 manifest 使 train/eval 的同任务关系可审计。

仍需实验验证的局限：

- (q/r) 衡量教师差异，不等同于任务成功的重要性；
- 每步同时跑教师与学生，完整 episode 的训练成本高；
- 16、四阶段、gap 4、5 updates 都是待消融的设计选择；
- 仅监督 action chunk 的第一个动作，未利用整个 action horizon；
- 纯学生 rollout 在很差的量化策略下可能长期停留于失败区域；
- 当前无 validation，最终 checkpoint 是固定训练预算而非验证集最优；
- train/eval 共用 task IDs 会高估对未见任务的迁移能力，只能解释为目标域适配；
- seed 0 只用于打通主链路，正式结论必须来自三组 train seed 和固定 Shared-560 eval seed。

因此，Shared-560 可以回答 OPQD 能否修复已知目标任务上的量化退化，但若要声称 held-out-task generalization，仍需另设不相交测试集。
