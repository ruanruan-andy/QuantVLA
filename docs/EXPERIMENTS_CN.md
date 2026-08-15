# 实验设计与记录表

## 1. 固定正式协议

| 项目 | 设置 |
|---|---|
| Methods | FP16、QuantVLA W4A8、QuantVLA-OPQD |
| Target tasks | Shared-560 first-20；每 suite/category 20；总计 560 |
| Train | OPQD only；与 eval 共用 task IDs；每 suite 140 episodes |
| Eval | 三种方法均在 Shared-560 上各跑 560 episodes |
| Horizons | spatial 220 / object 280 / goal 300 / libero_10 520 |
| Eval seed | 2026，所有方法固定 |
| Train seeds | 0、1、2 |
| Primary metric | episode success rate |
| Breakdown | suite、7 categories、difficulty |
| Uncertainty | 95% Wilson CI；同时报告 numerator/denominator |

这是 same-task transductive adaptation，不得表述为 held-out-task generalization。当前没有 validation：主结果使用固定 700-step final checkpoint，不按 Shared-560 eval success 选 checkpoint；如以后需要调参，应另建独立 development manifest。

## 2. 主结果表

| Method | Seed | Spatial | Object | Goal | Long | Overall 560 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | eval-2026 | — | — | — | — | — | — |
| QuantVLA | eval-2026 | — | — | — | — | — | — |
| OPQD | 0 | — | — | — | — | — | — |
| OPQD | 1 | — | — | — | — | — | — |
| OPQD | 2 | — | — | — | — | mean±std | — |

| Category | FP16 | QuantVLA | OPQD mean±std | OPQD−Quant (pp) |
|---|---:|---:|---:|---:|
| Camera Viewpoints | — | — | — | — |
| Robot Initial States | — | — | — | — |
| Language Instructions | — | — | — | — |
| Light Conditions | — | — | — | — |
| Background Textures | — | — | — | — |
| Sensor Noise | — | — | — | — |
| Objects Layout | — | — | — | — |

## 3. 必做消融

除目标变量外保持 Shared-560、seed、horizon、总计 700 optimizer steps 与 eval seed 2026 不变。

| ID | 变量 | 候选值 | 回答的问题 | 优先级 |
|---|---|---|---|---|
| A1 | selection | random-only / priority-only / mixed | q/r priority 是否有效 | P0 |
| A2 | states/episode | **16** / 32 / 64 | 稀疏预算与收益/成本 | P0 |
| A3 | phases | no phase / 4 phases | 阶段覆盖是否有效 | P0 |
| A4 | min gap | 0 / 2 / **4** / 8 | 时间冗余影响 | P1 |
| A5 | temporal risk | q-only / q+r | (r_t) 是否贡献 | P0 |
| A6 | clean anchor | 0 / **0.1** | OOD 收益与 clean 保持 | P0 |
| A7 | updates/episode | 1 / **5** | 重复更新是否必要 | P1 |
| A8 | LoRA rank | 8 / **16** / 32 | 容量敏感性 | P2 |

32/64-state 版本必须同步定义每阶段 priority/random 配额。64-state 在 spatial horizon 220 下无法同时严格满足 gap 4（理论最多约 55 个），因此 A2 必须报告每阶段实际 gap；不能把它与 16-state 主配置当作完全同约束比较。

## 4. 诊断表

| Seed | Suite | Episode success | selection | phase gaps | q mean/max | grad norm | episode sec | ETA |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 0 | spatial | — | —/16 | —/—/—/— | — | — | — | — |
| 0 | object | — | —/16 | —/—/—/— | — | — | — | — |
| 0 | goal | — | —/16 | —/—/—/— | — | — | — | — |
| 0 | long | — | —/16 | —/—/—/— | — | — | — | — |

建议额外画出 phase 的选择数量、timestep 分布、q/r 分布及 checkpoint 曲线；这用于解释方法，不替代 Test success rate。

## 5. 运行顺序与机器分配

| 阶段 | 机器 | GPU | 任务 |
|---|---|---|---|
| 1 | suzhou-C | 0–3 | OPQD seed 0，Shared-560 四 suite |
| 1 | suzhou-C | 4–7 | FP16 Shared-560，四 suite eval |
| 1 | suzhou-I | 0–3 | QuantVLA Shared-560，四 suite eval |
| 2 | suzhou-C/I | 空闲卡 | OPQD seed 1/2 train |
| 3 | 任一空闲机 | 4 cards | OPQD checkpoints Shared-560 eval |

所有启动前重新检查 GPU、端口和现有进程。smoke 输出必须放在 `output/smoke/`，正式结果只能使用规范目录。

## 6. 运行台账

| Date | Host:GPU | Method | Seed | Suite | Port(s) | Output | Status |
|---|---|---:|---|---|---|---|---|
| YYYY-MM-DD | suzhou-C:0 | OPQD train | 0 | spatial | 31000/31001 | `output/train/...` | planned |
| YYYY-MM-DD | suzhou-C:4 | FP16 eval | 2026 | spatial | 31100 | `output/eval/...` | planned |
| YYYY-MM-DD | suzhou-I:0 | Quant eval | 2026 | spatial | 31200 | `output/eval/...` | planned |

完整性判定：每个 method/seed 必须达到 manifest 规定数量、无重复 episode key、无 error、四 suite 和七 category 均齐全；OPQD train/eval 的 manifest SHA256 还必须完全一致。只有 `./collect_eval.sh --require-complete` 通过后才能填写最终表格。
