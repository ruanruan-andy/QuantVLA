# 实验设计与记录

## 1. 主实验矩阵

| Method | 训练 | Standard LIBERO eval | LIBERO-Plus eval | 主要目的 |
|---|---:|---:|---:|---|
| FP16 | — | 必做 | 必做 | 全精度上界 |
| QuantVLA W4A8 | — | 必做 | 必做 | 量化退化 |
| QuantVLA-OPQD | LIBERO-Plus calibration | 必做 | 必做 | 量化恢复与 clean-domain 保持 |

四个 suite：`libero_spatial`、`libero_goal`、`libero_object`、`libero_10`。

| Benchmark | 每 suite | 总 rollout/method | 分组 |
|---|---:|---:|---|
| Standard LIBERO | 10 tasks × 5 states = 50 | 200 | suite、task |
| LIBERO-Plus first-24 | 7 categories × 6 tasks = 42 | 168 | suite、category、difficulty |

## 2. 推荐实验列表

| ID | 优先级 | 实验 | 自变量 | 主要指标 | 状态 |
|---|---|---|---|---|---|
| E01 | P0 | 三方法主对比 | method | success rate、95% CI | 待运行 |
| E02 | P0 | 七类泛化对比 | method × category | category success rate | 待运行 |
| E03 | P0 | clean-domain 保持 | method × Standard LIBERO suite | success rate、OPQD-FP16 | 待运行 |
| E04 | P1 | OPQD 瞬时项消融 | `alpha_q={0,1}` | OPQD-Quant delta | 待运行 |
| E05 | P1 | OPQD 未来风险消融 | `beta_r={0,1}` | OPQD-Quant delta | 待运行 |
| E06 | P1 | IID anchor 消融 | `lambda_anchor={0,0.05,0.1}` | Plus 提升与 clean 回退 | 待运行 |
| E07 | P1 | temporal horizon | `H={0,2,4,8}` | success、q/r、稳定性 | 待运行 |
| E08 | P1 | checkpoint 曲线 | iteration `20/40/60/80/100` | success、训练成本 | 待运行 |
| E09 | P2 | LoRA 容量 | rank `4/8/16/32` | success、参数量 | 待运行 |
| E10 | P2 | calibration 规模 | 每类 `6/12/24/...` | sample efficiency | 待设计 |
| E11 | P2 | W/A bit 配置 | W4A8 等 | success、动作误差 | 待设计 |
| E12 | P2 | 推理效率 | method | latency、吞吐、显存 | 待运行 |
| E13 | P2 | 未见 holdout | train/eval manifest 不重叠 | unseen success rate | 待设计 |

`sqrt`、`block1` 和 camera-noise 旧实验不属于当前三方法主线，已从仓库与默认汇总中移除；若未来重新研究，应使用新的 run-name 和独立实验说明，不能混入主结果。

## 3. 公平比较约束

- 相同 suite checkpoint、task IDs、initial state、policy seed、denoising steps；
- 相同 evaluator 代码、MAX_STEPS、headless 设置和 manifest；
- 三方法使用相同 `run-name`，但输出目录按 method 隔离；
- `episodes.jsonl` 是事实来源，`summary.json` 仅用于一致性校验；
- partial 结果只能用于进度观察，最终表要求目标 rollout 全部完成且 errors=0；
- 报告 OPQD 相对 FP16 与 QuantVLA 的百分点差，同时给出 95% Wilson CI；
- fake-quant 成功率实验与真实 INT4 效率实验分开陈述。

## 4. 主结果表模板

| Benchmark | Suite | FP16 | QuantVLA | QuantVLA-OPQD | OPQD-FP16 (pp) | OPQD-Quant (pp) |
|---|---|---|---|---|---:|---:|
| LIBERO-Plus | spatial | — | — | — | — | — |
| LIBERO-Plus | goal | — | — | — | — | — |
| LIBERO-Plus | object | — | — | — | — | — |
| LIBERO-Plus | libero_10 | — | — | — | — | — |
| Standard LIBERO | spatial | — | — | — | — | — |
| Standard LIBERO | goal | — | — | — | — | — |
| Standard LIBERO | object | — | — | — | — | — |
| Standard LIBERO | libero_10 | — | — | — | — | — |

## 5. 七分类结果模板

| Category | FP16 | QuantVLA | QuantVLA-OPQD | OPQD-Quant (pp) |
|---|---|---|---|---:|
| Camera Viewpoints | — | — | — | — |
| Robot Initial States | — | — | — | — |
| Language Instructions | — | — | — | — |
| Light Conditions | — | — | — | — |
| Background Textures | — | — | — | — |
| Sensor Noise | — | — | — | — |
| Objects Layout | — | — | — | — |

## 6. OPQD 消融模板

| Run | α | β | H | γ | λ_anchor | Rank | Plus success | LIBERO success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quant-only | 0 | 0 | 0 | 0.9 | 0 | — | — | — |
| instant | 1 | 0 | 0 | 0.9 | 0 | 16 | — | — |
| temporal | 1 | 1 | 4 | 0.9 | 0 | 16 | — | — |
| full | 1 | 1 | 4 | 0.9 | 0.1 | 16 | — | — |

## 7. 运行登记模板

| Date | Host/GPU | Method | Benchmark | Suite | Run name | Port | Checkpoint | 状态 |
|---|---|---|---|---|---|---:|---|---|
| YYYY-MM-DD | suzhou-C:0 | fp16 | libero-plus | libero_spatial | main-v1 | 5700 | suite default | planned |

完成后使用：

```bash
./collect_eval.sh \
  --run-name main-v1 \
  --benchmarks libero-plus libero \
  --require-complete
```

归档 `report.md`、`summary.json`、全部 CSV、manifest、训练 `config.json` 和对应 checkpoint 路径。
