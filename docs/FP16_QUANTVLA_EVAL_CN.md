# FP16 与 QuantVLA Eval 教程

本教程覆盖从环境检查到 LIBERO/LIBERO-Plus 评测、恢复、监控和汇总。两个入口都会自动启动推理服务、等待端口 ready、运行 evaluator，并在结束或异常时关闭本次服务。

## 1. 准备

```bash
ssh suzhou-C
cd /lumos-vePFS/suda/ruan/QuantVLA
```

需要：

- `groot_test`：加载 GR00T/QuantVLA 推理服务；
- `libero_test`：运行 simulator evaluator；
- Standard LIBERO：默认 `/lumos-vePFS/suda/ruan/LIBERO`；
- LIBERO-Plus：默认 `/lumos-vePFS/suda/ruan/LIBERO-plus`；
- suite checkpoint 可从 Hugging Face cache 或本地 `--checkpoint` 加载；
- QuantVLA 需要 suite 对应的 `atm_alpha_beta_*.json` 和 `model/quantvla/.../duquant_pack`。

快速检查：

```bash
nvidia-smi
./eval_fp16.sh --help
./eval_quantvla.sh --help
```

先用 `--dry-run` 检查路径和最终命令，不启动进程也不创建 output：

```bash
./eval_fp16.sh --suite libero_spatial --gpu 0 --port 5700 --dry-run
```

若模型已在 `model/` 缓存，而集群无法稳定访问 Hugging Face，在任一命令末尾加
`--offline`。首次下载模型时不要使用该选项。

## 2. FP16

LIBERO-Plus：

```bash
./eval_fp16.sh \
  --benchmark libero-plus \
  --suite libero_spatial \
  --gpu 0 \
  --port 5700 \
  --run-name main-v1
```

Standard LIBERO：

```bash
./eval_fp16.sh \
  --benchmark libero \
  --suite libero_spatial \
  --gpu 0 \
  --port 5700 \
  --run-name main-v1
```

指定本地 checkpoint 或 Hugging Face model id：

```bash
./eval_fp16.sh \
  --benchmark libero-plus --suite libero_goal --gpu 1 --port 5701 \
  --checkpoint youliangtan/gr00t-n1.5-libero-goal-posttrain \
  --run-name main-v1
```

不传 `--checkpoint` 时，脚本根据 suite 使用仓库内置 checkpoint 映射。

## 3. QuantVLA W4A8

LIBERO-Plus：

```bash
./eval_quantvla.sh \
  --benchmark libero-plus \
  --suite libero_spatial \
  --gpu 1 \
  --port 5710 \
  --run-name main-v1
```

Standard LIBERO：

```bash
./eval_quantvla.sh \
  --benchmark libero \
  --suite libero_spatial \
  --gpu 1 \
  --port 5710 \
  --run-name main-v1
```

指定基础 checkpoint：

```bash
./eval_quantvla.sh \
  --benchmark libero-plus --suite libero_object --gpu 2 --port 5712 \
  --checkpoint /path/to/base-checkpoint \
  --run-name main-v1
```

QuantVLA 会复用 suite 专属 DuQuant pack；不要把其他 suite 或其他量化配置的 pack 混用。

## 4. Suite、manifest 与 rollout 数

支持：`libero_spatial`、`libero_goal`、`libero_object`、`libero_10`。

Standard LIBERO 每个 suite 为 10 tasks × 5 initial states = 50 rollouts。LIBERO-Plus 默认 manifest：

```text
configs/libero_plus/first_24_per_category.json
```

它在每个 suite/category 取前 6 条，因此每 suite 42 条，四个 suite 合计每个 category 24 条、每个 method 168 条。指定其他 manifest：

```bash
./eval_fp16.sh \
  --benchmark libero-plus --suite libero_goal --gpu 0 --port 5701 \
  --manifest configs/libero_plus/first_100_per_category.json \
  --run-name first100-v1
```

## 5. 输出

默认规范：

```text
output/eval/<benchmark>/<method>/<suite>/<run-name>/
├── episodes.jsonl
├── summary.json
├── server.log
├── pipeline.log
├── server_command.txt
├── eval_command.txt
└── libero*_eval_*.log
```

指定根目录：

```bash
./eval_fp16.sh ... --output-root /lumos-vePFS/suda/ruan/experiments/output
```

直接指定完整目录：

```bash
./eval_fp16.sh ... --output-dir /absolute/path/to/eval-output
```

若 `episodes.jsonl` 已存在，脚本默认拒绝覆盖。继续未完成任务：

```bash
./eval_fp16.sh \
  --benchmark libero-plus --suite libero_spatial --gpu 0 --port 5700 \
  --run-name main-v1 --resume
```

默认 headless 且不保存视频；使用 `--save-video` 保存视频，使用 `--no-headless` 关闭 headless。

## 6. 四个 suite 的建议分配

每个运行命令占一张 GPU 和一个唯一端口：

| Suite | FP16 GPU/port | QuantVLA GPU/port |
|---|---|---|
| spatial | 0 / 5700 | 4 / 5740 |
| goal | 1 / 5701 | 5 / 5741 |
| object | 2 / 5702 | 6 / 5742 |
| libero_10 | 3 / 5703 | 7 / 5743 |

后台运行示例：

```bash
tmux new-session -d -s fp16-spatial \
  "cd '$PWD' && ./eval_fp16.sh --benchmark libero-plus --suite libero_spatial --gpu 0 --port 5700 --run-name main-v1"
```

每个 suite 分别创建 session，避免把多条长任务串在同一个 shell 后误触发重跑。

## 7. Monitor 与 collect

查看同一 run-name：

```bash
./monitor_eval.sh \
  --run-name main-v1 \
  --methods fp16 quantvla quantvla-opqd \
  --benchmarks libero-plus libero
```

只打印一次：

```bash
./monitor_eval.sh --once --run-name main-v1
```

中期汇总允许 partial：

```bash
./collect_eval.sh \
  --run-name main-v1 \
  --benchmarks libero-plus libero
```

最终汇总：

```bash
./collect_eval.sh \
  --run-name main-v1 \
  --benchmarks libero-plus libero \
  --require-complete
```

默认报告位于 `output/reports/<run-name>/`。使用自定义 output root 时，monitor 和 collect 同样传入 `--output-root`。

## 8. 故障检查

```bash
tail -n 80 output/eval/libero-plus/fp16/libero_spatial/main-v1/server.log
tail -n 80 output/eval/libero-plus/fp16/libero_spatial/main-v1/pipeline.log
ss -ltnp | grep 5700
nvidia-smi
```

`summary.json` 完整但 `episodes.jsonl` 为空表示结果不一致，不能使用旧 summary；改用新的 run-name 完整重跑。`Ctrl+C` 会终止 evaluator，并由 trap 关闭本次启动的推理服务。
