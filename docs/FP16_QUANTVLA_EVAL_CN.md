# FP16 与 QuantVLA：从零开始 Eval

`fp16` 与 `quantvla` 都不训练。正式协议为 LIBERO-Plus Test-560：每个 suite 七类各 20 条，共 140；四个 suite 共 560，方法间固定相同任务、初始状态和 eval seed。

## 1. 环境检查

```bash
ssh suzhou-C
cd /lumos-vePFS/suda/ruan/QuantVLA
git status --short
nvidia-smi
conda env list | grep -E 'groot_test|libero_test'
ss -ltnp | grep -E ':31100|:31200' || true
./eval_fp16.sh --help
./eval_quantvla.sh --help
```

默认 manifest 为 `configs/libero_plus/splits/test560-split2026.json`，与 Train-560 task ID 无交集。

## 2. dry-run

```bash
./eval_fp16.sh --benchmark libero-plus --suite libero_spatial \
  --gpu 0 --port 31100 --eval-seed 2026 --dry-run

./eval_quantvla.sh --benchmark libero-plus --suite libero_spatial \
  --gpu 0 --port 31200 --eval-seed 2026 --dry-run
```

dry-run 不启动服务、不创建结果。

## 3. 单个 suite

```bash
./eval_fp16.sh --benchmark libero-plus --suite libero_spatial \
  --gpu 0 --port 31100 \
  --manifest configs/libero_plus/splits/test560-split2026.json \
  --eval-seed 2026

./eval_quantvla.sh --benchmark libero-plus --suite libero_spatial \
  --gpu 0 --port 31200 \
  --manifest configs/libero_plus/splits/test560-split2026.json \
  --eval-seed 2026
```

默认输出：

```text
output/eval/libero-plus/test560-split2026/{fp16|quantvla}/<suite>/
├── run.json
├── metrics/
│   ├── episodes.jsonl
│   └── summary.json
├── logs/
│   ├── server.log
│   ├── pipeline.log
│   └── evaluator.log
└── videos/                      # 仅 --save-video 时产生
```

已有 `metrics/episodes.jsonl` 时默认拒绝覆盖；同一协议断点续跑使用 `--resume`。自定义根目录用 `--output-root ROOT`，完全指定用 `--output-dir DIR`，额外实验用 `--run-name NAME`。

## 4. 四 suite 并行

suzhou-C GPU4–7 跑 FP16：

```bash
for spec in 'libero_spatial 4 31100' 'libero_object 5 31110' 'libero_goal 6 31120' 'libero_10 7 31130'; do
  set -- $spec
  tmux new-session -d -s "fp16-$1" \
    "cd /lumos-vePFS/suda/ruan/QuantVLA && ./eval_fp16.sh --benchmark libero-plus --suite $1 --gpu $2 --port $3 --eval-seed 2026"
done
```

suzhou-I GPU0–3 跑 QuantVLA：

```bash
for spec in 'libero_spatial 0 31200' 'libero_object 1 31210' 'libero_goal 2 31220' 'libero_10 3 31230'; do
  set -- $spec
  tmux new-session -d -s "quant-$1" \
    "cd /lumos-vePFS/suda/ruan/QuantVLA && ./eval_quantvla.sh --benchmark libero-plus --suite $1 --gpu $2 --port $3 --eval-seed 2026"
done
```

## 5. 参数速查

| 目标 | 参数 |
|---|---|
| 卡 / suite / 端口 | `--gpu N --suite NAME --port PORT` |
| Standard LIBERO | `--benchmark libero` |
| LIBERO-Plus | `--benchmark libero-plus --manifest MANIFEST` |
| 模型或 ckpt | `--checkpoint PATH_OR_HF_ID` |
| 固定采样 | `--eval-seed 2026` |
| 输出 | `--output-root ROOT` / `--output-dir DIR` / `--run-name NAME` |
| 视频 | `--save-video`；正式 eval 默认关闭 |
| 续跑 | `--resume` |
| evaluator 高级参数 | 放在 `--` 后，例如 `-- --max-tasks 1` |

视频调试使用独立目录，例如：

```bash
./eval_fp16.sh --benchmark libero-plus --suite libero_spatial \
  --gpu 4 --port 31100 --save-video --run-name video-debug
```

## 6. Monitor、汇总与排错

```bash
./monitor_eval.sh --once
./monitor_eval.sh --refresh-seconds 20
./collect_eval.sh
./collect_eval.sh --require-complete
```

monitor 一屏显示方法、总进度、成功率、ETA、四 suite 与 `4 suite × 7 category`。seed 0 报告默认写到 `output/reports/libero-plus/test560-split2026/opqd-seed-000/`。

```bash
tail -n 80 output/eval/libero-plus/test560-split2026/fp16/libero_spatial/logs/server.log
tail -n 80 output/eval/libero-plus/test560-split2026/fp16/libero_spatial/logs/pipeline.log
```
