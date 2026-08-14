# FP16 与 QuantVLA：从零开始 Eval

本文只覆盖两个无需训练的基线：`fp16` 与 `quantvla`。默认正式协议是 LIBERO-Plus Test-560；两者使用相同任务、初始状态和 eval seed。

## 1. 环境检查

```bash
ssh suzhou-C                     # FP16；QuantVLA 正式任务放 suzhou-I
cd /lumos-vePFS/suda/ruan/QuantVLA
git status --short
nvidia-smi
conda env list | grep -E 'groot_test|libero_test'
./eval_fp16.sh --help
./eval_quantvla.sh --help
```

端口和 GPU 必须空闲：

```bash
ss -ltnp | grep -E ':31100|:31200' || true
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

默认 manifest：`configs/libero_plus/splits/test560-split2026.json`。每个 suite 的七类各 20 条，共 140；四个 suite 共 560。它与同样为 560 条的 Train-560 无交集。

## 2. 先 dry-run

```bash
./eval_fp16.sh --benchmark libero-plus --suite libero_spatial \
  --gpu 0 --port 31100 --eval-seed 2026 --dry-run

./eval_quantvla.sh --benchmark libero-plus --suite libero_spatial \
  --gpu 0 --port 31200 --eval-seed 2026 --dry-run
```

dry-run 不启动服务、不创建结果。

## 3. 单个 suite

FP16：

```bash
./eval_fp16.sh \
  --benchmark libero-plus \
  --suite libero_spatial \
  --gpu 0 --port 31100 \
  --manifest configs/libero_plus/splits/test560-split2026.json \
  --eval-seed 2026 --run-name default
```

QuantVLA W4A8：

```bash
./eval_quantvla.sh \
  --benchmark libero-plus \
  --suite libero_spatial \
  --gpu 0 --port 31200 \
  --manifest configs/libero_plus/splits/test560-split2026.json \
  --eval-seed 2026 --run-name default
```

默认输出：

```text
output/eval/libero-plus/test560-split2026/{fp16|quantvla}/<suite>/default/
├── episodes.jsonl
├── summary.json
├── server.log
└── pipeline.log
```

已有 `episodes.jsonl` 时默认拒绝覆盖；确认属于同一协议后加 `--resume`。自定义根目录用 `--output-root /path/to/output`，完全自定义目录用 `--output-dir /absolute/path`。

## 4. 四个 suite 并行

在 suzhou-C 的 GPU4–7 跑 FP16（GPU0–3 留给 OPQD 训练）：

```bash
for spec in 'libero_spatial 4 31100' 'libero_object 5 31110' 'libero_goal 6 31120' 'libero_10 7 31130'; do
  set -- $spec
  tmux new-session -d -s "fp16-$1" \
    "cd /lumos-vePFS/suda/ruan/QuantVLA && ./eval_fp16.sh --benchmark libero-plus --suite $1 --gpu $2 --port $3 --eval-seed 2026 --run-name default"
done
```

在 suzhou-I 跑 QuantVLA：

```bash
for spec in 'libero_spatial 0 31200' 'libero_object 1 31210' 'libero_goal 2 31220' 'libero_10 3 31230'; do
  set -- $spec
  tmux new-session -d -s "quant-$1" \
    "cd /lumos-vePFS/suda/ruan/QuantVLA && ./eval_quantvla.sh --benchmark libero-plus --suite $1 --gpu $2 --port $3 --eval-seed 2026 --run-name default"
done
```

## 5. 参数速查

| 目标 | 参数 |
|---|---|
| 指定卡 / suite / 端口 | `--gpu N --suite NAME --port PORT` |
| Standard LIBERO | `--benchmark libero` |
| LIBERO-Plus | `--benchmark libero-plus --manifest MANIFEST` |
| 指定模型或本地 ckpt | `--checkpoint PATH_OR_HF_ID` |
| 固定采样 | `--eval-seed 2026` |
| 指定输出 | `--output-root ROOT` 或 `--output-dir DIR` |
| 保存视频 | `--save-video` |
| 断点续跑 | `--resume` |
| evaluator 高级参数 | 放在 `--` 后，例如 `-- --max-tasks 1` |

视频默认关闭，正式 Test-560 建议保持默认。需要可视化个别 rollout 时显式开启：

```bash
./eval_fp16.sh --benchmark libero-plus --suite libero_spatial \
  --gpu 4 --port 31100 --save-video --run-name video-debug

./eval_quantvla.sh --benchmark libero-plus --suite libero_spatial \
  --gpu 0 --port 31200 --save-video --run-name video-debug
```

视频路径会写入对应 episode 的 `video_path`；请为视频调试使用独立 `--run-name` 或 `--output-dir`，不要混入正式 `default` 结果。

Standard LIBERO 与自定义 checkpoint 示例：

```bash
./eval_fp16.sh --benchmark libero --suite libero_goal --gpu 0 --port 31140 \
  --checkpoint /path/to/model --output-dir /path/to/result
```

## 6. Monitor 与汇总

```bash
./monitor_eval.sh --once
./monitor_eval.sh --refresh-seconds 20
./collect_eval.sh
./collect_eval.sh --require-complete
```

monitor 一屏显示三个方法的总进度、成功率、ETA、四 suite 以及 `4 suite × 7 category` 对比。seed 0 完整报告默认写入 `output/reports/libero-plus/test560-split2026/opqd-seed-000/default/`。

排错先看：

```bash
tail -n 80 output/eval/libero-plus/test560-split2026/fp16/libero_spatial/default/server.log
tail -n 80 output/eval/libero-plus/test560-split2026/fp16/libero_spatial/default/pipeline.log
```
