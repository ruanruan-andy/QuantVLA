# GR00T FP16 and QuantVLA W4A8 evaluation

This repository evaluates the same four suite-specific GR00T N1.5 checkpoints in two modes:

| Model variant | Server | Precision behavior |
|---|---|---|
| `groot-fp16` | `run_inference_server.sh` | Full-precision baseline |
| `groot-quantvla-w4a8` | `run_quantvla.sh` | W4A8 fake quantization with ATM and OHB |

The W4A8 implementation simulates quantization error with floating-point tensors. It is suitable for task-success comparisons, but its process memory is not a measurement of packed INT4 deployment memory.

## Output layout

Results never overlap between models or benchmarks:

```text
output/
├── libero/
│   ├── groot-fp16/<suite>/
│   └── groot-quantvla-w4a8/<suite>/
├── libero-plus/
│   ├── groot-fp16/<suite>/
│   └── groot-quantvla-w4a8/<suite>/
└── summary/
```

Each suite directory contains `episodes.jsonl`, `summary.json`, the text log, and rollout videos. `--resume` skips keys already committed without an infrastructure error.

## LIBERO-Plus sampling protocol

Full LIBERO-Plus contains 10,030 tasks. This evaluation uses the deterministic manifest:

```text
configs/libero_plus/first_100_per_category.json
```

For every generalization category, it selects the first 25 tasks by task index from each suite:

```text
25 spatial + 25 goal + 25 object + 25 libero_10 = 100 tasks/category
7 categories × 100 = 700 LIBERO-Plus rollouts/model
```

The identical task list is used for FP16 and W4A8. Each selected LIBERO-Plus task has one rollout. Standard LIBERO uses 10 tasks × 5 initial states = 50 rollouts per suite, or 200 per model.

## Manual two-terminal commands

The server and evaluator are separate long-running processes and normally run in two terminals.

### FP16 example

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=6 GR00T_PORT=5556 \
    ./run_inference_server.sh libero_spatial
```

Terminal 2, standard LIBERO:

```bash
CUDA_VISIBLE_DEVICES=6 GR00T_PORT=5556 \
    ./run_libero_eval.sh libero_spatial \
    --model-variant groot-fp16 --headless --resume
```

Terminal 2, LIBERO-Plus:

```bash
CUDA_VISIBLE_DEVICES=7 GR00T_PORT=5560 \
    ./run_libero_plus_eval.sh libero_spatial \
    --model-variant groot-fp16 --headless --resume
```

### QuantVLA W4A8 example

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=4 GR00T_PORT=5570 \
    ./run_quantvla.sh libero_spatial
```

Terminal 2, standard LIBERO:

```bash
CUDA_VISIBLE_DEVICES=4 GR00T_PORT=5570 \
    ./run_libero_eval.sh libero_spatial \
    --model-variant groot-quantvla-w4a8 --headless --resume
```

Terminal 2, LIBERO-Plus:

```bash
CUDA_VISIBLE_DEVICES=5 GR00T_PORT=5574 \
    ./run_libero_plus_eval.sh libero_spatial \
    --model-variant groot-quantvla-w4a8 --headless --resume
```

`run_libero_plus_eval.sh` automatically supplies the 700-task sampling manifest unless `--sample-manifest` is explicitly provided.

## Four-GPU layout used for the comparison

| GPU | Model | Benchmark | Ports |
|---:|---|---|---|
| 6 | FP16 | Standard LIBERO | 5556–5559 |
| 7 | FP16 | LIBERO-Plus subset | 5560–5563 |
| 4 | QuantVLA W4A8 | Standard LIBERO | 5570–5573 |
| 5 | QuantVLA W4A8 | LIBERO-Plus subset | 5574–5577 |

Within each range, ports follow `spatial`, `goal`, `object`, `libero_10` order.

## Monitor

Print one snapshot:

```bash
./monitor_eval.sh --once
```

Refresh at a chosen interval:

```bash
./monitor_eval.sh --interval 1
./monitor_eval.sh --interval 0.5
```

The dashboard reports process/port health, per-model suite progress, success rates, errors, typical duration, ETA, and all seven LIBERO-Plus categories. It intentionally does not print GPU utilization.

## Collect

```bash
./collect_eval.sh
```

Reports are written to `output/summary/`, including:

```text
report.md
summary.json
suites.csv
episodes.csv
standard_libero_by_task.csv
libero_plus_by_category.csv
libero_plus_by_suite_category.csv
libero_plus_by_difficulty.csv
libero_plus_by_suite_difficulty.csv
libero_plus_by_category_difficulty.csv
fp16_vs_w4a8.csv
```

`fp16_vs_w4a8.csv` contains matched success-rate differences by suite and generalization category.
