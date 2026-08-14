# QuantVLA-OPQD 训练与评测教程

## 1. 准备

```bash
ssh suzhou-C
cd /lumos-vePFS/suda/ruan/QuantVLA
nvidia-smi
./train_quantvla_opqd.sh --help
./eval_quantvla_opqd.sh --help
```

若所需基础模型已缓存且集群无法稳定访问 Hugging Face，在训练或评测命令末尾加
`--offline`；首次下载模型时不要使用。

训练需要 `groot_test`、`libero_test`、Standard LIBERO、LIBERO-Plus、suite checkpoint、ATM/OHB JSON 和 DuQuant pack。默认用 LIBERO-Plus first-24 做 OOD calibration，并用 Standard LIBERO 生成 IID anchor。

## 2. 训练一个 suite

先 dry-run：

```bash
./train_quantvla_opqd.sh \
  --suite libero_spatial --gpu 0 \
  --env-port 5600 --clean-env-port 5601 \
  --run-name opqd-v1 --dry-run
```

正式训练：

```bash
./train_quantvla_opqd.sh \
  --suite libero_spatial \
  --gpu 0 \
  --env-port 5600 \
  --clean-env-port 5601 \
  --max-iterations 100 \
  --save-every 20 \
  --run-name opqd-v1
```

默认输出：

```text
output/train/libero-plus/quantvla-opqd/libero_spatial/opqd-v1/
├── config.json
├── selected_ood_tasks.json
├── metrics.jsonl
├── train.log
├── train_command.txt
├── ood_env_service.log
├── clean_env_service.log
└── checkpoint-000100/
    ├── adapter/
    └── trainer_state.pt
```

指定输出根目录或完整目录：

```bash
./train_quantvla_opqd.sh ... --output-root /path/to/output
./train_quantvla_opqd.sh ... --output-dir /absolute/path/to/train-run
```

## 3. 参数与恢复

常用 OPQD 参数放在 `--` 后转发给 trainer：

```bash
./train_quantvla_opqd.sh \
  --suite libero_goal --gpu 1 \
  --env-port 5610 --clean-env-port 5611 \
  --max-iterations 100 --save-every 20 --run-name opqd-v1 \
  -- \
  --rollout-horizon 16 \
  --temporal-horizon 4 \
  --temporal-discount 0.9 \
  --alpha-q 1.0 \
  --beta-r 1.0 \
  --lambda-anchor 0.1 \
  --lora-rank 16 \
  --lora-alpha 32
```

trainer 默认自动寻找输出目录中最新的完整 checkpoint 并恢复。显式指定：

```bash
./train_quantvla_opqd.sh \
  --suite libero_spatial --gpu 0 \
  --env-port 5600 --clean-env-port 5601 \
  --run-name opqd-v1 \
  --resume-from output/train/libero-plus/quantvla-opqd/libero_spatial/opqd-v1/checkpoint-000060 \
  --max-iterations 100 --save-every 20
```

恢复包含 LoRA、optimizer、iteration、task schedule 和随机状态；`metrics.jsonl` 会对齐到恢复 iteration。

## 4. 评测 checkpoint

`--checkpoint` 可指向 `checkpoint-xxxxxx` 或其中的 `adapter/`。

LIBERO-Plus：

```bash
./eval_quantvla_opqd.sh \
  --benchmark libero-plus \
  --suite libero_spatial \
  --gpu 0 \
  --port 5700 \
  --checkpoint output/train/libero-plus/quantvla-opqd/libero_spatial/opqd-v1/checkpoint-000100 \
  --run-name main-v1
```

Standard LIBERO clean-domain 评测：

```bash
./eval_quantvla_opqd.sh \
  --benchmark libero \
  --suite libero_spatial \
  --gpu 0 \
  --port 5700 \
  --checkpoint output/train/libero-plus/quantvla-opqd/libero_spatial/opqd-v1/checkpoint-000100 \
  --run-name main-v1
```

评测不会删除任何 checkpoint。已有结果默认拒绝覆盖，继续运行时加 `--resume`。

## 5. 四 suite 并行建议

| Suite | GPU | OOD/Clean ports | Eval port |
|---|---:|---|---:|
| spatial | 0 | 5600 / 5601 | 5700 |
| goal | 1 | 5610 / 5611 | 5710 |
| object | 2 | 5620 / 5621 | 5720 |
| libero_10 | 3 | 5630 / 5631 | 5730 |

每个 suite 使用独立 tmux session：

```bash
tmux new-session -d -s opqd-spatial \
  "cd '$PWD' && ./train_quantvla_opqd.sh --suite libero_spatial --gpu 0 --env-port 5600 --clean-env-port 5601 --run-name opqd-v1 --max-iterations 100 --save-every 20"
```

训练完成后显式运行 eval；不要在训练命令后用未审查的 shell 串联多个评测。

## 6. 训练监控

```bash
tail -f output/train/libero-plus/quantvla-opqd/libero_spatial/opqd-v1/train.log
tail -n 1 output/train/libero-plus/quantvla-opqd/libero_spatial/opqd-v1/metrics.jsonl
watch -n 3 nvidia-smi -i 0
```

关键字段：`q_mean/q_max`、`r_mean/r_max`、`weight_mean/weight_max`、`loss_opqd_unweighted_mean`、`loss_anchor_mean`、`gradient_norm`。

## 7. 三方法汇总

三方法 eval 使用相同 `run-name`：

```bash
./monitor_eval.sh --run-name main-v1 --benchmarks libero-plus libero

./collect_eval.sh \
  --run-name main-v1 \
  --benchmarks libero-plus libero \
  --require-complete
```

若使用自定义根目录，两条命令都加 `--output-root /path/to/output`。报告位于 `<output-root>/reports/main-v1/`。

## 8. 实验解释

默认 OPQD train 与 LIBERO-Plus eval 使用同一 first-24 calibration tasks，因此该结果是量化恢复实验。要研究未见泛化，应建立与训练 task IDs 不重叠的 holdout manifest；实验设计见 `docs/EXPERIMENTS_CN.md`，方法定义见 `docs/METHOD_CN.md`。
