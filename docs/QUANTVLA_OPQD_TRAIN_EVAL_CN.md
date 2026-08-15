# QuantVLA-OPQD：从零开始 Train 与 Eval

OPQD 是唯一需要训练的方法：在 LIBERO-Plus Shared-560 first-20 上做 student-on-policy rollout，并在相同 task IDs 上评测三种方法。这是 same-task transductive adaptation，不是 held-out-task generalization。

## 1. 正式配置

| 项目 | 默认值 |
|---|---|
| train / eval | Shared-560 first-20，task IDs 完全相同 |
| 每 suite | 7 类 × 20 = 140 episodes |
| horizon | spatial 220 / object 280 / goal 300 / libero_10 520；成功即结束 |
| 状态选择 | 4 phases × (2 priority + 2 random) = 16 |
| 时间间隔 | 每阶段目标 gap 4，不足时只在该阶段按 4→3→2→1 补齐固定配额 |
| 更新 | 5 updates/episode，700 optimizer steps/suite |
| LoRA | action-head Q/K/V，rank 16，alpha 32，dropout 0.05 |
| seed | 先跑 0，正式结果补 1、2 |

selector 不跨阶段补位、不重复；每个有效 episode 必须得到 16 个状态。`metrics.jsonl` 和 monitor 会显示 `selection_valid`、各阶段数量及实际 gap。train/eval 均记录 manifest SHA256，正式比较前必须一致。

## 2. 环境检查与 dry-run

```bash
ssh suzhou-C
cd /lumos-vePFS/suda/ruan/QuantVLA
nvidia-smi
conda env list | grep -E 'groot_test|libero_test'
ss -ltnp | grep -E ':31000|:31001' || true
./train_quantvla_opqd.sh --help

./train_quantvla_opqd.sh --suite libero_spatial --gpu 0 \
  --env-port 31000 --clean-env-port 31001 --seed 0 --dry-run
```

## 3. 训练一个 suite

```bash
./train_quantvla_opqd.sh \
  --suite libero_spatial --gpu 0 \
  --env-port 31000 --clean-env-port 31001 \
  --manifest configs/libero_plus/shared560-first20.json \
  --seed 0 --episodes 140 --updates-per-episode 5 \
  --save-every-steps 70 --keep-last-checkpoints 2
```

常用覆盖项：`--episode-horizon N`、`--output-dir /absolute/path`、`--run-name NAME`。Tyro 高级参数放在 `--` 后，例如 `-- --min-temporal-gap 4 --save-timestep-scores true`；全轨迹分数默认不保存。

默认输出：

```text
output/train/libero-plus/opqd-v2-s16-shared560-first20/seed-000/<suite>/
├── run.json
├── status.json
├── metrics.jsonl
├── logs/
│   ├── train.log
│   ├── plus_rollout_service.log
│   └── iid_anchor_service.log
└── checkpoints/                 # 默认只保留最近 2 个完整 checkpoint
    ├── step-000630/
    └── step-000700/
```

`run.json` 是唯一运行配置与溯源入口；`metrics.jsonl` 是逐 episode 指标；服务日志与训练日志只放 `logs/`。

## 4. suzhou-C 四 suite 并行

```bash
for spec in 'libero_spatial 0 31000 31001' 'libero_object 1 31010 31011' 'libero_goal 2 31020 31021' 'libero_10 3 31030 31031'; do
  set -- $spec
  tmux new-session -d -s "opqd-s0-$1" \
    "cd /lumos-vePFS/suda/ruan/QuantVLA && ./train_quantvla_opqd.sh --suite $1 --gpu $2 --env-port $3 --clean-env-port $4 --seed 0"
done
```

```bash
./monitor_eval.sh --once --opqd-train-seed 0
tmux ls
tail -f output/train/libero-plus/opqd-v2-s16-shared560-first20/seed-000/libero_spatial/logs/train.log
```

## 5. 断点恢复

```bash
./train_quantvla_opqd.sh --suite libero_spatial --gpu 0 \
  --env-port 31000 --clean-env-port 31001 --seed 0 \
  --resume-from output/train/libero-plus/opqd-v2-s16-shared560-first20/seed-000/libero_spatial/checkpoints/step-000630
```

suite、seed、manifest 和关键超参必须一致；程序恢复 adapter、optimizer、scheduler、RNG 与 task schedule。

## 6. 评测 adapter

```bash
./eval_quantvla_opqd.sh \
  --benchmark libero-plus --suite libero_spatial \
  --gpu 4 --port 31240 \
  --checkpoint output/train/libero-plus/opqd-v2-s16-shared560-first20/seed-000/libero_spatial/checkpoints/step-000700 \
  --train-seed 0 --eval-seed 2026
```

默认输出：

```text
output/eval/libero-plus/shared560-first20/opqd-v2-s16/seed-000/<suite>/
├── run.json
├── metrics/{episodes.jsonl,summary.json}
├── logs/{server.log,pipeline.log,evaluator.log}
└── videos/                      # 仅 --save-video 时产生
```

正式 eval 默认不保存视频。需要行为诊断时加 `--save-video --run-name video-debug`；已有相同 episode 时加 `--resume`，不得直接覆盖。Standard LIBERO 使用 `--benchmark libero`。

## 7. Seeds、汇总与 smoke test

seed 1/2 同时改变训练 `--seed`、端口以及评测 `--train-seed`；三组 eval 都固定 `--eval-seed 2026`。

```bash
./monitor_eval.sh --refresh-seconds 20 --opqd-train-seed 0
./collect_eval.sh --opqd-train-seed 0
./collect_eval.sh --opqd-train-seed 0 --require-complete

./train_quantvla_opqd.sh --suite libero_spatial --gpu 0 \
  --env-port 31900 --clean-env-port 31901 --seed 0 \
  --episodes 1 --updates-per-episode 1 \
  --output-dir output/smoke/opqd-v2-s16/spatial
```

smoke test 必须使用独立目录，不能写入正式结果。
