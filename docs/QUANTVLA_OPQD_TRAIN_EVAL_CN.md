# QuantVLA-OPQD：从零开始 Train 与 Eval

OPQD 是唯一需要训练的方法。默认使用 LIBERO-Plus Train-560 在线 rollout 训练，并只在等量且不相交的 Test-560 上做正式评测。

## 1. 正式默认值

| 项目 | 默认值 |
|---|---|
| train / test | Train-560 / Test-560，各 560 条且 task ID 无交集 |
| 每 suite 训练 | 7 类 × 20 条 = 140 episodes |
| episode horizon | spatial 220，object 280，goal 300，libero_10 520 |
| 状态选择 | 4 phases × (4 priority + 4 random) = 32 |
| `min_temporal_gap` | 4 timesteps |
| 更新 | 5 updates/episode，共 700 optimizer steps/suite |
| LoRA | action-head Q/K/V，rank 16，alpha 32，dropout 0.05 |
| seed | 先跑 0，正式结果补 1、2 |

`min_temporal_gap=4` 表示任意两个已选 timestep 的距离至少为 4；例如选中 10 后，11–13 不再选，14 可以选。短轨迹无法满足 32 个状态时会保留唯一状态、放宽 gap，并写入 `selection_gap_relaxed=true`。

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

## 3. 单个 suite 训练

```bash
./train_quantvla_opqd.sh \
  --suite libero_spatial \
  --gpu 0 \
  --env-port 31000 --clean-env-port 31001 \
  --manifest configs/libero_plus/splits/train560-split2026.json \
  --seed 0 --episodes 140 --updates-per-episode 5 \
  --save-every-steps 70 --run-name default
```

常用覆盖项：

- horizon：`--episode-horizon 220`
- 完全自定义输出：`--output-dir /absolute/path`
- 高级 Tyro 配置：`-- --priority-per-phase 4 --random-per-phase 4 --min-temporal-gap 4`

默认输出：

```text
output/train/libero-plus/opqd-v2-train560-split2026/seed-000/<suite>/default/
├── config.json
├── run.json
├── status.json
├── metrics.jsonl
├── train.log
└── checkpoint-step-000070/
    ├── adapter/
    └── trainer_state.pt
```

每 70 steps 保存一次，最终 checkpoint 为 `checkpoint-step-000700`。

## 4. suzhou-C 四 suite 并行

```bash
for spec in 'libero_spatial 0 31000 31001' 'libero_object 1 31010 31011' 'libero_goal 2 31020 31021' 'libero_10 3 31030 31031'; do
  set -- $spec
  tmux new-session -d -s "opqd-s0-$1" \
    "cd /lumos-vePFS/suda/ruan/QuantVLA && ./train_quantvla_opqd.sh --suite $1 --gpu $2 --env-port $3 --clean-env-port $4 --seed 0 --run-name default"
done
```

查看状态：

```bash
./monitor_eval.sh --once --opqd-train-seed 0
tmux ls
tail -f output/train/libero-plus/opqd-v2-train560-split2026/seed-000/libero_spatial/default/train.log
```

## 5. 断点恢复

```bash
./train_quantvla_opqd.sh --suite libero_spatial --gpu 0 \
  --env-port 31000 --clean-env-port 31001 --seed 0 \
  --resume-from output/train/libero-plus/opqd-v2-train560-split2026/seed-000/libero_spatial/default/checkpoint-step-000350
```

必须保持 suite、seed、manifest 和关键超参不变；程序会恢复 adapter、optimizer、scheduler、随机状态和 task schedule。

## 6. 评测训练后的 adapter

```bash
./eval_quantvla_opqd.sh \
  --benchmark libero-plus \
  --suite libero_spatial \
  --gpu 4 --port 31240 \
  --checkpoint output/train/libero-plus/opqd-v2-train560-split2026/seed-000/libero_spatial/default/checkpoint-step-000700 \
  --train-seed 0 --eval-seed 2026 --run-name default
```

默认输出：

```text
output/eval/libero-plus/test560-split2026/opqd-v2/seed-000/<suite>/default/
```

评测 Standard LIBERO 只需改为 `--benchmark libero`；指定其他卡、端口、output、ckpt 的方式与基线完全相同。已有相同结果时显式加 `--resume`，不要直接覆盖。

正式 eval 默认不保存视频。只在需要检查行为时加 `--save-video`，并使用独立输出名：

```bash
./eval_quantvla_opqd.sh --benchmark libero-plus --suite libero_spatial \
  --gpu 4 --port 31240 --checkpoint /path/to/checkpoint --train-seed 0 \
  --save-video --run-name video-debug
```

视频路径会写入 `episodes.jsonl` 的 `video_path`；不加 `--save-video` 时不会生成 rollout 视频。

## 7. Seeds、监控与报告

seed 1/2 只需同时改变 `--seed`、端口和输出自动生成的 `seed-XXX` 路径。评测时对应使用 `--train-seed 1` 或 `2`；eval seed 保持 2026，确保方法间成对比较。

```bash
./monitor_eval.sh --refresh-seconds 20 --opqd-train-seed 0
./collect_eval.sh --opqd-train-seed 0
./collect_eval.sh --opqd-train-seed 0 --require-complete
```

smoke test 必须使用独立目录，不能混入正式输出：

```bash
./train_quantvla_opqd.sh --suite libero_spatial --gpu 0 --env-port 31900 \
  --clean-env-port 31901 --seed 0 --episodes 1 --updates-per-episode 1 \
  --output-dir output/smoke/opqd-v2/spatial
```
