# OOD GAP-OPQD 代码与运行说明

## 1. 共享 first-24 数据集

训练与评测共同使用：

```text
configs/libero_plus/first_24_per_category.json
```

该 manifest 在四个 suite 间严格均分，每个 suite/category 取前 6 条；每个 LIBERO-Plus 分类合计 24 条，七类共 168 条。

## 2. 文件结构

- `gr00t/experiment/gap_opqd.py`：$q_t$、$r_t$、critical weights 和加权损失；
- `scripts/libero_plus_env_service.py`：LIBERO-Plus OOD calibration 环境；
- `scripts/libero_iid_env_service.py`：标准 clean LIBERO anchor 环境；
- `scripts/train_gap_opqd.py`：OOD student-controlled trainer；
- `run_gap_opqd.sh`：同时启动两个环境服务和模型训练；
- `run_gap_opqd_inference.sh`：加载 W4A8 QuantVLA 与 adapter；
- `run_libero_plus_eval.sh`：默认使用同一 first-25 manifest；
- `configs/gap_opqd/libero_spatial.yaml`：参考超参数；
- `tests/test_gap_opqd.py`：核心数学测试。

两个 simulator 服务运行在 `libero_test`，模型训练运行在 `groot_test`。服务通过 localhost RPC 交换 observation 和 action，避免 LIBERO、LIBERO-Plus 与模型依赖冲突。

## 3. 基础检查

```bash
cd /lumos-vePFS/suda/ruan/QuantVLA
source /root/Users/miniconda3/etc/profile.d/conda.sh
conda activate groot_test
export PYTHONPATH="$PWD"

pytest -q tests/test_gap_opqd.py
python scripts/train_gap_opqd.py --dry-run
```

## 4. 单步 GPU smoke test

```bash
cd /lumos-vePFS/suda/ruan/QuantVLA
CUDA_VISIBLE_DEVICES=0 ./run_gap_opqd.sh libero_spatial \
  --max-iterations 1 \
  --rollout-horizon 2 \
  --clean-anchor-rollout-horizon 1 \
  --task-ids 0 \
  --anchor-batch-size 1 \
  --save-every 1 \
  --output-dir output/gap-opqd-ood-smoke/libero_spatial
```

注意：`--task-ids` 必须是该 suite 的 first-25 manifest 中的真实 LIBERO-Plus task ID。省略该参数时训练器自动使用 manifest 全集。

## 5. 正式快速训练

```bash
cd /lumos-vePFS/suda/ruan/QuantVLA
CUDA_VISIBLE_DEVICES=0 ./run_gap_opqd.sh libero_spatial \
  --max-iterations 100 \
  --rollout-horizon 16 \
  --temporal-horizon 4 \
  --temporal-discount 0.9 \
  --alpha-q 1.0 \
  --beta-r 1.0 \
  --lambda-anchor 0.1
```

其他 suite：

```bash
CUDA_VISIBLE_DEVICES=1 ./run_gap_opqd.sh libero_goal
CUDA_VISIBLE_DEVICES=2 ./run_gap_opqd.sh libero_object
CUDA_VISIBLE_DEVICES=3 ./run_gap_opqd.sh libero_10
```

每个 suite 使用自己的 FP checkpoint、QuantVLA pack，以及共享 manifest 中对应的 7 或 6 条/分类。

## 6. 核心参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `sample_manifest` | first-24 | 训练/eval 共享 task 清单 |
| `rollout_horizon` | 16 | OOD student rollout 长度 $K$ |
| `temporal_horizon` | 4 | $r_t$ 的窗口 $H$ |
| `temporal_discount` | 0.9 | 折扣 $\gamma$ |
| `alpha_q` | 1.0 | 即时 disagreement 权重 |
| `beta_r` | 1.0 | temporal risk 权重 |
| `weight_max` | 5.0 | 最大状态权重 |
| `lambda_anchor` | 0.1 | clean IID anchor 强度 |
| `clean_anchor_rollout_horizon` | 4 | 每轮 clean anchor rollout 长度 |
| `lora_rank` | 16 | action-head Q/K/V LoRA rank |
| `learning_rate` | `5e-5` | LoRA 学习率 |
| `save_every` | 100 | 只保存最终 checkpoint 100 |

## 7. 消融

```bash
# Uniform OOD distillation
./run_gap_opqd.sh libero_spatial --alpha-q 0 --beta-r 0 --lambda-anchor 0

# OOD q_t-only
./run_gap_opqd.sh libero_spatial --alpha-q 1 --beta-r 0 --lambda-anchor 0

# OOD q_t+r_t, no clean anchor
./run_gap_opqd.sh libero_spatial --alpha-q 1 --beta-r 1 --lambda-anchor 0

# Full OOD GAP-OPQD
./run_gap_opqd.sh libero_spatial --alpha-q 1 --beta-r 1 --lambda-anchor 0.1
```

不同方法必须共享 manifest、task 顺序/seed、flow noise、训练步数、teacher queries 和 LoRA budget。

## 8. 加载 adapter 并在同一 first-24 上评测

启动推理服务：

```bash
CUDA_VISIBLE_DEVICES=0 GR00T_PORT=5570 ./run_gap_opqd_inference.sh \
  libero_spatial \
  output/gap-opqd-ood/libero_spatial/checkpoint-000100/adapter
```

另一个终端运行 calibration-set eval：

```bash
GR00T_PORT=5570 EVAL_MODEL_VARIANT=groot-gap-opqd-w4a8 \
  ./run_libero_plus_eval.sh libero_spatial --headless
```

`run_libero_plus_eval.sh` 默认读取与训练完全相同的 `first_24_per_category.json`。

## 9. Resume 与 tmux

训练器默认 `--resume True`：如果输出目录存在完整的 `checkpoint-*`，自动选择编号最大的 checkpoint，恢复 LoRA、optimizer 和 iteration。新格式 checkpoint 还恢复 task schedule、Python/NumPy/Torch RNG 和 flow-noise generator。为避免文件过大，clean replay 不写入 checkpoint，重启后通过 clean rollout 重新预热。正式配置的 `save_every=100`，训练期间保留当前恢复基点，最终 eval 成功后只留下 checkpoint 100。

显式指定 checkpoint：

```bash
./run_gap_opqd.sh libero_spatial \
  --resume-from-checkpoint output/gap-opqd-ood/libero_spatial/checkpoint-000020
```

恢复时，`metrics.jsonl` 会裁剪到 checkpoint iteration，删除尚未保存模型的尾部指标，防止重复 iteration。

当前四个后台训练位于同一个 tmux session。训练达到 100 后，各窗口会自动加载最终 adapter，在共享 first-24 manifest 上评测；summary 成功生成后自动删除旧 checkpoint：

```bash
tmux attach -t gap-opqd
```

窗口 0–3 分别对应 spatial、goal、object 和 libero_10。使用 `Ctrl-b n`/`Ctrl-b p` 切换窗口，`Ctrl-b d` 安全退出 tmux 而不停止训练。

## 10. 输出和限制

训练目录包含：

- `config.json`：完整训练配置；
- `selected_ood_tasks.json`：本 suite 实际蒸馏 task；
- `metrics.jsonl`：逐 iteration 的 category、$q_t$、$r_t$、weights、loss 和 gradient；
- `checkpoint-*/adapter`：PEFT adapter；
- `checkpoint-*/trainer_state.pt`：optimizer state。

由于训练和 eval 使用同一批 OOD task，结果只能作为快速 calibration-set 验证，不能作为未见 OOD 泛化结论。
