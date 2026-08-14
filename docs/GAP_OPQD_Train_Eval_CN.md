# GAP-OPQD Train 与 Eval 指令

## 1. 工作目录与数据协议

```bash
ssh suzhou-I
cd /lumos-vePFS/suda/ruan/QuantVLA
```

训练和评测共同使用：

```text
configs/libero_plus/first_24_per_category.json
```

七个 LIBERO-Plus 分类各取 24 条，四个 suite 严格均分：每个 suite/category 取 task index 前 6 条，因此每个 suite 42 条、四个 suite 共 168 条。

训练和 eval 使用完全相同的 task IDs。结果属于 calibration-set evaluation，不能表述为未见 OOD 泛化。

## 2. 推荐：四卡 tmux 训练并自动评测

确认 GPU 4–7 可用：

```bash
nvidia-smi -i 4,5,6,7
```

启动四个 suite：

```bash
./scripts/launch_gap_opqd_tmux.sh
```

分配关系：

| tmux 窗口 | Suite | GPU | OOD/Clean 端口 | Eval 端口 |
|---:|---|---:|---|---:|
| 0 `spatial` | `libero_spatial` | 4 | 5600/5601 | 5700 |
| 1 `goal` | `libero_goal` | 5 | 5610/5611 | 5710 |
| 2 `object` | `libero_object` | 6 | 5620/5621 | 5720 |
| 3 `libero10` | `libero_10` | 7 | 5630/5631 | 5730 |

启动器执行完整流水线：

1. 自动寻找当前 suite 编号最大的完整 checkpoint；
2. 恢复 LoRA、optimizer、iteration、task schedule 和随机状态；
3. 训练到 iteration 100；
4. 保存 `checkpoint-000100`；
5. 自动加载最终 adapter 并运行同一 first-24 eval；
6. eval 成功生成 `summary.json` 后删除旧 checkpoint，只保留 checkpoint 100。

如果 `gap-opqd` session 已经存在，启动器会拒绝创建重复任务。

## 3. 查看 tqdm 进度

进入 tmux：

```bash
tmux attach -t gap-opqd
```

常用按键：

```text
Ctrl-b 0/1/2/3    切换到指定 suite
Ctrl-b n          下一个窗口
Ctrl-b p          上一个窗口
Ctrl-b d          退出查看，后台训练继续
```

不进入 tmux，直接读取窗口：

```bash
tmux capture-pane -p -t gap-opqd:spatial | tail -n 20
tmux capture-pane -p -t gap-opqd:goal | tail -n 20
tmux capture-pane -p -t gap-opqd:object | tail -n 20
tmux capture-pane -p -t gap-opqd:libero10 | tail -n 20
```

查看结构化训练指标：

```bash
for suite in libero_spatial libero_goal libero_object libero_10; do
  echo "===== $suite ====="
  tail -n 1 "output/gap-opqd-ood/$suite/metrics.jsonl"
done
```

查看 GPU：

```bash
watch -n 3 nvidia-smi -i 4,5,6,7
```

## 4. 单独训练一个 suite

以 `libero_spatial` 为例：

```bash
CUDA_VISIBLE_DEVICES=4 \
GAP_OPQD_ENV_PORT=5600 \
GAP_OPQD_CLEAN_ENV_PORT=5601 \
GAP_OPQD_OUTPUT_DIR="$PWD/output/gap-opqd-ood/libero_spatial" \
./run_gap_opqd.sh libero_spatial \
  --max-iterations 100 \
  --save-every 100 \
  --rollout-horizon 16 \
  --temporal-horizon 4 \
  --temporal-discount 0.9 \
  --alpha-q 1.0 \
  --beta-r 1.0 \
  --lambda-anchor 0.1
```

默认 `--resume True`。如果输出目录存在完整 checkpoint，会自动从最新 checkpoint 继续。

显式指定恢复点：

```bash
./run_gap_opqd.sh libero_spatial \
  --resume-from-checkpoint \
  output/gap-opqd-ood/libero_spatial/checkpoint-000030 \
  --save-every 100
```

恢复时，`metrics.jsonl` 会裁剪到 checkpoint iteration，避免出现重复 iteration。clean replay 不写入 checkpoint，恢复后由新的短 clean rollout 自动预热。

## 5. 自动最终 Eval

使用 `launch_gap_opqd_tmux.sh` 时无需手动操作。训练完成后，每个窗口自动运行：

```bash
./scripts/eval_gap_opqd_final.sh \
  <suite> \
  output/gap-opqd-ood/<suite>/checkpoint-000100/adapter \
  <eval_port>
```

该脚本会：

- 启动 W4A8 QuantVLA + GAP-OPQD LoRA inference server；
- 等待服务 ready；
- 在共享 first-24 manifest 上运行 headless eval；
- 不保存 rollout 视频；
- 成功后停止 inference server；
- 删除 checkpoint 100 以外的旧 checkpoint。

## 6. 手动评测最终 checkpoint

以 spatial 为例：

```bash
CUDA_VISIBLE_DEVICES=4 \
./scripts/eval_gap_opqd_final.sh \
  libero_spatial \
  output/gap-opqd-ood/libero_spatial/checkpoint-000100/adapter \
  5700
```

其他 suite：

```bash
CUDA_VISIBLE_DEVICES=5 ./scripts/eval_gap_opqd_final.sh \
  libero_goal output/gap-opqd-ood/libero_goal/checkpoint-000100/adapter 5710

CUDA_VISIBLE_DEVICES=6 ./scripts/eval_gap_opqd_final.sh \
  libero_object output/gap-opqd-ood/libero_object/checkpoint-000100/adapter 5720

CUDA_VISIBLE_DEVICES=7 ./scripts/eval_gap_opqd_final.sh \
  libero_10 output/gap-opqd-ood/libero_10/checkpoint-000100/adapter 5730
```

## 7. 输出位置

训练输出：

```text
output/gap-opqd-ood/<suite>/
├── config.json
├── selected_ood_tasks.json
├── metrics.jsonl
├── tmux.log
├── final_eval.log
├── final_eval_server.log
└── checkpoint-000100/
    ├── adapter/
    └── trainer_state.pt
```

最终评测：

```text
output/libero-plus/groot-gap-opqd-w4a8/<suite>/episodes.jsonl
output/libero-plus/groot-gap-opqd-w4a8/<suite>/summary.json
```

汇总查看：

```bash
for suite in libero_spatial libero_goal libero_object libero_10; do
  echo "===== $suite ====="
  python -m json.tool \
    "output/libero-plus/groot-gap-opqd-w4a8/$suite/summary.json"
done
```

## 8. 停止与重新启动

安全停止当前 tmux 训练：

```bash
tmux send-keys -t gap-opqd:spatial C-c
tmux send-keys -t gap-opqd:goal C-c
tmux send-keys -t gap-opqd:object C-c
tmux send-keys -t gap-opqd:libero10 C-c
```

确认 trainer 和环境服务退出后，可以删除旧 tmux session：

```bash
tmux kill-session -t gap-opqd
```

再次运行 `./scripts/launch_gap_opqd_tmux.sh` 时会从最新完整 checkpoint 自动恢复。

## 9. 不要删除 DuQuant pack

以下目录是 QuantVLA W4A8 的预计算旋转/排列 pack，不是临时日志：

```text
model/quantvla/groot-n1.5/libero_spatial/duquant_pack
model/quantvla/groot-n1.5/libero_goal/duquant_pack
model/quantvla/groot-n1.5/libero_object/duquant_pack
model/quantvla/groot-n1.5/libero_10/duquant_pack
```

删除后模型会在下次启动时重新执行 116 个层的 `pack_weight()`，增加启动时间并降低实验复现的稳定性。当前训练和最终 eval 完成前不要删除。

## 10. Monitor 与 Collect

实时查看三模型的四-suite、七分类、真实进程状态和数据一致性 warning：

```bash
./monitor_eval.sh
```

只查看一次：

```bash
./monitor_eval.sh --once
```

生成允许 partial 的中间报告：

```bash
./collect_eval.sh \
  --output-dir output/summary/gap-opqd-first24
```

生成最终报告时必须启用严格模式：

```bash
./collect_eval.sh \
  --require-complete \
  --output-dir output/summary/gap-opqd-first24-final
```

严格模式要求选中模型全部达到每 suite `42/42`、总计 `168/168`，并且没有 JSON 损坏、重复/超量记录或 `episodes.jsonl` 与 `summary.json` 不一致。

完整字段、输出文件和故障处理见：

```text
docs/GAP_OPQD_Monitor_Collect_CN.md
```
