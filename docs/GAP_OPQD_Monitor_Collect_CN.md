# GAP-OPQD Eval Monitor 与指标收集

## 1. 适用范围与数据协议

本文档用于比较以下三个模型：

- `groot-fp16`
- `groot-quantvla-w4a8`
- `groot-gap-opqd-w4a8`

默认读取：

```text
configs/libero_plus/first_24_per_category.json
```

该 manifest 包含七个 LIBERO-Plus 分类。每个 suite/category 取 6 个任务，因此：

- 每个 suite：`7 × 6 = 42` 条；
- 每个分类：`4 × 6 = 24` 条；
- 每个模型：`4 × 42 = 168` 条。

monitor 和 collect 都以 `episodes.jsonl` 为实时指标的事实来源；`summary.json` 用于一致性核对，不会在逐条记录被清空时静默回退到旧 summary。

## 2. 实时 Monitor

进入项目：

```bash
ssh suzhou-I
cd /lumos-vePFS/suda/ruan/QuantVLA
```

持续刷新：

```bash
./monitor_eval.sh
```

只打印一次：

```bash
./monitor_eval.sh --once
```

调整刷新间隔：

```bash
./monitor_eval.sh --interval 10
```

显式指定 manifest：

```bash
./monitor_eval.sh --once \
  --manifest configs/libero_plus/first_24_per_category.json
```

默认只显示三组主实验和 LIBERO-Plus。需要额外实验或 Standard LIBERO 时可显式选择：

```bash
./monitor_eval.sh --once \
  --models groot-fp16 groot-quantvla-w4a8 groot-gap-opqd-w4a8 \
  --benchmarks libero-plus libero
```

按 `Ctrl+C` 只退出 monitor，不会停止 tmux、推理服务或 eval。

## 3. 状态与指标含义

Suite 状态：

| 状态 | 含义 |
|---|---|
| `RUNNING` | 找到匹配 model/suite 的真实 eval 进程 |
| `STARTING` | 服务已启动，但还没有有效 episode |
| `DONE` | `episodes.jsonl` 已达到 manifest 目标数 |
| `STOPPED` | 存在部分结果，但没有对应 eval 进程 |
| `EMPTY` | 输出文件存在但为空，且没有运行进程 |
| `NOT STARTED` | 尚未找到输出文件或运行进程 |

分类状态：

- `FINAL`：完成数等于该分类目标数 24；
- `PARTIAL`：只完成部分任务，中间成功率不能作为最终提升；
- `N/A`：没有数据，不会错误显示为 `0%`。

计数规则：

- 进度统计已经写入的 episode，包括带 `error` 的 episode；
- 成功率以无 error 的有效 episode 为分母；
- error 单独计数；
- resume 产生相同 task ID 的重复行时，以文件中最后一条为准，并显示 duplicate warning；
- JSON 损坏行、超出 manifest 数量和 `summary/episodes` 不一致都会显示 warning。

运行状态直接从 `/proc` 中发现 `run_libero_plus_eval.py` 和 `inference_service.py`，不依赖手工维护的 `processes.tsv`。

## 4. 三模型对比

monitor 会显示两张对比表：

- 四个 suite：FP16、QuantVLA、GAP-OPQD；
- 七个分类：FP16、QuantVLA、GAP-OPQD。

单元格中的 `*` 表示该模型/分组未完成。`OPQD-Quant` 使用百分点（percentage points, pp），部分结果的差值仅作实时观察。

## 5. Collect：生成可保存的报告

允许部分结果，写入默认目录 `output/summary`：

```bash
./collect_eval.sh
```

建议为实验指定独立目录：

```bash
./collect_eval.sh \
  --output-dir output/summary/gap-opqd-first24
```

只有三模型全部完成且没有一致性 warning 时才生成最终报告：

```bash
./collect_eval.sh \
  --require-complete \
  --output-dir output/summary/gap-opqd-first24-final
```

严格模式会在写文件之前退出，因此不会用不完整数据覆盖最终报告。

## 6. Collect 输出文件

| 文件 | 内容 |
|---|---|
| `summary.json` | 完整结构化快照、manifest、partial/inconsistent 标记和 warning |
| `report.md` | 三模型的四-suite、七分类和对比报告 |
| `suites.csv` | 每模型/suite 的进度、成功率、CI、ETA 和运行状态 |
| `episodes.csv` | 有效解析出的逐 episode 明细 |
| `libero_plus_by_category.csv` | 七分类汇总 |
| `libero_plus_by_suite_category.csv` | suite × 分类汇总 |
| `libero_plus_by_difficulty.csv` | 难度汇总 |
| `libero_plus_by_suite_difficulty.csv` | suite × 难度汇总 |
| `libero_plus_by_category_difficulty.csv` | 分类 × 难度汇总 |
| `model_comparisons.csv` | 三模型完整/部分 suite 与分类对比 |
| `matched_task_comparisons.csv` | 三模型都有有效记录的相同 task ID 交集对比 |

所有文件先写入同目录临时文件，再以原子替换方式发布，避免读取到半写入 CSV/JSON。

## 7. Partial 与 matched-task 的区别

普通对比展示各模型当前所有已完成任务。若 OPQD 尚未完成，任务组成可能与完整 baseline 不同，不能直接声称提升。

`matched_task_comparisons.csv` 只保留三模型均有有效记录的相同 `(suite, task_index)`，适合观察运行中的同任务差异。它仍然不是最终结论；最终论文指标应以：

```text
每个模型 168/168
每个 suite 42/42
每个分类 24/24
errors = 0
inconsistent = false
```

为验收条件。

## 8. 常见 Warning

### episodes/summary mismatch

示例：

```text
episodes=0, summary=42
```

通常表示某个新 eval 使用 `resume=False` 重新打开了 `episodes.jsonl`，而旧 `summary.json` 尚未被新结果更新。此时旧 summary 不能与空 episodes 混用。应先确认是否存在意外重跑进程，再决定停止重跑或完整重评。

### duplicate/resumed records

说明同一 task ID 在 JSONL 中出现多次。monitor/collect 使用最后一条，但最终归档前应确认这是预期 resume 行为。

### STOPPED

说明存在部分结果但找不到对应 evaluator。先查看 tmux 与日志：

```bash
tmux list-windows -t gap-opqd
tmux capture-pane -p -t gap-opqd:goal | tail -n 40
tail -n 80 output/gap-opqd-ood/libero_goal/final_eval_server.log
```

不要仅因 server 进程存在就判断 eval 正常；以 evaluator、episode 递增和错误日志三者共同判断。
