# LIBERO-Plus integration

LIBERO-Plus is connected to the same GR00T inference service used by Standard
LIBERO. The maintained user workflow is documented at the repository root:

- `docs/FP16_QUANTVLA_EVAL_CN.md`: FP16 and QuantVLA evaluation;
- `docs/QUANTVLA_OPQD_TRAIN_EVAL_CN.md`: QuantVLA-OPQD training/evaluation;
- `docs/METHOD_CN.md`: implementation and objective;
- `docs/EXPERIMENTS_CN.md`: experiment matrix and result templates.

The deterministic fast manifest is
`configs/libero_plus/first_24_per_category.json`. It selects six tasks from
each `(suite, category)` pair: 42 rollouts per suite, 24 per category across
four suites, and 168 per method.

Use the user-facing launchers instead of manually pairing server and evaluator
terminals:

```bash
./eval_fp16.sh --benchmark libero-plus --suite libero_spatial --gpu 0 --port 5700 --run-name main-v1
./eval_quantvla.sh --benchmark libero-plus --suite libero_spatial --gpu 1 --port 5710 --run-name main-v1
./eval_quantvla_opqd.sh --benchmark libero-plus --suite libero_spatial --gpu 2 --port 5720 \
  --checkpoint output/train/libero-plus/quantvla-opqd/libero_spatial/opqd-v1/checkpoint-000100 \
  --run-name main-v1
```

Outputs are isolated below
`output/eval/<benchmark>/<method>/<suite>/<run-name>/`. Existing
`episodes.jsonl` is never overwritten unless `--resume` is explicit.
