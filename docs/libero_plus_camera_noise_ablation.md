# LIBERO-Plus Camera and Sensor-Noise Ablation

This experiment compares two QuantVLA variants on the `Camera Viewpoints` and
`Sensor Noise` LIBERO-Plus categories across all four task suites.

| Variant | Quantization / scaling | Host GPUs | Ports |
| --- | --- | --- | --- |
| `block1` | `GR00T_DUQUANT_BLOCK=1`, independently generated pack and standard ATM/OHB calibration | `suzhou-C:4,5,6,7` | `5574-5577` |
| `sqrt` | block 64 and `sqrt(teacher/quant)` ATM and OHB calibration | `suzhou-I:0,1,2,4` | `5580-5583` |

The per-host suite order is `libero_spatial`, `libero_goal`, `libero_object`,
and `libero_10`. All inputs, packs, calibration JSONs, logs, summaries and
rollout videos are isolated below `experiments/libero_plus_camera_noise/`.

## Correctness checks

1. A block-1 server log must contain `block_in=1 block_out=1` for replaced
   layers. The pack directory must never be reused from a block-64 run.
2. Both variants require suite-specific ATM/OHB JSON produced from their own
   pack. `sqrt` calibration is selected with `--sqrt-scaling`.
3. Each evaluation passes exactly one category: `Camera Viewpoints` or
   `Sensor Noise`. LIBERO-Plus itself enforces one rollout per task.
4. The default sampling manifest selects the first 25 tasks per suite/category:
   100 rollouts per category and 200 rollouts per variant.

## Useful inspection commands

```bash
# Check progress and errors for one output directory.
tail -f experiments/libero_plus_camera_noise/block1/libero_spatial/camera/eval/libero_plus_eval_libero_spatial.log
cat experiments/libero_plus_camera_noise/sqrt/libero_goal/noise/eval/summary.json

# Verify actual block sizes in a block-1 inference log.
grep -E 'GR00T-DUQUANT.*REPLACED.*block_in=1 block_out=1' \
  experiments/libero_plus_camera_noise/block1/libero_spatial/server.log | head
```

## Resume an interrupted category

Run the same evaluator command again with `--resume`, retaining the same
output directory and server port. Completed tasks with no recorded error are
skipped using `episodes.jsonl`.

## Run commands

For each suite, create a calibration JSON before starting its server. The example
below shows block-1 spatial; replace suite, port, GPU and variant paths using
the allocation table above.

```bash
CUDA_VISIBLE_DEVICES=4 GR00T_DUQUANT_BLOCK=1 GR00T_DUQUANT_BLOCK_OUT=1 \\
GR00T_DUQUANT_PACKDIR=$PWD/experiments/libero_plus_camera_noise/block1/libero_spatial/duquant_pack \\
python tools/calibrate_atm_dit.py --teacher-checkpoint youliangtan/gr00t-n1.5-libero-spatial-posttrain \\
  --quant-checkpoint youliangtan/gr00t-n1.5-libero-spatial-posttrain --task-suite libero_spatial \\
  --steps 32 --calibrate-ohb 1 --out $PWD/experiments/libero_plus_camera_noise/block1/libero_spatial/atm_alpha_beta.json

CUDA_VISIBLE_DEVICES=4 GR00T_PORT=5574 GR00T_DUQUANT_BLOCK=1 GR00T_DUQUANT_BLOCK_OUT=1 \\
GR00T_DUQUANT_PACKDIR=$PWD/experiments/libero_plus_camera_noise/block1/libero_spatial/duquant_pack \\
GR00T_ATM_ALPHA_PATH=$PWD/experiments/libero_plus_camera_noise/block1/libero_spatial/atm_alpha_beta.json \\
./run_quantvla.sh libero_spatial

LIBERO_PLUS_OUTPUT_DIR=$PWD/experiments/libero_plus_camera_noise/block1/libero_spatial/camera/eval \\
GR00T_PORT=5574 ./run_libero_plus_eval.sh libero_spatial --model-variant groot-quantvla-w4a8 \\
  --headless --category Camera\\ Viewpoints

LIBERO_PLUS_OUTPUT_DIR=$PWD/experiments/libero_plus_camera_noise/block1/libero_spatial/noise/eval \\
GR00T_PORT=5574 ./run_libero_plus_eval.sh libero_spatial --model-variant groot-quantvla-w4a8 \\
  --headless --category Sensor\\ Noise
```

For sqrt, retain block 64 and add `--sqrt-scaling` to calibration. Its JSON
contains the applied coefficients; inference loads them and does not recompute them.

