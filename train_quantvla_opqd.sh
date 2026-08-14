#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/scripts/experiment_common.sh"

usage() {
    cat <<'EOF'
Train the QuantVLA-OPQD LoRA adapter on the balanced LIBERO-Plus subset.

Options:
  --suite NAME              suite to train (default: libero_spatial)
  --gpu INDEX               one physical GPU index (default: 0)
  --env-port PORT           LIBERO-Plus OOD service port (default: 5590)
  --clean-env-port PORT     clean LIBERO anchor service port (default: 5591)
  --output-root PATH        root for normalized outputs (default: ./output)
  --output-dir PATH         exact train directory; overrides output-root/run-name
  --run-name NAME           final output component (default: default)
  --manifest PATH           LIBERO-Plus selection manifest
  --resume-from PATH        explicit checkpoint directory
  --max-iterations N        training iterations (default: 100)
  --save-every N            checkpoint interval (default: 100)
  --offline                 use only locally cached Hugging Face files
  --dry-run                 validate and print command without executing
  --help                    show this message
  -- ARGS...                forward advanced Tyro options to train_gap_opqd.py
EOF
}

SUITE="libero_spatial"
GPU="0"
ENV_PORT="5590"
CLEAN_ENV_PORT="5591"
OUTPUT_ROOT="$REPO_ROOT/output"
OUTPUT_DIR=""
RUN_NAME="default"
MANIFEST="$REPO_ROOT/configs/libero_plus/first_24_per_category.json"
RESUME_FROM=""
MAX_ITERATIONS="100"
SAVE_EVERY="100"
OFFLINE=0
DRY_RUN=0
EXTRA_ARGS=()

while (($#)); do
    case "$1" in
        --suite) quantvla_require_value "$1" "${2:-}"; SUITE="$2"; shift 2 ;;
        --gpu) quantvla_require_value "$1" "${2:-}"; GPU="$2"; shift 2 ;;
        --env-port) quantvla_require_value "$1" "${2:-}"; ENV_PORT="$2"; shift 2 ;;
        --clean-env-port) quantvla_require_value "$1" "${2:-}"; CLEAN_ENV_PORT="$2"; shift 2 ;;
        --output-root) quantvla_require_value "$1" "${2:-}"; OUTPUT_ROOT="$2"; shift 2 ;;
        --output-dir|--output) quantvla_require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --run-name) quantvla_require_value "$1" "${2:-}"; RUN_NAME="$2"; shift 2 ;;
        --manifest) quantvla_require_value "$1" "${2:-}"; MANIFEST="$2"; shift 2 ;;
        --resume-from|--resume-from-checkpoint) quantvla_require_value "$1" "${2:-}"; RESUME_FROM="$2"; shift 2 ;;
        --max-iterations) quantvla_require_value "$1" "${2:-}"; MAX_ITERATIONS="$2"; shift 2 ;;
        --save-every) quantvla_require_value "$1" "${2:-}"; SAVE_EVERY="$2"; shift 2 ;;
        --offline) OFFLINE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

quantvla_validate_suite "$SUITE"
quantvla_validate_gpu "$GPU"
quantvla_validate_port "$ENV_PORT"
quantvla_validate_port "$CLEAN_ENV_PORT"
quantvla_validate_run_name "$RUN_NAME"
if [[ ! "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ || ! "$SAVE_EVERY" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-iterations and --save-every must be positive integers" >&2
    exit 2
fi

OUTPUT_ROOT="$(quantvla_abs_path "$OUTPUT_ROOT")"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$OUTPUT_ROOT/train/libero-plus/quantvla-opqd/$SUITE/$RUN_NAME"
else
    OUTPUT_DIR="$(quantvla_abs_path "$OUTPUT_DIR")"
fi
MANIFEST="$(quantvla_abs_path "$MANIFEST")"
[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST" >&2; exit 2; }
if [[ -n "$RESUME_FROM" ]]; then
    RESUME_FROM="$(quantvla_abs_path "$RESUME_FROM")"
    [[ -d "$RESUME_FROM" ]] || { echo "Resume checkpoint not found: $RESUME_FROM" >&2; exit 2; }
fi

TRAIN_ENV=(
    "CUDA_VISIBLE_DEVICES=$GPU"
    "GAP_OPQD_ENV_PORT=$ENV_PORT"
    "GAP_OPQD_CLEAN_ENV_PORT=$CLEAN_ENV_PORT"
    "GAP_OPQD_OUTPUT_DIR=$OUTPUT_DIR"
    "GAP_OPQD_SAMPLE_MANIFEST=$MANIFEST"
)
if (( OFFLINE )); then
    TRAIN_ENV+=(
        "HF_HUB_OFFLINE=1"
        "TRANSFORMERS_OFFLINE=1"
        "HF_HUB_DISABLE_TELEMETRY=1"
    )
fi
TRAIN_CMD=(
    "$REPO_ROOT/run_gap_opqd.sh" "$SUITE"
    --max-iterations "$MAX_ITERATIONS"
    --save-every "$SAVE_EVERY"
)
[[ -n "$RESUME_FROM" ]] && TRAIN_CMD+=(--resume-from-checkpoint "$RESUME_FROM")
TRAIN_CMD+=("${EXTRA_ARGS[@]}")

echo "Method:       quantvla-opqd"
echo "Train data:   libero-plus"
echo "Suite:        $SUITE"
echo "GPU:          $GPU"
echo "Env ports:    $ENV_PORT / $CLEAN_ENV_PORT"
echo "HF offline:   $OFFLINE"
echo "Output:       $OUTPUT_DIR"
echo "Training command:"
quantvla_print_command env "${TRAIN_ENV[@]}" "${TRAIN_CMD[@]}"

if (( DRY_RUN )); then
    echo "Dry run complete; no process started and no output created."
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
printf '%q ' env "${TRAIN_ENV[@]}" "${TRAIN_CMD[@]}" >"$OUTPUT_DIR/train_command.txt"
printf '\n' >>"$OUTPUT_DIR/train_command.txt"
set -o pipefail
env "${TRAIN_ENV[@]}" "${TRAIN_CMD[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
