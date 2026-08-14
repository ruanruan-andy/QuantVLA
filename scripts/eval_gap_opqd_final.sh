#!/bin/bash
set -euo pipefail

# Evaluate the final GAP-OPQD adapter on the same first-24 calibration manifest.
# Usage: eval_gap_opqd_final.sh <suite> <adapter_dir> <server_port>

TASK="${1:?suite is required}"
ADAPTER_PATH="${2:?adapter directory is required}"
SERVER_PORT="${3:?inference server port is required}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$(cd "$(dirname "$ADAPTER_PATH")/.." && pwd)"
EVAL_DIR="$REPO_ROOT/output/libero-plus/groot-gap-opqd-w4a8/$TASK"
SERVER_LOG="$TRAIN_DIR/final_eval_server.log"

case "$TASK" in
    libero_spatial|libero_goal|libero_object|libero_10) ;;
    *) echo "Unsupported suite: $TASK" >&2; exit 1 ;;
esac
if [[ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
    echo "Final adapter is missing: $ADAPTER_PATH" >&2
    exit 1
fi
case "$TRAIN_DIR" in
    "$REPO_ROOT/output/gap-opqd-ood/"*) ;;
    *) echo "Refusing unexpected training directory: $TRAIN_DIR" >&2; exit 1 ;;
esac

mkdir -p "$EVAL_DIR"
(
    export PYTHONUNBUFFERED=1
    export GR00T_PORT="$SERVER_PORT"
    exec "$REPO_ROOT/run_gap_opqd_inference.sh" "$TASK" "$ADAPTER_PATH"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

SERVER_READY=0
for _ in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Inference server exited before becoming ready; see $SERVER_LOG" >&2
        exit 1
    fi
    if grep -q "Server is ready and listening" "$SERVER_LOG"; then
        SERVER_READY=1
        break
    fi
    sleep 1
done
if [[ "$SERVER_READY" != 1 ]]; then
    echo "Timed out waiting for inference server; see $SERVER_LOG" >&2
    exit 1
fi

GR00T_PORT="$SERVER_PORT" \
EVAL_MODEL_VARIANT="groot-gap-opqd-w4a8" \
LIBERO_PLUS_OUTPUT_DIR="$EVAL_DIR" \
    "$REPO_ROOT/run_libero_plus_eval.sh" "$TASK" --headless --no-save-video

if [[ ! -f "$EVAL_DIR/summary.json" ]]; then
    echo "Evaluation completed without summary: $EVAL_DIR/summary.json" >&2
    exit 1
fi

# Evaluation succeeded: retain only the final checkpoint requested by the user.
find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
    ! -name 'checkpoint-000100' -exec rm -rf -- {} +
echo "Final evaluation completed: $EVAL_DIR/summary.json"
echo "Retained checkpoint: $TRAIN_DIR/checkpoint-000100"
