#!/bin/bash
set -euo pipefail

SESSION_NAME="${GAP_OPQD_TMUX_SESSION:-gap-opqd}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    echo "Attach with: tmux attach -t $SESSION_NAME" >&2
    exit 1
fi

launch_window() {
    local window="$1"
    local suite="$2"
    local gpu="$3"
    local ood_port="$4"
    local clean_port="$5"
    local eval_port="$6"
    local output_dir="$REPO_ROOT/output/gap-opqd-ood/$suite"
    mkdir -p "$output_dir"
    tmux send-keys -t "$SESSION_NAME:$window" \
        "set -o pipefail; cd '$REPO_ROOT' || exit 1; export CUDA_VISIBLE_DEVICES=$gpu GAP_OPQD_ENV_PORT=$ood_port GAP_OPQD_CLEAN_ENV_PORT=$clean_port GAP_OPQD_OUTPUT_DIR='$output_dir'; ./run_gap_opqd.sh '$suite' --save-every 100 2>&1 | tee -a '$output_dir/tmux.log' && ./scripts/eval_gap_opqd_final.sh '$suite' '$output_dir/checkpoint-000100/adapter' $eval_port 2>&1 | tee -a '$output_dir/final_eval.log'" \
        C-m
}

tmux new-session -d -s "$SESSION_NAME" -n spatial
tmux new-window -t "$SESSION_NAME" -n goal
tmux new-window -t "$SESSION_NAME" -n object
tmux new-window -t "$SESSION_NAME" -n libero10
tmux set-option -t "$SESSION_NAME" remain-on-exit on
tmux set-option -t "$SESSION_NAME" history-limit 50000

launch_window spatial libero_spatial 4 5600 5601 5700
launch_window goal libero_goal 5 5610 5611 5710
launch_window object libero_object 6 5620 5621 5720
launch_window libero10 libero_10 7 5630 5631 5730

echo "Started GAP-OPQD in tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
tmux list-windows -t "$SESSION_NAME"
