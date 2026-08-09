#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MONITOR_PYTHON:-/root/Users/miniconda3/envs/libero_test/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN=python
fi

exec "$PYTHON_BIN" "$REPO_ROOT/scripts/eval_monitor.py" \
    --repo-root "$REPO_ROOT" "$@"

