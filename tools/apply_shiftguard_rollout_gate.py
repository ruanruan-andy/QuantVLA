#!/usr/bin/env python
"""Apply a paired micro-rollout acceptance gate and otherwise roll back to QuantVLA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import stats


CATEGORIES = (
    "Camera Viewpoints", "Robot Initial States", "Language Instructions",
    "Light Conditions", "Background Textures", "Sensor Noise", "Objects Layout",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-tie", action="store_true")
    return parser.parse_args()


def subset(path: Path, per_category: int) -> dict[int, dict]:
    effective = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            effective[int(row["task_index"])] = row
    selected = {}
    for category in CATEGORIES:
        rows = sorted(
            (row for row in effective.values() if row.get("category") == category),
            key=lambda row: int(row["task_index"]),
        )[:per_category]
        if len(rows) != per_category:
            raise RuntimeError(f"{path}: need {per_category} rows for {category}, found {len(rows)}")
        selected.update({int(row["task_index"]): row for row in rows})
    return selected


def main() -> None:
    args = parse_args()
    base_path = args.repo_root / "output" / "libero-plus" / args.baseline_model / args.suite / "episodes.jsonl"
    cand_path = args.repo_root / "output" / "libero-plus" / args.candidate_model / args.suite / "episodes.jsonl"
    base, cand = subset(base_path, args.per_category), subset(cand_path, args.per_category)
    if set(base) != set(cand):
        raise ValueError("Baseline and candidate task IDs differ")
    keys = sorted(base)
    errors = sum(base[key].get("error") is not None or cand[key].get("error") is not None for key in keys)
    base_successes = sum(bool(base[key]["success"]) for key in keys)
    cand_successes = sum(bool(cand[key]["success"]) for key in keys)
    candidate_only = sum(not base[key]["success"] and cand[key]["success"] for key in keys)
    baseline_only = sum(base[key]["success"] and not cand[key]["success"] for key in keys)
    discordant = candidate_only + baseline_only
    p_value = float(stats.binomtest(candidate_only, discordant, 0.5).pvalue) if discordant else 1.0
    accepted = errors == 0 and (
        cand_successes >= base_successes if args.allow_tie else cand_successes > base_successes
    )
    result = {
        "suite": args.suite,
        "per_category": args.per_category,
        "episodes": len(keys),
        "baseline_model": args.baseline_model,
        "candidate_model": args.candidate_model,
        "baseline_successes": base_successes,
        "candidate_successes": cand_successes,
        "candidate_only_successes": candidate_only,
        "baseline_only_successes": baseline_only,
        "mcnemar_exact_p": p_value,
        "errors": errors,
        "allow_tie": args.allow_tie,
        "accepted": accepted,
        "selected_model": args.candidate_model if accepted else args.baseline_model,
        "reason": "candidate passed paired rollout gate" if accepted else "candidate failed paired rollout gate; rollback",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
