#!/usr/bin/env python
"""Paired validation comparison for ShiftGuard rotation candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def load(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["key"]: row for row in payload["samples"]}


def paired_summary(
    baseline: dict[str, dict], candidate: dict[str, dict], rng: np.random.Generator,
    bootstrap: int, permutations: int,
) -> dict:
    if set(baseline) != set(candidate):
        raise ValueError("Candidate and baseline sample keys differ")
    keys = sorted(baseline)
    base = np.asarray([baseline[key]["action_discrepancy"] for key in keys])
    cand = np.asarray([candidate[key]["action_discrepancy"] for key in keys])
    delta = cand - base
    indices = rng.integers(0, len(delta), size=(bootstrap, len(delta)))
    boot = delta[indices].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutations, len(delta)))
    permuted = (signs * delta).mean(axis=1)
    observed = float(delta.mean())
    sign_flip_p = float((np.count_nonzero(np.abs(permuted) >= abs(observed)) + 1) / (permutations + 1))
    try:
        wilcoxon_p = float(stats.wilcoxon(delta, alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    domains = sorted({baseline[key]["domain"] for key in keys})
    by_domain = {}
    for domain in domains:
        mask = np.asarray([baseline[key]["domain"] == domain for key in keys])
        by_domain[domain] = {
            "count": int(mask.sum()),
            "baseline_mean": float(base[mask].mean()),
            "candidate_mean": float(cand[mask].mean()),
            "mean_delta": float(delta[mask].mean()),
            "relative_change_percent": float(100.0 * delta[mask].mean() / base[mask].mean()),
            "improved_samples": int(np.count_nonzero(delta[mask] < 0)),
        }
    return {
        "count": len(keys),
        "baseline_mean": float(base.mean()),
        "candidate_mean": float(cand.mean()),
        "mean_delta": observed,
        "relative_change_percent": float(100.0 * observed / base.mean()),
        "bootstrap_95_ci_mean_delta": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "sign_flip_p_two_sided": sign_flip_p,
        "wilcoxon_p_two_sided": wilcoxon_p,
        "improved_samples": int(np.count_nonzero(delta < 0)),
        "worsened_samples": int(np.count_nonzero(delta > 0)),
        "by_domain": by_domain,
    }


def main() -> None:
    args = parse_args()
    baseline = load(args.baseline)
    rng = np.random.default_rng(args.seed)
    result = {
        path.stem: paired_summary(
            baseline, load(path), rng, args.bootstrap, args.permutations
        )
        for path in args.candidates
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
