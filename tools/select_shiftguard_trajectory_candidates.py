#!/usr/bin/env python
"""Select trajectory-aware rotation candidates with held-out safety rollback."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-max-risk-regression", type=float, default=0.02)
    parser.add_argument("--min-improved-domains", type=int, default=5)
    parser.add_argument("--max-domain-regression", type=float, default=0.05)
    return parser.parse_args()


def expand(values: list[str]) -> list[Path]:
    paths = []
    for value in values:
        paths.extend(Path(path) for path in sorted(glob.glob(value)))
    return paths


def trajectory_rows(payload: dict) -> list[dict]:
    grouped = defaultdict(list)
    for row in payload["samples"]:
        grouped[row["trajectory_id"]].append(row)
    result = []
    for trajectory_id, rows in sorted(grouped.items()):
        values = np.asarray([float(row["normalized_action_discrepancy"]) for row in rows])
        result.append({
            "trajectory_id": trajectory_id,
            "split": rows[0]["split"],
            "domain": rows[0]["domain"],
            "success": rows[0]["trajectory_success"],
            "mean": float(values.mean()),
            "max": float(values.max()),
        })
    return result


def summarize(rows: list[dict], split: str, kappa: float) -> dict:
    selected = [row for row in rows if row["split"] == split]
    means = np.asarray([row["mean"] for row in selected])
    maxima = np.asarray([row["max"] for row in selected])
    return {
        "trajectories": len(selected),
        "mean": float(means.mean()),
        "std": float(means.std()),
        "robust_mean": float(means.mean() + kappa * means.std()),
        "max_risk_mean": float(maxima.mean()),
        "by_domain": {row["domain"]: row["mean"] for row in selected},
    }


def main() -> None:
    args = parse_args()
    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_rows = trajectory_rows(baseline_payload)
    baseline = {
        split: summarize(baseline_rows, split, args.kappa)
        for split in ("calibration", "validation")
    }
    candidates = []
    for path in expand(args.candidates):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = trajectory_rows(payload)
        calibration = summarize(rows, "calibration", args.kappa)
        calibration_safe = calibration["max_risk_mean"] <= baseline["calibration"]["max_risk_mean"] * (1.0 + args.max_max_risk_regression)
        candidates.append({
            "candidate": path.stem,
            "path": str(path),
            "calibration": calibration,
            "calibration_safe": calibration_safe,
        })
    candidates.sort(key=lambda row: row["calibration"]["robust_mean"])
    shortlist = [row for row in candidates if row["calibration_safe"]][: args.top_k]

    validations = []
    for row in shortlist:
        payload = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
        validation = summarize(trajectory_rows(payload), "validation", args.kappa)
        relative_by_domain = {
            domain: (value - baseline["validation"]["by_domain"][domain]) / baseline["validation"]["by_domain"][domain]
            for domain, value in validation["by_domain"].items()
        }
        improved_domains = sum(value < 0 for value in relative_by_domain.values())
        max_safe = validation["max_risk_mean"] <= baseline["validation"]["max_risk_mean"] * (1.0 + args.max_max_risk_regression)
        eligible = (
            validation["robust_mean"] < baseline["validation"]["robust_mean"]
            and validation["mean"] < baseline["validation"]["mean"]
            and max_safe
            and improved_domains >= args.min_improved_domains
            and max(relative_by_domain.values()) <= args.max_domain_regression
        )
        validations.append({
            "candidate": row["candidate"],
            "calibration": row["calibration"],
            "validation": validation,
            "validation_relative_mean": (validation["mean"] - baseline["validation"]["mean"]) / baseline["validation"]["mean"],
            "validation_relative_robust": (validation["robust_mean"] - baseline["validation"]["robust_mean"]) / baseline["validation"]["robust_mean"],
            "validation_relative_max": (validation["max_risk_mean"] - baseline["validation"]["max_risk_mean"]) / baseline["validation"]["max_risk_mean"],
            "relative_by_domain": relative_by_domain,
            "improved_domains": improved_domains,
            "max_safe": max_safe,
            "eligible": eligible,
        })
    eligible = [row for row in validations if row["eligible"]]
    selected = min(eligible, key=lambda row: row["validation"]["robust_mean"]) if eligible else {
        "candidate": "baseline", "reason": "No trajectory candidate passed the held-out safety gate"
    }
    result = {
        "parameters": {
            "kappa": args.kappa,
            "top_k": args.top_k,
            "max_max_risk_regression": args.max_max_risk_regression,
            "min_improved_domains": args.min_improved_domains,
            "max_domain_regression": args.max_domain_regression,
        },
        "baseline": baseline,
        "calibration_ranking": candidates,
        "shortlist": shortlist,
        "validation": validations,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "shortlist": shortlist, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
