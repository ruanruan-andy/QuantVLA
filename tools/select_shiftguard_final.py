#!/usr/bin/env python
"""Two-stage calibration/validation selection with an explicit rollback gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-ood", type=float, default=0.8)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-id-regression", type=float, default=0.01)
    parser.add_argument("--min-improved-domains", type=int, default=5)
    parser.add_argument("--max-domain-regression", type=float, default=0.01)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_by_domain(payload: dict, exclude_id: bool = False) -> dict[str, float]:
    return {
        key: float(value["action_discrepancy"])
        for key, value in payload["by_domain"].items()
        if not (exclude_id and key == "ID")
    }


def main() -> None:
    args = parse_args()
    calibration = []
    for path in sorted(args.calibration_dir.glob("*.json")):
        payload = load(path)
        domains = mean_by_domain(payload, exclude_id=True)
        ood = np.asarray(list(domains.values()), dtype=float)
        id_loss = float(payload["by_domain"]["ID"]["action_discrepancy"])
        robust_ood = float(ood.mean() + args.kappa * ood.std())
        score = float((1.0 - args.alpha_ood) * id_loss + args.alpha_ood * robust_ood)
        calibration.append({
            "candidate": path.stem,
            "score": score,
            "id": id_loss,
            "ood_mean": float(ood.mean()),
            "ood_std": float(ood.std()),
            "robust_ood": robust_ood,
        })
    calibration.sort(key=lambda row: row["score"])
    baseline_cal = next(row for row in calibration if row["candidate"] == "baseline")
    shortlist = [row for row in calibration if row["candidate"] != "baseline"][: args.top_k]

    baseline_val = load(args.validation_dir / "baseline.json")
    base_domains = mean_by_domain(baseline_val)
    base_mean = float(np.mean(list(base_domains.values())))
    validation = []
    for row in shortlist:
        payload = load(args.validation_dir / f"{row['candidate']}.json")
        domains = mean_by_domain(payload)
        relative_by_domain = {
            key: (domains[key] - base_domains[key]) / base_domains[key]
            for key in sorted(base_domains)
        }
        val_mean = float(np.mean(list(domains.values())))
        id_regression = (row["id"] - baseline_cal["id"]) / baseline_cal["id"]
        improved_domains = sum(value < 0 for value in relative_by_domain.values())
        worst_domain_regression = max(relative_by_domain.values())
        eligible = (
            val_mean < base_mean
            and id_regression <= args.max_id_regression
            and improved_domains >= args.min_improved_domains
            and worst_domain_regression <= args.max_domain_regression
        )
        validation.append({
            **row,
            "validation_mean": val_mean,
            "validation_relative_change": (val_mean - base_mean) / base_mean,
            "id_relative_change": id_regression,
            "improved_domains": improved_domains,
            "worst_domain_regression": worst_domain_regression,
            "relative_change_by_domain": relative_by_domain,
            "eligible": eligible,
        })
    eligible = [row for row in validation if row["eligible"]]
    selected = min(eligible, key=lambda row: row["validation_mean"]) if eligible else {
        "candidate": "baseline", "reason": "No candidate passed the rollback gate"
    }
    result = {
        "parameters": vars(args) | {
            "calibration_dir": str(args.calibration_dir),
            "validation_dir": str(args.validation_dir),
            "output": str(args.output),
        },
        "calibration_baseline": baseline_cal,
        "calibration_ranking": calibration,
        "shortlist": shortlist,
        "validation_baseline_mean": base_mean,
        "validation": validation,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"shortlist": shortlist, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
