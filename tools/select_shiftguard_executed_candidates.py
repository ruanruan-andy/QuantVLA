#!/usr/bin/env python
"""Select uniform-W4 candidates with separate localization and control-safety signals."""

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
    parser.add_argument("--min-improved-domains", type=int, default=5)
    parser.add_argument("--max-domain-regression", type=float, default=0.05)
    return parser.parse_args()


def expand(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        paths.extend(Path(path) for path in sorted(glob.glob(value)))
    return paths


def trajectory_rows(payload: dict) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in payload["samples"]:
        grouped[row["trajectory_id"]].append(row)
    result = []
    for trajectory_id, rows in sorted(grouped.items()):
        full32 = np.asarray([row["full32_action_discrepancy"] for row in rows], dtype=float)
        executed = np.asarray([row["normalized_action_discrepancy"] for row in rows], dtype=float)
        gripper = np.asarray([row["gripper_disagreement"] for row in rows], dtype=float)
        critical = np.asarray([
            row["gripper_disagreement"] for row in rows if row["gripper_switch"] or row["tail_state"]
        ], dtype=float)
        result.append({
            "trajectory_id": trajectory_id,
            "split": rows[0]["split"],
            "domain": rows[0]["domain"],
            "full32_mean": float(full32.mean()),
            "executed_mean": float(executed.mean()),
            "executed_max": float(executed.max()),
            "gripper_mean": float(gripper.mean()),
            "critical_gripper": float(critical.mean()) if len(critical) else float(gripper.mean()),
        })
    return result


def summarize(rows: list[dict], split: str, kappa: float) -> dict:
    selected = [row for row in rows if row["split"] == split]
    result = {"trajectories": len(selected)}
    for key in ("full32_mean", "executed_mean", "executed_max", "gripper_mean", "critical_gripper"):
        values = np.asarray([row[key] for row in selected], dtype=float)
        result[key] = float(values.mean())
        result[f"{key}_std"] = float(values.std())
    result["localization_robust"] = result["full32_mean"] + kappa * result["full32_mean_std"]
    result["executed_by_domain"] = {row["domain"]: row["executed_mean"] for row in selected}
    return result


def relative(value: float, baseline: float) -> float:
    return (value - baseline) / max(abs(baseline), 1e-12)


def main() -> None:
    args = parse_args()
    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_rows = trajectory_rows(baseline_payload)
    baseline = {split: summarize(baseline_rows, split, args.kappa) for split in ("calibration", "validation")}
    candidates = []
    for path in expand(args.candidates):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = trajectory_rows(payload)
        calibration = summarize(rows, "calibration", args.kappa)
        calibration_safe = (
            calibration["executed_max"] <= baseline["calibration"]["executed_max"]
            and calibration["critical_gripper"] <= baseline["calibration"]["critical_gripper"]
        )
        candidates.append({
            "candidate": path.stem,
            "path": str(path),
            "calibration": calibration,
            "calibration_safe": calibration_safe,
        })
    candidates.sort(key=lambda row: row["calibration"]["localization_robust"])
    shortlist = [row for row in candidates if row["calibration_safe"]][: args.top_k]

    validations = []
    base = baseline["validation"]
    for row in shortlist:
        payload = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
        validation = summarize(trajectory_rows(payload), "validation", args.kappa)
        relative_by_domain = {
            domain: relative(value, base["executed_by_domain"][domain])
            for domain, value in validation["executed_by_domain"].items()
        }
        improved_domains = sum(value < 0 for value in relative_by_domain.values())
        deltas = {
            key: relative(validation[key], base[key])
            for key in ("full32_mean", "executed_mean", "executed_max", "gripper_mean", "critical_gripper")
        }
        eligible = (
            deltas["full32_mean"] < 0
            and deltas["executed_mean"] < 0
            and deltas["executed_max"] <= 0
            and deltas["critical_gripper"] <= 0
            and improved_domains >= args.min_improved_domains
            and max(relative_by_domain.values()) <= args.max_domain_regression
        )
        validations.append({
            "candidate": row["candidate"],
            "path": row["path"],
            "calibration": row["calibration"],
            "validation": validation,
            "relative": deltas,
            "relative_by_domain": relative_by_domain,
            "improved_domains": improved_domains,
            "eligible": eligible,
        })
    eligible = [row for row in validations if row["eligible"]]
    selected = min(eligible, key=lambda row: row["validation"]["executed_max"]) if eligible else {
        "candidate": "baseline",
        "reason": "No candidate passed executed-action and critical-gripper safety gates",
    }
    result = {
        "parameters": {
            "kappa": args.kappa,
            "top_k": args.top_k,
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
    print(json.dumps({
        "shortlist": [row["candidate"] for row in shortlist],
        "validation": [
            {
                "candidate": row["candidate"],
                "relative_pct": {key: 100 * value for key, value in row["relative"].items()},
                "improved_domains": row["improved_domains"],
                "eligible": row["eligible"],
            }
            for row in validations
        ],
        "selected": selected["candidate"],
    }, indent=2))


if __name__ == "__main__":
    main()
