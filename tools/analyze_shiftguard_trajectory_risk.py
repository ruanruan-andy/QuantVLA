#!/usr/bin/env python
"""Validate which trajectory-risk reduction best transfers across held-out tasks."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = ("initial_risk", "mean_risk", "max_risk", "tail_cvar", "critical_cvar", "trajectory_risk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def expand(values: list[str]) -> list[Path]:
    paths = []
    for value in values:
        paths.extend(Path(path) for path in sorted(glob.glob(value)))
    return paths


def permutation_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, count: int) -> tuple[float, float]:
    observed = float(stats.spearmanr(x, y).statistic)
    permuted = np.empty(count, dtype=float)
    for index in range(count):
        permuted[index] = stats.spearmanr(x, rng.permutation(y)).statistic
    p_value = float((np.count_nonzero(np.abs(permuted) >= abs(observed)) + 1) / (count + 1))
    return observed, p_value


def aggregate(trajectories: list[dict], split: str, metric: str) -> float:
    values = [float(row[metric]) for row in trajectories if row["split"] == split]
    return float(np.mean(values))


def cvar(values: list[float], fraction: float) -> float:
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.mean(np.sort(np.asarray(values, dtype=float))[-count:]))


def rebuild_trajectories(payload: dict, value_key: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in payload["samples"]:
        grouped[row["trajectory_id"]].append(row)
    result = []
    for trajectory_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["policy_step"])
        risks = [float(row[value_key]) for row in rows]
        growth = [max(risks[index] - risks[index - 1], 0.0) for index in range(1, len(risks))]
        critical = [
            float(row[value_key]) for row in rows if row["gripper_switch"] or row["tail_state"]
        ]
        tail_risk = cvar(risks, float(payload["cvar_fraction"]))
        growth_risk = cvar(growth, float(payload["cvar_fraction"])) if growth else 0.0
        critical_risk = cvar(critical, float(payload["cvar_fraction"])) if critical else 0.0
        result.append({
            "trajectory_id": trajectory_id,
            "split": rows[0]["split"],
            "success": rows[0]["trajectory_success"],
            "initial_risk": risks[0],
            "mean_risk": float(np.mean(risks)),
            "max_risk": float(np.max(risks)),
            "tail_cvar": tail_risk,
            "critical_cvar": critical_risk,
            "trajectory_risk": tail_risk + float(payload["growth_weight"]) * growth_risk + float(payload["critical_weight"]) * critical_risk,
        })
    return result


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    wins = sum(float(p > n) + 0.5 * float(p == n) for p in positive for n in negative)
    return float(wins / (len(positive) * len(negative)))


def main() -> None:
    args = parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in expand(args.inputs)]
    blocks = [payload for payload in payloads if payload["target"].startswith("dit:")]
    blocks.sort(key=lambda payload: int(payload["target"].split(":")[1]))
    rng = np.random.default_rng(args.seed)
    correlations = {}
    trajectory_views = {
        "executed_raw": {payload["target"]: payload["trajectories"] for payload in payloads},
        "executed_normalized": {
            payload["target"]: rebuild_trajectories(payload, "normalized_action_discrepancy")
            for payload in payloads
        },
    }
    optional_views = {
        "full32": "full32_action_discrepancy",
        "model7_normalized": "normalized_model7_action_discrepancy",
        "translation": "translation_discrepancy",
        "rotation": "rotation_discrepancy",
        "gripper_disagreement": "gripper_disagreement",
    }
    for view_name, value_key in optional_views.items():
        if all(value_key in row for payload in payloads for row in payload.get("samples", [])):
            trajectory_views[view_name] = {
                payload["target"]: rebuild_trajectories(payload, value_key)
                for payload in payloads
            }
    for view_name, views in trajectory_views.items():
        for metric in METRICS:
            calibration = np.asarray([aggregate(views[payload["target"]], "calibration", metric) for payload in blocks])
            validation = np.asarray([aggregate(views[payload["target"]], "validation", metric) for payload in blocks])
            rho, p_value = permutation_spearman(calibration, validation, rng, args.permutations)
            key = f"{view_name}.{metric}"
            correlations[key] = {
                "spearman_rho": rho,
                "permutation_p": p_value,
                "calibration": {payload["target"]: float(value) for payload, value in zip(blocks, calibration)},
                "validation": {payload["target"]: float(value) for payload, value in zip(blocks, validation)},
            }

    all_payload = next((payload for payload in payloads if payload["target"] == "all"), None)
    failure_prediction = {}
    if all_payload is not None:
        for view_name, views in trajectory_views.items():
            trajectories = views["all"]
            labels = np.asarray([not bool(row["success"]) for row in trajectories], dtype=int)
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in trajectories])
                correlation = stats.pointbiserialr(labels, values) if len(set(labels.tolist())) == 2 else None
                key = f"{view_name}.{metric}"
                failure_prediction[key] = {
                    "failure_auc": auc(labels, values),
                    "point_biserial_r": float(correlation.statistic) if correlation else None,
                    "point_biserial_p": float(correlation.pvalue) if correlation else None,
                    "failure_mean": float(values[labels == 1].mean()) if np.any(labels == 1) else None,
                    "success_mean": float(values[labels == 0].mean()) if np.any(labels == 0) else None,
                }
    best = max(correlations, key=lambda key: correlations[key]["spearman_rho"])
    significant = [key for key, value in correlations.items() if value["permutation_p"] < 0.05]
    best_significant = max(significant, key=lambda key: correlations[key]["spearman_rho"]) if significant else None
    result = {
        "blocks": [payload["target"] for payload in blocks],
        "num_blocks": len(blocks),
        "correlations": correlations,
        "best_transfer_metric": best,
        "best_significant_transfer_metric": best_significant,
        "failure_prediction_full_w4a8": failure_prediction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "num_blocks": len(blocks),
        "correlations": {key: {"rho": value["spearman_rho"], "p": value["permutation_p"]} for key, value in correlations.items()},
        "best_transfer_metric": best,
        "best_significant_transfer_metric": best_significant,
        "failure_prediction_full_w4a8": failure_prediction,
    }, indent=2))


if __name__ == "__main__":
    main()
