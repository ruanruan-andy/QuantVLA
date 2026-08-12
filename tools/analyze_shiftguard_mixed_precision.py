#!/usr/bin/env python
"""Analyze trajectory-aware mixed-precision candidates against W4A8."""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


QUANTIZED_WEIGHT_PARAMS = 905_969_664
DIT_BLOCK_WEIGHT_PARAMS = 2 * 1536 * 6144


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--max-max-risk-regression", type=float, default=0.02)
    parser.add_argument("--min-improved-domains", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260811)
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
        values = np.asarray(
            [float(row["normalized_action_discrepancy"]) for row in rows], dtype=float
        )
        result.append({
            "trajectory_id": trajectory_id,
            "split": rows[0]["split"],
            "domain": rows[0]["domain"],
            "mean": float(values.mean()),
            "max": float(values.max()),
        })
    return result


def summarize(rows: list[dict], split: str, kappa: float) -> dict:
    selected = [row for row in rows if row["split"] == split]
    means = np.asarray([row["mean"] for row in selected], dtype=float)
    maxima = np.asarray([row["max"] for row in selected], dtype=float)
    return {
        "trajectories": len(selected),
        "mean": float(means.mean()),
        "std": float(means.std()),
        "robust_mean": float(means.mean() + kappa * means.std()),
        "max_risk_mean": float(maxima.mean()),
        "by_domain": {row["domain"]: row["mean"] for row in selected},
        "by_trajectory": {row["trajectory_id"]: row["mean"] for row in selected},
    }


def paired_statistics(candidate: dict, baseline: dict, bootstrap_samples: int, seed: int) -> dict:
    keys = sorted(set(candidate) & set(baseline))
    differences = np.asarray([candidate[key] - baseline[key] for key in keys], dtype=float)
    observed = float(differences.mean())
    permutation = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permutation.append(float(np.mean(differences * np.asarray(signs))))
    one_sided_p = float(np.mean(np.asarray(permutation) <= observed))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return {
        "pairs": len(keys),
        "mean_difference": observed,
        "relative_difference": observed / float(np.mean([baseline[key] for key in keys])),
        "improved_trajectories": int(np.sum(differences < 0)),
        "exact_one_sided_signflip_p": one_sided_p,
        "bootstrap_95ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
    }


def budget(payload: dict) -> dict:
    blocks = payload.get("high_precision_blocks", [])
    bits = int(payload.get("high_precision_weight_bits", 4))
    extra_bits = len(blocks) * DIT_BLOCK_WEIGHT_PARAMS * max(bits - 4, 0)
    baseline_bits = QUANTIZED_WEIGHT_PARAMS * 4
    return {
        "high_precision_blocks": blocks,
        "high_precision_weight_bits": bits,
        "theoretical_average_weight_bits": (baseline_bits + extra_bits) / QUANTIZED_WEIGHT_PARAMS,
        "theoretical_weight_storage_mib": (baseline_bits + extra_bits) / 8 / (2**20),
        "extra_weight_storage_mib": extra_bits / 8 / (2**20),
    }


def main() -> None:
    args = parse_args()
    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_rows = trajectory_rows(baseline_payload)
    baseline = {
        split: summarize(baseline_rows, split, args.kappa)
        for split in ("calibration", "validation")
    }
    results = []
    for index, path in enumerate(expand(args.candidates)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = trajectory_rows(payload)
        split_summaries = {
            split: summarize(rows, split, args.kappa)
            for split in ("calibration", "validation")
        }
        validation = split_summaries["validation"]
        base_validation = baseline["validation"]
        relative_by_domain = {
            domain: (value - base_validation["by_domain"][domain]) / base_validation["by_domain"][domain]
            for domain, value in validation["by_domain"].items()
        }
        improved_domains = sum(value < 0 for value in relative_by_domain.values())
        max_safe = validation["max_risk_mean"] <= base_validation["max_risk_mean"] * (
            1.0 + args.max_max_risk_regression
        )
        eligible = (
            validation["robust_mean"] < base_validation["robust_mean"]
            and validation["mean"] < base_validation["mean"]
            and max_safe
            and improved_domains >= args.min_improved_domains
        )
        results.append({
            "candidate": path.stem,
            "path": str(path),
            "budget": budget(payload),
            **split_summaries,
            "validation_relative_mean": (validation["mean"] - base_validation["mean"]) / base_validation["mean"],
            "validation_relative_robust": (validation["robust_mean"] - base_validation["robust_mean"]) / base_validation["robust_mean"],
            "validation_relative_max": (validation["max_risk_mean"] - base_validation["max_risk_mean"]) / base_validation["max_risk_mean"],
            "relative_by_domain": relative_by_domain,
            "improved_domains": improved_domains,
            "max_safe": max_safe,
            "eligible": eligible,
            "paired_validation": paired_statistics(
                validation["by_trajectory"],
                base_validation["by_trajectory"],
                args.bootstrap_samples,
                args.seed + index,
            ),
        })
    results.sort(key=lambda row: (row["budget"]["extra_weight_storage_mib"], row["validation"]["robust_mean"]))
    eligible = [row for row in results if row["eligible"]]
    selected = min(
        eligible,
        key=lambda row: (row["validation"]["robust_mean"], row["budget"]["extra_weight_storage_mib"]),
    ) if eligible else {"candidate": "baseline", "reason": "No candidate passed the held-out gate"}
    output = {
        "parameters": vars(args) | {
            "baseline": str(args.baseline),
            "candidates": args.candidates,
            "output": str(args.output),
        },
        "constants": {
            "quantized_weight_params": QUANTIZED_WEIGHT_PARAMS,
            "dit_block_weight_params": DIT_BLOCK_WEIGHT_PARAMS,
        },
        "baseline": baseline,
        "candidates": results,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidates": [
            {
                "candidate": row["candidate"],
                "extra_mib": row["budget"]["extra_weight_storage_mib"],
                "validation_delta_pct": 100 * row["validation_relative_mean"],
                "validation_robust_delta_pct": 100 * row["validation_relative_robust"],
                "improved_domains": row["improved_domains"],
                "p": row["paired_validation"]["exact_one_sided_signflip_p"],
                "eligible": row["eligible"],
            }
            for row in results
        ],
        "selected": selected.get("candidate"),
    }, indent=2))


if __name__ == "__main__":
    main()
