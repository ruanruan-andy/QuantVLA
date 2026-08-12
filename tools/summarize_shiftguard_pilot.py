#!/usr/bin/env python
"""Summarize a fixed LIBERO-Plus subset and paired model differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")
CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--per-suite-category", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    effective = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            effective[int(row["task_index"])] = row
    return list(effective.values())


def fixed_subset(
    repo_root: Path, model: str, per_group: int, suites: list[str]
) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for suite in suites:
        path = repo_root / "output" / "libero-plus" / model / suite / "episodes.jsonl"
        records = load_records(path)
        for category in CATEGORIES:
            candidates = sorted(
                (row for row in records if row.get("category") == category),
                key=lambda row: int(row["task_index"]),
            )
            if len(candidates) < per_group:
                raise RuntimeError(
                    f"{model}/{suite}/{category} has {len(candidates)} records; need {per_group}"
                )
            for row in candidates[:per_group]:
                result[(suite, int(row["task_index"]))] = row
    return result


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in records if row.get("error") is None]
    successes = sum(bool(row.get("success")) for row in valid)
    return {
        "episodes": len(records),
        "valid_episodes": len(valid),
        "errors": len(records) - len(valid),
        "successes": successes,
        "success_rate": successes / len(valid) if valid else 0.0,
    }


def model_summary(
    records: dict[tuple[str, int], dict[str, Any]], suites: list[str]
) -> dict[str, Any]:
    rows = list(records.values())
    return {
        "overall": aggregate(rows),
        "by_category": {
            category: aggregate([row for row in rows if row.get("category") == category])
            for category in CATEGORIES
        },
        "by_suite": {
            suite: aggregate([row for (row_suite, _), row in records.items() if row_suite == suite])
            for suite in suites
        },
    }


def paired_statistics(
    reference: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    rng: np.random.Generator,
    bootstrap: int,
    category: str | None = None,
) -> dict[str, Any]:
    keys = sorted(set(reference) & set(candidate))
    if category is not None:
        keys = [key for key in keys if reference[key].get("category") == category]
    keys = [
        key
        for key in keys
        if reference[key].get("error") is None and candidate[key].get("error") is None
    ]
    ref = np.asarray([bool(reference[key]["success"]) for key in keys], dtype=float)
    cand = np.asarray([bool(candidate[key]["success"]) for key in keys], dtype=float)
    differences = cand - ref
    gains = int(np.sum((ref == 0) & (cand == 1)))
    losses = int(np.sum((ref == 1) & (cand == 0)))
    discordant = gains + losses
    mcnemar_p = (
        float(stats.binomtest(gains, discordant, 0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    samples = rng.integers(0, len(keys), size=(bootstrap, len(keys)))
    bootstrap_diff = differences[samples].mean(axis=1)
    return {
        "paired_episodes": len(keys),
        "reference_success_rate": float(ref.mean()),
        "candidate_success_rate": float(cand.mean()),
        "difference": float(differences.mean()),
        "bootstrap_ci95": [
            float(np.percentile(bootstrap_diff, 2.5)),
            float(np.percentile(bootstrap_diff, 97.5)),
        ],
        "candidate_only_successes": gains,
        "reference_only_successes": losses,
        "mcnemar_exact_p": mcnemar_p,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    records = {
        model: fixed_subset(args.repo_root, model, args.per_suite_category, args.suites)
        for model in args.models
    }
    result: dict[str, Any] = {
        "protocol": {
            "suites": list(args.suites),
            "categories": list(CATEGORIES),
            "per_suite_category": args.per_suite_category,
            "episodes_per_model": len(args.suites) * len(CATEGORIES) * args.per_suite_category,
        },
        "models": {model: model_summary(rows, args.suites) for model, rows in records.items()},
        "comparisons": {},
    }
    reference = args.models[0]
    for candidate in args.models[1:]:
        result["comparisons"][f"{candidate}_vs_{reference}"] = {
            "overall": paired_statistics(
                records[reference], records[candidate], rng, args.bootstrap
            ),
            "by_category": {
                category: paired_statistics(
                    records[reference], records[candidate], rng, args.bootstrap, category
                )
                for category in CATEGORIES
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "protocol": result["protocol"],
        "overall": {model: value["overall"] for model, value in result["models"].items()},
        "comparisons": {
            name: value["overall"] for name, value in result["comparisons"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
