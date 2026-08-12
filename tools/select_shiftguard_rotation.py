#!/usr/bin/env python
"""Select a ShiftGuard rotation candidate using the robust excess action-risk objective."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--lambda-shift", type=float, default=1.0)
    parser.add_argument("--baseline", default="baseline")
    return parser.parse_args()


def expand(values: list[str]) -> list[Path]:
    result = []
    for value in values:
        result.extend(Path(path) for path in sorted(glob.glob(value)))
    return result


def main() -> None:
    args = parse_args()
    rows = []
    for path in expand(args.inputs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_domain = payload["by_domain"]
        id_loss = float(by_domain["ID"]["action_discrepancy"])
        ood = np.asarray(
            [
                value["action_discrepancy"]
                for domain, value in by_domain.items()
                if domain != "ID"
            ],
            dtype=float,
        )
        excess = ood - id_loss
        smooth_worst = float(
            args.temperature
            * (logsumexp(excess / args.temperature) - np.log(len(excess)))
        )
        objective = id_loss + args.lambda_shift * smooth_worst
        rows.append(
            {
                "candidate": path.stem,
                "path": str(path),
                "objective": objective,
                "id_action_discrepancy": id_loss,
                "ood_mean": float(np.mean(ood)),
                "ood_std": float(np.std(ood)),
                "ood_max": float(np.max(ood)),
                "smooth_worst_excess": smooth_worst,
            }
        )
    rows.sort(key=lambda row: row["objective"])
    baseline = next(row for row in rows if row["candidate"] == args.baseline)
    for row in rows:
        row["objective_change_vs_baseline"] = row["objective"] - baseline["objective"]
        row["ood_mean_change_vs_baseline"] = row["ood_mean"] - baseline["ood_mean"]
    result = {
        "temperature": args.temperature,
        "lambda_shift": args.lambda_shift,
        "selected": rows[0],
        "baseline": baseline,
        "ranking": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
