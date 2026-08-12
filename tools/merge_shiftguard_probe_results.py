#!/usr/bin/env python
"""Merge ID and OOD branch-discrepancy probe outputs by candidate name."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-dir", type=Path, required=True)
    parser.add_argument("--ood-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summary(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "action_discrepancy": float(np.mean([row["action_discrepancy"] for row in rows])),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for id_path in sorted(args.id_dir.glob("*.json")):
        ood_path = args.ood_dir / id_path.name
        if not ood_path.exists():
            raise FileNotFoundError(ood_path)
        id_payload = json.loads(id_path.read_text(encoding="utf-8"))
        ood_payload = json.loads(ood_path.read_text(encoding="utf-8"))
        rows = id_payload["samples"] + ood_payload["samples"]
        if len({row["key"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate sample key while merging {id_path.stem}")
        by_domain = defaultdict(list)
        for row in rows:
            by_domain[row["domain"]].append(row)
        payload = {
            "branch": ood_payload["branch"],
            "overall": summary(rows),
            "by_domain": {key: summary(value) for key, value in sorted(by_domain.items())},
            "samples": rows,
        }
        (args.output_dir / id_path.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"merged": len(list(args.output_dir.glob('*.json')))}, indent=2))


if __name__ == "__main__":
    main()
