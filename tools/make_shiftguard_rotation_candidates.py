#!/usr/bin/env python
"""Create orthogonal, fixed-W4A8 ShiftGuard rotation candidates around a QuantVLA pack."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-pack", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dit-blocks", nargs="+", type=int, required=True)
    parser.add_argument("--angles", nargs="+", type=float, default=[-0.1, -0.05, 0.05, 0.1])
    parser.add_argument("--pairing-seed", type=int, default=20260811)
    return parser.parse_args()


def givens_bank(dimension: int, angle: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + dimension)
    order = rng.permutation(dimension)
    matrix = np.eye(dimension, dtype=np.float64)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    for offset in range(0, dimension - 1, 2):
        left = int(order[offset])
        right = int(order[offset + 1])
        matrix[left, left] = cosine
        matrix[right, right] = cosine
        matrix[left, right] = -sine
        matrix[right, left] = sine
    return matrix


def candidate_specs(angles: list[float]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [{"name": "baseline", "scope": "none", "angle": 0.0}]
    for scope in ("input", "output", "both"):
        for angle in angles:
            label = str(angle).replace("-", "m").replace(".", "p")
            specs.append({"name": f"{scope}_{label}", "scope": scope, "angle": angle})
    return specs


def initialize_overlay(baseline: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for source in baseline.glob("*.npz"):
        os.symlink(source.resolve(), destination / source.name)


def transform_pack(
    source: Path,
    destination: Path,
    scope: str,
    angle: float,
    seed: int,
) -> float:
    with np.load(source, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    errors = []
    if scope in ("input", "both"):
        for name in [key for key in arrays if key.startswith("Rin_")]:
            rotation = arrays[name].astype(np.float64)
            update = givens_bank(rotation.shape[0], angle, seed)
            rotation = rotation @ update
            errors.append(float(np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[0]))))
            arrays[name] = rotation.astype(np.float32)
    if scope in ("output", "both"):
        for name in [key for key in arrays if key.startswith("Rout_")]:
            rotation = arrays[name].astype(np.float64)
            update = givens_bank(rotation.shape[0], angle, seed + 1009)
            rotation = update @ rotation
            errors.append(float(np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[0]))))
            arrays[name] = rotation.astype(np.float32)
    destination.unlink()
    temporary = destination.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(destination)
    return max(errors, default=0.0)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    prefixes = [f"action_head.model.transformer_blocks.{index}." for index in args.dit_blocks]
    selected_files = sorted(
        path
        for path in args.baseline_pack.glob("*.npz")
        if any(path.name.startswith(prefix) for prefix in prefixes)
    )
    if len(selected_files) != 2 * len(args.dit_blocks):
        raise RuntimeError(
            f"Expected {2 * len(args.dit_blocks)} selected DiT packs, found {len(selected_files)}"
        )

    manifest = {
        "baseline_pack": str(args.baseline_pack.resolve()),
        "dit_blocks": args.dit_blocks,
        "pairing_seed": args.pairing_seed,
        "selected_files": [path.name for path in selected_files],
        "candidates": [],
    }
    for spec in candidate_specs(args.angles):
        destination = args.output_root / str(spec["name"])
        initialize_overlay(args.baseline_pack, destination)
        max_error = 0.0
        if spec["scope"] != "none":
            for offset, source in enumerate(selected_files):
                max_error = max(
                    max_error,
                    transform_pack(
                        source,
                        destination / source.name,
                        str(spec["scope"]),
                        float(spec["angle"]),
                        args.pairing_seed + offset * 17,
                    ),
                )
        manifest["candidates"].append(
            {
                **spec,
                "pack_dir": str(destination),
                "max_orthogonality_error": max_error,
            }
        )
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
