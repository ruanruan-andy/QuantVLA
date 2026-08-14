"""Create deterministic disjoint Train-560 and Test-560 manifests."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from libero.libero import benchmark, get_libero_path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


@dataclass
class SplitConfig:
    output_dir: str = "configs/libero_plus/splits"
    split_seed: int = 2026
    train_per_suite_category: int = 20
    gap_per_suite_category: int = 10
    test_per_suite_category: int = 20
    overwrite: bool = False


def _stable_seed(seed: int, suite: str, category: str, difficulty: str) -> int:
    payload = f"{seed}|{suite}|{category}|{difficulty}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _difficulty_interleave(
    candidates: list[int],
    *,
    metadata_by_id: dict[int, dict[str, Any]],
    seed: int,
    suite: str,
    category: str,
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for task_id in candidates:
        difficulty = str(metadata_by_id[task_id].get("difficulty_level", "unspecified"))
        groups[difficulty].append(task_id)
    for difficulty, values in groups.items():
        random.Random(_stable_seed(seed, suite, category, difficulty)).shuffle(values)

    ordered: list[int] = []
    keys = sorted(groups)
    while any(groups[key] for key in keys):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
    return ordered


def _manifest(
    *,
    split: str,
    seed: int,
    per_group: int,
    task_ids_by_suite: dict[str, list[int]],
    task_metadata_by_suite: dict[str, list[dict[str, Any]]],
    gap_per_group: int,
) -> dict[str, Any]:
    total = per_group * len(SUITES) * len(CATEGORIES)
    return {
        "version": 3,
        "protocol": "libero-plus-disjoint-opqd-v2",
        "split": split,
        "selection": "explicit_task_ids",
        "split_seed": seed,
        "per_suite_per_category": per_group,
        "excluded_gap_per_suite_category": gap_per_group,
        "total_tasks": total,
        "initial_state_ids": [0],
        "categories": list(CATEGORIES),
        "suites": list(SUITES),
        "task_ids_by_suite": task_ids_by_suite,
        "task_metadata_by_suite": task_metadata_by_suite,
    }


def build(config: SplitConfig) -> None:
    if config.train_per_suite_category <= 0 or config.test_per_suite_category <= 0:
        raise ValueError("train and test quotas must be positive")
    if config.gap_per_suite_category < 0:
        raise ValueError("gap quota must be non-negative")
    output_dir = Path(config.output_dir).expanduser().resolve()
    train_total = config.train_per_suite_category * len(SUITES) * len(CATEGORIES)
    test_total = config.test_per_suite_category * len(SUITES) * len(CATEGORIES)
    train_path = output_dir / f"train{train_total}-split{config.split_seed}.json"
    test_path = output_dir / f"test{test_total}-split{config.split_seed}.json"
    if not config.overwrite and (train_path.exists() or test_path.exists()):
        raise FileExistsError("split output exists; pass --overwrite to replace it")

    classification_path = (
        Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    )
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    benchmark_dict = benchmark.get_benchmark_dict()
    train_by_suite: dict[str, list[int]] = {}
    test_by_suite: dict[str, list[int]] = {}
    train_metadata_by_suite: dict[str, list[dict[str, Any]]] = {}
    test_metadata_by_suite: dict[str, list[dict[str, Any]]] = {}
    overlap_report: dict[str, int] = {}

    for suite in SUITES:
        task_names = benchmark_dict[suite]().get_task_names()
        metadata_by_name = {str(item["name"]): item for item in classification[suite]}
        missing = [name for name in task_names if name not in metadata_by_name]
        if missing:
            raise RuntimeError(f"classification missing {suite} task {missing[0]!r}")
        metadata_by_id = {
            task_id: metadata_by_name[name] for task_id, name in enumerate(task_names)
        }
        train_ids: list[int] = []
        test_ids: list[int] = []
        required = (
            config.train_per_suite_category
            + config.gap_per_suite_category
            + config.test_per_suite_category
        )
        for category in CATEGORIES:
            candidates = [
                task_id
                for task_id, metadata in metadata_by_id.items()
                if str(metadata["category"]) == category
            ]
            if len(candidates) < required:
                raise ValueError(
                    f"{suite}/{category} has {len(candidates)} tasks; requires {required}"
                )
            ordered = _difficulty_interleave(
                candidates,
                metadata_by_id=metadata_by_id,
                seed=config.split_seed,
                suite=suite,
                category=category,
            )
            train_ids.extend(ordered[: config.train_per_suite_category])
            test_start = config.train_per_suite_category + config.gap_per_suite_category
            test_ids.extend(ordered[test_start:required])
        overlap = set(train_ids) & set(test_ids)
        if overlap:
            raise AssertionError(f"train/test overlap in {suite}: {sorted(overlap)[:5]}")
        train_by_suite[suite] = train_ids
        test_by_suite[suite] = test_ids
        def serialize(task_id: int) -> dict[str, Any]:
            metadata = metadata_by_id[task_id]
            return {
                "task_id": task_id,
                "task_name": task_names[task_id],
                "classification_id": int(metadata["id"]),
                "category": str(metadata["category"]),
                "difficulty_level": metadata.get("difficulty_level"),
            }

        train_metadata_by_suite[suite] = [serialize(task_id) for task_id in train_ids]
        test_metadata_by_suite[suite] = [serialize(task_id) for task_id in test_ids]
        overlap_report[suite] = len(overlap)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path.write_text(
        json.dumps(
            _manifest(
                split="train",
                seed=config.split_seed,
                per_group=config.train_per_suite_category,
                task_ids_by_suite=train_by_suite,
                task_metadata_by_suite=train_metadata_by_suite,
                gap_per_group=config.gap_per_suite_category,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    test_path.write_text(
        json.dumps(
            _manifest(
                split="test",
                seed=config.split_seed,
                per_group=config.test_per_suite_category,
                task_ids_by_suite=test_by_suite,
                task_metadata_by_suite=test_metadata_by_suite,
                gap_per_group=config.gap_per_suite_category,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train_manifest": str(train_path),
                "test_manifest": str(test_path),
                "train_tasks": sum(map(len, train_by_suite.values())),
                "test_tasks": sum(map(len, test_by_suite.values())),
                "excluded_gap_per_suite_category": config.gap_per_suite_category,
                "overlap_by_suite": overlap_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    build(tyro.cli(SplitConfig))
