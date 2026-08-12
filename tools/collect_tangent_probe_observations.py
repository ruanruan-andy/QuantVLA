#!/usr/bin/env python
"""Collect a small, non-evaluation LIBERO observation buffer for tangent probing."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


def install_optional_wand_stub() -> None:
    """Allow non-motion-blur LIBERO-Plus tasks when ImageMagick is unavailable."""
    if os.environ.get("TANGENT_PROBE_STUB_WAND", "0") != "1":
        return

    class DummyFunction:
        argtypes = None

        def __call__(self, wand, radius, _sigma, angle):
            import cv2

            kernel_size = max(3, int(round(float(radius))) * 2 + 1)
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            center = ((kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0)
            rotation = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
            kernel /= max(float(kernel.sum()), 1e-12)
            wand._array = cv2.filter2D(wand._array, -1, kernel)

    class DummyLibrary:
        MagickMotionBlurImage = DummyFunction()

    class DummyWandImage:
        def __init__(self, *_args, blob=None, **_kwargs):
            import cv2

            if blob is None:
                raise ValueError("The motion-blur compatibility image requires a PNG blob")
            encoded = np.frombuffer(blob, dtype=np.uint8)
            self._array = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if self._array is None:
                raise ValueError("Could not decode the motion-blur input image")
            self.wand = self

        def make_blob(self):
            import cv2

            ok, encoded = cv2.imencode(".png", self._array)
            if not ok:
                raise RuntimeError("Could not encode the motion-blur output image")
            return encoded.tobytes()

    wand_module = types.ModuleType("wand")
    api_module = types.ModuleType("wand.api")
    image_module = types.ModuleType("wand.image")
    api_module.library = DummyLibrary()
    image_module.Image = DummyWandImage
    wand_module.api = api_module
    wand_module.image = image_module
    sys.modules.update(
        {"wand": wand_module, "wand.api": api_module, "wand.image": image_module}
    )


install_optional_wand_stub()

from libero.libero import benchmark, get_libero_path

from examples.Libero.eval.utils import get_libero_env, get_libero_image, quat2axisangle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("standard", "plus"), required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--category", default=None)
    parser.add_argument("--num-observations", type=int, default=16)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--trial-offset", type=int, default=0)
    parser.add_argument(
        "--skip-first",
        type=int,
        default=25,
        help="For LIBERO-Plus, skip the fixed evaluation tasks in this category.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def convert_observation(obs: dict[str, Any], language: str) -> dict[str, np.ndarray]:
    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"].copy())
    gripper = obs["robot0_gripper_qpos"].copy()
    image, wrist_image = get_libero_image(obs)
    return {
        "video.image": np.expand_dims(image, axis=0).astype(np.uint8),
        "video.wrist_image": np.expand_dims(wrist_image, axis=0).astype(np.uint8),
        "state.x": np.array([[xyz[0]]], dtype=np.float32),
        "state.y": np.array([[xyz[1]]], dtype=np.float32),
        "state.z": np.array([[xyz[2]]], dtype=np.float32),
        "state.roll": np.array([[rpy[0]]], dtype=np.float32),
        "state.pitch": np.array([[rpy[1]]], dtype=np.float32),
        "state.yaw": np.array([[rpy[2]]], dtype=np.float32),
        "state.gripper": np.expand_dims(gripper.astype(np.float32), axis=0),
        "annotation.human.action.task_description": np.asarray([language]),
    }


def category_matches(value: str, requested: str) -> bool:
    value_norm = value.casefold().replace("_", " ")
    requested_norm = requested.casefold().replace("_", " ")
    aliases = {
        "camera": "camera viewpoints",
        "robot": "robot initial states",
        "language": "language instructions",
        "light": "light conditions",
        "background": "background textures",
        "noise": "sensor noise",
        "layout": "objects layout",
    }
    target = aliases.get(requested_norm, requested_norm)
    return value_norm == target or value_norm.startswith(requested_norm)


def select_plus_task_ids(task_suite, suite_name: str, category: str, skip_first: int) -> list[int]:
    classification_path = (
        Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    )
    with classification_path.open(encoding="utf-8") as stream:
        classifications = json.load(stream)[suite_name]
    metadata_by_name = {str(item["name"]): item for item in classifications}
    candidates = []
    for task_id, task_name in enumerate(task_suite.get_task_names()):
        metadata = metadata_by_name[task_name]
        if category_matches(str(metadata["category"]), category):
            candidates.append(task_id)
    if len(candidates) <= skip_first:
        raise RuntimeError(
            f"Only {len(candidates)} tasks matched category {category!r}; cannot skip {skip_first}."
        )
    return candidates[skip_first:]


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.source == "plus":
        if not args.category:
            raise ValueError("--category is required for --source plus")
        task_ids = select_plus_task_ids(task_suite, args.suite, args.category, args.skip_first)
        domain = args.category
    else:
        task_ids = list(range(task_suite.n_tasks))
        domain = "ID"

    if args.task_offset:
        offset = args.task_offset % len(task_ids)
        task_ids = task_ids[offset:] + task_ids[:offset]

    records: list[dict[str, Any]] = []
    task_cursor = 0
    trial_cursor = args.trial_offset
    while len(records) < args.num_observations:
        if task_cursor >= len(task_ids):
            task_cursor = 0
            trial_cursor += 1
        task_id = task_ids[task_cursor]
        task_cursor += 1
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if not len(initial_states):
            continue
        trial_id = trial_cursor % len(initial_states)
        env = None
        try:
            env, fallback_language = get_libero_env(task, resolution=256)
            env.reset()
            obs = env.set_init_state(initial_states[trial_id])
            language = str(getattr(env, "language_instruction", fallback_language))
            records.append(
                {
                    "domain": domain,
                    "source": args.source,
                    "suite": args.suite,
                    "task_id": int(task_id),
                    "trial_id": int(trial_id),
                    "task_name": str(task.name),
                    "observation": convert_observation(obs, language),
                }
            )
        finally:
            if env is not None:
                env.close()
    return records


def main() -> None:
    args = parse_args()
    records = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source": args.source,
                "suite": args.suite,
                "category": args.category,
                "records": len(records),
                "task_ids": [record["task_id"] for record in records],
                "trial_ids": [record["trial_id"] for record in records],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
