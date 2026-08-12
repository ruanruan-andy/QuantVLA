#!/usr/bin/env python
"""Collect sparse closed-loop LIBERO-Plus trajectories for ShiftGuard-v2 risk probes."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

os.environ["TANGENT_PROBE_STUB_WAND"] = "1"
from collect_tangent_probe_observations import install_optional_wand_stub

install_optional_wand_stub()

from libero.libero import benchmark, get_libero_path

from examples.Libero.eval.run_libero_eval import GR00TPolicy
from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env


MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 600, "libero_10": 1000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial", choices=tuple(MAX_STEPS))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task-ids", nargs="+", type=int, required=True)
    parser.add_argument("--split", choices=("calibration", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--tail-states", type=int, default=5)
    parser.add_argument("--max-states-per-task", type=int, default=24)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    return parser.parse_args()


def metadata_by_name(suite: str) -> dict[str, dict[str, Any]]:
    path = Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    return {str(row["name"]): row for row in json.loads(path.read_text(encoding="utf-8"))[suite]}


def normalized_observation(policy: GR00TPolicy, obs: dict, language: str) -> dict[str, np.ndarray]:
    processed = policy._process_observation(obs, language)
    return {key: np.asarray(value).copy() for key, value in processed.items()}


def main() -> None:
    args = parse_args()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    names = suite.get_task_names()
    metadata = metadata_by_name(args.suite)
    policy = GR00TPolicy(host="localhost", port=args.port, headless=True)
    all_records: list[dict[str, Any]] = []
    trajectory_summaries = []

    for task_id in args.task_ids:
        if not 0 <= task_id < len(names):
            raise ValueError(f"Invalid task id {task_id}; suite has {len(names)} tasks")
        task = suite.get_task(task_id)
        task_name = names[task_id]
        task_meta = metadata[task_name]
        initial_states = suite.get_task_init_states(task_id)
        if not len(initial_states):
            raise RuntimeError(f"Task {task_id} has no initial state")
        env = None
        sampled: dict[int, dict[str, Any]] = {}
        tail: deque[dict[str, Any]] = deque(maxlen=args.tail_states)
        success = False
        steps = 0
        policy_step = 0
        previous_gripper: float | None = None
        max_steps = MAX_STEPS[args.suite]
        try:
            env, fallback_language = get_libero_env(
                task, resolution=256, horizon=max(1000, max_steps + args.num_steps_wait)
            )
            language = str(getattr(env, "language_instruction", fallback_language))
            obs = env.reset()
            obs = env.set_init_state(initial_states[0])
            done = False
            while steps < max_steps + args.num_steps_wait:
                if steps < args.num_steps_wait:
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    steps += 1
                    continue
                observation = normalized_observation(policy, obs, language)
                action_chunk = policy.policy.get_action(observation)
                action = policy._convert_to_libero_action(action_chunk, 0)
                gripper = float(action[-1])
                gripper_switch = previous_gripper is not None and gripper != previous_gripper
                previous_gripper = gripper
                record = {
                    "key": f"{args.split}:{args.suite}:{task_id}:0:{policy_step}",
                    "split": args.split,
                    "domain": str(task_meta["category"]),
                    "suite": args.suite,
                    "task_id": int(task_id),
                    "trial_id": 0,
                    "trajectory_id": f"{args.suite}:{task_id}:0",
                    "task_name": task_name,
                    "step": int(steps),
                    "policy_step": int(policy_step),
                    "phase": float(policy_step / max(max_steps - 1, 1)),
                    "gripper_switch": bool(gripper_switch),
                    "action_norm": float(np.linalg.norm(action[:6])),
                    "tail_state": False,
                    "policy_action": action.copy(),
                    "observation": observation,
                }
                tail.append(record)
                if policy_step % args.stride == 0 or gripper_switch:
                    sampled[policy_step] = record
                obs, _, done, _ = env.step(action.tolist())
                steps += 1
                policy_step += 1
                if done:
                    success = True
                    break
        finally:
            if env is not None:
                env.close()

        for record in tail:
            record["tail_state"] = True
            sampled[record["policy_step"]] = record
        ordered = [sampled[key] for key in sorted(sampled)]
        if len(ordered) > args.max_states_per_task:
            critical = [row for row in ordered if row["gripper_switch"] or row["tail_state"]]
            critical_steps = {row["policy_step"] for row in critical}
            ordinary = [row for row in ordered if row["policy_step"] not in critical_steps]
            if len(critical) > args.max_states_per_task:
                indices = np.linspace(0, len(critical) - 1, args.max_states_per_task).round().astype(int)
                critical = [critical[index] for index in sorted(set(indices.tolist()))]
            remaining = max(args.max_states_per_task - len(critical), 0)
            if remaining and ordinary:
                indices = np.linspace(0, len(ordinary) - 1, remaining).round().astype(int)
                ordinary = [ordinary[index] for index in sorted(set(indices.tolist()))]
            ordered = sorted((critical + ordinary)[: args.max_states_per_task], key=lambda row: row["policy_step"])
        for record in ordered:
            record["trajectory_success"] = bool(success)
            record["trajectory_steps"] = int(steps)
            record["phase"] = float(record["policy_step"] / max(policy_step - 1, 1))
        all_records.extend(ordered)
        summary = {
            "split": args.split,
            "task_id": task_id,
            "domain": str(task_meta["category"]),
            "success": success,
            "steps": steps,
            "policy_steps": policy_step,
            "saved_states": len(ordered),
        }
        trajectory_summaries.append(summary)
        print(json.dumps(summary), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(all_records, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps({
        "suite": args.suite,
        "split": args.split,
        "task_ids": args.task_ids,
        "states": len(all_records),
        "trajectories": trajectory_summaries,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "states": len(all_records)}, indent=2))


if __name__ == "__main__":
    main()
