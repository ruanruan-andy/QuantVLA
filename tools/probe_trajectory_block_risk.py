#!/usr/bin/env python
"""Measure single-block W4A8 risk on sparse closed-loop trajectory states."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from probe_tangent_risk import (
    block_paths,
    differentiable_action,
    prepare_normalized_input,
    set_quant_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--packdir", required=True)
    parser.add_argument("--target", required=True, help="dit:N or all")
    parser.add_argument(
        "--high-precision-blocks",
        nargs="*",
        type=int,
        default=[],
        help="DiT block indices whose weights use --high-precision-weight-bits when --target=all",
    )
    parser.add_argument("--high-precision-weight-bits", type=int, default=8)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atm-alpha", type=Path)
    parser.add_argument("--cvar-fraction", type=float, default=0.2)
    parser.add_argument("--growth-weight", type=float, default=0.5)
    parser.add_argument("--critical-weight", type=float, default=0.5)
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open("rb") as stream:
            records.extend(pickle.load(stream))
    return records


def libero_executed_action(policy, normalized_action: torch.Tensor) -> torch.Tensor:
    """Return the 7 physical controls that the LIBERO wrapper actually executes."""
    unnormalized = policy._get_unnormalized_action(normalized_action.float())
    keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    components = []
    for key in keys:
        value = np.asarray(unnormalized[f"action.{key}"])
        if value.ndim >= 3 and value.shape[-1] == 1:
            value = value[..., 0]
        components.append(value)
    action = np.stack(components, axis=-1).astype(np.float32, copy=False)
    # This exactly matches examples.Libero.eval.utils.normalize_gripper_action.
    action[..., -1] = np.sign(1.0 - 2.0 * action[..., -1])
    return torch.from_numpy(action)


def configure(
    model,
    target: str,
    active: bool,
    high_precision_blocks: set[int] | None = None,
    high_precision_weight_bits: int = 8,
) -> int:
    from gr00t.quantization.duquant_layers import DuQuantLinear

    prefix = None if target == "all" else block_paths(target)[1]
    count = 0
    high_precision_blocks = high_precision_blocks or set()
    for name, module in model.named_modules():
        if not isinstance(module, DuQuantLinear):
            continue
        if not getattr(module, "_trajectory_probe_private_cfg", False):
            module.cfg = copy.copy(module.cfg)
            module._trajectory_probe_private_cfg = True
        selected = target == "all" or name.startswith(str(prefix))
        enabled = active and selected
        weight_bits = 4
        if target == "all" and high_precision_blocks:
            for block_index in high_precision_blocks:
                block_prefix = f"action_head.model.transformer_blocks.{block_index}."
                if name.startswith(block_prefix):
                    weight_bits = high_precision_weight_bits
                    break
        module.weight_bits = weight_bits if enabled else 0
        module.cfg.act_bits = 8 if enabled else 0
        module._cached_weight_key = None
        module._weight_quantized_cached = False
        count += int(enabled)
    return count


def cvar(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.mean(np.sort(np.asarray(values, dtype=float))[-count:]))


def summarize_trajectories(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trajectory_id"]].append(row)
    trajectories = []
    for trajectory_id, values in sorted(grouped.items()):
        values.sort(key=lambda row: row["policy_step"])
        risks = [row["action_discrepancy"] for row in values]
        growth = [max(risks[index] - risks[index - 1], 0.0) for index in range(1, len(risks))]
        critical = [
            row["action_discrepancy"]
            for row in values
            if row["gripper_switch"] or row["tail_state"]
        ]
        tail_risk = cvar(risks, args.cvar_fraction)
        growth_risk = cvar(growth, args.cvar_fraction)
        critical_risk = cvar(critical, args.cvar_fraction)
        trajectory_risk = tail_risk + args.growth_weight * growth_risk + args.critical_weight * critical_risk
        trajectories.append({
            "trajectory_id": trajectory_id,
            "split": values[0]["split"],
            "domain": values[0]["domain"],
            "task_id": values[0]["task_id"],
            "success": values[0]["trajectory_success"],
            "states": len(values),
            "mean_risk": float(np.mean(risks)),
            "max_risk": float(np.max(risks)),
            "initial_risk": float(risks[0]),
            "tail_cvar": tail_risk,
            "growth_cvar": growth_risk,
            "critical_cvar": critical_risk,
            "trajectory_risk": trajectory_risk,
        })
    by_split = {}
    for split in sorted({row["split"] for row in trajectories}):
        values = [row for row in trajectories if row["split"] == split]
        by_split[split] = {
            "trajectories": len(values),
            "successes": sum(bool(row["success"]) for row in values),
            "mean_risk": float(np.mean([row["mean_risk"] for row in values])),
            "initial_risk": float(np.mean([row["initial_risk"] for row in values])),
            "tail_cvar": float(np.mean([row["tail_cvar"] for row in values])),
            "trajectory_risk": float(np.mean([row["trajectory_risk"] for row in values])),
        }
    return {"trajectories": trajectories, "by_split": by_split}


def main() -> None:
    args = parse_args()
    set_quant_environment(args.packdir)
    if args.atm_alpha:
        os.environ.update({
            "GR00T_ATM_ENABLE": "1",
            "GR00T_ATM_ALPHA_PATH": str(args.atm_alpha),
            "GR00T_ATM_SCOPE": "dit",
            "GR00T_OHB_ENABLE": "1",
            "GR00T_OHB_SCOPE": "dit",
            "GR00T_OHB_FALLBACK": "1.0",
        })
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    data_config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=args.denoising_steps,
        device="cuda",
    )
    model = policy.model.eval()
    records = load_records(args.inputs)
    high_precision_blocks = set(args.high_precision_blocks)
    rows = []
    for index, record in enumerate(records):
        normalized = prepare_normalized_input(policy, record["observation"])
        head = model.action_head
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed + index)
        noise = torch.randn(
            (1, head.config.action_horizon, head.config.action_dim),
            generator=generator, device="cuda", dtype=head.dtype,
        )
        active_linears = configure(
            model,
            args.target,
            True,
            high_precision_blocks=high_precision_blocks,
            high_precision_weight_bits=args.high_precision_weight_bits,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action_quant = differentiable_action(model, normalized, noise)
        configure(
            model,
            args.target,
            False,
            high_precision_blocks=high_precision_blocks,
            high_precision_weight_bits=args.high_precision_weight_bits,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action_fp = differentiable_action(model, normalized, noise)
        # GR00T predicts 32 padded dimensions, while LIBERO executes only seven.
        # The primary metric must be computed after inverse normalization and
        # gripper binarization; full-32 MSE is retained only as a diagnostic.
        action_quant_f = action_quant.float()
        action_fp_f = action_fp.float()
        action_quant_chunk = libero_executed_action(policy, action_quant_f)
        action_fp_chunk = libero_executed_action(policy, action_fp_f)
        chunk_delta = action_quant_chunk - action_fp_chunk
        # examples/Libero/eval/run_libero_eval.py always calls
        # _convert_to_libero_action(action_chunk, 0): only the first action is executed.
        action_quant_exec = action_quant_chunk[:, 0, :]
        action_fp_exec = action_fp_chunk[:, 0, :]
        executed_delta = action_quant_exec - action_fp_exec
        discrepancy = float(torch.mean(executed_delta ** 2))
        fp_energy = float(torch.mean(action_fp_exec ** 2))
        model_delta_7 = action_quant_f[..., :7] - action_fp_f[..., :7]
        model_fp_energy_7 = float(torch.mean(action_fp_f[..., :7] ** 2).cpu())
        full32_discrepancy = float(torch.mean((action_quant_f - action_fp_f) ** 2).cpu())
        row = {
            key: record[key]
            for key in (
                "key", "split", "domain", "suite", "task_id", "trial_id", "trajectory_id",
                "step", "policy_step", "phase", "gripper_switch", "tail_state",
                "trajectory_success", "trajectory_steps",
            )
        }
        row.update({
            "action_discrepancy": discrepancy,
            "normalized_action_discrepancy": discrepancy / max(fp_energy, 1e-8),
            "fp_action_energy": fp_energy,
            "full32_action_discrepancy": full32_discrepancy,
            "model7_action_discrepancy": float(torch.mean(model_delta_7 ** 2).cpu()),
            "normalized_model7_action_discrepancy": float(torch.mean(model_delta_7 ** 2).cpu())
            / max(model_fp_energy_7, 1e-8),
            "translation_discrepancy": float(torch.mean(executed_delta[..., :3] ** 2)),
            "rotation_discrepancy": float(torch.mean(executed_delta[..., 3:6] ** 2)),
            "gripper_disagreement": float(torch.mean((executed_delta[..., 6] != 0).float())),
            "chunk_executed_action_discrepancy": float(torch.mean(chunk_delta ** 2)),
            "chunk_gripper_disagreement": float(torch.mean((chunk_delta[..., 6] != 0).float())),
            "active_linears": active_linears,
        })
        rows.append(row)
        print(json.dumps({"progress": f"{index + 1}/{len(records)}", **row}), flush=True)

    trajectory = summarize_trajectories(rows, args)
    payload = {
        "target": args.target,
        "high_precision_blocks": sorted(high_precision_blocks),
        "high_precision_weight_bits": args.high_precision_weight_bits,
        "cvar_fraction": args.cvar_fraction,
        "growth_weight": args.growth_weight,
        "critical_weight": args.critical_weight,
        "atm_alpha": str(args.atm_alpha) if args.atm_alpha else None,
        "states": len(rows),
        "samples": rows,
        **trajectory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["by_split"], indent=2))


if __name__ == "__main__":
    main()
