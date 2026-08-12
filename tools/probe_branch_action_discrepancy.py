#!/usr/bin/env python
"""Measure action discrepancy when all LLM, DiT, or both branches are W4A8."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from probe_tangent_risk import differentiable_action, prepare_normalized_input, set_quant_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--packdir", required=True)
    parser.add_argument("--branch", choices=("llm", "dit", "all"), required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open("rb") as stream:
            records.extend(pickle.load(stream))
    return records


def configure(model, branch: str, active: bool) -> int:
    from gr00t.quantization.duquant_layers import DuQuantLinear

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, DuQuantLinear):
            continue
        if not getattr(module, "_branch_probe_private_cfg", False):
            module.cfg = copy.copy(module.cfg)
            module._branch_probe_private_cfg = True
        in_llm = name.startswith("backbone.eagle_model.language_model.")
        in_dit = name.startswith("action_head.model.transformer_blocks.")
        selected = branch == "all" or (branch == "llm" and in_llm) or (branch == "dit" and in_dit)
        enabled = active and selected
        module.weight_bits = 4 if enabled else 0
        module.cfg.act_bits = 8 if enabled else 0
        module._cached_weight_key = None
        module._weight_quantized_cached = False
        count += int(enabled)
    return count


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "action_discrepancy": float(np.mean([row["action_discrepancy"] for row in rows])),
    }


def main() -> None:
    args = parse_args()
    set_quant_environment(args.packdir)
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
        device=args.device,
    )
    model = policy.model.eval()
    rows = []
    records = load_records(args.inputs)
    for record_index, record in enumerate(records):
        normalized = prepare_normalized_input(policy, record["observation"])
        head = model.action_head
        generator = torch.Generator(device=args.device)
        generator.manual_seed(args.seed + record_index)
        noise = torch.randn(
            (1, head.config.action_horizon, head.config.action_dim),
            generator=generator,
            device=args.device,
            dtype=head.dtype,
        )
        active_linears = configure(model, args.branch, True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action_quant = differentiable_action(model, normalized, noise)
        configure(model, args.branch, False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action_fp = differentiable_action(model, normalized, noise)
        row = {
            "key": f"{record['domain']}:{record['suite']}:{record['task_id']}:{record['trial_id']}",
            "domain": record["domain"],
            "task_id": record["task_id"],
            "action_discrepancy": float(
                torch.mean((action_quant.float() - action_fp.float()) ** 2).cpu()
            ),
            "active_linears": active_linears,
        }
        rows.append(row)
        print(json.dumps({"progress": f"{record_index + 1}/{len(records)}", **row}))

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    payload = {
        "branch": args.branch,
        "overall": summarize(rows),
        "by_domain": {domain: summarize(values) for domain, values in sorted(by_domain.items())},
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
