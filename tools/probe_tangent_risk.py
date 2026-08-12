#!/usr/bin/env python
"""Compare block tangent risk with isolated-block action discrepancy in GR00T."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats
from transformers.feature_extraction_utils import BatchFeature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--packdir")
    parser.add_argument("--block", help="llm:<index> or dit:<index>")
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--num-probes", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--aggregate", nargs="+", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def exact_spearman_permutation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    observed = float(np.dot(x_centered, y_centered) / denominator)
    count = 0
    total = 0
    for permutation in itertools.permutations(y_rank.tolist()):
        rho = float(np.dot(x_centered, np.asarray(permutation) - np.mean(y_rank)) / denominator)
        count += int(abs(rho) >= abs(observed) - 1e-12)
        total += 1
    return observed, count / total


def exact_stratified_spearman_permutation(
    x: np.ndarray, y: np.ndarray, groups: list[list[int]]
) -> tuple[float, float]:
    observed = float(stats.spearmanr(x, y).statistic)
    grouped_permutations = [
        list(itertools.permutations(y[indexes].tolist())) for indexes in groups
    ]
    count = 0
    total = 0
    for grouped_values in itertools.product(*grouped_permutations):
        permutation = y.copy()
        for indexes, values in zip(groups, grouped_values):
            permutation[indexes] = values
        rho = float(stats.spearmanr(x, permutation).statistic)
        count += int(abs(rho) >= abs(observed) - 1e-12)
        total += 1
    return observed, count / total


def aggregate(args: argparse.Namespace) -> None:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.aggregate]
    payloads.sort(key=lambda item: item["block"])
    tangent = np.asarray([item["overall"]["tangent_risk"] for item in payloads], dtype=float)
    action = np.asarray([item["overall"]["action_discrepancy"] for item in payloads], dtype=float)
    rho, exact_p = exact_spearman_permutation(tangent, action)
    pearson = stats.pearsonr(tangent, action)

    family_indices = {
        family: [index for index, item in enumerate(payloads) if item["block"].startswith(f"{family}:")]
        for family in ("llm", "dit")
    }
    _, stratified_p = exact_stratified_spearman_permutation(
        tangent, action, list(family_indices.values())
    )

    sample_keys = sorted(
        set.intersection(*(set(item["samples_by_key"].keys()) for item in payloads))
    )
    rng = np.random.default_rng(args.seed)
    bootstrap_rhos = []
    if sample_keys:
        for _ in range(args.bootstrap):
            sampled_keys = rng.choice(sample_keys, size=len(sample_keys), replace=True)
            tangent_b = []
            action_b = []
            for item in payloads:
                values = [item["samples_by_key"][str(key)] for key in sampled_keys]
                tangent_b.append(np.mean([value["tangent_risk"] for value in values]))
                action_b.append(np.mean([value["action_discrepancy"] for value in values]))
            bootstrap_rhos.append(float(stats.spearmanr(tangent_b, action_b).statistic))
    ci = (
        [float(np.nanpercentile(bootstrap_rhos, 2.5)), float(np.nanpercentile(bootstrap_rhos, 97.5))]
        if bootstrap_rhos
        else [float("nan"), float("nan")]
    )
    robust = {}
    for estimator, reducer in (
        ("median", np.median),
        ("trimmed_mean_10pct", lambda values: stats.trim_mean(values, 0.1)),
    ):
        tangent_robust = np.asarray(
            [reducer([item["samples_by_key"][key]["tangent_risk"] for key in sample_keys]) for item in payloads]
        )
        action_robust = np.asarray(
            [reducer([item["samples_by_key"][key]["action_discrepancy"] for key in sample_keys]) for item in payloads]
        )
        robust_rho, robust_p = exact_spearman_permutation(tangent_robust, action_robust)
        _, robust_stratified_p = exact_stratified_spearman_permutation(
            tangent_robust, action_robust, list(family_indices.values())
        )
        robust[estimator] = {
            "spearman_rho": robust_rho,
            "spearman_exact_permutation_p": robust_p,
            "spearman_family_stratified_permutation_p": robust_stratified_p,
        }
    domains = sorted(set.intersection(*(set(item["by_domain"].keys()) for item in payloads)))
    by_domain = {}
    for domain in domains:
        tangent_domain = np.asarray(
            [item["by_domain"][domain]["tangent_risk"] for item in payloads], dtype=float
        )
        action_domain = np.asarray(
            [item["by_domain"][domain]["action_discrepancy"] for item in payloads], dtype=float
        )
        domain_rho, domain_exact_p = exact_spearman_permutation(tangent_domain, action_domain)
        domain_pearson = stats.pearsonr(tangent_domain, action_domain)
        by_domain[domain] = {
            "spearman_rho": domain_rho,
            "spearman_exact_permutation_p": domain_exact_p,
            "pearson_r": float(domain_pearson.statistic),
            "pearson_p": float(domain_pearson.pvalue),
        }

    by_family = {}
    for family in ("llm", "dit"):
        selected = family_indices[family]
        tangent_family = tangent[selected]
        action_family = action[selected]
        family_rho, family_exact_p = exact_spearman_permutation(tangent_family, action_family)
        family_pearson = stats.pearsonr(tangent_family, action_family)
        by_family[family] = {
            "num_blocks": len(selected),
            "spearman_rho": family_rho,
            "spearman_exact_permutation_p": family_exact_p,
            "pearson_r": float(family_pearson.statistic),
            "pearson_p": float(family_pearson.pvalue),
        }
    result = {
        "blocks": [item["block"] for item in payloads],
        "num_blocks": len(payloads),
        "common_samples": len(sample_keys),
        "spearman_rho": rho,
        "spearman_exact_permutation_p": exact_p,
        "spearman_family_stratified_permutation_p": stratified_p,
        "spearman_bootstrap_ci95": ci,
        "robust_aggregation": robust,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "by_domain": by_domain,
        "by_family": by_family,
        "ranking": sorted(
            [
                {
                    "block": item["block"],
                    "tangent_risk": item["overall"]["tangent_risk"],
                    "action_discrepancy": item["overall"]["action_discrepancy"],
                }
                for item in payloads
            ],
            key=lambda row: row["tangent_risk"],
            reverse=True,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def set_quant_environment(packdir: str) -> None:
    values = {
        "GR00T_DUQUANT_SCOPE": "",
        "GR00T_DUQUANT_INCLUDE": r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*",
        "GR00T_DUQUANT_EXCLUDE": r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)",
        "GR00T_DUQUANT_WBITS_DEFAULT": "4",
        "GR00T_DUQUANT_ABITS": "8",
        "GR00T_DUQUANT_BLOCK": "64",
        "GR00T_DUQUANT_PERMUTE": "0",
        "GR00T_DUQUANT_ROW_ROT": "restore",
        "GR00T_DUQUANT_ACT_PCT": "99.9",
        "GR00T_DUQUANT_CALIB_STEPS": "32",
        "GR00T_DUQUANT_LS": "0.15",
        "GR00T_DUQUANT_PACKDIR": packdir,
        "GR00T_DUQUANT_PRECACHE_WEIGHTS": "1",
        "GR00T_ATM_ENABLE": "0",
        "GR00T_OHB_ENABLE": "0",
    }
    os.environ.update(values)


def block_paths(block: str) -> tuple[str, str]:
    family, raw_index = block.split(":", 1)
    index = int(raw_index)
    if family == "llm":
        return (
            f"backbone.eagle_model.language_model.model.layers.{index}",
            f"backbone.eagle_model.language_model.model.layers.{index}.",
        )
    if family == "dit":
        return (
            f"action_head.model.transformer_blocks.{index}",
            f"action_head.model.transformer_blocks.{index}.",
        )
    raise ValueError(f"Unsupported block family: {family!r}")


def first_tensor(output: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(output, torch.Tensor):
        return output, lambda value: value
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], lambda value: (value, *output[1:])
    if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
        return output[0], lambda value: [value, *output[1:]]
    raise TypeError(f"Unsupported hooked block output type: {type(output)!r}")


class BlockCapture:
    def __init__(self) -> None:
        self.tensors: list[torch.Tensor] = []
        self.make_leaf = False

    def hook(self, _module, _inputs, output):
        tensor, rebuild = first_tensor(output)
        if self.make_leaf:
            tensor = tensor.detach().requires_grad_(True)
            self.tensors.append(tensor)
            return rebuild(tensor)
        self.tensors.append(tensor.detach())
        return None

    def reset(self, *, make_leaf: bool) -> None:
        self.tensors.clear()
        self.make_leaf = make_leaf


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open("rb") as stream:
            records.extend(pickle.load(stream))
    return records


def prepare_normalized_input(policy, observation: dict[str, Any]) -> dict[str, Any]:
    from gr00t.model.policy import unsqueeze_dict_values

    obs_copy = {key: np.asarray(value).copy() for key, value in observation.items()}
    normalized = policy.apply_transforms(unsqueeze_dict_values(obs_copy))
    return normalized


def differentiable_action(model, normalized_input, initial_noise: torch.Tensor) -> torch.Tensor:
    backbone_inputs, action_input = model.prepare_input(normalized_input)
    backbone_output = model.backbone(backbone_inputs)
    head = model.action_head
    backbone_output = head.process_backbone_output(backbone_output)
    vl_embs = backbone_output.backbone_features
    vl_attn_mask = backbone_output.backbone_attention_mask
    embodiment_id = action_input.embodiment_id
    state_features = head.state_encoder(action_input.state, embodiment_id)
    actions = initial_noise.to(device=vl_embs.device, dtype=vl_embs.dtype)
    num_steps = head.num_inference_timesteps
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_cont = step / float(num_steps)
        t_discretized = int(t_cont * head.num_timestep_buckets)
        timesteps = torch.full(
            (actions.shape[0],), t_discretized, dtype=torch.long, device=actions.device
        )
        action_features = head.action_encoder(actions, timesteps, embodiment_id)
        if head.config.add_pos_embed:
            position_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=actions.device
            )
            action_features = action_features + head.position_embedding(position_ids).unsqueeze(0)
        future_tokens = head.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        model_output = head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=timesteps,
        )
        prediction = head.action_decoder(model_output, embodiment_id)
        actions = actions + dt * prediction[:, -head.action_horizon :]
    return actions


def configure_modules(model, target_prefix: str, *, target_active: bool) -> int:
    from gr00t.quantization.duquant_layers import DuQuantLinear

    active_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, DuQuantLinear):
            continue
        if not getattr(module, "_probe_private_cfg", False):
            module.cfg = copy.copy(module.cfg)
            module._probe_private_cfg = True
        active = target_active and name.startswith(target_prefix)
        module.weight_bits = 4 if active else 0
        module.cfg.act_bits = 8 if active else 0
        module._cached_weight_key = None
        module._weight_quantized_cached = False
        if active:
            active_count += 1
    return active_count


def run_probe(args: argparse.Namespace) -> None:
    if not args.checkpoint or not args.packdir or not args.block or not args.inputs:
        raise ValueError("--checkpoint, --packdir, --block and --inputs are required")
    set_quant_environment(args.packdir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
    model = policy.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    module_path, target_prefix = block_paths(args.block)
    named_modules = dict(model.named_modules())
    if module_path not in named_modules:
        available = [name for name in named_modules if "transformer_blocks" in name or ".layers." in name]
        raise KeyError(f"Block {module_path!r} not found. Examples: {available[:20]}")
    capture = BlockCapture()
    handle = named_modules[module_path].register_forward_hook(capture.hook)
    records = load_records(args.inputs)
    if args.max_records is not None:
        records = records[: args.max_records]
    rows = []
    try:
        for record_index, record in enumerate(records):
            normalized_input = prepare_normalized_input(policy, record["observation"])
            head = model.action_head
            generator = torch.Generator(device=args.device)
            generator.manual_seed(args.seed + record_index)
            initial_noise = torch.randn(
                (1, head.config.action_horizon, head.config.action_dim),
                generator=generator,
                device=args.device,
                dtype=head.dtype,
            )

            active_count = configure_modules(model, target_prefix, target_active=True)
            if active_count == 0:
                raise RuntimeError(f"No DuQuantLinear modules matched {target_prefix!r}")
            capture.reset(make_leaf=False)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                action_quant = differentiable_action(model, normalized_input, initial_noise)
            quant_hidden = list(capture.tensors)

            configure_modules(model, target_prefix, target_active=False)
            capture.reset(make_leaf=False)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                action_fp_reference = differentiable_action(model, normalized_input, initial_noise)
            fp_reference_hidden = list(capture.tensors)

            capture.reset(make_leaf=True)
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                action_fp = differentiable_action(model, normalized_input, initial_noise)
            fp_hidden = list(capture.tensors)
            if len(quant_hidden) != len(fp_reference_hidden) or len(quant_hidden) != len(fp_hidden):
                raise RuntimeError(
                    f"Capture count mismatch for {args.block}: quant={len(quant_hidden)} "
                    f"fp_reference={len(fp_reference_hidden)} fp_grad={len(fp_hidden)}"
                )
            deltas = [
                quant.to(torch.float32) - full.to(torch.float32)
                for quant, full in zip(quant_hidden, fp_reference_hidden)
            ]
            hidden_mse_by_step = [float(delta.square().mean().detach().cpu()) for delta in deltas]
            action_discrepancy = float(
                torch.mean((action_quant.float() - action_fp_reference.float()) ** 2).cpu()
            )
            fp_replay_discrepancy = float(
                torch.mean((action_fp_reference.float() - action_fp.detach().float()) ** 2).cpu()
            )
            tangent_values = []
            tangent_values_by_step = [[] for _ in deltas]
            jacobian_energy_by_step = [[] for _ in deltas]
            for probe_index in range(args.num_probes):
                probe_generator = torch.Generator(device=args.device)
                probe_generator.manual_seed(args.seed + 100000 + record_index * 100 + probe_index)
                random_bits = torch.randint(
                    0,
                    2,
                    action_fp.shape,
                    generator=probe_generator,
                    device=action_fp.device,
                )
                probe = random_bits.to(action_fp.dtype).mul_(2).sub_(1)
                scalar = torch.sum(action_fp * probe)
                gradients = torch.autograd.grad(
                    scalar,
                    fp_hidden,
                    retain_graph=probe_index + 1 < args.num_probes,
                    allow_unused=False,
                )
                projected_by_step = [
                    torch.sum(gradient.float() * delta)
                    for gradient, delta in zip(gradients, deltas)
                ]
                projected = sum(projected_by_step)
                tangent_values.append(float((projected.square() / action_fp.numel()).detach().cpu()))
                for step_index, (step_projected, gradient) in enumerate(
                    zip(projected_by_step, gradients)
                ):
                    tangent_values_by_step[step_index].append(
                        float((step_projected.square() / action_fp.numel()).detach().cpu())
                    )
                    jacobian_energy_by_step[step_index].append(
                        float((gradient.float().square().sum() / action_fp.numel()).detach().cpu())
                    )
            tangent_risk = float(np.mean(tangent_values))
            tangent_risk_by_step = [float(np.mean(values)) for values in tangent_values_by_step]
            jacobian_energy_by_step_mean = [
                float(np.mean(values)) for values in jacobian_energy_by_step
            ]
            key = f"{record['domain']}:{record['suite']}:{record['task_id']}:{record['trial_id']}"
            row = {
                "key": key,
                "domain": record["domain"],
                "task_id": record["task_id"],
                "trial_id": record["trial_id"],
                "action_discrepancy": action_discrepancy,
                "fp_replay_discrepancy": fp_replay_discrepancy,
                "tangent_risk": tangent_risk,
                "tangent_risk_by_step": tangent_risk_by_step,
                "hidden_mse": float(np.mean(hidden_mse_by_step)),
                "hidden_mse_by_step": hidden_mse_by_step,
                "jacobian_energy": float(np.mean(jacobian_energy_by_step_mean)),
                "jacobian_energy_by_step": jacobian_energy_by_step_mean,
                "capture_count": len(fp_hidden),
                "active_linears": active_count,
            }
            rows.append(row)
            print(json.dumps({"block": args.block, "progress": f"{record_index + 1}/{len(records)}", **row}))
            del action_quant, action_fp_reference, action_fp
            del fp_reference_hidden, fp_hidden, quant_hidden, deltas
            torch.cuda.empty_cache()
    finally:
        handle.remove()

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)

    def summarize(values: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "count": len(values),
            "tangent_risk": float(np.mean([value["tangent_risk"] for value in values])),
            "tangent_risk_by_step": np.mean(
                [value["tangent_risk_by_step"] for value in values], axis=0
            ).tolist(),
            "hidden_mse": float(np.mean([value["hidden_mse"] for value in values])),
            "hidden_mse_by_step": np.mean(
                [value["hidden_mse_by_step"] for value in values], axis=0
            ).tolist(),
            "jacobian_energy": float(np.mean([value["jacobian_energy"] for value in values])),
            "jacobian_energy_by_step": np.mean(
                [value["jacobian_energy_by_step"] for value in values], axis=0
            ).tolist(),
            "action_discrepancy": float(
                np.mean([value["action_discrepancy"] for value in values])
            ),
            "fp_replay_discrepancy": float(
                np.mean([value["fp_replay_discrepancy"] for value in values])
            ),
        }

    payload = {
        "block": args.block,
        "checkpoint": args.checkpoint,
        "packdir": args.packdir,
        "num_probes": args.num_probes,
        "denoising_steps": args.denoising_steps,
        "overall": summarize(rows),
        "by_domain": {domain: summarize(values) for domain, values in sorted(by_domain.items())},
        "samples": rows,
        "samples_by_key": {row["key"]: row for row in rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["overall"], indent=2))


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_probe(args)


if __name__ == "__main__":
    main()
