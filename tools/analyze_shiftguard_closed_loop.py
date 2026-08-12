#!/usr/bin/env python
"""Analyze ShiftGuard block-risk predictors on disjoint calibration/validation tasks."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats


OOD_DOMAINS = ("Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout")
SEEN_DOMAINS = ("Camera", "Background", "Noise")
UNSEEN_DOMAINS = ("Robot", "Language", "Light", "Layout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def expand_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        paths.extend(Path(match) for match in matches)
    return paths


def load_payloads(values: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for path in expand_paths(values):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[payload["block"]] = payload
    return result


def permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
    groups: list[np.ndarray] | None = None,
) -> float:
    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    observed = float(np.dot(x_centered, y_centered) / denominator)
    extreme = 0
    for _ in range(permutations):
        permuted = y_centered.copy()
        if groups is None:
            permuted = rng.permutation(permuted)
        else:
            for indexes in groups:
                permuted[indexes] = rng.permutation(permuted[indexes])
        rho = float(np.dot(x_centered, permuted) / denominator)
        extreme += int(abs(rho) >= abs(observed) - 1e-12)
    return (extreme + 1) / (permutations + 1)


def correlation(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
    groups: list[np.ndarray] | None = None,
) -> dict[str, float]:
    spearman = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)
    return {
        "spearman_rho": float(spearman.statistic),
        "spearman_p_asymptotic": float(spearman.pvalue),
        "spearman_permutation_p": permutation_p(x, y, rng, permutations, groups),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
    }


def family(block: str) -> str:
    return block.split(":", 1)[0]


def domain_values(payload: dict[str, Any], field: str, domains: tuple[str, ...]) -> np.ndarray:
    return np.asarray([payload["by_domain"][domain][field] for domain in domains], dtype=float)


def scalar_reducer(name: str) -> Callable[[np.ndarray], float]:
    reducers = {
        "mean": lambda values: float(np.mean(values)),
        "mean_plus_half_std": lambda values: float(np.mean(values) + 0.5 * np.std(values)),
        "top2_mean": lambda values: float(np.mean(np.sort(values)[-2:])),
        "max": lambda values: float(np.max(values)),
    }
    return reducers[name]


def step_score(payload: dict[str, Any], domains: tuple[str, ...], mode: str) -> float:
    rows = np.asarray(
        [payload["by_domain"][domain]["tangent_risk_by_step"] for domain in domains],
        dtype=float,
    )
    steps = rows.shape[1]
    if mode == "uniform":
        weights = np.ones(steps)
    elif mode == "late_linear":
        weights = np.arange(1, steps + 1, dtype=float)
    elif mode == "late_quadratic":
        weights = np.arange(1, steps + 1, dtype=float) ** 2
    elif mode == "last":
        weights = np.zeros(steps)
        weights[-1] = 1.0
    else:
        raise KeyError(mode)
    weights /= weights.sum()
    return float(np.mean(rows @ weights))


def build_predictors(payload: dict[str, Any], domains: tuple[str, ...]) -> dict[str, float]:
    predictors: dict[str, float] = {}
    for field, prefix in (
        ("hidden_mse", "activation_mse"),
        ("jacobian_energy", "jacobian_energy"),
        ("tangent_risk", "tangent"),
        ("action_discrepancy", "action_risk"),
    ):
        values = domain_values(payload, field, domains)
        for reducer_name in ("mean", "mean_plus_half_std", "top2_mean", "max"):
            predictors[f"{prefix}_{reducer_name}"] = scalar_reducer(reducer_name)(values)
    if family(payload["block"]) == "dit":
        for mode in ("uniform", "late_linear", "late_quadratic", "last"):
            predictors[f"tangent_step_{mode}"] = step_score(payload, domains, mode)
    id_row = payload.get("by_domain", {}).get("ID")
    if id_row is not None:
        id_risk = float(id_row["tangent_risk"])
        ood = domain_values(payload, "tangent_risk", domains)
        predictors["tangent_excess_top2"] = float(np.mean(np.sort(ood - id_risk)[-2:]))
        predictors["tangent_ratio_top2"] = float(
            np.mean(np.sort(ood / max(id_risk, 1e-12))[-2:])
        )
        id_action = float(id_row["action_discrepancy"])
        ood_action = domain_values(payload, "action_discrepancy", domains)
        predictors["action_risk_excess_top2"] = float(
            np.mean(np.sort(ood_action - id_action)[-2:])
        )
        predictors["action_risk_ratio_top2"] = float(
            np.mean(np.sort(ood_action / max(id_action, 1e-12))[-2:])
        )
    return predictors


def target(payload: dict[str, Any], domains: tuple[str, ...]) -> float:
    return float(np.mean(domain_values(payload, "action_discrepancy", domains)))


def evaluate_predictors(
    blocks: list[str],
    predictors: dict[str, dict[str, float]],
    targets: dict[str, float],
    rng: np.random.Generator,
    permutations: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    families = np.asarray([family(block) for block in blocks])
    groups = [np.flatnonzero(families == value) for value in sorted(set(families))]
    predictor_names = sorted(set.union(*(set(predictors[block]) for block in blocks)))
    y = np.asarray([targets[block] for block in blocks], dtype=float)
    for name in predictor_names:
        row: dict[str, Any] = {}
        if all(name in predictors[block] for block in blocks):
            x = np.asarray([predictors[block][name] for block in blocks], dtype=float)
            row["all"] = correlation(x, y, rng, permutations, groups=groups)
        for family_name in ("llm", "dit"):
            indexes = np.flatnonzero(families == family_name)
            family_blocks = [blocks[index] for index in indexes]
            if all(name in predictors[block] for block in family_blocks):
                x_family = np.asarray(
                    [predictors[block][name] for block in family_blocks], dtype=float
                )
                row[family_name] = correlation(
                    x_family, y[indexes], rng, permutations, groups=None
                )
        result[name] = row
    return result


def best_by_family(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for family_name in ("llm", "dit"):
        ranked = sorted(
            (
                {
                    "predictor": name,
                    **values[family_name],
                }
                for name, values in metrics.items()
                if family_name in values
            ),
            key=lambda row: (row["spearman_rho"], -row["spearman_permutation_p"]),
            reverse=True,
        )
        result[family_name] = ranked[0]
    return result


def main() -> None:
    args = parse_args()
    calibration = load_payloads(args.calibration)
    validation = load_payloads(args.validation)
    blocks = sorted(set(calibration) & set(validation))
    if not blocks:
        raise RuntimeError("No common blocks between calibration and validation")
    rng = np.random.default_rng(args.seed)

    calibration_predictors = {
        block: build_predictors(calibration[block], OOD_DOMAINS) for block in blocks
    }
    calibration_targets = {
        block: target(calibration[block], OOD_DOMAINS) for block in blocks
    }
    validation_targets = {block: target(validation[block], OOD_DOMAINS) for block in blocks}
    fit_metrics = evaluate_predictors(
        blocks, calibration_predictors, calibration_targets, rng, args.permutations
    )
    validation_metrics = evaluate_predictors(
        blocks, calibration_predictors, validation_targets, rng, args.permutations
    )

    seen_predictors = {
        block: build_predictors(calibration[block], SEEN_DOMAINS) for block in blocks
    }
    unseen_targets = {block: target(validation[block], UNSEEN_DOMAINS) for block in blocks}
    unseen_metrics = evaluate_predictors(
        blocks, seen_predictors, unseen_targets, rng, args.permutations
    )

    selected = best_by_family(fit_metrics)
    best_validation = best_by_family(validation_metrics)
    best_unseen = best_by_family(unseen_metrics)
    selected_validation = {
        family_name: {
            "predictor": row["predictor"],
            **validation_metrics[row["predictor"]][family_name],
        }
        for family_name, row in selected.items()
    }
    selected_unseen = {
        family_name: {
            "predictor": row["predictor"],
            **unseen_metrics[row["predictor"]][family_name],
        }
        for family_name, row in selected.items()
    }
    result = {
        "blocks": blocks,
        "num_blocks": len(blocks),
        "domains": list(OOD_DOMAINS),
        "seen_domains": list(SEEN_DOMAINS),
        "unseen_domains": list(UNSEEN_DOMAINS),
        "calibration_fit": fit_metrics,
        "heldout_task_validation": validation_metrics,
        "heldout_shift_validation": unseen_metrics,
        "selected_on_calibration": selected,
        "selected_heldout_task_validation": selected_validation,
        "selected_heldout_shift_validation": selected_unseen,
        "best_on_heldout_task_validation": best_validation,
        "best_on_heldout_shift_validation": best_unseen,
        "calibration_predictors": calibration_predictors,
        "dit_robust_action_risk_ranking": sorted(
            [
                {
                    "block": block,
                    "score": calibration_predictors[block]["action_risk_mean_plus_half_std"],
                    "calibration_action_discrepancy": calibration_targets[block],
                    "validation_action_discrepancy": validation_targets[block],
                }
                for block in blocks
                if family(block) == "dit"
            ],
            key=lambda row: row["score"],
            reverse=True,
        ),
        "block_targets": {
            block: {
                "family": family(block),
                "calibration_action_discrepancy": calibration_targets[block],
                "validation_action_discrepancy": validation_targets[block],
                "unseen_action_discrepancy": unseen_targets[block],
            }
            for block in blocks
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "num_blocks": len(blocks),
        "selected_on_calibration": result["selected_on_calibration"],
        "selected_heldout_task_validation": result["selected_heldout_task_validation"],
        "selected_heldout_shift_validation": result["selected_heldout_shift_validation"],
        "best_on_heldout_task_validation": result["best_on_heldout_task_validation"],
        "best_on_heldout_shift_validation": result["best_on_heldout_shift_validation"],
        "dit_robust_action_risk_top5": result["dit_robust_action_risk_ranking"][:5],
    }, indent=2))


if __name__ == "__main__":
    main()
