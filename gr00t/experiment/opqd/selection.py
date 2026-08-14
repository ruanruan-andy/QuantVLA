"""Phase-balanced prioritized sampling for OPQD trajectories."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .config import OPQDSelectionConfig


@dataclass(frozen=True)
class SelectionResult:
    """Selected trajectory states and the metadata needed for logging."""

    indices: tuple[int, ...]
    phases: tuple[int, ...]
    reasons: tuple[str, ...]
    priority_scores: torch.Tensor
    weights: torch.Tensor
    gap_relaxed: bool


def _percentile_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic ranks in ``[0, 1]`` with average tied ranks."""
    if values.ndim != 1:
        raise ValueError("percentile ranks require a one-dimensional tensor")
    count = values.numel()
    if count == 0:
        return values.clone().float()
    if count == 1:
        return torch.ones_like(values, dtype=torch.float32)

    values = values.float()
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(count, dtype=torch.float32, device=values.device)
    start = 0
    while start < count:
        end = start + 1
        while end < count and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + end - 1) / 2.0) / (count - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _phase_for_index(index: int, trajectory_length: int, phase_bins: int) -> int:
    return min(phase_bins - 1, index * phase_bins // trajectory_length)


def _respects_gap(index: int, selected: list[int], gap: int) -> bool:
    return gap == 0 or all(abs(index - other) >= gap for other in selected)


def _append_candidates(
    candidates: list[int],
    *,
    count: int,
    reason: str,
    selected: list[int],
    reasons: dict[int, str],
    min_gap: int,
) -> None:
    for index in candidates:
        if index in reasons or not _respects_gap(index, selected, min_gap):
            continue
        selected.append(index)
        reasons[index] = reason
        count -= 1
        if count == 0:
            return


def select_phase_balanced_states(
    q_values: torch.Tensor,
    r_values: torch.Tensor,
    config: OPQDSelectionConfig,
    *,
    rng: random.Random,
) -> SelectionResult:
    """Select a deterministic priority/random mixture across temporal bins.

    The configured temporal gap is enforced first.  Very short or highly
    constrained trajectories then use distinct no-gap fallbacks so the update
    still receives up to ``states_per_episode`` states without duplicates.
    """
    config.validate()
    if q_values.ndim != 1 or r_values.ndim != 1 or q_values.shape != r_values.shape:
        raise ValueError("q_values and r_values must be matching 1D tensors")
    if not torch.isfinite(q_values).all() or not torch.isfinite(r_values).all():
        raise ValueError("q_values and r_values must be finite")
    if torch.any(q_values < 0) or torch.any(r_values < 0):
        raise ValueError("q_values and r_values must be non-negative")

    trajectory_length = q_values.numel()
    if trajectory_length == 0:
        return SelectionResult((), (), (), torch.empty(0), torch.empty(0), False)

    phase_ids = [
        _phase_for_index(index, trajectory_length, config.phase_bins)
        for index in range(trajectory_length)
    ]
    scores = torch.zeros(trajectory_length, dtype=torch.float32)
    phase_candidates: list[list[int]] = []
    for phase in range(config.phase_bins):
        candidates = [index for index, value in enumerate(phase_ids) if value == phase]
        phase_candidates.append(candidates)
        if not candidates:
            continue
        candidate_tensor = torch.as_tensor(candidates, dtype=torch.long)
        q_rank = _percentile_ranks(q_values[candidate_tensor])
        r_rank = _percentile_ranks(r_values[candidate_tensor])
        scores[candidate_tensor] = config.alpha_q * q_rank + config.beta_r * r_rank

    selected: list[int] = []
    reason_by_index: dict[int, str] = {}
    for phase, candidates in enumerate(phase_candidates):
        priority = sorted(candidates, key=lambda index: (-float(scores[index]), index))
        _append_candidates(
            priority,
            count=config.priority_per_phase,
            reason=f"priority_phase_{phase}",
            selected=selected,
            reasons=reason_by_index,
            min_gap=config.min_temporal_gap,
        )

        random_candidates = [index for index in candidates if index not in reason_by_index]
        rng.shuffle(random_candidates)
        _append_candidates(
            random_candidates,
            count=config.random_per_phase,
            reason=f"random_phase_{phase}",
            selected=selected,
            reasons=reason_by_index,
            min_gap=config.min_temporal_gap,
        )

    budget = min(config.states_per_episode, trajectory_length)
    gap_relaxed = len(selected) < budget
    if gap_relaxed:
        remaining = [index for index in range(trajectory_length) if index not in reason_by_index]
        remaining.sort(key=lambda index: (-float(scores[index]), index))
        for index in remaining[: budget - len(selected)]:
            selected.append(index)
            reason_by_index[index] = "fallback_relaxed_gap"

    selected.sort()
    selected_tensor = torch.as_tensor(selected, dtype=torch.long)
    selected_scores = scores[selected_tensor]
    mean_score = selected_scores.mean().clamp_min(config.epsilon)
    weights = (selected_scores / mean_score).clamp(config.weight_min, config.weight_max)
    return SelectionResult(
        indices=tuple(selected),
        phases=tuple(phase_ids[index] for index in selected),
        reasons=tuple(reason_by_index[index] for index in selected),
        priority_scores=selected_scores.detach(),
        weights=weights.detach(),
        gap_relaxed=gap_relaxed,
    )
