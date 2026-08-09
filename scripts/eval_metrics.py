from __future__ import annotations

import json
import math
import pathlib
import socket
import statistics
from collections import OrderedDict
from datetime import datetime
from typing import Any


SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")
DISPLAY_SUITE = {
    "libero_spatial": "spatial",
    "libero_goal": "goal",
    "libero_object": "object",
    "libero_10": "libero_10",
}
MODELS = OrderedDict(
    (
        (
            "groot-fp16",
            {
                "label": "GR00T N1.5 · FP16",
                "precision": "FP16",
                "quantization_backend": "none",
            },
        ),
        (
            "groot-quantvla-w4a8",
            {
                "label": "GR00T N1.5 · QuantVLA W4A8",
                "precision": "W4A8",
                "quantization_backend": "fake-quant",
            },
        ),
    )
)
BENCHMARKS = OrderedDict(
    (
        ("libero", {"label": "Standard LIBERO"}),
        ("libero-plus", {"label": "LIBERO-Plus · stratified subset"}),
    )
)
CATEGORY_ORDER = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)
CATEGORY_LABEL = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    rounded = int(round(seconds))
    days, remainder = divmod(rounded, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def wilson_interval(successes: int, episodes: int, z: float = 1.96) -> tuple[float, float]:
    if episodes <= 0:
        return 0.0, 0.0
    rate = successes / episodes
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes * episodes))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def load_sample_manifest(repo_root: pathlib.Path) -> dict[str, Any]:
    path = repo_root / "configs" / "libero_plus" / "first_100_per_category.json"
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    result["path"] = str(path)
    return result


def benchmark_totals(repo_root: pathlib.Path) -> dict[str, dict[str, int]]:
    manifest = load_sample_manifest(repo_root)
    plus_per_suite = int(manifest["per_suite_per_category"]) * len(manifest["categories"])
    return {
        "libero": {suite: 50 for suite in SUITES},
        "libero-plus": {suite: plus_per_suite for suite in SUITES},
    }


def _record_key(benchmark: str, record: dict[str, Any]) -> tuple[int, ...]:
    task_index = int(record["task_index"])
    if benchmark == "libero":
        return task_index, int(record.get("episode_index", 0))
    return (task_index,)


def load_effective_records(
    path: pathlib.Path, benchmark: str
) -> tuple[list[dict[str, Any]], int, int]:
    if not path.exists():
        return [], 0, 0
    effective: dict[tuple[int, ...], tuple[int, dict[str, Any]]] = {}
    malformed = 0
    valid_lines = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for sequence, line in enumerate(stream):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = _record_key(benchmark, record)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                malformed += 1
                continue
            valid_lines += 1
            effective[key] = (sequence, record)
    ordered = [item[1] for item in sorted(effective.values(), key=lambda item: item[0])]
    return ordered, malformed, valid_lines - len(ordered)


def _safe_number(record: dict[str, Any], field: str) -> float | None:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def suite_metrics(
    repo_root: pathlib.Path,
    model: str,
    benchmark: str,
    suite: str,
    total: int,
    eta_window: int,
) -> dict[str, Any]:
    path = repo_root / "output" / benchmark / model / suite / "episodes.jsonl"
    records, malformed, duplicates = load_effective_records(path, benchmark)
    completed_records = [record for record in records if record.get("error") is None]
    error_records = [record for record in records if record.get("error") is not None]
    successes = sum(bool(record.get("success")) for record in completed_records)
    durations = [
        value
        for record in completed_records
        if (value := _safe_number(record, "duration_seconds")) is not None and value > 0
    ]
    steps = [
        value
        for record in completed_records
        if (value := _safe_number(record, "steps")) is not None and value >= 0
    ]
    recent = durations[-eta_window:] if eta_window > 0 else durations
    typical = statistics.median(recent) if recent else None
    completed = len(completed_records)
    low, high = wilson_interval(successes, completed)
    return {
        "model": model,
        "model_label": MODELS[model]["label"],
        "precision": MODELS[model]["precision"],
        "quantization_backend": MODELS[model]["quantization_backend"],
        "benchmark": benchmark,
        "benchmark_label": BENCHMARKS[benchmark]["label"],
        "suite": suite,
        "suite_label": DISPLAY_SUITE[suite],
        "completed": completed,
        "total": total,
        "completion_rate": completed / total if total else 0.0,
        "successes": successes,
        "success_rate": successes / completed if completed else 0.0,
        "success_ci95_low": low,
        "success_ci95_high": high,
        "errors": len(error_records),
        "malformed_records": malformed,
        "resume_duplicates": duplicates,
        "typical_seconds": typical,
        "eta_seconds": max(0, total - completed) * typical if typical is not None else None,
        "mean_duration_seconds": statistics.fmean(durations) if durations else None,
        "median_duration_seconds": statistics.median(durations) if durations else None,
        "mean_steps": statistics.fmean(steps) if steps else None,
        "median_steps": statistics.median(steps) if steps else None,
        "records": records,
        "episodes_path": str(path),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = sum(row["completed"] for row in rows)
    total = sum(row["total"] for row in rows)
    successes = sum(row["successes"] for row in rows)
    low, high = wilson_interval(successes, completed)
    etas = [row["eta_seconds"] for row in rows if row["eta_seconds"] is not None]
    return {
        "completed": completed,
        "total": total,
        "completion_rate": completed / total if total else 0.0,
        "successes": successes,
        "success_rate": successes / completed if completed else 0.0,
        "success_ci95_low": low,
        "success_ci95_high": high,
        "errors": sum(row["errors"] for row in rows),
        "malformed_records": sum(row.get("malformed_records", 0) for row in rows),
        "resume_duplicates": sum(row.get("resume_duplicates", 0) for row in rows),
        "eta_seconds": max(etas) if etas else None,
    }


def _difficulty_label(value: Any) -> str:
    return "Unspecified" if value is None else str(value)


def _selected_plus_metadata(repo_root: pathlib.Path) -> tuple[pathlib.Path, list[dict[str, Any]]]:
    plus_root = repo_root.parent / "LIBERO-plus"
    path = plus_root / "libero" / "libero" / "benchmark" / "task_classification.json"
    with path.open(encoding="utf-8") as stream:
        classification = json.load(stream)
    manifest = load_sample_manifest(repo_root)
    per_group = int(manifest["per_suite_per_category"])
    categories = [str(value) for value in manifest["categories"]]
    selected: list[dict[str, Any]] = []
    for suite in SUITES:
        rows = classification[suite]
        for category in categories:
            candidates = [item for item in rows if str(item["category"]) == category]
            for item in candidates[:per_group]:
                selected.append(
                    {
                        "suite": suite,
                        "category": category,
                        "difficulty": _difficulty_label(item.get("difficulty_level")),
                    }
                )
    return path, selected


def _aggregate_group(records: list[dict[str, Any]], total: int) -> dict[str, Any]:
    completed_records = [record for record in records if record.get("error") is None]
    successes = sum(bool(record.get("success")) for record in completed_records)
    durations = [
        value
        for record in completed_records
        if (value := _safe_number(record, "duration_seconds")) is not None and value > 0
    ]
    steps = [
        value
        for record in completed_records
        if (value := _safe_number(record, "steps")) is not None and value >= 0
    ]
    completed = len(completed_records)
    low, high = wilson_interval(successes, completed)
    return {
        "completed": completed,
        "total": total,
        "completion_rate": completed / total if total else 0.0,
        "successes": successes,
        "success_rate": successes / completed if completed else 0.0,
        "success_ci95_low": low,
        "success_ci95_high": high,
        "errors": sum(record.get("error") is not None for record in records),
        "mean_steps": statistics.fmean(steps) if steps else None,
        "median_steps": statistics.median(steps) if steps else None,
        "mean_duration_seconds": statistics.fmean(durations) if durations else None,
        "median_duration_seconds": statistics.median(durations) if durations else None,
    }


def plus_group_metrics(
    repo_root: pathlib.Path, suite_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    classification_path, metadata = _selected_plus_metadata(repo_root)
    records_by_suite = {row["suite"]: row["records"] for row in suite_rows}
    all_records = [record for row in suite_rows for record in row["records"]]
    categories = []
    suite_categories = []
    for category in CATEGORY_ORDER:
        expected = sum(item["category"] == category for item in metadata)
        records = [record for record in all_records if record.get("category") == category]
        categories.append(
            {
                "category": category,
                "category_label": CATEGORY_LABEL[category],
                **_aggregate_group(records, expected),
            }
        )
        for suite in SUITES:
            suite_expected = sum(
                item["suite"] == suite and item["category"] == category for item in metadata
            )
            suite_records = [
                record
                for record in records_by_suite[suite]
                if record.get("category") == category
            ]
            suite_categories.append(
                {
                    "suite": suite,
                    "category": category,
                    "category_label": CATEGORY_LABEL[category],
                    **_aggregate_group(suite_records, suite_expected),
                }
            )
    difficulty_order = ["1", "2", "3", "4", "5", "Unspecified"]
    difficulties = []
    suite_difficulties = []
    category_difficulties = []
    for difficulty in difficulty_order:
        expected = sum(item["difficulty"] == difficulty for item in metadata)
        if expected:
            records = [
                record
                for record in all_records
                if _difficulty_label(record.get("difficulty_level")) == difficulty
            ]
            difficulties.append({"difficulty": difficulty, **_aggregate_group(records, expected)})
        for suite in SUITES:
            suite_expected = sum(
                item["suite"] == suite and item["difficulty"] == difficulty for item in metadata
            )
            if suite_expected:
                records = [
                    record
                    for record in records_by_suite[suite]
                    if _difficulty_label(record.get("difficulty_level")) == difficulty
                ]
                suite_difficulties.append(
                    {"suite": suite, "difficulty": difficulty, **_aggregate_group(records, suite_expected)}
                )
        for category in CATEGORY_ORDER:
            category_expected = sum(
                item["category"] == category and item["difficulty"] == difficulty
                for item in metadata
            )
            if category_expected:
                records = [
                    record
                    for record in all_records
                    if record.get("category") == category
                    and _difficulty_label(record.get("difficulty_level")) == difficulty
                ]
                category_difficulties.append(
                    {
                        "category": category,
                        "category_label": CATEGORY_LABEL[category],
                        "difficulty": difficulty,
                        **_aggregate_group(records, category_expected),
                    }
                )
    return {
        "classification_path": str(classification_path),
        "categories": categories,
        "suite_categories": suite_categories,
        "difficulties": difficulties,
        "suite_difficulties": suite_difficulties,
        "category_difficulties": category_difficulties,
    }


def runtime_status(repo_root: pathlib.Path) -> dict[str, Any]:
    manifest = repo_root / "runtime_logs" / "eval" / "processes.tsv"
    processes: list[dict[str, Any]] = []
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) != 7:
                continue
            kind, model, benchmark, suite, pid_text, gpu_text, port_text = fields
            try:
                pid, gpu, port = int(pid_text), int(gpu_text), int(port_text)
            except ValueError:
                continue
            processes.append(
                {
                    "kind": kind,
                    "model": model,
                    "benchmark": benchmark,
                    "suite": suite,
                    "pid": pid,
                    "gpu": gpu,
                    "port": port,
                    "alive": pathlib.Path(f"/proc/{pid}").exists(),
                }
            )
    server_ports = sorted({item["port"] for item in processes if item["kind"] == "server"})
    ports: dict[str, bool] = {}
    for port in server_ports:
        connection = socket.socket()
        connection.settimeout(0.15)
        ports[str(port)] = connection.connect_ex(("127.0.0.1", port)) == 0
        connection.close()
    return {
        "manifest": str(manifest),
        "processes": processes,
        "servers_alive": sum(item["alive"] for item in processes if item["kind"] == "server"),
        "servers_total": sum(item["kind"] == "server" for item in processes),
        "evaluators_alive": sum(item["alive"] for item in processes if item["kind"] == "eval"),
        "evaluators_total": sum(item["kind"] == "eval" for item in processes),
        "ports_alive": sum(ports.values()),
        "ports_total": len(ports),
        "ports": ports,
    }


def build_snapshot(repo_root: pathlib.Path, eta_window: int = 10) -> dict[str, Any]:
    totals_config = benchmark_totals(repo_root)
    models: dict[str, Any] = OrderedDict()
    for model in MODELS:
        benchmark_rows: dict[str, list[dict[str, Any]]] = OrderedDict()
        totals: dict[str, dict[str, Any]] = OrderedDict()
        for benchmark in BENCHMARKS:
            rows = [
                suite_metrics(
                    repo_root,
                    model,
                    benchmark,
                    suite,
                    totals_config[benchmark][suite],
                    eta_window,
                )
                for suite in SUITES
            ]
            benchmark_rows[benchmark] = rows
            totals[benchmark] = aggregate_rows(rows)
        models[model] = {
            "config": MODELS[model],
            "benchmarks": benchmark_rows,
            "totals": totals,
            "libero_plus_groups": plus_group_metrics(repo_root, benchmark_rows["libero-plus"]),
        }
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "eta_window": eta_window,
        "sample_manifest": load_sample_manifest(repo_root),
        "models": models,
        "runtime": runtime_status(repo_root),
    }
