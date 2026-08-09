from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
from collections import defaultdict
from typing import Any, Iterable

from rich.console import Console

from eval_metrics import BENCHMARKS, MODELS, build_snapshot, format_duration, wilson_interval


def _atomic_text(path: pathlib.Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: pathlib.Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _number(record: dict[str, Any], field: str) -> float | None:
    try:
        return float(record[field])
    except (KeyError, TypeError, ValueError):
        return None


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("error") is None]
    successes = sum(bool(record.get("success")) for record in completed)
    durations = [value for record in completed if (value := _number(record, "duration_seconds"))]
    steps = [
        value
        for record in completed
        if (value := _number(record, "steps")) is not None and value >= 0
    ]
    low, high = wilson_interval(successes, len(completed))
    return {
        "completed": len(completed),
        "successes": successes,
        "success_rate": successes / len(completed) if completed else 0.0,
        "success_ci95_low": low,
        "success_ci95_high": high,
        "errors": sum(record.get("error") is not None for record in records),
        "mean_steps": statistics.fmean(steps) if steps else None,
        "median_steps": statistics.median(steps) if steps else None,
        "mean_duration_seconds": statistics.fmean(durations) if durations else None,
        "median_duration_seconds": statistics.median(durations) if durations else None,
    }


def _suite_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, model_data in snapshot["models"].items():
        for benchmark, suites in model_data["benchmarks"].items():
            for row in suites:
                rows.append({key: value for key, value in row.items() if key != "records"})
    return rows


def _episode_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, model_data in snapshot["models"].items():
        config = MODELS[model]
        for benchmark, suites in model_data["benchmarks"].items():
            for suite_row in suites:
                for record in suite_row["records"]:
                    rows.append(
                        {
                            "model": model,
                            "precision": config["precision"],
                            "quantization_backend": config["quantization_backend"],
                            "benchmark": benchmark,
                            "suite": suite_row["suite"],
                            "task_index": record.get("task_index"),
                            "episode_index": record.get("episode_index"),
                            "classification_id": record.get("classification_id"),
                            "task_name": record.get("task_name"),
                            "task_description": record.get("task_description"),
                            "category": record.get("category"),
                            "difficulty_level": record.get("difficulty_level"),
                            "success": record.get("success"),
                            "steps": record.get("steps"),
                            "duration_seconds": record.get("duration_seconds"),
                            "error": record.get("error"),
                            "video_path": record.get("video_path"),
                        }
                    )
    return rows


def _plus_dimension_rows(snapshot: dict[str, Any], field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, model_data in snapshot["models"].items():
        for row in model_data["libero_plus_groups"][field]:
            rows.append(
                {
                    "model": model,
                    "precision": MODELS[model]["precision"],
                    "quantization_backend": MODELS[model]["quantization_backend"],
                    **row,
                }
            )
    return rows


def _standard_task_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for model, model_data in snapshot["models"].items():
        for suite_row in model_data["benchmarks"]["libero"]:
            for record in suite_row["records"]:
                groups[(model, suite_row["suite"], int(record["task_index"]))].append(record)
    return [
        {
            "model": model,
            "precision": MODELS[model]["precision"],
            "suite": suite,
            "task_index": task_index,
            **_aggregate_records(records),
        }
        for (model, suite, task_index), records in sorted(groups.items())
    ]


def _comparison_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    fp16 = snapshot["models"]["groot-fp16"]
    quant = snapshot["models"]["groot-quantvla-w4a8"]
    rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        fp_rows = {row["suite"]: row for row in fp16["benchmarks"][benchmark]}
        quant_rows = {row["suite"]: row for row in quant["benchmarks"][benchmark]}
        for suite in fp_rows:
            base = fp_rows[suite]
            candidate = quant_rows[suite]
            rows.append(
                {
                    "dimension": "suite",
                    "benchmark": benchmark,
                    "name": suite,
                    "fp16_completed": base["completed"],
                    "fp16_success_rate": base["success_rate"],
                    "w4a8_completed": candidate["completed"],
                    "w4a8_success_rate": candidate["success_rate"],
                    "success_rate_delta_points": (candidate["success_rate"] - base["success_rate"]) * 100,
                }
            )
    fp_categories = {
        row["category_label"]: row for row in fp16["libero_plus_groups"]["categories"]
    }
    quant_categories = {
        row["category_label"]: row for row in quant["libero_plus_groups"]["categories"]
    }
    for name, base in fp_categories.items():
        candidate = quant_categories[name]
        rows.append(
            {
                "dimension": "category",
                "benchmark": "libero-plus",
                "name": name,
                "fp16_completed": base["completed"],
                "fp16_success_rate": base["success_rate"],
                "w4a8_completed": candidate["completed"],
                "w4a8_success_rate": candidate["success_rate"],
                "success_rate_delta_points": (candidate["success_rate"] - base["success_rate"]) * 100,
            }
        )
    return rows


def _markdown_model(snapshot: dict[str, Any], model: str) -> str:
    data = snapshot["models"][model]
    lines = [f'## {MODELS[model]["label"]}', ""]
    for benchmark, config in BENCHMARKS.items():
        lines.extend(
            [
                f'### {config["label"]}',
                "",
                "| Suite | Progress | Success | Success rate | Errors | Typical | ETA |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in data["benchmarks"][benchmark]:
            lines.append(
                f'| {row["suite_label"]} | {row["completed"]}/{row["total"]} | '
                f'{row["successes"]} | {row["success_rate"] * 100:.1f}% | '
                f'{row["errors"]} | {format_duration(row["typical_seconds"])} | '
                f'{format_duration(row["eta_seconds"])} |'
            )
        total = data["totals"][benchmark]
        lines.append(
            f'| **Total** | **{total["completed"]}/{total["total"]}** | '
            f'**{total["successes"]}** | **{total["success_rate"] * 100:.1f}%** | '
            f'**{total["errors"]}** | — | **{format_duration(total["eta_seconds"])}** |'
        )
        lines.append("")
    lines.extend(
        [
            "### LIBERO-Plus categories",
            "",
            "| Category | Progress | Success | Success rate | Errors |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in data["libero_plus_groups"]["categories"]:
        lines.append(
            f'| {row["category_label"]} | {row["completed"]}/{row["total"]} | '
            f'{row["successes"]} | {row["success_rate"] * 100:.1f}% | {row["errors"]} |'
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect QuantVLA evaluation metrics")
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--eta-window", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or (repo_root / "output" / "summary")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot(repo_root, args.eta_window)
    suites = _suite_rows(snapshot)
    episodes = _episode_rows(snapshot)
    categories = _plus_dimension_rows(snapshot, "categories")
    suite_categories = _plus_dimension_rows(snapshot, "suite_categories")
    difficulties = _plus_dimension_rows(snapshot, "difficulties")
    suite_difficulties = _plus_dimension_rows(snapshot, "suite_difficulties")
    category_difficulties = _plus_dimension_rows(snapshot, "category_difficulties")
    standard_tasks = _standard_task_rows(snapshot)
    comparisons = _comparison_rows(snapshot)

    clean_summary = {
        "generated_at": snapshot["generated_at"],
        "repo_root": snapshot["repo_root"],
        "sample_manifest": snapshot["sample_manifest"],
        "partial": any(
            total["completed"] < total["total"]
            for model_data in snapshot["models"].values()
            for total in model_data["totals"].values()
        ),
        "models": snapshot["models"],
        "comparisons": comparisons,
        "runtime": snapshot["runtime"],
    }
    for model_data in clean_summary["models"].values():
        for rows in model_data["benchmarks"].values():
            for row in rows:
                row.pop("records", None)
    _atomic_text(output_dir / "summary.json", json.dumps(clean_summary, indent=2, ensure_ascii=False) + "\n")

    suite_fields = [
        "model", "model_label", "precision", "quantization_backend", "benchmark", "suite",
        "completed", "total", "completion_rate", "successes", "success_rate",
        "success_ci95_low", "success_ci95_high", "errors", "malformed_records",
        "resume_duplicates", "typical_seconds", "eta_seconds", "mean_steps", "median_steps",
        "mean_duration_seconds", "median_duration_seconds", "episodes_path",
    ]
    _write_csv(output_dir / "suites.csv", suite_fields, suites)
    _write_csv(
        output_dir / "episodes.csv",
        [
            "model", "precision", "quantization_backend", "benchmark", "suite", "task_index",
            "episode_index", "classification_id", "task_name", "task_description", "category",
            "difficulty_level", "success", "steps", "duration_seconds", "error", "video_path",
        ],
        episodes,
    )
    group_fields = [
        "model", "precision", "quantization_backend", "suite", "category", "category_label",
        "difficulty", "completed", "total", "completion_rate", "successes", "success_rate",
        "success_ci95_low", "success_ci95_high", "errors", "mean_steps", "median_steps",
        "mean_duration_seconds", "median_duration_seconds",
    ]
    _write_csv(output_dir / "libero_plus_by_category.csv", group_fields, categories)
    _write_csv(output_dir / "libero_plus_by_suite_category.csv", group_fields, suite_categories)
    _write_csv(output_dir / "libero_plus_by_difficulty.csv", group_fields, difficulties)
    _write_csv(output_dir / "libero_plus_by_suite_difficulty.csv", group_fields, suite_difficulties)
    _write_csv(output_dir / "libero_plus_by_category_difficulty.csv", group_fields, category_difficulties)
    _write_csv(
        output_dir / "standard_libero_by_task.csv",
        [
            "model", "precision", "suite", "task_index", "completed", "successes", "success_rate",
            "success_ci95_low", "success_ci95_high", "errors", "mean_steps", "median_steps",
            "mean_duration_seconds", "median_duration_seconds",
        ],
        standard_tasks,
    )
    _write_csv(
        output_dir / "fp16_vs_w4a8.csv",
        [
            "dimension", "benchmark", "name", "fp16_completed", "fp16_success_rate",
            "w4a8_completed", "w4a8_success_rate", "success_rate_delta_points",
        ],
        comparisons,
    )

    report = "\n\n".join(
        [
            "# GR00T FP16 vs QuantVLA W4A8 Evaluation Report",
            f'Generated: `{snapshot["generated_at"]}`',
            "Subset: 25 tasks per suite/category, 100 per category, 700 LIBERO-Plus rollouts per model.",
            _markdown_model(snapshot, "groot-fp16"),
            _markdown_model(snapshot, "groot-quantvla-w4a8"),
        ]
    )
    _atomic_text(output_dir / "report.md", report + "\n")

    console = Console()
    console.print("[bold green]Evaluation metrics collected[/bold green]")
    for model, model_data in snapshot["models"].items():
        console.print(f'  [bold]{MODELS[model]["label"]}[/bold]')
        for benchmark, total in model_data["totals"].items():
            console.print(
                f'    {BENCHMARKS[benchmark]["label"]}: {total["completed"]}/{total["total"]}, '
                f'success {total["success_rate"] * 100:.1f}%, errors {total["errors"]}'
            )
    console.print(f"  Report directory: [cyan]{output_dir}[/cyan]")


if __name__ == "__main__":
    main()
