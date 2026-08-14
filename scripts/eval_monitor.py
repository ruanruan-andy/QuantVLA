from __future__ import annotations

import argparse
import pathlib
import time

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from eval_metrics import (
    BENCHMARKS,
    CATEGORY_ORDER,
    METHOD_TO_MODEL,
    MODELS,
    PRIMARY_MODELS,
    build_snapshot,
    format_duration,
)


METHOD_NAMES = {
    "groot-fp16": "FP16",
    "groot-quantvla-w4a8": "QuantVLA",
    "groot-gap-opqd-w4a8": "QuantVLA-OPQD",
}


def _percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def _rate_style(rate: float | None, completed: int) -> str:
    if completed == 0 or rate is None:
        return "dim"
    if rate >= 0.9:
        return "green"
    if rate >= 0.5:
        return "yellow"
    return "red"


def _state_text(state: str) -> Text:
    styles = {
        "done": "bold green",
        "running": "bold cyan",
        "starting": "cyan",
        "stopped": "bold yellow",
        "empty": "yellow",
        "not-started": "dim",
    }
    labels = {
        "done": "DONE",
        "running": "RUNNING",
        "starting": "STARTING",
        "stopped": "STOPPED",
        "empty": "EMPTY",
        "not-started": "NOT STARTED",
    }
    return Text(labels.get(state, state.upper()), style=styles.get(state, "yellow"))


def _benchmark_table(title: str, rows: list[dict], total: dict) -> Table:
    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("Suite", no_wrap=True)
    table.add_column("状态", no_wrap=True)
    table.add_column("进度", justify="right", no_wrap=True)
    table.add_column("完成率", justify="right", no_wrap=True)
    table.add_column("成功", justify="right", no_wrap=True)
    table.add_column("成功率", justify="right", no_wrap=True)
    table.add_column("错误", justify="right", no_wrap=True)
    table.add_column("单次典型耗时", justify="right", no_wrap=True)
    table.add_column("预计剩余", justify="right", no_wrap=True)
    for row in rows:
        table.add_row(
            row["suite_label"],
            _state_text(row["runtime_state"]),
            f'{row["completed"]} / {row["total"]}',
            _percent(row["completion_rate"], 2 if row["completion_rate"] < 0.01 else 1),
            str(row["successes"]),
            Text(_percent(row["success_rate"]), style=_rate_style(row["success_rate"], row["evaluated"])),
            Text(str(row["errors"]), style="red" if row["errors"] else "green"),
            format_duration(row["typical_seconds"]),
            format_duration(row["eta_seconds"]),
        )
    table.add_section()
    table.add_row(
        Text("合计", style="bold"),
        Text("FINAL" if total["completed"] == total["total"] else "PARTIAL", style="bold green" if total["completed"] == total["total"] else "bold yellow"),
        Text(f'{total["completed"]} / {total["total"]}', style="bold"),
        Text(_percent(total["completion_rate"], 2 if total["completion_rate"] < 0.01 else 1), style="bold"),
        Text(str(total["successes"]), style="bold"),
        Text(_percent(total["success_rate"]), style=f'bold {_rate_style(total["success_rate"], total["evaluated"])}'),
        Text(str(total["errors"]), style="bold red" if total["errors"] else "bold green"),
        "—",
        Text(f'约 {format_duration(total["eta_seconds"])}', style="bold"),
    )
    return table


def _category_table(title: str, rows: list[dict], total: dict) -> Table:
    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("Category", no_wrap=True)
    table.add_column("状态", no_wrap=True)
    table.add_column("进度", justify="right", no_wrap=True)
    table.add_column("完成率", justify="right", no_wrap=True)
    table.add_column("成功", justify="right", no_wrap=True)
    table.add_column("成功率", justify="right", no_wrap=True)
    table.add_column("错误", justify="right", no_wrap=True)
    for row in rows:
        table.add_row(
            row["category_label"],
            Text("FINAL", style="green") if row["completed"] == row["total"] else Text("PARTIAL", style="yellow") if row["completed"] else Text("N/A", style="dim"),
            f'{row["completed"]} / {row["total"]}',
            _percent(row["completion_rate"], 2 if row["completion_rate"] < 0.01 else 1),
            str(row["successes"]),
            Text(_percent(row["success_rate"]), style=_rate_style(row["success_rate"], row["evaluated"])),
            Text(str(row["errors"]), style="red" if row["errors"] else "green"),
        )
    table.add_section()
    table.add_row(
        Text("合计", style="bold"),
        Text("FINAL" if total["completed"] == total["total"] else "PARTIAL", style="bold green" if total["completed"] == total["total"] else "bold yellow"),
        Text(f'{total["completed"]} / {total["total"]}', style="bold"),
        Text(_percent(total["completion_rate"], 2 if total["completion_rate"] < 0.01 else 1), style="bold"),
        Text(str(total["successes"]), style="bold"),
        Text(_percent(total["success_rate"]), style=f'bold {_rate_style(total["success_rate"], total["evaluated"])}'),
        Text(str(total["errors"]), style="bold red" if total["errors"] else "bold green"),
    )
    return table


def _metric_cell(row: dict) -> str:
    rate = _percent(row["success_rate"])
    marker = "" if row["completed"] == row["total"] else "*"
    return f'{rate}{marker} ({row["completed"]}/{row["total"]})'


def _delta(candidate: float | None, reference: float | None) -> str:
    if candidate is None or reference is None:
        return "—"
    return f"{(candidate - reference) * 100:+.1f} pp"


def _comparison_table(snapshot: dict, dimension: str) -> Table | None:
    required = PRIMARY_MODELS
    if not all(model in snapshot["models"] for model in required):
        return None
    table = Table(
        title="Three-model suite comparison" if dimension == "suite" else "Three-model category comparison",
        box=box.SIMPLE_HEAVY,
        header_style="bold white",
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("Suite" if dimension == "suite" else "Category", no_wrap=True)
    table.add_column("FP16", justify="right", no_wrap=True)
    table.add_column("QuantVLA", justify="right", no_wrap=True)
    table.add_column("QuantVLA-OPQD", justify="right", no_wrap=True)
    table.add_column("OPQD-Quant", justify="right", no_wrap=True)
    if dimension == "suite":
        benchmark = "libero-plus"
        if benchmark not in snapshot["selected_benchmarks"]:
            return None
        indexed = {
            model: {row["suite"]: row for row in snapshot["models"][model]["benchmarks"][benchmark]}
            for model in required
        }
        names = [(suite, indexed[required[0]][suite]["suite_label"]) for suite in indexed[required[0]]]
    else:
        if "libero-plus" not in snapshot["selected_benchmarks"]:
            return None
        indexed = {
            model: {
                row["category"]: row
                for row in snapshot["models"][model]["libero_plus_groups"]["categories"]
            }
            for model in required
        }
        names = [(category, indexed[required[0]][category]["category_label"]) for category in indexed[required[0]]]
    for key, label in names:
        fp16, quant, opqd = (indexed[model][key] for model in required)
        table.add_row(
            label,
            _metric_cell(fp16),
            _metric_cell(quant),
            _metric_cell(opqd),
            _delta(opqd["success_rate"], quant["success_rate"]),
        )
    return table


def _compact_cell(row: dict, *, runtime: bool = True) -> Text:
    state = row.get("runtime_state", "") if runtime else ""
    if row["completed"] == row["total"]:
        marker, style = "✓", "green"
    elif state == "running":
        marker, style = "▶", "bold cyan"
    elif row["completed"]:
        marker, style = "…", "yellow"
    else:
        marker, style = "·", "dim"
    rate = _percent(row.get("success_rate"))
    error = f' E{row["errors"]}' if row.get("errors") else ""
    return Text(
        f'{marker} {rate} {row["completed"]}/{row["total"]}{error}',
        style="red" if row.get("errors") else style,
    )


def _overview_table(snapshot: dict) -> Table:
    table = Table(title="Method overview", box=box.SIMPLE_HEAVY, pad_edge=False)
    table.add_column("Method", no_wrap=True, style="bold")
    table.add_column("Benchmark", no_wrap=True)
    table.add_column("Progress", justify="right", no_wrap=True)
    table.add_column("Success", justify="right", no_wrap=True)
    table.add_column("Errors", justify="right", no_wrap=True)
    table.add_column("Running", justify="right", no_wrap=True)
    table.add_column("ETA", justify="right", no_wrap=True)
    for model in snapshot["selected_models"]:
        data = snapshot["models"][model]
        for benchmark in snapshot["selected_benchmarks"]:
            total = data["totals"][benchmark]
            rows = data["benchmarks"][benchmark]
            running = sum(row["runtime_state"] == "running" for row in rows)
            table.add_row(
                METHOD_NAMES.get(model, model),
                "Plus" if benchmark == "libero-plus" else "LIBERO",
                f'{total["completed"]}/{total["total"]}',
                _percent(total["success_rate"]),
                str(total["errors"]),
                str(running),
                format_duration(total["eta_seconds"]),
            )
    return table


def _suite_matrix(snapshot: dict, benchmark: str) -> Table:
    table = Table(
        title=f'{BENCHMARKS[benchmark]["label"]} · suite comparison',
        box=box.SIMPLE_HEAVY,
        pad_edge=False,
    )
    table.add_column("Suite", no_wrap=True, style="bold")
    for model in snapshot["selected_models"]:
        table.add_column(METHOD_NAMES.get(model, model), justify="right", no_wrap=True)
    if all(model in snapshot["models"] for model in PRIMARY_MODELS):
        table.add_column("OPQD-Quant", justify="right", no_wrap=True)
    table.add_column("Slowest ETA", justify="right", no_wrap=True)
    indexed = {
        model: {
            row["suite"]: row
            for row in snapshot["models"][model]["benchmarks"][benchmark]
        }
        for model in snapshot["selected_models"]
    }
    first_model = snapshot["selected_models"][0]
    for suite, first_row in indexed[first_model].items():
        method_rows = [indexed[model][suite] for model in snapshot["selected_models"]]
        cells: list[object] = [first_row["suite_label"]]
        cells.extend(_compact_cell(row) for row in method_rows)
        if all(model in indexed for model in PRIMARY_MODELS):
            cells.append(
                _delta(
                    indexed[PRIMARY_MODELS[2]][suite]["success_rate"],
                    indexed[PRIMARY_MODELS[1]][suite]["success_rate"],
                )
            )
        incomplete_rows = [row for row in method_rows if row["completed"] < row["total"]]
        if not incomplete_rows:
            eta = "0s"
        elif any(row["eta_seconds"] is None for row in incomplete_rows):
            eta = "—"
        else:
            slowest = max(incomplete_rows, key=lambda row: row["eta_seconds"])
            eta = f'{METHOD_NAMES.get(slowest["model"], slowest["model"])} {format_duration(slowest["eta_seconds"])}'
        cells.append(eta)
        table.add_row(*cells)
    return table


def _suite_category_matrix(snapshot: dict) -> Table | None:
    if "libero-plus" not in snapshot["selected_benchmarks"]:
        return None
    available = [
        model
        for model in snapshot["selected_models"]
        if snapshot["models"][model]["libero_plus_groups"] is not None
    ]
    if not available:
        return None
    table = Table(
        title="LIBERO-Plus · 4 suites × 7 generalization categories",
        box=box.SIMPLE_HEAVY,
        pad_edge=False,
    )
    table.add_column("Suite", no_wrap=True, style="bold")
    table.add_column("Category", no_wrap=True)
    for model in available:
        table.add_column(METHOD_NAMES.get(model, model), justify="right", no_wrap=True)
    if all(model in available for model in PRIMARY_MODELS):
        table.add_column("OPQD-Quant", justify="right", no_wrap=True)
    indexed = {
        model: {
            (row["suite"], row["category"]): row
            for row in snapshot["models"][model]["libero_plus_groups"]["suite_categories"]
        }
        for model in available
    }
    suite_labels = {
        row["suite"]: row["suite_label"]
        for row in snapshot["models"][available[0]]["benchmarks"]["libero-plus"]
    }
    for suite_index, suite in enumerate(suite_labels):
        if suite_index:
            table.add_section()
        for category_index, category in enumerate(CATEGORY_ORDER):
            first = indexed[available[0]][suite, category]
            cells: list[object] = [
                suite_labels[suite] if category_index == 0 else "",
                first["category_label"],
            ]
            cells.extend(_compact_cell(indexed[model][suite, category], runtime=False) for model in available)
            if all(model in indexed for model in PRIMARY_MODELS):
                cells.append(
                    _delta(
                        indexed[PRIMARY_MODELS[2]][suite, category]["success_rate"],
                        indexed[PRIMARY_MODELS[1]][suite, category]["success_rate"],
                    )
                )
            table.add_row(*cells)
    return table


def build_dashboard(snapshot: dict) -> Group:
    runtime = snapshot["runtime"]
    header = Text()
    header.append("QuantVLA Eval Monitor", style="bold")
    header.append(f'  run={snapshot["run_name"]}  {snapshot["generated_at"]}\n', style="dim")
    header.append(
        f'active: servers={runtime["servers_alive"]}  evaluators={runtime["evaluators_alive"]}  '
        f'ports={runtime["ports_alive"]}/{runtime["ports_total"]}  output={snapshot["output_root"]}',
        style="cyan" if runtime["evaluators_alive"] else "dim",
    )
    blocks: list[object] = [header, _overview_table(snapshot)]
    for benchmark in snapshot["selected_benchmarks"]:
        blocks.append(_suite_matrix(snapshot, benchmark))
    category_matrix = _suite_category_matrix(snapshot)
    if category_matrix is not None:
        blocks.append(category_matrix)
    if snapshot["warnings"]:
        shown = snapshot["warnings"][:6]
        warning_text = Text(f'Data warnings ({len(snapshot["warnings"])})\n', style="bold red")
        for warning in shown:
            warning_text.append(f"- {warning}\n", style="yellow")
        if len(snapshot["warnings"]) > len(shown):
            warning_text.append(
                f'- … {len(snapshot["warnings"]) - len(shown)} more; run collect for details\n',
                style="dim",
            )
        blocks.append(warning_text)
    blocks.append(
        Text(
            "Cell = success rate + completed/expected; ✓ complete, ▶ running, … partial. "
            f'ETA uses the median of the latest {snapshot["eta_window"]} valid rollouts. '
            "Ctrl+C exits only the monitor.",
            style="dim",
        )
    )
    return Group(*blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live QuantVLA evaluation monitor")
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    parser.add_argument("--eta-window", type=int, default=10, help="Recent rollouts used for ETA")
    parser.add_argument("--manifest", type=pathlib.Path, default=None, help="LIBERO-Plus sample manifest")
    parser.add_argument(
        "--methods",
        "--models",
        dest="methods",
        nargs="+",
        choices=tuple(METHOD_TO_MODEL) + tuple(MODELS),
        default=list(METHOD_TO_MODEL),
        help="Methods to display; legacy groot-* model ids are also accepted",
    )
    parser.add_argument("--benchmarks", nargs="+", choices=tuple(BENCHMARKS), default=["libero-plus"])
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=None,
        help="Normalized output root (default: <repo>/output)",
    )
    parser.add_argument("--run-name", default="default", help="Run name below each suite")
    parser.add_argument(
        "--no-legacy-fallback",
        action="store_true",
        help="Read only normalized output/eval paths",
    )
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--no-color", action="store_true", help="Disable terminal colors")
    return parser.parse_args()


def _build_snapshot(args: argparse.Namespace, repo_root: pathlib.Path) -> dict:
    selected_models = [METHOD_TO_MODEL.get(value, value) for value in args.methods]
    return build_snapshot(
        repo_root,
        args.eta_window,
        manifest_path=args.manifest,
        selected_models=selected_models,
        selected_benchmarks=args.benchmarks,
        output_root=args.output_root,
        run_name=args.run_name,
        legacy_fallback=not args.no_legacy_fallback,
    )


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    if args.eta_window <= 0:
        raise SystemExit("--eta-window must be positive")
    repo_root = args.repo_root.resolve()
    console = Console(no_color=args.no_color)
    if args.once:
        console.print(build_dashboard(_build_snapshot(args, repo_root)))
        return
    try:
        with Live(
            build_dashboard(_build_snapshot(args, repo_root)),
            console=console,
            refresh_per_second=max(1, min(10, int(1 / min(args.interval, 1)))),
            screen=False,
        ) as live:
            while True:
                time.sleep(args.interval)
                live.update(build_dashboard(_build_snapshot(args, repo_root)), refresh=True)
    except KeyboardInterrupt:
        console.print("\n监控已退出；评测进程继续运行。", style="green")


if __name__ == "__main__":
    main()
