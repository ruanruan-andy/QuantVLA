from __future__ import annotations

import argparse
import pathlib
import time

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from eval_metrics import BENCHMARKS, MODELS, build_snapshot, format_duration


def _percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _rate_style(rate: float, completed: int) -> str:
    if completed == 0:
        return "dim"
    if rate >= 0.9:
        return "green"
    if rate >= 0.5:
        return "yellow"
    return "red"


def _benchmark_table(title: str, rows: list[dict], total: dict) -> Table:
    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("Suite", no_wrap=True)
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
            f'{row["completed"]} / {row["total"]}',
            _percent(row["completion_rate"], 2 if row["completion_rate"] < 0.01 else 1),
            str(row["successes"]),
            Text(_percent(row["success_rate"]), style=_rate_style(row["success_rate"], row["completed"])),
            Text(str(row["errors"]), style="red" if row["errors"] else "green"),
            format_duration(row["typical_seconds"]),
            format_duration(row["eta_seconds"]),
        )
    table.add_section()
    table.add_row(
        Text("合计", style="bold"),
        Text(f'{total["completed"]} / {total["total"]}', style="bold"),
        Text(_percent(total["completion_rate"], 2 if total["completion_rate"] < 0.01 else 1), style="bold"),
        Text(str(total["successes"]), style="bold"),
        Text(_percent(total["success_rate"]), style=f'bold {_rate_style(total["success_rate"], total["completed"])}'),
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
    table.add_column("进度", justify="right", no_wrap=True)
    table.add_column("完成率", justify="right", no_wrap=True)
    table.add_column("成功", justify="right", no_wrap=True)
    table.add_column("成功率", justify="right", no_wrap=True)
    table.add_column("错误", justify="right", no_wrap=True)
    for row in rows:
        table.add_row(
            row["category_label"],
            f'{row["completed"]} / {row["total"]}',
            _percent(row["completion_rate"], 2 if row["completion_rate"] < 0.01 else 1),
            str(row["successes"]),
            Text(_percent(row["success_rate"]), style=_rate_style(row["success_rate"], row["completed"])),
            Text(str(row["errors"]), style="red" if row["errors"] else "green"),
        )
    table.add_section()
    table.add_row(
        Text("合计", style="bold"),
        Text(f'{total["completed"]} / {total["total"]}', style="bold"),
        Text(_percent(total["completion_rate"], 2 if total["completion_rate"] < 0.01 else 1), style="bold"),
        Text(str(total["successes"]), style="bold"),
        Text(_percent(total["success_rate"]), style=f'bold {_rate_style(total["success_rate"], total["completed"])}'),
        Text(str(total["errors"]), style="bold red" if total["errors"] else "bold green"),
    )
    return table


def build_dashboard(snapshot: dict) -> Group:
    runtime = snapshot["runtime"]
    populated = runtime["servers_total"] > 0 and runtime["evaluators_total"] > 0
    status_ok = populated and (
        runtime["servers_alive"] == runtime["servers_total"]
        and runtime["evaluators_alive"] == runtime["evaluators_total"]
        and runtime["ports_alive"] == runtime["ports_total"]
    )
    header = Text()
    header.append("QuantVLA Eval Monitor", style="bold")
    header.append(f'  {snapshot["generated_at"]}  ')
    header.append(
        f'服务 {runtime["servers_alive"]}/{runtime["servers_total"]}  '
        f'Eval {runtime["evaluators_alive"]}/{runtime["evaluators_total"]}  '
        f'端口 {runtime["ports_alive"]}/{runtime["ports_total"]}',
        style="green" if status_ok else "yellow" if not populated else "red",
    )
    blocks: list[object] = [header]
    for model, model_config in MODELS.items():
        data = snapshot["models"][model]
        blocks.append(Text(model_config["label"], style="bold white on blue"))
        for benchmark, benchmark_config in BENCHMARKS.items():
            blocks.append(
                _benchmark_table(
                    benchmark_config["label"],
                    data["benchmarks"][benchmark],
                    data["totals"][benchmark],
                )
            )
        blocks.append(
            _category_table(
                "LIBERO-Plus · Generalization Categories",
                data["libero_plus_groups"]["categories"],
                data["totals"]["libero-plus"],
            )
        )
    blocks.append(
        Text(
            f'ETA 使用最近 {snapshot["eta_window"]} 个成功完成 rollout 的耗时中位数；'
            "并行总 ETA 取最慢 suite。Ctrl+C 只退出监控，不停止评测。",
            style="dim",
        )
    )
    return Group(*blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live QuantVLA evaluation monitor")
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    parser.add_argument("--eta-window", type=int, default=10, help="Recent rollouts used for ETA")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--no-color", action="store_true", help="Disable terminal colors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    if args.eta_window <= 0:
        raise SystemExit("--eta-window must be positive")
    repo_root = args.repo_root.resolve()
    console = Console(no_color=args.no_color)
    if args.once:
        console.print(build_dashboard(build_snapshot(repo_root, args.eta_window)))
        return
    try:
        with Live(
            build_dashboard(build_snapshot(repo_root, args.eta_window)),
            console=console,
            refresh_per_second=max(1, min(10, int(1 / min(args.interval, 1)))),
            screen=False,
        ) as live:
            while True:
                time.sleep(args.interval)
                live.update(build_dashboard(build_snapshot(repo_root, args.eta_window)), refresh=True)
    except KeyboardInterrupt:
        console.print("\n监控已退出；评测进程继续运行。", style="green")


if __name__ == "__main__":
    main()
