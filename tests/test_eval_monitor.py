from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_monitor import _category_cell, _cell, _success_style  # noqa: E402


def test_success_rate_color_thresholds() -> None:
    assert _success_style(None, 0) == "dim"
    assert _success_style(0.0, 4) == "bold red"
    assert _success_style(0.499, 4) == "bold red"
    assert _success_style(0.5, 4) == "yellow"
    assert _success_style(0.799, 4) == "yellow"
    assert _success_style(0.8, 4) == "bold green"
    assert _success_style(1.0, 4) == "bold green"


def test_metric_cell_keeps_compact_progress_text() -> None:
    empty = _cell({"completed": 0, "evaluated": 0, "total": 20, "success_rate": None})
    partial = _cell({"completed": 5, "evaluated": 5, "total": 20, "success_rate": 0.4})
    complete = _cell({"completed": 20, "evaluated": 20, "total": 20, "success_rate": 0.9})

    assert empty.plain == "0/20" and str(empty.style) == "dim"
    assert partial.plain == "40.0% 5/20" and str(partial.style) == "bold red"
    assert complete.plain == "90.0%" and str(complete.style) == "bold green"


def test_category_cell_fits_compact_monitor() -> None:
    empty = _category_cell(
        {"completed": 0, "evaluated": 0, "total": 20, "success_rate": None}
    )
    partial = _category_cell(
        {"completed": 14, "evaluated": 14, "total": 20, "success_rate": 9 / 14}
    )

    assert empty.plain == "—(0)" and str(empty.style) == "dim"
    assert partial.plain == "64%(14)" and str(partial.style) == "yellow"
