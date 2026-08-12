#!/usr/bin/env python
"""Run LIBERO-Plus eval with the OpenCV motion-blur fallback used by probe collection."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> None:
    os.environ["TANGENT_PROBE_STUB_WAND"] = "1"
    from collect_tangent_probe_observations import install_optional_wand_stub

    install_optional_wand_stub()
    target = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "LiberoPlus"
        / "eval"
        / "run_libero_plus_eval.py"
    )
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
