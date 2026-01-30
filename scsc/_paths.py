from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_stage_paths() -> None:
    root = project_root()
    stage1 = root / "stage1_semantic"
    stage2 = root / "stage2_color"

    for p in (stage1, stage2):
        p_str = os.fspath(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

