from __future__ import annotations

import argparse
import os
import runpy
import sys

from scsc._paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser("SCSC_Uncertainty Stage2 (Color) trainer runner")
    parser.add_argument(
        "--opt",
        type=str,
        default=os.fspath(project_root() / "stage2_color" / "options" / "train" / "train_ddcolor_uncertainty.yml"),
        help="DDColor 训练配置（YAML）路径",
    )
    args, unknown = parser.parse_known_args()

    train_py = project_root() / "stage2_color" / "basicsr" / "train.py"
    sys.argv = [os.fspath(train_py), "-opt", args.opt, *unknown]
    runpy.run_path(os.fspath(train_py), run_name="__main__")


if __name__ == "__main__":
    main()

