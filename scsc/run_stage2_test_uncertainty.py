from __future__ import annotations

import argparse
import os
import runpy
import sys

from scsc._paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser("SCSC_Uncertainty Stage2 (Color) uncertainty tester runner")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.fspath(project_root() / "stage2_color" / "pretrain" / "uncert_net_g_330000.pth"),
    )
    parser.add_argument("--input", type=str, required=True, help="测试图像文件夹")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--model_size", type=str, default="large", choices=["tiny", "large"])
    parser.add_argument("--uncertainty_threshold", type=float, default=0.5)
    parser.add_argument("--save_uncertainty_viz", action="store_true")
    args, unknown = parser.parse_known_args()

    script = project_root() / "stage2_color" / "test_with_uncertainty.py"
    sys.argv = [
        os.fspath(script),
        "--model_path",
        args.model_path,
        "--input",
        args.input,
        "--output",
        args.output,
        "--input_size",
        str(args.input_size),
        "--model_size",
        args.model_size,
        "--uncertainty_threshold",
        str(args.uncertainty_threshold),
        *(["--save_uncertainty_viz"] if args.save_uncertainty_viz else []),
        *unknown,
    ]
    runpy.run_path(os.fspath(script), run_name="__main__")


if __name__ == "__main__":
    main()

