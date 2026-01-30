from __future__ import annotations

import argparse
import os
import runpy
import sys

from scsc._paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser("SCSC_Uncertainty Stage1 (Semantic) runner")
    parser.add_argument("--training", action="store_true", help="训练模式；不加则为测试模式")
    parser.add_argument("--trainset", type=str, default="DIV2K", choices=["CIFAR10", "DIV2K"])
    parser.add_argument("--testset", type=str, default="kodak", choices=["kodak", "CLIC21"])
    parser.add_argument("--distortion-metric", type=str, default="MSE", choices=["MSE", "MS-SSIM"])
    parser.add_argument("--model", type=str, default="WITT", choices=["WITT", "WITT_W/O"])
    parser.add_argument("--channel-type", type=str, default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--C", type=int, default=96)
    parser.add_argument("--multiple-snr", type=str, default="1,4,7,10,13")
    args, unknown = parser.parse_known_args()

    stage1_train = project_root() / "stage1_semantic" / "train.py"
    sys.argv = [
        os.fspath(stage1_train),
        *(["--training"] if args.training else []),
        "--trainset",
        args.trainset,
        "--testset",
        args.testset,
        "--distortion-metric",
        args.distortion_metric,
        "--model",
        args.model,
        "--channel-type",
        args.channel_type,
        "--C",
        str(args.C),
        "--multiple-snr",
        args.multiple_snr,
        *unknown,
    ]
    runpy.run_path(os.fspath(stage1_train), run_name="__main__")


if __name__ == "__main__":
    main()

