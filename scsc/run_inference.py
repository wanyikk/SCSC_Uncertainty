from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from scsc.stage1_semantic import SemanticStage1
from scsc.stage2_color import ColorStage2
from scsc.image_io import iter_images, read_rgb_uint8, resize_rgb_uint8, save_rgb_uint8
from scsc.colormap import jet_colormap
from scsc.uncertainty import assess_relative, normalize_percentile, summarize_map


def _rgb_to_gray3_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)
    gray3 = np.stack([gray, gray, gray], axis=-1)
    t = torch.from_numpy(gray3.transpose(2, 0, 1)).unsqueeze(0)
    return t


def _tensor_to_rgb_uint8(rgb_tensor_1chw: torch.Tensor) -> np.ndarray:
    rgb = rgb_tensor_1chw.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()
    return (rgb * 255.0).round().astype(np.uint8)


def _save_uncertainty(
    out_dir: Path,
    name_stem: str,
    uncertainty_data: dict,
    p_low: float,
    p_high: float,
    gamma: float,
) -> None:
    total = uncertainty_data["total_uncertainty"].astype(np.float32)
    raw_stats = summarize_map(total)

    total_rel = normalize_percentile(total, p_low=p_low, p_high=p_high, gamma=gamma)
    rel_stats = summarize_map(total_rel)
    rel_assess = assess_relative(rel_stats.mean)

    raw_vis = np.clip(total / (raw_stats.p95 + 1e-6), 0.0, 1.0)
    raw_col = jet_colormap(raw_vis)

    rel_col = jet_colormap(total_rel)

    save_rgb_uint8(raw_col, out_dir / f"{name_stem}_uncertainty_raw.png")
    save_rgb_uint8(rel_col, out_dir / f"{name_stem}_uncertainty_relative.png")

    report_path = out_dir / f"{name_stem}_uncertainty_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("不确定性报告（SCSC_Uncertainty）\n")
        f.write("\n[原始总不确定性：模型输出的数值尺度]\n")
        f.write(f"  mean={raw_stats.mean:.6f}, std={raw_stats.std:.6f}, max={raw_stats.max:.6f}\n")
        f.write(f"  p50={raw_stats.p50:.6f}, p90={raw_stats.p90:.6f}, p95={raw_stats.p95:.6f}\n")
        f.write("\n[相对总不确定性：分位归一化 + gamma 显示压缩]\n")
        f.write(f"  normalize: p_low={p_low:.1f}, p_high={p_high:.1f}, gamma={gamma:.2f}\n")
        f.write(f"  mean={rel_stats.mean:.6f}, std={rel_stats.std:.6f}, max={rel_stats.max:.6f}\n")
        f.write(f"  p50={rel_stats.p50:.6f}, p90={rel_stats.p90:.6f}, p95={rel_stats.p95:.6f}\n")
        f.write(f"  assessment={rel_assess}\n")


def main() -> None:
    parser = argparse.ArgumentParser("SCSC_Uncertainty end-to-end inference")
    parser.add_argument("--input", type=str, required=True, help="单张图片路径或文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--stage1_ckpt", type=str, required=True, help="Stage1 语义通信权重路径")
    parser.add_argument("--stage2_ckpt", type=str, required=True, help="Stage2 颜色恢复权重路径")
    parser.add_argument("--snr", type=int, default=None, help="指定 SNR（dB），不填则随机采样")
    parser.add_argument("--channel_type", type=str, default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--stage1_C", type=int, default=96)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--stage2_model_size", type=str, default="large", choices=["tiny", "large"])
    parser.add_argument("--save_uncertainty", action="store_true")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.5)
    parser.add_argument("--uncertainty_p_low", type=float, default=5.0)
    parser.add_argument("--uncertainty_p_high", type=float, default=95.0)
    parser.add_argument("--uncertainty_gamma", type=float, default=2.2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "stage1_recon").mkdir(parents=True, exist_ok=True)
    (out_dir / "stage2_color").mkdir(parents=True, exist_ok=True)
    if args.save_uncertainty:
        (out_dir / "uncertainty").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stage1 = SemanticStage1(
        ckpt_path=args.stage1_ckpt,
        snr_list="1,4,7,10,13",
        channel_type=args.channel_type,
        C=args.stage1_C,
        device=device,
    )
    stage2 = ColorStage2(
        ckpt_path=args.stage2_ckpt,
        input_size=args.input_size,
        model_size=args.stage2_model_size,
        enable_uncertainty=args.save_uncertainty,
    )

    for img_path in iter_images(args.input):
        rgb = read_rgb_uint8(img_path)
        rgb = resize_rgb_uint8(rgb, args.input_size)

        gray3_tensor = _rgb_to_gray3_tensor(rgb)
        recon_info = stage1.reconstruct(gray3_tensor, snr=args.snr)
        recon_rgb = _tensor_to_rgb_uint8(recon_info["recon"])

        name_stem = img_path.stem
        save_rgb_uint8(recon_rgb, out_dir / "stage1_recon" / f"{name_stem}_recon_gray.png")

        color_res = stage2.colorize(
            recon_rgb,
            return_uncertainty=True,
            uncertainty_threshold=args.uncertainty_threshold,
        )
        save_rgb_uint8(color_res.rgb, out_dir / "stage2_color" / f"{name_stem}_color.png")

        if args.save_uncertainty and color_res.uncertainty is not None:
            _save_uncertainty(
                out_dir=out_dir / "uncertainty",
                name_stem=name_stem,
                uncertainty_data=color_res.uncertainty,
                p_low=args.uncertainty_p_low,
                p_high=args.uncertainty_p_high,
                gamma=args.uncertainty_gamma,
            )


if __name__ == "__main__":
    main()
