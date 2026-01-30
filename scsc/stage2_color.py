from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from scsc._paths import add_stage_paths
from scsc.color_space import lab_to_rgb01, rgb01_to_lab


@dataclass(frozen=True)
class ColorResult:
    rgb: np.ndarray
    uncertainty: Optional[dict[str, Any]] = None


class ColorStage2:
    def __init__(
        self,
        ckpt_path: str,
        input_size: int = 256,
        model_size: str = "large",
        enable_uncertainty: bool = True,
    ) -> None:
        add_stage_paths()
        import os
        from PIL import Image
        import torch

        from basicsr.archs.ddcolor_arch_uncertainty import DDColorWithUncertainty

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file not found: {ckpt_path}")

        self.input_size = int(input_size)
        self.enable_uncertainty = bool(enable_uncertainty)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_name = "convnext-t" if model_size == "tiny" else "convnext-l"

        self.model = DDColorWithUncertainty(
            encoder_name=encoder_name,
            decoder_name="MultiScaleColorDecoder",
            num_input_channels=3,
            input_size=(self.input_size, self.input_size),
            num_output_channels=2,
            last_norm="Spectral",
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
            encoder_from_pretrain=False,
            num_ensemble_heads=3,
            uncertainty_weight=0.1,
        ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self._pil = Image

    def colorize(
        self,
        gray_rgb_uint8: np.ndarray,
        return_uncertainty: bool = True,
        uncertainty_threshold: float = 0.5,
    ) -> ColorResult:
        import torch
        import torch.nn.functional as F

        if gray_rgb_uint8.dtype != np.uint8 or gray_rgb_uint8.ndim != 3 or gray_rgb_uint8.shape[2] != 3:
            raise ValueError("gray_rgb_uint8 must be uint8 RGB image with shape (H, W, 3)")

        h, w = int(gray_rgb_uint8.shape[0]), int(gray_rgb_uint8.shape[1])
        rgb01 = gray_rgb_uint8.astype(np.float32) / 255.0

        orig_lab = rgb01_to_lab(rgb01)
        orig_l = orig_lab[:, :, :1]

        pil_img = self._pil.fromarray(gray_rgb_uint8, mode="RGB")
        pil_img = pil_img.resize((self.input_size, self.input_size), resample=self._pil.BICUBIC)
        rgb01_small = (np.array(pil_img, dtype=np.uint8).astype(np.float32) / 255.0)
        lab_small = rgb01_to_lab(rgb01_small)
        l_small = lab_small[:, :, :1]

        gray_lab_small = np.concatenate([l_small, np.zeros_like(l_small), np.zeros_like(l_small)], axis=-1)
        gray_rgb01_small = lab_to_rgb01(gray_lab_small)

        inp = torch.from_numpy(gray_rgb01_small.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        inp = inp.float()

        if return_uncertainty and self.enable_uncertainty:
            output_ab, uncertainty, _ = self.model(inp)
        else:
            output_ab = self.model(inp)[0] if isinstance(self.model(inp), (tuple, list)) else self.model(inp)
            uncertainty = None

        output_ab = F.interpolate(output_ab, size=(h, w), mode="bilinear", align_corners=False)[0]
        output_ab_np = output_ab.detach().cpu().numpy().transpose(1, 2, 0)

        out_lab = np.concatenate([orig_l, output_ab_np], axis=-1)
        out_rgb01 = lab_to_rgb01(out_lab)
        out_rgb_u8 = (np.clip(out_rgb01, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        if uncertainty is None:
            return DDColorResult(rgb=out_rgb_u8, uncertainty=None)

        uncertainty = F.interpolate(uncertainty, size=(h, w), mode="bilinear", align_corners=False)[0]
        uncertainty_np = uncertainty.detach().cpu().numpy()
        total_u_map = np.mean(uncertainty_np, axis=0).astype(np.float32)
        high_mask = total_u_map > float(uncertainty_threshold)

        return DDColorResult(
            rgb=out_rgb_u8,
            uncertainty={
                "total_uncertainty": total_u_map,
                "high_uncertainty_mask": high_mask,
                "uncertainty_threshold": float(uncertainty_threshold),
            },
        )
