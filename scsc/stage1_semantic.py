from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from scsc._paths import add_stage_paths


@dataclass(frozen=True)
class SemanticConfig:
    pass_channel: bool = True
    CUDA: bool = True
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    norm: bool = False
    downsample: int = 4
    encoder_kwargs: dict = None
    decoder_kwargs: dict = None
    logger: object = None

    @staticmethod
    def default_div2k(C: int = 96) -> "SemanticConfig":
        encoder_kwargs = dict(
            img_size=(256, 256),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 192, 256, 320],
            depths=[2, 2, 6, 2],
            num_heads=[4, 6, 8, 10],
            C=C,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(256, 256),
            embed_dims=[320, 256, 192, 128],
            depths=[2, 6, 2, 2],
            num_heads=[10, 8, 6, 4],
            C=C,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        return SemanticConfig(encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs)


class SemanticStage1:
    def __init__(
        self,
        ckpt_path: str,
        snr_list: str = "1,4,7,10,13",
        channel_type: str = "awgn",
        model_variant: str = "WITT",
        C: int = 96,
        device: Optional[torch.device] = None,
    ) -> None:
        add_stage_paths()
        from net.network import WITT

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.config = SemanticConfig.default_div2k(C=C)
        object.__setattr__(self.config, "device", device)
        object.__setattr__(self.config, "CUDA", device.type == "cuda")

        self.args = SimpleNamespace(
            multiple_snr=snr_list,
            channel_type=channel_type,
            model=model_variant,
            distortion_metric="MSE",
        )

        self.net: nn.Module = WITT(self.args, self.config).to(self.device)
        self._load_weights(ckpt_path)
        self.net.eval()

    def _load_weights(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.net.load_state_dict(state, strict=True)

    @torch.no_grad()
    def reconstruct(self, gray3_tensor: torch.Tensor, snr: Optional[int] = None):
        gray3_tensor = gray3_tensor.to(self.device)
        recon, cbr, chan_param, mse, loss_g = self.net(gray3_tensor, given_SNR=snr)
        return {
            "recon": recon.detach().cpu(),
            "cbr": float(cbr),
            "snr": int(chan_param),
            "mse": float(mse),
            "loss": float(loss_g),
        }

