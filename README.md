# SCSC_Uncertainty

This project integrates two stages into an end-to-end pipeline:

1. **Semantic Communication Reconstruction (Stage 1 / Semantic)**: Input color image → Convert to grayscale (3-channel replication) → Swin-Transformer Encoding → Wireless Channel Simulation (AWGN / Rayleigh) → Swin-Transformer Decoding → Reconstructed grayscale image.
2. **Color Restoration (Stage 2 / Color + Uncertainty)**: Input reconstructed grayscale image → Color Decoder + Pixel Decoder → Output color image, and provide uncertainty (Aleatoric/Epistemic/Total Uncertainty).

Directory Structure:
- `stage1_semantic/`: Stage 1 (Semantic Communication Module).
- `stage2_color/`: Stage 2 (Color Restoration Module).
- `scsc/`: End-to-end integration code and unified entry scripts.

## End-to-End Inference

Run in the `SCSC_Uncertainty` directory:

```bash
python -m scsc.run_inference ^
  --input "\path\to\images_or_image.png" ^
  --output_dir "\path\to\out" ^
  --stage1_ckpt "SCSC_Uncertainty\stage1_semantic\Semantic_AWGN_DIV2K_random_snr_psnr_C96.model" ^
  --stage2_ckpt "\path\to\color_uncertainty.pth" ^
  --snr 10 ^
  --channel_type awgn ^
  --save_uncertainty
```

Notes:
- `--input` supports single image or folder.
- `--snr` specifies the Signal-to-Noise Ratio (dB). If not specified, it will sample randomly according to `multiple_snr` in Stage 1.
- `--save_uncertainty` will additionally output uncertainty visualization and reports: saving both "Raw Uncertainty" and "Relative Uncertainty (Quantile Normalization)" to avoid the perception bias of "always high uncertainty" caused by fixed thresholds.

## Single Stage Training/Testing Entry Points

Stage 1 (Semantic):

```bash
python -m scsc.run_stage1 --training
python -m scsc.run_stage1
```

Stage 2 (Color Training, default uncertainty configuration):

```bash
python -m scsc.run_stage2_train --opt "stage2_color/options/train/train_color_uncertainty.yml"
```

Stage 2 (Uncertainty Testing/Reporting, changed to "Relative Uncertainty" quantile scale to avoid high perception bias from fixed thresholds):

```bash
python -m scsc.run_stage2_test_uncertainty ^
  --input "\\path\\to\\test_images" ^
  --output "\\path\\to\\out_uncert"
```
