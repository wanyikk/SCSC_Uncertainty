# SCSC_Uncertainty

本项目将两个阶段整合为一个端到端流水线：

1. **语义通信重构（Stage 1 / Semantic）**：输入彩色图像 → 转灰度（3 通道复制）→ Swin-Transformer 编码 → 无线信道仿真（AWGN / Rayleigh）→ Swin-Transformer 解码 → 得到重构灰度图。
2. **颜色恢复（Stage 2 / Color + Uncertainty）**：输入重构灰度图 → 颜色解码器 + 像素解码器 → 输出彩色图像，并给出不确定性（偶然/认知/总不确定性）。

目录结构：
- `stage1_semantic/`：Stage 1（语义通信模块）。
- `stage2_color/`：Stage 2（颜色恢复模块）。
- `scsc/`：端到端集成代码与统一入口脚本。

## 端到端推理

在 `SCSC_Uncertainty` 目录下运行：

```bash
python -m scsc.run_inference ^
  --input "Y:\path\to\images_or_image.png" ^
  --output_dir "Y:\path\to\out" ^
  --stage1_ckpt "Y:\teacherguo\Semantic Communications\WITT-main_02\SCSC_Uncertainty\stage1_semantic\Semantic_AWGN_DIV2K_random_snr_psnr_C96.model" ^
  --stage2_ckpt "Y:\path\to\ddcolor_uncertainty.pth" ^
  --snr 10 ^
  --channel_type awgn ^
  --save_uncertainty
```

说明：
- `--input` 支持单张图片或文件夹。
- `--snr` 为指定信噪比（dB）。如果不指定，将按 Stage 1 的 `multiple_snr` 随机采样。
- `--save_uncertainty` 会额外输出不确定性可视化与报告：同时保存“原始不确定性”与“相对不确定性（分位归一化）”，避免阈值固定导致“总是高不确定性”的观感偏差。

## 单阶段训练/测试入口

Stage 1（Semantic）：

```bash
python -m scsc.run_stage1 --training
python -m scsc.run_stage1
```

Stage 2（Color 训练，默认不确定性版配置）：

```bash
python -m scsc.run_stage2_train --opt "stage2_color/options/train/train_color_uncertainty.yml"
```

Stage 2（不确定性测试/报告，已改为“相对不确定性”分档口径，避免固定阈值导致偏高观感）：

```bash
python -m scsc.run_stage2_test_uncertainty ^
  --input "Y:\\path\\to\\test_images" ^
  --output "Y:\\path\\to\\out_uncert"
```

