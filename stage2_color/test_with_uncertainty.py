"""
Author:wanyii
time:2023/10
"""
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from basicsr.archs.ddcolor_arch_uncertainty import DDColorWithUncertainty
from basicsr.utils.img_util import tensor_lab2rgb


class ImageColorizationWithUncertainty(object):
    """Image colorization pipeline with uncertainty estimation"""

    def __init__(self, model_path, input_size=256, model_size='large', uncertainty_threshold=0.5):

        self.input_size = input_size
        self.uncertainty_threshold = uncertainty_threshold

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if model_size == 'tiny':
            self.encoder_name = 'convnext-t'
        else:
            self.encoder_name = 'convnext-l'

        self.decoder_type = "MultiScaleColorDecoder"

        # Initialize model with uncertainty
        self.model = DDColorWithUncertainty(
            encoder_name=self.encoder_name,
            decoder_name='MultiScaleColorDecoder',
            input_size=[self.input_size, self.input_size],
            num_output_channels=2,
            last_norm='Spectral',
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
            num_ensemble_heads=3,
            uncertainty_weight=0.1
        ).to(self.device)

        # Load model weights
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['params'], strict=False)
        self.model.eval()

    @torch.no_grad()
    def process(self, img, return_uncertainty=True):
        """Process image and return colorized result with uncertainty"""
        self.height, self.width = img.shape[:2]

        # Preprocess
        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        # Resize and convert to grayscale RGB
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        # Convert to tensor
        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)

        # Forward pass with uncertainty
        if return_uncertainty:
            output_ab, uncertainty, _ = self.model(tensor_gray_rgb, return_uncertainty=True)

            # Resize outputs
            output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[
                0].float().cpu().numpy().transpose(1, 2, 0)
            uncertainty_resize = F.interpolate(uncertainty, size=(self.height, self.width))[
                0].float().cpu().numpy().transpose(1, 2, 0)

            # Combine with original L channel
            output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
            output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
            output_img = (output_bgr * 255.0).round().astype(np.uint8)

            # Calculate uncertainty metrics
            uncertainty_metrics = self._calculate_uncertainty_metrics(uncertainty_resize)

            return output_img, uncertainty_resize, uncertainty_metrics
        else:
            output_ab = self.model(tensor_gray_rgb, return_uncertainty=False)
            output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[
                0].float().cpu().numpy().transpose(1, 2, 0)
            output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
            output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
            output_img = (output_bgr * 255.0).round().astype(np.uint8)

            return output_img

    def _calculate_uncertainty_metrics(self, uncertainty):
        """Calculate various uncertainty metrics"""
        uncertainty_map = np.mean(uncertainty, axis=2).astype(np.float32)
        mean_uncertainty = float(np.mean(uncertainty_map))
        std_uncertainty = float(np.std(uncertainty_map))
        max_uncertainty = float(np.max(uncertainty_map))

        relative_map = self._relative_uncertainty_map(uncertainty_map)
        relative_mean = float(np.mean(relative_map))
        relative_std = float(np.std(relative_map))
        relative_max = float(np.max(relative_map))

        high_uncertainty_mask = relative_map > self.uncertainty_threshold
        high_uncertainty_ratio = float(np.mean(high_uncertainty_mask) * 100)

        # Spatial analysis
        h, w = uncertainty.shape[:2]
        regions = {
            'top_half': uncertainty[:h // 2, :],
            'bottom_half': uncertainty[h // 2:, :],
            'left_half': uncertainty[:, :w // 2],
            'right_half': uncertainty[:, w // 2:],
            'center': uncertainty[h // 4:3 * h // 4, w // 4:3 * w // 4],
            'edges': np.concatenate([
                uncertainty[:h // 8, :].flatten(),
                uncertainty[-h // 8:, :].flatten(),
                uncertainty[:, :w // 8].flatten(),
                uncertainty[:, -w // 8:].flatten()
            ])
        }

        region_stats = {}
        for name, region in regions.items():
            if name == 'edges':
                region_stats[name] = {
                    'mean': np.mean(region),
                    'std': np.std(region)
                }
            else:
                region_stats[name] = {
                    'mean': np.mean(region),
                    'std': np.std(region)
                }

        return {
            'mean': mean_uncertainty,
            'std': std_uncertainty,
            'max': max_uncertainty,
            'relative_mean': relative_mean,
            'relative_std': relative_std,
            'relative_max': relative_max,
            'high_uncertainty_ratio': high_uncertainty_ratio,
            'region_stats': region_stats,
            'assessment': self._assess_uncertainty(relative_mean)
        }

    def _assess_uncertainty(self, mean_uncertainty):
        """不确定性口径说明：这里使用“相对不确定性”（分位归一化后的均值）进行分档。"""
        if mean_uncertainty < 0.35:
            return "相对不确定性较低（相对高置信）"
        elif mean_uncertainty < 0.55:
            return "相对不确定性中等（相对中等置信）"
        elif mean_uncertainty < 0.75:
            return "相对不确定性较高（相对低置信）"
        else:
            return "相对不确定性很高（相对很低置信）"

    def _relative_uncertainty_map(self, uncertainty_map, p_low=5.0, p_high=95.0, gamma=2.2, eps=1e-6):
        lo = float(np.percentile(uncertainty_map, p_low))
        hi = float(np.percentile(uncertainty_map, p_high))
        if hi - lo < eps:
            rel = np.zeros_like(uncertainty_map, dtype=np.float32)
        else:
            rel = np.clip((uncertainty_map - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)
        return np.power(rel, gamma).astype(np.float32)

    def create_uncertainty_visualization(self, uncertainty, output_img):
        """Create comprehensive uncertainty visualization"""
        # Average uncertainty across channels
        uncertainty_map = np.mean(uncertainty, axis=2)

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Uncertainty heatmap
        rel_map = self._relative_uncertainty_map(uncertainty_map)
        im1 = axes[0, 0].imshow(rel_map, cmap='hot', vmin=0, vmax=1)
        axes[0, 0].set_title('Relative Uncertainty Heatmap')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

        # 2. High uncertainty regions
        high_uncertainty_mask = rel_map > self.uncertainty_threshold
        axes[0, 1].imshow(high_uncertainty_mask, cmap='RdYlGn_r')
        axes[0, 1].set_title(f'High Uncertainty Regions (>{self.uncertainty_threshold})')
        axes[0, 1].axis('off')

        # 3. Uncertainty overlay on colorized image
        overlay = output_img.copy()
        uncertainty_overlay = (rel_map * 255).astype(np.uint8)
        uncertainty_colored = cv2.applyColorMap(uncertainty_overlay, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 0.7, uncertainty_colored, 0.3, 0)
        axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Uncertainty Overlay')
        axes[0, 2].axis('off')

        # 4. Uncertainty histogram
        axes[1, 0].hist(rel_map.flatten(), bins=50, density=True, alpha=0.7, color='blue')
        axes[1, 0].axvline(self.uncertainty_threshold, color='red', linestyle='--',
                           label=f'Threshold={self.uncertainty_threshold}')
        axes[1, 0].set_xlabel('Uncertainty Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Uncertainty Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Channel-wise uncertainty
        for i, channel in enumerate(['A channel', 'B channel']):
            axes[1, 1].hist(uncertainty[:, :, i].flatten(), bins=30, alpha=0.5, label=channel, density=True)
        axes[1, 1].set_xlabel('Uncertainty Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Channel-wise Uncertainty')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Spatial uncertainty profile
        h_profile = np.mean(rel_map, axis=1)
        v_profile = np.mean(rel_map, axis=0)
        axes[1, 2].plot(h_profile, label='Horizontal', alpha=0.7)
        axes[1, 2].plot(v_profile, label='Vertical', alpha=0.7)
        axes[1, 2].set_xlabel('Position')
        axes[1, 2].set_ylabel('Mean Uncertainty')
        axes[1, 2].set_title('Spatial Uncertainty Profile')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model with uncertainty')
    parser.add_argument('--input', type=str, default='./assets/test_images/', help='Input test image folder')
    parser.add_argument('--output', type=str, default='./results_uncertainty/', help='Output folder')
    parser.add_argument('--input_size', type=int, default=512, help='Input size for model')
    parser.add_argument('--model_size', type=str, default='large', help='Model size (tiny or large)')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.5, help='Threshold for high uncertainty')
    parser.add_argument('--save_uncertainty_viz', action='store_true', help='Save uncertainty visualizations')

    args = parser.parse_args()

    print(f'Output path: {args.output}')
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'uncertainty_analysis'), exist_ok=True)

    img_list = os.listdir(args.input)
    assert len(img_list) > 0, "No images found in input directory"

    # Initialize colorizer with uncertainty
    colorizer = ImageColorizationWithUncertainty(
        model_path=args.model_path,
        input_size=args.input_size,
        model_size=args.model_size,
        uncertainty_threshold=args.uncertainty_threshold
    )

    # Summary statistics
    all_metrics = []

    for name in tqdm(img_list):
        if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Read image
        img_path = os.path.join(args.input, name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to read {name}, skipping...")
            continue

        # Process with uncertainty
        output_img, uncertainty, metrics = colorizer.process(img, return_uncertainty=True)
        all_metrics.append(metrics)

        # Save colorized image
        output_path = os.path.join(args.output, name)
        cv2.imwrite(output_path, output_img)

        # Save uncertainty map
        uncertainty_avg = np.mean(uncertainty, axis=2)
        uncertainty_vis = (uncertainty_avg * 255).astype(np.uint8)
        uncertainty_colored = cv2.applyColorMap(uncertainty_vis, cv2.COLORMAP_JET)
        uncertainty_path = os.path.join(args.output, 'uncertainty_analysis', f'{name[:-4]}_uncertainty.png')
        cv2.imwrite(uncertainty_path, uncertainty_colored)

        # Save detailed uncertainty visualization
        if args.save_uncertainty_viz:
            fig = colorizer.create_uncertainty_visualization(uncertainty, output_img)
            viz_path = os.path.join(args.output, 'uncertainty_analysis', f'{name[:-4]}_uncertainty_analysis.png')
            fig.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Save uncertainty report
        report_path = os.path.join(args.output, 'uncertainty_analysis', f'{name[:-4]}_uncertainty_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Uncertainty Analysis Report for {name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Metrics:\n")
            f.write(f"  Raw Mean Uncertainty: {metrics['mean']:.6f}\n")
            f.write(f"  Raw Std Uncertainty: {metrics['std']:.6f}\n")
            f.write(f"  Raw Max Uncertainty: {metrics['max']:.6f}\n")
            f.write(f"  Relative Mean Uncertainty: {metrics['relative_mean']:.6f}\n")
            f.write(f"  Relative Std Uncertainty: {metrics['relative_std']:.6f}\n")
            f.write(f"  Relative Max Uncertainty: {metrics['relative_max']:.6f}\n")
            f.write(f"  High Uncertainty Ratio (relative > {args.uncertainty_threshold}): {metrics['high_uncertainty_ratio']:.2f}%\n")
            f.write(f"  Assessment: {metrics['assessment']}\n\n")

            f.write("Regional Analysis:\n")
            for region, stats in metrics['region_stats'].items():
                f.write(f"  {region}:\n")
                f.write(f"    Mean: {stats['mean']:.4f}\n")
                f.write(f"    Std: {stats['std']:.4f}\n")

            f.write("\nInterpretation:\n")
            if metrics['relative_mean'] > args.uncertainty_threshold:
                f.write("  - 相对不确定性偏高：建议人工复核或调整模型/数据分布。\n")
            else:
                f.write("  - 相对不确定性偏低：整体置信度表现良好。\n")

            # Regional insights
            edge_uncertainty = metrics['region_stats']['edges']['mean']
            center_uncertainty = metrics['region_stats']['center']['mean']
            if edge_uncertainty > center_uncertainty * 1.5:
                f.write("  - Higher uncertainty at edges suggests difficulty with boundaries.\n")
            elif center_uncertainty > edge_uncertainty * 1.5:
                f.write("  - Higher uncertainty in center suggests complex main subject.\n")

    # Save summary report
    summary_path = os.path.join(args.output, 'uncertainty_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Uncertainty Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total images processed: {len(all_metrics)}\n\n")

        # Calculate summary statistics
        mean_uncertainties = [m['mean'] for m in all_metrics]
        relative_mean_uncertainties = [m['relative_mean'] for m in all_metrics]
        high_uncertainty_images = sum(1 for m in all_metrics if m['relative_mean'] > args.uncertainty_threshold)

        f.write(f"Overall Statistics:\n")
        f.write(f"  Average raw uncertainty: {np.mean(mean_uncertainties):.6f} ± {np.std(mean_uncertainties):.6f}\n")
        f.write(f"  Average relative uncertainty: {np.mean(relative_mean_uncertainties):.6f} ± {np.std(relative_mean_uncertainties):.6f}\n")
        f.write(f"  Min raw uncertainty: {np.min(mean_uncertainties):.6f}\n")
        f.write(f"  Max raw uncertainty: {np.max(mean_uncertainties):.6f}\n")
        f.write(
            f"  High uncertainty images: {high_uncertainty_images} ({high_uncertainty_images / len(all_metrics) * 100:.1f}%)\n\n")

        # Sort images by uncertainty
        sorted_indices = np.argsort(relative_mean_uncertainties)[::-1]

        f.write("Top 10 highest uncertainty images:\n")
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            f.write(f"  {i + 1}. {img_list[idx]} - Relative Uncertainty: {relative_mean_uncertainties[idx]:.6f}\n")

        f.write("\nTop 10 lowest uncertainty images:\n")
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[-(i + 1)]
            f.write(f"  {i + 1}. {img_list[idx]} - Relative Uncertainty: {relative_mean_uncertainties[idx]:.6f}\n")

    print(f"\nProcessing complete!")
    print(f"Results saved to: {args.output}")
    print(f"Uncertainty analysis saved to: {os.path.join(args.output, 'uncertainty_analysis')}")
    print(f"Summary report: {summary_path}")


if __name__ == '__main__':
    main()
