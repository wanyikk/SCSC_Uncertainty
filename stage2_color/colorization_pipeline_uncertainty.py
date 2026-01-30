"""
Author:wanyii
time:2023/10
"""
# !/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from basicsr.archs.ddcolor_arch_uncertainty import DDColorUncertainty
except ImportError:
    print("Warning: Could not import DDColorUncertainty")
    DDColorUncertainty = None


class ImageColorizationPipelineUncertainty(object):

    def __init__(self, model_path, input_size=256, model_size='large', enable_uncertainty=True):

        self.input_size = input_size
        self.enable_uncertainty = enable_uncertainty

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if model_size == 'tiny':
            self.encoder_name = 'convnext-t'
        else:
            self.encoder_name = 'convnext-l'

        self.decoder_type = "MultiScaleColorDecoder"

        # 创建模型
        if DDColorUncertainty is None:
            raise ImportError("DDColorUncertainty not available. Please check installation.")

        if self.decoder_type == 'MultiScaleColorDecoder':
            self.model = DDColorUncertainty(
                encoder_name=self.encoder_name,
                decoder_name='MultiScaleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
                enable_uncertainty=self.enable_uncertainty,
                uncertainty_mode='evidential'
            ).to(self.device)
        else:
            self.model = DDColorUncertainty(
                encoder_name=self.encoder_name,
                decoder_name='SingleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=256,
                enable_uncertainty=self.enable_uncertainty,
                uncertainty_mode='evidential'
            ).to(self.device)

        # 加载模型权重
        self._load_model_weights(model_path)
        self.model.eval()

    def _load_model_weights(self, model_path):
        """加载模型权重，支持兼容性检查"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
            else:
                state_dict = checkpoint

            # 获取模型状态字典
            model_dict = self.model.state_dict()

            # 检查兼容性并加载权重
            if self.enable_uncertainty:
                # 不确定性模式：只加载兼容的权重
                compatible_dict = {}
                incompatible_keys = []

                for k, v in state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        compatible_dict[k] = v
                    else:
                        incompatible_keys.append(k)

                print(f"Loading {len(compatible_dict)}/{len(state_dict)} compatible parameters")
                if incompatible_keys:
                    print(f"Skipping {len(incompatible_keys)} incompatible parameters")

                # 更新模型字典
                model_dict.update(compatible_dict)
                self.model.load_state_dict(model_dict, strict=False)
            else:
                # 普通模式：直接加载
                self.model.load_state_dict(state_dict, strict=False)

            print(f"Model loaded successfully from {model_path}")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Model will be initialized with random weights")

    @torch.no_grad()
    def process(self, img, return_uncertainty=None, uncertainty_threshold=0.5):
        """
        处理图像并返回彩色化结果

        Args:
            img: 输入灰度图像
            return_uncertainty: 是否返回不确定性信息
            uncertainty_threshold: 不确定性阈值，用于标记高不确定性区域
        """
        if return_uncertainty is None:
            return_uncertainty = self.enable_uncertainty

        self.height, self.width = img.shape[:2]

        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        # resize rgb image -> lab -> get grey -> rgb
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)

        if return_uncertainty and self.enable_uncertainty:
            # 获取不确定性信息
            model_output = self.model(tensor_gray_rgb, return_uncertainty=True)
            output_ab = model_output['prediction'].cpu()
            aleatoric_uncertainty = model_output['aleatoric_uncertainty'].cpu()
            epistemic_uncertainty = model_output['epistemic_uncertainty'].cpu()
            total_uncertainty = model_output['total_uncertainty'].cpu()
        else:
            # 只获取预测结果
            output_ab = self.model(tensor_gray_rgb, return_uncertainty=False).cpu()

        # resize ab -> concat original l -> rgb
        output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float().numpy().transpose(1, 2,
                                                                                                                 0)
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        output_img = (output_bgr * 255.0).round().astype(np.uint8)

        if return_uncertainty and self.enable_uncertainty:
            # 调整不确定性图的尺寸
            aleatoric_uncertainty_resize = F.interpolate(aleatoric_uncertainty, size=(self.height, self.width))[
                0].numpy()
            epistemic_uncertainty_resize = F.interpolate(epistemic_uncertainty, size=(self.height, self.width))[
                0].numpy()
            total_uncertainty_resize = F.interpolate(total_uncertainty, size=(self.height, self.width))[0].numpy()

            # 计算平均不确定性（跨通道）
            aleatoric_uncertainty_map = np.mean(aleatoric_uncertainty_resize, axis=0)
            epistemic_uncertainty_map = np.mean(epistemic_uncertainty_resize, axis=0)
            total_uncertainty_map = np.mean(total_uncertainty_resize, axis=0)

            # 创建高不确定性掩码
            high_uncertainty_mask = total_uncertainty_map > uncertainty_threshold

            return {
                'colorized_image': output_img,
                'aleatoric_uncertainty': aleatoric_uncertainty_map,
                'epistemic_uncertainty': epistemic_uncertainty_map,
                'total_uncertainty': total_uncertainty_map,
                'high_uncertainty_mask': high_uncertainty_mask,
                'uncertainty_stats': {
                    'mean_aleatoric': float(np.mean(aleatoric_uncertainty_map)),
                    'mean_epistemic': float(np.mean(epistemic_uncertainty_map)),
                    'mean_total': float(np.mean(total_uncertainty_map)),
                    'max_uncertainty': float(np.max(total_uncertainty_map)),
                    'high_uncertainty_ratio': float(np.sum(high_uncertainty_mask) / high_uncertainty_mask.size)
                }
            }
        else:
            return output_img

    def save_uncertainty_visualization(self, uncertainty_data, save_path, title="Uncertainty Analysis"):
        """保存不确定性可视化图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)

        # 原始彩色化结果
        axes[0, 0].imshow(cv2.cvtColor(uncertainty_data['colorized_image'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Colorized Image')
        axes[0, 0].axis('off')

        # 偶然不确定性
        im1 = axes[0, 1].imshow(uncertainty_data['aleatoric_uncertainty'], cmap='hot')
        axes[0, 1].set_title(
            f'Aleatoric Uncertainty\n(Mean: {uncertainty_data["uncertainty_stats"]["mean_aleatoric"]:.3f})')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 认知不确定性
        im2 = axes[0, 2].imshow(uncertainty_data['epistemic_uncertainty'], cmap='hot')
        axes[0, 2].set_title(
            f'Epistemic Uncertainty\n(Mean: {uncertainty_data["uncertainty_stats"]["mean_epistemic"]:.3f})')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # 总不确定性
        im3 = axes[1, 0].imshow(uncertainty_data['total_uncertainty'], cmap='hot')
        axes[1, 0].set_title(f'Total Uncertainty\n(Mean: {uncertainty_data["uncertainty_stats"]["mean_total"]:.3f})')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 高不确定性掩码
        axes[1, 1].imshow(uncertainty_data['high_uncertainty_mask'], cmap='binary')
        axes[1, 1].set_title(
            f'High Uncertainty Mask\n(Ratio: {uncertainty_data["uncertainty_stats"]["high_uncertainty_ratio"]:.1%})')
        axes[1, 1].axis('off')

        # 不确定性统计
        stats_text = f"""Uncertainty Statistics:

    Max Uncertainty: {uncertainty_data["uncertainty_stats"]["max_uncertainty"]:.3f}
    Mean Aleatoric: {uncertainty_data["uncertainty_stats"]["mean_aleatoric"]:.3f}
    Mean Epistemic: {uncertainty_data["uncertainty_stats"]["mean_epistemic"]:.3f}
    Mean Total: {uncertainty_data["uncertainty_stats"]["mean_total"]:.3f}
    High Uncertainty Ratio: {uncertainty_data["uncertainty_stats"]["high_uncertainty_ratio"]:.1%}
            """
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_confidence_overlay(self, uncertainty_data, alpha=0.3):
        """创建带有置信度覆盖的图像"""
        colorized = uncertainty_data['colorized_image'].copy()
        uncertainty_map = uncertainty_data['total_uncertainty']

        # 归一化不确定性到0-1
        uncertainty_norm = (uncertainty_map - uncertainty_map.min())  # !/usr/bin/env python3

        # 创建热图覆盖
        uncertainty_colored = cm.hot(uncertainty_norm)[:, :, :3] * 255
        uncertainty_colored = uncertainty_colored.astype(np.uint8)

        # 混合原图和不确定性图
        overlay = cv2.addWeighted(colorized, 1 - alpha, uncertainty_colored, alpha, 0)

        return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./pretrain/ddcolor_modelscope.pth')
    parser.add_argument('--input', type=str, default='./assets/test_images/',
                        help='input test image folder or video path')
    parser.add_argument('--output', type=str, default='./results/', help='output folder or video path')
    parser.add_argument('--input_size', type=int, default=512, help='input size for model')
    parser.add_argument('--model_size', type=str, default='large', help='ddcolor model size')
    parser.add_argument('--enable_uncertainty', action='store_true', help='enable uncertainty quantification')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.5, help='uncertainty threshold for masking')
    parser.add_argument('--save_uncertainty_vis', action='store_true', help='save uncertainty visualization')
    args = parser.parse_args()

    print(f'Output path: {args.output}')
    os.makedirs(args.output, exist_ok=True)

    if args.save_uncertainty_vis:
        uncertainty_output_dir = os.path.join(args.output, 'uncertainty_analysis')
        os.makedirs(uncertainty_output_dir, exist_ok=True)

    img_list = os.listdir(args.input)
    assert len(img_list) > 0

    colorizer = ImageColorizationPipelineUncertainty(
        model_path=args.model_path,
        input_size=args.input_size,
        model_size=args.model_size,
        enable_uncertainty=args.enable_uncertainty
    )

    for name in tqdm(img_list):
        img = cv2.imread(os.path.join(args.input, name))

        if args.enable_uncertainty:
            result = colorizer.process(img, return_uncertainty=True, uncertainty_threshold=args.uncertainty_threshold)

            # 保存彩色化结果
            cv2.imwrite(os.path.join(args.output, name), result['colorized_image'])

            # 打印不确定性统计
            stats = result['uncertainty_stats']
            print(f"{name} - Uncertainty Stats:")
            print(f"  Mean Total Uncertainty: {stats['mean_total']:.3f}")
            print(f"  High Uncertainty Ratio: {stats['high_uncertainty_ratio']:.1%}")

            if args.save_uncertainty_vis:
                # 保存不确定性可视化
                base_name = os.path.splitext(name)[0]
                uncertainty_vis_path = os.path.join(uncertainty_output_dir, f"{base_name}_uncertainty_analysis.png")
                colorizer.save_uncertainty_visualization(result, uncertainty_vis_path, f"Uncertainty Analysis - {name}")

                # 保存置信度覆盖图
                confidence_overlay = colorizer.create_confidence_overlay(result)
                confidence_path = os.path.join(uncertainty_output_dir, f"{base_name}_confidence_overlay.png")
                cv2.imwrite(confidence_path, confidence_overlay)

                # 保存各种不确定性图
                uncertainty_maps = {
                    'aleatoric': result['aleatoric_uncertainty'],
                    'epistemic': result['epistemic_uncertainty'],
                    'total': result['total_uncertainty']
                }

                for uncertainty_type, uncertainty_map in uncertainty_maps.items():
                    # 保存为热图
                    plt.figure(figsize=(8, 8))
                    plt.imshow(uncertainty_map, cmap='hot')
                    plt.colorbar(label='Uncertainty')
                    plt.title(f'{uncertainty_type.capitalize()} Uncertainty - {name}')
                    plt.axis('off')
                    uncertainty_heatmap_path = os.path.join(uncertainty_output_dir,
                                                            f"{base_name}_{uncertainty_type}_uncertainty.png")
                    plt.savefig(uncertainty_heatmap_path, bbox_inches='tight', dpi=150)
                    plt.close()
        else:
            # 普通模式
            image_out = colorizer.process(img, return_uncertainty=False)
            cv2.imwrite(os.path.join(args.output, name), image_out)

    print("Processing completed!")
    if args.enable_uncertainty and args.save_uncertainty_vis:
        print(f"Uncertainty visualizations saved to: {uncertainty_output_dir}")



if __name__ == '__main__':
