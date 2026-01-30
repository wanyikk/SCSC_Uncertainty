import os
import torch
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.img_util import tensor_lab2rgb
from basicsr.utils.dist_util import master_only
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.metrics.custom_fid import INCEPTION_V3_FID, get_activations, calculate_activation_statistics, \
    calculate_frechet_distance
from basicsr.utils.color_enhance import color_enhacne_blend


@MODEL_REGISTRY.register()
class ColorModelWithUncertainty(BaseModel):
    """Colorization model with uncertainty estimation"""

    def __init__(self, opt):
        super(ColorModelWithUncertainty, self).__init__(opt)

        # define network net_g with uncertainty
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # Uncertainty tracking
        self.uncertainty_threshold = opt.get('uncertainty_threshold', 0.5)
        self.uncertainty_history = []
        self.high_uncertainty_count = 0
        self.total_count = 0

        # load pretrained model for net_g
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

        # Uncertainty tracking - 修改初始阈值
        self.uncertainty_threshold = opt.get('uncertainty_threshold', 2.0)  # 从0.5改为2.0
        self.uncertainty_history = []
        self.high_uncertainty_count = 0
        self.total_count = 0

        # 添加warmup配置
        self.uncertainty_warmup_iter = opt['train'].get('uncertainty_warmup_iter', 50000)

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained model for net_d
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        if train_opt.get('colorfulness_opt'):
            self.cri_colorfulness = build_loss(train_opt['colorfulness_opt']).to(self.device)
        else:
            self.cri_colorfulness = None

        # Add uncertainty loss
        if train_opt.get('uncertainty_opt'):
            self.cri_uncertainty = build_loss(train_opt['uncertainty_opt']).to(self.device)
        else:
            self.cri_uncertainty = None

        # Add calibration loss
        if train_opt.get('calibration_opt'):
            self.cri_calibration = build_loss(train_opt['calibration_opt']).to(self.device)
        else:
            self.cri_calibration = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_uncertainty is None:
            raise ValueError('At least one loss should be defined.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        # set real dataset cache for fid metric computing
        self.real_mu, self.real_sigma = None, None
        if self.opt['val'].get('metrics') is not None and self.opt['val']['metrics'].get('fid') is not None:
            self._prepare_inception_model_fid()

        # Add uncertainty loss
        if train_opt.get('uncertainty_opt'):
            self.cri_uncertainty = build_loss(train_opt['uncertainty_opt']).to(self.device)
        else:
            self.cri_uncertainty = None

        # Add calibration loss
        if train_opt.get('calibration_opt'):
            self.cri_calibration = build_loss(train_opt['calibration_opt']).to(self.device)
        else:
            self.cri_calibration = None

        # 添加不确定性正则化损失
        if train_opt.get('uncertainty_reg_opt'):
            self.cri_uncertainty_reg = build_loss(train_opt['uncertainty_reg_opt']).to(self.device)
        else:
            self.cri_uncertainty_reg = None

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params_g = self.net_g.parameters()

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq_rgb = tensor_lab2rgb(torch.cat([self.lq, torch.zeros_like(self.lq), torch.zeros_like(self.lq)], dim=1))
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_lab = torch.cat([self.lq, self.gt], dim=1)
            self.gt_rgb = tensor_lab2rgb(self.gt_lab)

            if self.opt['train'].get('color_enhance', False):
                for i in range(self.gt_rgb.shape[0]):
                    self.gt_rgb[i] = color_enhacne_blend(self.gt_rgb[i],
                                                         factor=self.opt['train'].get('color_enhance_factor'))

    def optimize_parameters(self, current_iter):
        # 启用异常检测
        torch.autograd.set_detect_anomaly(True)

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        # Forward with uncertainty
        self.output_ab, self.uncertainty, self.ensemble_predictions = self.net_g(self.lq_rgb, return_uncertainty=True)
        self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
        self.output_rgb = tensor_lab2rgb(self.output_lab.clone()).clone()

        l_g_total = 0
        loss_dict = OrderedDict()

        # 计算warmup系数
        if current_iter < self.uncertainty_warmup_iter:
            uncertainty_scale = current_iter / self.uncertainty_warmup_iter
        else:
            uncertainty_scale = 1.0

        # Uncertainty-aware loss with warmup
        if self.cri_uncertainty:
            l_g_uncertainty, uncertainty_components = self.cri_uncertainty(
                self.output_ab, self.gt, self.uncertainty, self.ensemble_predictions
            )
            l_g_total += l_g_uncertainty * uncertainty_scale
            loss_dict['l_g_uncertainty'] = l_g_uncertainty
            for k, v in uncertainty_components.items():
                loss_dict[f'l_g_{k}'] = v
        else:
            # Standard pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output_ab, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

        # 添加不确定性正则化损失
        if self.cri_uncertainty_reg:
            l_g_uncertainty_reg = self.cri_uncertainty_reg(self.uncertainty)
            l_g_total += l_g_uncertainty_reg * uncertainty_scale
            loss_dict['l_g_uncertainty_reg'] = l_g_uncertainty_reg

        # Calibration loss with warmup
        if self.cri_calibration:
            l_g_calib = self.cri_calibration(self.output_ab, self.gt, self.uncertainty)
            l_g_total += l_g_calib * uncertainty_scale
            loss_dict['l_g_calib'] = l_g_calib

        # perceptual loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output_rgb.clone(), self.gt_rgb.clone())
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        # gan loss
        if self.cri_gan:
            fake_g_pred = self.net_d(self.output_rgb.clone())
            l_g_gan = self.cri_gan(fake_g_pred, target_is_real=True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        # colorfulness loss
        if self.cri_colorfulness:
            l_g_color = self.cri_colorfulness(self.output_rgb.clone())
            l_g_total += l_g_color
            loss_dict['l_g_color'] = l_g_color

        l_g_total.backward()
        self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()

        real_d_pred = self.net_d(self.gt_rgb.clone())
        fake_d_pred = self.net_d(self.output_rgb.detach().clone())
        l_d = self.cri_gan(real_d_pred, target_is_real=True, is_disc=True) + self.cri_gan(fake_d_pred,
                                                                                          target_is_real=False,
                                                                                          is_disc=True)
        loss_dict['l_d'] = l_d
        loss_dict['real_score'] = real_d_pred.detach().mean()
        loss_dict['fake_score'] = fake_d_pred.detach().mean()

        l_d.backward()
        self.optimizer_d.step()

        # Track uncertainty metrics - 保持为 tensor
        with torch.no_grad():
            mean_uncertainty = self.uncertainty.mean()
            max_uncertainty = self.uncertainty.max()

            # 修改高不确定性比例的计算
            high_uncertainty_ratio = (self.uncertainty.mean(dim=1) > self.uncertainty_threshold).float().mean()

            # 更保守的动态阈值调整策略
            if current_iter > self.uncertainty_warmup_iter:  # 只在warmup后调整
                if high_uncertainty_ratio.item() > 0.8 and current_iter % 1000 == 0:  # 从0.95降低到0.8，频率从100改为1000
                    # 更小的调整步长
                    self.uncertainty_threshold = min(self.uncertainty_threshold * 1.02, 2.0)  # 从1.1改为1.02，最大值从5.0改为2.0
                    logger = get_root_logger()
                    logger.info(f"High uncertainty ratio {high_uncertainty_ratio.item():.2%}. "
                                f"Gradually increasing threshold to {self.uncertainty_threshold:.3f}")

            # 添加异常检测
            if mean_uncertainty.item() > 50.0:  # 如果平均不确定性过高
                logger = get_root_logger()
                logger.warning(f"Abnormally high uncertainty detected: {mean_uncertainty.item():.2f}. "
                               f"Consider reducing uncertainty loss weights or checking initialization.")

            loss_dict['uncertainty_mean'] = mean_uncertainty
            loss_dict['uncertainty_max'] = max_uncertainty
            loss_dict['high_uncertainty_ratio'] = high_uncertainty_ratio
            loss_dict['uncertainty_threshold'] = torch.tensor(self.uncertainty_threshold, device=self.device)

            # 更新历史记录时添加异常值过滤
            if mean_uncertainty.item() < 100.0:  # 只记录合理的值
                self.uncertainty_history.append(mean_uncertainty.item())
                if len(self.uncertainty_history) > 1000:
                    self.uncertainty_history.pop(0)

            # Count high uncertainty samples
            if mean_uncertainty.item() > self.uncertainty_threshold:
                self.high_uncertainty_count += 1
            self.total_count += 1

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq_rgb.detach().cpu()
        out_dict['result'] = self.output_rgb.detach().cpu()

        # Add uncertainty visualization
        if hasattr(self, 'uncertainty'):
            # Normalize uncertainty for visualization
            uncertainty_vis = self.uncertainty.mean(dim=1, keepdim=True)  # Average over a,b channels
            uncertainty_vis = (uncertainty_vis - uncertainty_vis.min()) / (
                        uncertainty_vis.max() - uncertainty_vis.min() + 1e-8)
            uncertainty_vis = uncertainty_vis.repeat(1, 3, 1, 1)  # Convert to RGB
            out_dict['uncertainty'] = uncertainty_vis.detach().cpu()

            # Create uncertainty heatmap
            uncertainty_heatmap = self._create_uncertainty_heatmap(self.uncertainty)
            out_dict['uncertainty_heatmap'] = uncertainty_heatmap.detach().cpu()

        if self.opt['logger'].get('save_snapshot_verbose', False):
            self.output_lab_chroma = torch.cat([torch.ones_like(self.lq) * 50, self.output_ab], dim=1)
            self.output_rgb_chroma = tensor_lab2rgb(self.output_lab_chroma)
            out_dict['result_chroma'] = self.output_rgb_chroma.detach().cpu()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt_rgb.detach().cpu()
            if self.opt['logger'].get('save_snapshot_verbose', False):
                self.gt_lab_chroma = torch.cat([torch.ones_like(self.lq) * 50, self.gt], dim=1)
                self.gt_rgb_chroma = tensor_lab2rgb(self.gt_lab_chroma)
                out_dict['gt_chroma'] = self.gt_rgb_chroma.detach().cpu()
        return out_dict

    def _create_uncertainty_heatmap(self, uncertainty):
        """Create a color-coded heatmap of uncertainty"""
        # Average uncertainty across channels
        uncertainty_map = uncertainty.mean(dim=1, keepdim=True)

        # Normalize to [0, 1]
        uncertainty_map = (uncertainty_map - uncertainty_map.min()) / (
                    uncertainty_map.max() - uncertainty_map.min() + 1e-8)

        # Create heatmap (blue->green->yellow->red)
        heatmap = torch.zeros(uncertainty_map.shape[0], 3, uncertainty_map.shape[2], uncertainty_map.shape[3],
                              device=uncertainty_map.device)

        # Red channel: increases with uncertainty
        heatmap[:, 0, :, :] = uncertainty_map.squeeze(1)

        # Green channel: peaks at medium uncertainty
        heatmap[:, 1, :, :] = 1.0 - torch.abs(uncertainty_map.squeeze(1) - 0.5) * 2

        # Blue channel: decreases with uncertainty
        heatmap[:, 2, :, :] = 1.0 - uncertainty_map.squeeze(1)

        return heatmap

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output_ab, self.uncertainty, _ = self.net_g_ema(self.lq_rgb, return_uncertainty=True)
                self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
                self.output_rgb = tensor_lab2rgb(self.output_lab)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output_ab, self.uncertainty, _ = self.net_g(self.lq_rgb, return_uncertainty=True)
                self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
                self.output_rgb = tensor_lab2rgb(self.output_lab)
            self.net_g.train()

    def test_with_uncertainty_analysis(self):
        """Test with detailed uncertainty analysis"""
        self.test()

        with torch.no_grad():
            # Calculate uncertainty metrics
            mean_uncertainty = self.uncertainty.mean().item()
            std_uncertainty = self.uncertainty.std().item()
            max_uncertainty = self.uncertainty.max().item()

            # Calculate percentage of high uncertainty pixels
            high_uncertainty_mask = self.uncertainty.mean(dim=1) > self.uncertainty_threshold
            high_uncertainty_percentage = high_uncertainty_mask.float().mean().item() * 100

            # Spatial uncertainty analysis
            uncertainty_spatial = self.uncertainty.mean(dim=1).squeeze()  # [H, W]
            h, w = uncertainty_spatial.shape

            # Divide image into regions and analyze
            regions = {'top_left': uncertainty_spatial[:h // 2, :w // 2],
                       'top_right': uncertainty_spatial[:h // 2, w // 2:],
                       'bottom_left': uncertainty_spatial[h // 2:, :w // 2],
                       'bottom_right': uncertainty_spatial[h // 2:, w // 2:]}

            region_stats = {}
            for name, region in regions.items():
                region_stats[name] = {
                    'mean': region.mean().item(),
                    'std': region.std().item(),
                    'max': region.max().item()
                }

            return {
                'mean_uncertainty': mean_uncertainty,
                'std_uncertainty': std_uncertainty,
                'max_uncertainty': max_uncertainty,
                'high_uncertainty_percentage': high_uncertainty_percentage,
                'uncertainty_assessment': self._assess_uncertainty(mean_uncertainty),
                'region_stats': region_stats
            }

    def _assess_uncertainty(self, mean_uncertainty):
        """Assess uncertainty level and provide interpretation"""
        if mean_uncertainty < 0.3:
            return "Low uncertainty - High confidence in colorization"
        elif mean_uncertainty < 0.5:
            return "Medium uncertainty - Moderate confidence in colorization"
        elif mean_uncertainty < 0.7:
            return "High uncertainty - Low confidence in colorization"
        else:
            return "Very high uncertainty - Very low confidence in colorization"

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # Add uncertainty metrics
            self.metric_results['uncertainty_mean'] = 0
            self.metric_results['high_uncertainty_ratio'] = 0

        if with_metrics:
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        if self.opt['val']['metrics'].get('fid') is not None:
            fake_acts_set, acts_set = [], []

        uncertainty_values = []

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            if hasattr(self, 'gt'):
                del self.gt
            self.feed_data(val_data)

            # Test with uncertainty analysis
            uncertainty_stats = self.test_with_uncertainty_analysis()
            uncertainty_values.append(uncertainty_stats['mean_uncertainty'])

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img

            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_dir = osp.join(self.opt['path']['visualization'], img_name)
                    os.makedirs(save_dir, exist_ok=True)
                    for key in visuals:
                        save_path = os.path.join(save_dir, '{}_{}.png'.format(current_iter, key))
                        img = tensor2img(visuals[key])
                        imwrite(img, save_path)

                    # Save uncertainty stats
                    stats_path = os.path.join(save_dir, '{}_uncertainty_stats.txt'.format(current_iter))
                    with open(stats_path, 'w') as f:
                        f.write(f"Uncertainty Statistics for {img_name}:\n")
                        f.write(f"Mean Uncertainty: {uncertainty_stats['mean_uncertainty']:.4f}\n")
                        f.write(f"Std Uncertainty: {uncertainty_stats['std_uncertainty']:.4f}\n")
                        f.write(f"Max Uncertainty: {uncertainty_stats['max_uncertainty']:.4f}\n")
                        f.write(
                            f"High Uncertainty Percentage: {uncertainty_stats['high_uncertainty_percentage']:.2f}%\n")
                        f.write(f"Assessment: {uncertainty_stats['uncertainty_assessment']}\n")
                        f.write("\nRegion Statistics:\n")
                        for region, stats in uncertainty_stats['region_stats'].items():
                            f.write(
                                f"  {region}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, max={stats['max']:.4f}\n")
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                    imwrite(sr_img, save_img_path)

                    # Save uncertainty visualization
                    uncertainty_img = tensor2img(visuals['uncertainty_heatmap'])
                    uncertainty_path = save_img_path.replace('.png', '_uncertainty.png')
                    imwrite(uncertainty_img, uncertainty_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name == 'fid':
                        pred, gt = visuals['result'].cuda(), visuals['gt'].cuda()
                        fake_act = get_activations(pred, self.inception_model_fid, 1)
                        fake_acts_set.append(fake_act)
                        if self.real_mu is None:
                            real_act = get_activations(gt, self.inception_model_fid, 1)
                            acts_set.append(real_act)
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

                # Add uncertainty metrics
                self.metric_results['uncertainty_mean'] += uncertainty_stats['mean_uncertainty']
                self.metric_results['high_uncertainty_ratio'] += (
                            uncertainty_stats['mean_uncertainty'] > self.uncertainty_threshold)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name} (Uncertainty: {uncertainty_stats["mean_uncertainty"]:.3f})')

        if use_pbar:
            pbar.close()

        if with_metrics:
            if self.opt['val']['metrics'].get('fid') is not None:
                if self.real_mu is None:
                    acts_set = np.concatenate(acts_set, 0)
                    self.real_mu, self.real_sigma = calculate_activation_statistics(acts_set)
                fake_acts_set = np.concatenate(fake_acts_set, 0)
                fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)

                fid_score = calculate_frechet_distance(self.real_mu, self.real_sigma, fake_mu, fake_sigma)
                self.metric_results['fid'] = fid_score

            for metric in self.metric_results.keys():
                if metric not in ['fid', 'uncertainty_mean', 'high_uncertainty_ratio']:
                    self.metric_results[metric] /= (idx + 1)
                elif metric in ['uncertainty_mean', 'high_uncertainty_ratio']:
                    self.metric_results[metric] /= (idx + 1)

                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

            # Log uncertainty statistics
            logger = get_root_logger()
            logger.info(f'Validation Uncertainty Statistics:')
            logger.info(f'\t Average Uncertainty: {np.mean(uncertainty_values):.4f} ± {np.std(uncertainty_values):.4f}')
            logger.info(f'\t High Uncertainty Ratio: {self.metric_results["high_uncertainty_ratio"]:.2%}')

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results') and metric in self.best_metric_results[dataset_name]:
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def _prepare_inception_model_fid(self,
                                     path='/home/sunbowen/snap/DDColor-master_create01/pretrain/inception_v3_google-1a9a5a14.pth'):
        incep_state_dict = torch.load(path, map_location='cpu')
        block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model_fid = INCEPTION_V3_FID(incep_state_dict, [block_idx])
        self.inception_model_fid.cuda()
        self.inception_model_fid.eval()

    @master_only
    def save_training_images(self, current_iter):
        visuals = self.get_current_visuals()
        save_dir = osp.join(self.opt['root_path'], 'experiments', self.opt['name'], 'training_images_snapshot')
        os.makedirs(save_dir, exist_ok=True)

        for key in visuals:
            save_path = os.path.join(save_dir, '{}_{}.png'.format(current_iter, key))
            img = tensor2img(visuals[key])
            imwrite(img, save_path)

        # Save uncertainty statistics
        stats_path = os.path.join(save_dir, '{}_uncertainty_stats.txt'.format(current_iter))
        with open(stats_path, 'w') as f:
            f.write(f"Training Uncertainty Statistics at iteration {current_iter}:\n")
            f.write(f"Average uncertainty over last 1000 iterations: {np.mean(self.uncertainty_history):.4f}\n")
            f.write(f"High uncertainty ratio: {self.high_uncertainty_count / max(self.total_count, 1):.2%}\n")

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
