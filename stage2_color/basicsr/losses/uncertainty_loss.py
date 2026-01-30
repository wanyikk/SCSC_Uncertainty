import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import weighted_loss


@LOSS_REGISTRY.register()
class UncertaintyAwareLoss(nn.Module):
    """Loss function that incorporates uncertainty estimation"""

    def __init__(self, base_loss_weight=1.0, uncertainty_weight=0.1, nll_weight=0.1, diversity_weight=0.1):
        super().__init__()
        self.base_loss_weight = base_loss_weight
        self.uncertainty_weight = uncertainty_weight
        self.nll_weight = nll_weight
        self.diversity_weight = diversity_weight

    def forward(self, predictions, target, uncertainty, ensemble_predictions=None):
        # Base reconstruction loss
        base_loss = F.l1_loss(predictions, target)

        # 添加不确定性范围检查和修正
        uncertainty = torch.clamp(uncertainty, min=1e-6, max=10.0)  # 限制不确定性范围

        # Negative log-likelihood loss with uncertainty
        eps = 1e-6
        diff_squared = (predictions - target) ** 2

        # 修改NLL计算，添加稳定性项
        nll_loss = 0.5 * (torch.log(uncertainty + eps) + diff_squared / (uncertainty + eps))

        # 添加异常值处理
        nll_loss = torch.where(torch.isnan(nll_loss) | torch.isinf(nll_loss),
                               torch.zeros_like(nll_loss), nll_loss)
        nll_loss = nll_loss.mean()

        # 修改不确定性正则化，使其更温和
        log_uncertainty = torch.log(uncertainty + eps)
        uncertainty_reg = torch.mean(log_uncertainty ** 2) * 0.1  # 添加缩放因子

        # Total loss
        total_loss = (self.base_loss_weight * base_loss +
                      self.nll_weight * nll_loss +
                      self.uncertainty_weight * uncertainty_reg)

        # Ensemble diversity loss with gradient clipping
        diversity_loss = 0
        if ensemble_predictions is not None and self.diversity_weight > 0:
            num_heads = ensemble_predictions.shape[0]
            if num_heads > 1:
                for i in range(num_heads):
                    for j in range(i + 1, num_heads):
                        pred_i = ensemble_predictions[i].flatten(start_dim=1)
                        pred_j = ensemble_predictions[j].flatten(start_dim=1)
                        cosine_sim = F.cosine_similarity(pred_i, pred_j, dim=1).mean()
                        diversity_loss -= cosine_sim

                diversity_loss = diversity_loss / (num_heads * (num_heads - 1) / 2)
                # 限制diversity_loss的范围
                diversity_loss = torch.clamp(diversity_loss, min=-2.0, max=2.0)
                total_loss += self.diversity_weight * diversity_loss

        loss_components = {
            'base_loss': base_loss,
            'nll_loss': nll_loss,
            'uncertainty_reg': uncertainty_reg,
            'diversity_loss': diversity_loss if ensemble_predictions is not None else torch.tensor(0.0)
        }

        return total_loss, loss_components


@LOSS_REGISTRY.register()
class CalibrationLoss(nn.Module):
    """Loss for uncertainty calibration"""

    def __init__(self, num_bins=10, loss_weight=0.1, temperature=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, predictions, target, uncertainty):
        """
        Expected Calibration Error (ECE) based loss
        """
        # Calculate prediction errors
        errors = torch.abs(predictions - target).mean(dim=1, keepdim=True)  # [B, 1, H, W]
        uncertainties = uncertainty.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Flatten
        errors = errors.flatten()
        uncertainties = uncertainties.flatten()

        # Sort by uncertainty
        sorted_indices = torch.argsort(uncertainties)
        sorted_errors = errors[sorted_indices]
        sorted_uncertainties = uncertainties[sorted_indices]

        # Compute calibration error
        calibration_error = 0
        bin_size = len(sorted_errors) // self.num_bins

        for i in range(self.num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < self.num_bins - 1 else len(sorted_errors)

            if end_idx > start_idx:
                bin_errors = sorted_errors[start_idx:end_idx]
                bin_uncertainties = sorted_uncertainties[start_idx:end_idx]

                # Mean error should match mean uncertainty
                bin_mean_error = bin_errors.mean()
                bin_mean_uncertainty = bin_uncertainties.mean()

                # Calibration error for this bin
                calibration_error += torch.abs(bin_mean_error - bin_mean_uncertainty * self.temperature)

        calibration_error = calibration_error / self.num_bins

        return self.loss_weight * calibration_error


@LOSS_REGISTRY.register()
class AdaptiveUncertaintyLoss(nn.Module):
    """Adaptive loss that weights samples based on uncertainty"""

    def __init__(self, base_loss_type='l1', uncertainty_mode='adaptive', loss_weight=1.0):
        super().__init__()
        self.base_loss_type = base_loss_type
        self.uncertainty_mode = uncertainty_mode
        self.loss_weight = loss_weight

        if base_loss_type == 'l1':
            self.base_loss_fn = nn.L1Loss(reduction='none')
        elif base_loss_type == 'l2':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unsupported base loss type: {base_loss_type}")

    def forward(self, predictions, target, uncertainty):
        """
        Adaptive weighting based on uncertainty
        """
        # Base loss per pixel
        pixel_loss = self.base_loss_fn(predictions, target)

        if self.uncertainty_mode == 'adaptive':
            # Weight loss inversely by uncertainty (more confident predictions get higher weight)
            eps = 1e-6
            weights = 1.0 / (uncertainty + eps)
            weights = weights / weights.mean()  # Normalize weights
            weighted_loss = (pixel_loss * weights).mean()

        elif self.uncertainty_mode == 'threshold':
            # Only consider pixels with low uncertainty
            low_uncertainty_mask = uncertainty < 0.5
            if low_uncertainty_mask.any():
                weighted_loss = pixel_loss[low_uncertainty_mask].mean()
            else:
                weighted_loss = pixel_loss.mean()

        else:
            weighted_loss = pixel_loss.mean()

        return self.loss_weight * weighted_loss


@LOSS_REGISTRY.register()
class EnsembleConsistencyLoss(nn.Module):
    """Encourage consistency among ensemble predictions in low uncertainty regions"""

    def __init__(self, loss_weight=0.1, uncertainty_threshold=0.3):
        super().__init__()
        self.loss_weight = loss_weight
        self.uncertainty_threshold = uncertainty_threshold

    def forward(self, ensemble_predictions, uncertainty):
        """
        Args:
            ensemble_predictions: [num_heads, B, C, H, W]
            uncertainty: [B, C, H, W]
        """
        if ensemble_predictions is None or ensemble_predictions.shape[0] < 2:
            return torch.tensor(0.0, device=uncertainty.device)

        # Low uncertainty mask
        low_uncertainty_mask = uncertainty.mean(dim=1, keepdim=True) < self.uncertainty_threshold

        # Calculate variance among predictions
        ensemble_mean = ensemble_predictions.mean(dim=0)
        ensemble_var = ((ensemble_predictions - ensemble_mean.unsqueeze(0)) ** 2).mean(dim=0)

        # Penalize variance in low uncertainty regions
        if low_uncertainty_mask.any():
            consistency_loss = ensemble_var[low_uncertainty_mask.expand_as(ensemble_var)].mean()
        else:
            consistency_loss = torch.tensor(0.0, device=uncertainty.device)

        return self.loss_weight * consistency_loss


@LOSS_REGISTRY.register()
class UncertaintyRegularizationLoss(nn.Module):
    """Regularize uncertainty to prevent collapse or explosion"""

    def __init__(self, min_uncertainty=0.01, max_uncertainty=1.0, loss_weight=0.01):
        super().__init__()
        self.min_uncertainty = min_uncertainty
        self.max_uncertainty = max_uncertainty
        self.loss_weight = loss_weight

    def forward(self, uncertainty):
        """
        Penalize uncertainty values outside desired range
        """
        # Penalty for too low uncertainty
        low_penalty = F.relu(self.min_uncertainty - uncertainty).mean()

        # Penalty for too high uncertainty
        high_penalty = F.relu(uncertainty - self.max_uncertainty).mean()

        # 添加均值约束
        mean_penalty = F.relu(uncertainty.mean() - 2.0) * 5.0

        # Entropy regularization to encourage diversity in uncertainty values
        uncertainty_flat = uncertainty.flatten()
        uncertainty_norm = (uncertainty_flat - uncertainty_flat.min()) / (
                    uncertainty_flat.max() - uncertainty_flat.min() + 1e-8)
        uncertainty_prob = F.softmax(uncertainty_norm * 10, dim=0)  # 温度参数
        entropy = -(uncertainty_prob * torch.log(uncertainty_prob + 1e-8)).sum()
        entropy_reg = -entropy * 0.01

        total_reg = low_penalty + high_penalty + entropy_reg

        return self.loss_weight * total_reg


@weighted_loss
def uncertainty_aware_l1_loss(pred, target, uncertainty=None):
    """L1 loss weighted by uncertainty"""
    if uncertainty is not None:
        # Weight by inverse uncertainty
        weights = 1.0 / (uncertainty + 1e-6)
        return F.l1_loss(pred, target, reduction='none') * weights
    else:
        return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def uncertainty_aware_l2_loss(pred, target, uncertainty=None):
    """L2 loss weighted by uncertainty"""
    if uncertainty is not None:
        # Weight by inverse uncertainty
        weights = 1.0 / (uncertainty + 1e-6)
        return F.mse_loss(pred, target, reduction='none') * weights
    else:
        return F.mse_loss(pred, target, reduction='none')


@LOSS_REGISTRY.register()
class UncertaintyAwareL1Loss(nn.Module):
    """Wrapper for uncertainty-aware L1 loss"""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, uncertainty=None):
        return self.loss_weight * uncertainty_aware_l1_loss(
            pred, target, weight, reduction=self.reduction, uncertainty=uncertainty
        )


@LOSS_REGISTRY.register()
class UncertaintyAwareL2Loss(nn.Module):
    """Wrapper for uncertainty-aware L2 loss"""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, uncertainty=None):
        return self.loss_weight * uncertainty_aware_l2_loss(
            pred, target, weight, reduction=self.reduction, uncertainty=uncertainty
        )