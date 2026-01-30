import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Gamma
import math
import numpy as np

from basicsr.archs.ddcolor_arch_utils.unet import Hook, CustomPixelShuffle_ICNR, UnetBlockWide, NormType, \
    custom_conv_layer
from basicsr.archs.ddcolor_arch_utils.convnext import ConvNeXt
from basicsr.archs.ddcolor_arch_utils.transformer_utils import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from basicsr.archs.ddcolor_arch_utils.position_encoding import PositionEmbeddingSine
from basicsr.archs.ddcolor_arch_utils.transformer import Transformer
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DDColorWithUncertainty(nn.Module):
    """DDColor with uncertainty estimation capabilities"""

    def __init__(self,
                 encoder_name='convnext-l',
                 decoder_name='MultiScaleColorDecoder',
                 num_input_channels=3,
                 input_size=(256, 256),
                 nf=512,
                 num_output_channels=3,
                 last_norm='Weight',
                 do_normalize=False,
                 num_queries=256,
                 num_scales=3,
                 dec_layers=9,
                 encoder_from_pretrain=False,
                 num_ensemble_heads=3,  # Number of ensemble heads for uncertainty
                 uncertainty_weight=0.1):  # Weight for uncertainty loss
        super().__init__()

        self.num_ensemble_heads = num_ensemble_heads
        self.uncertainty_weight = uncertainty_weight

        # Original DDColor components
        self.encoder = Encoder(encoder_name, ['norm0', 'norm1', 'norm2', 'norm3'], from_pretrain=encoder_from_pretrain)
        self.encoder.eval()
        test_input = torch.randn(1, num_input_channels, *input_size)
        self.encoder(test_input)

        self.decoder = DecoderWithUncertainty(
            self.encoder.hooks,
            nf=nf,
            last_norm=last_norm,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
            decoder_name=decoder_name,
            num_ensemble_heads=num_ensemble_heads
        )

        # Multiple prediction heads for ensemble
        self.refine_nets = nn.ModuleList([
            nn.Sequential(custom_conv_layer(num_queries + 3, num_output_channels, ks=1, use_activ=False,
                                            norm_type=NormType.Spectral))
            for _ in range(num_ensemble_heads)
        ])

        # Uncertainty estimation head
        self.uncertainty_net = nn.Sequential(
            custom_conv_layer(num_queries + 3, 64, ks=3, stride=1, padding=1, norm_type=NormType.Spectral),
            nn.ReLU(inplace=False),
            custom_conv_layer(64, 32, ks=3, stride=1, padding=1, norm_type=NormType.Spectral),
            nn.ReLU(inplace=False),
            custom_conv_layer(32, 2, ks=1, use_activ=False)  # Output variance for a and b channels
        )

        self.do_normalize = do_normalize
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 初始化不确定性网络，使其输出较小的初始值
        with torch.no_grad():
            # 找到uncertainty_net的最后一层
            for module in self.uncertainty_net.modules():
                if isinstance(module, nn.Conv2d) and module.out_channels == 2:
                    # 这是最后一层 - 修改这里的初始化
                    module.weight.data *= 0.001  # 从0.01改为0.001，进一步减小权重
                    if module.bias is not None:
                        # 从-2.3改为-4.6，对应exp(-4.6) ≈ 0.01的方差
                        module.bias.data = torch.full_like(module.bias.data, -4.6)
                    break

        # 额外添加：对所有不确定性网络层进行更保守的初始化
        for module in self.uncertainty_net.modules():
            if isinstance(module, nn.Conv2d):
                # 使用更小的初始化方差
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # 初始化集成头使其有差异
        for i, net in enumerate(self.refine_nets):
            with torch.no_grad():
                for param in net.parameters():
                    noise = torch.randn_like(param) * 0.02 * (i + 1)
                    param.data += noise

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, x, return_uncertainty=True):
        if x.shape[1] == 3:
            x = self.normalize(x)

        self.encoder(x)
        out_feat = self.decoder()

        # Generate multiple predictions using ensemble heads
        predictions = []
        for refine_net in self.refine_nets:
            coarse_input = torch.cat([out_feat, x], dim=1)
            pred = refine_net(coarse_input)
            predictions.append(pred)

        # Stack predictions: [num_heads, batch, channels, height, width]
        predictions = torch.stack(predictions, dim=0)

        # Calculate mean prediction
        mean_prediction = predictions.mean(dim=0)

        # Calculate uncertainty (variance across ensemble)
        if return_uncertainty:
            # Ensemble uncertainty (epistemic)
            ensemble_variance = predictions.var(dim=0)

            # Aleatoric uncertainty from uncertainty network
            uncertainty_input = torch.cat([out_feat, x], dim=1)
            aleatoric_log_var = self.uncertainty_net(uncertainty_input)

            # 更严格的裁剪范围，防止数值爆炸
            aleatoric_log_var = torch.clamp(aleatoric_log_var, min=-6.0, max=0.0)  # 从(-3.0, 2.0)改为(-6.0, 0.0)
            aleatoric_variance = torch.exp(aleatoric_log_var)

            # 对集成不确定性也进行裁剪
            ensemble_variance = torch.clamp(ensemble_variance, min=0.0, max=1.0)

            # Total uncertainty with weighting
            # 降低偶然不确定性的权重
            total_uncertainty = ensemble_variance + 0.1 * aleatoric_variance  # 添加权重因子

            if self.do_normalize:
                mean_prediction = self.denormalize(mean_prediction)

            return mean_prediction, total_uncertainty, predictions
        else:
            if self.do_normalize:
                mean_prediction = self.denormalize(mean_prediction)
            return mean_prediction


class DecoderWithUncertainty(nn.Module):
    """Decoder with uncertainty-aware features"""

    def __init__(self,
                 hooks,
                 nf=512,
                 blur=True,
                 last_norm='Weight',
                 num_queries=256,
                 num_scales=3,
                 dec_layers=9,
                 decoder_name='MultiScaleColorDecoder',
                 num_ensemble_heads=3):
        super().__init__()
        self.hooks = hooks
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)
        self.decoder_name = decoder_name
        self.num_ensemble_heads = num_ensemble_heads

        self.layers = self.make_layers()
        embed_dim = nf // 2

        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm,
                                                 scale=4)

        if self.decoder_name == 'MultiScaleColorDecoder':
            self.color_decoder = MultiScaleColorDecoderWithDropout(
                in_channels=[512, 512, 256],
                num_queries=num_queries,
                num_scales=num_scales,
                dec_layers=dec_layers,
                dropout_rate=0.1  # Add dropout for uncertainty
            )
        else:
            self.color_decoder = SingleColorDecoder(
                in_channels=hooks[-1].feature.shape[1],
                num_queries=num_queries,
            )

    def forward(self):
        encode_feat = self.hooks[-1].feature
        out0 = self.layers[0](encode_feat)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out1)
        out3 = self.last_shuf(out2)

        if self.decoder_name == 'MultiScaleColorDecoder':
            out = self.color_decoder([out0, out1, out2], out3)
        else:
            out = self.color_decoder(out3, encode_feat)

        return out

    def make_layers(self):
        decoder_layers = []
        e_in_c = self.hooks[-1].feature.shape[1]
        in_c = e_in_c
        out_c = self.nf
        setup_hooks = self.hooks[-2::-1]
        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if layer_index == len(setup_hooks) - 1:
                out_c = out_c // 2
            decoder_layers.append(
                UnetBlockWide(
                    in_c, feature_c, out_c, hook, blur=self.blur, self_attention=False, norm_type=NormType.Spectral))
            in_c = out_c
        return nn.Sequential(*decoder_layers)


class MultiScaleColorDecoderWithDropout(nn.Module):
    """MultiScaleColorDecoder with dropout for uncertainty estimation"""

    def __init__(
            self,
            in_channels,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,
            pre_norm=False,
            color_embed_dim=256,
            enforce_input_project=True,
            num_scales=3,
            dropout_rate=0.1
    ):
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()  # Add dropout layers

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=dropout_rate,  # Increased dropout
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=dropout_rate,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout_rate,
                    normalize_before=pre_norm,
                )
            )
            self.dropout_layers.append(nn.Dropout(dropout_rate))

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable color query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable color query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding
        self.num_feature_levels = num_scales
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # input projections
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            if in_channels[i] != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    def forward(self, x, img_features):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []

        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](output)
            # Apply dropout for uncertainty
            output = self.dropout_layers[i](output)

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [N, bs, C]  -> [bs, N, C]
        color_embed = self.color_embed(decoder_output)
        out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)

        return out


# Keep the original Encoder class as is
class Encoder(nn.Module):
    def __init__(self, encoder_name, hook_names, from_pretrain, **kwargs):
        super().__init__()

        if encoder_name == 'convnext-t' or encoder_name == 'convnext':
            self.arch = ConvNeXt()
        elif encoder_name == 'convnext-s':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
        elif encoder_name == 'convnext-b':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        elif encoder_name == 'convnext-l':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        else:
            raise NotImplementedError

        self.encoder_name = encoder_name
        self.hook_names = hook_names
        self.hooks = self.setup_hooks()

        if from_pretrain:
            self.load_pretrain_model()

    def setup_hooks(self):
        hooks = [Hook(self.arch._modules[name]) for name in self.hook_names]
        return hooks

    def forward(self, x):
        return self.arch(x)

    def load_pretrain_model(self):
        if self.encoder_name == 'convnext-t' or self.encoder_name == 'convnext':
            self.load('pretrain/convnext_tiny_22k_224.pth')
        elif self.encoder_name == 'convnext-s':
            self.load('pretrain/convnext_small_22k_224.pth')
        elif self.encoder_name == 'convnext-b':
            self.load('pretrain/convnext_base_22k_224.pth')
        elif self.encoder_name == 'convnext-l':
            self.load('/home/sunbowen/snap/DDColor-master_create01/pretrain/convnext_large_22k_224.pth')
        else:
            raise NotImplementedError
        print('Loaded pretrained convnext model.')

    def load(self, path):
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        if not path:
            logger.info("No checkpoint found. Initializing model from scratch")
            return
        logger.info("[Encoder] Loading from {} ...".format(path))
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        checkpoint_state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        incompatible = self.arch.load_state_dict(checkpoint_state_dict, strict=False)

        if incompatible.missing_keys:
            msg = "Some model parameters or buffers are not found in the checkpoint:\n"
            msg += str(incompatible.missing_keys)
            logger.warning(msg)
        if incompatible.unexpected_keys:
            msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
            msg += str(incompatible.unexpected_keys)
            logger.warning(msg)


# Keep the original SingleColorDecoder class if needed
class SingleColorDecoder(nn.Module):
    def __init__(
            self,
            in_channels=768,
            hidden_dim=256,
            num_queries=256,
            nheads=8,
            dropout=0.1,
            dim_feedforward=2048,
            enc_layers=0,
            dec_layers=6,
            pre_norm=False,
            deep_supervision=True,
            enforce_input_project=True,
    ):
        super().__init__()

        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
        self.num_queries = num_queries
        self.transformer = transformer

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            nn.init.kaiming_uniform_(self.input_proj.weight, a=1)
            if self.input_proj.bias is not None:
                nn.init.constant_(self.input_proj.bias, 0)
        else:
            self.input_proj = nn.Sequential()

    def forward(self, img_features, encode_feat):
        pos = self.pe_layer(encode_feat)
        src = encode_feat
        mask = None
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)
        color_embed = hs[-1]
        color_preds = torch.einsum('bqc,bchw->bqhw', color_embed, img_features)
        return color_preds