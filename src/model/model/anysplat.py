import os
from copy import deepcopy
import time
from typing import Optional
from einops import rearrange
import huggingface_hub
from omegaconf import DictConfig, OmegaConf
import torch.distributed
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass

from src.model.types import Gaussians
from src.model.encoder import act_gs, sh_utils
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.decoder.decoder_splatting_cuda import (
    DecoderSplattingCUDA,
    DecoderSplattingCUDACfg,
)
from src.model.encoder.anysplat import (
    EncoderAnySplat,
    EncoderAnySplatCfg,
    OpacityMappingCfg,
)


def _group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim * 2,
            kernel_size=3,
            padding=1,
        )
        self.candidate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden is None:
            hidden = x.new_zeros(
                x.shape[0],
                self.hidden_dim,
                x.shape[-2],
                x.shape[-1],
            )

        gate_input = torch.cat([x, hidden], dim=1)
        reset_gate, update_gate = self.gates(gate_input).chunk(2, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        candidate_input = torch.cat([x, reset_gate * hidden], dim=1)
        candidate = torch.tanh(self.candidate(candidate_input))
        return (1.0 - update_gate) * hidden + update_gate * candidate


class GaussianResidualRefiner(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        sh_dim: int,
        hidden_dim: int = 64,
        num_iters: int = 4,
    ) -> None:
        super().__init__()
        evidence_dim = 9
        self.sh_dim = sh_dim
        self.num_iters = max(1, int(num_iters))
        norm_groups = _group_count(hidden_dim)
        error_context_dim = 5 + 8

        self.evidence_encoder = nn.Sequential(
            nn.Conv2d(feature_dim + evidence_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.error_feature_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(8), 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
        )
        self.error_context_encoder = nn.Sequential(
            nn.Conv2d(error_context_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(norm_groups, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.context_gate = nn.Parameter(torch.tensor(-4.0))
        self.iter_embed = nn.Parameter(torch.zeros(self.num_iters, hidden_dim, 1, 1))
        self.update_block = ConvGRUCell(hidden_dim, hidden_dim)

        self.geometry_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 3 + 4 + 1, kernel_size=1),
        )
        self.density_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1 + 3 + 1, kernel_size=1),
        )
        self.appearance_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, sh_dim + 1, kernel_size=1),
        )
        for head in (self.geometry_head, self.density_head, self.appearance_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def encode_feature_error(
        self,
        render_color: torch.Tensor,
        target_color: torch.Tensor,
    ) -> torch.Tensor:
        b, s, c, h, w = render_color.shape
        render_color = rearrange(render_color.detach().float(), "b s c h w -> (b s) c h w")
        target_color = rearrange(target_color.detach().float(), "b s c h w -> (b s) c h w")
        feature_error = self.error_feature_encoder(render_color) - self.error_feature_encoder(target_color)
        return rearrange(feature_error, "(b s) c h w -> b s c h w", b=b, s=s)

    def forward(
        self,
        features: torch.Tensor,
        rgb_residual: torch.Tensor,
        depth_residual: torch.Tensor,
        alpha: torch.Tensor,
        depth_conf: torch.Tensor,
        depth_uncertainty: torch.Tensor,
        opacity: torch.Tensor,
        scale_norm: torch.Tensor,
        view_gate: torch.Tensor,
        causal_error_context: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
        iter_idx: int = 0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        b, s, c, h, w = features.shape
        x = torch.cat(
            [
                features,
                rgb_residual,
                depth_residual,
                alpha,
                depth_conf,
                depth_uncertainty,
                opacity,
                scale_norm,
            ],
            dim=2,
        )
        x = rearrange(x, "b s c h w -> (b s) c h w")
        evidence = self.evidence_encoder(x)
        if causal_error_context is not None:
            causal_error_context = rearrange(
                causal_error_context.to(evidence.dtype),
                "b s c h w -> (b s) c h w",
            )
            context_evidence = self.error_context_encoder(causal_error_context)
            evidence = evidence + torch.sigmoid(self.context_gate).to(evidence.dtype) * context_evidence
        iter_embed = self.iter_embed[min(max(int(iter_idx), 0), self.num_iters - 1)]
        evidence = evidence + iter_embed
        hidden_state = self.update_block(evidence, hidden_state)

        geometry = self.geometry_head(hidden_state)
        density = self.density_head(hidden_state)
        appearance = self.appearance_head(hidden_state)

        geometry = rearrange(geometry, "(b s) c h w -> b s c h w", b=b, s=s)
        density = rearrange(density, "(b s) c h w -> b s c h w", b=b, s=s)
        appearance = rearrange(appearance, "(b s) c h w -> b s c h w", b=b, s=s)

        delta_mean = geometry[:, :, 0:3]
        delta_quat = geometry[:, :, 3:7]
        geometry_gate = torch.sigmoid(geometry[:, :, 7:8]) * view_gate

        delta_opacity = density[:, :, 0:1]
        delta_scale = density[:, :, 1:4]
        density_gate = torch.sigmoid(density[:, :, 4:5]) * view_gate

        delta_sh = appearance[:, :, : self.sh_dim]
        appearance_gate = torch.sigmoid(appearance[:, :, self.sh_dim : self.sh_dim + 1]) * view_gate

        return (
            geometry_gate * delta_mean.tanh(),
            geometry_gate * delta_quat.tanh(),
            density_gate * delta_opacity.tanh(),
            density_gate * delta_scale.tanh(),
            appearance_gate * delta_sh.tanh(),
            hidden_state,
        )


class AnySplat(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(
        self,
        encoder_cfg: EncoderAnySplatCfg,
        decoder_cfg: DecoderSplattingCUDACfg,
    ):
        super(AnySplat, self).__init__()
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.build_encoder(encoder_cfg)
        self.build_decoder(decoder_cfg)
        self.build_gs_refiner()

    def convert_nested_config(self, cfg_dict: dict, target_class: type):
        """Convert nested dictionary config to dataclass instance

        Args:
            cfg_dict: Configuration dictionary or already converted object
            target_class: Target dataclass type to convert to

        Returns:
            Instance of target_class
        """
        if isinstance(cfg_dict, dict):
            # Convert dict to dataclass
            return target_class(**cfg_dict)
        elif isinstance(cfg_dict, target_class):
            # Already converted, return as is
            return cfg_dict
        elif hasattr(cfg_dict, "__dict__"):
            # Accept equivalent dataclasses from sibling encoder variants.
            return target_class(**cfg_dict.__dict__)
        elif cfg_dict is None:
            # Handle None case
            return None
        else:
            raise ValueError(f"Cannot convert {type(cfg_dict)} to {target_class}")

    def convert_config_recursively(self, cfg_obj, conversion_map: dict):
        """Convert nested configurations recursively using a conversion map

        Args:
            cfg_obj: Configuration object to convert
            conversion_map: Dict mapping field names to their target classes
                           e.g., {'gaussian_adapter': GaussianAdapterCfg}

        Returns:
            Converted configuration object
        """
        if not hasattr(cfg_obj, "__dict__"):
            return cfg_obj

        cfg_dict = cfg_obj.__dict__.copy()

        for field_name, target_class in conversion_map.items():
            if field_name in cfg_dict:
                cfg_dict[field_name] = self.convert_nested_config(
                    cfg_dict[field_name], target_class
                )

        # Return new instance of the same type
        return type(cfg_obj)(**cfg_dict)

    def convert_encoder_config(
        self, encoder_cfg: EncoderAnySplatCfg
    ) -> EncoderAnySplatCfg:
        """Convert all nested configurations in encoder_cfg"""
        conversion_map = {
            "gaussian_adapter": GaussianAdapterCfg,
            "opacity_mapping": OpacityMappingCfg,
        }

        return self.convert_config_recursively(encoder_cfg, conversion_map)

    def build_encoder(self, encoder_cfg: EncoderAnySplatCfg):
        # Convert nested configurations using the helper method
        encoder_cfg = self.convert_encoder_config(encoder_cfg)
        self.encoder = EncoderAnySplat(encoder_cfg)

    def build_decoder(self, decoder_cfg: DecoderSplattingCUDACfg):
        self.decoder = DecoderSplattingCUDA(decoder_cfg)

    def build_gs_refiner(self):
        cfg = self.encoder.cfg
        self.gs_residual_refiner = None
        if not getattr(cfg, "gs_refine_enabled", False):
            return

        feature_dim = self.encoder.feature_dim // 2
        sh_dim = self.encoder.nums_sh * 3
        self.gs_residual_refiner = GaussianResidualRefiner(
            feature_dim=feature_dim,
            sh_dim=sh_dim,
            hidden_dim=cfg.gs_refine_hidden_dim,
            num_iters=cfg.gs_refine_iters,
        )

    def _build_gaussians_from_refine_state(
        self,
        refine_info: dict,
        means_raw: torch.Tensor,
        quats_raw: torch.Tensor,
        scales_raw: torch.Tensor,
        opacities_raw: torch.Tensor,
        res_sh_raw: torch.Tensor,
    ) -> Gaussians:
        return self._build_gaussians_from_raw_state(
            refine_info["base_sh"],
            means_raw,
            quats_raw,
            scales_raw,
            opacities_raw,
            res_sh_raw,
        )

    @staticmethod
    def _build_gaussians_from_raw_state(
        base_sh_raw: torch.Tensor,
        means_raw: torch.Tensor,
        quats_raw: torch.Tensor,
        scales_raw: torch.Tensor,
        opacities_raw: torch.Tensor,
        res_sh_raw: torch.Tensor,
    ) -> Gaussians:
        b, s, n, _ = means_raw.shape
        means = means_raw.reshape(b, s * n, 3)
        rotations = act_gs.reg_dense_rotation(quats_raw).reshape(b, s * n, 4)
        scales = act_gs.reg_dense_scales(scales_raw).clamp_max(0.1).reshape(b, s * n, 3)
        opacities = act_gs.reg_dense_opacities(opacities_raw).reshape(b, s * n)
        harmonics = (base_sh_raw + res_sh_raw).reshape(b, s * n, -1).unsqueeze(-2)
        return Gaussians(
            means=means,
            harmonics=harmonics,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )

    @staticmethod
    def _concat_gaussians(first: Gaussians, second: Gaussians) -> Gaussians:
        return Gaussians(
            means=torch.cat([first.means, second.means], dim=1),
            harmonics=torch.cat([first.harmonics, second.harmonics], dim=1),
            opacities=torch.cat([first.opacities, second.opacities], dim=1),
            scales=torch.cat([first.scales, second.scales], dim=1),
            rotations=torch.cat([first.rotations, second.rotations], dim=1),
        )

    @staticmethod
    def _normalize_render_depth(depth: torch.Tensor, b: int, views: int, h: int, w: int) -> torch.Tensor:
        if depth.dim() == 2:
            depth = depth.view(1, 1, h, w)
        elif depth.dim() == 3:
            if depth.shape[0] == b * views:
                depth = depth.view(b, views, h, w)
            elif depth.shape[0] == views and b == 1:
                depth = depth.unsqueeze(0)
            else:
                depth = depth.view(b, views, h, w)
        elif depth.dim() != 4:
            depth = depth.reshape(b, views, h, w)
        return depth.unsqueeze(2)

    @staticmethod
    def _normalize_render_alpha(alpha: torch.Tensor, b: int, views: int, h: int, w: int) -> torch.Tensor:
        if alpha.dim() == 2:
            alpha = alpha.view(1, 1, h, w)
        elif alpha.dim() == 3:
            if alpha.shape[0] == b * views:
                alpha = alpha.view(b, views, h, w)
            elif alpha.shape[0] == views and b == 1:
                alpha = alpha.unsqueeze(0)
            else:
                alpha = alpha.view(b, views, h, w)
        elif alpha.dim() != 4:
            alpha = alpha.reshape(b, views, h, w)
        return alpha.unsqueeze(2)

    def _refine_gaussians(
        self,
        encoder_output,
        context_image: torch.Tensor,
        pred_all_extrinsic: torch.Tensor,
        pred_context_pose: dict,
        ctx_img_num: int,
        near: float,
        far: float,
    ) -> Gaussians:
        if self.gs_residual_refiner is None:
            return encoder_output.gaussians
        refine_info = None if encoder_output.infos is None else encoder_output.infos.get("gs_refine")
        if refine_info is None:
            return encoder_output.gaussians

        cfg = self.encoder.cfg
        features = refine_info["features"]
        b, s, _, h, w = features.shape
        if s <= 1:
            return encoder_output.gaussians

        device = features.device
        evidence_features = features.detach() if cfg.gs_refine_detach_evidence else features
        render_scale = float(max(0.0, min(1.0, cfg.gs_refine_render_scale)))
        low_h = max(8, int(round(h * render_scale)))
        low_w = max(8, int(round(w * render_scale)))

        means_raw = refine_info["means"]
        quats_raw = refine_info["quats"]
        scales_raw = refine_info["scales_raw"]
        opacities_raw = refine_info["opacities_raw"]
        res_sh_raw = refine_info["res_sh_raw"]
        base_sh_raw = refine_info["base_sh"]

        depth = refine_info["depth"]
        if cfg.gs_refine_detach_evidence:
            depth = depth.detach()
        depth_low = rearrange(depth, "b s h w c -> (b s) c h w")
        depth_low = F.interpolate(depth_low.float(), size=(low_h, low_w), mode="bilinear", align_corners=False)
        depth_low = rearrange(depth_low, "(b s) c h w -> b s c h w", b=b, s=s)

        depth_conf = refine_info["depth_conf"]
        if cfg.gs_refine_detach_evidence:
            depth_conf = depth_conf.detach()
        depth_conf = rearrange(depth_conf, "b s h w c -> (b s) c h w")
        depth_conf = F.interpolate(depth_conf.float(), size=(h, w), mode="bilinear", align_corners=False)
        depth_conf = rearrange(depth_conf, "(b s) c h w -> b s c h w", b=b, s=s)

        depth_uncertainty = refine_info["depth_uncertainty"]
        if depth_uncertainty is None:
            depth_uncertainty = torch.zeros((b, s, 1, h, w), device=device, dtype=features.dtype)
        else:
            if cfg.gs_refine_detach_evidence:
                depth_uncertainty = depth_uncertainty.detach()
            depth_uncertainty = rearrange(depth_uncertainty, "b s h w c -> (b s) c h w")
            depth_uncertainty = F.interpolate(
                depth_uncertainty.float(), size=(h, w), mode="bilinear", align_corners=False
            )
            depth_uncertainty = rearrange(depth_uncertainty, "(b s) c h w -> b s c h w", b=b, s=s)

        target_low = rearrange(context_image[:, :s], "b s c h w -> (b s) c h w")
        target_low = F.interpolate(target_low.float(), size=(low_h, low_w), mode="bilinear", align_corners=False)
        target_low = rearrange(target_low, "(b s) c h w -> b s c h w", b=b, s=s)

        render_intrinsics = pred_context_pose["intrinsic"][:, 0:1].detach()
        near_tensor = torch.full((b, 1), near, device=device)
        far_tensor = torch.full((b, 1), far, device=device)

        def refine_current_view(
            view_idx: int,
            means_state: torch.Tensor,
            quats_state: torch.Tensor,
            scales_state: torch.Tensor,
            opacities_state: torch.Tensor,
            sh_state: torch.Tensor,
        ):
            # The historical map is a fixed pseudo-state. Gradients only train
            # how the current frame consumes that state and updates its own GS.
            with torch.no_grad():
                old_gaussians = self._build_gaussians_from_raw_state(
                    base_sh_raw[:, :view_idx].detach(),
                    means_state[:, :view_idx].detach(),
                    quats_state[:, :view_idx].detach(),
                    scales_state[:, :view_idx].detach(),
                    opacities_state[:, :view_idx].detach(),
                    sh_state[:, :view_idx].detach(),
                )
                old_output = self.decoder.forward(
                    old_gaussians,
                    pred_all_extrinsic[:, view_idx : view_idx + 1].detach(),
                    render_intrinsics,
                    near_tensor,
                    far_tensor,
                    (low_h, low_w),
                    "depth",
                )
                render_color = old_output.color.detach()
                render_depth = self._normalize_render_depth(
                    old_output.depth.detach(), b, 1, low_h, low_w
                )
                render_alpha = self._normalize_render_alpha(
                    old_output.alpha.detach(), b, 1, low_h, low_w
                )

            current_target_low = target_low[:, view_idx : view_idx + 1]
            rgb_residual_low = (render_color - current_target_low).to(features.dtype)
            current_depth_low = depth_low[:, view_idx : view_idx + 1].clamp_min(1e-4)
            depth_residual_low = ((render_depth - current_depth_low) / current_depth_low).clamp(-1.0, 1.0)
            depth_residual_low = depth_residual_low.to(features.dtype)
            alpha_low = render_alpha.to(features.dtype)

            def upsample_error(error: torch.Tensor) -> torch.Tensor:
                error = rearrange(error, "b s c h w -> (b s) c h w")
                error = F.interpolate(error, size=(h, w), mode="bilinear", align_corners=False)
                return rearrange(error, "(b s) c h w -> b s c h w", b=b, s=1)

            rgb_residual = upsample_error(rgb_residual_low)
            depth_residual = upsample_error(depth_residual_low)
            alpha = upsample_error(alpha_low)
            feature_error = self.gs_residual_refiner.encode_feature_error(
                render_color,
                current_target_low,
            )
            feature_error = upsample_error(feature_error)
            error_context = torch.cat(
                [
                    rgb_residual.detach().float(),
                    depth_residual.detach().float(),
                    alpha.detach().float(),
                    feature_error.float(),
                ],
                dim=2,
            )

            current_means = means_state[:, view_idx : view_idx + 1]
            current_quats = quats_state[:, view_idx : view_idx + 1]
            current_scales = scales_state[:, view_idx : view_idx + 1]
            current_opacities = opacities_state[:, view_idx : view_idx + 1]
            current_sh = sh_state[:, view_idx : view_idx + 1]
            current_features = evidence_features[:, view_idx : view_idx + 1]
            current_depth_conf = depth_conf[:, view_idx : view_idx + 1]
            current_depth_uncertainty = depth_uncertainty[:, view_idx : view_idx + 1]
            mean_step_source = depth[:, view_idx : view_idx + 1]
            if cfg.gs_refine_detach_evidence:
                mean_step_source = mean_step_source.detach()
            mean_step = rearrange(mean_step_source.float(), "b s h w c -> b s (h w) c")
            view_gate = torch.ones((b, 1, 1, h, w), device=device, dtype=features.dtype)
            refiner_hidden = None

            for refine_iter in range(max(0, int(cfg.gs_refine_iters))):
                opacity_source = current_opacities.detach() if cfg.gs_refine_detach_evidence else current_opacities
                scale_source = current_scales.detach() if cfg.gs_refine_detach_evidence else current_scales
                opacity = act_gs.reg_dense_opacities(opacity_source)
                opacity = rearrange(opacity, "b s (h w) c -> b s c h w", h=h, w=w)
                scale_norm = act_gs.reg_dense_scales(scale_source).clamp_max(0.1).norm(dim=-1, keepdim=True)
                scale_norm = rearrange(scale_norm, "b s (h w) c -> b s c h w", h=h, w=w)

                (
                    delta_mean,
                    delta_quat,
                    delta_opacity,
                    delta_scale,
                    delta_sh,
                    refiner_hidden,
                ) = self.gs_residual_refiner(
                    current_features.float(),
                    rgb_residual.float(),
                    depth_residual.float(),
                    alpha.float(),
                    current_depth_conf.float(),
                    current_depth_uncertainty.float(),
                    opacity.float(),
                    scale_norm.float(),
                    view_gate.float(),
                    error_context,
                    refiner_hidden,
                    refine_iter,
                )

                delta_mean = rearrange(delta_mean, "b s c h w -> b s (h w) c")
                delta_quat = rearrange(delta_quat, "b s c h w -> b s (h w) c")
                delta_opacity = rearrange(delta_opacity, "b s c h w -> b s (h w) c")
                delta_scale = rearrange(delta_scale, "b s c h w -> b s (h w) c")
                delta_sh = rearrange(delta_sh, "b s c h w -> b s (h w) c")

                current_means = current_means + mean_step.to(current_means.dtype) * delta_mean.to(current_means.dtype)
                current_quats = current_quats + delta_quat.to(current_quats.dtype)
                current_opacities = current_opacities + cfg.gs_refine_step_opacity * delta_opacity.to(current_opacities.dtype)
                current_scales = current_scales + cfg.gs_refine_step_scale * delta_scale.to(current_scales.dtype)
                current_sh = current_sh + cfg.gs_refine_step_sh * delta_sh.to(current_sh.dtype)

            current_gaussians = self._build_gaussians_from_raw_state(
                base_sh_raw[:, view_idx : view_idx + 1],
                current_means,
                current_quats,
                current_scales,
                current_opacities,
                current_sh,
            )
            return (
                current_means,
                current_quats,
                current_scales,
                current_opacities,
                current_sh,
                old_gaussians,
                current_gaussians,
            )

        if self.training:
            # One randomly sampled transition trains the streaming update rule
            # without backpropagating through a full sequential rollout.
            refine_view_idx = int(torch.randint(1, s, (), device=device).item())
            (
                current_means,
                current_quats,
                current_scales,
                current_opacities,
                current_sh,
                old_gaussians,
                current_gaussians,
            ) = refine_current_view(
                refine_view_idx,
                means_raw,
                quats_raw,
                scales_raw,
                opacities_raw,
                res_sh_raw,
            )
            refined_gaussians = self._concat_gaussians(old_gaussians, current_gaussians)
        else:
            # Validation/test simulate streaming: each refined current frame is
            # appended to the map and becomes history for the next frame.
            for refine_view_idx in range(1, s):
                (
                    current_means,
                    current_quats,
                    current_scales,
                    current_opacities,
                    current_sh,
                    _,
                    _,
                ) = refine_current_view(
                    refine_view_idx,
                    means_raw,
                    quats_raw,
                    scales_raw,
                    opacities_raw,
                    res_sh_raw,
                )

                def replace_view(state: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
                    return torch.cat(
                        [state[:, :refine_view_idx], current, state[:, refine_view_idx + 1 :]],
                        dim=1,
                    )

                means_raw = replace_view(means_raw, current_means)
                quats_raw = replace_view(quats_raw, current_quats)
                scales_raw = replace_view(scales_raw, current_scales)
                opacities_raw = replace_view(opacities_raw, current_opacities)
                res_sh_raw = replace_view(res_sh_raw, current_sh)

            refined_gaussians = self._build_gaussians_from_refine_state(
                refine_info,
                means_raw,
                quats_raw,
                scales_raw,
                opacities_raw,
                res_sh_raw,
            )

        if encoder_output.infos is not None:
            encoder_output.infos.pop("gs_refine", None)
            encoder_output.infos["gs_refine_steps"] = torch.tensor(
                int(cfg.gs_refine_iters), device=device
            )
            encoder_output.infos["gs_refine_history_views"] = torch.tensor(
                refine_view_idx, device=device
            )
        return refined_gaussians

    @torch.no_grad()
    def inference(
        self,
        context_image: torch.Tensor,
    ):
        self.encoder.distill = False
        encoder_output = self.encoder(
            context_image, global_step=0, visualization_dump=None
        )
        gaussians, pred_context_pose = (
            encoder_output.gaussians,
            encoder_output.pred_context_pose,
        )
        return gaussians, pred_context_pose

    def forward(
        self,
        context_image: torch.Tensor,
        ctx_index: list = None, 
        global_step: int = 0,
        near: float = 0.01,
        far: float = 100.0,
    ):
        b, v, c, h, w = context_image.shape
        device = context_image.device

        encoder_output, pred_all_extrinsic, ctx_img_num = self.encoder(
            context_image, ctx_index, global_step=global_step
        )
        gaussians, pred_context_pose = (
            encoder_output.gaussians,
            encoder_output.pred_context_pose,
        )
        gaussians = self._refine_gaussians(
            encoder_output,
            context_image,
            pred_all_extrinsic,
            pred_context_pose,
            ctx_img_num,
            near,
            far,
        )
        encoder_output.gaussians = gaussians

        # num_context_view = ctx_img_num
        # pred_all_context_extrinsic, pred_all_target_extrinsic = (
        #     pred_all_extrinsic[:, :num_context_view],
        #     pred_all_extrinsic[:, num_context_view:],
        # )
        # scale_factor = (
        #     pred_context_pose["extrinsic"][:, :, :3, 3].mean()
        #     / pred_all_context_extrinsic[:, :, :3, 3].mean()
        # )
        # pred_all_target_extrinsic[..., :3, 3] = (
        #     pred_all_target_extrinsic[..., :3, 3] * scale_factor
        # )
        # pred_all_context_extrinsic[..., :3, 3] = (
        #     pred_all_context_extrinsic[..., :3, 3] * scale_factor
        # )
        # pred_context_ex = torch.cat(
        #     (pred_context_pose["extrinsic"], pred_all_target_extrinsic), dim=1
        # )

        output = self.decoder.forward(
            gaussians,
            pred_all_extrinsic.detach(),
            pred_context_pose["intrinsic"][:, 0:1, ...].repeat(1, v, 1, 1).detach(),
            torch.ones(b, v, device=device) * near,
            torch.ones(b, v, device=device) * far,
            (h, w),
            "depth",
        )
        output.depth = output.depth[:, :ctx_img_num, ...]

        return encoder_output, output
