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
from src.model.streaming_gir import (
    DominantGIR,
    DominantGIRRenderer,
    GIRUpdateHead,
    StreamingGaussianState,
)
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
        self.reprojection_encoder = nn.Sequential(
            nn.Conv2d(16, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.context_gate = nn.Parameter(torch.tensor(-4.0))
        self.reprojection_gate = nn.Parameter(torch.tensor(0.0))
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
        nn.init.zeros_(self.reprojection_encoder[-1].weight)
        nn.init.zeros_(self.reprojection_encoder[-1].bias)

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
        reprojection_evidence: Optional[torch.Tensor] = None,
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
        if reprojection_evidence is not None:
            reprojection_evidence = rearrange(
                reprojection_evidence.to(evidence.dtype),
                "b s c h w -> (b s) c h w",
            )
            reprojection_feature = self.reprojection_encoder(reprojection_evidence)
            reprojection_feature = F.interpolate(
                reprojection_feature,
                size=evidence.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            evidence = evidence + (
                torch.sigmoid(self.reprojection_gate).to(evidence.dtype)
                * reprojection_feature
            )
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
        self.gir_renderer = None
        self.gir_update_head = None
        if getattr(cfg, "gir_enabled", False):
            if getattr(cfg, "gs_refine_enabled", False):
                raise ValueError(
                    "gir_enabled and gs_refine_enabled are mutually exclusive."
                )
            self.gir_renderer = DominantGIRRenderer()
            self.gir_update_head = GIRUpdateHead(
                feature_dim=self.encoder.feature_dim // 2,
                harmonic_dim=self.encoder.nums_sh * 3,
                hidden_dim=cfg.gir_hidden_dim,
            )
            return
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

    @staticmethod
    def _slice_gaussian_view(
        gaussians: Gaussians,
        view_idx: int,
        gaussians_per_view: int,
    ) -> Gaussians:
        start = view_idx * gaussians_per_view
        end = start + gaussians_per_view
        return Gaussians(
            means=gaussians.means[:, start:end],
            harmonics=gaussians.harmonics[:, start:end],
            opacities=gaussians.opacities[:, start:end],
            scales=gaussians.scales[:, start:end],
            rotations=gaussians.rotations[:, start:end],
        )

    def _update_streaming_gaussians(
        self,
        encoder_output,
        context_image: torch.Tensor,
        pred_all_extrinsic: torch.Tensor,
        pred_context_pose: dict,
        ctx_img_num: int,
        near: float,
        far: float,
    ) -> Gaussians:
        refine_info = None if encoder_output.infos is None else encoder_output.infos.get("gs_refine")
        if refine_info is None:
            raise RuntimeError("GIR is enabled, but the encoder did not return per-view GS data.")

        cfg = self.encoder.cfg
        features = refine_info["features"]
        b, source_views, _, h, w = features.shape
        if source_views != ctx_img_num:
            raise RuntimeError(
                "GIR source-view mismatch: "
                f"encoder returned {source_views}, expected {ctx_img_num}."
            )

        render_scale = float(max(0.05, min(1.0, cfg.gir_render_scale)))
        low_h = max(8, int(round(h * render_scale)))
        low_w = max(8, int(round(w * render_scale)))
        gaussians_per_view = h * w
        intrinsics = pred_context_pose["intrinsic"]
        if intrinsics.shape[1] == 1 and source_views > 1:
            intrinsics = intrinsics.expand(-1, source_views, -1, -1)

        depth = refine_info["depth"]
        depth_confidence = refine_info["depth_conf"]
        state: Optional[StreamingGaussianState] = None
        auxiliary_losses = []
        regularization_losses = []
        add_gates = []
        historical_gates = []
        visible_ratios = []
        residual_magnitudes = []

        for view_idx in range(source_views):
            current_feature = features[:, view_idx]
            current_rgb = context_image[:, view_idx]
            current_depth = depth[:, view_idx].permute(0, 3, 1, 2)
            current_depth_confidence = depth_confidence[:, view_idx].permute(0, 3, 1, 2)
            current_gaussians = self._slice_gaussian_view(
                encoder_output.gaussians,
                view_idx,
                gaussians_per_view,
            )

            if state is None:
                gir = DominantGIR.empty(
                    b,
                    low_h,
                    low_w,
                    features.device,
                    features.dtype,
                )
            else:
                gir = self.gir_renderer(
                    state,
                    pred_all_extrinsic[:, view_idx],
                    intrinsics[:, view_idx],
                    (low_h, low_w),
                )

            prediction = self.gir_update_head(
                current_feature,
                current_rgb,
                current_depth,
                current_depth_confidence,
                gir,
            )
            prediction.historical_gate = prediction.historical_gate - 4.0

            if state is not None:
                state = state.update_historical(
                    gir,
                    prediction,
                    pred_all_extrinsic[:, view_idx],
                )

            coverage = gir.valid.to(prediction.add_logit.dtype)
            visible_ratios.append(coverage.mean())
            if state is None:
                # Keep frame zero identical to the base GS prediction.
                add_gate_low = torch.ones_like(prediction.add_logit) + (
                    0.0 * prediction.add_logit
                )
            else:
                add_prior = torch.where(
                    gir.valid,
                    prediction.add_logit.new_full((), -2.0),
                    prediction.add_logit.new_full((), 6.0),
                )
                add_gate_low = torch.sigmoid(prediction.add_logit + add_prior)
            add_gate = F.interpolate(
                add_gate_low,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            add_gates.append(add_gate_low.mean())
            historical_gates.append(
                (prediction.historical_gate.sigmoid() * coverage).sum()
                / coverage.sum().clamp_min(1.0)
            )
            valid_normalizer = coverage.sum().clamp_min(1.0)
            residual_energy = (
                prediction.delta_mean_camera.square().sum(dim=1, keepdim=True)
                + prediction.delta_rotation.square().sum(dim=1, keepdim=True)
                + prediction.delta_log_scale.square().sum(dim=1, keepdim=True)
                + prediction.delta_opacity_logit.square()
                + prediction.delta_harmonics.square().mean(dim=1, keepdim=True)
            )
            residual_magnitudes.append(residual_energy.mean().sqrt())
            covered_add = add_gate_low.square() * coverage
            uncovered_missing = (1.0 - add_gate_low).square() * (1.0 - coverage)
            regularization_losses.append(
                (residual_energy * coverage).sum() / valid_normalizer
                + covered_add.sum() / valid_normalizer
                + uncovered_missing.mean()
            )

            if state is None:
                state = StreamingGaussianState.from_current(
                    current_gaussians,
                    add_gate,
                )
            else:
                state = state.append(current_gaussians, add_gate)

            if self.training and cfg.gir_aux_loss_weight > 0:
                replay_count = max(0, int(cfg.gir_replay_views))
                replay_indices = list(
                    range(max(0, view_idx - replay_count), view_idx)
                )
                render_indices = replay_indices + [view_idx]
                render_views = len(render_indices)
                render_output = self.decoder.forward(
                    state.gaussians,
                    pred_all_extrinsic[:, render_indices],
                    intrinsics[:, render_indices],
                    torch.full(
                        (b, render_views), near, device=features.device
                    ),
                    torch.full(
                        (b, render_views), far, device=features.device
                    ),
                    (low_h, low_w),
                    "depth",
                )
                target = context_image[:, render_indices]
                target = rearrange(target, "b v c h w -> (b v) c h w")
                target = F.interpolate(
                    target.float(),
                    size=(low_h, low_w),
                    mode="bilinear",
                    align_corners=False,
                )
                target = rearrange(
                    target,
                    "(b v) c h w -> b v c h w",
                    b=b,
                    v=render_views,
                ).to(render_output.color.dtype)
                difference = render_output.color - target
                auxiliary_losses.append(
                    torch.sqrt(difference.square() + 1e-6).mean()
                )

            chunk_size = max(0, int(cfg.gir_tbptt_chunk))
            if (
                self.training
                and chunk_size > 0
                and (view_idx + 1) % chunk_size == 0
                and view_idx + 1 < source_views
            ):
                state = state.detach()

        if state is None:
            return encoder_output.gaussians

        if encoder_output.infos is not None:
            encoder_output.infos.pop("gs_refine", None)
            encoder_output.infos["gir_history_views"] = torch.tensor(
                max(source_views - 1, 0), device=features.device
            )
            encoder_output.infos["gir_map_gaussians"] = torch.tensor(
                state.num_gaussians, device=features.device
            )
            encoder_output.infos["gir_add_gate"] = torch.stack(add_gates).mean()
            encoder_output.infos["gir_historical_gate"] = torch.stack(
                historical_gates
            ).mean()
            encoder_output.infos["gir_visible_ratio"] = torch.stack(
                visible_ratios
            ).mean()
            encoder_output.infos["gir_residual_magnitude"] = torch.stack(
                residual_magnitudes
            ).mean()
            if auxiliary_losses:
                encoder_output.infos["gir_aux_loss"] = torch.stack(
                    auxiliary_losses
                ).mean()
            encoder_output.infos["gir_regularization_loss"] = torch.stack(
                regularization_losses
            ).mean()

        return state.gaussians

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
        if self.gir_update_head is not None:
            return self._update_streaming_gaussians(
                encoder_output,
                context_image,
                pred_all_extrinsic,
                pred_context_pose,
                ctx_img_num,
                near,
                far,
            )
        if self.gs_residual_refiner is None:
            return encoder_output.gaussians
        refine_info = None if encoder_output.infos is None else encoder_output.infos.get("gs_refine")
        if refine_info is None:
            return encoder_output.gaussians

        cfg = self.encoder.cfg
        features = refine_info["features"]
        b, s, _, h, w = features.shape
        if s <= 1:
            if self.training:
                raise RuntimeError(
                    "Old-only GS refinement requires at least two source views. "
                    "Increase the training sampler's minimum total view count to four."
                )
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

        with torch.no_grad():
            reprojection_features_low = rearrange(
                features.detach().float(), "b s c h w -> (b s) c h w"
            )
            reprojection_features_low = F.interpolate(
                reprojection_features_low,
                size=(low_h, low_w),
                mode="bilinear",
                align_corners=False,
            )
            reprojection_features_low = F.normalize(
                reprojection_features_low, dim=1, eps=1e-6
            )
            reprojection_features_low = rearrange(
                reprojection_features_low,
                "(b s) c h w -> b s c h w",
                b=b,
                s=s,
            )
            source_w2c = torch.linalg.inv(
                pred_all_extrinsic[:, :s].detach().float()
            )
            source_intrinsics = pred_context_pose["intrinsic"].detach().float()
            if source_intrinsics.shape[1] == 1 and s > 1:
                source_intrinsics = source_intrinsics.expand(-1, s, -1, -1)
            source_intrinsics = source_intrinsics[:, :s]
            history_rgb_depth_low = torch.cat(
                [target_low.detach().float(), depth_low.detach().float()], dim=2
            )

            reprojection_offsets = torch.tensor(
                [
                    [-1, -1],
                    [0, -1],
                    [1, -1],
                    [-1, 0],
                    [0, 0],
                    [1, 0],
                    [-1, 1],
                    [0, 1],
                    [1, 1],
                ],
                device=device,
                dtype=torch.float32,
            )
            reprojection_offsets[:, 0] *= 2.0 / float(low_w)
            reprojection_offsets[:, 1] *= 2.0 / float(low_h)

        def upsample_error(error: torch.Tensor) -> torch.Tensor:
            error = rearrange(error, "b s c h w -> (b s) c h w")
            error = F.interpolate(error, size=(h, w), mode="bilinear", align_corners=False)
            return rearrange(error, "(b s) c h w -> b s c h w", b=b)

        def build_history_reprojection_evidence(
            view_idx: int,
            means_state: torch.Tensor,
        ) -> torch.Tensor:
            if view_idx <= 0:
                return torch.zeros(
                    (b, 1, 16, low_h, low_w),
                    device=device,
                    dtype=torch.float32,
                )

            with torch.no_grad():
                current_means = rearrange(
                    means_state[:, view_idx].detach().float(),
                    "b (h w) c -> b c h w",
                    h=h,
                    w=w,
                )
                current_means = F.interpolate(
                    current_means,
                    size=(low_h, low_w),
                    mode="bilinear",
                    align_corners=False,
                )
                current_means = rearrange(
                    current_means, "b c h w -> b h w c"
                )
                current_means_h = torch.cat(
                    [
                        current_means,
                        torch.ones_like(current_means[..., :1]),
                    ],
                    dim=-1,
                )

                current_feature = reprojection_features_low[:, view_idx]
                current_rgb = target_low[:, view_idx].detach().float()
                pair_evidence = []
                pair_scores = []
                pair_masks = []
                pair_max_correlations = []
                pair_confidences = []

                for history_idx in range(view_idx):
                    points_history = torch.einsum(
                        "bij,bhwj->bhwi",
                        source_w2c[:, history_idx],
                        current_means_h,
                    )
                    z_history = points_history[..., 2]
                    z_safe = z_history.clamp_min(1e-4)

                    intrinsic = source_intrinsics[:, history_idx]
                    fx = intrinsic[:, 0, 0].view(b, 1, 1) * float(low_w)
                    fy = intrinsic[:, 1, 1].view(b, 1, 1) * float(low_h)
                    cx = intrinsic[:, 0, 2].view(b, 1, 1) * float(low_w)
                    cy = intrinsic[:, 1, 2].view(b, 1, 1) * float(low_h)
                    u_pixel = fx * points_history[..., 0] / z_safe + cx
                    v_pixel = fy * points_history[..., 1] / z_safe + cy
                    grid = torch.stack(
                        [
                            2.0 * (u_pixel + 0.5) / float(low_w) - 1.0,
                            2.0 * (v_pixel + 0.5) / float(low_h) - 1.0,
                        ],
                        dim=-1,
                    )

                    projection_valid = (
                        (z_history > 1e-4)
                        & (u_pixel >= 0.0)
                        & (u_pixel <= float(low_w - 1))
                        & (v_pixel >= 0.0)
                        & (v_pixel <= float(low_h - 1))
                    ).unsqueeze(1)

                    sampled_rgb_depth = F.grid_sample(
                        history_rgb_depth_low[:, history_idx],
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    )
                    sampled_rgb = sampled_rgb_depth[:, :3]
                    sampled_depth = sampled_rgb_depth[:, 3:4]

                    offset_grid = grid.unsqueeze(3) + reprojection_offsets.view(
                        1, 1, 1, 9, 2
                    )
                    offset_grid = offset_grid.reshape(
                        b, low_h, low_w * 9, 2
                    )
                    sampled_feature = F.grid_sample(
                        reprojection_features_low[:, history_idx],
                        offset_grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    )
                    sampled_feature = sampled_feature.reshape(
                        b,
                        sampled_feature.shape[1],
                        low_h,
                        low_w,
                        9,
                    )
                    sampled_feature = F.normalize(
                        sampled_feature, dim=1, eps=1e-6
                    )
                    correlation = (
                        current_feature.unsqueeze(-1) * sampled_feature
                    ).sum(dim=1)
                    correlation = rearrange(
                        correlation, "b h w n -> b n h w"
                    ).clamp(-1.0, 1.0)

                    relative_depth = (
                        z_history.unsqueeze(1) - sampled_depth
                    ) / sampled_depth.clamp_min(1e-4)
                    relative_depth = relative_depth.clamp(-1.0, 1.0)
                    max_correlation = correlation.max(dim=1, keepdim=True).values
                    depth_consistency = torch.exp(
                        -relative_depth.abs() / 0.15
                    )
                    feature_consistency = (
                        (max_correlation + 1.0) * 0.5
                    ).clamp(1e-3, 1.0)
                    visibility = (
                        projection_valid
                        & (sampled_depth > 1e-4)
                        & (
                            z_history.unsqueeze(1)
                            <= sampled_depth * 1.10
                        )
                    )
                    confidence = depth_consistency * feature_consistency
                    score = torch.log(confidence.clamp_min(1e-6))

                    pair_evidence.append(
                        torch.cat(
                            [
                                current_rgb - sampled_rgb,
                                relative_depth,
                                correlation,
                            ],
                            dim=1,
                        )
                    )
                    pair_scores.append(score)
                    pair_masks.append(visibility)
                    pair_max_correlations.append(max_correlation)
                    pair_confidences.append(confidence)

                pair_evidence = torch.stack(pair_evidence, dim=1)
                pair_scores = torch.stack(pair_scores, dim=1)
                pair_masks = torch.stack(pair_masks, dim=1)
                pair_max_correlations = torch.stack(
                    pair_max_correlations, dim=1
                )
                pair_confidences = torch.stack(pair_confidences, dim=1)

                masked_scores = pair_scores.masked_fill(~pair_masks, -1e4)
                score_max = masked_scores.max(dim=1, keepdim=True).values
                raw_weights = (
                    torch.exp((pair_scores - score_max).clamp(-30.0, 30.0))
                    * pair_masks.to(pair_scores.dtype)
                )
                weights = raw_weights / raw_weights.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-6)
                aggregated = (weights * pair_evidence).sum(dim=1)

                any_visible = pair_masks.any(dim=1)
                max_correlation = pair_max_correlations.masked_fill(
                    ~pair_masks, -1.0
                ).max(dim=1).values
                max_correlation = torch.where(
                    any_visible, max_correlation, torch.zeros_like(max_correlation)
                )
                visible_support = pair_masks.float().mean(dim=1)
                aggregated_confidence = (
                    weights * pair_confidences
                ).sum(dim=1)

                evidence = torch.cat(
                    [
                        aggregated,
                        max_correlation,
                        visible_support,
                        aggregated_confidence,
                    ],
                    dim=1,
                )
                return evidence.unsqueeze(1)

        def build_causal_evidence(
            view_idx: int,
            means_state: torch.Tensor,
            quats_state: torch.Tensor,
            scales_state: torch.Tensor,
            opacities_state: torch.Tensor,
            sh_state: torch.Tensor,
            include_current: bool,
        ):
            # The first iteration observes history only. Later iterations render
            # the updated causal prefix, including the current view, to close the
            # refinement loop without exposing any future views.
            prefix_end = view_idx + 1 if include_current else view_idx
            with torch.no_grad():
                prefix_gaussians = self._build_gaussians_from_raw_state(
                    base_sh_raw[:, :prefix_end].detach(),
                    means_state[:, :prefix_end].detach(),
                    quats_state[:, :prefix_end].detach(),
                    scales_state[:, :prefix_end].detach(),
                    opacities_state[:, :prefix_end].detach(),
                    sh_state[:, :prefix_end].detach(),
                )
                prefix_output = self.decoder.forward(
                    prefix_gaussians,
                    pred_all_extrinsic[:, view_idx : view_idx + 1].detach(),
                    render_intrinsics,
                    near_tensor,
                    far_tensor,
                    (low_h, low_w),
                    "depth",
                )
                render_color = prefix_output.color.detach()
                render_depth = self._normalize_render_depth(
                    prefix_output.depth.detach(), b, 1, low_h, low_w
                )
                render_alpha = self._normalize_render_alpha(
                    prefix_output.alpha.detach(), b, 1, low_h, low_w
                )

            current_target_low = target_low[:, view_idx : view_idx + 1]
            rgb_residual_low = (render_color - current_target_low).to(features.dtype)
            current_depth_low = depth_low[:, view_idx : view_idx + 1].clamp_min(1e-4)
            depth_residual_low = ((render_depth - current_depth_low) / current_depth_low).clamp(-1.0, 1.0)
            depth_residual_low = depth_residual_low.to(features.dtype)
            alpha_low = render_alpha.to(features.dtype)

            rgb_residual = upsample_error(rgb_residual_low)
            depth_residual = upsample_error(depth_residual_low)
            alpha = upsample_error(alpha_low)
            feature_error = self.gs_residual_refiner.encode_feature_error(
                render_color,
                current_target_low,
            )
            feature_error = upsample_error(feature_error)
            reprojection_evidence = build_history_reprojection_evidence(
                view_idx,
                means_state,
            )
            return (
                rgb_residual,
                depth_residual,
                alpha,
                feature_error,
                reprojection_evidence,
            )

        def select_views(state: torch.Tensor, view_indices: list[int]) -> torch.Tensor:
            indices = torch.tensor(view_indices, device=state.device, dtype=torch.long)
            return state.index_select(1, indices)

        def refine_view_batch_once(
            view_indices: list[int],
            means_state: torch.Tensor,
            quats_state: torch.Tensor,
            scales_state: torch.Tensor,
            opacities_state: torch.Tensor,
            sh_state: torch.Tensor,
            evidence: list[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ]
            ],
            refiner_hidden: Optional[torch.Tensor],
            refine_iter: int,
        ):
            rgb_residual = torch.cat([item[0] for item in evidence], dim=1)
            depth_residual = torch.cat([item[1] for item in evidence], dim=1)
            alpha = torch.cat([item[2] for item in evidence], dim=1)
            feature_error = torch.cat([item[3] for item in evidence], dim=1)
            reprojection_evidence = torch.cat(
                [item[4] for item in evidence], dim=1
            )
            error_context = torch.cat(
                [
                    rgb_residual.detach().float(),
                    depth_residual.detach().float(),
                    alpha.detach().float(),
                    feature_error.float(),
                ],
                dim=2,
            )

            current_means = select_views(means_state, view_indices)
            current_quats = select_views(quats_state, view_indices)
            current_scales = select_views(scales_state, view_indices)
            current_opacities = select_views(opacities_state, view_indices)
            current_sh = select_views(sh_state, view_indices)
            current_features = select_views(evidence_features, view_indices)
            current_depth_conf = select_views(depth_conf, view_indices)
            current_depth_uncertainty = select_views(depth_uncertainty, view_indices)
            mean_step_source = select_views(depth, view_indices)
            if cfg.gs_refine_detach_evidence:
                mean_step_source = mean_step_source.detach()
            mean_step = rearrange(mean_step_source.float(), "b s h w c -> b s (h w) c")
            view_gate = torch.ones(
                (b, len(view_indices), 1, h, w),
                device=device,
                dtype=features.dtype,
            )

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
                causal_error_context=error_context,
                reprojection_evidence=reprojection_evidence,
                hidden_state=refiner_hidden,
                iter_idx=refine_iter,
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

            return (
                current_means,
                current_quats,
                current_scales,
                current_opacities,
                current_sh,
                refiner_hidden,
            )

        def replace_view(
            state: torch.Tensor,
            view_idx: int,
            current: torch.Tensor,
        ) -> torch.Tensor:
            return torch.cat(
                [state[:, :view_idx], current, state[:, view_idx + 1 :]],
                dim=1,
            )

        num_refine_iters = max(0, int(cfg.gs_refine_iters))
        if num_refine_iters == 0:
            return encoder_output.gaussians

        if self.training:
            # Full-t synchronous causal refinement. Each round renders the
            # states produced by the previous round, while every view is
            # restricted to its own prefix and all updates remain batched.
            refine_view_indices = list(range(1, s))
            refiner_hidden = None
            for refine_iter in range(num_refine_iters):
                evidence = [
                    build_causal_evidence(
                        view_idx,
                        means_raw,
                        quats_raw,
                        scales_raw,
                        opacities_raw,
                        res_sh_raw,
                        include_current=refine_iter > 0,
                    )
                    for view_idx in refine_view_indices
                ]
                (
                    current_means,
                    current_quats,
                    current_scales,
                    current_opacities,
                    current_sh,
                    refiner_hidden,
                ) = refine_view_batch_once(
                    refine_view_indices,
                    means_raw,
                    quats_raw,
                    scales_raw,
                    opacities_raw,
                    res_sh_raw,
                    evidence,
                    refiner_hidden,
                    refine_iter,
                )

                means_raw = torch.cat([means_raw[:, :1], current_means], dim=1)
                quats_raw = torch.cat([quats_raw[:, :1], current_quats], dim=1)
                scales_raw = torch.cat([scales_raw[:, :1], current_scales], dim=1)
                opacities_raw = torch.cat(
                    [opacities_raw[:, :1], current_opacities], dim=1
                )
                res_sh_raw = torch.cat([res_sh_raw[:, :1], current_sh], dim=1)

            refined_gaussians = self._build_gaussians_from_refine_state(
                refine_info,
                means_raw,
                quats_raw,
                scales_raw,
                opacities_raw,
                res_sh_raw,
            )
        else:
            # Validation/test perform a streaming rollout. Previous views are
            # fully refined and frozen before the newly arrived view runs its
            # own closed-loop iterations.
            for refine_view_idx in range(1, s):
                refiner_hidden = None
                for refine_iter in range(num_refine_iters):
                    evidence = [
                        build_causal_evidence(
                            refine_view_idx,
                            means_raw,
                            quats_raw,
                            scales_raw,
                            opacities_raw,
                            res_sh_raw,
                            include_current=refine_iter > 0,
                        )
                    ]
                    (
                        current_means,
                        current_quats,
                        current_scales,
                        current_opacities,
                        current_sh,
                        refiner_hidden,
                    ) = refine_view_batch_once(
                        [refine_view_idx],
                        means_raw,
                        quats_raw,
                        scales_raw,
                        opacities_raw,
                        res_sh_raw,
                        evidence,
                        refiner_hidden,
                        refine_iter,
                    )

                    means_raw = replace_view(means_raw, refine_view_idx, current_means)
                    quats_raw = replace_view(quats_raw, refine_view_idx, current_quats)
                    scales_raw = replace_view(scales_raw, refine_view_idx, current_scales)
                    opacities_raw = replace_view(
                        opacities_raw, refine_view_idx, current_opacities
                    )
                    res_sh_raw = replace_view(res_sh_raw, refine_view_idx, current_sh)

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
                s - 1, device=device
            )
            encoder_output.infos["gs_refine_reprojection_gate"] = torch.sigmoid(
                self.gs_residual_refiner.reprojection_gate.detach()
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
