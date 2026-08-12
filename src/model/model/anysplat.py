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
    apply_current_gaussian_residual,
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
            if getattr(cfg, "gir_dominant_id_enabled", False) and not getattr(
                cfg, "gir_raster_evidence_enabled", False
            ):
                raise ValueError(
                    "gir_dominant_id_enabled requires gir_raster_evidence_enabled."
                )
            self.gir_renderer = DominantGIRRenderer()
            self.gir_update_head = GIRUpdateHead(
                feature_dim=self.encoder.feature_dim // 2,
                harmonic_dim=self.encoder.nums_sh * 3,
                hidden_dim=cfg.gir_hidden_dim,
                use_raster_evidence=getattr(
                    cfg, "gir_raster_evidence_enabled", False
                ),
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

    @torch.no_grad()
    def _render_old_map_gir_evidence(
        self,
        gir: DominantGIR,
        state: StreamingGaussianState,
        camera_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
        image_shape: tuple[int, int],
        use_dominant_ids: bool,
        min_dominant_weight: float,
        num_top_contributors: int = 1,
    ) -> DominantGIR:
        b = state.batch_size
        h, w = image_shape
        render_output = self.decoder.render_gir_evidence(
            state.gaussians,
            camera_to_world.detach(),
            intrinsics.detach(),
            image_shape,
            num_top_contributors=num_top_contributors,
        )

        render_alpha = torch.nan_to_num(
            render_output.alpha.float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_(0.0, 1.0)
        render_depth = render_output.expected_depth
        render_depth = torch.nan_to_num(
            render_depth.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        render_depth = torch.where(
            render_alpha > 1e-4,
            render_depth.clamp_min(1e-6),
            torch.zeros_like(render_depth),
        )

        render_color = torch.nan_to_num(
            render_output.color.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        gir.rgb = render_color.clamp_(0.0, 1.0).to(gir.rgb.dtype)
        gir.raster_depth = render_depth.to(gir.depth.dtype)
        gir.raster_alpha = render_alpha.to(gir.opacity.dtype)
        contributor_ids = render_output.contributor_ids
        contributor_weights = render_output.contributor_weights
        if contributor_ids is not None and contributor_weights is not None:
            contributor_weights = torch.nan_to_num(
                contributor_weights.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min_(0.0)
            contributor_valid = (
                (contributor_ids >= 0)
                & (contributor_ids < state.num_gaussians)
                & (contributor_weights >= min_dominant_weight)
                & (render_alpha > 1e-4)
            )
            gir.contributor_ids = torch.where(
                contributor_valid,
                contributor_ids,
                torch.full_like(contributor_ids, -1),
            )
            gir.contributor_weights = torch.where(
                contributor_valid,
                contributor_weights,
                torch.zeros_like(contributor_weights),
            )

        if use_dominant_ids:
            ids = render_output.dominant_ids
            weights = torch.nan_to_num(
                render_output.dominant_weights.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min_(0.0)
            valid = (
                (ids >= 0)
                & (ids < state.num_gaussians)
                & (weights[:, 0] >= min_dominant_weight)
                & (render_alpha[:, 0] > 1e-4)
            )
            safe_ids = ids.clamp(min=0, max=max(state.num_gaussians - 1, 0))
            flat_ids = safe_ids.reshape(b, -1)

            def gather_scalar(values: torch.Tensor) -> torch.Tensor:
                gathered = values.gather(1, flat_ids).reshape(b, 1, h, w)
                return torch.where(
                    valid[:, None], gathered, torch.zeros_like(gathered)
                )

            means = state.gaussians.means.float()
            ones = torch.ones(
                (b, state.num_gaussians, 1),
                device=means.device,
                dtype=means.dtype,
            )
            means_h = torch.cat([means, ones], dim=-1)
            world_to_camera = torch.linalg.inv(camera_to_world.detach().float())
            camera_points = torch.einsum("bij,bnj->bni", world_to_camera, means_h)
            dominant_depth = gather_scalar(camera_points[..., 2])
            valid = valid & (dominant_depth[:, 0] > 1e-5)

            gir.indices = torch.where(valid, ids, torch.full_like(ids, -1))
            stable_ids = state.stable_ids.gather(1, flat_ids).reshape(b, h, w)
            gir.stable_ids = torch.where(
                valid, stable_ids, torch.full_like(stable_ids, -1)
            )
            gir.valid = valid[:, None]
            gir.depth = torch.where(
                gir.valid,
                dominant_depth.to(gir.depth.dtype),
                torch.zeros_like(gir.depth),
            )
            gir.opacity = gather_scalar(state.gaussians.opacities).to(
                gir.opacity.dtype
            )
            gir.scale = gather_scalar(state.gaussians.scales.norm(dim=-1)).to(
                gir.scale.dtype
            )
            gir.observation_count = gather_scalar(state.observation_count).to(
                gir.observation_count.dtype
            )
            gir.dominant_weight = torch.where(
                gir.valid,
                weights.to(gir.dominant_weight.dtype),
                torch.zeros_like(gir.dominant_weight),
            )

        return gir

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
        test_add_gate_prune_threshold: float = 0.0,
        test_top1_confidence_mode: str = "inherit",
        test_top1_confidence_floor: float = 0.25,
        test_correspondence_diagnostics: bool = False,
        test_correspondence_topk: int = 8,
        test_old_gs_final_prune_enabled: bool = False,
        test_old_gs_prune_min_candidate_views: int = 2,
        test_old_gs_prune_max_top1_rate: float = 0.0,
        test_old_gs_prune_max_mean_weight: float = 0.01,
        test_old_gs_prune_max_opacity: float = 1.0,
    ) -> Gaussians:
        refine_info = None if encoder_output.infos is None else encoder_output.infos.get("gs_refine")
        if refine_info is None:
            raise RuntimeError("GIR is enabled, but the encoder did not return per-view GS data.")

        cfg = self.encoder.cfg
        prune_threshold = float(test_add_gate_prune_threshold)
        if not 0.0 <= prune_threshold <= 1.0:
            raise ValueError(
                "GIR add-gate prune threshold must be in [0, 1], "
                f"got {prune_threshold}."
            )
        if self.training and prune_threshold > 0.0:
            raise RuntimeError("GIR add-gate pruning is test-only.")
        correspondence_diagnostics = bool(test_correspondence_diagnostics)
        if self.training and correspondence_diagnostics:
            raise RuntimeError("GIR correspondence diagnostics are test-only.")
        correspondence_topk = max(2, min(16, int(test_correspondence_topk)))
        old_gs_final_prune_enabled = bool(test_old_gs_final_prune_enabled)
        if self.training and old_gs_final_prune_enabled:
            raise RuntimeError("Historical GS pruning is test-only.")
        if old_gs_final_prune_enabled and not correspondence_diagnostics:
            raise RuntimeError(
                "Historical GS pruning requires correspondence diagnostics."
            )
        old_gs_prune_min_candidate_views = max(
            1, int(test_old_gs_prune_min_candidate_views)
        )
        old_gs_prune_max_top1_rate = max(
            0.0, min(1.0, float(test_old_gs_prune_max_top1_rate))
        )
        old_gs_prune_max_mean_weight = max(
            0.0, float(test_old_gs_prune_max_mean_weight)
        )
        old_gs_prune_max_opacity = max(
            0.0, min(1.0, float(test_old_gs_prune_max_opacity))
        )
        requested_confidence_mode = str(test_top1_confidence_mode)
        if requested_confidence_mode not in {
            "inherit",
            "none",
            "floor_sqrt",
            "sqrt",
        }:
            raise ValueError(
                "GIR top-1 confidence mode must be one of "
                "inherit, none, floor_sqrt, sqrt; "
                f"got {requested_confidence_mode}."
            )
        if requested_confidence_mode == "inherit":
            top1_confidence_mode = str(
                getattr(cfg, "gir_top1_confidence_mode", "none")
            )
            confidence_floor = float(
                getattr(cfg, "gir_top1_confidence_floor", 0.25)
            )
        else:
            top1_confidence_mode = requested_confidence_mode
            confidence_floor = float(test_top1_confidence_floor)
        if top1_confidence_mode not in {"none", "floor_sqrt", "sqrt"}:
            raise ValueError(
                "Configured GIR top-1 confidence mode must be one of "
                "none, floor_sqrt, sqrt; "
                f"got {top1_confidence_mode}."
            )
        confidence_floor = max(0.0, min(1.0, confidence_floor))
        use_raster_evidence = bool(
            getattr(cfg, "gir_raster_evidence_enabled", False)
        )
        use_dominant_ids = bool(getattr(cfg, "gir_dominant_id_enabled", False))
        if correspondence_diagnostics and not (
            use_raster_evidence and use_dominant_ids
        ):
            raise RuntimeError(
                "GIR correspondence diagnostics require raster evidence and "
                "dominant IDs."
            )
        min_dominant_weight = float(
            max(0.0, getattr(cfg, "gir_dominant_min_weight", 1e-4))
        )
        soft_update_topk = max(
            1, min(16, int(getattr(cfg, "gir_soft_update_topk", 1)))
        )
        if soft_update_topk > 1 and not (
            use_raster_evidence and use_dominant_ids
        ):
            raise RuntimeError(
                "Soft GIR updates require raster evidence and dominant IDs."
            )
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
        history_adapt_losses = []
        history_preserve_losses = []
        history_before_errors = []
        history_after_errors = []
        history_mask_strengths = []
        history_past_before_errors = []
        history_past_after_errors = []
        history_past_degradations = []
        add_suppression_losses = []
        regularization_losses = []
        add_gates = []
        add_targets = []
        covered_add_gates = []
        uncovered_add_gates = []
        supported_add_gates = []
        unsupported_add_gates = []
        effective_new_ratios = []
        new_opacity_mass_ratios = []
        low_add_gate_ratios = {0.1: [], 0.2: []}
        effective_new_threshold_ratios = {
            0.001: [],
            0.003: [],
            0.005: [],
            0.01: [],
        }
        test_pruned_new_ratios = []
        top1_ownership_means = []
        top1_ownership_above_0_1 = []
        top1_ownership_above_0_25 = []
        top1_ownership_above_0_5 = []
        top1_confidence_means = []
        soft_update_coverages = []
        corr_top1_weights = []
        corr_top2_weights = []
        corr_top2_to_top1 = []
        corr_top1_top2_relative_gaps = []
        corr_top2_over_0_5 = []
        corr_top2_over_0_8 = []
        corr_contributor_counts = []
        corr_multi_contributor_ratios = []
        corr_contributor_cap_ratios = []
        corr_matched_old_gs_ratios = []
        corr_pixels_per_matched_gs = []
        corr_matched_gs_ge_4_pixels = []
        corr_match_view_hits = None
        corr_match_pixel_hits = None
        corr_match_opportunities = None
        corr_candidate_view_hits = None
        corr_candidate_weight_sum = None
        corr_candidate_weight_samples = None
        historical_gates = []
        visible_ratios = []
        residual_magnitudes = []
        raster_alpha_means = []
        dominant_weight_means = []
        new_residual_magnitudes = []
        new_residual_gates = []

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
            has_history = state is not None

            if state is None:
                gir = DominantGIR.empty(
                    b,
                    low_h,
                    low_w,
                    features.device,
                    features.dtype,
                )
            else:
                if use_dominant_ids:
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
                if use_raster_evidence:
                    gir = self._render_old_map_gir_evidence(
                        gir,
                        state,
                        pred_all_extrinsic[:, view_idx],
                        intrinsics[:, view_idx],
                        (low_h, low_w),
                        use_dominant_ids,
                        min_dominant_weight,
                        max(
                            soft_update_topk,
                            correspondence_topk
                            if correspondence_diagnostics
                            else 1,
                        ),
                    )

            if use_raster_evidence:
                raster_alpha_means.append(gir.raster_alpha.mean())
            if use_dominant_ids and state is not None:
                valid_count = gir.valid.sum().clamp_min(1)
                dominant_weight_means.append(
                    (gir.dominant_weight * gir.valid).sum() / valid_count
                )

            if correspondence_diagnostics and has_history:
                if (
                    gir.contributor_ids is None
                    or gir.contributor_weights is None
                ):
                    raise RuntimeError(
                        "GIR correspondence diagnostics require contributor IDs "
                        "and weights from the rasterizer."
                    )
                contributor_weights = gir.contributor_weights.detach().float()
                valid_float_corr = gir.valid.detach().float()
                valid_count_corr = valid_float_corr.sum().clamp_min(1.0)
                top1_weight = contributor_weights[:, 0:1]
                top2_weight = contributor_weights[:, 1:2]
                top2_ratio = top2_weight / top1_weight.clamp_min(1e-8)
                significant_count = (
                    contributor_weights >= min_dominant_weight
                ).sum(dim=1, keepdim=True).float()

                def masked_pixel_mean(values: torch.Tensor) -> torch.Tensor:
                    return (
                        values * valid_float_corr
                    ).sum() / valid_count_corr

                corr_top1_weights.append(masked_pixel_mean(top1_weight))
                corr_top2_weights.append(masked_pixel_mean(top2_weight))
                corr_top2_to_top1.append(masked_pixel_mean(top2_ratio))
                corr_top1_top2_relative_gaps.append(
                    masked_pixel_mean(1.0 - top2_ratio)
                )
                corr_top2_over_0_5.append(
                    masked_pixel_mean((top2_ratio >= 0.5).float())
                )
                corr_top2_over_0_8.append(
                    masked_pixel_mean((top2_ratio >= 0.8).float())
                )
                corr_contributor_counts.append(
                    masked_pixel_mean(significant_count)
                )
                corr_multi_contributor_ratios.append(
                    masked_pixel_mean((significant_count >= 2).float())
                )
                corr_contributor_cap_ratios.append(
                    masked_pixel_mean(
                        (significant_count >= correspondence_topk).float()
                    )
                )

                if (
                    corr_match_view_hits is None
                    or corr_match_pixel_hits is None
                    or corr_match_opportunities is None
                    or corr_candidate_view_hits is None
                    or corr_candidate_weight_sum is None
                    or corr_candidate_weight_samples is None
                ):
                    raise RuntimeError(
                        "GIR correspondence counters were not initialized."
                    )
                corr_match_opportunities = corr_match_opportunities + 1
                per_gs_pixel_hits = torch.zeros_like(corr_match_pixel_hits)
                per_gs_candidate_hits = torch.zeros_like(corr_match_pixel_hits)
                per_gs_candidate_weight_sum = torch.zeros_like(
                    corr_match_pixel_hits
                )
                per_gs_candidate_weight_samples = torch.zeros_like(
                    corr_match_pixel_hits
                )
                for batch_idx in range(b):
                    valid_ids = gir.indices[batch_idx][
                        gir.valid[batch_idx, 0]
                    ]
                    if valid_ids.numel() > 0:
                        per_gs_pixel_hits[batch_idx].scatter_add_(
                            0,
                            valid_ids,
                            torch.ones_like(
                                valid_ids,
                                dtype=per_gs_pixel_hits.dtype,
                            ),
                        )
                    contributor_ids = gir.contributor_ids[batch_idx].reshape(-1)
                    contributor_weights = gir.contributor_weights[batch_idx].reshape(
                        -1
                    ).float()
                    contributor_valid = (
                        contributor_weights >= min_dominant_weight
                    ) & (contributor_ids >= 0) & (
                        contributor_ids < state.num_gaussians
                    )
                    contributor_ids = contributor_ids[contributor_valid]
                    contributor_weights = contributor_weights[contributor_valid]
                    if contributor_ids.numel() > 0:
                        per_gs_candidate_hits[batch_idx].scatter_add_(
                            0,
                            contributor_ids,
                            torch.ones_like(
                                contributor_ids,
                                dtype=per_gs_candidate_hits.dtype,
                            ),
                        )
                        per_gs_candidate_weight_sum[batch_idx].scatter_add_(
                            0,
                            contributor_ids,
                            contributor_weights,
                        )
                        per_gs_candidate_weight_samples[batch_idx].scatter_add_(
                            0,
                            contributor_ids,
                            torch.ones_like(contributor_weights),
                        )
                matched_old_gs = per_gs_pixel_hits > 0
                candidate_old_gs = per_gs_candidate_hits > 0
                matched_count = matched_old_gs.sum().clamp_min(1)
                corr_match_pixel_hits = (
                    corr_match_pixel_hits + per_gs_pixel_hits
                )
                corr_match_view_hits = corr_match_view_hits + matched_old_gs
                corr_candidate_view_hits = (
                    corr_candidate_view_hits + candidate_old_gs
                )
                corr_candidate_weight_sum = (
                    corr_candidate_weight_sum + per_gs_candidate_weight_sum
                )
                corr_candidate_weight_samples = (
                    corr_candidate_weight_samples
                    + per_gs_candidate_weight_samples
                )
                corr_matched_old_gs_ratios.append(
                    matched_old_gs.float().mean()
                )
                corr_pixels_per_matched_gs.append(
                    per_gs_pixel_hits.sum() / matched_count
                )
                corr_matched_gs_ge_4_pixels.append(
                    ((per_gs_pixel_hits >= 4) & matched_old_gs).sum()
                    / matched_count
                )

            prediction = self.gir_update_head(
                current_feature,
                current_rgb,
                current_depth,
                current_depth_confidence,
                gir,
            )
            prediction.historical_gate = prediction.historical_gate + float(
                getattr(cfg, "gir_history_gate_bias", -2.0)
            )

            if has_history:
                history_interval = max(
                    1, int(getattr(cfg, "gir_history_loss_interval", 4))
                )
                supervise_history = (
                    (view_idx + 1) % history_interval == 0
                    or view_idx + 1 == source_views
                )
                adapt_weight = max(
                    0.0,
                    float(getattr(cfg, "gir_history_adapt_weight", 0.05)),
                )
                preserve_weight = max(
                    0.0,
                    float(getattr(cfg, "gir_history_preserve_weight", 0.10)),
                )
                replay_indices = []
                history_before_render = None

                if (
                    self.training
                    and supervise_history
                    and preserve_weight > 0
                ):
                    replay_count = min(
                        max(0, int(getattr(cfg, "gir_history_replay_views", 1))),
                        view_idx,
                    )
                    if replay_count > 0:
                        # Replay the source views from the previous TBPTT chunk.
                        # This is deterministic across DDP ranks and strictly causal.
                        replay_start = max(0, view_idx - history_interval)
                        replay_indices = list(
                            range(
                                replay_start,
                                min(view_idx, replay_start + replay_count),
                            )
                        )
                        replay_views = len(replay_indices)
                        with torch.no_grad():
                            history_before_render = self.decoder.forward(
                                state.gaussians,
                                pred_all_extrinsic[:, replay_indices],
                                intrinsics[:, replay_indices],
                                torch.full(
                                    (b, replay_views),
                                    near,
                                    device=features.device,
                                ),
                                torch.full(
                                    (b, replay_views),
                                    far,
                                    device=features.device,
                                ),
                                (low_h, low_w),
                                "depth",
                            )

                evidence_alpha = (
                    gir.raster_alpha if use_raster_evidence else gir.opacity
                ).detach().float()
                valid_float = gir.valid.detach().float()
                valid_count = valid_float.sum().clamp_min(1.0)
                top1_ownership = (
                    gir.dominant_weight.detach().float()
                    / evidence_alpha.clamp_min(1e-6)
                ).clamp(0.0, 1.0)
                if (
                    soft_update_topk > 1
                    and gir.contributor_weights is not None
                ):
                    correspondence_weight = gir.contributor_weights[
                        :, :soft_update_topk
                    ].sum(dim=1, keepdim=True).detach().float()
                    soft_coverage = (
                        correspondence_weight
                        / evidence_alpha.clamp_min(1e-6)
                    ).clamp(0.0, 1.0)
                    soft_update_coverages.append(
                        (soft_coverage * gir.valid.detach().float()).sum()
                        / valid_count
                    )
                top1_ownership_means.append(
                    (top1_ownership * valid_float).sum() / valid_count
                )
                top1_ownership_above_0_1.append(
                    ((top1_ownership > 0.1).float() * valid_float).sum()
                    / valid_count
                )
                top1_ownership_above_0_25.append(
                    ((top1_ownership > 0.25).float() * valid_float).sum()
                    / valid_count
                )
                top1_ownership_above_0_5.append(
                    ((top1_ownership > 0.5).float() * valid_float).sum()
                    / valid_count
                )
                historical_update_confidence = None
                if top1_confidence_mode == "sqrt":
                    historical_update_confidence = top1_ownership.sqrt()
                elif top1_confidence_mode == "floor_sqrt":
                    historical_update_confidence = confidence_floor + (
                        1.0 - confidence_floor
                    ) * top1_ownership.sqrt()
                if historical_update_confidence is None:
                    top1_confidence_means.append(top1_ownership.new_ones(()))
                else:
                    top1_confidence_means.append(
                        (historical_update_confidence * valid_float).sum()
                        / valid_count
                    )

                state = state.update_historical(
                    gir,
                    prediction,
                    pred_all_extrinsic[:, view_idx],
                    update_confidence=historical_update_confidence,
                    num_contributors=soft_update_topk,
                )

                if (
                    self.training
                    and supervise_history
                    and (adapt_weight > 0 or replay_indices)
                ):
                    render_indices = [view_idx] + replay_indices
                    render_views = len(render_indices)
                    history_after_render = self.decoder.forward(
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
                    history_target = F.interpolate(
                        current_rgb.float(),
                        size=(low_h, low_w),
                        mode="bilinear",
                        align_corners=False,
                    ).to(history_after_render.color.dtype)
                    current_depth_low = F.interpolate(
                        current_depth.detach().float(),
                        size=(low_h, low_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    history_depth = (
                        gir.raster_depth if use_raster_evidence else gir.depth
                    ).detach().float()
                    history_alpha = (
                        gir.raster_alpha if use_raster_evidence else gir.opacity
                    ).detach().float()
                    alpha_threshold = min(
                        0.99,
                        max(
                            0.0,
                            float(
                                getattr(cfg, "gir_history_alpha_threshold", 0.10)
                            ),
                        ),
                    )
                    depth_tolerance = max(
                        1e-3,
                        float(
                            getattr(cfg, "gir_history_depth_tolerance", 0.20)
                        ),
                    )
                    alpha_support = (
                        (history_alpha - alpha_threshold)
                        / max(1.0 - alpha_threshold, 1e-3)
                    ).clamp(0.0, 1.0)
                    relative_depth_error = (
                        (history_depth - current_depth_low).abs()
                        / current_depth_low.clamp_min(1e-4)
                    )
                    depth_support = torch.exp(
                        -0.5 * (relative_depth_error / depth_tolerance).square()
                    )
                    depth_valid = (
                        (history_depth > 1e-5) & (current_depth_low > 1e-5)
                    ).to(depth_support.dtype)
                    history_valid = gir.valid.detach().float() * depth_valid
                    history_mask = (
                        history_valid * alpha_support * depth_support
                    ).to(history_after_render.color.dtype)
                    history_normalizer = history_mask.sum().clamp_min(1.0)
                    history_mask_strengths.append(
                        history_mask.sum()
                        / history_valid.sum().clamp_min(1.0).to(history_mask.dtype)
                    )
                    before_error = torch.sqrt(
                        (gir.rgb.to(history_target.dtype) - history_target).square()
                        + 1e-6
                    ).mean(dim=1, keepdim=True)
                    after_error = torch.sqrt(
                        (history_after_render.color[:, 0] - history_target).square()
                        + 1e-6
                    ).mean(dim=1, keepdim=True)
                    history_after = (
                        after_error * history_mask
                    ).sum() / history_normalizer
                    if adapt_weight > 0:
                        history_before_errors.append(
                            (before_error * history_mask).sum()
                            / history_normalizer
                        )
                        history_after_errors.append(history_after.detach())
                        history_adapt_losses.append(history_after)

                    if replay_indices and history_before_render is not None:
                        replay_views = len(replay_indices)
                        replay_target = context_image[:, replay_indices]
                        replay_target = rearrange(
                            replay_target,
                            "b v c h w -> (b v) c h w",
                        )
                        replay_target = F.interpolate(
                            replay_target.float(),
                            size=(low_h, low_w),
                            mode="bilinear",
                            align_corners=False,
                        )
                        replay_target = rearrange(
                            replay_target,
                            "(b v) c h w -> b v c h w",
                            b=b,
                            v=replay_views,
                        ).to(history_after_render.color.dtype)

                        replay_before_color = history_before_render.color
                        replay_after_color = history_after_render.color[:, 1:]
                        replay_before_error = torch.sqrt(
                            (replay_before_color - replay_target).square()
                            + 1e-6
                        ).mean(dim=2, keepdim=True)
                        replay_after_error = torch.sqrt(
                            (replay_after_color - replay_target).square()
                            + 1e-6
                        ).mean(dim=2, keepdim=True)

                        if history_before_render.alpha is None:
                            replay_mask = torch.ones_like(replay_before_error)
                        else:
                            replay_alpha = self._normalize_render_alpha(
                                history_before_render.alpha,
                                b,
                                replay_views,
                                low_h,
                                low_w,
                            )
                            replay_mask = (replay_alpha > 1e-4).to(
                                replay_before_error.dtype
                            )

                        replay_normalizer = replay_mask.sum().clamp_min(1.0)
                        replay_degradation = (
                            replay_after_error - replay_before_error.detach()
                        )
                        preserve_margin = max(
                            0.0,
                            float(
                                getattr(
                                    cfg,
                                    "gir_history_preserve_margin",
                                    0.002,
                                )
                            ),
                        )
                        preserve_penalty = F.relu(
                            replay_degradation - preserve_margin
                        )
                        history_preserve_losses.append(
                            (preserve_penalty * replay_mask).sum()
                            / replay_normalizer
                        )
                        history_past_before_errors.append(
                            (
                                replay_before_error.detach() * replay_mask
                            ).sum()
                            / replay_normalizer
                        )
                        history_past_after_errors.append(
                            (
                                replay_after_error.detach() * replay_mask
                            ).sum()
                            / replay_normalizer
                        )
                        history_past_degradations.append(
                            (
                                replay_degradation.detach() * replay_mask
                            ).sum()
                            / replay_normalizer
                        )

                current_gaussians = apply_current_gaussian_residual(
                    current_gaussians,
                    prediction,
                    pred_all_extrinsic[:, view_idx],
                    current_depth,
                )
                new_residual_energy = (
                    prediction.current_delta_mean_camera.square().sum(
                        dim=1, keepdim=True
                    )
                    + prediction.current_delta_rotation.square().sum(
                        dim=1, keepdim=True
                    )
                    + prediction.current_delta_log_scale.square().sum(
                        dim=1, keepdim=True
                    )
                    + prediction.current_delta_opacity_logit.square()
                    + prediction.current_delta_harmonics.square().mean(
                        dim=1, keepdim=True
                    )
                )
                new_residual_magnitudes.append(new_residual_energy.mean().sqrt())
                new_residual_gates.append(
                    prediction.current_residual_gate.sigmoid().mean()
                )
            else:
                new_residual_energy = prediction.add_logit.new_zeros(())

            coverage = gir.valid.to(prediction.add_logit.dtype)
            visible_ratios.append(coverage.mean())
            if state is None:
                # Keep frame zero identical to the base GS prediction.
                add_gate_low = torch.ones_like(prediction.add_logit) + (
                    0.0 * prediction.add_logit
                )
            else:
                with torch.no_grad():
                    current_depth_low = F.interpolate(
                        current_depth.detach().float(),
                        size=(low_h, low_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    historical_depth = (
                        gir.raster_depth if use_raster_evidence else gir.depth
                    ).detach().float()
                    historical_alpha = (
                        gir.raster_alpha if use_raster_evidence else gir.opacity
                    ).detach().float()

                    depth_tolerance = max(
                        1e-3,
                        float(getattr(cfg, "gir_add_depth_tolerance", 0.15)),
                    )
                    alpha_threshold = max(
                        1e-3,
                        float(getattr(cfg, "gir_add_alpha_threshold", 0.5)),
                    )
                    prior_floor = min(
                        0.49,
                        max(
                            1e-4,
                            float(getattr(cfg, "gir_add_prior_floor", 0.02)),
                        ),
                    )
                    relative_depth_error = (
                        (historical_depth - current_depth_low).abs()
                        / current_depth_low.clamp_min(1e-4)
                    )
                    depth_support = torch.exp(
                        -0.5 * (relative_depth_error / depth_tolerance).square()
                    )
                    alpha_support = (
                        historical_alpha / alpha_threshold
                    ).clamp(0.0, 1.0)
                    valid_depth = (
                        (historical_depth > 1e-5)
                        & (current_depth_low > 1e-5)
                    ).to(alpha_support.dtype)
                    history_support = (
                        coverage.float()
                        * valid_depth
                        * alpha_support
                        * depth_support
                    ).clamp(0.0, 1.0)
                    add_target = (1.0 - history_support).clamp(
                        prior_floor, 1.0 - prior_floor
                    )
                    add_prior = torch.logit(add_target).to(
                        prediction.add_logit.dtype
                    )

                add_gate_low = torch.sigmoid(prediction.add_logit + add_prior)
                add_suppression_losses.append(
                    (add_gate_low.float() - add_target).square().mean()
                )
                add_targets.append(add_target.mean())

                def masked_mean(
                    values: torch.Tensor, mask: torch.Tensor
                ) -> torch.Tensor:
                    mask = mask.to(values.dtype)
                    return (values * mask).sum() / mask.sum().clamp_min(1.0)

                supported = (history_support >= 0.5).to(add_gate_low.dtype)
                unsupported = coverage * (1.0 - supported)
                covered_add_gates.append(masked_mean(add_gate_low, coverage))
                uncovered_add_gates.append(
                    masked_mean(add_gate_low, 1.0 - coverage)
                )
                supported_add_gates.append(masked_mean(add_gate_low, supported))
                unsupported_add_gates.append(
                    masked_mean(add_gate_low, unsupported)
                )
            add_gate = F.interpolate(
                add_gate_low,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            add_gates.append(add_gate_low.mean())
            if has_history:
                effective_opacity = current_gaussians.opacities * add_gate.reshape(
                    b, gaussians_per_view
                ).to(current_gaussians.opacities.dtype)
                effective_new_ratios.append(
                    (effective_opacity > cfg.opacity_threshold).float().mean()
                )
                if not self.training:
                    original_opacity_float = current_gaussians.opacities.float()
                    effective_opacity_float = effective_opacity.float()
                    new_opacity_mass_ratios.append(
                        effective_opacity_float.sum()
                        / original_opacity_float.sum().clamp_min(1e-8)
                    )
                    for threshold, ratios in low_add_gate_ratios.items():
                        ratios.append((add_gate < threshold).float().mean())
                    for threshold, ratios in effective_new_threshold_ratios.items():
                        ratios.append(
                            (effective_opacity_float > threshold).float().mean()
                        )
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
            regularization_losses.append(
                (residual_energy * coverage).sum() / valid_normalizer
                + new_residual_energy.mean()
            )

            if state is None:
                state = StreamingGaussianState.from_current(
                    current_gaussians,
                    add_gate,
                )
                if correspondence_diagnostics:
                    counter_shape = (b, state.num_gaussians)
                    corr_match_view_hits = torch.zeros(
                        counter_shape,
                        device=features.device,
                        dtype=torch.float32,
                    )
                    corr_match_pixel_hits = torch.zeros_like(
                        corr_match_view_hits
                    )
                    corr_match_opportunities = torch.zeros_like(
                        corr_match_view_hits
                    )
                    corr_candidate_view_hits = torch.zeros_like(
                        corr_match_view_hits
                    )
                    corr_candidate_weight_sum = torch.zeros_like(
                        corr_match_view_hits
                    )
                    corr_candidate_weight_samples = torch.zeros_like(
                        corr_match_view_hits
                    )
            else:
                previous_gaussian_count = state.num_gaussians
                if not self.training:
                    test_pruned_new_ratios.append(
                        (add_gate < prune_threshold).float().mean()
                        if prune_threshold > 0.0
                        else add_gate.new_zeros(())
                    )
                state = state.append(
                    current_gaussians,
                    add_gate,
                    prune_threshold=prune_threshold,
                )
                if correspondence_diagnostics:
                    added_gaussians = state.num_gaussians - previous_gaussian_count
                    if added_gaussians < 0:
                        raise RuntimeError(
                            "GIR append unexpectedly reduced the map size."
                        )
                    if added_gaussians > 0:
                        counter_padding = torch.zeros(
                            (b, added_gaussians),
                            device=features.device,
                            dtype=torch.float32,
                        )
                        corr_match_view_hits = torch.cat(
                            [corr_match_view_hits, counter_padding], dim=1
                        )
                        corr_match_pixel_hits = torch.cat(
                            [corr_match_pixel_hits, counter_padding], dim=1
                        )
                        corr_match_opportunities = torch.cat(
                            [corr_match_opportunities, counter_padding], dim=1
                        )
                        corr_candidate_view_hits = torch.cat(
                            [corr_candidate_view_hits, counter_padding], dim=1
                        )
                        corr_candidate_weight_sum = torch.cat(
                            [corr_candidate_weight_sum, counter_padding], dim=1
                        )
                        corr_candidate_weight_samples = torch.cat(
                            [
                                corr_candidate_weight_samples,
                                counter_padding,
                            ],
                            dim=1,
                        )

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

        map_gaussians_before_old_prune = state.num_gaussians
        old_gs_prune_stats = None
        if old_gs_final_prune_enabled:
            if b != 1:
                raise RuntimeError(
                    "Test-only historical GS pruning currently requires batch size 1."
                )
            if (
                corr_candidate_view_hits is None
                or corr_candidate_weight_sum is None
                or corr_candidate_weight_samples is None
                or corr_match_view_hits is None
            ):
                raise RuntimeError(
                    "Historical GS pruning requires completed correspondence counters."
                )

            candidate_views = corr_candidate_view_hits[0]
            top1_views = corr_match_view_hits[0]
            mean_candidate_weight = corr_candidate_weight_sum[0] / (
                corr_candidate_weight_samples[0].clamp_min(1.0)
            )
            candidate_eligible = (
                candidate_views >= old_gs_prune_min_candidate_views
            )
            top1_rate = top1_views / candidate_views.clamp_min(1.0)
            opacity = state.gaussians.opacities[0].detach().float()
            prune_mask = (
                candidate_eligible
                & (top1_rate <= old_gs_prune_max_top1_rate)
                & (mean_candidate_weight <= old_gs_prune_max_mean_weight)
                & (opacity <= old_gs_prune_max_opacity)
            )
            keep_mask = ~prune_mask
            safeguard_kept = False
            if not keep_mask.any():
                safeguard_index = opacity.argmax()
                keep_mask[safeguard_index] = True
                prune_mask[safeguard_index] = False
                safeguard_kept = True

            map_count_before = state.num_gaussians
            removed_opacity_mass = opacity[prune_mask].sum()
            total_opacity_mass = opacity.sum().clamp_min(1e-8)
            state = state.select(keep_mask)
            old_gs_prune_stats = {
                "gir_test_old_gs_prune_enabled": torch.tensor(
                    1.0, device=features.device
                ),
                "gir_test_old_gs_prune_min_candidate_views": torch.tensor(
                    old_gs_prune_min_candidate_views,
                    device=features.device,
                    dtype=torch.float32,
                ),
                "gir_test_old_gs_prune_max_top1_rate": torch.tensor(
                    old_gs_prune_max_top1_rate, device=features.device
                ),
                "gir_test_old_gs_prune_max_mean_weight": torch.tensor(
                    old_gs_prune_max_mean_weight, device=features.device
                ),
                "gir_test_old_gs_prune_max_opacity": torch.tensor(
                    old_gs_prune_max_opacity, device=features.device
                ),
                "gir_test_old_gs_prune_map_before": torch.tensor(
                    map_count_before,
                    device=features.device,
                    dtype=torch.float32,
                ),
                "gir_test_old_gs_prune_map_after": torch.tensor(
                    state.num_gaussians,
                    device=features.device,
                    dtype=torch.float32,
                ),
                "gir_test_old_gs_prune_ratio": prune_mask.float().mean(),
                "gir_test_old_gs_prune_candidate_eligible_ratio": (
                    candidate_eligible.float().mean()
                ),
                "gir_test_old_gs_prune_removed_opacity_mass_ratio": (
                    removed_opacity_mass / total_opacity_mass
                ),
                "gir_test_old_gs_prune_safeguard_kept": torch.tensor(
                    float(safeguard_kept), device=features.device
                ),
            }

        if encoder_output.infos is not None:
            encoder_output.infos.pop("gs_refine", None)
            encoder_output.infos["gir_history_views"] = torch.tensor(
                max(source_views - 1, 0), device=features.device
            )
            encoder_output.infos["gir_map_gaussians"] = torch.tensor(
                state.num_gaussians, device=features.device
            )
            if old_gs_prune_stats is not None:
                encoder_output.infos.update(old_gs_prune_stats)
            if test_pruned_new_ratios:
                unpruned_map_gaussians = source_views * gaussians_per_view
                encoder_output.infos["gir_test_prune_threshold"] = torch.tensor(
                    prune_threshold,
                    device=features.device,
                )
                encoder_output.infos["gir_test_pruned_new_ratio"] = torch.stack(
                    test_pruned_new_ratios
                ).mean()
                encoder_output.infos["gir_test_map_reduction_ratio"] = torch.tensor(
                    1.0
                    - map_gaussians_before_old_prune
                    / max(unpruned_map_gaussians, 1),
                    device=features.device,
                )
            if top1_ownership_means:
                encoder_output.infos[
                    "gir_top1_confidence_mode_code"
                ] = torch.tensor(
                    {"none": 0.0, "floor_sqrt": 1.0, "sqrt": 2.0}[
                        top1_confidence_mode
                    ],
                    device=features.device,
                )
                encoder_output.infos[
                    "gir_top1_confidence_floor"
                ] = torch.tensor(confidence_floor, device=features.device)
                encoder_output.infos["gir_top1_ownership_mean"] = torch.stack(
                    top1_ownership_means
                ).mean()
                encoder_output.infos[
                    "gir_top1_ownership_above_0_1_ratio"
                ] = torch.stack(top1_ownership_above_0_1).mean()
                encoder_output.infos[
                    "gir_top1_ownership_above_0_25_ratio"
                ] = torch.stack(top1_ownership_above_0_25).mean()
                encoder_output.infos[
                    "gir_top1_ownership_above_0_5_ratio"
                ] = torch.stack(top1_ownership_above_0_5).mean()
                encoder_output.infos["gir_top1_confidence_mean"] = torch.stack(
                    top1_confidence_means
                ).mean()
            if soft_update_coverages:
                encoder_output.infos["gir_soft_update_topk"] = torch.tensor(
                    soft_update_topk,
                    device=features.device,
                    dtype=torch.float32,
                )
                encoder_output.infos[
                    "gir_soft_update_coverage_mean"
                ] = torch.stack(soft_update_coverages).mean()
            if correspondence_diagnostics and corr_top1_weights:
                encoder_output.infos[
                    "gir_test_corr_top1_weight_mean"
                ] = torch.stack(corr_top1_weights).mean()
                encoder_output.infos[
                    "gir_test_corr_top2_weight_mean"
                ] = torch.stack(corr_top2_weights).mean()
                encoder_output.infos[
                    "gir_test_corr_top2_to_top1_mean"
                ] = torch.stack(corr_top2_to_top1).mean()
                encoder_output.infos[
                    "gir_test_corr_top1_top2_relative_gap_mean"
                ] = torch.stack(corr_top1_top2_relative_gaps).mean()
                encoder_output.infos[
                    "gir_test_corr_top2_over_0_5_ratio"
                ] = torch.stack(corr_top2_over_0_5).mean()
                encoder_output.infos[
                    "gir_test_corr_top2_over_0_8_ratio"
                ] = torch.stack(corr_top2_over_0_8).mean()
                encoder_output.infos[
                    "gir_test_corr_significant_contributors_mean"
                ] = torch.stack(corr_contributor_counts).mean()
                encoder_output.infos[
                    "gir_test_corr_multi_contributor_pixel_ratio"
                ] = torch.stack(corr_multi_contributor_ratios).mean()
                encoder_output.infos[
                    "gir_test_corr_contributor_cap_ratio"
                ] = torch.stack(corr_contributor_cap_ratios).mean()
                encoder_output.infos[
                    "gir_test_corr_matched_old_gs_per_view_ratio"
                ] = torch.stack(corr_matched_old_gs_ratios).mean()
                encoder_output.infos[
                    "gir_test_corr_pixels_per_matched_gs"
                ] = torch.stack(corr_pixels_per_matched_gs).mean()
                encoder_output.infos[
                    "gir_test_corr_matched_gs_ge_4_pixels_ratio"
                ] = torch.stack(corr_matched_gs_ge_4_pixels).mean()

                if (
                    corr_match_view_hits is None
                    or corr_match_pixel_hits is None
                    or corr_match_opportunities is None
                    or corr_candidate_view_hits is None
                ):
                    raise RuntimeError(
                        "GIR correspondence counters are missing at rollout end."
                    )
                eligible = corr_match_opportunities > 0
                eligible_count = eligible.sum().clamp_min(1)
                long_term = corr_match_opportunities >= 2
                long_term_count = long_term.sum().clamp_min(1)
                candidate_visible = corr_candidate_view_hits > 0
                candidate_visible_count = candidate_visible.sum().clamp_min(1)
                encoder_output.infos[
                    "gir_test_corr_old_gs_with_future_view_ratio"
                ] = eligible.float().mean()
                encoder_output.infos[
                    "gir_test_corr_never_top1_after_future_view_ratio"
                ] = (
                    ((corr_match_view_hits == 0) & eligible).sum()
                    / eligible_count
                )
                encoder_output.infos[
                    "gir_test_corr_never_top1_after_2_future_views_ratio"
                ] = (
                    ((corr_match_view_hits == 0) & long_term).sum()
                    / long_term_count
                )
                encoder_output.infos[
                    "gir_test_corr_old_gs_future_view_top1_rate"
                ] = (
                    (corr_match_view_hits / corr_match_opportunities.clamp_min(1))
                    * eligible
                ).sum() / eligible_count
                encoder_output.infos[
                    "gir_test_corr_old_gs_top1_pixels_per_future_view"
                ] = (
                    (corr_match_pixel_hits / corr_match_opportunities.clamp_min(1))
                    * eligible
                ).sum() / eligible_count
                encoder_output.infos[
                    "gir_test_corr_old_gs_candidate_visible_ratio"
                ] = candidate_visible.sum() / eligible_count
                encoder_output.infos[
                    "gir_test_corr_candidate_never_top1_ratio"
                ] = (
                    ((corr_match_view_hits == 0) & candidate_visible).sum()
                    / candidate_visible_count
                )
                encoder_output.infos[
                    "gir_test_corr_top1_given_candidate_view_rate"
                ] = (
                    (
                        corr_match_view_hits
                        / corr_candidate_view_hits.clamp_min(1)
                    )
                    * candidate_visible
                ).sum() / candidate_visible_count
                encoder_output.infos[
                    "gir_test_corr_topk"
                ] = torch.tensor(
                    correspondence_topk,
                    device=features.device,
                    dtype=torch.float32,
                )
            encoder_output.infos["gir_add_gate"] = torch.stack(add_gates).mean()
            if add_suppression_losses:
                encoder_output.infos["gir_add_loss"] = torch.stack(
                    add_suppression_losses
                ).mean()
                encoder_output.infos["gir_add_target"] = torch.stack(
                    add_targets
                ).mean()
                encoder_output.infos["gir_add_gate_covered"] = torch.stack(
                    covered_add_gates
                ).mean()
                encoder_output.infos["gir_add_gate_uncovered"] = torch.stack(
                    uncovered_add_gates
                ).mean()
                encoder_output.infos["gir_add_gate_supported"] = torch.stack(
                    supported_add_gates
                ).mean()
                encoder_output.infos["gir_add_gate_unsupported"] = torch.stack(
                    unsupported_add_gates
                ).mean()
                encoder_output.infos["gir_effective_new_ratio"] = torch.stack(
                    effective_new_ratios
                ).mean()
                if new_opacity_mass_ratios:
                    encoder_output.infos[
                        "gir_new_opacity_mass_ratio"
                    ] = torch.stack(new_opacity_mass_ratios).mean()
                    for threshold, ratios in low_add_gate_ratios.items():
                        suffix = str(threshold).replace(".", "_")
                        encoder_output.infos[
                            f"gir_add_gate_below_{suffix}_ratio"
                        ] = torch.stack(ratios).mean()
                    for threshold, ratios in effective_new_threshold_ratios.items():
                        suffix = str(threshold).replace(".", "_")
                        encoder_output.infos[
                            f"gir_effective_new_above_{suffix}_ratio"
                        ] = torch.stack(ratios).mean()
            encoder_output.infos["gir_historical_gate"] = torch.stack(
                historical_gates
            ).mean()
            encoder_output.infos["gir_visible_ratio"] = torch.stack(
                visible_ratios
            ).mean()
            encoder_output.infos["gir_residual_magnitude"] = torch.stack(
                residual_magnitudes
            ).mean()
            if raster_alpha_means:
                encoder_output.infos["gir_raster_alpha"] = torch.stack(
                    raster_alpha_means
                ).mean()
            if dominant_weight_means:
                encoder_output.infos["gir_dominant_weight"] = torch.stack(
                    dominant_weight_means
                ).mean()
            if new_residual_magnitudes:
                encoder_output.infos["gir_new_residual_magnitude"] = torch.stack(
                    new_residual_magnitudes
                ).mean()
                encoder_output.infos["gir_new_residual_gate"] = torch.stack(
                    new_residual_gates
                ).mean()
            if history_adapt_losses:
                encoder_output.infos["gir_history_adapt_loss"] = torch.stack(
                    history_adapt_losses
                ).mean()
                encoder_output.infos["gir_history_before_error"] = torch.stack(
                    history_before_errors
                ).mean()
                encoder_output.infos["gir_history_after_error"] = torch.stack(
                    history_after_errors
                ).mean()
                encoder_output.infos["gir_history_mask_strength"] = torch.stack(
                    history_mask_strengths
                ).mean()
            if history_preserve_losses:
                encoder_output.infos["gir_history_preserve_loss"] = torch.stack(
                    history_preserve_losses
                ).mean()
                encoder_output.infos["gir_history_past_before_error"] = torch.stack(
                    history_past_before_errors
                ).mean()
                encoder_output.infos["gir_history_past_after_error"] = torch.stack(
                    history_past_after_errors
                ).mean()
                encoder_output.infos["gir_history_past_degradation"] = torch.stack(
                    history_past_degradations
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
        test_add_gate_prune_threshold: float = 0.0,
        test_top1_confidence_mode: str = "inherit",
        test_top1_confidence_floor: float = 0.25,
        test_correspondence_diagnostics: bool = False,
        test_correspondence_topk: int = 8,
        test_old_gs_final_prune_enabled: bool = False,
        test_old_gs_prune_min_candidate_views: int = 2,
        test_old_gs_prune_max_top1_rate: float = 0.0,
        test_old_gs_prune_max_mean_weight: float = 0.01,
        test_old_gs_prune_max_opacity: float = 1.0,
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
                test_add_gate_prune_threshold,
                test_top1_confidence_mode,
                test_top1_confidence_floor,
                test_correspondence_diagnostics,
                test_correspondence_topk,
                test_old_gs_final_prune_enabled,
                test_old_gs_prune_min_candidate_views,
                test_old_gs_prune_max_top1_rate,
                test_old_gs_prune_max_mean_weight,
                test_old_gs_prune_max_opacity,
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
