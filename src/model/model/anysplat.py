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
from dataclasses import dataclass, replace

from src.model.types import Gaussians
from src.model.streaming_gir import (
    apply_projected_new_gaussian_residual,
    build_mean_gradient_evidence,
    build_opacity_gradient_evidence,
    build_point_map_append_mask,
    build_provisional_gaussians,
    build_projected_gaussian_evidence,
    build_projected_new_gaussian_evidence,
    compute_old_opacity_gradient_effective_stats,
    DominantGIR,
    DominantGIRRenderer,
    gate_front_conflict_decay,
    GIRUpdateHead,
    MEAN_GRADIENT_STAT_KEYS,
    OLD_OPACITY_GRADIENT_EFFECTIVE_STAT_KEYS,
    OPACITY_GRADIENT_STAT_KEYS,
    ProjectedGSResidualHead,
    select_current_and_recent_replay_gradients,
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
from src.misc.cuda_memory import log_cuda_memory


NEW_RESIDUAL_STAT_KEYS = (
    "candidate_ratio",
    "gate_mean",
    "raw_mean",
    "mean_shift_mean",
    "raw_rotation",
    "rotation_shift_mean",
    "raw_log_scale",
    "scale_relative_shift_mean",
    "raw_opacity_logit",
    "opacity_shift_mean",
    "raw_harmonics",
    "harmonics_shift_mean",
    "regularization",
    *OPACITY_GRADIENT_STAT_KEYS,
    *MEAN_GRADIENT_STAT_KEYS,
)


def _group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def _causal_replay_indices(view_idx: int, max_views: int, name: str) -> list[int]:
    """Select prior views, with -1 meaning the complete causal history."""
    max_views = int(max_views)
    if max_views < -1:
        raise ValueError(
            f"{name} must be -1, 0, or a positive integer; got {max_views}."
        )
    if max_views == 0 or view_idx <= 0:
        return []
    replay_start = 0 if max_views == -1 else max(0, view_idx - max_views)
    return list(range(replay_start, view_idx))


def _bound_tanh_input(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Encode a scaled tanh bound for the activation in update_historical."""
    if scale <= 0:
        return value * 0.0
    bounded = (torch.tanh(value) * min(float(scale), 0.999)).clamp(
        -0.999, 0.999
    )
    return torch.atanh(bounded)


def _gir_debug_tensor(name: str, value: torch.Tensor) -> None:
    """Print compact finite/range statistics for a GIR tensor."""
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    ):
        return
    if value is None:
        print(f"[GIR DEBUG] {name}: None", flush=True)
        return
    detached = value.detach().float()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    total_count = detached.numel()
    if finite_count == 0:
        print(
            f"[GIR DEBUG] {name}: finite=0/{total_count}, all_nonfinite",
            flush=True,
        )
        return
    finite_values = detached[finite]
    print(
        f"[GIR DEBUG] {name}: finite={finite_count}/{total_count} "
        f"min={finite_values.min().item():.6e} "
        f"max={finite_values.max().item():.6e} "
        f"absmax={finite_values.abs().max().item():.6e} "
        f"mean={finite_values.mean().item():.6e}",
        flush=True,
    )


def _mask_gir(gir: DominantGIR, mask: torch.Tensor) -> DominantGIR:
    """Keep GIR evidence only at pixels selected for historical updates."""
    mask = mask.bool()
    mask_hw = mask[:, 0]
    return replace(
        gir,
        indices=torch.where(mask_hw, gir.indices, torch.full_like(gir.indices, -1)),
        stable_ids=torch.where(
            mask_hw, gir.stable_ids, torch.full_like(gir.stable_ids, -1)
        ),
        valid=gir.valid & mask,
        dominant_weight=torch.where(
            mask, gir.dominant_weight, torch.zeros_like(gir.dominant_weight)
        ),
        depth=torch.where(mask, gir.depth, torch.zeros_like(gir.depth)),
        opacity=torch.where(mask, gir.opacity, torch.zeros_like(gir.opacity)),
        scale=torch.where(mask, gir.scale, torch.zeros_like(gir.scale)),
        observation_count=torch.where(
            mask, gir.observation_count, torch.zeros_like(gir.observation_count)
        ),
        raster_depth=torch.where(
            mask, gir.raster_depth, torch.zeros_like(gir.raster_depth)
        ),
        raster_alpha=torch.where(
            mask, gir.raster_alpha, torch.zeros_like(gir.raster_alpha)
        ),
        contributor_ids=(
            None
            if gir.contributor_ids is None
            else torch.where(
                # contributor_ids is laid out as [B, K, H, W]. The mask is
                # already [B, 1, H, W], so adding another axis would create
                # [B, 1, K, H, W] through broadcasting and collapse the
                # apparent contributor dimension to 1 downstream.
                mask,
                gir.contributor_ids,
                torch.full_like(gir.contributor_ids, -1),
            )
        ),
        contributor_weights=(
            None
            if gir.contributor_weights is None
            else torch.where(
                mask,
                gir.contributor_weights,
                torch.zeros_like(gir.contributor_weights),
            )
        ),
        contributor_depth=(
            None
            if gir.contributor_depth is None
            else torch.where(
                mask,
                gir.contributor_depth,
                torch.zeros_like(gir.contributor_depth),
            )
        ),
        contributor_opacity=(
            None
            if gir.contributor_opacity is None
            else torch.where(
                mask,
                gir.contributor_opacity,
                torch.zeros_like(gir.contributor_opacity),
            )
        ),
        contributor_scale=(
            None
            if gir.contributor_scale is None
            else torch.where(
                mask,
                gir.contributor_scale,
                torch.zeros_like(gir.contributor_scale),
            )
        ),
        contributor_observation_count=(
            None
            if gir.contributor_observation_count is None
            else torch.where(
                mask,
                gir.contributor_observation_count,
                torch.zeros_like(gir.contributor_observation_count),
            )
        ),
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
        self.build_gir()

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

    def build_gir(self):
        cfg = self.encoder.cfg
        # GIR is the only streaming Gaussian refinement path.
        self.gir_renderer = None
        self.gir_update_head = None
        self.gir_projected_update_head = None
        if not getattr(cfg, "gir_enabled", False):
            return
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
        old_update_mode = str(
            getattr(cfg, "gir_old_update_mode", "pixel_topk")
        ).lower()
        if old_update_mode not in {"pixel_topk", "projected_shared"}:
            raise ValueError(
                "gir_old_update_mode must be pixel_topk or projected_shared; "
                f"got {old_update_mode!r}."
            )
        new_residual_enabled = bool(
            getattr(cfg, "gir_new_residual_enabled", False)
        )
        new_opacity_gradient_enabled = bool(
            getattr(cfg, "gir_new_opacity_gradient_enabled", False)
        )
        if new_opacity_gradient_enabled and not new_residual_enabled:
            raise ValueError(
                "New-GS opacity-gradient conditioning requires "
                "gir_new_residual_enabled=true."
            )
        new_mean_gradient_enabled = bool(
            getattr(cfg, "gir_new_mean_gradient_enabled", False)
        )
        if new_mean_gradient_enabled and not new_residual_enabled:
            raise ValueError(
                "New-GS mean-gradient conditioning requires "
                "gir_new_residual_enabled=true."
            )
        if old_update_mode == "projected_shared" or new_residual_enabled:
            if bool(getattr(cfg, "gir_old_decay_enabled", False)):
                if old_update_mode == "projected_shared":
                    raise ValueError(
                        "projected_shared includes front-surface opacity decay; "
                        "set gir_old_decay_enabled=false to avoid duplicate updates."
                    )
            self.gir_projected_update_head = ProjectedGSResidualHead(
                feature_dim=self.encoder.feature_dim // 2,
                harmonic_dim=self.encoder.nums_sh * 3,
                hidden_dim=int(getattr(cfg, "gir_projected_hidden_dim", 128)),
                chunk_size=int(getattr(cfg, "gir_projected_chunk_size", 65536)),
            )
        residual_topk = max(1, int(getattr(cfg, "gir_residual_topk", 1)))
        if old_update_mode == "pixel_topk" and residual_topk > 1 and bool(
            getattr(cfg, "gir_residual_independent_heads", True)
        ):
            self.gir_update_head.configure_historical_prediction_heads(
                residual_topk
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

        if (
            render_output.contributor_ids is not None
            and render_output.contributor_weights is not None
        ):
            contributor_ids = render_output.contributor_ids.long()
            contributor_weights = torch.nan_to_num(
                render_output.contributor_weights.float(),
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
            safe_contributor_ids = contributor_ids.clamp(
                min=0, max=max(state.num_gaussians - 1, 0)
            )
            flat_contributor_ids = safe_contributor_ids.reshape(b, -1)

            def gather_contributor_scalar(values: torch.Tensor) -> torch.Tensor:
                return values.gather(1, flat_contributor_ids).reshape_as(
                    safe_contributor_ids
                )

            means = state.gaussians.means.detach().float()
            ones = torch.ones(
                (b, state.num_gaussians, 1),
                device=means.device,
                dtype=means.dtype,
            )
            means_h = torch.cat([means, ones], dim=-1)
            world_to_camera = torch.linalg.inv(camera_to_world.detach().float())
            camera_points = torch.einsum("bij,bnj->bni", world_to_camera, means_h)
            contributor_depth = gather_contributor_scalar(camera_points[..., 2])
            contributor_valid = contributor_valid & (contributor_depth > 1e-5)
            gir.contributor_ids = torch.where(
                contributor_valid,
                contributor_ids,
                torch.full_like(contributor_ids, -1),
            )
            gir.contributor_weights = torch.where(
                contributor_valid,
                contributor_weights.to(gir.dominant_weight.dtype),
                torch.zeros_like(contributor_weights).to(
                    gir.dominant_weight.dtype
                ),
            )
            gir.contributor_depth = torch.where(
                contributor_valid,
                contributor_depth.to(gir.depth.dtype),
                torch.zeros_like(contributor_depth).to(gir.depth.dtype),
            )
            gir.contributor_opacity = torch.where(
                contributor_valid,
                gather_contributor_scalar(state.gaussians.opacities).to(
                    gir.opacity.dtype
                ),
                torch.zeros_like(contributor_weights).to(gir.opacity.dtype),
            )
            gir.contributor_scale = torch.where(
                contributor_valid,
                gather_contributor_scalar(
                    state.gaussians.scales.norm(dim=-1)
                ).to(gir.scale.dtype),
                torch.zeros_like(contributor_weights).to(gir.scale.dtype),
            )
            gir.contributor_observation_count = torch.where(
                contributor_valid,
                gather_contributor_scalar(state.observation_count).to(
                    gir.observation_count.dtype
                ),
                torch.zeros_like(contributor_weights).to(
                    gir.observation_count.dtype
                ),
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

    @staticmethod
    def _project_selected_points_for_debug(
        point_map: torch.Tensor,
        selected_mask: torch.Tensor,
        camera_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Project a GS-space boolean mask into the current image for debugging."""
        height, width = image_shape
        batch_size, point_count, _ = point_map.shape
        if selected_mask.shape != (batch_size, point_count):
            raise ValueError(
                "Debug point mask must match the point map; got "
                f"{tuple(selected_mask.shape)} and {(batch_size, point_count)}."
            )

        projected = torch.zeros(
            (batch_size, 1, height, width),
            device=point_map.device,
            dtype=torch.float32,
        )
        world_to_camera = torch.linalg.inv(camera_to_world.detach().float())
        for batch_idx in range(batch_size):
            points = point_map[batch_idx, selected_mask[batch_idx]].detach().float()
            if points.numel() == 0:
                continue
            ones = torch.ones(
                (points.shape[0], 1), device=points.device, dtype=points.dtype
            )
            camera_points = (
                world_to_camera[batch_idx]
                @ torch.cat([points, ones], dim=-1).transpose(0, 1)
            ).transpose(0, 1)[..., :3]
            z = camera_points[:, 2]
            z_safe = z.clamp_min(1e-6)
            pixel_x = (
                intrinsics[batch_idx, 0, 0].detach().float()
                * width
                * camera_points[:, 0]
                / z_safe
                + intrinsics[batch_idx, 0, 2].detach().float() * width
            ).round().long()
            pixel_y = (
                intrinsics[batch_idx, 1, 1].detach().float()
                * height
                * camera_points[:, 1]
                / z_safe
                + intrinsics[batch_idx, 1, 2].detach().float() * height
            ).round().long()
            valid = (
                torch.isfinite(camera_points).all(dim=-1)
                & (z > 1e-5)
                & (pixel_x >= 0)
                & (pixel_x < width)
                & (pixel_y >= 0)
                & (pixel_y < height)
            )
            linear_index = pixel_y[valid] * width + pixel_x[valid]
            projected[batch_idx, 0].view(-1).scatter_add_(
                0,
                linear_index,
                torch.ones_like(linear_index, dtype=projected.dtype),
            )

        # A one-pixel projected center is hard to inspect in a large panel.
        return F.max_pool2d(
            (projected > 0).float(), kernel_size=3, stride=1, padding=1
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
        global_step: int = 0,
        test_add_gate_prune_threshold: float = 0.0,
        test_top1_confidence_mode: str = "inherit",
        test_top1_confidence_floor: float = 0.25,
        return_debug_intermediates: bool = False,
        debug_view_stride: int = 1,
    ) -> Gaussians:
        refine_info = None if encoder_output.infos is None else encoder_output.infos.get("gir")
        if refine_info is None:
            raise RuntimeError("GIR is enabled, but the encoder did not return per-view GS data.")

        debug_view_stride = int(debug_view_stride)
        if debug_view_stride < 1:
            raise ValueError(
                "GIR debug view stride must be at least 1, "
                f"got {debug_view_stride}."
            )

        cfg = self.encoder.cfg
        prune_threshold = float(test_add_gate_prune_threshold)
        if not 0.0 <= prune_threshold <= 1.0:
            raise ValueError(
                "GIR add-gate prune threshold must be in [0, 1], "
                f"got {prune_threshold}."
            )
        if self.training and prune_threshold > 0.0:
            raise RuntimeError("GIR add-gate pruning is test-only.")
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
        old_residual_enabled = bool(
            getattr(cfg, "gir_old_residual_enabled", True)
        )
        old_update_mode = str(
            getattr(cfg, "gir_old_update_mode", "pixel_topk")
        ).lower()
        if old_update_mode not in {"pixel_topk", "projected_shared"}:
            raise ValueError(
                "GIR old update mode must be pixel_topk or projected_shared; "
                f"got {old_update_mode!r}."
            )
        projected_update_enabled = (
            old_residual_enabled and old_update_mode == "projected_shared"
        )
        if projected_update_enabled and self.gir_projected_update_head is None:
            raise RuntimeError(
                "projected_shared is enabled, but the projected GS head is missing."
            )
        old_opacity_gradient_enabled = bool(
            getattr(cfg, "gir_old_opacity_gradient_enabled", False)
        )
        new_opacity_gradient_enabled = bool(
            getattr(cfg, "gir_new_opacity_gradient_enabled", False)
        )
        if (
            old_opacity_gradient_enabled
            and old_residual_enabled
            and old_update_mode != "projected_shared"
        ):
            raise ValueError(
                "Old-GS opacity-gradient conditioning requires "
                "gir_old_update_mode=projected_shared."
            )
        opacity_gradient_replay_views = int(
            getattr(cfg, "gir_opacity_gradient_replay_views", 6)
        )
        if opacity_gradient_replay_views < -1:
            raise ValueError(
                "gir_opacity_gradient_replay_views must be -1, 0, or positive; "
                f"got {opacity_gradient_replay_views}."
            )
        opacity_gradient_scale = float(
            getattr(cfg, "gir_opacity_gradient_scale", 1000.0)
        )
        opacity_gradient_clip = float(
            getattr(cfg, "gir_opacity_gradient_clip", 20.0)
        )
        opacity_gradient_threshold = float(
            getattr(cfg, "gir_opacity_gradient_threshold", 1e-3)
        )
        opacity_gradient_current_weight = float(
            getattr(cfg, "gir_opacity_gradient_current_weight", 1.0)
        )
        opacity_gradient_replay_weight = float(
            getattr(cfg, "gir_opacity_gradient_replay_weight", 0.5)
        )
        opacity_gradient_front_decay_gate_enabled = bool(
            getattr(
                cfg,
                "gir_opacity_gradient_front_decay_gate_enabled",
                False,
            )
        )
        opacity_gradient_front_decay_threshold = float(
            getattr(
                cfg,
                "gir_opacity_gradient_front_decay_threshold",
                0.0,
            )
        )
        mean_gradient_enabled = bool(
            getattr(cfg, "gir_mean_gradient_enabled", False)
        )
        if mean_gradient_enabled and not projected_update_enabled:
            raise ValueError(
                "Old-GS mean-gradient conditioning requires "
                "gir_old_residual_enabled=true and "
                "gir_old_update_mode=projected_shared."
            )
        mean_gradient_scale = float(
            getattr(cfg, "gir_mean_gradient_scale", 1.0)
        )
        mean_gradient_clip = float(
            getattr(cfg, "gir_mean_gradient_clip", 20.0)
        )
        mean_gradient_threshold = float(
            getattr(cfg, "gir_mean_gradient_threshold", 1e-6)
        )
        mean_gradient_replay_views = int(
            getattr(
                cfg,
                "gir_mean_gradient_replay_views",
                opacity_gradient_replay_views,
            )
        )
        if mean_gradient_replay_views < -1:
            raise ValueError(
                "gir_mean_gradient_replay_views must be -1, 0, or positive; "
                f"got {mean_gradient_replay_views}."
            )
        old_mean_gradient_current_weight = float(
            getattr(cfg, "gir_old_mean_gradient_current_weight", 1.0)
        )
        old_mean_gradient_replay_weight = float(
            getattr(cfg, "gir_old_mean_gradient_replay_weight", 0.5)
        )
        new_mean_gradient_replay_views = int(
            getattr(cfg, "gir_new_mean_gradient_replay_views", 0)
        )
        if new_mean_gradient_replay_views < -1:
            raise ValueError(
                "gir_new_mean_gradient_replay_views must be -1, 0, or "
                f"positive; got {new_mean_gradient_replay_views}."
            )
        new_mean_gradient_current_weight = float(
            getattr(cfg, "gir_new_mean_gradient_current_weight", 1.0)
        )
        new_mean_gradient_replay_weight = float(
            getattr(cfg, "gir_new_mean_gradient_replay_weight", 0.0)
        )
        old_decay_enabled = bool(
            getattr(cfg, "gir_old_decay_enabled", False)
        )
        old_decay_topk = max(
            1, int(getattr(cfg, "gir_old_decay_topk", 4))
        )
        old_decay_strength = max(
            0.0, float(getattr(cfg, "gir_old_decay_strength", 0.02))
        )
        old_decay_prune_threshold = max(
            0.0,
            float(getattr(cfg, "gir_old_decay_prune_threshold", 0.005)),
        )
        residual_topk = max(1, int(getattr(cfg, "gir_residual_topk", 1)))
        independent_residual_heads = bool(
            getattr(cfg, "gir_residual_independent_heads", True)
        )
        mean_update_mode = str(
            getattr(cfg, "gir_mean_update_mode", "absolute")
        ).lower()
        if mean_update_mode not in {"absolute", "relative_depth"}:
            raise ValueError(
                "GIR mean update mode must be absolute or relative_depth; "
                f"got {mean_update_mode!r}."
            )
        raw_scale_residual = bool(
            getattr(cfg, "gir_raw_scale_residual", True)
        )
        raw_opacity_residual = bool(
            getattr(cfg, "gir_raw_opacity_residual", True)
        )
        raw_rotation_residual = bool(
            getattr(cfg, "gir_raw_rotation_residual", False)
        )
        raw_harmonics_residual = bool(
            getattr(cfg, "gir_raw_harmonics_residual", False)
        )
        mean_relative_scale = float(
            getattr(cfg, "gir_mean_relative_scale", 0.02)
        )
        rotation_scale = float(getattr(cfg, "gir_rotation_scale", 0.05))
        harmonics_scale = float(getattr(cfg, "gir_harmonics_scale", 0.10))
        if (
            old_update_mode == "pixel_topk"
            and residual_topk > 1
            and not use_raster_evidence
        ):
            raise ValueError(
                "Top-k historical residuals require "
                "gir_raster_evidence_enabled=true."
            )
        if old_decay_enabled and old_decay_topk > 1 and not use_raster_evidence:
            raise ValueError(
                "Top-k old-GS decay requires gir_raster_evidence_enabled=true."
            )
        use_dominant_ids = bool(
            getattr(cfg, "gir_dominant_id_enabled", False)
        ) and old_update_mode == "pixel_topk"
        min_dominant_weight = float(
            max(0.0, getattr(cfg, "gir_dominant_min_weight", 1e-4))
        )
        old_delete_enabled = bool(
            getattr(cfg, "gir_old_delete_enabled", False)
        )
        if old_delete_enabled and old_decay_enabled:
            raise ValueError(
                "gir_old_delete_enabled and gir_old_decay_enabled are "
                "mutually exclusive."
            )
        old_delete_threshold = float(
            getattr(cfg, "gir_old_delete_threshold", 0.5)
        )
        if not 0.0 <= old_delete_threshold <= 1.0:
            raise ValueError(
                "GIR old-delete threshold must be in [0, 1], "
                f"got {old_delete_threshold}."
            )
        old_delete_temperature = max(
            1e-4, float(getattr(cfg, "gir_old_delete_temperature", 1.0))
        )
        old_delete_min_observations = max(
            1, int(getattr(cfg, "gir_old_delete_min_observations", 2))
        )
        old_delete_test_prune = bool(
            getattr(cfg, "gir_old_delete_test_prune_enabled", True)
        )
        old_delete_warmup_steps = max(
            0, int(getattr(cfg, "gir_old_delete_warmup_steps", 1000))
        )
        old_delete_target_ratio = max(
            0.0,
            min(1.0, float(getattr(cfg, "gir_old_delete_target_ratio", 0.01))),
        )
        old_delete_warmup_progress = (
            1.0
            if not self.training or old_delete_warmup_steps == 0
            else min(
                1.0,
                max(0.0, float(global_step) / old_delete_warmup_steps),
            )
        )
        scheduled_old_delete_target = (
            old_delete_target_ratio * old_delete_warmup_progress
        )
        projected_warmup_steps = max(
            0, int(getattr(cfg, "gir_projected_warmup_steps", 1000))
        )
        projected_warmup_progress = (
            1.0
            if not self.training or projected_warmup_steps == 0
            else min(
                1.0,
                max(0.0, float(global_step) / projected_warmup_steps),
            )
        )
        add_gate_enabled = bool(
            getattr(cfg, "gir_add_gate_enabled", True)
        )
        append_mode = str(
            getattr(
                cfg,
                "gir_append_mode",
                "learned" if add_gate_enabled else "all",
            )
        ).lower()
        if append_mode not in {"all", "point_map", "learned"}:
            raise ValueError(
                "gir_append_mode must be all, point_map, or learned; "
                f"got {append_mode!r}."
            )
        if append_mode == "learned" and not add_gate_enabled:
            raise ValueError(
                "gir_append_mode=learned requires gir_add_gate_enabled=true."
            )
        point_append_depth_check = bool(
            getattr(cfg, "gir_point_append_depth_check_enabled", True)
        )
        point_append_hybrid = bool(
            getattr(cfg, "gir_point_append_hybrid_enabled", False)
        )
        point_append_current_in_front = bool(
            getattr(
                cfg,
                "gir_point_append_current_in_front_enabled",
                True,
            )
        )
        point_append_history_in_front = bool(
            getattr(
                cfg,
                "gir_point_append_history_in_front_enabled",
                True,
            )
        )
        point_append_tolerance = float(
            getattr(cfg, "gir_point_append_depth_relative_tolerance", 0.15)
        )
        point_append_rgb_threshold = float(
            getattr(cfg, "gir_point_append_rgb_bad_threshold", 0.08)
        )
        point_append_old_alpha_threshold = float(
            getattr(cfg, "gir_point_append_old_alpha_threshold", 0.10)
        )
        point_append_min_depth_confidence = float(
            getattr(cfg, "gir_point_append_min_depth_confidence", 0.0)
        )
        point_append_front_decay = float(
            getattr(
                cfg,
                "gir_point_append_front_conflict_opacity_decay",
                0.30,
            )
        )
        point_append_front_confirm_views = int(
            getattr(
                cfg,
                "gir_point_append_front_conflict_confirm_views",
                1,
            )
        )
        point_append_max_ratio = float(
            getattr(cfg, "gir_point_append_max_ratio", 1.0)
        )
        point_append_dilation = int(
            getattr(cfg, "gir_point_append_dilation_radius", 2)
        )
        point_append_chunk_size = int(
            getattr(cfg, "gir_point_append_chunk_size", 65536)
        )
        new_residual_enabled = bool(
            getattr(cfg, "gir_new_residual_enabled", False)
        )
        new_mean_gradient_enabled = bool(
            getattr(cfg, "gir_new_mean_gradient_enabled", False)
        )
        if new_mean_gradient_enabled and not new_residual_enabled:
            raise ValueError(
                "New-GS mean-gradient conditioning requires "
                "gir_new_residual_enabled=true."
            )
        historical_detach_mode = str(
            getattr(cfg, "gir_historical_detach_mode", "means")
        ).lower()
        if historical_detach_mode not in {"all", "means", "none"}:
            raise ValueError(
                "GIR historical detach mode must be one of all, means, none; "
                f"got {historical_detach_mode!r}."
            )
        add_gate_warmup_steps = max(
            0, int(getattr(cfg, "gir_add_gate_warmup_steps", 1000))
        )
        training_run = str(getattr(cfg, "mode", "train")) == "train"
        add_gate_warmup_progress = (
            1.0
            if not training_run or add_gate_warmup_steps == 0
            else min(
                1.0,
                max(0.0, float(global_step) / add_gate_warmup_steps),
            )
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
        state: Optional[StreamingGaussianState] = None
        # Keep the streaming RGB terms separate so append holes can receive
        # focused supervision without changing current/replay normalization.
        base_rgb_losses = []
        current_rgb_losses = []
        new_hole_rgb_losses = []
        replay_rgb_losses = []
        history_adapt_losses = []
        history_preserve_losses = []
        history_before_errors = []
        history_after_errors = []
        history_past_before_errors = []
        history_past_after_errors = []
        history_past_degradations = []
        add_rate_ratios = []
        regularization_losses = []
        old_delete_budget_losses = []
        old_delete_candidate_probabilities = []
        old_delete_candidate_rates = []
        old_delete_candidate_counts = []
        old_delete_hard_ratios = []
        old_delete_removed_opacity_ratios = []
        old_delete_map_counts_before = []
        old_delete_map_counts_after = []
        old_decay_budget_losses = []
        old_decay_candidate_probabilities = []
        old_decay_candidate_counts = []
        old_decay_opacity_mass_ratios = []
        old_decay_hard_ratios = []
        old_decay_map_counts_before = []
        old_decay_map_counts_after = []
        projected_candidate_ratios = []
        projected_consistent_ratios = []
        projected_front_ratios = []
        projected_gate_means = []
        projected_opacity_shifts = []
        projected_front_decay_probabilities = []
        projected_gradient_stats = {
            key: []
            for key in (
                *OPACITY_GRADIENT_STAT_KEYS,
                *OLD_OPACITY_GRADIENT_EFFECTIVE_STAT_KEYS,
                *MEAN_GRADIENT_STAT_KEYS,
            )
        }
        point_coverage_ratios = []
        point_append_ratios = []
        point_front_conflict_ratios = []
        point_front_gradient_decay_gate_ratios = []
        point_depth_scales = []
        point_front_decay_mass_ratios = []
        old_delete_graph_anchor = features.new_zeros(())
        add_gates = []
        learned_add_gates = []
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
        historical_gates = []
        visible_ratios = []
        residual_magnitudes = []
        raster_alpha_means = []
        dominant_weight_means = []
        new_residual_stats = {key: [] for key in NEW_RESIDUAL_STAT_KEYS}
        debug_intermediates = []
        auxiliary_enabled = bool(
            self.training
            and float(getattr(cfg, "gir_aux_loss_weight", 0.0)) > 0.0
        )
        debug_numerics = bool(getattr(cfg, "gir_debug_numerics", False))
        debug_max_views = max(
            0, int(getattr(cfg, "gir_debug_numerics_max_views", 2))
        )

        for view_idx in range(source_views):
            # This is the same temporal boundary used by the overfit
            # experiment.  It prevents historical GS geometry from retaining
            # a gradient path through every earlier view.
            if state is not None:
                if historical_detach_mode == "all":
                    state = state.detach()
                elif historical_detach_mode == "means":
                    state = state.detach_means()

            current_feature = features[:, view_idx]
            current_rgb = context_image[:, view_idx]
            current_depth = depth[:, view_idx].permute(0, 3, 1, 2)
            # Match the overfit rollout, which does not condition streaming
            # evidence or point-map fusion on predicted depth confidence.
            current_depth_confidence = torch.ones_like(current_depth)
            current_gaussians = self._slice_gaussian_view(
                encoder_output.gaussians,
                view_idx,
                gaussians_per_view,
            )
            has_history = state is not None
            projected_alignment_context = None

            if auxiliary_enabled and view_idx == 0:
                # Render the first-view GS before any historical update or
                # later-view append. This is loss_base_first in overfit.
                base_render = self.decoder.forward(
                    current_gaussians,
                    pred_all_extrinsic[:, :1],
                    intrinsics[:, :1],
                    torch.full((b, 1), near, device=features.device),
                    torch.full((b, 1), far, device=features.device),
                    (low_h, low_w),
                    "depth",
                )
                base_target = F.interpolate(
                    current_rgb.float(),
                    size=(low_h, low_w),
                    mode="bilinear",
                    align_corners=False,
                )[:, None].to(base_render.color.dtype)
                base_rgb_losses.append(
                    (base_render.color - base_target).square().mean()
                )

            projected_raster = None
            old_coverage = None
            depth_consistent = None
            old_in_front = None
            rgb_error = None
            residual_mask = None
            if state is None:
                gir = DominantGIR.empty(
                    b,
                    low_h,
                    low_w,
                    features.device,
                    features.dtype,
                )
            else:
                if old_update_mode == "projected_shared":
                    gir = DominantGIR.empty(
                        b,
                        low_h,
                        low_w,
                        features.device,
                        features.dtype,
                    )
                    projected_raster = self.decoder.render_projected_gs_evidence(
                        state.gaussians,
                        pred_all_extrinsic[:, view_idx],
                        intrinsics[:, view_idx],
                        (low_h, low_w),
                    )
                    projected_alpha = torch.nan_to_num(
                        projected_raster.alpha.float(),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).clamp(0.0, 1.0)
                    projected_depth = torch.nan_to_num(
                        projected_raster.expected_depth.float(),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    projected_depth = torch.where(
                        projected_alpha > 1e-4,
                        projected_depth.clamp_min(1e-6),
                        torch.zeros_like(projected_depth),
                    )
                    gir.rgb = torch.nan_to_num(
                        projected_raster.color.float(),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).clamp(0.0, 1.0).to(gir.rgb.dtype)
                    gir.depth = projected_depth.to(gir.depth.dtype)
                    gir.raster_depth = projected_depth.to(gir.raster_depth.dtype)
                    gir.raster_alpha = projected_alpha.to(gir.raster_alpha.dtype)
                    gir.opacity = projected_alpha.to(gir.opacity.dtype)
                    gir.dominant_weight = projected_alpha.to(
                        gir.dominant_weight.dtype
                    )
                    gir.valid = projected_alpha >= float(
                        getattr(cfg, "gir_residual_alpha_threshold", 0.02)
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
                                residual_topk if old_residual_enabled else 1,
                                old_decay_topk if old_decay_enabled else 1,
                            ),
                        )

            if use_raster_evidence:
                raster_alpha_means.append(gir.raster_alpha.mean())
            if use_dominant_ids and state is not None:
                valid_count = gir.valid.sum().clamp_min(1)
                dominant_weight_means.append(
                    (gir.dominant_weight * gir.valid).sum() / valid_count
                )

            residual_gir = gir
            projected_update_stats = None
            if has_history:
                # Match the overfit residual selection: update historical GS
                # only where the old map is geometrically consistent but RGB
                # is wrong, or where the old map lies behind the current
                # surface.  All masks are evaluated at GIR resolution.
                current_depth_low = F.interpolate(
                    current_depth.float(),
                    size=(low_h, low_w),
                    mode="bilinear",
                    align_corners=False,
                )
                current_rgb_low = F.interpolate(
                    current_rgb.float(),
                    size=(low_h, low_w),
                    mode="bilinear",
                    align_corners=False,
                )
                valid_depth_low = current_depth_low > 0
                old_coverage = gir.valid & (
                    gir.raster_alpha
                    >= float(getattr(cfg, "gir_residual_alpha_threshold", 0.02))
                )
                relative_depth_delta = (
                    gir.depth.float() - current_depth_low
                ) / current_depth_low.clamp_min(1e-4)
                depth_tolerance = float(
                    getattr(cfg, "gir_residual_depth_relative_tolerance", 0.08)
                )
                depth_consistent = (
                    old_coverage
                    & valid_depth_low
                    & (relative_depth_delta.abs() <= depth_tolerance)
                )
                old_in_front = (
                    old_coverage
                    & valid_depth_low
                    & (relative_depth_delta < -depth_tolerance)
                )
                rgb_error = (
                    gir.rgb.float() - current_rgb_low.float()
                ).abs().mean(dim=1, keepdim=True)
                rgb_bad = rgb_error >= float(
                    getattr(cfg, "gir_residual_rgb_bad_threshold", 0.08)
                )
                residual_mask = (depth_consistent & rgb_bad) | old_in_front
                residual_gir = _mask_gir(gir, residual_mask)

            projected_evidence = None
            projected_prediction = None
            old_gradient_evidence = None
            old_mean_gradient_evidence = None
            point_append_mask = None
            point_coverage = None
            point_front_conflict_mask = None
            point_front_decay_mask = None
            point_depth_scale = None

            # Match overfit: decide append/front-conflict ownership from the
            # incoming historical map before applying its learned residual.
            if has_history and append_mode == "point_map":
                (
                    point_append_mask,
                    point_coverage,
                    point_front_conflict_mask,
                    next_front_conflict_count,
                    point_depth_scale,
                ) = build_point_map_append_mask(
                    state=state,
                    current_depth=current_depth,
                    camera_to_world=pred_all_extrinsic[:, view_idx],
                    intrinsics=intrinsics[:, view_idx],
                    current_rgb=current_rgb,
                    old_rgb=gir.rgb,
                    old_depth=gir.raster_depth,
                    old_alpha=gir.raster_alpha,
                    current_depth_confidence=current_depth_confidence,
                    depth_relative_tolerance=point_append_tolerance,
                    rgb_bad_threshold=point_append_rgb_threshold,
                    old_alpha_threshold=point_append_old_alpha_threshold,
                    min_depth_confidence=point_append_min_depth_confidence,
                    dilation_radius=point_append_dilation,
                    chunk_size=point_append_chunk_size,
                    depth_check_enabled=point_append_depth_check,
                    hybrid_enabled=point_append_hybrid,
                    append_current_in_front_enabled=(
                        point_append_current_in_front
                    ),
                    append_history_in_front_enabled=(
                        point_append_history_in_front
                    ),
                    front_conflict_confirm_views=(
                        point_append_front_confirm_views
                    ),
                    max_append_ratio=point_append_max_ratio,
                )
                state = state.with_front_conflict_count(
                    next_front_conflict_count
                )
                point_coverage_ratios.append(point_coverage.float().mean())
                point_append_ratios.append(point_append_mask.float().mean())
                point_depth_scales.append(point_depth_scale.mean())
                valid_old = (
                    (state.stable_ids >= 0).float().sum().clamp_min(1.0)
                )
                point_front_conflict_ratios.append(
                    point_front_conflict_mask.float().sum() / valid_old
                )

            if projected_update_enabled and has_history:
                if projected_raster is None:
                    raise RuntimeError(
                        "Projected old-GS update requires old-map raster evidence."
                    )
                if old_opacity_gradient_enabled or mean_gradient_enabled:
                    opacity_uses_all_replay = (
                        old_opacity_gradient_enabled
                        and opacity_gradient_replay_views == -1
                    )
                    mean_uses_all_replay = (
                        mean_gradient_enabled
                        and mean_gradient_replay_views == -1
                    )
                    gradient_replay_limit = (
                        -1
                        if opacity_uses_all_replay or mean_uses_all_replay
                        else max(
                            opacity_gradient_replay_views
                            if old_opacity_gradient_enabled
                            else 0,
                            mean_gradient_replay_views
                            if mean_gradient_enabled
                            else 0,
                        )
                    )
                    gradient_replay_indices = _causal_replay_indices(
                        view_idx,
                        gradient_replay_limit,
                        "combined GIR gradient replay views",
                    )
                    gradient_view_indices = [view_idx, *gradient_replay_indices]
                    old_gradient_active_mask = (
                        (state.stable_ids >= 0)
                        & (state.gaussians.opacities.detach() > 0)
                    )
                    old_gradients = self.decoder.render_parameter_gradients(
                        state.gaussians,
                        pred_all_extrinsic[:, gradient_view_indices],
                        intrinsics[:, gradient_view_indices],
                        context_image[:, gradient_view_indices],
                        (low_h, low_w),
                        active_mask=old_gradient_active_mask,
                        mean_camera_enabled=mean_gradient_enabled,
                    )
                    if old_opacity_gradient_enabled:
                        old_gradient_evidence = build_opacity_gradient_evidence(
                            select_current_and_recent_replay_gradients(
                                old_gradients.opacity_logit,
                                opacity_gradient_replay_views,
                            ),
                            gradient_scale=opacity_gradient_scale,
                            gradient_clip=opacity_gradient_clip,
                            gradient_threshold=opacity_gradient_threshold,
                            current_weight=opacity_gradient_current_weight,
                            replay_weight=opacity_gradient_replay_weight,
                            active_mask=old_gradient_active_mask,
                        )
                    if mean_gradient_enabled:
                        if old_gradients.mean_camera is None:
                            raise RuntimeError(
                                "Mean-gradient rendering did not return mean "
                                "gradients."
                            )
                        old_mean_gradient_evidence = build_mean_gradient_evidence(
                            select_current_and_recent_replay_gradients(
                                old_gradients.mean_camera,
                                mean_gradient_replay_views,
                            ),
                            gradient_scale=mean_gradient_scale,
                            gradient_clip=mean_gradient_clip,
                            gradient_threshold=mean_gradient_threshold,
                            current_weight=old_mean_gradient_current_weight,
                            replay_weight=old_mean_gradient_replay_weight,
                            active_mask=old_gradient_active_mask,
                        )
                projected_evidence = build_projected_gaussian_evidence(
                    state=state,
                    current_feature=current_feature,
                    current_rgb=current_rgb,
                    current_depth=current_depth,
                    current_depth_confidence=current_depth_confidence,
                    old_rgb=gir.rgb,
                    old_depth=gir.raster_depth,
                    old_alpha=gir.raster_alpha,
                    camera_to_world=pred_all_extrinsic[:, view_idx],
                    intrinsics=intrinsics[:, view_idx],
                    alpha_threshold=float(
                        getattr(cfg, "gir_residual_alpha_threshold", 0.02)
                    ),
                    depth_relative_tolerance=float(
                        getattr(
                            cfg,
                            "gir_residual_depth_relative_tolerance",
                            0.08,
                        )
                    ),
                    rgb_bad_threshold=float(
                        getattr(cfg, "gir_residual_rgb_bad_threshold", 0.08)
                    ),
                    min_projected_radius=float(
                        getattr(cfg, "gir_projected_min_radius", 0.1)
                    ),
                    covariance_extent_sigma=float(
                        getattr(cfg, "gir_projected_covariance_sigma", 3.0)
                    ),
                    chunk_size=int(
                        getattr(cfg, "gir_projected_chunk_size", 65536)
                    ),
                    opacity_gradient_evidence=old_gradient_evidence,
                    mean_gradient_evidence=old_mean_gradient_evidence,
                )
                log_cuda_memory(
                    "gir_projected_evidence",
                    step=global_step,
                    view=f"{view_idx + 1}/{source_views}",
                    state_gs=state.num_gaussians,
                    projected_candidates=int(projected_evidence.descriptor.shape[0]),
                )
                projected_prediction = self.gir_projected_update_head(
                    projected_evidence
                )
                log_cuda_memory(
                    "gir_projected_prediction",
                    step=global_step,
                    view=f"{view_idx + 1}/{source_views}",
                    projected_candidates=int(projected_evidence.descriptor.shape[0]),
                )

            prediction = self.gir_update_head(
                current_feature,
                current_rgb,
                current_depth,
                current_depth_confidence,
                residual_gir,
                num_historical_predictions=(
                    residual_topk
                    if (
                        old_update_mode == "pixel_topk"
                        and independent_residual_heads
                        and old_residual_enabled
                    )
                    else 1
                ),
                decay_gir=gir,
                num_decay_predictions=(
                    old_decay_topk
                    if old_decay_enabled and has_history
                    else 0
                ),
            )
            debug_view = debug_numerics and view_idx < debug_max_views
            if debug_view:
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} current_feature",
                    current_feature,
                )
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} current_depth",
                    current_depth,
                )
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} gir_depth",
                    residual_gir.depth,
                )
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} gir_alpha",
                    residual_gir.raster_alpha,
                )
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} delta_mean_raw",
                    prediction.delta_mean_camera,
                )
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} delta_scale_raw",
                    prediction.delta_log_scale,
                )
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} delta_opacity_raw",
                    prediction.delta_opacity_logit,
                )
            if mean_update_mode == "relative_depth":
                prediction.delta_mean_camera = _bound_tanh_input(
                    prediction.delta_mean_camera, mean_relative_scale
                )
            if not raw_rotation_residual:
                prediction.delta_rotation = _bound_tanh_input(
                    prediction.delta_rotation, rotation_scale
                )
            if not raw_harmonics_residual:
                prediction.delta_harmonics = _bound_tanh_input(
                    prediction.delta_harmonics, harmonics_scale
                )
            if prediction.rank_delta_mean_camera is not None:
                if mean_update_mode == "relative_depth":
                    prediction.rank_delta_mean_camera = _bound_tanh_input(
                        prediction.rank_delta_mean_camera,
                        mean_relative_scale,
                    )
                if not raw_rotation_residual:
                    prediction.rank_delta_rotation = _bound_tanh_input(
                        prediction.rank_delta_rotation, rotation_scale
                    )
                if not raw_harmonics_residual:
                    prediction.rank_delta_harmonics = _bound_tanh_input(
                        prediction.rank_delta_harmonics, harmonics_scale
                    )
            old_delete_graph_anchor = (
                old_delete_graph_anchor + 0.0 * prediction.delete_logit.sum()
            )
            # The overfit experiment uses a selected-update mask and an
            # always-on historical update for the selected pixels.  Keep the
            # same behavior here; the mask, not a learned gate, decides which
            # old GS receive the residual.
            prediction.historical_gate = torch.full_like(
                prediction.historical_gate, 12.0
            )
            if prediction.rank_historical_gate is not None:
                prediction.rank_historical_gate = torch.full_like(
                    prediction.rank_historical_gate, 12.0
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
                    replay_indices = _causal_replay_indices(
                        view_idx,
                        int(getattr(cfg, "gir_history_replay_views", 1)),
                        "gir_history_replay_views",
                    )
                    if replay_indices:
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
                ownership = (
                    gir.dominant_weight.detach().float()
                    / evidence_alpha.clamp_min(1e-6)
                ).clamp(0.0, 1.0)
                valid_float = gir.valid.detach().float()
                valid_count = valid_float.sum().clamp_min(1.0)
                top1_ownership_means.append(
                    (ownership * valid_float).sum() / valid_count
                )
                top1_ownership_above_0_1.append(
                    ((ownership > 0.1).float() * valid_float).sum()
                    / valid_count
                )
                top1_ownership_above_0_25.append(
                    ((ownership > 0.25).float() * valid_float).sum()
                    / valid_count
                )
                top1_ownership_above_0_5.append(
                    ((ownership > 0.5).float() * valid_float).sum()
                    / valid_count
                )
                historical_update_confidence = None
                if top1_confidence_mode == "sqrt":
                    historical_update_confidence = ownership.sqrt()
                elif top1_confidence_mode == "floor_sqrt":
                    historical_update_confidence = confidence_floor + (
                        1.0 - confidence_floor
                    ) * ownership.sqrt()
                if historical_update_confidence is None:
                    top1_confidence_means.append(ownership.new_ones(()))
                else:
                    top1_confidence_means.append(
                        (historical_update_confidence * valid_float).sum()
                        / valid_count
                    )

                if old_residual_enabled:
                    if old_update_mode == "projected_shared":
                        if (
                            projected_evidence is None
                            or projected_prediction is None
                        ):
                            raise RuntimeError(
                                "Projected old-GS evidence/prediction is missing."
                            )
                        projected_original_opacities = state.gaussians.opacities
                        state, projected_update_stats = (
                            state.update_projected_historical(
                                projected_evidence,
                                projected_prediction,
                                pred_all_extrinsic[:, view_idx],
                                opacity_logit_scale=float(
                                    getattr(
                                        cfg,
                                        "gir_projected_opacity_logit_scale",
                                        1.0,
                                    )
                                ),
                                front_decay_strength=float(
                                    getattr(
                                        cfg,
                                        "gir_projected_front_decay_strength",
                                        0.02,
                                    )
                                ),
                                decay_schedule=projected_warmup_progress,
                                mean_update_mode=mean_update_mode,
                                raw_scale_residual=raw_scale_residual,
                                raw_opacity_residual=raw_opacity_residual,
                                raw_rotation_residual=raw_rotation_residual,
                                raw_harmonics_residual=raw_harmonics_residual,
                                mean_relative_scale=mean_relative_scale,
                                rotation_scale=rotation_scale,
                                harmonics_scale=harmonics_scale,
                            )
                        )
                        projected_alignment_context = (
                            projected_evidence,
                            projected_original_opacities,
                        )
                        projected_candidate_ratios.append(
                            projected_update_stats["candidate_ratio"]
                        )
                        projected_consistent_ratios.append(
                            projected_update_stats["surface_consistent_ratio"]
                        )
                        projected_front_ratios.append(
                            projected_update_stats["old_in_front_ratio"]
                        )
                        projected_gate_means.append(
                            projected_update_stats["gate_mean"]
                        )
                        projected_opacity_shifts.append(
                            projected_update_stats["opacity_shift_mean"]
                        )
                        projected_front_decay_probabilities.append(
                            projected_update_stats[
                                "front_decay_probability_mean"
                            ]
                        )
                        for key in (
                            *OPACITY_GRADIENT_STAT_KEYS,
                            *OLD_OPACITY_GRADIENT_EFFECTIVE_STAT_KEYS,
                            *MEAN_GRADIENT_STAT_KEYS,
                        ):
                            projected_gradient_stats[key].append(
                                projected_update_stats[key]
                            )
                    else:
                        state = state.update_historical(
                            residual_gir,
                            prediction,
                            pred_all_extrinsic[:, view_idx],
                            update_confidence=historical_update_confidence,
                            num_contributors=residual_topk,
                            mean_update_mode=mean_update_mode,
                            raw_scale_residual=raw_scale_residual,
                            raw_opacity_residual=raw_opacity_residual,
                            raw_rotation_residual=raw_rotation_residual,
                            raw_harmonics_residual=raw_harmonics_residual,
                        )
                    if debug_view:
                        _gir_debug_tensor(
                            f"step={global_step} view={view_idx} "
                            "historical_state.means",
                            state.gaussians.means,
                        )
                        _gir_debug_tensor(
                            f"step={global_step} view={view_idx} "
                            "historical_state.scales",
                            state.gaussians.scales,
                        )
                        _gir_debug_tensor(
                            f"step={global_step} view={view_idx} "
                            "historical_state.opacities",
                            state.gaussians.opacities,
                        )

                if old_decay_enabled:
                    if prediction.rank_decay_logits is None:
                        raise RuntimeError(
                            "Top-k old-GS decay is enabled, but the GIR head "
                            "did not return contributor-conditioned logits."
                        )
                    state, decay_stats = state.decay_historical_opacity(
                        gir,
                        prediction.rank_decay_logits,
                        min_observations=old_delete_min_observations,
                        temperature=old_delete_temperature,
                        decay_strength=old_decay_strength,
                        decay_schedule=old_delete_warmup_progress,
                        min_contributor_weight=min_dominant_weight,
                        physical_prune=(
                            not self.training and old_delete_test_prune
                        ),
                        prune_threshold=old_decay_prune_threshold,
                    )
                    old_decay_candidate_probabilities.append(
                        decay_stats["candidate_probability_mean"]
                    )
                    old_decay_candidate_counts.append(
                        decay_stats["candidate_count"]
                    )
                    old_decay_opacity_mass_ratios.append(
                        decay_stats["decay_opacity_mass_ratio"]
                    )
                    old_decay_hard_ratios.append(
                        decay_stats["hard_pruned_ratio"]
                    )
                    old_decay_map_counts_before.append(
                        decay_stats["map_count_before"]
                    )
                    old_decay_map_counts_after.append(
                        decay_stats["map_count_after"]
                    )
                    if self.training:
                        # Keep the loss graph identical across DDP ranks. A
                        # sample with no eligible historical contributors only
                        # contributes zero through the detached candidate mask.
                        candidate_gate = (
                            decay_stats["candidate_count"] > 0
                        ).to(features.dtype).detach()
                        old_decay_budget_losses.append(
                            (
                                decay_stats["decay_opacity_mass_ratio"]
                                - scheduled_old_delete_target
                            ).abs()
                            * candidate_gate
                        )

                elif old_delete_enabled:
                    state, delete_stats = state.delete_historical(
                        gir,
                        prediction.delete_logit,
                        min_observations=old_delete_min_observations,
                        threshold=old_delete_threshold,
                        temperature=old_delete_temperature,
                        physical_prune=(
                            not self.training and old_delete_test_prune
                        ),
                    )
                    old_delete_candidate_probabilities.append(
                        delete_stats["candidate_probability_mean"]
                    )
                    old_delete_candidate_counts.append(
                        delete_stats["candidate_count"]
                    )
                    old_delete_candidate_rates.append(
                        delete_stats["candidate_delete_rate"]
                    )
                    old_delete_hard_ratios.append(
                        delete_stats["hard_deleted_ratio"]
                    )
                    old_delete_removed_opacity_ratios.append(
                        delete_stats["removed_opacity_mass_ratio"]
                    )
                    old_delete_map_counts_before.append(
                        delete_stats["map_count_before"]
                    )
                    old_delete_map_counts_after.append(
                        delete_stats["map_count_after"]
                    )

                    if self.training and bool(
                        delete_stats["candidate_count"] > 0
                    ):
                        old_delete_budget_losses.append(
                            (
                                delete_stats["candidate_delete_rate"]
                                - scheduled_old_delete_target
                            ).abs()
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
                    history_mask = gir.valid.to(history_after_render.color.dtype)
                    history_normalizer = history_mask.sum().clamp_min(1.0)
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

            new_residual_energy = prediction.add_logit.new_zeros(())

            coverage = gir.valid.to(prediction.add_logit.dtype)
            residual_coverage = residual_gir.valid.to(
                prediction.add_logit.dtype
            )
            visible_ratios.append(coverage.mean())
            if has_history and append_mode == "point_map":
                if point_front_conflict_mask is None:
                    raise RuntimeError(
                        "Point-map front-conflict mask is missing."
                    )
                (
                    point_front_decay_mask,
                    point_front_gradient_decay_gate_ratio,
                ) = gate_front_conflict_decay(
                    point_front_conflict_mask,
                    old_gradient_evidence,
                    enabled=opacity_gradient_front_decay_gate_enabled,
                    threshold=opacity_gradient_front_decay_threshold,
                )
                point_front_gradient_decay_gate_ratios.append(
                    point_front_gradient_decay_gate_ratio
                )
                opacity_before_decay = state.gaussians.opacities
                state = state.decay_masked_opacity(
                    point_front_decay_mask,
                    point_append_front_decay,
                )
                point_front_decay_mass_ratios.append(
                    (
                        opacity_before_decay.detach().float().sum()
                        - state.gaussians.opacities.detach().float().sum()
                    )
                    / opacity_before_decay.detach().float().sum().clamp_min(1e-8)
                )

            if projected_alignment_context is not None and state is not None:
                alignment_evidence, original_opacities = (
                    projected_alignment_context
                )
                if original_opacities.shape == state.gaussians.opacities.shape:
                    final_alignment_stats = (
                        compute_old_opacity_gradient_effective_stats(
                            alignment_evidence,
                            original_opacities,
                            state.gaussians.opacities,
                        )
                    )
                    for key in OLD_OPACITY_GRADIENT_EFFECTIVE_STAT_KEYS:
                        projected_gradient_stats[key][-1] = final_alignment_stats[
                            key
                        ]

            if state is None or append_mode == "all":
                # The first frame initializes the point map; all mode is the
                # explicit dense-append ablation.
                learned_add_gate_low = torch.ones_like(prediction.add_logit)
                add_gate_low = torch.ones_like(prediction.add_logit) + (
                    0.0 * prediction.add_logit
                )
                if state is not None:
                    learned_add_gates.append(learned_add_gate_low.mean())
                add_gate = F.interpolate(
                    add_gate_low,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
            elif append_mode == "point_map":
                if point_append_mask is None:
                    raise RuntimeError("Point-map append mask is missing.")
                add_gate = point_append_mask.to(prediction.add_logit.dtype)
                add_gate_low = F.interpolate(
                    add_gate,
                    size=(low_h, low_w),
                    mode="nearest",
                )
                learned_add_gate_low = add_gate_low
                learned_add_gates.append(learned_add_gate_low.mean())
            else:
                learned_add_gate_low = torch.sigmoid(prediction.add_logit)
                add_gate_low = 1.0 - add_gate_warmup_progress * (
                    1.0 - learned_add_gate_low
                )
                learned_add_gates.append(learned_add_gate_low.mean())
                add_gate = F.interpolate(
                    add_gate_low,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
            add_gates.append(add_gate_low.mean())

            if (
                return_debug_intermediates
                and has_history
                and (view_idx - 1) % debug_view_stride == 0
            ):
                point_map = (
                    state.point_map
                    if state.point_map is not None
                    else state.gaussians.means
                )
                front_conflict_projection = (
                    self._project_selected_points_for_debug(
                        point_map,
                        point_front_conflict_mask,
                        pred_all_extrinsic[:, view_idx],
                        intrinsics[:, view_idx],
                        (h, w),
                    )
                    if point_front_conflict_mask is not None
                    else None
                )
                front_decay_projection = (
                    self._project_selected_points_for_debug(
                        point_map,
                        point_front_decay_mask,
                        pred_all_extrinsic[:, view_idx],
                        intrinsics[:, view_idx],
                        (h, w),
                    )
                    if point_front_decay_mask is not None
                    else None
                )
                active_old = (state.stable_ids >= 0)
                active_old_count = active_old.float().sum().clamp_min(1.0)

                def snapshot(value: torch.Tensor | None):
                    return (
                        None
                        if value is None
                        else value.detach().float().cpu()
                    )

                debug_intermediates.append(
                    {
                        "view_index": view_idx,
                        "current_rgb": snapshot(current_rgb),
                        "old_rgb": snapshot(gir.rgb),
                        "rgb_error": snapshot(rgb_error),
                        "current_depth": snapshot(current_depth),
                        "old_depth": snapshot(gir.raster_depth),
                        "old_alpha": snapshot(gir.raster_alpha),
                        "old_coverage": snapshot(old_coverage),
                        "depth_consistent": snapshot(depth_consistent),
                        "old_in_front": snapshot(old_in_front),
                        "residual_mask": snapshot(residual_mask),
                        "point_coverage": snapshot(point_coverage),
                        "append_mask": snapshot(point_append_mask),
                        "add_gate": snapshot(add_gate),
                        "front_conflict_projection": snapshot(
                            front_conflict_projection
                        ),
                        "front_decay_projection": snapshot(
                            front_decay_projection
                        ),
                        "summary": {
                            "old_gaussians": int(state.num_gaussians),
                            "old_active_gaussians": int(active_old.sum().item()),
                            "coverage_ratio": float(
                                point_coverage.float().mean().item()
                            )
                            if point_coverage is not None
                            else None,
                            "append_ratio": float(
                                point_append_mask.float().mean().item()
                            )
                            if point_append_mask is not None
                            else None,
                            "appended_gaussians": int(
                                point_append_mask.sum().item()
                            )
                            if point_append_mask is not None
                            else None,
                            "old_coverage_ratio": float(
                                old_coverage.float().mean().item()
                            )
                            if old_coverage is not None
                            else None,
                            "depth_consistent_ratio": float(
                                depth_consistent.float().mean().item()
                            )
                            if depth_consistent is not None
                            else None,
                            "old_in_front_ratio": float(
                                old_in_front.float().mean().item()
                            )
                            if old_in_front is not None
                            else None,
                            "residual_candidate_ratio": float(
                                residual_mask.float().mean().item()
                            )
                            if residual_mask is not None
                            else None,
                            "rgb_error_mean": float(rgb_error.mean().item())
                            if rgb_error is not None
                            else None,
                            "old_alpha_mean": float(
                                gir.raster_alpha.float().mean().item()
                            ),
                            "add_gate_mean": float(add_gate.mean().item()),
                            "front_conflict_count": int(
                                point_front_conflict_mask.sum().item()
                            )
                            if point_front_conflict_mask is not None
                            else None,
                            "front_conflict_ratio": float(
                                point_front_conflict_mask.float().sum().item()
                                / active_old_count.item()
                            )
                            if point_front_conflict_mask is not None
                            else None,
                            "front_decay_count": int(
                                point_front_decay_mask.sum().item()
                            )
                            if point_front_decay_mask is not None
                            else None,
                            "front_decay_ratio": float(
                                point_front_decay_mask.float().sum().item()
                                / active_old_count.item()
                            )
                            if point_front_decay_mask is not None
                            else None,
                            "point_depth_scale": float(
                                point_depth_scale.mean().item()
                            )
                            if point_depth_scale is not None
                            else None,
                        },
                    }
                )
            if has_history and new_residual_enabled:
                if self.gir_projected_update_head is None:
                    raise RuntimeError(
                        "New-GS residuals require gir_projected_update_head."
                    )
                new_candidate_mask = (
                    point_append_mask
                    if append_mode == "point_map"
                    else torch.ones(
                        (b, 1, h, w),
                        device=current_depth.device,
                        dtype=torch.bool,
                    )
                )
                new_gradient_evidence = None
                new_mean_gradient_evidence = None
                if new_opacity_gradient_enabled or new_mean_gradient_enabled:
                    provisional, provisional_active = build_provisional_gaussians(
                        state.gaussians,
                        current_gaussians,
                        torch.ones_like(current_gaussians.opacities),
                        new_candidate_mask,
                    )
                    opacity_uses_all_replay = (
                        new_opacity_gradient_enabled
                        and opacity_gradient_replay_views == -1
                    )
                    mean_uses_all_replay = (
                        new_mean_gradient_enabled
                        and new_mean_gradient_replay_views == -1
                    )
                    gradient_replay_limit = (
                        -1
                        if opacity_uses_all_replay or mean_uses_all_replay
                        else max(
                            opacity_gradient_replay_views
                            if new_opacity_gradient_enabled
                            else 0,
                            new_mean_gradient_replay_views
                            if new_mean_gradient_enabled
                            else 0,
                        )
                    )
                    gradient_replay_indices = _causal_replay_indices(
                        view_idx,
                        gradient_replay_limit,
                        "combined GIR gradient replay views",
                    )
                    gradient_view_indices = [view_idx, *gradient_replay_indices]
                    provisional_gradients = self.decoder.render_parameter_gradients(
                        provisional,
                        pred_all_extrinsic[:, gradient_view_indices],
                        intrinsics[:, gradient_view_indices],
                        context_image[:, gradient_view_indices],
                        (low_h, low_w),
                        active_mask=provisional_active,
                        mean_camera_enabled=new_mean_gradient_enabled,
                    )
                    new_count = current_gaussians.opacities.shape[1]
                    new_active_mask = new_candidate_mask.reshape(b, new_count)
                    if new_opacity_gradient_enabled:
                        new_gradient_evidence = build_opacity_gradient_evidence(
                            select_current_and_recent_replay_gradients(
                                provisional_gradients.opacity_logit[
                                    :, :, -new_count:
                                ],
                                opacity_gradient_replay_views,
                            ),
                            gradient_scale=opacity_gradient_scale,
                            gradient_clip=opacity_gradient_clip,
                            gradient_threshold=opacity_gradient_threshold,
                            current_weight=opacity_gradient_current_weight,
                            replay_weight=opacity_gradient_replay_weight,
                            active_mask=new_active_mask,
                        )
                    if new_mean_gradient_enabled:
                        if provisional_gradients.mean_camera is None:
                            raise RuntimeError(
                                "New-GS mean-gradient rendering returned no "
                                "mean gradients."
                            )
                        new_mean_gradient_evidence = build_mean_gradient_evidence(
                            select_current_and_recent_replay_gradients(
                                provisional_gradients.mean_camera[
                                    :, :, -new_count:
                                ],
                                new_mean_gradient_replay_views,
                            ),
                            gradient_scale=mean_gradient_scale,
                            gradient_clip=mean_gradient_clip,
                            gradient_threshold=mean_gradient_threshold,
                            current_weight=new_mean_gradient_current_weight,
                            replay_weight=new_mean_gradient_replay_weight,
                            active_mask=new_active_mask,
                        )
                new_evidence = build_projected_new_gaussian_evidence(
                    current=current_gaussians,
                    append_mask=new_candidate_mask,
                    current_feature=current_feature,
                    current_rgb=current_rgb,
                    current_depth=current_depth,
                    current_depth_confidence=current_depth_confidence,
                    old_rgb=gir.rgb,
                    old_depth=gir.raster_depth,
                    old_alpha=gir.raster_alpha,
                    camera_to_world=pred_all_extrinsic[:, view_idx],
                    intrinsics=intrinsics[:, view_idx],
                    alpha_threshold=float(
                        getattr(cfg, "gir_point_append_old_alpha_threshold", 0.10)
                    ),
                    depth_relative_tolerance=point_append_tolerance,
                    covariance_extent_sigma=float(
                        getattr(cfg, "gir_projected_covariance_sigma", 3.0)
                    ),
                    opacity_gradient_evidence=new_gradient_evidence,
                    mean_gradient_evidence=new_mean_gradient_evidence,
                )
                new_prediction = self.gir_projected_update_head.forward_new(
                    new_evidence
                )
                current_gaussians, new_update_stats = (
                    apply_projected_new_gaussian_residual(
                        current_gaussians,
                        new_evidence,
                        new_prediction,
                        pred_all_extrinsic[:, view_idx],
                        opacity_logit_scale=float(
                            getattr(
                                cfg,
                                "gir_projected_opacity_logit_scale",
                                1.0,
                            )
                        ),
                        mean_update_mode=mean_update_mode,
                        raw_scale_residual=raw_scale_residual,
                        raw_opacity_residual=raw_opacity_residual,
                        raw_rotation_residual=raw_rotation_residual,
                        raw_harmonics_residual=raw_harmonics_residual,
                        mean_relative_scale=mean_relative_scale,
                        rotation_scale=rotation_scale,
                        harmonics_scale=harmonics_scale,
                    )
                )
                new_residual_energy = new_update_stats["regularization"]
                for key in NEW_RESIDUAL_STAT_KEYS:
                    new_residual_stats[key].append(new_update_stats[key])
            if has_history:
                learned_add_gate = F.interpolate(
                    learned_add_gate_low,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, gaussians_per_view)
                opacity_weight = current_gaussians.opacities.detach().float()
                add_rate_ratios.append(
                    (
                        learned_add_gate.float() * opacity_weight
                    ).sum() / opacity_weight.sum().clamp_min(1e-8)
                )
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
            if old_update_mode == "projected_shared" and has_history:
                if projected_update_stats is None:
                    raise RuntimeError("Projected old-GS update stats are missing.")
                historical_gates.append(projected_update_stats["gate_mean"])
                residual_magnitudes.append(
                    projected_update_stats["opacity_shift_mean"]
                )
                residual_energy = projected_update_stats["regularization"]
                old_residual_regularization = residual_energy
            else:
                historical_gates.append(
                    (prediction.historical_gate.sigmoid() * coverage).sum()
                    / coverage.sum().clamp_min(1.0)
                )
                if prediction.rank_delta_mean_camera is not None:
                    # Include all independent contributor heads in the residual
                    # statistic and regularizer, rather than only rank 1.
                    residual_energy = (
                        prediction.rank_delta_mean_camera.square().sum(
                            dim=2, keepdim=True
                        )
                        + prediction.rank_delta_rotation.square().sum(
                            dim=2, keepdim=True
                        )
                        + prediction.rank_delta_log_scale.square().sum(
                            dim=2, keepdim=True
                        )
                        + prediction.rank_delta_opacity_logit.square()
                        + prediction.rank_delta_harmonics.square().mean(
                            dim=2, keepdim=True
                        )
                    ).mean(dim=1)
                else:
                    residual_energy = (
                        prediction.delta_mean_camera.square().sum(
                            dim=1, keepdim=True
                        )
                        + prediction.delta_rotation.square().sum(
                            dim=1, keepdim=True
                        )
                        + prediction.delta_log_scale.square().sum(
                            dim=1, keepdim=True
                        )
                        + prediction.delta_opacity_logit.square()
                        + prediction.delta_harmonics.square().mean(
                            dim=1, keepdim=True
                        )
                    )
                residual_magnitudes.append(residual_energy.mean().sqrt())
                old_residual_regularization = (
                    (residual_energy * residual_coverage).sum()
                    / residual_coverage.sum().clamp_min(1.0)
                    if old_residual_enabled
                    else residual_energy.sum() * 0.0
                )
            regularization_losses.append(
                old_residual_regularization + new_residual_energy
            )

            if debug_view:
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} residual_energy",
                    residual_energy,
                )
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} old_reg",
                    old_residual_regularization,
                )
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} new_reg",
                    new_residual_energy,
                )

            if state is None:
                state = StreamingGaussianState.from_current(
                    current_gaussians,
                    add_gate,
                    point_map=refine_info["means"][:, view_idx],
                )
            else:
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
                    append_mask=point_append_mask,
                    point_map=refine_info["means"][:, view_idx],
                )

            log_cuda_memory(
                "gir_after_append",
                step=global_step,
                view=f"{view_idx + 1}/{source_views}",
                state_gs=state.num_gaussians,
            )

            if debug_view:
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} state.means",
                    state.gaussians.means,
                )
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} state.scales",
                    state.gaussians.scales,
                )
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} state.opacities",
                    state.gaussians.opacities,
                )
                _gir_debug_tensor(
                    f"step={global_step} view={view_idx} state.rotations",
                    state.gaussians.rotations,
                )

            if auxiliary_enabled and view_idx > 0:
                replay_indices = _causal_replay_indices(
                    view_idx,
                    int(cfg.gir_replay_views),
                    "gir_replay_views",
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
                # Match the overfit experiment's replay/current RGB objective.
                # It uses plain MSE rather than the previous Charbonnier-like
                # sqrt(error^2 + eps) objective.
                difference = render_output.color - target
                current_difference = difference[:, -1:]
                current_rgb_losses.append(current_difference.square().mean())
                if point_append_mask is not None:
                    hole_mask = F.interpolate(
                        point_append_mask.float(),
                        size=(low_h, low_w),
                        mode="nearest",
                    ).unsqueeze(1)
                    hole_mask = hole_mask.expand(
                        -1,
                        -1,
                        current_difference.shape[2],
                        -1,
                        -1,
                    ).to(current_difference.dtype)
                    new_hole_rgb_losses.append(
                        (current_difference.square() * hole_mask).sum()
                        / hole_mask.sum().clamp_min(1.0)
                    )
                if replay_indices:
                    replay_rgb_losses.append(
                        difference[:, :-1].square().mean()
                    )

            projected_candidate_count = (
                int(projected_evidence.descriptor.shape[0])
                if projected_evidence is not None
                else 0
            )
            append_selected = (
                int(point_append_mask.sum().item())
                if point_append_mask is not None
                else 0
            )
            log_cuda_memory(
                "gir_view_end",
                step=global_step,
                view=f"{view_idx + 1}/{source_views}",
                state_gs=state.num_gaussians,
                projected_candidates=projected_candidate_count,
                append_selected=append_selected,
            )

            # Truncated BPTT for the streaming map. The loss for the current
            # view has already been built above, so cutting here prevents a
            # later view from backpropagating through older chunks while still
            # allowing gradients within the current chunk. This is separate
            # from gir_historical_detach_mode, which is applied at every view
            # boundary and may only detach historical means.
            tbptt_chunk = int(getattr(cfg, "gir_tbptt_chunk", 0))
            if tbptt_chunk < -1:
                raise ValueError(
                    "gir_tbptt_chunk must be -1, 0, or a positive integer; "
                    f"got {tbptt_chunk}."
                )
            if (
                self.training
                and tbptt_chunk > 0
                and (view_idx + 1) % tbptt_chunk == 0
                and view_idx + 1 < source_views
            ):
                state = state.detach()

        if state is None:
            return encoder_output.gaussians

        if encoder_output.infos is not None:
            encoder_output.infos.pop("gir", None)
            if return_debug_intermediates:
                encoder_output.infos[
                    "gir_debug_intermediates"
                ] = debug_intermediates
            encoder_output.infos["gir_history_views"] = torch.tensor(
                max(source_views - 1, 0), device=features.device
            )
            encoder_output.infos["gir_map_gaussians"] = torch.tensor(
                state.num_gaussians, device=features.device
            )
            encoder_output.infos["gir_old_residual_enabled"] = torch.tensor(
                float(old_residual_enabled), device=features.device
            )
            encoder_output.infos["gir_residual_topk"] = torch.tensor(
                float(residual_topk), device=features.device
            )
            encoder_output.infos["gir_residual_independent_heads"] = torch.tensor(
                float(independent_residual_heads), device=features.device
            )
            encoder_output.infos["gir_old_update_mode_code"] = torch.tensor(
                1.0 if old_update_mode == "projected_shared" else 0.0,
                device=features.device,
            )
            encoder_output.infos["gir_new_residual_enabled"] = torch.tensor(
                float(new_residual_enabled), device=features.device
            )
            encoder_output.infos["gir_new_mean_gradient_enabled"] = torch.tensor(
                float(new_mean_gradient_enabled), device=features.device
            )
            encoder_output.infos["gir_mean_gradient_replay_views"] = torch.tensor(
                float(mean_gradient_replay_views), device=features.device
            )
            encoder_output.infos[
                "gir_new_mean_gradient_replay_views"
            ] = torch.tensor(
                float(new_mean_gradient_replay_views), device=features.device
            )
            encoder_output.infos[
                "gir_old_mean_gradient_current_weight"
            ] = torch.tensor(
                old_mean_gradient_current_weight, device=features.device
            )
            encoder_output.infos[
                "gir_old_mean_gradient_replay_weight"
            ] = torch.tensor(
                old_mean_gradient_replay_weight, device=features.device
            )
            encoder_output.infos[
                "gir_new_mean_gradient_current_weight"
            ] = torch.tensor(
                new_mean_gradient_current_weight, device=features.device
            )
            encoder_output.infos[
                "gir_new_mean_gradient_replay_weight"
            ] = torch.tensor(
                new_mean_gradient_replay_weight, device=features.device
            )
            encoder_output.infos["gir_historical_detach_mode_code"] = torch.tensor(
                {"all": 0.0, "means": 1.0, "none": 2.0}[historical_detach_mode],
                device=features.device,
            )
            encoder_output.infos["gir_tbptt_chunk"] = torch.tensor(
                float(int(getattr(cfg, "gir_tbptt_chunk", 0))),
                device=features.device,
            )
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
                    1.0 - state.num_gaussians / max(unpruned_map_gaussians, 1),
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
            encoder_output.infos["gir_add_gate"] = torch.stack(add_gates).mean()
            encoder_output.infos["gir_add_gate_enabled"] = torch.tensor(
                float(add_gate_enabled), device=features.device
            )
            encoder_output.infos["gir_append_mode_code"] = torch.tensor(
                {"all": 0.0, "point_map": 1.0, "learned": 2.0}[append_mode],
                device=features.device,
            )
            if point_coverage_ratios:
                encoder_output.infos["gir_point_append_depth_check_enabled"] = (
                    features.new_tensor(float(point_append_depth_check))
                )
                encoder_output.infos["gir_point_append_hybrid_enabled"] = (
                    features.new_tensor(float(point_append_hybrid))
                )
                encoder_output.infos[
                    "gir_point_append_current_in_front_enabled"
                ] = features.new_tensor(float(point_append_current_in_front))
                encoder_output.infos[
                    "gir_point_append_history_in_front_enabled"
                ] = features.new_tensor(float(point_append_history_in_front))
                encoder_output.infos["gir_point_coverage_ratio"] = torch.stack(
                    point_coverage_ratios
                ).mean()
                encoder_output.infos["gir_point_append_ratio"] = torch.stack(
                    point_append_ratios
                ).mean()
                encoder_output.infos[
                    "gir_point_front_conflict_ratio"
                ] = torch.stack(point_front_conflict_ratios).mean()
                encoder_output.infos[
                    "gir_point_front_gradient_decay_gate_ratio"
                ] = torch.stack(
                    point_front_gradient_decay_gate_ratios
                ).mean()
                encoder_output.infos[
                    "gir_point_front_decay_mass_ratio"
                ] = torch.stack(point_front_decay_mass_ratios).mean()
                encoder_output.infos["gir_point_depth_scale"] = torch.stack(
                    point_depth_scales
                ).mean()
            encoder_output.infos["gir_add_gate_warmup_progress"] = torch.tensor(
                add_gate_warmup_progress, device=features.device
            )
            if add_rate_ratios:
                add_rate_ratio = torch.stack(add_rate_ratios).mean()
                encoder_output.infos["gir_add_gate_learned"] = torch.stack(
                    learned_add_gates
                ).mean()
                encoder_output.infos["gir_add_rate"] = add_rate_ratio
                encoder_output.infos["gir_add_rate_loss"] = (
                    add_rate_ratio * add_gate_warmup_progress
                )
            if effective_new_ratios:
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
            if projected_candidate_ratios:
                encoder_output.infos[
                    "gir_projected_candidate_ratio"
                ] = torch.stack(projected_candidate_ratios).mean()
                encoder_output.infos[
                    "gir_projected_consistent_ratio"
                ] = torch.stack(projected_consistent_ratios).mean()
                encoder_output.infos[
                    "gir_projected_front_ratio"
                ] = torch.stack(projected_front_ratios).mean()
                encoder_output.infos[
                    "gir_projected_gate_mean"
                ] = torch.stack(projected_gate_means).mean()
                encoder_output.infos[
                    "gir_projected_opacity_shift"
                ] = torch.stack(projected_opacity_shifts).mean()
                encoder_output.infos[
                    "gir_projected_front_decay_probability"
                ] = torch.stack(projected_front_decay_probabilities).mean()
                for key, values in projected_gradient_stats.items():
                    encoder_output.infos[
                        f"gir_projected_{key}"
                    ] = torch.stack(values).mean()
            if raster_alpha_means:
                encoder_output.infos["gir_raster_alpha"] = torch.stack(
                    raster_alpha_means
                ).mean()
            if dominant_weight_means:
                encoder_output.infos["gir_dominant_weight"] = torch.stack(
                    dominant_weight_means
                ).mean()
            if new_residual_stats["candidate_ratio"]:
                for key, values in new_residual_stats.items():
                    encoder_output.infos[
                        f"gir_new_residual_{key}"
                    ] = torch.stack(values).mean()
                # Preserve the original dashboard names while exposing the
                # unambiguous per-channel diagnostics above.
                encoder_output.infos["gir_new_residual_magnitude"] = (
                    encoder_output.infos["gir_new_residual_mean_shift_mean"]
                )
                encoder_output.infos["gir_new_residual_gate"] = (
                    encoder_output.infos["gir_new_residual_gate_mean"]
                )
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
            streaming_rgb_terms = []
            if base_rgb_losses:
                base_rgb_loss = torch.stack(base_rgb_losses).mean()
                encoder_output.infos["gir_base_rgb_loss"] = base_rgb_loss
                streaming_rgb_terms.append(
                    float(getattr(cfg, "gir_base_loss_weight", 1.0))
                    * base_rgb_loss
                )
            if current_rgb_losses:
                current_rgb_loss = torch.stack(current_rgb_losses).mean()
                encoder_output.infos["gir_current_rgb_loss"] = current_rgb_loss
                streaming_rgb_terms.append(
                    float(getattr(cfg, "gir_current_loss_weight", 1.0))
                    * current_rgb_loss
                )
            if new_hole_rgb_losses:
                new_hole_rgb_loss = torch.stack(new_hole_rgb_losses).mean()
                encoder_output.infos["gir_new_hole_rgb_loss"] = (
                    new_hole_rgb_loss
                )
                streaming_rgb_terms.append(
                    float(getattr(cfg, "gir_new_hole_loss_weight", 1.0))
                    * new_hole_rgb_loss
                )
            if replay_rgb_losses:
                replay_rgb_loss = torch.stack(replay_rgb_losses).mean()
                encoder_output.infos["gir_replay_rgb_loss"] = replay_rgb_loss
                streaming_rgb_terms.append(
                    float(getattr(cfg, "gir_replay_loss_weight", 0.5))
                    * replay_rgb_loss
                )
            if streaming_rgb_terms:
                encoder_output.infos["gir_aux_loss"] = sum(streaming_rgb_terms)
            encoder_output.infos["gir_regularization_loss"] = torch.stack(
                regularization_losses
            ).mean() + old_delete_graph_anchor
            if old_delete_enabled:
                if old_delete_budget_losses:
                    old_delete_budget_loss = torch.stack(
                        old_delete_budget_losses
                    ).mean()
                else:
                    old_delete_budget_loss = old_delete_graph_anchor
                encoder_output.infos[
                    "gir_old_delete_budget_loss"
                ] = old_delete_budget_loss + old_delete_graph_anchor
                encoder_output.infos["gir_old_delete_target_ratio"] = torch.tensor(
                    scheduled_old_delete_target,
                    device=features.device,
                )
                encoder_output.infos["gir_old_delete_threshold"] = torch.tensor(
                    old_delete_threshold, device=features.device
                )
                if old_delete_candidate_probabilities:
                    candidate_counts = torch.stack(old_delete_candidate_counts)
                    total_candidates = candidate_counts.sum().clamp_min(1.0)
                    encoder_output.infos[
                        "gir_old_delete_candidate_probability"
                    ] = (
                        torch.stack(old_delete_candidate_probabilities)
                        * candidate_counts
                    ).sum() / total_candidates
                    encoder_output.infos[
                        "gir_old_delete_candidate_count"
                    ] = candidate_counts.mean()
                    encoder_output.infos[
                        "gir_old_delete_candidate_rate"
                    ] = (
                        torch.stack(old_delete_candidate_rates)
                        * candidate_counts
                    ).sum() / total_candidates
                    encoder_output.infos[
                        "gir_old_delete_hard_ratio"
                    ] = torch.stack(old_delete_hard_ratios).mean()
                    encoder_output.infos[
                        "gir_old_delete_removed_opacity_mass_ratio"
                    ] = torch.stack(old_delete_removed_opacity_ratios).mean()
                    encoder_output.infos[
                        "gir_old_delete_map_count_before"
                    ] = torch.stack(old_delete_map_counts_before).mean()
                    encoder_output.infos[
                        "gir_old_delete_map_count_after"
                    ] = torch.stack(old_delete_map_counts_after).mean()
            if old_decay_enabled:
                if old_decay_budget_losses:
                    old_decay_budget_loss = torch.stack(
                        old_decay_budget_losses
                    ).mean()
                else:
                    old_decay_budget_loss = old_delete_graph_anchor
                encoder_output.infos[
                    "gir_old_decay_budget_loss"
                ] = old_decay_budget_loss + old_delete_graph_anchor
                encoder_output.infos["gir_old_decay_topk"] = torch.tensor(
                    float(old_decay_topk), device=features.device
                )
                encoder_output.infos["gir_old_decay_strength"] = torch.tensor(
                    old_decay_strength, device=features.device
                )
                encoder_output.infos[
                    "gir_old_decay_prune_threshold"
                ] = torch.tensor(
                    old_decay_prune_threshold, device=features.device
                )
                if old_decay_candidate_probabilities:
                    candidate_counts = torch.stack(old_decay_candidate_counts)
                    total_candidates = candidate_counts.sum().clamp_min(1.0)
                    encoder_output.infos[
                        "gir_old_decay_candidate_probability"
                    ] = (
                        torch.stack(old_decay_candidate_probabilities)
                        * candidate_counts
                    ).sum() / total_candidates
                    encoder_output.infos[
                        "gir_old_decay_candidate_count"
                    ] = candidate_counts.mean()
                    encoder_output.infos[
                        "gir_old_decay_opacity_mass_ratio"
                    ] = torch.stack(old_decay_opacity_mass_ratios).mean()
                    encoder_output.infos[
                        "gir_old_decay_hard_pruned_ratio"
                    ] = torch.stack(old_decay_hard_ratios).mean()
                    encoder_output.infos[
                        "gir_old_decay_map_count_before"
                    ] = torch.stack(old_decay_map_counts_before).mean()
                    encoder_output.infos[
                        "gir_old_decay_map_count_after"
                    ] = torch.stack(old_decay_map_counts_after).mean()

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
        global_step: int = 0,
        test_add_gate_prune_threshold: float = 0.0,
        test_top1_confidence_mode: str = "inherit",
        test_top1_confidence_floor: float = 0.25,
        return_debug_intermediates: bool = False,
        debug_view_stride: int = 1,
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
                global_step,
                test_add_gate_prune_threshold,
                test_top1_confidence_mode,
                test_top1_confidence_floor,
                return_debug_intermediates,
                debug_view_stride,
            )
        return encoder_output.gaussians

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
            global_step,
        )
        encoder_output.gaussians = gaussians
        log_cuda_memory(
            "before_final_render",
            step=global_step,
            map_gs=int(gaussians.means.shape[1]),
            render_views=v,
        )

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

        render_intrinsics = pred_context_pose["intrinsic"]
        if render_intrinsics.shape[1] == 1 and v > 1:
            render_intrinsics = render_intrinsics.expand(-1, v, -1, -1)
        if render_intrinsics.shape[1] != v:
            raise RuntimeError(
                "Predicted intrinsic view count does not match the render "
                f"sequence: intrinsics={render_intrinsics.shape[1]}, views={v}."
            )
        output = self.decoder.forward(
            gaussians,
            pred_all_extrinsic,
            render_intrinsics,
            torch.ones(b, v, device=device) * near,
            torch.ones(b, v, device=device) * far,
            (h, w),
            "depth",
        )
        log_cuda_memory(
            "after_final_render",
            step=global_step,
            map_gs=int(gaussians.means.shape[1]),
            render_views=v,
        )
        output.depth = output.depth[:, :ctx_img_num, ...]

        return encoder_output, output
