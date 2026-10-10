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
        residual_topk = max(1, int(getattr(cfg, "gir_residual_topk", 1)))
        if residual_topk > 1 and bool(
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
    ) -> Gaussians:
        refine_info = None if encoder_output.infos is None else encoder_output.infos.get("gir")
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
        if residual_topk > 1 and not use_raster_evidence:
            raise ValueError(
                "Top-k historical residuals require "
                "gir_raster_evidence_enabled=true."
            )
        if old_decay_enabled and old_decay_topk > 1 and not use_raster_evidence:
            raise ValueError(
                "Top-k old-GS decay requires gir_raster_evidence_enabled=true."
            )
        use_dominant_ids = bool(getattr(cfg, "gir_dominant_id_enabled", False))
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
        add_gate_enabled = bool(
            getattr(cfg, "gir_add_gate_enabled", True)
        )
        new_residual_enabled = bool(
            getattr(cfg, "gir_new_residual_enabled", True)
        )
        historical_detach_mode = str(
            getattr(cfg, "gir_historical_detach_mode", "means")
        ).lower()
        if historical_detach_mode not in {"all", "means", "none"}:
            raise ValueError(
                "GIR historical detach mode must be one of all, means, none; "
                f"got {historical_detach_mode!r}."
            )
        if not add_gate_enabled and prune_threshold > 0.0:
            raise ValueError(
                "GIR test pruning requires gir_add_gate_enabled=true."
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

        # Match the overfit experiment: camera/depth geometry is a frozen
        # condition for the streaming GIR rollout. The camera head is trained
        # only by the explicit GT camera loss in ModelWrapper. Without this
        # detach, the auxiliary/replay RGB renders send unstable rasterizer
        # gradients back into camera_mlp_head at step zero.
        pred_all_extrinsic = pred_all_extrinsic.detach()
        intrinsics = intrinsics.detach()

        depth = refine_info["depth"]
        depth_confidence = refine_info["depth_conf"]
        state: Optional[StreamingGaussianState] = None
        # Streaming RGB supervision is kept as three separate terms to match
        # the overfit experiment: initial map, newly arrived view, and replay.
        base_rgb_losses = []
        current_rgb_losses = []
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
        new_residual_magnitudes = []
        new_residual_gates = []
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
            current_depth_confidence = depth_confidence[:, view_idx].permute(0, 3, 1, 2)
            current_gaussians = self._slice_gaussian_view(
                encoder_output.gaussians,
                view_idx,
                gaussians_per_view,
            )
            has_history = state is not None

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

            prediction = self.gir_update_head(
                current_feature,
                current_rgb,
                current_depth,
                current_depth_confidence,
                residual_gir,
                num_historical_predictions=(
                    residual_topk
                    if independent_residual_heads and old_residual_enabled
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

                if new_residual_enabled:
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
                else:
                    # Keep the disabled current branch connected to the graph
                    # with zero weight, avoiding an unused DDP head while
                    # guaranteeing that no new-GS residual is written back.
                    new_residual_energy = (
                        prediction.current_delta_mean_camera.square().sum()
                        + prediction.current_delta_rotation.square().sum()
                        + prediction.current_delta_log_scale.square().sum()
                        + prediction.current_delta_opacity_logit.square().sum()
                        + prediction.current_delta_harmonics.square().sum()
                    ) * 0.0
                new_residual_magnitudes.append(new_residual_energy.mean().sqrt())
                new_residual_gates.append(
                    prediction.current_residual_gate.sigmoid().mean()
                )
            else:
                new_residual_energy = prediction.add_logit.new_zeros(())

            coverage = gir.valid.to(prediction.add_logit.dtype)
            residual_coverage = residual_gir.valid.to(
                prediction.add_logit.dtype
            )
            visible_ratios.append(coverage.mean())
            if state is None or not add_gate_enabled:
                # Frame zero and residual-only experiments keep every new GS.
                learned_add_gate_low = torch.ones_like(prediction.add_logit)
                add_gate_low = torch.ones_like(prediction.add_logit) + (
                    0.0 * prediction.add_logit
                )
                if state is not None:
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
            historical_gates.append(
                (prediction.historical_gate.sigmoid() * coverage).sum()
                / coverage.sum().clamp_min(1.0)
            )
            valid_normalizer = coverage.sum().clamp_min(1.0)
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
                    prediction.delta_mean_camera.square().sum(dim=1, keepdim=True)
                    + prediction.delta_rotation.square().sum(dim=1, keepdim=True)
                    + prediction.delta_log_scale.square().sum(dim=1, keepdim=True)
                    + prediction.delta_opacity_logit.square()
                    + prediction.delta_harmonics.square().mean(dim=1, keepdim=True)
                )
            residual_magnitudes.append(residual_energy.mean().sqrt())
            old_residual_regularization = (
                (residual_energy * residual_coverage).sum()
                / residual_coverage.sum().clamp_min(1.0)
                if old_residual_enabled
                else residual_energy.sum() * 0.0
            )
            regularization_losses.append(
                old_residual_regularization + new_residual_energy.mean()
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
                # Match the overfit experiment's replay/current RGB objective.
                # It uses plain MSE rather than the previous Charbonnier-like
                # sqrt(error^2 + eps) objective.
                difference = render_output.color - target
                current_rgb_losses.append(difference[:, -1:].square().mean())
                if replay_indices:
                    replay_rgb_losses.append(
                        difference[:, :-1].square().mean()
                    )

            # Truncated BPTT for the streaming map. The loss for the current
            # view has already been built above, so cutting here prevents a
            # later view from backpropagating through older chunks while still
            # allowing gradients within the current chunk. This is separate
            # from gir_historical_detach_mode, which is applied at every view
            # boundary and may only detach historical means.
            tbptt_chunk = max(0, int(getattr(cfg, "gir_tbptt_chunk", 0)))
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
            encoder_output.infos["gir_new_residual_enabled"] = torch.tensor(
                float(new_residual_enabled), device=features.device
            )
            encoder_output.infos["gir_historical_detach_mode_code"] = torch.tensor(
                {"all": 0.0, "means": 1.0, "none": 2.0}[historical_detach_mode],
                device=features.device,
            )
            encoder_output.infos["gir_tbptt_chunk"] = torch.tensor(
                float(max(0, int(getattr(cfg, "gir_tbptt_chunk", 0)))),
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
            pred_all_extrinsic.detach(),
            render_intrinsics.detach(),
            torch.ones(b, v, device=device) * near,
            torch.ones(b, v, device=device) * far,
            (h, w),
            "depth",
        )
        output.depth = output.depth[:, :ctx_img_num, ...]

        return encoder_output, output
