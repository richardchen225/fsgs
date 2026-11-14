from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from lpips import LPIPS
from omegaconf import DictConfig, OmegaConf
from torch.utils.checkpoint import checkpoint

from src.config import load_typed_root_config
from src.evaluation.metrics import compute_lpips
from src.geometry.sim3 import (
    apply_camera_sim3,
    apply_point_sim3,
    estimate_camera_sim3,
)
from src.experiments.overfit_old_gs_residual import (
    _build_base_gaussians,
    _cache_backbone_tokens,
    _cache_dav3_depth_targets,
    _extract_features,
    _load_checkpoint,
    _load_toy_data,
    _normalized_intrinsic,
    _predict_camera_geometry,
    _psnr,
    _render,
    _save_image,
    _save_mask,
    _sequence_scale_only_log_depth_loss,
)
from src.model.model_wrapper import (
    _camera_supervision_losses as _direct_camera_supervision_losses,
)
from src.model.analytic_gaussians import build_analytic_gaussian_anchors
from src.model.encoder.vggt.utils.rotation import mat_to_quat, quat_to_mat
from src.model.model import get_model
from src.model.ply_export import export_ply
from src.model.streaming_gir import (
    StreamingGaussianState,
    SymmetricGSResidualHead,
    apply_symmetric_multiview_residual,
    attach_symmetric_multiview_visual_evidence,
    build_point_map_append_mask,
    build_symmetric_multiview_gradient_evidence,
    build_symmetric_multiview_visual_evidence,
)
from src.model.types import Gaussians


def _training_lpips_mean(
    lpips_model: LPIPS,
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int,
    gradient_checkpointing: bool,
) -> torch.Tensor:
    """Compute the exact image-batch LPIPS mean with a bounded VGG peak."""
    if rendered.shape != target.shape or rendered.ndim != 4:
        raise ValueError(
            "Training LPIPS expects matching [N,3,H,W] tensors, got "
            f"{tuple(rendered.shape)} and {tuple(target.shape)}."
        )
    if rendered.shape[0] == 0:
        raise ValueError("Training LPIPS requires at least one image.")
    chunk_size = max(1, int(chunk_size))

    def evaluate(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return lpips_model.forward(prediction, reference, normalize=True)

    total = rendered.new_zeros(())
    for start in range(0, rendered.shape[0], chunk_size):
        stop = min(start + chunk_size, rendered.shape[0])
        prediction = rendered[start:stop]
        reference = target[start:stop]
        if gradient_checkpointing and prediction.requires_grad:
            values = checkpoint(
                evaluate,
                prediction,
                reference,
                use_reentrant=False,
            )
        else:
            values = evaluate(prediction, reference)
        total = total + values.sum()
    return total / rendered.shape[0]


def _requires_sim3_alignment(
    source_render_camera: str,
    heldout_test_camera: str,
    heldout_loss_enabled: bool,
    alignment_mode: str = "auto",
    geometry_camera_source: str = "predicted",
    heldout_loss_camera: str = "gt",
) -> bool:
    """Select whether the predicted reconstruction is moved into the GT gauge."""
    alignment_mode = str(alignment_mode).lower()
    if alignment_mode == "always":
        return True
    if alignment_mode == "never":
        return False
    if alignment_mode != "auto":
        raise ValueError(
            "offline.sim3_alignment_mode must be auto, always, or never, got "
            f"{alignment_mode!r}."
        )
    return (
        geometry_camera_source == "gt"
        or source_render_camera == "gt"
        or heldout_test_camera == "gt"
        or (heldout_loss_enabled and heldout_loss_camera == "gt")
    )


def _camera_error_metrics(
    predicted_c2w: torch.Tensor,
    predicted_intrinsic: torch.Tensor,
    gt_c2w: torch.Tensor,
    gt_intrinsic: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    center_error = (
        predicted_c2w[..., :3, 3] - gt_c2w[..., :3, 3]
    ).norm(dim=-1)
    relative_rotation = (
        predicted_c2w[..., :3, :3].transpose(-1, -2) @ gt_c2w[..., :3, :3]
    )
    rotation_angle = torch.acos(
        ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
        .clamp(-1.0, 1.0)
    )
    predicted_focal = predicted_intrinsic[..., (0, 1), (0, 1)]
    gt_focal = gt_intrinsic[..., (0, 1), (0, 1)]
    focal_abs_rel = (
        (predicted_focal - gt_focal).abs() / gt_focal.abs().clamp_min(1e-6)
    )
    return {
        f"{prefix}_center_rmse": float(center_error.square().mean().sqrt().item()),
        f"{prefix}_rotation_deg_mean": float(
            torch.rad2deg(rotation_angle).mean().item()
        ),
        f"{prefix}_focal_abs_rel": float(focal_abs_rel.mean().item()),
    }


def _camera_points_from_depth(
    depth: torch.Tensor,
    intrinsic_px: torch.Tensor,
) -> torch.Tensor:
    batch, views, height, width = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = (
        (xx[None, None] - intrinsic_px[..., 0, 2, None, None])
        * depth
        / intrinsic_px[..., 0, 0, None, None].clamp_min(1e-6)
    )
    y = (
        (yy[None, None] - intrinsic_px[..., 1, 2, None, None])
        * depth
        / intrinsic_px[..., 1, 1, None, None].clamp_min(1e-6)
    )
    return torch.stack([x, y, depth], dim=-1)


def _concatenate_gaussian_anchors(
    model,
    features: torch.Tensor,
    images: torch.Tensor,
    depth: torch.Tensor,
    predicted_c2w: torch.Tensor,
    predicted_intrinsic_px: torch.Tensor,
    sim3_rotation: torch.Tensor,
    sim3_translation: torch.Tensor,
    sim3_scale: torch.Tensor,
    cfg: DictConfig,
    gaussian_scale_factor: torch.Tensor | None = None,
) -> Gaussians:
    batch, views, _, height, width = images.shape
    flattened_batch = batch * views
    parameter_source = str(cfg.gaussian_parameter_source).lower()
    if parameter_source == "analytic":
        camera_points = _camera_points_from_depth(depth, predicted_intrinsic_px)
        homogeneous = torch.cat(
            [camera_points, torch.ones_like(camera_points[..., :1])], dim=-1
        )
        predicted_world_points = torch.einsum(
            "bvij,bvhwj->bvhwi", predicted_c2w, homogeneous
        )[..., :3]
        base_opacity = float(cfg.get("analytic_base_opacity", 0.5))
        if not 0.0 < base_opacity < 1.0:
            raise ValueError("offline.analytic_base_opacity must be in (0,1).")
        anchors = build_analytic_gaussian_anchors(
            predicted_world_points.reshape(flattened_batch, height * width, 3),
            images.reshape(flattened_batch, 3, height, width),
            depth.reshape(flattened_batch, height, width),
            predicted_intrinsic_px.reshape(flattened_batch, 3, 3),
            base_opacity=base_opacity,
            scale_multiplier=float(cfg.get("analytic_scale_multiplier", 1.0)),
            max_scale=float(cfg.get("analytic_max_scale", 0.1)),
        )
    elif parameter_source == "gs_head":
        anchors = _build_base_gaussians(
            model,
            features.reshape(flattened_batch, *features.shape[2:]),
            images.reshape(flattened_batch, 3, height, width),
            depth.reshape(flattened_batch, height, width),
            predicted_c2w.reshape(flattened_batch, 4, 4),
            predicted_intrinsic_px.reshape(flattened_batch, 3, 3),
        )
    else:
        raise ValueError(
            "offline.gaussian_parameter_source must be 'analytic' or "
            f"'gs_head', got {parameter_source!r}."
        )

    def merge_views(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch, views * height * width, *value.shape[2:])

    means = anchors.means.reshape(batch, views, height * width, 3)
    means = apply_point_sim3(
        means, sim3_rotation, sim3_translation, sim3_scale
    )
    scales = anchors.scales.reshape(batch, views, height * width, 3)
    if gaussian_scale_factor is None:
        gaussian_scale_factor = sim3_scale
    scales = scales * gaussian_scale_factor[:, None, None, None]
    rotation_matrices = quat_to_mat(
        anchors.rotations.reshape(batch, views, height * width, 4).float()
    )
    aligned_rotation_matrices = torch.einsum(
        "bij,bvnjk->bvnik", sim3_rotation.float(), rotation_matrices
    )
    rotations = mat_to_quat(aligned_rotation_matrices).reshape(
        batch, views * height * width, 4
    )
    return Gaussians(
        means=means.reshape(batch, views * height * width, 3),
        harmonics=merge_views(anchors.harmonics),
        opacities=merge_views(anchors.opacities),
        scales=scales.reshape(batch, views * height * width, 3),
        rotations=rotations,
    )


def _make_state(gaussians: Gaussians) -> StreamingGaussianState:
    batch, count = gaussians.opacities.shape
    stable_ids = torch.arange(count, device=gaussians.means.device).expand(batch, -1)
    return StreamingGaussianState(
        gaussians=gaussians,
        stable_ids=stable_ids,
        observation_count=torch.zeros_like(gaussians.opacities),
        point_map=gaussians.means,
    )


class SceneGaussianKeepGate(torch.nn.Module):
    """Scene-specific differentiable keep probabilities for a fixed all-append map."""

    def __init__(self, max_gaussians: int, initial_probability: float) -> None:
        super().__init__()
        if max_gaussians <= 0:
            raise ValueError("Keep gate requires at least one Gaussian.")
        probability = float(initial_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("Initial keep probability must be in (0, 1).")
        initial_logit = math.log(probability / (1.0 - probability))
        self.logits = torch.nn.Parameter(
            torch.full((max_gaussians,), initial_logit, dtype=torch.float32)
        )

    def forward(
        self,
        gaussian_count: int,
        batch_size: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if gaussian_count > self.logits.numel():
            raise ValueError(
                "Keep gate is smaller than the rolled-out map: "
                f"gate={self.logits.numel()}, map={gaussian_count}."
            )
        keep = self.logits[:gaussian_count].sigmoid().to(dtype=dtype)
        return keep.unsqueeze(0).expand(batch_size, -1)


def _apply_scene_keep_gate(
    state: StreamingGaussianState,
    keep_gate: SceneGaussianKeepGate | None,
) -> tuple[StreamingGaussianState, torch.Tensor]:
    if keep_gate is None:
        keep = torch.ones_like(state.gaussians.opacities)
        return state, keep
    keep = keep_gate(
        state.num_gaussians,
        state.batch_size,
        dtype=state.gaussians.opacities.dtype,
    )
    gated = StreamingGaussianState(
        gaussians=Gaussians(
            means=state.gaussians.means,
            harmonics=state.gaussians.harmonics,
            opacities=state.gaussians.opacities * keep,
            scales=state.gaussians.scales,
            rotations=state.gaussians.rotations,
        ),
        stable_ids=state.stable_ids,
        observation_count=state.observation_count,
        point_map=state.point_map,
        front_conflict_count=state.front_conflict_count,
        split_generation=state.split_generation,
    )
    return gated, keep


def _slice_gaussians(
    gaussians: Gaussians, start: int, end: int
) -> Gaussians:
    return Gaussians(
        means=gaussians.means[:, start:end],
        harmonics=gaussians.harmonics[:, start:end],
        opacities=gaussians.opacities[:, start:end],
        scales=gaussians.scales[:, start:end],
        rotations=gaussians.rotations[:, start:end],
    )


def _concatenate_anchor_views(anchors: list[Gaussians]) -> Gaussians:
    if not anchors:
        raise ValueError("At least one Gaussian anchor view is required.")
    return Gaussians(
        means=torch.cat([anchor.means for anchor in anchors], dim=1),
        harmonics=torch.cat([anchor.harmonics for anchor in anchors], dim=1),
        opacities=torch.cat([anchor.opacities for anchor in anchors], dim=1),
        scales=torch.cat([anchor.scales for anchor in anchors], dim=1),
        rotations=torch.cat([anchor.rotations for anchor in anchors], dim=1),
    )


def _build_source_anchor_views(
    model,
    cfg: DictConfig,
    anchor_inputs: dict[str, torch.Tensor],
    start: int = 0,
    end: int | None = None,
) -> list[Gaussians]:
    """Generate a contiguous range of per-view anchors with live head gradients."""
    images = anchor_inputs["images"]
    source_count = images.shape[1]
    if end is None:
        end = source_count
    if not 0 <= start < end <= source_count:
        raise ValueError(
            "Invalid source-anchor view range: "
            f"start={start}, end={end}, source_count={source_count}."
        )
    anchors = _concatenate_gaussian_anchors(
        model,
        anchor_inputs["features"][:, start:end],
        images[:, start:end],
        anchor_inputs["depth"][:, start:end],
        anchor_inputs["predicted_c2w"][:, start:end],
        anchor_inputs["predicted_intrinsic_px"][:, start:end],
        anchor_inputs["sim3_rotation"],
        anchor_inputs["sim3_translation"],
        anchor_inputs["sim3_scale"],
        cfg,
        gaussian_scale_factor=anchor_inputs.get("gaussian_scale_factor"),
    )
    height, width = images.shape[-2:]
    anchors_per_view = height * width
    return [
        _slice_gaussians(
            anchors,
            view_idx * anchors_per_view,
            (view_idx + 1) * anchors_per_view,
        )
        for view_idx in range(end - start)
    ]


def _source_prefix(data: dict, end: int) -> dict:
    return {
        "ids": data["ids"][:end],
        "images": data["images"][:, :end],
        "c2w": data["c2w"][:, :end],
        "intrinsic": data["intrinsic"][:, :end],
        "image_shape": data["image_shape"],
    }


def _select_replay_indices(view_idx: int, replay_views: int) -> list[int]:
    if view_idx <= 0 or replay_views == 0:
        return []
    start = 0 if replay_views < 0 else max(0, view_idx - replay_views)
    return list(range(start, view_idx))


def _append_point_map_view(
    model,
    cfg,
    state: StreamingGaussianState,
    current: Gaussians,
    view_idx: int,
    current_depth: torch.Tensor,
    current_c2w: torch.Tensor,
    current_intrinsic: torch.Tensor,
    image_shape: tuple[int, int],
) -> tuple[StreamingGaussianState, dict[str, float]]:
    raster = model.decoder.render_projected_gs_evidence(
        state.gaussians,
        current_c2w,
        current_intrinsic,
        image_shape,
    )
    append_mask, _, front_conflict, next_conflict_count, depth_scale = (
        build_point_map_append_mask(
            state=state,
            current_depth=current_depth[:, None],
            camera_to_world=current_c2w,
            intrinsics=current_intrinsic,
            old_rgb=raster.color,
            old_depth=raster.expected_depth,
            old_alpha=raster.alpha,
            current_depth_confidence=torch.ones_like(current_depth[:, None]),
            depth_relative_tolerance=float(
                cfg.get("point_append_depth_relative_tolerance", 0.15)
            ),
            rgb_bad_threshold=float(cfg.get("point_append_rgb_bad_threshold", 0.08)),
            old_alpha_threshold=float(
                cfg.get("point_append_old_alpha_threshold", 0.10)
            ),
            min_depth_confidence=float(
                cfg.get("point_append_min_depth_confidence", 0.0)
            ),
            dilation_radius=int(cfg.get("point_append_dilation_radius", 2)),
            chunk_size=int(cfg.get("point_append_chunk_size", 65536)),
            depth_check_enabled=bool(
                cfg.get("point_append_depth_check_enabled", False)
            ),
            hybrid_enabled=bool(cfg.get("point_append_hybrid_enabled", True)),
            append_current_in_front_enabled=bool(
                cfg.get("point_append_current_in_front_enabled", True)
            ),
            append_history_in_front_enabled=bool(
                cfg.get("point_append_history_in_front_enabled", True)
            ),
            front_conflict_confirm_views=int(
                cfg.get("point_append_front_conflict_confirm_views", 1)
            ),
            max_append_ratio=float(cfg.get("point_append_max_ratio", 1.0)),
        )
    )
    state = state.with_front_conflict_count(next_conflict_count)
    state = state.decay_masked_opacity(
        front_conflict,
        float(cfg.get("point_append_front_conflict_opacity_decay", 0.70)),
    )
    previous_count = state.num_gaussians
    state = state.append(
        current,
        torch.ones_like(current.opacities),
        append_mask=append_mask,
        point_map=current.means,
    )
    point_appended_count = state.num_gaussians - previous_count

    periodic_appended_count = 0
    periodic_keyframe = False
    periodic_enabled = bool(cfg.get("periodic_all_append_enabled", False))
    periodic_interval = int(cfg.get("periodic_all_append_interval", 0))
    if (
        periodic_enabled
        and periodic_interval > 0
        and (view_idx + 1) % periodic_interval == 0
    ):
        periodic_keyframe = True
        valid = torch.isfinite(current_depth) & (current_depth > 1e-5)
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape != append_mask.shape:
            raise ValueError(
                "Periodic append validity must match the point-map mask; got "
                f"{tuple(valid.shape)} and {tuple(append_mask.shape)}."
            )
        refinement_mask = valid & ~append_mask.bool()
        refinement = Gaussians(
            means=current.means,
            harmonics=current.harmonics,
            opacities=current.opacities
            * float(cfg.get("periodic_all_append_opacity_scale", 0.35)),
            scales=current.scales
            * float(cfg.get("periodic_all_append_scale_multiplier", 0.75)),
            rotations=current.rotations,
        )
        before_refinement = state.num_gaussians
        state = state.append(
            refinement,
            torch.ones_like(refinement.opacities),
            append_mask=refinement_mask,
            point_map=current.means,
        )
        periodic_appended_count = state.num_gaussians - before_refinement
        combined_append_mask = append_mask.bool() | refinement_mask
    else:
        combined_append_mask = append_mask.bool()

    appended_count = state.num_gaussians - previous_count
    return state, {
        "append_ratio": float(combined_append_mask.float().mean().item()),
        "appended_count": float(appended_count),
        "point_appended_count": float(point_appended_count),
        "periodic_appended_count": float(periodic_appended_count),
        "periodic_keyframe": float(periodic_keyframe),
        "point_depth_scale": float(depth_scale.mean().item()),
    }


def _append_view(
    model,
    cfg,
    state: StreamingGaussianState,
    current: Gaussians,
    view_idx: int,
    current_depth: torch.Tensor,
    current_c2w: torch.Tensor,
    current_intrinsic: torch.Tensor,
    image_shape: tuple[int, int],
) -> tuple[StreamingGaussianState, dict[str, float]]:
    append_mode = str(cfg.get("append_mode", "point_map")).lower()
    if append_mode == "point_map":
        return _append_point_map_view(
            model,
            cfg,
            state,
            current,
            view_idx,
            current_depth,
            current_c2w,
            current_intrinsic,
            image_shape,
        )
    if append_mode != "all":
        raise ValueError(
            "offline.append_mode must be point_map or all, got "
            f"{append_mode!r}."
        )

    valid = torch.isfinite(current_depth) & (current_depth > 1e-5)
    previous_count = state.num_gaussians
    state = state.append(
        current,
        torch.ones_like(current.opacities),
        append_mask=valid,
        point_map=current.means,
    )
    return state, {
        "append_ratio": float(valid.float().mean().item()),
        "appended_count": float(state.num_gaussians - previous_count),
        "point_appended_count": 0.0,
        "periodic_appended_count": 0.0,
        "periodic_keyframe": 0.0,
        "point_depth_scale": 1.0,
    }


def _build_gradient_evidence(model, cfg, state, data):
    active = state.gaussians.opacities > 0
    gradients = model.decoder.render_parameter_gradients(
        state.gaussians,
        data["c2w"],
        data["intrinsic"],
        data["images"],
        data["image_shape"],
        active_mask=active,
        mean_camera_enabled=True,
        harmonics_enabled=True,
        loss_reduction="mean",
    )
    if gradients.mean_camera is None or gradients.harmonics is None:
        raise RuntimeError("Offline evidence requires mean and harmonics gradients.")
    evidence = build_symmetric_multiview_gradient_evidence(
        gradients.opacity_logit,
        gradients.mean_camera,
        gradients.harmonics,
        active_mask=active,
        opacity_gradient_scale=float(cfg.opacity_gradient_scale),
        opacity_gradient_clip=float(cfg.opacity_gradient_clip),
        opacity_gradient_threshold=float(cfg.opacity_gradient_threshold),
        mean_gradient_scale=float(cfg.mean_gradient_scale),
        mean_gradient_clip=float(cfg.mean_gradient_clip),
        mean_gradient_threshold=float(cfg.mean_gradient_threshold),
        harmonics_gradient_scale=float(cfg.harmonics_gradient_scale),
        harmonics_gradient_clip=float(cfg.harmonics_gradient_clip),
        harmonics_gradient_threshold=float(cfg.harmonics_gradient_threshold),
    )
    render_cache = {
        "rgb": gradients.rendered_color,
        "depth": gradients.rendered_depth,
        "alpha": gradients.rendered_alpha,
    }
    return evidence, render_cache


def _attach_visual_evidence(head, cfg, state, data, evidence, render_cache):
    if not bool(cfg.visual_enabled):
        return evidence
    if head.visual_encoder is None:
        raise RuntimeError("Visual evidence is enabled but its encoder is missing.")
    candidate_mode = str(
        cfg.get("visual_candidate_mode", "gradient_active")
    ).lower()
    if candidate_mode == "gradient_active":
        visual_active_mask = evidence.active_mask
    elif candidate_mode == "all_active":
        visual_active_mask = state.gaussians.opacities > 0
    else:
        raise ValueError(
            "offline.visual_candidate_mode must be gradient_active or "
            f"all_active, got {candidate_mode!r}."
        )
    visual, view_count, weight_sum = build_symmetric_multiview_visual_evidence(
        state.gaussians,
        data["images"],
        render_cache["rgb"],
        render_cache["depth"],
        render_cache["alpha"],
        data["c2w"],
        data["intrinsic"],
        head.visual_encoder,
        active_mask=visual_active_mask,
        chunk_size=int(cfg.chunk_size),
        image_scale=float(cfg.visual_image_scale),
        depth_temperature=float(cfg.visual_depth_temperature),
        min_alpha=float(cfg.visual_min_alpha),
    )
    return attach_symmetric_multiview_visual_evidence(
        evidence, visual, view_count, weight_sum
    )


def _predict_corrected_state(head, cfg, base_state, evidence, reference_c2w):
    prediction = head(
        base_state.gaussians,
        evidence.visible_view_count,
        base_state.split_generation,
        evidence,
        torch.zeros_like(base_state.stable_ids, dtype=torch.bool),
    )
    return apply_symmetric_multiview_residual(
        base_state,
        evidence,
        prediction,
        reference_c2w,
        max_mean_step_scale=float(cfg.max_mean_step_scale),
        max_rotation_step=float(cfg.max_rotation_step),
        max_log_scale_step=float(cfg.max_log_scale_step),
        max_opacity_logit_step=float(cfg.max_opacity_logit_step),
        max_harmonics_step=float(cfg.max_harmonics_step),
        unbounded_updates=bool(cfg.get("unbounded_residual_updates", False)),
    )


def _apply_incremental_residual(
    model,
    head,
    cfg,
    state: StreamingGaussianState,
    render_data: dict,
    view_idx: int,
) -> tuple[StreamingGaussianState, dict[str, torch.Tensor]]:
    if not bool(cfg.get("residual_enabled", True)):
        return state, _mean_update_stats([], state.gaussians.means)
    prefix = _source_prefix(render_data, view_idx + 1)
    gradient_evidence, render_cache = _build_gradient_evidence(
        model, cfg, state, prefix
    )
    evidence = _attach_visual_evidence(
        head, cfg, state, prefix, gradient_evidence, render_cache
    )
    updated, stats = _predict_corrected_state(
        head,
        cfg,
        state,
        evidence,
        render_data["c2w"][:, 0],
    )
    if bool(cfg.get("point_map_follow_mean_updates_enabled", True)):
        updated = updated.sync_point_map_to_means()
    return updated, stats


def _mean_update_stats(
    stats_per_view: list[dict[str, torch.Tensor]],
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not stats_per_view:
        return {
            key: reference.new_zeros(())
            for key in (
                "active_count",
                "active_ratio",
                "visible_views_mean",
                "mean_coherence",
                "mean_conflict_ratio",
                "gate_mean",
                "mean_shift_mean",
                "rotation_shift_mean",
                "scale_relative_shift_mean",
                "opacity_shift_mean",
                "harmonics_shift_mean",
                "visual_visible_views_mean",
                "visual_weight_mean",
                "visual_feature_norm",
                "regularization",
            )
        }
    return {
        key: torch.stack([stats[key].float() for stats in stats_per_view]).mean()
        for key in stats_per_view[0]
    }


def _rollout_incremental(
    model,
    head,
    cfg,
    anchors: list[Gaussians],
    geometry_data: dict,
    render_data: dict,
    *,
    apply_residual: bool,
) -> tuple[StreamingGaussianState, dict[str, torch.Tensor], dict[str, float]]:
    state = _make_state(anchors[0]).detach()
    stats_per_view = []
    appended_count = 0.0
    point_appended_count = 0.0
    periodic_appended_count = 0.0
    periodic_keyframes = 0.0
    append_ratios = []
    for view_idx in range(len(anchors)):
        if view_idx > 0:
            state, append_stats = _append_view(
                model,
                cfg,
                state,
                anchors[view_idx],
                view_idx,
                geometry_data["depth"][:, view_idx],
                geometry_data["c2w"][:, view_idx],
                geometry_data["intrinsic"][:, view_idx],
                render_data["image_shape"],
            )
            appended_count += append_stats["appended_count"]
            point_appended_count += append_stats["point_appended_count"]
            periodic_appended_count += append_stats["periodic_appended_count"]
            periodic_keyframes += append_stats["periodic_keyframe"]
            append_ratios.append(append_stats["append_ratio"])
        if apply_residual:
            state, stats = _apply_incremental_residual(
                model, head, cfg, state, render_data, view_idx
            )
            stats_per_view.append(stats)
        state = state.detach()
    stats = _mean_update_stats(
        stats_per_view, state.gaussians.means
    )
    append_summary = {
        "append_ratio": (
            float(sum(append_ratios) / len(append_ratios))
            if append_ratios
            else 0.0
        ),
        "appended_count": appended_count,
        "point_appended_count": point_appended_count,
        "periodic_appended_count": periodic_appended_count,
        "periodic_keyframes": periodic_keyframes,
    }
    return state, stats, append_summary


def _as_bvhw(value: torch.Tensor, batch: int, views: int) -> torch.Tensor:
    if value.ndim == 4 and value.shape[:2] == (batch, views):
        return value
    if value.ndim == 5 and value.shape[:3] == (batch, views, 1):
        return value[:, :, 0]
    raise ValueError(
        "Expected scalar render output [B,V,H,W] or [B,V,1,H,W], got "
        f"{tuple(value.shape)}."
    )


def _metrics_for_views(
    model,
    gaussians,
    views,
    image_shape,
    prefix,
    *,
    compute_perceptual: bool = False,
    perceptual_device: str = "cpu",
):
    if views["rgb"].shape[1] == 0:
        return {}, None, None
    rendering = _render(
        model, gaussians, views["c2w"], views["intrinsic"], image_shape
    )
    rendered = rendering.color
    if rendering.alpha is None:
        raise RuntimeError("Offline evaluation requires decoder alpha output.")
    alpha = _as_bvhw(rendering.alpha, rendered.shape[0], rendered.shape[1])
    alpha = torch.nan_to_num(alpha.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    metrics = {
        f"{prefix}_psnr": _psnr(rendered, views["rgb"]),
        f"{prefix}_alpha_mean": float(alpha.mean().item()),
        f"{prefix}_alpha_coverage_05": float((alpha >= 0.05).float().mean().item()),
        f"{prefix}_alpha_coverage_50": float((alpha >= 0.50).float().mean().item()),
        f"{prefix}_white_hole_ratio": float((alpha < 0.05).float().mean().item()),
    }
    for index, frame_id in enumerate(views["ids"]):
        metrics[f"{prefix}_frame_{int(frame_id):03d}_psnr"] = _psnr(
            rendered[:, index], views["rgb"][:, index]
        )
        metrics[f"{prefix}_frame_{int(frame_id):03d}_alpha_mean"] = float(
            alpha[:, index].mean().item()
        )
        metrics[f"{prefix}_frame_{int(frame_id):03d}_white_hole_ratio"] = float(
            (alpha[:, index] < 0.05).float().mean().item()
        )
    if compute_perceptual:
        device = torch.device(perceptual_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA LPIPS evaluation requested without CUDA.")
        per_view_lpips = []
        for index, frame_id in enumerate(views["ids"]):
            value = compute_lpips(
                views["rgb"][:, index].detach().float().to(device),
                rendered[:, index].detach().float().to(device),
            ).mean()
            frame_lpips = float(value.item())
            metrics[f"{prefix}_frame_{int(frame_id):03d}_lpips"] = frame_lpips
            per_view_lpips.append(frame_lpips)
        metrics[f"{prefix}_lpips"] = float(
            sum(per_view_lpips) / len(per_view_lpips)
        )
    return metrics, rendered, alpha


def _read_jsonl(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    by_step = {int(row["step"]): row for row in rows}
    return [by_step[step] for step in sorted(by_step)]


def _read_csv(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(handle)
        ]


def _record_training_metrics(output_dir: Path, row: dict[str, float]) -> None:
    path = output_dir / "metrics.csv"
    fieldnames = list(row)
    write_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _plot_metrics(output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluations = _read_jsonl(output_dir / "eval_metrics.jsonl")
    training = _read_csv(output_dir / "metrics.csv")
    figure, axes = plt.subplots(2, 4, figsize=(24, 9), constrained_layout=True)
    (
        psnr_ax,
        lpips_ax,
        loss_ax,
        alpha_ax,
        residual_ax,
        geometry_ax,
        gradient_ax,
        unused_ax,
    ) = axes.ravel()

    def plot_eval(axis, key, label, scale=1.0):
        points = [
            (row["step"], scale * row[key])
            for row in evaluations
            if key in row and math.isfinite(float(row[key]))
        ]
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=label,
            )

    def plot_train(axis, key, label):
        points = [
            (row["step"], row[key])
            for row in training
            if key in row and math.isfinite(float(row[key]))
        ]
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                linewidth=1.0,
                label=label,
            )

    plot_eval(psnr_ax, "base_source_psnr", "Base source")
    plot_eval(psnr_ax, "source_psnr", "Updated source")
    plot_eval(psnr_ax, "heldout_loss_psnr", "Heldout loss")
    plot_eval(psnr_ax, "heldout_test_psnr", "Heldout test")
    psnr_ax.set_title("PSNR")
    psnr_ax.set_ylabel("dB")

    plot_eval(lpips_ax, "heldout_loss_lpips", "Heldout loss")
    plot_eval(lpips_ax, "heldout_test_lpips", "Heldout test")
    lpips_ax.set_title("LPIPS (lower is better)")

    plot_train(loss_ax, "loss", "Total")
    plot_train(loss_ax, "loss_rgb_source", "Source RGB")
    plot_train(loss_ax, "loss_rgb_final_map", "Final-map RGB")
    plot_train(loss_ax, "loss_rgb_heldout", "Heldout RGB")
    plot_train(loss_ax, "loss_lpips", "LPIPS")
    plot_train(loss_ax, "loss_active_count", "Active count")
    plot_train(loss_ax, "loss_rgb_current", "Current RGB")
    plot_train(loss_ax, "loss_rgb_replay", "Replay RGB")
    plot_train(loss_ax, "loss_depth", "Depth")
    plot_train(loss_ax, "loss_camera", "Camera")
    plot_train(loss_ax, "loss_regularization", "Residual regularization")
    loss_ax.set_yscale("log")
    loss_ax.set_title("Training losses")

    plot_eval(alpha_ax, "source_alpha_coverage_05", "Source coverage", 100.0)
    plot_eval(alpha_ax, "heldout_loss_alpha_coverage_05", "Loss coverage", 100.0)
    plot_eval(alpha_ax, "heldout_test_alpha_coverage_05", "Test coverage", 100.0)
    plot_eval(alpha_ax, "source_white_hole_ratio", "Source holes", 100.0)
    plot_eval(alpha_ax, "heldout_test_white_hole_ratio", "Test holes", 100.0)
    alpha_ax.set_title("Alpha coverage")
    alpha_ax.set_ylabel("Pixels (%)")

    for key, label in (
        ("residual_mean_shift_mean", "Mean"),
        ("residual_rotation_shift_mean", "Rotation"),
        ("residual_scale_relative_shift_mean", "Scale"),
        ("residual_opacity_shift_mean", "Opacity"),
        ("residual_harmonics_shift_mean", "Harmonics"),
    ):
        plot_eval(residual_ax, key, label)
    residual_ax.set_title("Residual magnitudes")

    plot_eval(geometry_ax, "sim3_source_center_rmse", "Sim(3) center RMSE")
    plot_eval(geometry_ax, "sim3_source_rotation_deg_mean", "Sim(3) rotation deg")
    plot_eval(geometry_ax, "sim3_test_center_rmse", "Test center RMSE")
    plot_eval(geometry_ax, "sim3_test_rotation_deg_mean", "Test rotation deg")
    plot_eval(geometry_ax, "map_opacity_mean", "Map opacity mean")
    plot_eval(geometry_ax, "map_scale_mean", "Map scale mean")
    geometry_ax.set_title("Geometry and map state")

    plot_train(gradient_ax, "grad_norm", "Gradient norm")
    plot_train(gradient_ax, "camera_grad_norm", "Camera gradient norm")
    plot_train(gradient_ax, "source_train_psnr", "Train PSNR")
    plot_train(gradient_ax, "heldout_train_psnr", "Heldout PSNR")
    plot_eval(gradient_ax, "residual_gate_mean", "Update gate")
    gradient_ax.set_title("Optimization")

    plot_eval(unused_ax, "keep_probability_mean", "Mean keep")
    plot_eval(unused_ax, "keep_ratio_ge_01", "Keep >= 0.1")
    plot_eval(unused_ax, "keep_ratio_ge_05", "Keep >= 0.5")
    unused_ax.set_ylim(-0.02, 1.02)
    unused_ax.set_title("Learned Gaussian keep gate")

    for axis in axes.ravel():
        axis.set_xlabel("Step")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
    destination = output_dir / "metric_curves.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


@torch.no_grad()
def _evaluate(
    model,
    head,
    cfg,
    base_state,
    gradient_evidence,
    render_cache,
    data,
    step,
    output_dir,
    sim3_metrics,
):
    head.eval()
    if bool(cfg.get("residual_enabled", True)):
        evidence = _attach_visual_evidence(
            head, cfg, base_state, data, gradient_evidence, render_cache
        )
        corrected_state, stats = _predict_corrected_state(
            head, cfg, base_state, evidence, data["c2w"][:, 0]
        )
    else:
        corrected_state = base_state
        stats = _mean_update_stats([], base_state.gaussians.means)
    source_views = {
        "ids": data["ids"],
        "rgb": data["images"],
        "c2w": data["c2w"],
        "intrinsic": data["intrinsic"],
    }
    source_metrics, source_render, source_alpha = _metrics_for_views(
        model, corrected_state.gaussians, source_views, data["image_shape"], "source"
    )
    test_metrics, test_render, test_alpha = _metrics_for_views(
        model,
        corrected_state.gaussians,
        data["heldout_test"],
        data["image_shape"],
        "heldout_test",
        compute_perceptual=bool(cfg.get("eval_lpips_enabled", True)),
        perceptual_device=str(cfg.get("eval_lpips_device", "cpu")),
    )
    base_source_metrics, _, _ = _metrics_for_views(
        model, base_state.gaussians, source_views, data["image_shape"], "base_source"
    )
    valid_opacity = corrected_state.gaussians.opacities[
        corrected_state.gaussians.opacities > 0
    ]
    metrics = {
        "step": int(step),
        "training_mode": "offline_global",
        "gaussian_parameter_source": str(cfg.gaussian_parameter_source).lower(),
        "residual_enabled": bool(cfg.get("residual_enabled", True)),
        "lpips_loss_enabled": bool(cfg.get("lpips_loss_enabled", False)),
        "loss_lpips_weight": float(cfg.get("loss_lpips", 0.05)),
        "train_gs_head": bool(cfg.get("train_gs_head", False)),
        "train_camera_head": bool(cfg.get("train_camera_head", False)),
        "camera_grad_clip": float(cfg.get("camera_grad_clip", 0.0)),
        "camera_translation_direction_min_baseline_ratio": float(
            cfg.get("camera_translation_direction_min_baseline_ratio", 0.0)
        ),
        "camera_translation_direction_baseline_weighting": bool(
            cfg.get("camera_translation_direction_baseline_weighting", False)
        ),
        "camera_global_trajectory_weight": float(
            cfg.get("camera_global_trajectory_weight", 0.0)
        ),
        "train_depth_head": bool(cfg.get("train_depth_head", False)),
        "depth_loss_enabled": bool(cfg.get("depth_loss_enabled", False)),
        "detach_predicted_camera": bool(
            cfg.get("detach_predicted_camera", False)
        ),
        "detach_predicted_depth": bool(cfg.get("detach_predicted_depth", False)),
        "unbounded_residual_updates": bool(
            cfg.get("unbounded_residual_updates", False)
        ),
        "training_geometry_camera_source": str(
            cfg.training_geometry_camera_source
        ).lower(),
        "training_sim3_alignment_mode": str(
            cfg.training_sim3_alignment_mode
        ).lower(),
        "training_source_render_camera": str(
            cfg.training_source_render_camera
        ).lower(),
        "training_heldout_loss_camera": str(
            cfg.training_heldout_loss_camera
        ).lower(),
        "geometry_camera_source": str(
            cfg.evaluation_geometry_camera_source
        ).lower(),
        "source_render_camera": str(
            cfg.evaluation_source_render_camera
        ).lower(),
        "heldout_loss_camera": str(
            cfg.evaluation_heldout_loss_camera
        ).lower(),
        "heldout_test_camera": str(
            cfg.evaluation_heldout_test_camera
        ).lower(),
        "heldout_loss_camera_conditioned": (
            str(cfg.evaluation_heldout_loss_camera).lower()
            == "predicted"
        ),
        "heldout_camera_conditioned": (
            str(cfg.evaluation_heldout_test_camera).lower() == "predicted"
        ),
        "gaussian_count": int(base_state.num_gaussians),
        "map_opacity_mean": float(valid_opacity.mean().item()),
        "map_opacity_min": float(valid_opacity.min().item()),
        "map_opacity_max": float(valid_opacity.max().item()),
        "map_scale_mean": float(corrected_state.gaussians.scales.mean().item()),
        "map_scale_max": float(corrected_state.gaussians.scales.max().item()),
        **sim3_metrics,
        **base_source_metrics,
        **source_metrics,
        **test_metrics,
        **{
            f"residual_{key}": float(value.detach().float().item())
            for key, value in stats.items()
        },
    }
    with (output_dir / "eval_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")

    image_dir = output_dir / "renders" / f"step_{step:06d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, frame_id in enumerate(data["ids"]):
        _save_image(image_dir / f"source_{int(frame_id):03d}.png", source_render[:, index])
        _save_mask(
            image_dir / f"source_{int(frame_id):03d}_alpha.png",
            source_alpha[:, index],
        )
    if test_render is not None:
        for index, frame_id in enumerate(data["heldout_test"]["ids"]):
            _save_image(
                image_dir / f"test_{int(frame_id):03d}.png", test_render[:, index]
            )
            _save_mask(
                image_dir / f"test_{int(frame_id):03d}_alpha.png",
                test_alpha[:, index],
            )
    curve_path = _plot_metrics(output_dir)
    print(f"Updated offline metric curves: {curve_path}")
    print("[offline eval] " + json.dumps(metrics, sort_keys=True))
    head.train()
    return corrected_state


@torch.no_grad()
def _evaluate_incremental(
    model,
    head,
    keep_gate,
    cfg,
    anchors,
    geometry_data,
    render_data,
    step,
    output_dir,
    sim3_metrics,
):
    head.eval()
    base_state, _, base_append = _rollout_incremental(
        model,
        head,
        cfg,
        anchors,
        geometry_data,
        render_data,
        apply_residual=False,
    )
    corrected_state, stats, append_summary = _rollout_incremental(
        model,
        head,
        cfg,
        anchors,
        geometry_data,
        render_data,
        apply_residual=True,
    )
    corrected_state, keep = _apply_scene_keep_gate(corrected_state, keep_gate)
    source_views = {
        "ids": render_data["ids"],
        "rgb": render_data["images"],
        "c2w": render_data["c2w"],
        "intrinsic": render_data["intrinsic"],
    }
    source_metrics, source_render, source_alpha = _metrics_for_views(
        model,
        corrected_state.gaussians,
        source_views,
        render_data["image_shape"],
        "source",
    )
    test_metrics, test_render, test_alpha = _metrics_for_views(
        model,
        corrected_state.gaussians,
        render_data["heldout_test"],
        render_data["image_shape"],
        "heldout_test",
        compute_perceptual=bool(cfg.get("eval_lpips_enabled", True)),
        perceptual_device=str(cfg.get("eval_lpips_device", "cpu")),
    )
    heldout_loss_metrics, heldout_loss_render, heldout_loss_alpha = (
        _metrics_for_views(
            model,
            corrected_state.gaussians,
            render_data["heldout_loss"],
            render_data["image_shape"],
            "heldout_loss",
            compute_perceptual=bool(cfg.get("eval_lpips_enabled", True)),
            perceptual_device=str(cfg.get("eval_lpips_device", "cpu")),
        )
    )
    base_source_metrics, _, _ = _metrics_for_views(
        model,
        base_state.gaussians,
        source_views,
        render_data["image_shape"],
        "base_source",
    )
    valid_opacity = corrected_state.gaussians.opacities[
        corrected_state.gaussians.opacities > 0
    ]
    training_mode = str(
        cfg.get("training_mode", "incremental_per_frame")
    ).lower()
    metrics = {
        "step": int(step),
        "training_mode": training_mode,
        "append_mode": str(cfg.get("append_mode", "point_map")).lower(),
        "periodic_all_append_enabled": bool(
            cfg.get("periodic_all_append_enabled", False)
        ),
        "periodic_all_append_interval": int(
            cfg.get("periodic_all_append_interval", 0)
        ),
        "optimizer_updates_per_sweep": (
            len(anchors) if training_mode == "incremental_per_frame" else 1
        ),
        "gaussian_parameter_source": str(cfg.gaussian_parameter_source).lower(),
        "residual_enabled": bool(cfg.get("residual_enabled", True)),
        "visual_enabled": bool(cfg.get("visual_enabled", False)),
        "visual_candidate_mode": str(
            cfg.get("visual_candidate_mode", "gradient_active")
        ).lower(),
        "visual_image_scale": float(cfg.get("visual_image_scale", 0.5)),
        "heldout_loss_enabled": bool(cfg.get("heldout_loss_enabled", False)),
        "heldout_loss_frames": [
            int(frame_id) for frame_id in render_data["heldout_loss"]["ids"]
        ],
        "heldout_loss_weight": float(cfg.get("heldout_loss_weight", 1.0)),
        "keep_gate_enabled": keep_gate is not None,
        "keep_gate_lr": float(cfg.get("keep_gate_lr", 0.0)),
        "keep_gate_initial_probability": float(
            cfg.get("keep_gate_initial_probability", 0.99)
        ),
        "keep_gate_detach_heldout_map": bool(
            cfg.get("keep_gate_detach_heldout_map", True)
        ),
        "keep_gate_active_count_weight": float(
            cfg.get("keep_gate_active_count_weight", 0.0)
        ),
        "keep_probability_mean": float(keep.mean().item()),
        "keep_probability_min": float(keep.min().item()),
        "keep_probability_max": float(keep.max().item()),
        "keep_ratio_ge_01": float((keep >= 0.1).float().mean().item()),
        "keep_ratio_ge_05": float((keep >= 0.5).float().mean().item()),
        "effective_gs_count_ge_01": int((keep >= 0.1).sum().item()),
        "effective_gs_count_ge_05": int((keep >= 0.5).sum().item()),
        "lpips_loss_enabled": bool(cfg.get("lpips_loss_enabled", False)),
        "loss_lpips_weight": float(cfg.get("loss_lpips", 0.05)),
        "lpips_apply_after_step": int(cfg.get("lpips_apply_after_step", 0)),
        "train_gs_head": bool(cfg.get("train_gs_head", False)),
        "train_camera_head": bool(cfg.get("train_camera_head", False)),
        "camera_grad_clip": float(cfg.get("camera_grad_clip", 0.0)),
        "camera_translation_direction_min_baseline_ratio": float(
            cfg.get("camera_translation_direction_min_baseline_ratio", 0.0)
        ),
        "camera_translation_direction_baseline_weighting": bool(
            cfg.get("camera_translation_direction_baseline_weighting", False)
        ),
        "camera_global_trajectory_weight": float(
            cfg.get("camera_global_trajectory_weight", 0.0)
        ),
        "train_depth_head": bool(cfg.get("train_depth_head", False)),
        "depth_loss_enabled": bool(cfg.get("depth_loss_enabled", False)),
        "detach_predicted_camera": bool(
            cfg.get("detach_predicted_camera", False)
        ),
        "detach_predicted_depth": bool(cfg.get("detach_predicted_depth", False)),
        "unbounded_residual_updates": bool(
            cfg.get("unbounded_residual_updates", False)
        ),
        "training_geometry_camera_source": str(
            cfg.training_geometry_camera_source
        ).lower(),
        "training_sim3_alignment_mode": str(
            cfg.training_sim3_alignment_mode
        ).lower(),
        "training_source_render_camera": str(
            cfg.training_source_render_camera
        ).lower(),
        "training_heldout_loss_camera": str(
            cfg.training_heldout_loss_camera
        ).lower(),
        "geometry_camera_source": str(
            cfg.evaluation_geometry_camera_source
        ).lower(),
        "source_render_camera": str(
            cfg.evaluation_source_render_camera
        ).lower(),
        "heldout_loss_camera": str(
            cfg.evaluation_heldout_loss_camera
        ).lower(),
        "heldout_test_camera": str(
            cfg.evaluation_heldout_test_camera
        ).lower(),
        "heldout_loss_camera_conditioned": (
            str(cfg.evaluation_heldout_loss_camera).lower()
            == "predicted"
        ),
        "heldout_camera_conditioned": (
            str(cfg.evaluation_heldout_test_camera).lower() == "predicted"
        ),
        "initial_gs_count": int(anchors[0].means.shape[1]),
        "gaussian_count": int(corrected_state.num_gaussians),
        "final_gs_count": int(corrected_state.num_gaussians),
        "appended_gs": int(append_summary["appended_count"]),
        "point_appended_gs": int(append_summary["point_appended_count"]),
        "periodic_appended_gs": int(
            append_summary["periodic_appended_count"]
        ),
        "periodic_keyframes": int(append_summary["periodic_keyframes"]),
        "append_ratio": float(append_summary["append_ratio"]),
        "base_final_gs_count": int(base_state.num_gaussians),
        "base_appended_gs": int(base_append["appended_count"]),
        "base_point_appended_gs": int(base_append["point_appended_count"]),
        "base_periodic_appended_gs": int(
            base_append["periodic_appended_count"]
        ),
        "base_append_ratio": float(base_append["append_ratio"]),
        "map_opacity_mean": float(valid_opacity.mean().item()),
        "map_opacity_min": float(valid_opacity.min().item()),
        "map_opacity_max": float(valid_opacity.max().item()),
        "map_scale_mean": float(corrected_state.gaussians.scales.mean().item()),
        "map_scale_max": float(corrected_state.gaussians.scales.max().item()),
        **sim3_metrics,
        **base_source_metrics,
        **source_metrics,
        **heldout_loss_metrics,
        **test_metrics,
        **{
            f"residual_{key}": float(value.detach().float().item())
            for key, value in stats.items()
        },
    }
    with (output_dir / "eval_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")

    image_dir = output_dir / "renders" / f"step_{step:06d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, frame_id in enumerate(render_data["ids"]):
        _save_image(
            image_dir / f"source_{int(frame_id):03d}.png",
            source_render[:, index],
        )
        _save_mask(
            image_dir / f"source_{int(frame_id):03d}_alpha.png",
            source_alpha[:, index],
        )
    if test_render is not None:
        for index, frame_id in enumerate(render_data["heldout_test"]["ids"]):
            _save_image(
                image_dir / f"test_{int(frame_id):03d}.png",
                test_render[:, index],
            )
            _save_mask(
                image_dir / f"test_{int(frame_id):03d}_alpha.png",
                test_alpha[:, index],
            )
    if heldout_loss_render is not None:
        for index, frame_id in enumerate(render_data["heldout_loss"]["ids"]):
            _save_image(
                image_dir / f"heldout_loss_{int(frame_id):03d}.png",
                heldout_loss_render[:, index],
            )
            _save_mask(
                image_dir / f"heldout_loss_{int(frame_id):03d}_alpha.png",
                heldout_loss_alpha[:, index],
            )
    curve_path = _plot_metrics(output_dir)
    print(f"Updated incremental metric curves: {curve_path}")
    print("[incremental eval] " + json.dumps(metrics, sort_keys=True))
    head.train()
    return corrected_state


def _incremental_rgb_losses(
    model,
    cfg,
    state: StreamingGaussianState,
    render_data: dict,
    view_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_render = _render(
        model,
        state.gaussians,
        render_data["c2w"][:, view_idx : view_idx + 1],
        render_data["intrinsic"][:, view_idx : view_idx + 1],
        render_data["image_shape"],
    )
    current_loss = (
        current_render.color.float()
        - render_data["images"][:, view_idx : view_idx + 1].float()
    ).square().mean()

    replay_indices = _select_replay_indices(
        view_idx, int(cfg.get("incremental_replay_views", -1))
    )
    if not replay_indices:
        return current_loss, current_loss.new_zeros(())
    replay_render = _render(
        model,
        state.gaussians,
        render_data["c2w"][:, replay_indices],
        render_data["intrinsic"][:, replay_indices],
        render_data["image_shape"],
    )
    replay_loss = (
        replay_render.color.float()
        - render_data["images"][:, replay_indices].float()
    ).square().mean()
    return current_loss, replay_loss


def _build_full_graph_incremental_loss(
    model,
    head,
    cfg,
    anchors: list[Gaussians],
    geometry_data: dict,
    render_data: dict,
) -> tuple[
    StreamingGaussianState,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, float],
]:
    """Roll out every source view while preserving the cross-view graph."""
    # Frozen anchors have no graph. Trainable GS-head anchors must retain this
    # edge so first-view RGB loss can update the GS head in full-graph mode.
    state = _make_state(anchors[0])
    current_losses = []
    replay_losses = []
    stats_per_view = []
    append_ratios = []
    appended_count = 0.0
    point_appended_count = 0.0
    periodic_appended_count = 0.0
    periodic_keyframes = 0.0
    point_depth_scales = []
    replay_views = int(cfg.get("incremental_replay_views", -1))

    for view_idx in range(len(anchors)):
        if view_idx > 0:
            state, append_stats = _append_view(
                model,
                cfg,
                state,
                anchors[view_idx],
                view_idx,
                geometry_data["depth"][:, view_idx],
                geometry_data["c2w"][:, view_idx],
                geometry_data["intrinsic"][:, view_idx],
                render_data["image_shape"],
            )
            append_ratios.append(float(append_stats["append_ratio"]))
            appended_count += float(append_stats["appended_count"])
            point_appended_count += float(append_stats["point_appended_count"])
            periodic_appended_count += float(
                append_stats["periodic_appended_count"]
            )
            periodic_keyframes += float(append_stats["periodic_keyframe"])
            point_depth_scales.append(float(append_stats["point_depth_scale"]))

        state, stats = _apply_incremental_residual(
            model, head, cfg, state, render_data, view_idx
        )
        current_loss, replay_loss = _incremental_rgb_losses(
            model, cfg, state, render_data, view_idx
        )
        current_losses.append(current_loss)
        if view_idx > 0 and replay_views != 0:
            replay_losses.append(replay_loss)
        stats_per_view.append(stats)

    current_loss = torch.stack(current_losses).mean()
    replay_loss = (
        torch.stack(replay_losses).mean()
        if replay_losses
        else current_loss.new_zeros(())
    )
    weighted_rgb_loss = (
        float(cfg.get("incremental_current_loss_weight", 1.0)) * current_loss
        + float(cfg.get("incremental_replay_loss_weight", 0.5)) * replay_loss
    )
    stats = _mean_update_stats(stats_per_view, state.gaussians.means)
    regularization_loss = (
        float(cfg.residual_regularization_weight) * stats["regularization"]
    )
    loss = weighted_rgb_loss + regularization_loss
    append_summary = {
        "append_ratio": (
            float(sum(append_ratios) / len(append_ratios))
            if append_ratios
            else 0.0
        ),
        "appended_count": appended_count,
        "point_appended_count": point_appended_count,
        "periodic_appended_count": periodic_appended_count,
        "periodic_keyframes": periodic_keyframes,
        "point_depth_scale": (
            float(sum(point_depth_scales) / len(point_depth_scales))
            if point_depth_scales
            else 1.0
        ),
    }
    return (
        state,
        loss,
        current_loss,
        replay_loss,
        weighted_rgb_loss,
        regularization_loss,
        stats,
        append_summary,
    )


def _build_final_map_incremental_loss(
    model,
    head,
    lpips_model,
    keep_gate,
    cfg,
    anchors: list[Gaussians],
    geometry_data: dict,
    render_data: dict,
    step: int,
) -> tuple[
    StreamingGaussianState,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, float],
]:
    """Roll out causally, then supervise only the final map on all sources."""
    state = _make_state(anchors[0])
    stats_per_view = []
    append_ratios = []
    appended_count = 0.0
    point_appended_count = 0.0
    periodic_appended_count = 0.0
    periodic_keyframes = 0.0
    point_depth_scales = []

    for view_idx in range(len(anchors)):
        if view_idx > 0:
            state, append_stats = _append_view(
                model,
                cfg,
                state,
                anchors[view_idx],
                view_idx,
                geometry_data["depth"][:, view_idx],
                geometry_data["c2w"][:, view_idx],
                geometry_data["intrinsic"][:, view_idx],
                render_data["image_shape"],
            )
            append_ratios.append(float(append_stats["append_ratio"]))
            appended_count += float(append_stats["appended_count"])
            point_appended_count += float(append_stats["point_appended_count"])
            periodic_appended_count += float(
                append_stats["periodic_appended_count"]
            )
            periodic_keyframes += float(append_stats["periodic_keyframe"])
            point_depth_scales.append(float(append_stats["point_depth_scale"]))

        state, stats = _apply_incremental_residual(
            model, head, cfg, state, render_data, view_idx
        )
        stats_per_view.append(stats)

    ungated_state = state
    state, keep = _apply_scene_keep_gate(ungated_state, keep_gate)
    final_render = _render(
        model,
        state.gaussians,
        render_data["c2w"],
        render_data["intrinsic"],
        render_data["image_shape"],
    )
    final_rgb_loss = (
        final_render.color.float() - render_data["images"].float()
    ).square().mean()
    final_heldout_rgb_loss = final_rgb_loss.new_zeros(())
    if bool(cfg.get("heldout_loss_enabled", False)):
        heldout_state = state
        if (
            keep_gate is not None
            and bool(cfg.get("keep_gate_detach_heldout_map", True))
        ):
            heldout_state, _ = _apply_scene_keep_gate(
                ungated_state.detach(), keep_gate
            )
        heldout_render = _render(
            model,
            heldout_state.gaussians,
            render_data["heldout_loss"]["c2w"],
            render_data["heldout_loss"]["intrinsic"],
            render_data["image_shape"],
        )
        raw_heldout_rgb_loss = (
            heldout_render.color.float()
            - render_data["heldout_loss"]["rgb"].float()
        ).square().mean()
        final_heldout_rgb_loss = float(
            cfg.get("heldout_loss_weight", 1.0)
        ) * raw_heldout_rgb_loss
    final_lpips_loss = final_rgb_loss.new_zeros(())
    if (
        lpips_model is not None
        and step >= int(cfg.get("lpips_apply_after_step", 0))
    ):
        batch, views, channels, height, width = final_render.color.shape
        rendered_flat = final_render.color.float().reshape(
            batch * views, channels, height, width
        )
        target_flat = render_data["images"].float().reshape(
            batch * views, channels, height, width
        )
        raw_lpips = _training_lpips_mean(
            lpips_model,
            rendered_flat,
            target_flat,
            chunk_size=int(cfg.get("lpips_train_chunk_size", 1)),
            gradient_checkpointing=bool(
                cfg.get("lpips_gradient_checkpointing", True)
            ),
        )
        final_lpips_loss = float(cfg.get("loss_lpips", 0.05)) * torch.nan_to_num(
            raw_lpips, nan=0.0, posinf=0.0, neginf=0.0
        )
    stats = _mean_update_stats(stats_per_view, state.gaussians.means)
    regularization_loss = (
        float(cfg.residual_regularization_weight) * stats["regularization"]
    )
    active_count_loss = final_rgb_loss.new_zeros(())
    if keep_gate is not None:
        active_count_loss = float(
            cfg.get("keep_gate_active_count_weight", 0.0)
        ) * keep.mean()
    stats = {
        **stats,
        "keep_probability_mean": keep.mean(),
        "keep_probability_min": keep.min(),
        "keep_probability_max": keep.max(),
        "keep_ratio_ge_01": (keep >= 0.1).float().mean(),
        "keep_ratio_ge_05": (keep >= 0.5).float().mean(),
    }
    loss = (
        final_rgb_loss
        + final_heldout_rgb_loss
        + final_lpips_loss
        + active_count_loss
        + regularization_loss
    )
    append_summary = {
        "append_ratio": (
            float(sum(append_ratios) / len(append_ratios))
            if append_ratios
            else 0.0
        ),
        "appended_count": appended_count,
        "point_appended_count": point_appended_count,
        "periodic_appended_count": periodic_appended_count,
        "periodic_keyframes": periodic_keyframes,
        "point_depth_scale": (
            float(sum(point_depth_scales) / len(point_depth_scales))
            if point_depth_scales
            else 1.0
        ),
    }
    return (
        state,
        loss,
        final_rgb_loss,
        final_heldout_rgb_loss,
        final_lpips_loss,
        active_count_loss,
        regularization_loss,
        stats,
        append_summary,
    )


def _save_checkpoint(
    head,
    optimizer,
    step,
    output_dir,
    *,
    keep_gate: SceneGaussianKeepGate | None = None,
    gs_head: torch.nn.Module | None = None,
    camera_head: torch.nn.Module | None = None,
    depth_head: torch.nn.Module | None = None,
):
    path = output_dir / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": int(step),
        "residual_head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if keep_gate is not None:
        checkpoint["keep_gate"] = keep_gate.state_dict()
    if gs_head is not None:
        checkpoint["gs_head"] = gs_head.state_dict()
    if camera_head is not None:
        checkpoint["camera_head"] = camera_head.state_dict()
    if depth_head is not None:
        checkpoint["depth_head"] = depth_head.state_dict()
    torch.save(checkpoint, path)


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The offline GS-MLP experiment requires CUDA/gsplat.")
    if "offline" not in cfg_dict:
        raise KeyError("Use +experiment=offline_gs_mlp_overfit.")
    cfg = cfg_dict.offline
    if str(cfg.dataset_type).lower() != "dl3dv":
        raise ValueError("The first offline GS-MLP experiment supports DL3DV only.")
    source_frame_count = len(cfg.train_frames)
    heldout_loss_enabled = bool(cfg.get("heldout_loss_enabled", False))
    heldout_loss_count = len(cfg.heldout_loss_frame)
    if heldout_loss_enabled and heldout_loss_count == 0:
        raise ValueError(
            "offline.heldout_loss_enabled=true requires at least one "
            "offline.heldout_loss_frame."
        )
    if not heldout_loss_enabled and heldout_loss_count != 0:
        raise ValueError(
            "Set offline.heldout_loss_enabled=true or clear "
            "offline.heldout_loss_frame."
        )
    heldout_test_count = len(cfg.heldout_test_frame)
    if heldout_test_count < 1:
        raise ValueError("offline.heldout_test_frame must contain at least one view.")
    overlap = set(cfg.heldout_loss_frame) & set(cfg.heldout_test_frame)
    if overlap:
        raise ValueError(
            "Heldout-loss and heldout-test frames must be disjoint; "
            f"overlap={sorted(int(frame_id) for frame_id in overlap)}."
        )
    training_mode = str(cfg.get("training_mode", "incremental_per_frame")).lower()
    if training_mode not in {
        "incremental_per_frame",
        "incremental_full_graph",
        "incremental_final_map",
        "offline_global",
    }:
        raise ValueError(
            "offline.training_mode must be incremental_per_frame, "
            "incremental_full_graph, incremental_final_map, or offline_global."
        )
    append_mode = str(cfg.get("append_mode", "point_map")).lower()
    if append_mode not in {"point_map", "all"}:
        raise ValueError("offline.append_mode must be point_map or all.")
    periodic_all_append_enabled = bool(
        cfg.get("periodic_all_append_enabled", False)
    )
    periodic_all_append_interval = int(
        cfg.get("periodic_all_append_interval", 0)
    )
    if periodic_all_append_enabled:
        if append_mode != "point_map":
            raise ValueError(
                "offline.periodic_all_append_enabled=true requires "
                "offline.append_mode=point_map."
            )
        if periodic_all_append_interval <= 0:
            raise ValueError(
                "offline.periodic_all_append_interval must be positive."
            )
        opacity_scale = float(cfg.get("periodic_all_append_opacity_scale", 0.35))
        scale_multiplier = float(
            cfg.get("periodic_all_append_scale_multiplier", 0.75)
        )
        if not 0.0 <= opacity_scale <= 1.0:
            raise ValueError(
                "offline.periodic_all_append_opacity_scale must be in [0, 1]."
            )
        if scale_multiplier <= 0.0:
            raise ValueError(
                "offline.periodic_all_append_scale_multiplier must be positive."
            )
    if int(cfg.max_steps) <= 0:
        raise ValueError("offline.max_steps must be positive.")
    training_geometry_camera_source = str(
        cfg.training_geometry_camera_source
    ).lower()
    training_source_render_camera = str(
        cfg.training_source_render_camera
    ).lower()
    training_heldout_loss_camera = str(
        cfg.training_heldout_loss_camera
    ).lower()
    evaluation_geometry_camera_source = str(
        cfg.evaluation_geometry_camera_source
    ).lower()
    evaluation_source_render_camera = str(
        cfg.evaluation_source_render_camera
    ).lower()
    evaluation_heldout_loss_camera = str(
        cfg.evaluation_heldout_loss_camera
    ).lower()
    evaluation_heldout_test_camera = str(
        cfg.evaluation_heldout_test_camera
    ).lower()
    valid_render_cameras = {"gt", "predicted"}
    camera_modes = {
        "training_geometry_camera_source": training_geometry_camera_source,
        "training_source_render_camera": training_source_render_camera,
        "training_heldout_loss_camera": training_heldout_loss_camera,
        "evaluation_geometry_camera_source": evaluation_geometry_camera_source,
        "evaluation_source_render_camera": evaluation_source_render_camera,
        "evaluation_heldout_loss_camera": evaluation_heldout_loss_camera,
        "evaluation_heldout_test_camera": evaluation_heldout_test_camera,
    }
    for name, value in camera_modes.items():
        if value not in valid_render_cameras:
            raise ValueError(f"offline.{name} must be gt or predicted.")
    training_sim3_alignment_mode = str(
        cfg.training_sim3_alignment_mode
    ).lower()
    training_sim3_alignment_enabled = _requires_sim3_alignment(
        training_source_render_camera,
        "predicted",
        heldout_loss_enabled,
        training_sim3_alignment_mode,
        geometry_camera_source=training_geometry_camera_source,
        heldout_loss_camera=training_heldout_loss_camera,
    )
    evaluation_sim3_alignment_mode = str(
        cfg.evaluation_sim3_alignment_mode
    ).lower()
    evaluation_sim3_alignment_enabled = _requires_sim3_alignment(
        evaluation_source_render_camera,
        evaluation_heldout_test_camera,
        heldout_loss_enabled,
        evaluation_sim3_alignment_mode,
        geometry_camera_source=evaluation_geometry_camera_source,
        heldout_loss_camera=evaluation_heldout_loss_camera,
    )
    if (
        training_sim3_alignment_enabled or evaluation_sim3_alignment_enabled
    ) and source_frame_count < 3:
        raise ValueError(
            "At least 3 offline.train_frames are required when source-camera "
            "Sim(3) alignment is enabled."
        )
    if source_frame_count < 1:
        raise ValueError("offline.train_frames must contain at least one source view.")
    gaussian_parameter_source = str(cfg.gaussian_parameter_source).lower()
    if gaussian_parameter_source not in {"analytic", "gs_head"}:
        raise ValueError(
            "offline.gaussian_parameter_source must be analytic or gs_head."
        )
    train_gs_head = bool(cfg.get("train_gs_head", False))
    if train_gs_head and gaussian_parameter_source != "gs_head":
        raise ValueError(
            "offline.train_gs_head=true requires "
            "offline.gaussian_parameter_source=gs_head."
        )
    if train_gs_head and float(cfg.get("gs_head_lr", 0.0)) <= 0.0:
        raise ValueError("offline.gs_head_lr must be positive when training gs_head.")
    train_camera_head = bool(cfg.get("train_camera_head", False))
    train_depth_head = bool(cfg.get("train_depth_head", False))
    residual_enabled = bool(cfg.get("residual_enabled", True))
    keep_gate_enabled = bool(cfg.get("keep_gate_enabled", False))
    if keep_gate_enabled:
        if training_mode != "incremental_final_map":
            raise ValueError(
                "offline.keep_gate_enabled=true requires "
                "offline.training_mode=incremental_final_map."
            )
        if append_mode != "all":
            raise ValueError(
                "offline.keep_gate_enabled=true requires offline.append_mode=all "
                "so gate indices have a fixed all-append topology."
            )
        if not heldout_loss_enabled:
            raise ValueError(
                "offline.keep_gate_enabled=true requires heldout-loss supervision."
            )
        if float(cfg.get("keep_gate_lr", 0.0)) <= 0.0:
            raise ValueError("offline.keep_gate_lr must be positive.")
        initial_keep = float(cfg.get("keep_gate_initial_probability", 0.99))
        if not 0.0 < initial_keep < 1.0:
            raise ValueError(
                "offline.keep_gate_initial_probability must be in (0, 1)."
            )
        if float(cfg.get("keep_gate_active_count_weight", 0.0)) < 0.0:
            raise ValueError(
                "offline.keep_gate_active_count_weight must be non-negative."
            )
    if heldout_loss_enabled and training_mode != "incremental_final_map":
        raise ValueError(
            "Heldout-loss supervision currently requires "
            "offline.training_mode=incremental_final_map."
        )
    if float(cfg.get("heldout_loss_weight", 1.0)) < 0.0:
        raise ValueError("offline.heldout_loss_weight must be non-negative.")
    lpips_loss_enabled = bool(cfg.get("lpips_loss_enabled", False))
    if lpips_loss_enabled and training_mode != "incremental_final_map":
        raise ValueError(
            "offline.lpips_loss_enabled currently requires "
            "offline.training_mode=incremental_final_map."
        )
    if float(cfg.get("loss_lpips", 0.05)) < 0.0:
        raise ValueError("offline.loss_lpips must be non-negative.")
    if int(cfg.get("lpips_apply_after_step", 0)) < 0:
        raise ValueError("offline.lpips_apply_after_step must be non-negative.")
    depth_loss_enabled = bool(cfg.get("depth_loss_enabled", False))
    depth_loss_target = str(cfg.get("depth_loss_target", "dav3")).lower()
    if depth_loss_target not in {"dav3", "gt"}:
        raise ValueError("offline.depth_loss_target must be dav3 or gt.")
    if train_camera_head and float(cfg.get("camera_head_lr", 0.0)) <= 0.0:
        raise ValueError(
            "offline.camera_head_lr must be positive when training camera_head."
        )
    if train_depth_head and float(cfg.get("depth_head_lr", 0.0)) <= 0.0:
        raise ValueError(
            "offline.depth_head_lr must be positive when training depth_head."
        )
    if depth_loss_enabled and not train_depth_head:
        raise ValueError(
            "offline.depth_loss_enabled=true requires train_depth_head=true."
        )
    if depth_loss_enabled and depth_loss_target == "gt":
        raise ValueError(
            "DL3DV has no GT depth; use offline.depth_loss_target=dav3."
        )
    if not any(
        (
            residual_enabled,
            keep_gate_enabled,
            train_gs_head,
            train_camera_head,
            train_depth_head,
        )
    ):
        raise ValueError(
            "Enable at least one of keep gate, residual, GS, camera, or depth "
            "head training."
        )

    torch.manual_seed(int(cfg_dict.seed))
    np.random.seed(int(cfg_dict.seed))
    device = torch.device("cuda:0")
    output_dir = Path(HydraConfig.get()["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg_dict, output_dir / "resolved_config.yaml", resolve=True)

    root_cfg = load_typed_root_config(cfg_dict)
    model = get_model(root_cfg.model.encoder, root_cfg.model.decoder).to(device)
    if gaussian_parameter_source == "gs_head":
        _load_checkpoint(model, cfg.gs_head_checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if train_gs_head:
        model.encoder.gs_head.requires_grad_(True)
        model.encoder.gs_head.train()
    if train_camera_head:
        model.encoder.camera_head.requires_grad_(True)
        model.encoder.camera_head.train()
    depth_head_source = str(
        getattr(model.encoder.cfg, "depth_head_source", "native")
    ).lower()
    if depth_head_source not in {"native", "world_mirror"}:
        raise ValueError(
            "model.encoder.depth_head_source must be native or world_mirror."
        )
    native_depth_head = getattr(model.encoder, "depth_head", None)
    if native_depth_head is None:
        raise RuntimeError("The reconstruction backbone has no native depth head.")
    wm_depth_head = model.encoder.gaussian_param_head.scratch.output_conv2
    selected_depth_head = (
        native_depth_head if depth_head_source == "native" else wm_depth_head
    )
    if train_depth_head:
        selected_depth_head.requires_grad_(True)
        selected_depth_head.train()

    data = _load_toy_data(cfg, device)
    source_count = data["images"].shape[1]
    predict_evaluation_loss_camera = (
        heldout_loss_enabled
        and evaluation_heldout_loss_camera == "predicted"
    )
    predict_evaluation_test_camera = (
        evaluation_heldout_test_camera == "predicted"
    )
    reconstruction_backbone = str(
        model.encoder.cfg.reconstruction_backbone
    ).lower()
    if (
        predict_evaluation_loss_camera or predict_evaluation_test_camera
    ) and reconstruction_backbone not in {"zipmap", "streamvggt"}:
        raise ValueError(
            "Predicted heldout cameras require ZipMap or StreamVGGT, got "
            f"{reconstruction_backbone!r}."
        )
    camera_image_parts = [data["images"]]
    camera_gt_c2w_parts = [data["c2w"]]
    camera_cursor = source_count
    heldout_loss_camera_slice = slice(camera_cursor, camera_cursor)
    if predict_evaluation_loss_camera:
        predicted_loss_view_count = data["heldout_loss"]["rgb"].shape[1]
        heldout_loss_camera_slice = slice(
            camera_cursor, camera_cursor + predicted_loss_view_count
        )
        camera_cursor += predicted_loss_view_count
        camera_image_parts.append(data["heldout_loss"]["rgb"])
        camera_gt_c2w_parts.append(data["heldout_loss"]["c2w"])
    heldout_test_camera_slice = slice(camera_cursor, camera_cursor)
    if predict_evaluation_test_camera:
        predicted_test_view_count = data["heldout_test"]["rgb"].shape[1]
        heldout_test_camera_slice = slice(
            camera_cursor, camera_cursor + predicted_test_view_count
        )
        camera_image_parts.append(data["heldout_test"]["rgb"])
        camera_gt_c2w_parts.append(data["heldout_test"]["c2w"])
    camera_images = torch.cat(camera_image_parts, dim=1)
    camera_gt_c2w = torch.cat(camera_gt_c2w_parts, dim=1)
    with torch.no_grad():
        source_tokens, patch_start_idx = _cache_backbone_tokens(
            model, data["images"], bool(cfg.mixed_precision)
        )
        evaluation_camera_tokens = (
            source_tokens
            if camera_images.shape[1] == source_count
            else _cache_backbone_tokens(
                model,
                camera_images,
                bool(cfg.mixed_precision),
            )[0]
        )
        camera_view_count = camera_images.shape[1]
        if any(
            token.shape[1] < camera_view_count
            for token in evaluation_camera_tokens
        ):
            raise RuntimeError(
                "Backbone tokens do not contain every requested camera view: "
                f"requested={camera_view_count}, "
                "token_views="
                f"{[token.shape[1] for token in evaluation_camera_tokens]}."
            )
        cached_features, cached_depth, _ = _extract_features(
            model,
            source_tokens,
            patch_start_idx,
            data["images"],
            data["image_shape"],
            bool(cfg.mixed_precision),
        )
        cached_training_camera = _predict_camera_geometry(
            model,
            source_tokens,
            data["image_shape"],
            data["c2w"],
        )
        cached_evaluation_camera = _predict_camera_geometry(
            model,
            evaluation_camera_tokens,
            data["image_shape"],
            camera_gt_c2w,
        )
    if cached_evaluation_camera["c2w"].shape[1] != camera_view_count:
        raise RuntimeError(
            "Camera head returned an unexpected view count: "
            f"expected={camera_view_count}, "
            f"actual={cached_evaluation_camera['c2w'].shape[1]}."
        )
    if depth_loss_enabled:
        data["dav3_depth"] = _cache_dav3_depth_targets(
            data["images"],
            data["image_shape"],
            data["c2w"],
            data["intrinsic_px"],
            str(cfg.dav3_weights_path),
            bool(cfg.get("dav3_align_to_gt_camera_scale", True)),
        )

    # The backbone remains fixed. Camera/depth heads consume these cached
    # tokens on every step when trainable, preserving their gradient paths.
    model.encoder.aggregator.to("cpu")
    torch.cuda.empty_cache()

    def build_scene_components(
        *,
        phase: str,
        track_gradients: bool,
        anchor_start: int = 0,
        anchor_end: int | None = None,
    ) -> dict:
        if phase == "training":
            geometry_camera_source = training_geometry_camera_source
            source_render_camera = training_source_render_camera
            heldout_loss_camera = training_heldout_loss_camera
            heldout_test_camera = "gt"
            sim3_alignment_mode = training_sim3_alignment_mode
            sim3_alignment_enabled = training_sim3_alignment_enabled
            predict_loss_camera = False
            predict_test_camera = False
            camera_tokens = source_tokens
            camera_supervision_c2w = data["c2w"]
            cached_camera = cached_training_camera
        elif phase == "evaluation":
            geometry_camera_source = evaluation_geometry_camera_source
            source_render_camera = evaluation_source_render_camera
            heldout_loss_camera = evaluation_heldout_loss_camera
            heldout_test_camera = evaluation_heldout_test_camera
            sim3_alignment_mode = evaluation_sim3_alignment_mode
            sim3_alignment_enabled = evaluation_sim3_alignment_enabled
            predict_loss_camera = predict_evaluation_loss_camera
            predict_test_camera = predict_evaluation_test_camera
            camera_tokens = evaluation_camera_tokens
            camera_supervision_c2w = camera_gt_c2w
            cached_camera = cached_evaluation_camera
        else:
            raise ValueError(f"Unknown offline component phase {phase!r}.")
        with torch.set_grad_enabled(track_gradients):
            if train_depth_head:
                features, predicted_depth_raw, _ = _extract_features(
                    model,
                    source_tokens,
                    patch_start_idx,
                    data["images"],
                    data["image_shape"],
                    bool(cfg.mixed_precision),
                )
            else:
                features = cached_features
                predicted_depth_raw = cached_depth
            camera = (
                _predict_camera_geometry(
                    model,
                    camera_tokens,
                    data["image_shape"],
                    camera_supervision_c2w,
                )
                if train_camera_head
                else cached_camera
            )
            camera_losses = _direct_camera_supervision_losses(
                camera["pose_encoding"][:, :source_count],
                data["c2w"],
                data["intrinsic"],
                translation_weight=float(cfg.camera_translation_weight),
                rotation_weight=float(cfg.camera_rotation_weight),
                focal_weight=float(cfg.camera_focal_weight),
                pose_loss_mode=str(
                    cfg.get("camera_pose_loss_mode", "first_relative")
                ),
                translation_direction_weight=float(
                    cfg.get("camera_translation_direction_weight", 0.1)
                ),
                translation_direction_min_baseline_ratio=float(
                    cfg.get(
                        "camera_translation_direction_min_baseline_ratio",
                        0.0,
                    )
                ),
                translation_direction_baseline_weighting=bool(
                    cfg.get(
                        "camera_translation_direction_baseline_weighting",
                        False,
                    )
                ),
                global_trajectory_weight=float(
                    cfg.get("camera_global_trajectory_weight", 0.0)
                ),
            )

            if depth_loss_enabled:
                (
                    depth_loss,
                    depth_alignment_scale,
                    depth_structure_loss,
                    depth_absolute_scale_loss,
                ) = _sequence_scale_only_log_depth_loss(
                    predicted_depth_raw,
                    data["dav3_depth"],
                    absolute_log_scale_weight=float(
                        cfg.get("depth_absolute_log_scale_weight", 0.1)
                    ),
                )
            else:
                depth_loss = predicted_depth_raw.new_zeros(())
                depth_alignment_scale = depth_loss.reshape(1)
                depth_structure_loss = depth_loss
                depth_absolute_scale_loss = depth_loss

            predicted_depth = torch.nan_to_num(
                predicted_depth_raw.squeeze(-1).float(),
                nan=1e-4,
                posinf=100.0,
                neginf=1e-4,
            ).clamp(1e-4, 100.0)
            # Use the camera head's own first-source-relative gauge. The
            # camera["c2w"] convenience value is anchored to the first GT pose
            # and must not enter a fully predicted-camera experiment.
            predicted_geometry_c2w = camera["relative_c2w"]
            predicted_geometry_intrinsic_px = camera["intrinsic_px"]
            geometry_depth = predicted_depth
            if bool(cfg.get("detach_predicted_camera", False)):
                predicted_geometry_c2w = predicted_geometry_c2w.detach()
                predicted_geometry_intrinsic_px = (
                    predicted_geometry_intrinsic_px.detach()
                )
            if bool(cfg.get("detach_predicted_depth", False)):
                geometry_depth = geometry_depth.detach()

            if sim3_alignment_enabled:
                rotation, translation, scale = estimate_camera_sim3(
                    predicted_geometry_c2w[:, :source_count], data["c2w"]
                )
                aligned_all_c2w = apply_camera_sim3(
                    predicted_geometry_c2w, rotation, translation, scale
                )
            else:
                batch = predicted_geometry_c2w.shape[0]
                rotation = torch.eye(
                    3,
                    device=predicted_geometry_c2w.device,
                    dtype=predicted_geometry_c2w.dtype,
                ).unsqueeze(0).expand(batch, -1, -1)
                translation = torch.zeros(
                    batch,
                    3,
                    device=predicted_geometry_c2w.device,
                    dtype=predicted_geometry_c2w.dtype,
                )
                scale = torch.ones(
                    batch,
                    device=predicted_geometry_c2w.device,
                    dtype=predicted_geometry_c2w.dtype,
                )
                aligned_all_c2w = predicted_geometry_c2w
            aligned_source_c2w = aligned_all_c2w[:, :source_count]
            predicted_source_intrinsic = _normalized_intrinsic(
                predicted_geometry_intrinsic_px[:, :source_count],
                *data["image_shape"],
            )
            if geometry_camera_source == "gt":
                point_rotation = torch.eye(
                    3,
                    device=rotation.device,
                    dtype=rotation.dtype,
                ).unsqueeze(0).expand_as(rotation)
                point_translation = torch.zeros_like(translation)
                point_scale = torch.ones_like(scale)
                anchor_depth = (
                    geometry_depth
                    * scale.detach()[:, None, None, None]
                )
                anchor_c2w = data["c2w"]
                anchor_intrinsic_px = data["intrinsic_px"]
                gaussian_scale_factor = (
                    scale.detach()
                    if str(cfg.gaussian_parameter_source).lower() == "gs_head"
                    else torch.ones_like(scale)
                )
                source_geometry_c2w = data["c2w"]
                source_geometry_intrinsic = data["intrinsic"]
            else:
                point_rotation = rotation
                point_translation = translation
                point_scale = scale
                anchor_depth = geometry_depth
                anchor_c2w = predicted_geometry_c2w[:, :source_count]
                anchor_intrinsic_px = predicted_geometry_intrinsic_px[
                    :, :source_count
                ]
                gaussian_scale_factor = scale
                source_geometry_c2w = aligned_source_c2w
                source_geometry_intrinsic = predicted_source_intrinsic
            anchor_inputs = {
                "features": features.detach(),
                "images": data["images"],
                "depth": anchor_depth,
                "predicted_c2w": anchor_c2w,
                "predicted_intrinsic_px": anchor_intrinsic_px,
                "sim3_rotation": point_rotation,
                "sim3_translation": point_translation,
                "sim3_scale": point_scale,
                "gaussian_scale_factor": gaussian_scale_factor,
            }
            anchors = _build_source_anchor_views(
                model,
                cfg,
                anchor_inputs,
                start=anchor_start,
                end=anchor_end,
            )
            geometry_data = {
                "c2w": source_geometry_c2w,
                "intrinsic": source_geometry_intrinsic,
                "depth": anchor_depth,
            }

            if predict_loss_camera:
                aligned_loss_c2w = aligned_all_c2w[
                    :, heldout_loss_camera_slice
                ]
                predicted_loss_intrinsic = _normalized_intrinsic(
                    predicted_geometry_intrinsic_px[
                        :, heldout_loss_camera_slice
                    ],
                    *data["image_shape"],
                )
            else:
                aligned_loss_c2w = data["heldout_loss"]["c2w"]
                predicted_loss_intrinsic = data["heldout_loss"]["intrinsic"]
            if predict_test_camera:
                aligned_test_c2w = aligned_all_c2w[
                    :, heldout_test_camera_slice
                ]
                predicted_test_intrinsic = _normalized_intrinsic(
                    predicted_geometry_intrinsic_px[
                        :, heldout_test_camera_slice
                    ],
                    *data["image_shape"],
                )
            else:
                aligned_test_c2w = data["heldout_test"]["c2w"]
                predicted_test_intrinsic = data["heldout_test"]["intrinsic"]
            render_data = {
                **data,
                "c2w": (
                    aligned_source_c2w
                    if source_render_camera == "predicted"
                    else data["c2w"]
                ),
                "intrinsic": (
                    predicted_source_intrinsic
                    if source_render_camera == "predicted"
                    else data["intrinsic"]
                ),
                "heldout_loss": {
                    **data["heldout_loss"],
                    "c2w": (
                        aligned_loss_c2w
                        if heldout_loss_camera == "predicted"
                        else data["heldout_loss"]["c2w"]
                    ),
                    "intrinsic": (
                        predicted_loss_intrinsic
                        if heldout_loss_camera == "predicted"
                        else data["heldout_loss"]["intrinsic"]
                    ),
                },
                "heldout_test": {
                    **data["heldout_test"],
                    "c2w": (
                        aligned_test_c2w
                        if heldout_test_camera == "predicted"
                        else data["heldout_test"]["c2w"]
                    ),
                    "intrinsic": (
                        predicted_test_intrinsic
                        if heldout_test_camera == "predicted"
                        else data["heldout_test"]["intrinsic"]
                    ),
                },
            }
            sim3_metrics = {}
            if not track_gradients:
                sim3_metrics = {
                    "sim3_alignment_mode": sim3_alignment_mode,
                    "sim3_alignment_enabled": bool(sim3_alignment_enabled),
                    "camera_coordinate_frame": (
                        "gt_geometry"
                        if geometry_camera_source == "gt"
                        else (
                            "gt_aligned"
                            if sim3_alignment_enabled
                            else "predicted"
                        )
                    ),
                    "geometry_camera_source": geometry_camera_source,
                    "sim3_scale": float(scale.mean().item()),
                    "depth_supervision_loss": float(depth_loss.item()),
                    "depth_structure_loss": float(depth_structure_loss.item()),
                    "depth_absolute_scale_loss": float(
                        depth_absolute_scale_loss.item()
                    ),
                    "depth_alignment_scale_median": float(
                        depth_alignment_scale.median().item()
                    ),
                    "camera_supervision_loss": float(
                        camera_losses["total"].item()
                    ),
                    "camera_pose_loss_mode": str(
                        cfg.get("camera_pose_loss_mode", "first_relative")
                    ).lower(),
                    "camera_translation_loss": float(
                        camera_losses["translation"].item()
                    ),
                    "camera_translation_shape_loss": float(
                        camera_losses["translation_shape"].item()
                    ),
                    "camera_translation_direction_loss": float(
                        camera_losses["translation_direction"].item()
                    ),
                    "camera_translation_alignment_scale": float(
                        camera_losses["translation_alignment_scale"].item()
                    ),
                    "camera_translation_direction_valid_ratio": float(
                        camera_losses[
                            "translation_direction_valid_ratio"
                        ].item()
                    ),
                    "camera_translation_direction_pair_weight_mean": float(
                        camera_losses[
                            "translation_direction_pair_weight_mean"
                        ].item()
                    ),
                    "camera_translation_reference_baseline": float(
                        camera_losses["translation_reference_baseline"].item()
                    ),
                    "camera_rotation_loss": float(
                        camera_losses["rotation"].item()
                    ),
                    "camera_focal_loss": float(camera_losses["focal"].item()),
                    "camera_global_trajectory_loss": float(
                        camera_losses["global_trajectory"].item()
                    ),
                    "camera_global_translation_shape_loss": float(
                        camera_losses["global_translation_shape"].item()
                    ),
                    "camera_global_rotation_loss": float(
                        camera_losses["global_rotation"].item()
                    ),
                    "camera_global_translation_alignment_scale": float(
                        camera_losses[
                            "global_translation_alignment_scale"
                        ].item()
                    ),
                    **_camera_error_metrics(
                        aligned_source_c2w,
                        predicted_source_intrinsic,
                        data["c2w"],
                        data["intrinsic"],
                        (
                            "sim3_source"
                            if sim3_alignment_enabled
                            else "predicted_source"
                        ),
                    ),
                }
                if predict_test_camera:
                    sim3_metrics.update(
                        _camera_error_metrics(
                            aligned_test_c2w,
                            predicted_test_intrinsic,
                            data["heldout_test"]["c2w"],
                            data["heldout_test"]["intrinsic"],
                            (
                                "sim3_test"
                                if sim3_alignment_enabled
                                else "predicted_test"
                            ),
                        )
                    )
                if predict_loss_camera:
                    sim3_metrics.update(
                        _camera_error_metrics(
                            aligned_loss_c2w,
                            predicted_loss_intrinsic,
                            data["heldout_loss"]["c2w"],
                            data["heldout_loss"]["intrinsic"],
                            (
                                "sim3_heldout_loss"
                                if sim3_alignment_enabled
                                else "predicted_heldout_loss"
                            ),
                        )
                    )
            return {
                "anchors": anchors,
                "geometry_data": geometry_data,
                "render_data": render_data,
                "sim3_metrics": sim3_metrics,
                "depth_loss": depth_loss,
                "depth_structure_loss": depth_structure_loss,
                "depth_absolute_scale_loss": depth_absolute_scale_loss,
                "depth_alignment_scale": depth_alignment_scale,
                "camera_losses": camera_losses,
            }

    def build_frozen_scene_components(phase: str) -> dict:
        modules = [model.encoder.gs_head]
        if train_camera_head:
            modules.append(model.encoder.camera_head)
        if train_depth_head:
            modules.append(selected_depth_head)
        previous_modes = [module.training for module in modules]
        for module in modules:
            module.eval()
        try:
            return build_scene_components(
                phase=phase,
                track_gradients=False,
            )
        finally:
            for module, training in zip(modules, previous_modes):
                module.train(training)

    def build_evaluation_components() -> dict:
        return build_frozen_scene_components("evaluation")

    initial_components = build_evaluation_components()
    source_anchors = initial_components["anchors"]
    geometry_data = initial_components["geometry_data"]
    render_data = initial_components["render_data"]
    sim3_metrics = initial_components["sim3_metrics"]
    base_state = _make_state(_concatenate_anchor_views(source_anchors))
    print(
        "Offline base map initialization: "
        f"sources={source_count}, "
        f"heldout_loss={len(data['heldout_loss']['ids'])}, "
        f"heldout_test={len(data['heldout_test']['ids'])}, "
        f"gaussians={base_state.num_gaussians:,}, "
        f"parameter_source={gaussian_parameter_source}, "
        f"per-view opacity={base_state.gaussians.opacities.max().item():.6f}"
    )
    print(f"Training mode: {training_mode}")
    print(
        "Offline training cameras: "
        f"geometry={training_geometry_camera_source}, "
        f"source={training_source_render_camera}, "
        f"heldout_loss={training_heldout_loss_camera}; "
        f"sim3={training_sim3_alignment_mode}; "
        "camera_head=GT-supervised"
    )
    print(
        "Offline evaluation cameras: "
        f"geometry={evaluation_geometry_camera_source}, "
        f"source={evaluation_source_render_camera}, "
        f"heldout_loss={evaluation_heldout_loss_camera}, "
        f"heldout_test={evaluation_heldout_test_camera}"
    )
    if periodic_all_append_enabled:
        print(
            "Periodic point-map densification: every "
            f"{periodic_all_append_interval}th source view appends covered "
            "pixels as refinement GS with "
            "opacity_scale="
            f"{float(cfg.periodic_all_append_opacity_scale):g}, "
            "scale_multiplier="
            f"{float(cfg.periodic_all_append_scale_multiplier):g}."
        )
    if training_mode == "incremental_per_frame":
        print(
            f"Incremental optimization: {append_mode} append, one backward and "
            "optimizer.step per arriving frame, then state.detach()."
        )
    elif training_mode == "incremental_full_graph":
        print(
            f"Full-graph incremental optimization: {append_mode} append, no "
            "frame-to-frame detach, one backward and optimizer.step per "
            "complete source-view sweep."
        )
    elif training_mode == "incremental_final_map":
        print(
            f"Final-map incremental optimization: {append_mode} append and "
            "residual prediction per arriving frame, no intermediate RGB "
            "losses or frame-to-frame detach, then one all-source MSE/LPIPS "
            "objective plus heldout keep-gate supervision, backward, and "
            "optimizer.step per complete source-view sweep."
        )
    if evaluation_sim3_alignment_enabled:
        print(
            "Evaluation source-camera Sim(3): "
            + json.dumps(sim3_metrics, sort_keys=True)
        )
    else:
        print(
            "Evaluation source-camera Sim(3): disabled; predicted cameras, "
            "depth, and Gaussians remain in the reconstruction backbone's "
            "native gauge."
        )
    trainable_names = []
    if residual_enabled:
        trainable_names.append("offline shared GS residual MLP")
    if keep_gate_enabled:
        trainable_names.append("scene-specific Gaussian keep logits")
    if train_gs_head:
        trainable_names.append("final GS parameter head")
    if train_camera_head:
        trainable_names.append("camera head")
    if train_depth_head:
        trainable_names.append(f"{depth_head_source} depth head")
    print(
        "Frozen modules: reconstruction backbone, Gaussian/DPT feature head, "
        "decoder. Trainable modules: " + ", ".join(trainable_names) + "."
    )
    print(
        "Geometry supervision: "
        f"camera={train_camera_head} (weight={float(cfg.loss_camera):g}), "
        "camera_pose_loss_mode="
        f"{str(cfg.get('camera_pose_loss_mode', 'first_relative')).lower()}, "
        "camera_translation_direction_weight="
        f"{float(cfg.get('camera_translation_direction_weight', 0.1)):g}, "
        "camera_translation_direction_min_baseline_ratio="
        f"{float(cfg.get('camera_translation_direction_min_baseline_ratio', 0.0)):g}, "
        "camera_translation_direction_baseline_weighting="
        f"{bool(cfg.get('camera_translation_direction_baseline_weighting', False))}, "
        "camera_global_trajectory_weight="
        f"{float(cfg.get('camera_global_trajectory_weight', 0.0)):g}, "
        "camera_grad_clip="
        f"{float(cfg.get('camera_grad_clip', 0.0)):g}, "
        f"depth={depth_loss_enabled} "
        f"(target={depth_loss_target}, weight={float(cfg.loss_depth):g}); "
        "RGB geometry gradients: "
        f"camera={'detached' if bool(cfg.detach_predicted_camera) else 'enabled'}, "
        f"depth={'detached' if bool(cfg.detach_predicted_depth) else 'enabled'}"
    )
    print(
        "Final-map perceptual supervision: "
        f"enabled={lpips_loss_enabled}, "
        f"weight={float(cfg.get('loss_lpips', 0.05)):g}, "
        f"apply_after_step={int(cfg.get('lpips_apply_after_step', 0))}."
    )
    print(
        "Heldout keep-gate supervision: "
        f"enabled={keep_gate_enabled}, "
        f"frames={data['heldout_loss']['ids']}, "
        f"rgb_weight={float(cfg.get('heldout_loss_weight', 1.0)):g}, "
        "active_count="
        f"{float(cfg.get('keep_gate_active_count_weight', 0.0)):g}"
        " * mean(keep), "
        f"detach_map={bool(cfg.get('keep_gate_detach_heldout_map', True))}."
    )
    print(
        "Residual parameterization: "
        + (
            "disabled"
            if not residual_enabled
            else "unbounded direct MLP outputs (gates and consistency bounds disabled)"
            if bool(cfg.get("unbounded_residual_updates", False))
            else "bounded gated updates"
        )
    )
    print(
        "Render cameras: "
        f"training_source={training_source_render_camera}, "
        f"training_heldout_loss={training_heldout_loss_camera}; "
        f"evaluation_source={evaluation_source_render_camera}, "
        f"evaluation_heldout_loss={evaluation_heldout_loss_camera}, "
        f"evaluation_heldout_test={evaluation_heldout_test_camera}"
    )
    if predict_evaluation_loss_camera or predict_evaluation_test_camera:
        predicted_heldout_roles = []
        if predict_evaluation_loss_camera:
            predicted_heldout_roles.append("heldout-loss")
        if predict_evaluation_test_camera:
            predicted_heldout_roles.append("heldout-test")
        print(
            "Predicted "
            + "/".join(predicted_heldout_roles)
            + " cameras append their RGBs after source RGBs and run the "
            + f"{reconstruction_backbone} camera path; evaluation is "
            "camera-conditioned, not strict blind NVS."
        )
    head = SymmetricGSResidualHead(
        harmonic_dim=model.encoder.nums_sh * 3,
        hidden_dim=int(cfg.hidden_dim),
        chunk_size=int(cfg.chunk_size),
        visual_enabled=bool(cfg.visual_enabled),
        visual_feature_dim=int(cfg.visual_feature_dim),
        visual_hidden_dim=int(cfg.visual_hidden_dim),
    ).to(device)
    if bool(cfg.unbounded_residual_updates):
        # Direct residual mode does not consume learned update gates.
        head.attribute_decoders["gates"].requires_grad_(False)
    keep_gate = None
    if keep_gate_enabled:
        max_gate_gaussians = sum(
            int(anchor.means.shape[1]) for anchor in source_anchors
        )
        keep_gate = SceneGaussianKeepGate(
            max_gate_gaussians,
            float(cfg.get("keep_gate_initial_probability", 0.99)),
        ).to(device)
    lpips_model = None
    if lpips_loss_enabled:
        lpips_model = LPIPS(net="vgg").to(device).eval()
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)
    optimizer_groups = []
    if residual_enabled:
        optimizer_groups.append(
            {"params": head.parameters(), "lr": float(cfg.lr)}
        )
    if keep_gate is not None:
        optimizer_groups.append(
            {"params": keep_gate.parameters(), "lr": float(cfg.keep_gate_lr)}
        )
    if train_gs_head:
        optimizer_groups.append(
            {
                "params": model.encoder.gs_head.parameters(),
                "lr": float(cfg.gs_head_lr),
            }
        )
    if train_camera_head:
        optimizer_groups.append(
            {
                "params": model.encoder.camera_head.parameters(),
                "lr": float(cfg.camera_head_lr),
            }
        )
    if train_depth_head:
        optimizer_groups.append(
            {
                "params": selected_depth_head.parameters(),
                "lr": float(cfg.depth_head_lr),
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(cfg.weight_decay),
    )
    optimization_parameters = (
        list(head.parameters()) if residual_enabled else []
    )
    if keep_gate is not None:
        optimization_parameters.extend(keep_gate.parameters())
    if train_gs_head:
        optimization_parameters.extend(model.encoder.gs_head.parameters())
    if train_camera_head:
        optimization_parameters.extend(model.encoder.camera_head.parameters())
    if train_depth_head:
        optimization_parameters.extend(selected_depth_head.parameters())
    camera_optimization_parameters = (
        list(model.encoder.camera_head.parameters()) if train_camera_head else []
    )

    def clip_optimization_gradients(
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_grad_clip = float(cfg.get("camera_grad_clip", 0.0))
        if camera_optimization_parameters:
            camera_grad_norm = torch.nn.utils.clip_grad_norm_(
                camera_optimization_parameters,
                camera_grad_clip if camera_grad_clip > 0.0 else float("inf"),
            )
        else:
            camera_grad_norm = reference.new_zeros(())
        grad_clip = float(cfg.grad_clip)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            optimization_parameters,
            grad_clip if grad_clip > 0.0 else float("inf"),
        )
        return grad_norm, camera_grad_norm

    checkpoint_gs_head = model.encoder.gs_head if train_gs_head else None
    checkpoint_camera_head = (
        model.encoder.camera_head if train_camera_head else None
    )
    checkpoint_depth_head = selected_depth_head if train_depth_head else None

    def current_global_components(
        *,
        phase: str,
        track_anchor_gradients: bool,
    ) -> tuple[
        StreamingGaussianState,
        object,
        dict[str, torch.Tensor],
        dict,
    ]:
        components = (
            build_scene_components(
                phase=phase,
                track_gradients=True,
            )
            if track_anchor_gradients
            else build_frozen_scene_components(phase)
        )
        anchors = components["anchors"]
        state = _make_state(_concatenate_anchor_views(anchors))
        if residual_enabled:
            evidence, cache = _build_gradient_evidence(
                model, cfg, state, components["render_data"]
            )
            cache = {key: value.detach() for key, value in cache.items()}
        else:
            evidence, cache = None, {}
        return state, evidence, cache, components

    if training_mode == "incremental_per_frame":
        corrected_state = _evaluate_incremental(
            model,
            head,
            keep_gate,
            cfg,
            source_anchors,
            geometry_data,
            render_data,
            0,
            output_dir,
            sim3_metrics,
        )
        step = 0
        sweep = 0
        max_steps = int(cfg.max_steps)
        while step < max_steps:
            state = None
            for view_idx in range(source_count):
                if step >= max_steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                training_components = build_scene_components(
                    phase="training",
                    track_gradients=True,
                    anchor_start=view_idx,
                    anchor_end=view_idx + 1,
                )
                current_anchor = training_components["anchors"][0]
                geometry_data = training_components["geometry_data"]
                render_data = training_components["render_data"]
                append_stats = {
                    "append_ratio": 0.0,
                    "appended_count": 0.0,
                    "point_appended_count": 0.0,
                    "periodic_appended_count": 0.0,
                    "periodic_keyframe": 0.0,
                    "point_depth_scale": 1.0,
                }
                if view_idx == 0:
                    state = _make_state(current_anchor)
                else:
                    if state is None:
                        raise RuntimeError("Incremental state was not initialized.")
                    state = state.detach()
                    state, append_stats = _append_view(
                        model,
                        cfg,
                        state,
                        current_anchor,
                        view_idx,
                        geometry_data["depth"][:, view_idx],
                        geometry_data["c2w"][:, view_idx],
                        geometry_data["intrinsic"][:, view_idx],
                        render_data["image_shape"],
                    )

                if state is None:
                    raise RuntimeError("Incremental state was not initialized.")
                updated_state, stats = _apply_incremental_residual(
                    model, head, cfg, state, render_data, view_idx
                )
                current_loss, replay_loss = _incremental_rgb_losses(
                    model, cfg, updated_state, render_data, view_idx
                )
                weighted_rgb_loss = (
                    float(cfg.get("incremental_current_loss_weight", 1.0))
                    * current_loss
                    + float(cfg.get("incremental_replay_loss_weight", 0.5))
                    * replay_loss
                )
                regularization_loss = (
                    float(cfg.residual_regularization_weight)
                    * stats["regularization"]
                )
                depth_loss = (
                    float(cfg.loss_depth) * training_components["depth_loss"]
                )
                camera_loss = (
                    float(cfg.loss_camera)
                    * training_components["camera_losses"]["total"]
                    if train_camera_head
                    else weighted_rgb_loss.new_zeros(())
                )
                loss = (
                    weighted_rgb_loss
                    + regularization_loss
                    + depth_loss
                    + camera_loss
                )
                loss.backward()
                grad_norm, camera_grad_norm = clip_optimization_gradients(
                    loss
                )
                if (
                    not torch.isfinite(loss)
                    or not torch.isfinite(grad_norm)
                    or not torch.isfinite(camera_grad_norm)
                ):
                    raise FloatingPointError(
                        f"Non-finite incremental optimization at step {step + 1}, "
                        f"view={view_idx}: loss={loss.item()}, "
                        f"grad_norm={grad_norm.item()}, "
                        f"camera_grad_norm={camera_grad_norm.item()}."
                    )
                optimizer.step()
                state = updated_state.detach()
                step += 1

                current_psnr = -10.0 * math.log10(
                    max(current_loss.detach().item(), 1e-10)
                )
                replay_psnr = (
                    -10.0
                    * math.log10(max(replay_loss.detach().item(), 1e-10))
                    if view_idx > 0
                    and int(cfg.get("incremental_replay_views", -1)) != 0
                    else 0.0
                )
                _record_training_metrics(
                    output_dir,
                    {
                        "step": float(step),
                        "sweep": float(sweep),
                        "view_idx": float(view_idx),
                        "frame_id": float(render_data["ids"][view_idx]),
                        "loss": float(loss.detach().item()),
                        "loss_rgb_source": float(weighted_rgb_loss.detach().item()),
                        "loss_rgb_current": float(current_loss.detach().item()),
                        "loss_rgb_replay": float(replay_loss.detach().item()),
                        "loss_regularization": float(
                            regularization_loss.detach().item()
                        ),
                        "loss_depth": float(depth_loss.detach().item()),
                        "loss_depth_structure": float(
                            training_components["depth_structure_loss"]
                            .detach()
                            .item()
                        ),
                        "loss_depth_absolute_scale": float(
                            training_components["depth_absolute_scale_loss"]
                            .detach()
                            .item()
                        ),
                        "loss_camera": float(camera_loss.detach().item()),
                        "loss_camera_translation": float(
                            training_components["camera_losses"]["translation"]
                            .detach()
                            .item()
                        ),
                        "loss_camera_translation_shape": float(
                            training_components["camera_losses"]
                            ["translation_shape"]
                            .detach()
                            .item()
                        ),
                        "loss_camera_translation_direction": float(
                            training_components["camera_losses"]
                            ["translation_direction"]
                            .detach()
                            .item()
                        ),
                        "camera_translation_alignment_scale": float(
                            training_components["camera_losses"]
                            ["translation_alignment_scale"]
                            .detach()
                            .item()
                        ),
                        "camera_translation_direction_valid_ratio": float(
                            training_components["camera_losses"]
                            ["translation_direction_valid_ratio"]
                            .detach()
                            .item()
                        ),
                        "camera_translation_direction_pair_weight_mean": float(
                            training_components["camera_losses"]
                            ["translation_direction_pair_weight_mean"]
                            .detach()
                            .item()
                        ),
                        "loss_camera_rotation": float(
                            training_components["camera_losses"]["rotation"]
                            .detach()
                            .item()
                        ),
                        "loss_camera_focal": float(
                            training_components["camera_losses"]["focal"]
                            .detach()
                            .item()
                        ),
                        "loss_camera_global_trajectory": float(
                            training_components["camera_losses"]
                            ["global_trajectory"]
                            .detach()
                            .item()
                        ),
                        "loss_camera_global_translation_shape": float(
                            training_components["camera_losses"]
                            ["global_translation_shape"]
                            .detach()
                            .item()
                        ),
                        "loss_camera_global_rotation": float(
                            training_components["camera_losses"]
                            ["global_rotation"]
                            .detach()
                            .item()
                        ),
                        "source_train_psnr": float(current_psnr),
                        "current_train_psnr": float(current_psnr),
                        "replay_train_psnr": float(replay_psnr),
                        "grad_norm": float(grad_norm.detach().item()),
                        "camera_grad_norm": float(
                            camera_grad_norm.detach().item()
                        ),
                        "gaussian_count": float(state.num_gaussians),
                        "append_ratio": float(append_stats["append_ratio"]),
                        "appended_gs": float(append_stats["appended_count"]),
                        "point_appended_gs": float(
                            append_stats["point_appended_count"]
                        ),
                        "periodic_appended_gs": float(
                            append_stats["periodic_appended_count"]
                        ),
                        "periodic_keyframe": float(
                            append_stats["periodic_keyframe"]
                        ),
                        "point_depth_scale": float(
                            append_stats["point_depth_scale"]
                        ),
                        "residual_regularization": float(
                            stats["regularization"].detach().item()
                        ),
                        "residual_gate_mean": float(
                            stats["gate_mean"].detach().item()
                        ),
                        "residual_mean_shift_mean": float(
                            stats["mean_shift_mean"].detach().item()
                        ),
                        "residual_rotation_shift_mean": float(
                            stats["rotation_shift_mean"].detach().item()
                        ),
                        "residual_scale_relative_shift_mean": float(
                            stats["scale_relative_shift_mean"].detach().item()
                        ),
                        "residual_opacity_shift_mean": float(
                            stats["opacity_shift_mean"].detach().item()
                        ),
                        "residual_harmonics_shift_mean": float(
                            stats["harmonics_shift_mean"].detach().item()
                        ),
                    },
                )
                if step == 1 or step % int(cfg.log_every) == 0:
                    print(
                        f"[incremental train] step={step} sweep={sweep} "
                        f"frame={int(render_data['ids'][view_idx])} "
                        f"loss={loss.item():.6f} current_psnr={current_psnr:.3f} "
                        f"replay_psnr={replay_psnr:.3f} "
                        f"depth={depth_loss.item():.5f} "
                        f"camera={camera_loss.item():.5f} "
                        f"gs={state.num_gaussians:,} "
                        f"grad_norm={grad_norm.item():.4f} "
                        f"camera_grad_norm={camera_grad_norm.item():.4f}"
                    )
                if step % int(cfg.eval_every) == 0 or step == max_steps:
                    eval_components = build_evaluation_components()
                    corrected_state = _evaluate_incremental(
                        model,
                        head,
                        keep_gate,
                        cfg,
                        eval_components["anchors"],
                        eval_components["geometry_data"],
                        eval_components["render_data"],
                        step,
                        output_dir,
                        eval_components["sim3_metrics"],
                    )
                if bool(cfg.save_checkpoints) and (
                    step % int(cfg.save_every) == 0 or step == max_steps
                ):
                    _save_checkpoint(
                        head,
                        optimizer,
                        step,
                        output_dir,
                        keep_gate=keep_gate,
                        gs_head=checkpoint_gs_head,
                        camera_head=checkpoint_camera_head,
                        depth_head=checkpoint_depth_head,
                    )
            sweep += 1
    elif training_mode in {"incremental_full_graph", "incremental_final_map"}:
        corrected_state = _evaluate_incremental(
            model,
            head,
            keep_gate,
            cfg,
            source_anchors,
            geometry_data,
            render_data,
            0,
            output_dir,
            sim3_metrics,
        )
        # Step 0 is diagnostic only. Keeping this complete map (and the
        # duplicated initial component dictionaries) alive would overlap it
        # with every subsequent 1M+ Gaussian training rollout.
        del corrected_state
        del initial_components, source_anchors, geometry_data, render_data, base_state
        max_steps = int(cfg.max_steps)
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            training_components = build_scene_components(
                phase="training",
                track_gradients=True,
            )
            if training_mode == "incremental_final_map":
                (
                    state,
                    loss,
                    final_rgb_loss,
                    final_heldout_rgb_loss,
                    final_lpips_loss,
                    active_count_loss,
                    regularization_loss,
                    stats,
                    append_stats,
                ) = _build_final_map_incremental_loss(
                    model,
                    head,
                    lpips_model,
                    keep_gate,
                    cfg,
                    training_components["anchors"],
                    training_components["geometry_data"],
                    training_components["render_data"],
                    step,
                )
                current_loss = final_rgb_loss.new_zeros(())
                replay_loss = final_rgb_loss.new_zeros(())
                weighted_rgb_loss = final_rgb_loss
            else:
                (
                    state,
                    loss,
                    current_loss,
                    replay_loss,
                    weighted_rgb_loss,
                    regularization_loss,
                    stats,
                    append_stats,
                ) = _build_full_graph_incremental_loss(
                    model,
                    head,
                    cfg,
                    training_components["anchors"],
                    training_components["geometry_data"],
                    training_components["render_data"],
                )
                final_rgb_loss = weighted_rgb_loss.new_zeros(())
                final_heldout_rgb_loss = weighted_rgb_loss.new_zeros(())
                final_lpips_loss = weighted_rgb_loss.new_zeros(())
                active_count_loss = weighted_rgb_loss.new_zeros(())
            depth_loss = (
                float(cfg.loss_depth) * training_components["depth_loss"]
            )
            camera_loss = (
                float(cfg.loss_camera)
                * training_components["camera_losses"]["total"]
                if train_camera_head
                else loss.new_zeros(())
            )
            loss = loss + depth_loss + camera_loss
            loss.backward()
            grad_norm, camera_grad_norm = clip_optimization_gradients(
                loss
            )
            if (
                not torch.isfinite(loss)
                or not torch.isfinite(grad_norm)
                or not torch.isfinite(camera_grad_norm)
            ):
                raise FloatingPointError(
                    f"Non-finite {training_mode} optimization at step {step}: "
                    f"loss={loss.item()}, grad_norm={grad_norm.item()}, "
                    f"camera_grad_norm={camera_grad_norm.item()}."
                )
            optimizer.step()

            if training_mode == "incremental_final_map":
                source_train_psnr = -10.0 * math.log10(
                    max(final_rgb_loss.detach().item(), 1e-10)
                )
                heldout_weight = float(cfg.get("heldout_loss_weight", 1.0))
                heldout_train_mse = (
                    final_heldout_rgb_loss.detach().item() / heldout_weight
                    if heldout_loss_enabled and heldout_weight > 0.0
                    else 0.0
                )
                heldout_train_psnr = (
                    -10.0 * math.log10(max(heldout_train_mse, 1e-10))
                    if heldout_loss_enabled
                    else 0.0
                )
                current_psnr = 0.0
                replay_psnr = 0.0
            else:
                current_psnr = -10.0 * math.log10(
                    max(current_loss.detach().item(), 1e-10)
                )
                replay_psnr = (
                    -10.0
                    * math.log10(max(replay_loss.detach().item(), 1e-10))
                    if int(cfg.get("incremental_replay_views", -1)) != 0
                    else 0.0
                )
                source_train_psnr = current_psnr
                heldout_train_psnr = 0.0
            _record_training_metrics(
                output_dir,
                {
                    "step": float(step),
                    "sweep": float(step - 1),
                    "view_idx": -1.0,
                    "frame_id": -1.0,
                    "loss": float(loss.detach().item()),
                    "loss_rgb_source": float(weighted_rgb_loss.detach().item()),
                    "loss_rgb_final_map": float(final_rgb_loss.detach().item()),
                    "loss_rgb_heldout": float(
                        final_heldout_rgb_loss.detach().item()
                    ),
                    "loss_lpips": float(final_lpips_loss.detach().item()),
                    "loss_active_count": float(
                        active_count_loss.detach().item()
                    ),
                    "loss_rgb_current": float(current_loss.detach().item()),
                    "loss_rgb_replay": float(replay_loss.detach().item()),
                    "loss_regularization": float(
                        regularization_loss.detach().item()
                    ),
                    "loss_depth": float(depth_loss.detach().item()),
                    "loss_depth_structure": float(
                        training_components["depth_structure_loss"].detach().item()
                    ),
                    "loss_depth_absolute_scale": float(
                        training_components["depth_absolute_scale_loss"]
                        .detach()
                        .item()
                    ),
                    "loss_camera": float(camera_loss.detach().item()),
                    "loss_camera_translation": float(
                        training_components["camera_losses"]["translation"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_translation_shape": float(
                        training_components["camera_losses"]["translation_shape"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_translation_direction": float(
                        training_components["camera_losses"]
                        ["translation_direction"]
                        .detach()
                        .item()
                    ),
                    "camera_translation_alignment_scale": float(
                        training_components["camera_losses"]
                        ["translation_alignment_scale"]
                        .detach()
                        .item()
                    ),
                    "camera_translation_direction_valid_ratio": float(
                        training_components["camera_losses"]
                        ["translation_direction_valid_ratio"]
                        .detach()
                        .item()
                    ),
                    "camera_translation_direction_pair_weight_mean": float(
                        training_components["camera_losses"]
                        ["translation_direction_pair_weight_mean"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_rotation": float(
                        training_components["camera_losses"]["rotation"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_focal": float(
                        training_components["camera_losses"]["focal"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_global_trajectory": float(
                        training_components["camera_losses"]
                        ["global_trajectory"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_global_translation_shape": float(
                        training_components["camera_losses"]
                        ["global_translation_shape"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_global_rotation": float(
                        training_components["camera_losses"]
                        ["global_rotation"]
                        .detach()
                        .item()
                    ),
                    "source_train_psnr": float(source_train_psnr),
                    "heldout_train_psnr": float(heldout_train_psnr),
                    "current_train_psnr": float(current_psnr),
                    "replay_train_psnr": float(replay_psnr),
                    "grad_norm": float(grad_norm.detach().item()),
                    "camera_grad_norm": float(
                        camera_grad_norm.detach().item()
                    ),
                    "gaussian_count": float(state.num_gaussians),
                    "append_ratio": float(append_stats["append_ratio"]),
                    "appended_gs": float(append_stats["appended_count"]),
                    "point_appended_gs": float(
                        append_stats["point_appended_count"]
                    ),
                    "periodic_appended_gs": float(
                        append_stats["periodic_appended_count"]
                    ),
                    "periodic_keyframes": float(
                        append_stats["periodic_keyframes"]
                    ),
                    "point_depth_scale": float(
                        append_stats["point_depth_scale"]
                    ),
                    "residual_regularization": float(
                        stats["regularization"].detach().item()
                    ),
                    "residual_gate_mean": float(
                        stats["gate_mean"].detach().item()
                    ),
                    "residual_mean_shift_mean": float(
                        stats["mean_shift_mean"].detach().item()
                    ),
                    "residual_rotation_shift_mean": float(
                        stats["rotation_shift_mean"].detach().item()
                    ),
                    "residual_scale_relative_shift_mean": float(
                        stats["scale_relative_shift_mean"].detach().item()
                    ),
                    "residual_opacity_shift_mean": float(
                        stats["opacity_shift_mean"].detach().item()
                    ),
                    "residual_harmonics_shift_mean": float(
                        stats["harmonics_shift_mean"].detach().item()
                    ),
                    "keep_probability_mean": float(
                        stats.get(
                            "keep_probability_mean",
                            final_rgb_loss.new_ones(()),
                        )
                        .detach()
                        .item()
                    ),
                    "keep_ratio_ge_01": float(
                        stats.get(
                            "keep_ratio_ge_01",
                            final_rgb_loss.new_ones(()),
                        )
                        .detach()
                        .item()
                    ),
                    "keep_ratio_ge_05": float(
                        stats.get(
                            "keep_ratio_ge_05",
                            final_rgb_loss.new_ones(()),
                        )
                        .detach()
                        .item()
                    ),
                },
            )
            if step == 1 or step % int(cfg.log_every) == 0:
                if training_mode == "incremental_final_map":
                    print(
                        f"[final-map train] step={step} loss={loss.item():.6f} "
                        f"source_psnr={source_train_psnr:.3f} "
                        f"heldout_psnr={heldout_train_psnr:.3f} "
                        f"lpips_loss={final_lpips_loss.item():.5f} "
                        f"active_loss={active_count_loss.item():.5f} "
                        "keep_mean="
                        f"{stats['keep_probability_mean'].item():.4f} "
                        f"depth={depth_loss.item():.5f} "
                        f"camera={camera_loss.item():.5f} "
                        f"gs={state.num_gaussians:,} "
                        f"grad_norm={grad_norm.item():.4f} "
                        f"camera_grad_norm={camera_grad_norm.item():.4f}"
                    )
                else:
                    print(
                        f"[full-graph train] step={step} loss={loss.item():.6f} "
                        f"current_psnr={current_psnr:.3f} "
                        f"replay_psnr={replay_psnr:.3f} "
                        f"depth={depth_loss.item():.5f} "
                        f"camera={camera_loss.item():.5f} "
                        f"gs={state.num_gaussians:,} "
                        f"grad_norm={grad_norm.item():.4f} "
                        f"camera_grad_norm={camera_grad_norm.item():.4f}"
                    )
            if step % int(cfg.eval_every) == 0 or step == max_steps:
                eval_components = build_evaluation_components()
                corrected_state = _evaluate_incremental(
                    model,
                    head,
                    keep_gate,
                    cfg,
                    eval_components["anchors"],
                    eval_components["geometry_data"],
                    eval_components["render_data"],
                    step,
                    output_dir,
                    eval_components["sim3_metrics"],
                )
                del eval_components
                if step != max_steps:
                    del corrected_state
            if bool(cfg.save_checkpoints) and (
                step % int(cfg.save_every) == 0 or step == max_steps
            ):
                _save_checkpoint(
                    head,
                    optimizer,
                    step,
                    output_dir,
                    keep_gate=keep_gate,
                    gs_head=checkpoint_gs_head,
                    camera_head=checkpoint_camera_head,
                    depth_head=checkpoint_depth_head,
                )
            # Explicitly drop the previous autograd rollout before Python
            # evaluates the right-hand side that builds the next one.
            del (
                state,
                loss,
                final_rgb_loss,
                final_heldout_rgb_loss,
                final_lpips_loss,
                active_count_loss,
                regularization_loss,
                stats,
                append_stats,
                current_loss,
                replay_loss,
                weighted_rgb_loss,
                depth_loss,
                camera_loss,
                grad_norm,
                camera_grad_norm,
                training_components,
            )
    else:
        print(
            "Building equal-view gradient evidence from "
            f"{source_count} source views..."
        )
        (
            evaluation_base_state,
            evaluation_gradient_evidence,
            evaluation_render_cache,
            evaluation_components,
        ) = current_global_components(
            phase="evaluation",
            track_anchor_gradients=False,
        )
        corrected_state = _evaluate(
            model,
            head,
            cfg,
            evaluation_base_state,
            evaluation_gradient_evidence,
            evaluation_render_cache,
            evaluation_components["render_data"],
            0,
            output_dir,
            evaluation_components["sim3_metrics"],
        )
        head.train()
        train_scene_heads = train_gs_head or train_camera_head or train_depth_head
        if not train_scene_heads:
            (
                frozen_training_base_state,
                frozen_training_gradient_evidence,
                frozen_training_render_cache,
                frozen_training_components,
            ) = current_global_components(
                phase="training",
                track_anchor_gradients=False,
            )
        for step in range(1, int(cfg.max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            if train_scene_heads:
                (
                    training_base_state,
                    training_gradient_evidence,
                    training_render_cache,
                    training_components,
                ) = current_global_components(
                    phase="training",
                    track_anchor_gradients=True,
                )
            else:
                training_base_state = frozen_training_base_state
                training_gradient_evidence = frozen_training_gradient_evidence
                training_render_cache = frozen_training_render_cache
                training_components = frozen_training_components
            training_render_data = training_components["render_data"]
            if residual_enabled:
                evidence = _attach_visual_evidence(
                    head,
                    cfg,
                    training_base_state,
                    training_render_data,
                    training_gradient_evidence,
                    training_render_cache,
                )
                corrected_state, stats = _predict_corrected_state(
                    head,
                    cfg,
                    training_base_state,
                    evidence,
                    training_render_data["c2w"][:, 0],
                )
            else:
                corrected_state = training_base_state
                stats = _mean_update_stats(
                    [], training_base_state.gaussians.means
                )
            rendering = _render(
                model,
                corrected_state.gaussians,
                training_render_data["c2w"],
                training_render_data["intrinsic"],
                training_render_data["image_shape"],
            )
            rgb_loss = (
                rendering.color.float() - training_render_data["images"].float()
            ).square().mean()
            regularization_loss_tensor = (
                float(cfg.residual_regularization_weight)
                * stats["regularization"]
            )
            depth_loss = float(cfg.loss_depth) * training_components["depth_loss"]
            camera_loss = (
                float(cfg.loss_camera)
                * training_components["camera_losses"]["total"]
                if train_camera_head
                else rgb_loss.new_zeros(())
            )
            loss = (
                rgb_loss
                + regularization_loss_tensor
                + depth_loss
                + camera_loss
            )
            loss.backward()
            grad_norm, camera_grad_norm = clip_optimization_gradients(
                loss
            )
            if (
                not torch.isfinite(loss)
                or not torch.isfinite(grad_norm)
                or not torch.isfinite(camera_grad_norm)
            ):
                raise FloatingPointError(
                    f"Non-finite offline optimization at step {step}: "
                    f"loss={loss.item()}, grad_norm={grad_norm.item()}, "
                    f"camera_grad_norm={camera_grad_norm.item()}."
                )
            optimizer.step()

            source_train_psnr = -10.0 * math.log10(max(rgb_loss.item(), 1e-10))
            regularization_loss = (
                float(cfg.residual_regularization_weight)
                * stats["regularization"].detach().item()
            )
            _record_training_metrics(
                output_dir,
                {
                    "step": float(step),
                    "loss": float(loss.detach().item()),
                    "loss_rgb_source": float(rgb_loss.detach().item()),
                    "loss_regularization": float(regularization_loss),
                    "loss_depth": float(depth_loss.detach().item()),
                    "loss_depth_structure": float(
                        training_components["depth_structure_loss"].detach().item()
                    ),
                    "loss_depth_absolute_scale": float(
                        training_components["depth_absolute_scale_loss"]
                        .detach()
                        .item()
                    ),
                    "loss_camera": float(camera_loss.detach().item()),
                    "loss_camera_translation": float(
                        training_components["camera_losses"]["translation"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_translation_shape": float(
                        training_components["camera_losses"]["translation_shape"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_translation_direction": float(
                        training_components["camera_losses"]
                        ["translation_direction"]
                        .detach()
                        .item()
                    ),
                    "camera_translation_alignment_scale": float(
                        training_components["camera_losses"]
                        ["translation_alignment_scale"]
                        .detach()
                        .item()
                    ),
                    "camera_translation_direction_valid_ratio": float(
                        training_components["camera_losses"]
                        ["translation_direction_valid_ratio"]
                        .detach()
                        .item()
                    ),
                    "camera_translation_direction_pair_weight_mean": float(
                        training_components["camera_losses"]
                        ["translation_direction_pair_weight_mean"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_rotation": float(
                        training_components["camera_losses"]["rotation"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_focal": float(
                        training_components["camera_losses"]["focal"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_global_trajectory": float(
                        training_components["camera_losses"]
                        ["global_trajectory"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_global_translation_shape": float(
                        training_components["camera_losses"]
                        ["global_translation_shape"]
                        .detach()
                        .item()
                    ),
                    "loss_camera_global_rotation": float(
                        training_components["camera_losses"]
                        ["global_rotation"]
                        .detach()
                        .item()
                    ),
                    "source_train_psnr": float(source_train_psnr),
                    "grad_norm": float(grad_norm.detach().item()),
                    "camera_grad_norm": float(
                        camera_grad_norm.detach().item()
                    ),
                    "residual_regularization": float(
                        stats["regularization"].detach().item()
                    ),
                    "residual_gate_mean": float(stats["gate_mean"].detach().item()),
                    "residual_mean_shift_mean": float(
                        stats["mean_shift_mean"].detach().item()
                    ),
                    "residual_rotation_shift_mean": float(
                        stats["rotation_shift_mean"].detach().item()
                    ),
                    "residual_scale_relative_shift_mean": float(
                        stats["scale_relative_shift_mean"].detach().item()
                    ),
                    "residual_opacity_shift_mean": float(
                        stats["opacity_shift_mean"].detach().item()
                    ),
                    "residual_harmonics_shift_mean": float(
                        stats["harmonics_shift_mean"].detach().item()
                    ),
                },
            )

            if step == 1 or step % int(cfg.log_every) == 0:
                print(
                    f"[offline train] step={step} loss={loss.item():.6f} "
                    f"source_psnr={source_train_psnr:.3f} "
                    f"depth={depth_loss.item():.5f} "
                    f"camera={camera_loss.item():.5f} "
                    f"grad_norm={grad_norm.item():.4f} "
                    f"camera_grad_norm={camera_grad_norm.item():.4f}"
                )
            if step % int(cfg.eval_every) == 0 or step == int(cfg.max_steps):
                if train_scene_heads:
                    (
                        eval_base_state,
                        eval_gradient_evidence,
                        eval_render_cache,
                        eval_components,
                    ) = current_global_components(
                        phase="evaluation",
                        track_anchor_gradients=False,
                    )
                else:
                    eval_base_state = evaluation_base_state
                    eval_gradient_evidence = evaluation_gradient_evidence
                    eval_render_cache = evaluation_render_cache
                    eval_components = evaluation_components
                corrected_state = _evaluate(
                    model,
                    head,
                    cfg,
                    eval_base_state,
                    eval_gradient_evidence,
                    eval_render_cache,
                    eval_components["render_data"],
                    step,
                    output_dir,
                    eval_components["sim3_metrics"],
                )
            if bool(cfg.save_checkpoints) and (
                step % int(cfg.save_every) == 0 or step == int(cfg.max_steps)
            ):
                _save_checkpoint(
                    head,
                    optimizer,
                    step,
                    output_dir,
                    keep_gate=keep_gate,
                    gs_head=checkpoint_gs_head,
                    camera_head=checkpoint_camera_head,
                    depth_head=checkpoint_depth_head,
                )

    if bool(cfg.save_final_ply):
        gaussians = corrected_state.gaussians
        export_ply(
            gaussians.means[0],
            gaussians.scales[0],
            gaussians.rotations[0],
            gaussians.harmonics[0],
            gaussians.opacities[0],
            output_dir / "gaussians" / "final_unfiltered.ply",
            filter_gaussians=False,
        )
    _save_checkpoint(
        head,
        optimizer,
        int(cfg.max_steps),
        output_dir,
        keep_gate=keep_gate,
        gs_head=checkpoint_gs_head,
        camera_head=checkpoint_camera_head,
        depth_head=checkpoint_depth_head,
    )
    print(f"Offline GS-MLP experiment complete: {output_dir}")


if __name__ == "__main__":
    main()
