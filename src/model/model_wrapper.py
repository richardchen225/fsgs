from dataclasses import dataclass
from pathlib import Path
import gc
import random
import sys
from typing import Literal, Optional, Protocol, runtime_checkable, Any
import json
import torch
import math
import torchvision
import wandb
from PIL import Image
import cv2
import numpy as np
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from tabulate import tabulate
from torch import Tensor, nn, optim
import torch.nn.functional as F
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
from loss.loss_lpips import LossLpips
from loss.loss_mse import LossMse
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import (
    compute_lpips,
    compute_psnr,
    compute_ssim,
)
from ..evaluation.benchmark_io import summarize_scenes, write_results
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.cuda_memory import log_cuda_memory, reset_cuda_peak_memory
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import (
    inverse_normalize,
    vis_depth_map,
    confidence_map,
    get_overlap_tag,
)
from .camera_phase import resolve_camera_phase_settings
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat

from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .encoder.vggt.utils.rotation import quat_to_mat
from .ply_export import export_ply
from lightning.pytorch.loggers.wandb import WandbLogger
def _gir_training_metric_names(
    cfg,
    compact: bool = False,
) -> tuple[str, ...]:
    """Select compact, non-redundant diagnostics for active GIR paths."""
    if not bool(getattr(cfg, "gir_enabled", False)):
        return ()
    if compact:
        if not bool(
            getattr(cfg, "gir_symmetric_multiview_update_enabled", False)
        ):
            return ()
        return tuple(
            f"gir_symmetric_multiview_{key}"
            for key in (
                "active_ratio",
                "visible_views_mean",
                "mean_conflict_ratio",
                "mean_shift_mean",
                "opacity_shift_mean",
                "harmonics_shift_mean",
            )
        )

    names = ["gir_map_gaussians"]
    if bool(getattr(cfg, "gir_add_gate_enabled", False)):
        names.append("gir_add_gate")

    append_mode = str(getattr(cfg, "gir_append_mode", "learned")).lower()
    if append_mode in {"point_map", "tracker_point_map"}:
        names.extend((
            "gir_point_coverage_ratio",
            "gir_point_append_ratio",
            "gir_point_front_conflict_ratio",
            "gir_point_front_decay_mass_ratio",
            "gir_point_depth_scale",
        ))
        if bool(
            getattr(
                cfg,
                "gir_point_append_align_depth_scale_enabled",
                False,
            )
        ):
            names.extend((
                "gir_point_depth_scale_min",
                "gir_point_depth_scale_max",
            ))
        if bool(getattr(cfg, "gir_opacity_gradient_front_decay_gate_enabled", False)):
            names.append("gir_point_front_gradient_decay_gate_ratio")
    if append_mode in {"tracker", "tracker_point_map"}:
        names.extend((
            "gir_tracker_coverage_ratio",
            "gir_tracker_visibility_ratio",
            "gir_tracker_fb_consistency_ratio",
            "gir_tracker_cycle_error_mean",
        ))
        if append_mode == "tracker":
            names.extend((
                "gir_point_coverage_ratio",
                "gir_point_append_ratio",
            ))
        else:
            names.extend((
                "gir_tracker_point_union_coverage_ratio",
                "gir_confirmed_current_in_front_ratio",
            ))
    if append_mode == "learned":
        names.extend(("gir_add_gate_learned", "gir_add_rate"))

    if bool(
        getattr(cfg, "gir_symmetric_multiview_update_enabled", False)
    ):
        names.extend(
            f"gir_symmetric_multiview_{key}"
            for key in (
                "active_ratio",
                "visible_views_mean",
                "mean_coherence",
                "mean_conflict_ratio",
                "mean_shift_mean",
                "opacity_shift_mean",
                "harmonics_shift_mean",
            )
        )
    if bool(getattr(cfg, "gir_anchor_child_enabled", False)):
        names.extend(
            f"gir_anchor_child_{key}"
            for key in (
                "eligible_anchor_count",
                "expanded_anchor_count",
                "spawned_child_count",
                "active_anchor_count",
                "active_child_count",
                "selected_gradient_score_mean",
                "selected_coherence_mean",
                "map_count_after",
            )
        )

    opacity_keys = (
        "gradient_signal_abs_mean",
        "gradient_update_alignment",
    )
    old_opacity_effective_keys = (
        "gradient_effective_alignment",
        "gradient_current_replay_conflict_ratio",
    )
    mean_gradient_keys = (
        "mean_gradient_current_norm_mean",
        "mean_gradient_replay_norm_mean",
        "mean_gradient_current_replay_cosine",
        "mean_gradient_current_replay_conflict_ratio",
        "mean_gradient_effective_update_alignment",
    )
    old_enabled = bool(getattr(cfg, "gir_old_residual_enabled", True))
    projected = str(getattr(cfg, "gir_old_update_mode", "pixel_topk")).lower() == "projected_shared"
    if old_enabled and projected:
        names.extend((
            "gir_projected_candidate_ratio",
            "gir_projected_consistent_ratio",
            "gir_projected_front_ratio",
            "gir_projected_gate_mean",
            "gir_projected_opacity_shift",
        ))
        if float(getattr(cfg, "gir_projected_front_decay_strength", 0.02)) > 0.0:
            names.append("gir_projected_front_decay_probability")
        if bool(getattr(cfg, "gir_old_opacity_gradient_enabled", False)):
            names.extend(
                f"gir_projected_{key}"
                for key in (*opacity_keys, *old_opacity_effective_keys)
            )
        if bool(getattr(cfg, "gir_mean_gradient_enabled", False)):
            names.extend(f"gir_projected_{key}" for key in mean_gradient_keys)
        if bool(getattr(cfg, "gir_split_enabled", False)):
            names.extend((
                "gir_split_count",
                "gir_split_ratio",
                "gir_split_selected_radius",
                "gir_split_selected_gradient_norm",
                "gir_split_map_count_after",
            ))
        if bool(getattr(cfg, "gir_functional_update_enabled", False)):
            names.extend(
                f"gir_functional_update_{key}"
                for key in (
                    "selected_count",
                    "selected_ratio",
                    "mean_step_norm",
                    "opacity_logit_step_abs",
                    "harmonics_step_norm",
                )
            )
    elif old_enabled:
        names.extend((
            "gir_historical_gate",
            "gir_residual_magnitude",
            "gir_top1_ownership_mean",
            "gir_top1_ownership_above_0_25_ratio",
        ))
        if str(getattr(cfg, "gir_top1_confidence_mode", "none")).lower() != "none":
            names.append("gir_top1_confidence_mean")

    if bool(getattr(cfg, "gir_new_residual_enabled", False)):
        names.extend(
            f"gir_new_residual_{key}"
            for key in (
                "gate_mean",
                "mean_shift_mean",
                "rotation_shift_mean",
                "scale_relative_shift_mean",
                "opacity_shift_mean",
                "harmonics_shift_mean",
            )
        )
        if bool(getattr(cfg, "gir_new_opacity_gradient_enabled", False)):
            names.extend(f"gir_new_residual_{key}" for key in opacity_keys)
        if bool(getattr(cfg, "gir_new_mean_gradient_enabled", False)):
            names.extend(f"gir_new_residual_{key}" for key in mean_gradient_keys)

    if bool(getattr(cfg, "gir_old_delete_enabled", False)):
        names.extend(
            f"gir_old_delete_{key}"
            for key in (
                "candidate_rate", "hard_ratio", "removed_opacity_mass_ratio",
            )
        )
    if bool(getattr(cfg, "gir_old_decay_enabled", False)):
        names.extend(
            f"gir_old_decay_{key}"
            for key in (
                "opacity_mass_ratio", "hard_pruned_ratio",
            )
        )
    return tuple(names)


def _expected_render_depth_for_visualization(
    accumulated_depth: torch.Tensor,
    alpha: torch.Tensor,
    min_alpha: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert gsplat RGB+D output to displayable expected depth."""
    if accumulated_depth.shape != alpha.shape:
        raise ValueError(
            "Rendered depth and alpha must have matching shapes, got "
            f"depth={tuple(accumulated_depth.shape)}, "
            f"alpha={tuple(alpha.shape)}."
        )
    min_alpha = float(min_alpha)
    if not math.isfinite(min_alpha) or not 0.0 <= min_alpha < 1.0:
        raise ValueError(
            f"Rendered-depth min_alpha must be in [0,1), got {min_alpha}."
        )

    depth = accumulated_depth.detach().float()
    opacity = alpha.detach().float()
    valid = (
        torch.isfinite(depth)
        & torch.isfinite(opacity)
        & (opacity >= min_alpha)
    )
    expected_depth = depth / opacity.clamp_min(1e-6)
    valid &= torch.isfinite(expected_depth) & (expected_depth > 0.0)

    valid_values = expected_depth[valid]
    if valid_values.numel() == 0:
        fill_depth = expected_depth.new_tensor(1.0)
    else:
        # Invalid pixels must not push vis_depth_map's percentiles toward zero.
        fill_depth = valid_values.quantile(0.99).clamp_min(1e-6)
    display_depth = torch.where(valid, expected_depth, fill_depth)
    return display_depth, valid

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float
    pretrained_lr: float | None = None
    scratch_lr: float | None = None
    train_base_heads: bool = True
    train_gir: bool = False
    train_gaussian_param_head: bool | None = None
    train_gs_head: bool | None = None
    train_camera_head: bool | None = None
    train_depth_head: bool | None = None
    gaussian_param_head_lr: float | None = None
    gs_head_lr: float | None = None
    camera_lr: float | None = None
    depth_lr: float | None = None
    gir_lr: float | None = None
    base_weight_decay: float = 0.1
    gir_weight_decay: float | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    scheduler_mode: str = "cosine"


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    generate_video: bool
    mode: Literal["inference", "evaluation"]
    image_folder: str
    save_metrics: bool = True
    save_unfiltered_ply: bool = False
    gir_add_gate_prune_threshold: float = 0.0
    gir_test_top1_confidence_mode: Literal[
        "inherit", "none", "floor_sqrt", "sqrt"
    ] = "inherit"
    gir_test_top1_confidence_floor: float = 0.25
    gir_debug_save_intermediates: bool = False
    gir_debug_max_scenes: int = 10
    gir_debug_view_stride: int = 1


@dataclass
class TrainCfg:
    output_path: Path
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    max_val_comparisons: int = 16
    pose_loss_alpha: float = 1.0
    pose_loss_delta: float = 1.0
    camera_focal_weight: float = 1.0
    camera_pose_loss_mode: str = "adjacent_pair"
    camera_translation_direction_weight: float = 0.1
    camera_translation_direction_min_baseline_ratio: float = 0.0
    camera_translation_direction_baseline_weighting: bool = False
    camera_global_trajectory_weight: float = 0.0
    camera_grad_clip: float = 0.0
    debug_metrics_enabled: bool = False
    debug_metrics_every_n_steps: int = 10
    debug_metrics_filename: str = "train_debug_metrics.jsonl"
    debug_metrics_print: bool = True
    compact_logging: bool = False
    cxt_depth_weight: float = 0.01
    weight_pose: float = 1.0
    weight_depth: float = 1.0
    weight_normal: float = 1.0
    render_ba: bool = False
    render_ba_after_step: int = 0


def _camera_supervision_losses(
    predicted_pose_encoding: torch.Tensor,
    target_c2w: torch.Tensor,
    target_intrinsics: torch.Tensor,
    translation_weight: float,
    rotation_weight: float,
    focal_weight: float,
    pose_loss_mode: str = "adjacent_pair",
    translation_direction_weight: float = 0.1,
    translation_direction_min_baseline_ratio: float = 0.0,
    translation_direction_baseline_weighting: bool = False,
    global_trajectory_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Supervise the camera encoding directly or in a relative frame.

    Keep this path independent from the differentiable rendering camera. The
    overfit experiment supervises the predicted camera parameters directly;
    routing this loss through the full camera/intrinsics construction makes an
    unstable early camera prediction produce non-finite gradients.
    """
    predicted_pose_encoding = predicted_pose_encoding.float()
    target_c2w = target_c2w.to(
        device=predicted_pose_encoding.device,
        dtype=predicted_pose_encoding.dtype,
    )
    if predicted_pose_encoding.shape[:2] != target_c2w.shape[:2]:
        raise RuntimeError(
            "Predicted camera encoding and GT camera sequence have different "
            f"view counts: predicted={tuple(predicted_pose_encoding.shape)}, "
            f"gt={tuple(target_c2w.shape)}."
        )
    if predicted_pose_encoding.shape[:2] != target_intrinsics.shape[:2]:
        raise RuntimeError(
            "Predicted camera encoding and GT intrinsics have different "
            f"view counts: predicted={tuple(predicted_pose_encoding.shape)}, "
            f"gt={tuple(target_intrinsics.shape)}."
        )

    target_intrinsics = target_intrinsics.to(
        device=predicted_pose_encoding.device,
        dtype=predicted_pose_encoding.dtype,
    )
    connected_zero = torch.nan_to_num(
        predicted_pose_encoding,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).sum() * 0.0
    raw_quaternion = predicted_pose_encoding[..., 3:7]
    quaternion_finite = torch.isfinite(raw_quaternion).all(dim=-1)
    quaternion_norm = torch.nan_to_num(
        raw_quaternion,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).norm(dim=-1, keepdim=True)
    predicted_valid = (
        torch.isfinite(predicted_pose_encoding).all(dim=-1)
        & quaternion_finite
        & (quaternion_norm[..., 0] > 1e-6)
    )
    target_pose_valid = torch.isfinite(target_c2w).all(dim=(-2, -1))
    target_intrinsic_valid = (
        torch.isfinite(target_intrinsics).all(dim=(-2, -1))
        & (target_intrinsics[..., 0, 0] > 1e-6)
        & (target_intrinsics[..., 1, 1] > 1e-6)
    )
    view_valid = predicted_valid & target_pose_valid & target_intrinsic_valid
    safe_prediction = torch.nan_to_num(
        predicted_pose_encoding,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    identity_pose = torch.eye(
        4,
        device=target_c2w.device,
        dtype=target_c2w.dtype,
    ).view(1, 1, 4, 4).expand_as(target_c2w)
    safe_target_c2w = torch.where(
        target_pose_valid[..., None, None],
        torch.nan_to_num(target_c2w, nan=0.0, posinf=0.0, neginf=0.0),
        identity_pose,
    )
    safe_target_intrinsics = torch.nan_to_num(
        target_intrinsics,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    def masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = mask.to(values.dtype)
        if sample_weight is not None:
            sample_weight = sample_weight.to(
                device=values.device,
                dtype=values.dtype,
            )
            while sample_weight.ndim < weight.ndim:
                sample_weight = sample_weight.unsqueeze(-1)
            weight = weight * sample_weight
        while weight.ndim < values.ndim:
            weight = weight.unsqueeze(-1)
        weight = weight.expand_as(values)
        return (
            (values * weight).sum() / weight.sum().clamp_min(1.0)
            + connected_zero
        )

    def masked_smooth_l1(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return masked_mean(
            F.smooth_l1_loss(prediction, target, reduction="none"),
            mask,
            sample_weight,
        )

    # ZipMap encodes world-to-camera translation and rotation. Convert only
    # these compact predictions to c2w components; this avoids any 4x4 matrix
    # inverse or dependency on the render-camera construction.
    safe_quaternion = safe_prediction[..., 3:7]
    quaternion = safe_quaternion / quaternion_norm.clamp_min(1e-6)
    identity_quaternion = torch.zeros_like(quaternion)
    identity_quaternion[..., 3] = 1.0
    quaternion = torch.where(
        quaternion_norm > 1e-6, quaternion, identity_quaternion
    )
    rotation_w2c = quat_to_mat(quaternion)
    rotation_c2w = rotation_w2c.transpose(-1, -2)
    translation_w2c = safe_prediction[..., :3]
    translation_c2w = -torch.matmul(
        rotation_c2w, translation_w2c.unsqueeze(-1)
    ).squeeze(-1)

    def relative_c2w(
        rotation: torch.Tensor, translation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference_rotation = rotation[:, :1]
        reference_translation = translation[:, :1]
        reference_rotation_t = reference_rotation.transpose(-1, -2)
        relative_rotation = reference_rotation_t @ rotation
        relative_translation = (
            reference_rotation_t
            @ (translation - reference_translation).unsqueeze(-1)
        ).squeeze(-1)
        return relative_rotation, relative_translation

    predicted_relative_rotation, predicted_relative_translation = relative_c2w(
        rotation_c2w, translation_c2w
    )
    target_relative_rotation, target_relative_translation = relative_c2w(
        safe_target_c2w[..., :3, :3], safe_target_c2w[..., :3, 3]
    )
    relative_valid = view_valid[:, :1] & view_valid
    if (
        not math.isfinite(float(global_trajectory_weight))
        or float(global_trajectory_weight) < 0.0
    ):
        raise ValueError(
            "camera_global_trajectory_weight must be finite and non-negative, "
            f"got {global_trajectory_weight}."
        )
    pose_loss_mode = str(pose_loss_mode).lower()
    pair_weight = torch.ones_like(view_valid, dtype=predicted_pose_encoding.dtype)
    if pose_loss_mode == "absolute":
        target_rotation_w2c = safe_target_c2w[..., :3, :3].transpose(-1, -2)
        target_translation_w2c = -torch.matmul(
            target_rotation_w2c,
            safe_target_c2w[..., :3, 3].unsqueeze(-1),
        ).squeeze(-1)
        predicted_for_rotation = rotation_w2c
        predicted_for_translation = translation_w2c
        target_for_rotation = target_rotation_w2c
        target_for_translation = target_translation_w2c
        loss_valid = view_valid
    elif pose_loss_mode in {"adjacent_pair", "all_pairs"}:
        if predicted_relative_rotation.shape[1] < 2:
            raise ValueError(
                f"camera_pose_loss_mode={pose_loss_mode} requires at least two "
                "camera views."
            )

        if pose_loss_mode == "adjacent_pair":
            left_indices = torch.arange(
                predicted_relative_rotation.shape[1] - 1,
                device=predicted_relative_rotation.device,
            )
            right_indices = left_indices + 1
        else:
            left_indices, right_indices = torch.triu_indices(
                predicted_relative_rotation.shape[1],
                predicted_relative_rotation.shape[1],
                offset=1,
                device=predicted_relative_rotation.device,
            )

        def indexed_pair_c2w(
            rotation: torch.Tensor,
            translation: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            left_rotation_t = rotation[:, left_indices].transpose(-1, -2)
            pair_rotation = left_rotation_t @ rotation[:, right_indices]
            pair_translation = (
                left_rotation_t
                @ (
                    translation[:, right_indices]
                    - translation[:, left_indices]
                ).unsqueeze(-1)
            ).squeeze(-1)
            return pair_rotation, pair_translation

        predicted_for_rotation, predicted_for_translation = indexed_pair_c2w(
            predicted_relative_rotation, predicted_relative_translation
        )
        target_for_rotation, target_for_translation = indexed_pair_c2w(
            target_relative_rotation, target_relative_translation
        )
        loss_valid = (
            view_valid[:, left_indices] & view_valid[:, right_indices]
        )
        temporal_gap = (right_indices - left_indices).to(
            predicted_for_translation.dtype
        )
        pair_weight = temporal_gap.unsqueeze(0).expand_as(loss_valid)
    elif pose_loss_mode == "first_relative":
        predicted_for_rotation = predicted_relative_rotation
        predicted_for_translation = predicted_relative_translation
        target_for_rotation = target_relative_rotation
        target_for_translation = target_relative_translation
        loss_valid = relative_valid
    else:
        raise ValueError(
            "camera_pose_loss_mode must be absolute, first_relative, "
            "adjacent_pair, or all_pairs, "
            f"got {pose_loss_mode!r}."
        )

    if pose_loss_mode in {"adjacent_pair", "all_pairs"}:
        if (
            not math.isfinite(float(translation_direction_weight))
            or float(translation_direction_weight) < 0.0
        ):
            raise ValueError(
                "camera_translation_direction_weight must be finite and "
                f"non-negative, got {translation_direction_weight}."
            )
        if (
            not math.isfinite(float(translation_direction_min_baseline_ratio))
            or float(translation_direction_min_baseline_ratio) < 0.0
        ):
            raise ValueError(
                "camera_translation_direction_min_baseline_ratio must be "
                "finite and non-negative, got "
                f"{translation_direction_min_baseline_ratio}."
            )
        translation_mask = (
            loss_valid.to(predicted_for_translation.dtype)
            * pair_weight.to(predicted_for_translation.dtype)
        )[..., None]
        denominator = (
            predicted_for_translation.square() * translation_mask
        ).sum(dim=(1, 2))
        alignment_scale = (
            (
                predicted_for_translation
                * target_for_translation
                * translation_mask
            ).sum(dim=(1, 2))
            / denominator.clamp_min(1e-8)
        )
        alignment_scale = torch.where(
            denominator > 1e-8,
            alignment_scale.clamp(1e-3, 1e3),
            torch.ones_like(alignment_scale),
        ).detach()
        translation_shape = masked_smooth_l1(
            predicted_for_translation * alignment_scale[:, None, None],
            target_for_translation,
            loss_valid,
            pair_weight,
        )
        predicted_norm = predicted_for_translation.norm(dim=-1)
        target_norm = target_for_translation.norm(dim=-1)
        valid_pair_weight = (
            pair_weight.to(target_norm.dtype) * loss_valid.to(target_norm.dtype)
        )
        valid_pair_count = valid_pair_weight.sum(dim=-1).clamp_min(1e-8)
        reference_baseline = (
            (target_norm * valid_pair_weight).sum(dim=-1)
            / valid_pair_count
        ).clamp_min(1e-6)
        minimum_baseline = (
            float(translation_direction_min_baseline_ratio)
            * reference_baseline
        )
        aligned_predicted_norm = (
            predicted_norm * alignment_scale.abs()[:, None]
        )
        valid_direction = (
            loss_valid
            & (predicted_norm > 1e-6)
            & (target_norm > 1e-6)
            & (target_norm >= minimum_baseline[:, None])
            & (aligned_predicted_norm >= minimum_baseline[:, None])
        )
        direction_error = 1.0 - F.cosine_similarity(
            predicted_for_translation,
            target_for_translation,
            dim=-1,
            eps=1e-6,
        )
        if translation_direction_baseline_weighting:
            direction_pair_weight = (
                target_norm / reference_baseline[:, None]
            ).clamp(max=1.0)
        else:
            direction_pair_weight = torch.ones_like(target_norm)
        valid_weight = (
            pair_weight.to(direction_pair_weight.dtype)
            * direction_pair_weight
            * valid_direction.to(direction_pair_weight.dtype)
        )
        valid_weight_sum = valid_weight.sum()
        translation_direction = (
            (direction_error * valid_weight).sum()
            / valid_weight_sum.clamp_min(1e-8)
            + connected_zero
        )
        translation = (
            translation_shape
            + float(translation_direction_weight) * translation_direction
        )
    else:
        translation_shape = masked_smooth_l1(
            predicted_for_translation,
            target_for_translation,
            loss_valid,
        )
        translation_direction = connected_zero
        alignment_scale = torch.ones(
            predicted_for_translation.shape[0],
            device=predicted_for_translation.device,
            dtype=predicted_for_translation.dtype,
        )
        valid_direction = loss_valid
        direction_pair_weight = torch.ones_like(
            predicted_for_translation[..., 0]
        )
        target_norm = target_for_translation.norm(dim=-1)
        valid_pair_count = loss_valid.sum(dim=-1).clamp_min(1).to(target_norm.dtype)
        reference_baseline = (
            target_norm * loss_valid.to(target_norm.dtype)
        ).sum(dim=-1) / valid_pair_count
        translation = translation_shape
    rotation = masked_mean(
        (predicted_for_rotation - target_for_rotation).square(),
        loss_valid,
        pair_weight,
    )

    if pose_loss_mode == "absolute":
        # Absolute W2C supervision already fixes the complete world gauge.
        # Adding a separately scale-aligned relative term would weaken the
        # interpretation of this ablation and duplicate the same trajectory.
        global_alignment_scale = torch.ones(
            predicted_for_translation.shape[0],
            device=predicted_for_translation.device,
            dtype=predicted_for_translation.dtype,
        )
        global_translation_shape = connected_zero
        global_rotation = connected_zero
        global_trajectory = connected_zero
    else:
        # Optional first-relative trajectory supervision. This is useful with
        # local pair modes and can be disabled when all-pairs already supplies
        # the same long-range constraints.
        global_mask = relative_valid[..., None].to(
            predicted_relative_translation.dtype
        )
        global_denominator = (
            predicted_relative_translation.square() * global_mask
        ).sum(dim=(1, 2))
        global_alignment_scale = (
            (
                predicted_relative_translation
                * target_relative_translation
                * global_mask
            ).sum(dim=(1, 2))
            / global_denominator.clamp_min(1e-8)
        )
        global_alignment_scale = torch.where(
            global_denominator > 1e-8,
            global_alignment_scale.clamp(1e-3, 1e3),
            torch.ones_like(global_alignment_scale),
        ).detach()
        global_translation_shape = masked_smooth_l1(
            predicted_relative_translation
            * global_alignment_scale[:, None, None],
            target_relative_translation,
            relative_valid,
        )
        global_rotation = masked_mean(
            (predicted_relative_rotation - target_relative_rotation).square(),
            relative_valid,
        )
        global_trajectory = (
            translation_weight * global_translation_shape
            + rotation_weight * global_rotation
        )
    gt_fov_h = 2.0 * torch.atan(
        0.5 / safe_target_intrinsics[..., 1, 1].clamp_min(1e-6)
    )
    gt_fov_w = 2.0 * torch.atan(
        0.5 / safe_target_intrinsics[..., 0, 0].clamp_min(1e-6)
    )
    gt_fov = torch.stack([gt_fov_h, gt_fov_w], dim=-1)
    # Match offline_gs_mlp_overfit: supervise the raw FOV encoding. The
    # bounded copy is only for rendering and must not saturate this loss.
    focal = masked_smooth_l1(
        safe_prediction[..., 7:9],
        gt_fov.float(),
        view_valid,
    )
    total = (
        translation_weight * translation
        + rotation_weight * rotation
        + focal_weight * focal
        + float(global_trajectory_weight) * global_trajectory
    )
    return {
        "total": total,
        "translation": translation,
        "translation_shape": translation_shape,
        "translation_direction": translation_direction,
        "translation_alignment_scale": alignment_scale.mean() + connected_zero,
        "translation_direction_valid_ratio": (
            valid_direction.float().sum()
            / loss_valid.float().sum().clamp_min(1.0)
            + connected_zero
        ),
        "translation_direction_pair_weight_mean": masked_mean(
            direction_pair_weight,
            loss_valid,
        ),
        "temporal_gap_weight_mean": masked_mean(pair_weight, loss_valid),
        "relative_pair_count": loss_valid.float().sum() + connected_zero,
        "translation_reference_baseline": reference_baseline.mean(),
        "rotation": rotation,
        "focal": focal,
        "global_trajectory": global_trajectory,
        "global_translation_shape": global_translation_shape,
        "global_rotation": global_rotation,
        "global_translation_alignment_scale": (
            global_alignment_scale.mean() + connected_zero
        ),
        "invalid_ratio": (~view_valid).float().mean(),
        "valid_ratio": view_valid.float().mean(),
        "valid_view_count": view_valid.float().sum(),
    }


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model: nn.Module
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        model: nn.Module,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder_visualizer = None
        self.model = model
        self._configure_training_stage()
        self.data_shim = get_data_shim(self.model.encoder)
        self.losses = nn.ModuleList(losses)
        for loss in self.losses:
            if loss.name == "depth" and loss.cfg.teacher != self.model.encoder.cfg.depth_teacher:
                raise ValueError("Encoder depth_teacher must match loss.depth.teacher.")

        self.benchmarker = Benchmarker()
        self._skip_optimizer_step = False
        self._bad_grad_name = None
        self._bad_grad_summaries = []
        self._pending_train_debug_record: dict[str, Any] | None = None
        self._val_comparison_images: list[np.ndarray] = []
        self._val_comparison_captions: list[str] = []

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self is not None and hasattr(self.model, "encoder"):
            from src.misc.backbone_checkpoint import backbone_signature

            checkpoint["reconstruction_backbone"] = backbone_signature(self.model.encoder.cfg)
        # Fixed loss networks are initialized from their own pretrained weights.
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict):
            if key.startswith("losses."):
                del state_dict[key]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if hasattr(self, "model") and hasattr(self.model, "encoder"):
            from src.misc.backbone_checkpoint import validate_backbone_checkpoint

            validate_backbone_checkpoint(checkpoint, self.model.encoder.cfg)
        # Fill omitted loss weights for strict Lightning resume. Preserve any
        # loss weights stored by older checkpoints and keep model checks strict.
        state_dict = checkpoint["state_dict"]
        for key, value in self.losses.state_dict(prefix="losses.").items():
            state_dict.setdefault(key, value)

    def _configure_training_stage(self) -> None:
        train_base_heads = bool(self.optimizer_cfg.train_base_heads)
        train_gir = bool(self.optimizer_cfg.train_gir)
        granular_base_training = any(
            value is not None
            for value in (
                self.optimizer_cfg.train_gaussian_param_head,
                self.optimizer_cfg.train_gs_head,
                self.optimizer_cfg.train_camera_head,
                self.optimizer_cfg.train_depth_head,
            )
        )
        train_gaussian_param_head = (
            train_base_heads and not granular_base_training
            if self.optimizer_cfg.train_gaussian_param_head is None
            else bool(self.optimizer_cfg.train_gaussian_param_head)
        )
        train_gs_head = (
            train_base_heads
            if self.optimizer_cfg.train_gs_head is None
            else bool(self.optimizer_cfg.train_gs_head)
        )
        train_camera_head = (
            train_base_heads
            if self.optimizer_cfg.train_camera_head is None
            else bool(self.optimizer_cfg.train_camera_head)
        )
        train_depth_head = (
            train_base_heads
            if self.optimizer_cfg.train_depth_head is None
            else bool(self.optimizer_cfg.train_depth_head)
        )

        if not train_base_heads or granular_base_training:
            # Stage 2 starts from a fixed stage-1 model and only learns GIR.
            self.model.requires_grad_(False)

        for module_name in ("gaussian_param_head", "gs_head", "depth_head"):
            module = getattr(self.model.encoder, module_name, None)
            if module is None:
                raise RuntimeError(
                    f"Base training module model.encoder.{module_name} is missing."
                )
            if not granular_base_training:
                module.requires_grad_(train_base_heads)

        if granular_base_training:
            self.model.encoder.gaussian_param_head.requires_grad_(
                train_gaussian_param_head
            )
            self.model.encoder.gs_head.requires_grad_(train_gs_head)

        shared_new_decoder_enabled = bool(
            getattr(
                self.model.encoder.cfg,
                "gir_new_gs_shared_decoder_enabled",
                False,
            )
        )
        if shared_new_decoder_enabled:
            if train_gaussian_param_head:
                raise ValueError(
                    "optimizer.train_gaussian_param_head=true requires the "
                    "original gs_head path; disable "
                    "gir_new_gs_shared_decoder_enabled."
                )
            self.model.encoder.gs_head.requires_grad_(False)

        wm_depth_output = getattr(
            getattr(self.model.encoder.gaussian_param_head, "scratch", None),
            "output_conv2",
            None,
        )
        depth_head_source = str(
            getattr(self.model.encoder.cfg, "depth_head_source", "native")
        ).lower()
        if depth_head_source not in {"native", "world_mirror"}:
            raise ValueError(
                "model.encoder.depth_head_source must be native or "
                f"world_mirror, got {depth_head_source!r}."
            )
        self.model.encoder.depth_head.requires_grad_(
            train_depth_head and depth_head_source == "native"
        )
        if wm_depth_output is None and depth_head_source == "world_mirror":
            raise RuntimeError("WorldMirror depth output head is missing.")
        if wm_depth_output is not None:
            wm_depth_output.requires_grad_(
                train_depth_head and depth_head_source == "world_mirror"
            )

        camera_head = getattr(self.model.encoder, "camera_head", None)
        reconstruction_backbone = getattr(
            self.model.encoder.cfg, "reconstruction_backbone", "zipmap"
        )
        if reconstruction_backbone in {"zipmap", "streamvggt"}:
            if camera_head is None:
                raise RuntimeError(f"{reconstruction_backbone} training requires encoder.camera_head.")
            camera_head.requires_grad_(train_camera_head)
            # StreamVGGT detaches each refinement prediction before feeding it
            # to the next iteration. The empty token is used only by the first
            # iteration, so a loss on the final prediction cannot reach it.
            # Keep it fixed instead of registering a permanently unused DDP
            # parameter.
            empty_pose_tokens = getattr(camera_head, "empty_pose_tokens", None)
            if empty_pose_tokens is not None:
                empty_pose_tokens.requires_grad_(False)
            if train_camera_head and self.train_cfg.weight_pose != 0 and not any(
                parameter.requires_grad for parameter in camera_head.parameters()
            ):
                raise RuntimeError(
                    "GT camera loss is enabled, but the reconstruction camera head "
                    "has no trainable parameters."
                )
        elif reconstruction_backbone == "abot":
            # ABot's pose and local-point heads are part of its 1B frozen
            # network, while these adapters intentionally own no parameters.
            camera_head.requires_grad_(False)
            if depth_head_source == "native":
                self.model.encoder.depth_head.requires_grad_(False)
            print(
                "ABot native camera and depth remain frozen; base training "
                "updates the WorldMirror GS heads only."
            )

        gir_head = getattr(self.model, "gir_update_head", None)
        projected_gir_head = getattr(
            self.model, "gir_projected_update_head", None
        )
        symmetric_gir_head = getattr(
            self.model, "gir_symmetric_update_head", None
        )
        gir_enabled = bool(getattr(self.model.encoder.cfg, "gir_enabled", False))
        if train_gir and (
            not gir_enabled
            or (
                gir_head is None
                and projected_gir_head is None
                and symmetric_gir_head is None
            )
        ):
            raise ValueError(
                "optimizer.train_gir=true requires model.encoder.gir_enabled=true."
            )
        old_update_mode = str(
            getattr(
                self.model.encoder.cfg,
                "gir_old_update_mode",
                "pixel_topk",
            )
        ).lower()
        projected_update_enabled = bool(
            getattr(self.model.encoder.cfg, "gir_old_residual_enabled", True)
        ) and old_update_mode == "projected_shared"
        new_residual_enabled = bool(
            getattr(
                self.model.encoder.cfg,
                "gir_new_residual_enabled",
                False,
            )
        )
        old_opacity_gradient_enabled = bool(
            getattr(
                self.model.encoder.cfg,
                "gir_old_opacity_gradient_enabled",
                False,
            )
        )
        new_opacity_gradient_enabled = bool(
            getattr(
                self.model.encoder.cfg,
                "gir_new_opacity_gradient_enabled",
                False,
            )
        )
        mean_gradient_enabled = bool(
            getattr(
                self.model.encoder.cfg,
                "gir_mean_gradient_enabled",
                False,
            )
        )
        new_mean_gradient_enabled = bool(
            getattr(
                self.model.encoder.cfg,
                "gir_new_mean_gradient_enabled",
                False,
            )
        )
        symmetric_update_enabled = bool(
            getattr(
                self.model.encoder.cfg,
                "gir_symmetric_multiview_update_enabled",
                False,
            )
        )
        if train_gir and symmetric_update_enabled and symmetric_gir_head is None:
            raise RuntimeError(
                "Symmetric multiview GS updates require "
                "model.gir_symmetric_update_head."
            )
        if symmetric_gir_head is not None:
            symmetric_gir_head.requires_grad_(
                train_gir and symmetric_update_enabled
            )
        if train_gir and symmetric_update_enabled:
            visual_enabled = bool(
                getattr(
                    self.model.encoder.cfg,
                    "gir_symmetric_multiview_visual_enabled",
                    False,
                )
            )
            unbounded_updates = bool(
                getattr(
                    self.model.encoder.cfg,
                    "gir_symmetric_multiview_unbounded_updates",
                    False,
                )
            )
            if unbounded_updates:
                # Direct residual mode intentionally bypasses learned gates.
                # Keep that decoder out of the optimizer/DDP parameter set.
                symmetric_gir_head.attribute_decoders["gates"].requires_grad_(
                    False
                )
            print(
                "GIR symmetric multiview update: append old/new GS first, "
                "aggregate equal-role gradients from every causal view, "
                f"visual_context={visual_enabled}, then apply one shared "
                f"{'unbounded' if unbounded_updates else 'bounded'} residual head"
            )
        projected_head_needed = projected_update_enabled or new_residual_enabled
        if train_gir and projected_head_needed and projected_gir_head is None:
            raise RuntimeError(
                "Projected old/new GS residuals require "
                "model.gir_projected_update_head."
            )
        if projected_gir_head is not None:
            projected_gir_head.requires_grad_(False)
            projected_gir_head.decoder.requires_grad_(
                train_gir and projected_head_needed
            )
            projected_gir_head.opacity_gradient_adapter.requires_grad_(
                train_gir
                and (
                    (projected_update_enabled and old_opacity_gradient_enabled)
                    or (new_residual_enabled and new_opacity_gradient_enabled)
                )
            )
            projected_gir_head.mean_gradient_adapter.requires_grad_(
                train_gir and projected_update_enabled and mean_gradient_enabled
            )
            projected_gir_head.new_mean_gradient_adapter.requires_grad_(
                train_gir and new_residual_enabled and new_mean_gradient_enabled
            )
            projected_gir_head.output.requires_grad_(
                train_gir and projected_update_enabled
            )
            projected_gir_head.new_output.requires_grad_(
                train_gir and new_residual_enabled
            )
        if gir_head is not None:
            old_delete_enabled = bool(
                getattr(
                    self.model.encoder.cfg,
                    "gir_old_delete_enabled",
                    False,
                )
            )
            old_decay_enabled = bool(
                getattr(
                    self.model.encoder.cfg,
                    "gir_old_decay_enabled",
                    False,
                )
            )
            add_gate_enabled = bool(
                getattr(self.model.encoder.cfg, "gir_add_gate_enabled", True)
            )
            pixel_residual_enabled = bool(
                getattr(self.model.encoder.cfg, "gir_old_residual_enabled", True)
            ) and old_update_mode == "pixel_topk"
            pixel_head_needed = (
                pixel_residual_enabled
                or old_delete_enabled
                or old_decay_enabled
                or add_gate_enabled
            )
            gir_head.requires_grad_(train_gir and pixel_head_needed)

            if train_gir and pixel_head_needed:
                # Keep the optimizer/DDP parameter set consistent with the
                # streaming branches that are actually enabled.  The GIR
                # head contains several independent output heads, but the
                # current experiment trains only historical residuals.
                current_prediction = getattr(
                    gir_head, "current_prediction", None
                )
                if current_prediction is not None:
                    # New anchors now use the projected shared decoder. Keep
                    # the legacy dense current-pixel head out of the optimizer.
                    current_prediction.requires_grad_(False)

                delete_prediction = getattr(gir_head, "delete_prediction", None)
                if delete_prediction is not None and not old_delete_enabled:
                    # The shared delete head is used only by hard deletion.
                    # Soft decay has its own contributor-conditioned head.
                    delete_prediction.requires_grad_(False)

                contributor_decay_prediction = getattr(
                    gir_head, "contributor_decay_prediction", None
                )
                if (
                    contributor_decay_prediction is not None
                    and not old_decay_enabled
                ):
                    contributor_decay_prediction.requires_grad_(False)

                if not new_residual_enabled or (
                    not old_delete_enabled and not old_decay_enabled
                ):
                    print(
                        "GIR branch training: "
                        f"historical_residual={bool(getattr(self.model.encoder.cfg, 'gir_old_residual_enabled', True))}, "
                        f"new_residual={new_residual_enabled}, "
                        f"delete={old_delete_enabled or old_decay_enabled}; "
                        "disabled GIR heads are frozen"
                    )

        if train_gir and projected_update_enabled:
            print(
                "GIR old-map update: projected_shared residuals for "
                "mean/rotation/scale/opacity/SH; front errors decay opacity "
                "only; pixel rank heads and top-k decay are disabled"
            )
        if train_gir and new_residual_enabled:
            shared_base_opacity = float(
                getattr(
                    self.model.encoder.cfg,
                    "gir_new_gs_shared_decoder_base_opacity",
                    0.5,
                )
            )
            shared_scale_multiplier = float(
                getattr(
                    self.model.encoder.cfg,
                    "gir_new_gs_shared_decoder_scale_multiplier",
                    1.0,
                )
            )
            print(
                "GIR new-map update: "
                + (
                    "analytic anchors from every first-frame pixel and "
                    "append-selected later pixels use the shared projected "
                    "decoder; dense gs_head is frozen; base_opacity="
                    f"{shared_base_opacity:g}, "
                    "scale_multiplier="
                    f"{shared_scale_multiplier:g}"
                    if shared_new_decoder_enabled
                    else "append-selected DPT anchors use the shared projected "
                    "decoder with an independent new-GS head"
                )
            )
        append_mode = str(
            getattr(self.model.encoder.cfg, "gir_append_mode", "all")
        ).lower()
        tracker_backend = str(
            getattr(self.model.encoder.cfg, "gir_tracker_backend", "spot")
        ).lower()
        print(
            "GIR new-GS append mode: "
            f"{append_mode}"
            + (
                " (depth-consistent global point-map coverage)"
                if append_mode == "point_map"
                else (
                    f" (rolling previous-frame {tracker_backend} coverage only)"
                    if append_mode == "tracker"
                    else (
                        f" (union of rolling {tracker_backend} and "
                        "depth-consistent global point-map coverage)"
                        if append_mode == "tracker_point_map"
                        else ""
                    )
                )
            )
        )
        if append_mode in {"point_map", "tracker_point_map"}:
            print(
                "GIR point-map follows residual-updated means: "
                f"{bool(getattr(self.model.encoder.cfg, 'gir_point_map_follow_mean_updates_enabled', False))}"
            )
        if bool(
            getattr(self.model.encoder.cfg, "gir_anchor_child_enabled", False)
        ):
            print(
                "GIR point-map anchor children: persistent, non-recursive; "
                f"max_count={int(getattr(self.model.encoder.cfg, 'gir_anchor_child_max_count', 8))}, "
                f"add_count={int(getattr(self.model.encoder.cfg, 'gir_anchor_child_add_count', 1))}, "
                f"max_ratio={float(getattr(self.model.encoder.cfg, 'gir_anchor_child_max_ratio', 0.05)):g}, "
                f"max_gaussians={int(getattr(self.model.encoder.cfg, 'gir_anchor_child_max_gaussians', 500000))}"
            )
        if append_mode in {"tracker", "tracker_point_map"}:
            tracker_fb_enabled = bool(
                getattr(
                    self.model.encoder.cfg,
                    "gir_tracker_forward_backward_enabled",
                    False,
                )
            )
            tracker_cycle_absolute = float(
                getattr(
                    self.model.encoder.cfg,
                    "gir_tracker_cycle_absolute_threshold",
                    1.5,
                )
            )
            tracker_cycle_relative = float(
                getattr(
                    self.model.encoder.cfg,
                    "gir_tracker_cycle_relative_threshold",
                    0.05,
                )
            )
            print(
                f"{tracker_backend} forward-backward consistency: enabled="
                f"{tracker_fb_enabled}, "
                "cycle_threshold="
                f"{tracker_cycle_absolute:g}+{tracker_cycle_relative:g}"
                "*motion"
            )

        if not any(
            (
                train_gaussian_param_head,
                train_gs_head,
                train_camera_head,
                train_depth_head,
                train_gir,
            )
        ):
            raise ValueError(
                "The training stage disables both base GS heads and GIR; "
                "there would be no selected trainable model branch."
            )

        print(
            "Training stage: "
            f"gir_enabled={gir_enabled}, "
            f"train_gaussian_param_head={train_gaussian_param_head}, "
            f"train_gs_head={train_gs_head}, "
            f"train_camera_head={reconstruction_backbone in {'zipmap', 'streamvggt'} and train_camera_head}, "
            f"depth_head_source={depth_head_source}, "
            f"train_depth_head={train_depth_head and (depth_head_source == 'world_mirror' or reconstruction_backbone != 'abot')}, "
            f"train_gir={train_gir}"
        )
        training_cameras = resolve_camera_phase_settings(
            self.model.encoder.cfg, True
        )
        evaluation_cameras = resolve_camera_phase_settings(
            self.model.encoder.cfg, False
        )
        print(
            "Final-map geometry/rendering: "
            f"mode={getattr(self.model.encoder.cfg, 'gir_training_mode', 'streaming_aux')}; "
            "training="
            f"sim3:{training_cameras.sim3_alignment_mode},"
            f"geometry:{training_cameras.geometry_camera_source},"
            f"source:{training_cameras.source_render_camera},"
            f"heldout:{training_cameras.heldout_test_camera}; "
            "evaluation="
            f"sim3:{evaluation_cameras.sim3_alignment_mode},"
            f"geometry:{evaluation_cameras.geometry_camera_source},"
            f"source:{evaluation_cameras.source_render_camera},"
            f"heldout:{evaluation_cameras.heldout_test_camera}"
        )
        if train_camera_head:
            print(
                "Camera supervision: "
                f"mode={self.train_cfg.camera_pose_loss_mode}, "
                "direction_weight="
                f"{self.train_cfg.camera_translation_direction_weight:g}, "
                "min_baseline_ratio="
                f"{self.train_cfg.camera_translation_direction_min_baseline_ratio:g}, "
                "baseline_weighting="
                f"{self.train_cfg.camera_translation_direction_baseline_weighting}, "
                "global_trajectory_weight="
                f"{self.train_cfg.camera_global_trajectory_weight:g}, "
                f"grad_clip={self.train_cfg.camera_grad_clip:g}"
            )

        self._enforce_selected_module_modes()

    def _enforce_selected_module_modes(self) -> None:
        """Keep frozen feature extractors deterministic like the offline run."""
        for module_name in ("aggregator",):
            module = getattr(self.model.encoder, module_name, None)
            if module is not None:
                module.eval()
        self.model.decoder.eval()

        for module_name in (
            "gaussian_param_head",
            "gs_head",
            "camera_head",
            "depth_head",
        ):
            module = getattr(self.model.encoder, module_name, None)
            if module is not None:
                module.train(any(parameter.requires_grad for parameter in module.parameters()))
        for module_name in (
            "gir_update_head",
            "gir_projected_update_head",
            "gir_symmetric_update_head",
        ):
            module = getattr(self.model, module_name, None)
            if module is not None:
                module.train(any(parameter.requires_grad for parameter in module.parameters()))

    def on_train_epoch_start(self) -> None:
        self._enforce_selected_module_modes()
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        print(f"Train epoch start on rank {self.trainer.global_rank}")
        if hasattr(self.trainer.datamodule.train_loader.dataset, "set_epoch"):
            self.trainer.datamodule.train_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.train_loader.sampler, "set_epoch"):
            self.trainer.datamodule.train_loader.sampler.set_epoch(self.current_epoch)

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        self._enforce_selected_module_modes()

    def on_validation_epoch_start(self) -> None:
        print(f"Validation epoch start on rank {self.trainer.global_rank}")
        self._val_comparison_images = []
        self._val_comparison_captions = []
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.datamodule.val_loader.dataset, "set_epoch"):
            self.trainer.datamodule.val_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.val_loader.sampler, "set_epoch"):
            self.trainer.datamodule.val_loader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        if self.global_rank != 0 or len(self._val_comparison_images) == 0:
            return

        self.logger.log_image(
            "comparison",
            self._val_comparison_images,
            step=self.global_step,
            caption=self._val_comparison_captions,
        )

    @staticmethod
    def _debug_scalar(value: Any) -> float | int | str | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            scalar = float(value)
        elif torch.is_tensor(value) and value.numel() == 1:
            scalar = float(value.detach().float().item())
        else:
            return None
        if math.isnan(scalar):
            return "nan"
        if math.isinf(scalar):
            return "+inf" if scalar > 0.0 else "-inf"
        return scalar

    @staticmethod
    def _debug_gradient_norm(parameters: list[torch.Tensor]) -> float:
        squared_norm = None
        for parameter in parameters:
            if parameter.grad is None:
                continue
            contribution = parameter.grad.detach().float().square().sum()
            squared_norm = (
                contribution
                if squared_norm is None
                else squared_norm + contribution
            )
        if squared_norm is None:
            return 0.0
        return float(squared_norm.sqrt().item())

    def _write_train_debug_record(self, record: dict[str, Any]) -> None:
        if self.global_rank != 0:
            return
        path = self.train_cfg.output_path / self.train_cfg.debug_metrics_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.train_cfg.debug_metrics_print:
            print(
                "[train debug] "
                f"step={record.get('step')} "
                f"loss={record.get('loss_total')} "
                f"psnr={record.get('train_psnr')} "
                f"camera_loss={record.get('loss_camera')} "
                f"camera_invalid={record.get('camera_invalid_ratio')} "
                f"camera_grad={record.get('grad_camera_norm')} "
                f"sim3={record.get('sim3_scale')} "
                f"gs={record.get('gaussian_count')} "
                "symmetric_active="
                f"{record.get('gir_symmetric_multiview_active_count')} "
                f"status={record.get('status', 'ok')}"
            )

    def on_fit_start(self) -> None:
        if not self.train_cfg.debug_metrics_enabled:
            return
        if self.train_cfg.debug_metrics_every_n_steps <= 0:
            raise ValueError(
                "train.debug_metrics_every_n_steps must be positive when "
                "debug metrics are enabled."
            )
        if not self.train_cfg.debug_metrics_filename.strip():
            raise ValueError("train.debug_metrics_filename must not be empty.")
        if self.global_rank == 0:
            path = (
                self.train_cfg.output_path
                / self.train_cfg.debug_metrics_filename
            )
            print(f"Training debug metrics: enabled; path={path}")

    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        # if self.global_rank == 0:
        #     print(
        #         f"context = {batch['context']['index'].tolist()}"
        #     )
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat(
                                    [batch_combined[k][kk], batch_per_dl[k][kk]], dim=0
                                )
                        else:
                            raise NotImplementedError
            batch = batch_combined

        batch: BatchedExample = self.data_shim(batch)
        b, v, c, h, w = batch["context"]["image"].shape
        compact_logging = bool(self.train_cfg.compact_logging)
        self._pending_train_debug_record = None
        debug_due = (
            self.global_rank == 0
            and self.train_cfg.debug_metrics_enabled
            and self.train_cfg.debug_metrics_every_n_steps > 0
            and self.global_step
            % self.train_cfg.debug_metrics_every_n_steps
            == 0
        )
        if debug_due:
            self._pending_train_debug_record = {
                "step": int(self.global_step),
                "batch_idx": int(batch_idx),
                "batch_size": int(b),
                "view_count": int(v),
                "height": int(h),
                "width": int(w),
                "scenes": [str(scene) for scene in batch.get("scene", [])],
                "context_indices": batch["context"]["index"]
                .detach()
                .cpu()
                .tolist(),
            }
        context_image = (batch["context"]["image"] + 1) / 2
        reset_cuda_peak_memory()
        log_cuda_memory(
            "training_step_start",
            step=int(self.global_step),
            batch=b,
            views=v,
            height=h,
            width=w,
            scene=batch.get("scene"),
        )
        if (
            self.global_rank == 0
            and self.train_cfg.print_log_every_n_steps > 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"training step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        depth_loss_idx = list(get_cfg()["loss"].keys()).index("depth")
        depth_loss_module = self.losses[depth_loss_idx]
        depth_loss_weight = (
            self.train_cfg.cxt_depth_weight * self.train_cfg.weight_depth
        )
        # Training uses the first half of a multi-view batch as source views.
        # Run the frozen DAV3 teacher before AnySplat builds its large GIR graph,
        # otherwise the two activation peaks overlap on high-view-count batches.
        source_view_count = 1 if v == 1 else v // 2
        with torch.amp.autocast("cuda", enabled=False):
            prepared_context_depth_target = (
                depth_loss_module.prepare_context_depth_target(
                    batch,
                    source_view_count,
                    cxt_depth_weight=depth_loss_weight,
                )
            )
        log_cuda_memory(
            "after_depth_target",
            step=int(self.global_step),
            source_views=source_view_count,
        )

        # Run the model
#         if self.global_step < 50:
#             loss_align = self.model.encoder(context_image, batch["context"]["index"], self.global_step)
#             total_loss = 0
#             total_loss = total_loss + torch.nan_to_num(
#                     loss_align, nan=0.0, posinf=0.0, neginf=0.0
#                 ) * 1e-2
#             print(f'total_loss:{total_loss}')
#             self.log("loss/total", total_loss.item())
#             self.log("info/global_step", self.global_step)
#             # print(f"total_loss: {total_loss}")
#             # print(f"scene = {[x[:20] for x in batch['scene']]}; " )

#             # Tell the data loader processes about the current step.
#             if self.step_tracker is not None:
#                 self.step_tracker.set_step(self.global_step)

#             del batch
#             if self.global_step % 50 == 0:
#                 gc.collect()
#                 torch.cuda.empty_cache()

#             return total_loss
        encoder_output, output = self.model(
            context_image,
            batch["context"]["index"],
            self.global_step,
            gt_context_c2w=batch["context"]["extrinsics"],
            gt_context_intrinsics=batch["context"]["intrinsics"],
        )
        log_cuda_memory(
            "after_model_forward",
            step=int(self.global_step),
            map_gs=int(encoder_output.gaussians.means.shape[1]),
        )
        gaussians, pred_pose_enc_list, depth_dict = (
            encoder_output.gaussians,
            encoder_output.pred_pose_enc_list,
            encoder_output.depth_dict,
        )
        debug_record = self._pending_train_debug_record
        if debug_record is not None:
            debug_record["gaussian_count"] = int(gaussians.means.shape[1])
            for metric_name, metric_value in (encoder_output.infos or {}).items():
                scalar = self._debug_scalar(metric_value)
                if scalar is not None:
                    debug_record[metric_name] = scalar
        
        distill_infos = encoder_output.distill_infos
        if encoder_output.infos is not None and "sim3_scale" in encoder_output.infos:
            self.log(
                "train/sim3_scale",
                encoder_output.infos["sim3_scale"].float().mean(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            self.log(
                "train/sim3_fallback_ratio",
                encoder_output.infos["sim3_fallback_ratio"].float(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        if encoder_output.infos is not None and "gir_add_gate" in encoder_output.infos:
            for metric_name in _gir_training_metric_names(
                self.model.encoder.cfg,
                compact=compact_logging,
            ):
                if metric_name in encoder_output.infos:
                    self.log(
                        f"train/{metric_name}",
                        encoder_output.infos[metric_name].float(),
                    )
            if (
                not compact_logging
                and "gir_history_before_error" in encoder_output.infos
            ):
                history_before = encoder_output.infos[
                    "gir_history_before_error"
                ].float()
                history_after = encoder_output.infos[
                    "gir_history_after_error"
                ].float()
                self.log("train/gir_history_before_error", history_before)
                self.log("train/gir_history_after_error", history_after)
                self.log(
                    "train/gir_history_improvement",
                    history_before - history_after,
                )
            if (
                not compact_logging
                and "gir_history_past_before_error" in encoder_output.infos
            ):
                self.log(
                    "train/gir_history_past_before_error",
                    encoder_output.infos["gir_history_past_before_error"].float(),
                )
                self.log(
                    "train/gir_history_past_after_error",
                    encoder_output.infos["gir_history_past_after_error"].float(),
                )
                self.log(
                    "train/gir_history_past_degradation",
                    encoder_output.infos["gir_history_past_degradation"].float(),
                )
        
        target_gt = (batch["context"]["image"] + 1) / 2
        num_context_views = target_gt.shape[1]

        using_index = torch.arange(num_context_views, device=gaussians.means.device)
        batch["using_index"] = using_index

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean().item())
        if debug_record is not None:
            debug_record["train_psnr"] = float(
                psnr_probabilistic.mean().detach().item()
            )

        total_loss = 0

        if encoder_output.infos is not None and "gir_aux_loss" in encoder_output.infos:
            gir_aux_loss = encoder_output.infos["gir_aux_loss"]
            gir_aux_weight = float(self.model.encoder.cfg.gir_aux_loss_weight)
            if gir_aux_weight != 0.0 and not compact_logging:
                self.log("loss/gir_aux", gir_aux_loss.item())
            gir_rgb_loss_weights = {
                "gir_base_rgb_loss": float(
                    self.model.encoder.cfg.gir_base_loss_weight
                ),
                "gir_current_rgb_loss": float(
                    self.model.encoder.cfg.gir_current_loss_weight
                ),
                "gir_new_hole_rgb_loss": float(
                    self.model.encoder.cfg.gir_new_hole_loss_weight
                ),
                "gir_replay_rgb_loss": float(
                    self.model.encoder.cfg.gir_replay_loss_weight
                ),
            }
            for metric_name, component_weight in gir_rgb_loss_weights.items():
                if (
                    gir_aux_weight != 0.0
                    and component_weight != 0.0
                    and metric_name in encoder_output.infos
                    and not compact_logging
                ):
                    self.log(
                        f"loss/{metric_name}",
                        encoder_output.infos[metric_name].item(),
                    )
            if gir_aux_weight != 0.0:
                total_loss = total_loss + gir_aux_weight * gir_aux_loss
        if (
            encoder_output.infos is not None
            and "gir_old_delete_budget_loss" in encoder_output.infos
        ):
            old_delete_budget_loss = encoder_output.infos[
                "gir_old_delete_budget_loss"
            ]
            old_delete_budget_weight = float(
                self.model.encoder.cfg.gir_old_delete_budget_weight
            )
            if old_delete_budget_weight != 0.0:
                if not compact_logging:
                    self.log(
                        "loss/gir_old_delete_budget",
                        old_delete_budget_loss.item(),
                    )
                total_loss = (
                    total_loss
                    + old_delete_budget_weight * old_delete_budget_loss
                )
        if (
            encoder_output.infos is not None
            and "gir_old_decay_budget_loss" in encoder_output.infos
        ):
            old_decay_budget_loss = encoder_output.infos[
                "gir_old_decay_budget_loss"
            ]
            old_decay_budget_weight = float(
                self.model.encoder.cfg.gir_old_delete_budget_weight
            )
            if old_decay_budget_weight != 0.0:
                if not compact_logging:
                    self.log(
                        "loss/gir_old_decay_budget",
                        old_decay_budget_loss.item(),
                    )
                total_loss = total_loss + (
                    old_decay_budget_weight * old_decay_budget_loss
                )
        if (
            encoder_output.infos is not None
            and "gir_history_adapt_loss" in encoder_output.infos
        ):
            gir_history_adapt = encoder_output.infos["gir_history_adapt_loss"]
            adapt_weight = float(
                self.model.encoder.cfg.gir_history_adapt_weight
            )
            if adapt_weight != 0.0:
                if not compact_logging:
                    self.log(
                        "loss/gir_history_adapt",
                        gir_history_adapt.item(),
                    )
                total_loss = total_loss + adapt_weight * gir_history_adapt
        if (
            encoder_output.infos is not None
            and "gir_history_preserve_loss" in encoder_output.infos
        ):
            gir_history_preserve = encoder_output.infos[
                "gir_history_preserve_loss"
            ]
            preserve_weight = float(
                self.model.encoder.cfg.gir_history_preserve_weight
            )
            if preserve_weight != 0.0:
                if not compact_logging:
                    self.log(
                        "loss/gir_history_preserve",
                        gir_history_preserve.item(),
                    )
                total_loss = total_loss + preserve_weight * gir_history_preserve
        if (
            encoder_output.infos is not None
            and "gir_add_rate_loss" in encoder_output.infos
        ):
            gir_add_rate_loss = encoder_output.infos["gir_add_rate_loss"]
            gir_add_rate_weight = float(
                self.model.encoder.cfg.gir_add_rate_loss_weight
            )
            if gir_add_rate_weight != 0.0:
                if not compact_logging:
                    self.log("loss/gir_add_rate", gir_add_rate_loss.item())
                total_loss = (
                    total_loss + gir_add_rate_weight * gir_add_rate_loss
                )
        if (
            encoder_output.infos is not None
            and "gir_regularization_loss" in encoder_output.infos
        ):
            gir_regularization = encoder_output.infos["gir_regularization_loss"]
            regularization_weight = float(
                self.model.encoder.cfg.gir_regularization_weight
            )
            if regularization_weight != 0.0:
                if not compact_logging:
                    self.log(
                        "loss/gir_regularization",
                        gir_regularization.item(),
                    )
                total_loss = total_loss + regularization_weight * gir_regularization

        with torch.amp.autocast("cuda", enabled=False):
            loss_depth_ctx = depth_loss_module.ctx_depth_loss(
                depth_dict["depth"],
                batch,
                cxt_depth_weight=depth_loss_weight,
                prepared_target=prepared_context_depth_target,
            )
            log_cuda_memory("after_depth_loss", step=int(self.global_step))

            self.log(
                "loss/loss_depth_ctx",
                loss_depth_ctx.item())
            if debug_record is not None:
                debug_record["loss_depth"] = float(
                    loss_depth_ctx.detach().item()
                )
            if not compact_logging:
                for component_name, component_value in getattr(
                    depth_loss_module, "last_depth_components", {}
                ).items():
                    if component_name == "depth_absolute_log_scale_weighted":
                        continue
                    namespace = (
                        "diagnostics"
                        if component_name.startswith(
                            "depth_camera_scale_alignment_"
                        )
                        else "loss"
                    )
                    self.log(
                        f"{namespace}/{component_name}",
                        component_value,
                        on_step=True,
                        on_epoch=False,
                        sync_dist=True,
                    )
                    if debug_record is not None:
                        scalar = self._debug_scalar(component_value)
                        if scalar is not None:
                            debug_record[f"depth_{component_name}"] = scalar
                # Teacher diagnostics are detached and not part of the loss.
                for metric_name, metric_value in getattr(
                    depth_loss_module, "last_teacher_gt_metrics", {}
                ).items():
                    self.log(
                        f"diagnostics/{metric_name}",
                        metric_value,
                        on_step=True,
                        on_epoch=False,
                        sync_dist=True,
                    )
            total_loss = total_loss + loss_depth_ctx
            
            # depth_loss_idx = list(get_cfg()["loss"].keys()).index("depth")
            # depth_loss_fn = self.losses[depth_loss_idx].ctx_depth_loss
            # loss_depth_ctx = depth_loss_fn(
            #     depth_dict["depth"],
            #     batch,
            #     cxt_depth_weight=self.train_cfg.cxt_depth_weight,
            # )
            # # print(f'loss_depth_ctx :{loss_depth_ctx}')
            # self.log("loss/loss_depth_ctx", loss_depth_ctx.item())
            # total_loss = total_loss + loss_depth_ctx

            for loss_fn in self.losses:
                if loss_fn.name == "depth":
                    break
                loss = loss_fn.forward(output, batch, gaussians, depth_dict, self.global_step)
                log_cuda_memory(
                    f"after_{loss_fn.name}_loss",
                    step=int(self.global_step),
                )
                self.log(f"loss/{loss_fn.name}", loss.item())
                if debug_record is not None:
                    debug_record[f"loss_{loss_fn.name}"] = float(
                        loss.detach().item()
                    )
                total_loss = total_loss + loss

            predicted_context_c2w = None
            predicted_context_pose_encoding = None
            if encoder_output.infos is not None:
                predicted_context_c2w = encoder_output.infos.get(
                    "pred_context_c2w"
                )
                predicted_context_pose_encoding = encoder_output.infos.get(
                    "pred_context_pose_encoding"
                )
            gt_context_c2w = batch["context"].get("extrinsics")
            gt_context_intrinsics = batch["context"].get("intrinsics")
            if (
                self.train_cfg.weight_pose != 0
                and (
                    predicted_context_pose_encoding is None
                    or gt_context_c2w is None
                    or gt_context_intrinsics is None
                )
            ):
                raise RuntimeError(
                    "GT camera supervision is enabled, but the model did not "
                    "provide complete differentiable pose/FOV predictions or "
                    "the batch has no GT cameras."
                )
            if (
                self.train_cfg.weight_pose != 0
                and predicted_context_pose_encoding is not None
                and gt_context_c2w is not None
                and gt_context_intrinsics is not None
            ):
                source_view_count = int(depth_dict["depth"].shape[1])
                camera_losses = _camera_supervision_losses(
                    predicted_context_pose_encoding[:, :source_view_count],
                    gt_context_c2w[:, :source_view_count],
                    gt_context_intrinsics[:, :source_view_count],
                    translation_weight=float(self.train_cfg.pose_loss_alpha),
                    rotation_weight=float(self.train_cfg.pose_loss_delta),
                    focal_weight=float(self.train_cfg.camera_focal_weight),
                    pose_loss_mode=self.train_cfg.camera_pose_loss_mode,
                    translation_direction_weight=float(
                        self.train_cfg.camera_translation_direction_weight
                    ),
                    translation_direction_min_baseline_ratio=float(
                        self.train_cfg.camera_translation_direction_min_baseline_ratio
                    ),
                    translation_direction_baseline_weighting=bool(
                        self.train_cfg.camera_translation_direction_baseline_weighting
                    ),
                    global_trajectory_weight=float(
                        self.train_cfg.camera_global_trajectory_weight
                    ),
                )
                loss_camera = camera_losses["total"]
                if debug_record is not None:
                    for metric_name, metric_value in camera_losses.items():
                        scalar = self._debug_scalar(metric_value)
                        if scalar is not None:
                            debug_record[f"camera_{metric_name}"] = scalar
                    debug_record["loss_camera"] = float(
                        loss_camera.detach().item()
                    )
                self.log("loss/camera", loss_camera.item())
                self.log(
                    "loss/camera_translation",
                    camera_losses["translation"].item(),
                )
                self.log(
                    "loss/camera_rotation",
                    camera_losses["rotation"].item(),
                )
                self.log(
                    "camera/translation_alignment_scale",
                    camera_losses["translation_alignment_scale"].item(),
                )
                self.log("loss/camera_focal", camera_losses["focal"].item())
                if not compact_logging:
                    self.log(
                        "loss/camera_translation_shape",
                        camera_losses["translation_shape"].item(),
                    )
                    self.log(
                        "loss/camera_translation_direction",
                        camera_losses["translation_direction"].item(),
                    )
                    self.log(
                        "camera/translation_direction_valid_ratio",
                        camera_losses[
                            "translation_direction_valid_ratio"
                        ].item(),
                    )
                    self.log(
                        "camera/translation_direction_pair_weight_mean",
                        camera_losses[
                            "translation_direction_pair_weight_mean"
                        ].item(),
                    )
                    self.log(
                        "camera/translation_reference_baseline",
                        camera_losses[
                            "translation_reference_baseline"
                        ].item(),
                    )
                    self.log(
                        "loss/camera_global_trajectory",
                        camera_losses["global_trajectory"].item(),
                    )
                    self.log(
                        "loss/camera_global_translation_shape",
                        camera_losses[
                            "global_translation_shape"
                        ].item(),
                    )
                    self.log(
                        "loss/camera_global_rotation",
                        camera_losses["global_rotation"].item(),
                    )
                    self.log(
                        "camera/global_translation_alignment_scale",
                        camera_losses[
                            "global_translation_alignment_scale"
                        ].item(),
                    )
                self.log(
                    "camera/invalid_ratio",
                    camera_losses["invalid_ratio"],
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
                if not compact_logging:
                    self.log(
                        "camera/valid_view_count",
                        camera_losses["valid_view_count"],
                        on_step=True,
                        on_epoch=False,
                        sync_dist=True,
                    )
                total_loss = total_loss + self.train_cfg.weight_pose * loss_camera

            # int loss
            # loss_ca1 = F.mse_loss(
            #     pred_pose_enc_list, distill_infos["pred_pose_enc_list"][:, 0:1, -1]
            # )
            # loss_ca = 10 * loss_ca1
            # self.log("loss/loss_ca", loss_ca.item())
            # total_loss = total_loss + loss_ca
            # print(f'loss_sparsity:{loss_sparsity}')
            # self.log("loss/loss_new", loss_sparsity.item())
            # total_loss = total_loss + loss_sparsity
            # gt ex loss
#             gt_pose = batch["context"]["extrinsics"]
#             gt_t = gt_pose[:, :, :3, 3].float()  # BxSx3
#             gt_R_c2w = gt_pose[:, :, :3, :3].float()  # BxSx3x3
            
#             weights=[0.25, 0.5, 0.75, 1.0]
   
#             delta_t_gt = gt_t - gt_t[:, 0:1, :]

#             for i, pred_9d in enumerate(pred_list):
#                 # 1. 拆分 W2C 定义的 9 维向量
#                 t_w2c = pred_9d[..., :3]     # [B, S, 3]
#                 q_w2c = pred_9d[..., 3:7]    # [B, S, 4]

#                 # 2. 🌟 核心转换：W2C -> C2W 🌟
#                 # 获取 W2C 旋转矩阵
#                 R_w2c = quat_to_mat(q_w2c) # [B, S, 3, 3]

#                 # 相机在世界坐标系的旋转 R_c2w = R_w2c^T
#                 R_c2w_pred = R_w2c.transpose(-1, -2)

#                 # 相机在世界坐标系的位置 t_c2w = -R_w2c^T @ t_w2c
#                 # 这一步算出的才是相机真正的运动轨迹！
#                 t_c2w_pred = -torch.matmul(R_c2w_pred, t_w2c.unsqueeze(-1)).squeeze(-1) # [B, S, 3]

#                 # 3. 🌟 尺度不变对齐 (Procrustes Alignment) 🌟
#                 # 只在 C2W 轨迹空间算相对位移
#                 delta_t_pred = t_c2w_pred - t_c2w_pred[:, 0:1, :]

#                 # 计算最优缩放因子 alpha
#                 num = torch.sum(delta_t_pred * delta_t_gt, dim=(1, 2))
#                 den = torch.sum(delta_t_pred * delta_t_pred, dim=(1, 2)) + 1e-6
#                 alpha = (num / den).view(-1, 1, 1)

#                 # 平移 Loss (尺度对齐后)
#                 loss_t = F.mse_loss(alpha * delta_t_pred, delta_t_gt)

#                 # 4. 旋转 Loss (直接在 C2W 矩阵空间对比)
#                 loss_R = F.mse_loss(R_c2w_pred, gt_R_c2w)

#                 # 6. 组合阶段 Loss
#                 stage_loss = 10.0 * loss_t + 10.0 * loss_R
#                 total_loss += weights[i] * stage_loss

        self.log("loss/total", total_loss.item())
        if debug_record is not None:
            debug_record["loss_total"] = float(total_loss.detach().item())
        # ModelCheckpoint monitors this metric. Keep it in Lightning's callback
        # metrics even when compact mode suppresses it from the external logger.
        self.log(
            "info/global_step",
            self.global_step,
            logger=not compact_logging,
        )
        # print(f"total_loss: {total_loss}")
        # print(f"scene = {[x[:20] for x in batch['scene']]}; " )
        
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        del batch
        if self.global_step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        encoder_cfg = getattr(getattr(get_cfg(), "model", None), "encoder", None)
        if bool(getattr(encoder_cfg, "gir_debug_numerics", False)):
            debug_losses = {"total": float(total_loss.detach().float().item())}
            for key in (
                "gir_regularization_loss",
                "gir_aux_loss",
                "gir_base_rgb_loss",
                "gir_current_rgb_loss",
                "gir_new_hole_rgb_loss",
                "gir_replay_rgb_loss",
            ):
                value = (encoder_output.infos or {}).get(key)
                if value is not None:
                    debug_losses[key] = float(value.detach().float().item())
            self._gir_debug_losses = debug_losses
        
        log_cuda_memory("training_step_return", step=int(self.global_step))
        return total_loss

    def on_after_backward(self):
        log_cuda_memory("after_backward", step=int(self.global_step))
        # Keep DDP's strict unused-parameter check enabled. If a future graph
        # change disconnects a trainable branch, print exact names before DDP
        # raises on the next forward pass.
        if self.global_step > 1:
            return
        unused = [
            name
            for name, param in self.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        if unused:
            if self._pending_train_debug_record is not None:
                self._pending_train_debug_record["status"] = (
                    "disconnected_parameters"
                )
                self._pending_train_debug_record["unused_parameters"] = unused
                self._write_train_debug_record(
                    self._pending_train_debug_record
                )
                self._pending_train_debug_record = None
            details = (
                f"[DDP unused parameters][rank={self.global_rank}]\n"
                + "\n".join(unused)
            )
            print(details, file=sys.stderr, flush=True)
            raise RuntimeError(
                "Trainable parameters are disconnected from the training loss; "
                "see the parameter list printed immediately above."
            )

    def on_before_optimizer_step(self, optimizer):
        # 这个 hook 里检查的是“真正要 step 的梯度”
        self._skip_optimizer_step = False
        self._bad_grad_name = None
        self._bad_grad_summaries = []

        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            finite = torch.isfinite(p.grad)
            if not bool(finite.all()):
                self._skip_optimizer_step = True
                if self._bad_grad_name is None:
                    self._bad_grad_name = name
                self._bad_grad_summaries.append(
                    (name, int((~finite).sum().item()), p.grad.numel())
                )

        debug_record = self._pending_train_debug_record
        if debug_record is not None:
            group_norms = []
            for group_index, group in enumerate(optimizer.param_groups):
                group_name = str(group.get("name", f"group_{group_index}"))
                group_parameters = list(group["params"])
                group_norm = self._debug_gradient_norm(group_parameters)
                group_norms.append(group_norm)
                debug_record[f"lr_{group_name}"] = float(group["lr"])
                debug_record[f"grad_{group_name}_norm"] = group_norm
            debug_record["grad_total_norm"] = math.sqrt(
                sum(norm * norm for norm in group_norms)
            )

        if self._skip_optimizer_step:
            if self.global_rank == 0:
                print(
                    f"[skip step] global_step={self.global_step}, "
                    f"bad_param={self._bad_grad_name}, "
                    f"bad_param_count={len(self._bad_grad_summaries)}"
                )
                for name, bad_count, element_count in self._bad_grad_summaries[:8]:
                    print(
                        f"[bad grad] {name}: "
                        f"nonfinite={bad_count}/{element_count}",
                        flush=True,
                    )
                encoder_cfg = getattr(
                    getattr(get_cfg(), "model", None), "encoder", None
                )
                if bool(getattr(encoder_cfg, "gir_debug_numerics", False)):
                    print(
                        f"[GIR DEBUG] losses={getattr(self, '_gir_debug_losses', {})}",
                        flush=True,
                    )
                    gir_head = getattr(self.model, "gir_update_head", None)
                    if gir_head is not None:
                        for name, parameter in gir_head.named_parameters():
                            if not parameter.requires_grad:
                                continue
                            value = parameter.detach().float()
                            grad = parameter.grad
                            if grad is None:
                                print(
                                    f"[GIR DEBUG] grad {name}: None "
                                    f"param_absmax={value.abs().max().item():.6e}",
                                    flush=True,
                                )
                                continue
                            grad = grad.detach().float()
                            finite = torch.isfinite(grad)
                            finite_count = int(finite.sum().item())
                            if finite_count > 0:
                                finite_grad = grad[finite]
                                grad_min = finite_grad.min().item()
                                grad_max = finite_grad.max().item()
                                grad_absmax = finite_grad.abs().max().item()
                                grad_norm = torch.linalg.vector_norm(
                                    finite_grad
                                ).item()
                            else:
                                grad_min = float("nan")
                                grad_max = float("nan")
                                grad_absmax = float("nan")
                                grad_norm = float("nan")
                            print(
                                f"[GIR DEBUG] grad {name}: "
                                f"finite={finite_count}/{grad.numel()} "
                                f"min={grad_min:.6e} max={grad_max:.6e} "
                                f"absmax={grad_absmax:.6e} "
                                f"norm={grad_norm:.6e} "
                                f"param_absmax={value.abs().max().item():.6e}",
                                flush=True,
                            )

        if not self._skip_optimizer_step:
            camera_head = getattr(self.model.encoder, "camera_head", None)
            camera_parameters = (
                [
                    parameter
                    for parameter in camera_head.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if camera_head is not None
                else []
            )
            if camera_parameters:
                camera_grad_clip = float(self.train_cfg.camera_grad_clip)
                camera_grad_norm = torch.nn.utils.clip_grad_norm_(
                    camera_parameters,
                    (
                        camera_grad_clip
                        if camera_grad_clip > 0.0
                        else float("inf")
                    ),
                )
                if debug_record is not None:
                    camera_grad_norm_value = float(
                        camera_grad_norm.detach().item()
                    )
                    debug_record["grad_camera_norm"] = (
                        camera_grad_norm_value
                    )
                    debug_record["camera_grad_clip"] = camera_grad_clip
                    debug_record["camera_grad_clipped"] = int(
                        camera_grad_clip > 0.0
                        and camera_grad_norm_value > camera_grad_clip
                    )
                self.log(
                    "train/camera_grad_norm",
                    camera_grad_norm,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )

        if debug_record is not None:
            if self._skip_optimizer_step:
                debug_record["status"] = "nonfinite_gradient"
                debug_record["bad_gradients"] = [
                    {
                        "name": name,
                        "nonfinite": bad_count,
                        "elements": element_count,
                    }
                    for name, bad_count, element_count in self._bad_grad_summaries
                ]
            else:
                debug_record["status"] = "ok"
            self._write_train_debug_record(debug_record)
            self._pending_train_debug_record = None

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # 如果这一轮梯度坏了：不更新参数，只清梯度
        if self._skip_optimizer_step:
            optimizer.zero_grad(set_to_none=True)
            self._skip_optimizer_step = False
            self._bad_grad_name = None
            self._bad_grad_summaries = []
            return

        # 正常情况才 step
        optimizer.step(closure=optimizer_closure)
        log_cuda_memory("after_optimizer_step", step=int(self.global_step))
        self._skip_optimizer_step = False
        self._bad_grad_name = None
        self._bad_grad_summaries = []
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # if self.global_step < 20:
        #     return
        batch: BatchedExample = self.data_shim(batch)
        total_batches = len(self.trainer.datamodule.val_loader)
        if self.global_rank == 0:
            print(f"Rank {self.global_rank}, batch {batch_idx+1}/{total_batches}")
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        encoder_output, output = self.model(
            (batch["context"]["image"] + 1) / 2,
            batch["context"]["index"],
            self.global_step,
            gt_context_c2w=batch["context"]["extrinsics"],
            gt_context_intrinsics=batch["context"]["intrinsics"],
        )

        # Compute validation metrics over every sample/view in the batch.
        rgb_pred = rearrange(output.color.float(), "b v c h w -> (b v) c h w")
        rgb_gt = rearrange((batch["context"]["image"].float() + 1) / 2, "b v c h w -> (b v) c h w")
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr, sync_dist=True)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips, sync_dist=True)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim, sync_dist=True)

        if self.global_rank != 0:
            return

        depth_dict = encoder_output.depth_dict
        scenes = batch["scene"]
        for i in range(len(scenes)):
            if len(self._val_comparison_images) >= self.train_cfg.max_val_comparisons:
                break

            context_img = inverse_normalize(batch["context"]["image"][i])
            context = [context_img[j] for j in range(context_img.shape[0])]

            model_depth_pred = depth_dict["depth"].squeeze(-1)[i]
            model_depth_pred = vis_depth_map(model_depth_pred)

            accumulated_rendered_depth = output.depth[i]
            rendered_alpha = output.alpha[
                i, : accumulated_rendered_depth.shape[0]
            ]
            if accumulated_rendered_depth.ndim == 2:
                accumulated_rendered_depth = accumulated_rendered_depth.unsqueeze(0)
                rendered_alpha = rendered_alpha.unsqueeze(0)
            rendered_depth, rendered_depth_valid = (
                _expected_render_depth_for_visualization(
                    accumulated_rendered_depth, rendered_alpha
                )
            )
            depth_pred = vis_depth_map(rendered_depth)
            depth_pred = depth_pred * rendered_depth_valid.unsqueeze(-3)
            rgb_gt_i = (batch["context"]["image"][i].float() + 1) / 2
            rgb_pred_i = output.color[i].float()

            comparison = hcat(
                add_label(vcat(*context), "Context"),
                add_label(vcat(*rgb_gt_i), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred_i), "Rendered Target"),
                add_label(vcat(*model_depth_pred), "GS Depth"),
                add_label(vcat(*depth_pred), "Rendered Depth"),
            )

            comparison = torch.nn.functional.interpolate(
                comparison.unsqueeze(0),
                scale_factor=0.5,
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

            self._val_comparison_images.append(prep_image(add_border(comparison)))
            self._val_comparison_captions.append(
                f"step={self.global_step}, batch={batch_idx}, index={i}, scene={scenes[i]}"
            )

    def on_test_epoch_start(self) -> None:
        self._test_scene_rows = []
        self._test_result_summary = None
        self.running_metrics = None
        for name in ("_test_gir_diagnostic_sums", "_test_gir_diagnostic_count"):
            if hasattr(self, name):
                delattr(self, name)

    def _save_test_gir_debug_intermediates(
        self,
        scene: str,
        context_indices: torch.Tensor,
        records: list[dict[str, Any]],
    ) -> None:
        if not records:
            print(f"[GIR DEBUG] scene={scene}: no update-view intermediates.")
            return

        safe_scene = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in scene
        )
        run_name = str(get_cfg()["wandb"]["name"])
        output_dir = (
            self.test_cfg.output_path
            / run_name
            / "gir_debug"
            / f"rank_{int(self.global_rank):02d}"
            / safe_scene
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        for record in records:
            view_idx = int(record["view_index"])
            frame_idx = int(context_indices[0, view_idx].item())
            current_rgb = record["current_rgb"][0].float().clamp(0.0, 1.0)
            image_shape = tuple(current_rgb.shape[-2:])

            def resize(value: torch.Tensor | None, mode: str) -> torch.Tensor:
                if value is None:
                    return torch.zeros((1, *image_shape), dtype=torch.float32)
                value = value[0].float()
                if value.ndim == 2:
                    value = value.unsqueeze(0)
                if tuple(value.shape[-2:]) != image_shape:
                    interpolation_args = {}
                    if mode in {"bilinear", "bicubic"}:
                        interpolation_args["align_corners"] = False
                    value = F.interpolate(
                        value.unsqueeze(0),
                        size=image_shape,
                        mode=mode,
                        **interpolation_args,
                    ).squeeze(0)
                return value

            def rgb(key: str) -> torch.Tensor:
                value = resize(record.get(key), "bilinear")
                if value.shape[0] == 1:
                    value = value.repeat(3, 1, 1)
                return value[:3].clamp(0.0, 1.0)

            def mask(key: str) -> torch.Tensor:
                value = resize(record.get(key), "nearest")[:1].clamp(0.0, 1.0)
                return value.repeat(3, 1, 1)

            def heat(key: str, color_map: str = "magma") -> torch.Tensor:
                value = resize(record.get(key), "bilinear")[:1].clamp(0.0, 1.0)
                return apply_color_map_to_image(value[0], color_map)

            current_depth = resize(record.get("current_depth"), "bilinear")[:1]
            old_depth = resize(record.get("old_depth"), "bilinear")[:1]
            valid_depth_values = []
            for depth_value in (current_depth, old_depth):
                valid = torch.isfinite(depth_value) & (depth_value > 1e-6)
                if bool(valid.any()):
                    valid_depth_values.append(depth_value[valid].log())
            if valid_depth_values:
                log_depth_values = torch.cat(valid_depth_values)
                depth_near = log_depth_values.quantile(0.01)
                depth_far = log_depth_values.quantile(0.99)
                depth_far = torch.maximum(
                    depth_far, depth_near + depth_near.new_tensor(1e-6)
                )
            else:
                depth_near = torch.tensor(0.0)
                depth_far = torch.tensor(1.0)

            def depth_image(depth_value: torch.Tensor) -> torch.Tensor:
                valid = torch.isfinite(depth_value) & (depth_value > 1e-6)
                safe_depth = torch.where(
                    valid, depth_value, depth_near.exp().expand_as(depth_value)
                )
                normalized = 1.0 - (
                    safe_depth.log() - depth_near
                ) / (depth_far - depth_near)
                colored = apply_color_map_to_image(
                    normalized[0].clamp(0.0, 1.0), "turbo"
                )
                return colored * valid[0].float().unsqueeze(0)

            append_mask = mask("append_mask")
            append_overlay = current_rgb * (1.0 - 0.45 * append_mask[:1])
            append_overlay = append_overlay + torch.tensor(
                [1.0, 0.0, 0.0], dtype=current_rgb.dtype
            )[:, None, None] * (0.45 * append_mask[:1])

            def tile(image: torch.Tensor, label: str) -> torch.Tensor:
                return add_label(image, label, font_size=18)

            panel = vcat(
                hcat(
                    tile(current_rgb, "Current RGB"),
                    tile(rgb("old_rgb"), "Old-map render"),
                    tile(heat("rgb_error", "turbo"), "Absolute RGB error"),
                    gap=4,
                ),
                hcat(
                    tile(depth_image(current_depth), "Current predicted depth"),
                    tile(depth_image(old_depth), "Old-map rendered depth"),
                    tile(heat("old_alpha"), "Old-map alpha"),
                    gap=4,
                ),
                hcat(
                    tile(mask("point_coverage"), "Point-map coverage"),
                    tile(append_mask, "Append mask"),
                    tile(append_overlay.clamp(0.0, 1.0), "Append overlay"),
                    gap=4,
                ),
                hcat(
                    tile(mask("old_coverage"), "Old raster coverage"),
                    tile(mask("depth_consistent"), "Depth consistent"),
                    tile(mask("old_in_front"), "Old in front"),
                    gap=4,
                ),
                hcat(
                    tile(mask("residual_mask"), "Old residual candidates"),
                    tile(
                        mask("front_conflict_projection"),
                        "Front-conflict GS projection",
                    ),
                    tile(
                        mask("front_decay_projection"),
                        "Applied opacity-decay projection",
                    ),
                    gap=4,
                ),
                gap=4,
            )
            stem = f"update_{view_idx:02d}_frame_{frame_idx:06d}"
            save_image(panel, output_dir / f"{stem}_panel.png")
            for key in (
                "append_mask",
                "point_coverage",
                "residual_mask",
                "front_conflict_projection",
                "front_decay_projection",
                "add_gate",
            ):
                if record.get(key) is not None:
                    save_image(mask(key), output_dir / f"{stem}_{key}.png")

            summaries.append(
                {
                    "view_index": view_idx,
                    "frame_index": frame_idx,
                    **record["summary"],
                }
            )

        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "scene": scene,
                    "rank": int(self.global_rank),
                    "updates": summaries,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(
            f"[GIR DEBUG] saved {len(records)} update views for scene={scene} "
            f"to {output_dir}"
        )

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        scene_name = str(batch["scene"][0])
        debug_max_scenes = int(self.test_cfg.gir_debug_max_scenes)
        if debug_max_scenes < -1:
            raise ValueError(
                "test.gir_debug_max_scenes must be -1 or non-negative, "
                f"got {debug_max_scenes}."
            )
        debug_view_stride = int(self.test_cfg.gir_debug_view_stride)
        if debug_view_stride < 1:
            raise ValueError(
                "test.gir_debug_view_stride must be at least 1, "
                f"got {debug_view_stride}."
            )
        world_size = max(1, int(getattr(self.trainer, "world_size", 1)))
        debug_scene_ordinal = batch_idx * world_size + int(self.global_rank)
        save_gir_debug = bool(
            self.test_cfg.gir_debug_save_intermediates
            and (
                debug_max_scenes == -1
                or debug_scene_ordinal < debug_max_scenes
            )
        )
        test_append_masks = None
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")
        
        target_image = batch["target"]["image"]
        target_view_count = target_image.shape[1]
        evaluation_cameras = resolve_camera_phase_settings(
            self.model.encoder.cfg, False
        )
        context_indices = batch["context"]["index"]
        target_indices = batch["target"]["index"]
        if context_indices.shape[1] <= target_view_count:
            raise RuntimeError(
                "Test input must contain source views followed by duplicated target "
                "views for causal target-pose prediction."
            )
        if not torch.equal(
            context_indices[:, -target_view_count:], target_indices
        ):
            raise RuntimeError(
                "Test context/target mismatch: the final context views must match "
                "the target views in the same order."
            )
        
        with torch.no_grad():
            with self.benchmarker.time("encoder"):
                (
                    encoder_output,
                    pred_all_extrinsic,
                    intrinsic, 
                    depth_map, 
                    ctx_img_num,
                    # single_opacities
                    # primitive_mask_hard,
                    # sampling_vis
                ) = self.model.encoder(
                    (batch["context"]["image"] + 1) / 2,
                    batch["context"]["index"],
                    global_step=self.global_step, 
                    # decoder = self.model.decoder, 
                    name = batch['scene'],
                    target_view_count=target_view_count,
                    return_gir_data=True,
                    gt_context_c2w=batch["context"]["extrinsics"],
                    gt_context_intrinsics=batch["context"]["intrinsics"],
                    # global_rank = self.global_rank
                )
                gaussians = self.model._refine_gaussians(
                    encoder_output,
                    (batch["context"]["image"] + 1) / 2,
                    pred_all_extrinsic,
                    encoder_output.pred_context_pose,
                    ctx_img_num,
                    0.01,
                    100.0,
                    global_step=self.global_step,
                    test_add_gate_prune_threshold=(
                        self.test_cfg.gir_add_gate_prune_threshold
                    ),
                    test_top1_confidence_mode=(
                        self.test_cfg.gir_test_top1_confidence_mode
                    ),
                    test_top1_confidence_floor=(
                        self.test_cfg.gir_test_top1_confidence_floor
                    ),
                    return_debug_intermediates=save_gir_debug,
                    debug_view_stride=debug_view_stride,
                    return_append_masks=bool(self.test_cfg.save_compare),
                    residual_camera_to_world=(
                        batch["context"]["extrinsics"][:, :ctx_img_num]
                        if evaluation_cameras.source_render_camera == "gt"
                        else pred_all_extrinsic[:, :ctx_img_num]
                    ),
                    residual_intrinsics=(
                        batch["context"]["intrinsics"][:, :ctx_img_num]
                        if evaluation_cameras.source_render_camera == "gt"
                        else intrinsic[:, :ctx_img_num]
                    ),
                )
                if encoder_output.infos is not None:
                    test_append_masks = encoder_output.infos.pop(
                        "gir_append_masks", None
                    )
                if self.test_cfg.save_unfiltered_ply:
                    safe_scene_name = scene_name.replace("/", "_").replace(
                        "\\", "_"
                    )
                    ply_path = (
                        self.test_cfg.output_path
                        / "gaussians_unfiltered"
                        / f"{safe_scene_name}.ply"
                    )
                    export_ply(
                        gaussians.means[0],
                        gaussians.scales[0],
                        gaussians.rotations[0],
                        gaussians.harmonics[0],
                        gaussians.opacities[0],
                        ply_path,
                        shift_and_scale=False,
                        filter_gaussians=False,
                    )
                    print(
                        f"[test] exported {gaussians.means.shape[1]} "
                        f"unfiltered GS to {ply_path}"
                    )
                if (
                    encoder_output.infos is not None
                    and "gir_history_views" in encoder_output.infos
                ):
                    final_history = int(
                        encoder_output.infos["gir_history_views"].item()
                    )
                    if final_history != ctx_img_num - 1:
                        raise RuntimeError(
                            "GIR test rollout did not consume all causal context views: "
                            f"history={final_history}, expected={ctx_img_num - 1}."
                        )

        gir_diagnostics = {}
        if (
            encoder_output.infos is not None
            and "gir_map_gaussians" in encoder_output.infos
        ):
            opacity = gaussians.opacities.detach().float()
            gir_diagnostics = {
                "map_gaussians": float(opacity.numel()),
                "map_opacity_mean": opacity.mean().item(),
                "map_opacity_mass": opacity.sum().item(),
            }
            for threshold in (0.001, 0.003, 0.005, 0.01):
                suffix = str(threshold).replace(".", "_")
                gir_diagnostics[f"map_above_{suffix}_ratio"] = (
                    (opacity > threshold).float().mean().item()
                )

            info_keys = (
                "gir_old_residual_enabled",
                "gir_add_gate",
                "gir_append_mode_code",
                "gir_point_append_depth_check_enabled",
                "gir_point_append_align_depth_scale_enabled",
                "gir_point_map_follow_mean_updates_enabled",
                "gir_point_coverage_ratio",
                "gir_point_append_ratio",
                "gir_tracker_coverage_ratio",
                "gir_tracker_point_union_coverage_ratio",
                "gir_confirmed_current_in_front_ratio",
                "gir_tracker_visibility_ratio",
                "gir_tracker_fb_consistency_ratio",
                "gir_tracker_cycle_error_mean",
                "gir_point_append_hybrid_enabled",
                "gir_point_front_conflict_ratio",
                "gir_point_front_gradient_decay_gate_ratio",
                "gir_point_front_decay_mass_ratio",
                "gir_point_depth_scale",
                "gir_point_depth_scale_min",
                "gir_point_depth_scale_max",
                "gir_add_gate_learned",
                "gir_add_gate_warmup_progress",
                "gir_add_rate",
                "gir_new_opacity_mass_ratio",
                "gir_add_gate_below_0_1_ratio",
                "gir_add_gate_below_0_2_ratio",
                "gir_effective_new_above_0_001_ratio",
                "gir_effective_new_above_0_003_ratio",
                "gir_effective_new_above_0_005_ratio",
                "gir_effective_new_above_0_01_ratio",
                "gir_test_prune_threshold",
                "gir_test_pruned_new_ratio",
                "gir_test_map_reduction_ratio",
                "gir_top1_ownership_mean",
                "gir_top1_ownership_above_0_1_ratio",
                "gir_top1_ownership_above_0_25_ratio",
                "gir_top1_ownership_above_0_5_ratio",
                "gir_top1_confidence_mean",
                "gir_top1_confidence_mode_code",
                "gir_top1_confidence_floor",
                "gir_old_delete_target_ratio",
                "gir_old_delete_threshold",
                "gir_old_delete_candidate_probability",
                "gir_old_delete_candidate_count",
                "gir_old_delete_candidate_rate",
                "gir_old_delete_hard_ratio",
                "gir_old_delete_removed_opacity_mass_ratio",
                "gir_old_delete_map_count_before",
                "gir_old_delete_map_count_after",
                "gir_old_decay_topk",
                "gir_old_decay_strength",
                "gir_old_decay_prune_threshold",
                "gir_old_decay_candidate_probability",
                "gir_old_decay_candidate_count",
                "gir_old_decay_opacity_mass_ratio",
                "gir_old_decay_hard_pruned_ratio",
                "gir_old_decay_map_count_before",
                "gir_old_decay_map_count_after",
                "gir_split_enabled",
                "gir_split_eligible_count",
                "gir_split_count",
                "gir_split_ratio",
                "gir_split_selected_radius",
                "gir_split_selected_gradient_norm",
                "gir_split_map_count_before",
                "gir_split_map_count_after",
                "gir_functional_update_selected_count",
                "gir_functional_update_selected_ratio",
                "gir_functional_update_score_mean",
                "gir_functional_update_mean_gradient_norm",
                "gir_functional_update_mean_step_norm",
                "gir_functional_update_opacity_gradient_abs",
                "gir_functional_update_opacity_logit_step_abs",
                "gir_functional_update_harmonics_gradient_norm",
                "gir_functional_update_harmonics_step_norm",
            )
            info_keys += tuple(
                key
                for key in encoder_output.infos
                if key.startswith(("gir_projected_", "gir_new_residual_"))
                and "gradient_" in key
                and key not in info_keys
            )
            for key in info_keys:
                if key in encoder_output.infos:
                    gir_diagnostics[key.removeprefix("gir_")] = float(
                        encoder_output.infos[key].detach().float().item()
                    )

            if not self.test_cfg.compute_scores:
                self.log_dict(
                    {
                        f"test/gir_{key}": torch.tensor(value, device=opacity.device)
                        for key, value in gir_diagnostics.items()
                    },
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            self._record_test_gir_diagnostics(gir_diagnostics)

            print(
                "[GIR TEST] "
                f"scene={scene_name} "
                f"old_residual="
                f"{gir_diagnostics.get('old_residual_enabled', float('nan')):.0f} "
                f"top1_mode_code="
                f"{gir_diagnostics.get('top1_confidence_mode_code', float('nan')):.0f} "
                f"old_delete_p="
                f"{gir_diagnostics.get('old_delete_candidate_probability', float('nan')):.4f} "
                f"old_delete_candidate_rate="
                f"{gir_diagnostics.get('old_delete_candidate_rate', 0.0):.4f} "
                f"old_deleted="
                f"{gir_diagnostics.get('old_delete_hard_ratio', 0.0):.4f} "
                f"old_map="
                f"{gir_diagnostics.get('old_delete_map_count_before', gir_diagnostics['map_gaussians']):.0f}->"
                f"{gir_diagnostics.get('old_delete_map_count_after', gir_diagnostics['map_gaussians']):.0f} "
                f"prune_threshold="
                f"{gir_diagnostics.get('test_prune_threshold', 0.0):.3f} "
                f"pruned_new="
                f"{gir_diagnostics.get('test_pruned_new_ratio', 0.0):.4f} "
                f"map_reduction="
                f"{gir_diagnostics.get('test_map_reduction_ratio', 0.0):.4f} "
                f"map={int(gir_diagnostics['map_gaussians'])} "
                f"opacity_mass_ratio="
                f"{gir_diagnostics.get('new_opacity_mass_ratio', float('nan')):.4f} "
                f"add_gate={gir_diagnostics.get('add_gate', float('nan')):.4f} "
                f"gate<0.1="
                f"{gir_diagnostics.get('add_gate_below_0_1_ratio', float('nan')):.4f} "
                f"map>0.001={gir_diagnostics['map_above_0_001_ratio']:.4f} "
                f"map>0.005={gir_diagnostics['map_above_0_005_ratio']:.4f} "
                f"map>0.01={gir_diagnostics['map_above_0_01_ratio']:.4f}"
            )

        if save_gir_debug:
            debug_records = (encoder_output.infos or {}).pop(
                "gir_debug_intermediates", []
            )
            self._save_test_gir_debug_intermediates(
                scene_name,
                context_indices,
                debug_records,
            )
        # if self.global_rank == 0:
        # export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0].permute(0,2,1), single_opacities, Path(f"gaussians_{[x[:20] for x in batch['scene']]}.ply"))
        
#         import re    
#         prediction_path = '/openbayes/home/AnySplat'
#         img_folder = os.path.join(prediction_path, 'output_frames') 
#         video_save_path = os.path.join(prediction_path, 'output_video.mp4') 
#         fps = 30 
#         if self.global_rank == 0:
#             images = [img for img in os.listdir(img_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
#             if len(images) > 0:
#                 images.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else x)
#                 first_img_path = os.path.join(img_folder, images[0])
#                 first_img = cv2.imread(first_img_path)
#                 H, W, _ = first_img.shape

#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (W, H))

#                 print(f"开始将 {len(images)} 张图片合成视频...")

#                 for img_name in images:
#                     img_path = os.path.join(img_folder, img_name)
#                     frame = cv2.imread(img_path)

#                     if frame is None:
#                         print(f"警告：无法读取图片 {img_name}，跳过。")
#                         continue

#                     if (frame.shape[1], frame.shape[0]) != (W, H):
#                         frame = cv2.resize(frame, (W, H))

#                     video_writer.write(frame)

#                 video_writer.release()
#                 print(f"视频已成功保存至: {video_save_path}")
#             else:
#                 print(f"错误：文件夹 {img_folder} 中没有找到图片。")
                
        def save_depth_and_presence_for_eval(
            pred_depth,              # [b, v, h, w, 1]
            primitive_mask_hard,     # [b*v*h*w, 1]
            out_dir,
            batch_idx=0,
        ):
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            assert pred_depth.ndim == 5, pred_depth.shape
            assert pred_depth.shape[-1] == 1, pred_depth.shape

            b, v, h, w, _ = pred_depth.shape

            assert primitive_mask_hard.shape == (b * v * h * w, 1), (
                primitive_mask_hard.shape,
                (b * v * h * w, 1),
            )

            # depth: [b, v, h, w, 1] -> [b, v, h, w]
            depth_np = pred_depth.detach().float().cpu().numpy()[..., 0]

            # mask: [b*v*h*w, 1] -> [b, v, h, w]
            mask_np = (
                primitive_mask_hard
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(b, v, h, w)
            )
            mask_int = mask_np.astype("uint8")

            total_ones = int(mask_int.sum())
            total_pixels = int(mask_int.size)
            total_zeros = total_pixels - total_ones

            print(f"[MASK DEBUG] total pixels: {total_pixels}")
            print(f"[MASK DEBUG] total ones : {total_ones}")
            print(f"[MASK DEBUG] total zeros: {total_zeros}")
            print(f"[MASK DEBUG] keep ratio  : {total_ones / max(total_pixels, 1):.6f}")
            for view_id in range(v):
                depth = depth_np[batch_idx, view_id].astype("float32")  # [h, w]
                mask = mask_np[batch_idx, view_id].astype("uint8") * 255

                depth_path = out_dir / f"pred_depth_{view_id:03d}.exr"
                mask_path = out_dir / f"existence_{view_id:03d}.png"

                ok = cv2.imwrite(str(depth_path), depth)
                if not ok:
                    raise RuntimeError(f"Failed to save depth: {depth_path}")

                Image.fromarray(mask).save(mask_path)

            print(f"[DONE] saved {v} views to {out_dir}")
            
        # save_depth_and_presence_for_eval(
        #     pred_depth=depth_map,                  # [b, v, h, w, 1]
        #     primitive_mask_hard=primitive_mask_hard, # [b*v*h*w, 1]
        #     out_dir=f"/openbayes/home/AnySplat/{batch['scene']}",
        #     batch_idx=0,
        # )
        from .utils import make_sampling_heatmap_overlay_tensors
        num_context_view = ctx_img_num
        pred_all_target_extrinsic = pred_all_extrinsic[:, ctx_img_num:]
        pred_all_target_intrinsic = intrinsic[:, ctx_img_num:]
        if evaluation_cameras.heldout_test_camera == "gt":
            pred_all_target_extrinsic = batch["target"]["extrinsics"]
            pred_all_target_intrinsic = batch["target"]["intrinsics"]
        render_view_count = pred_all_target_extrinsic.shape[1]
        if render_view_count != target_view_count:
            raise RuntimeError(
                "Test target-pose count mismatch: "
                f"predicted={render_view_count}, expected={target_view_count}."
            )
        if pred_all_target_intrinsic.shape[1] != target_view_count:
            raise RuntimeError(
                "Test target-intrinsic count mismatch: "
                f"predicted={pred_all_target_intrinsic.shape[1]}, "
                f"expected={target_view_count}."
            )
        render_device = gaussians.means.device
                
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.model.decoder.forward(
                gaussians,
                pred_all_target_extrinsic,
                pred_all_target_intrinsic.float(),
                torch.ones(1, render_view_count, device=render_device) * 0.01,
                torch.ones(1, render_view_count, device=render_device) * 100,
                (h, w),
            )

        rgb_pred = output.color[0].float()
        rgb_gt = target_image[0].float()
        psnr = None
        with torch.no_grad():
            if self.test_cfg.compute_scores:
                psnr = compute_psnr(rgb_gt, rgb_pred).mean().item()
                all_metrics = {
                    f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean().item(),
                    f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean().item(),
                    f"psnr_ours": psnr,
                }
                methods = ["ours"]
                self._test_scene_rows.append({
                    "scene": str(batch["scene"][0]),
                    "source_indices": context_indices[0, :ctx_img_num].tolist(),
                    "target_indices": target_indices[0].tolist(),
                    "image_shape": [h, w],
                    **all_metrics,
                    "gir": gir_diagnostics,
                })
                self.print_preview_metrics(all_metrics, methods)

        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        
        # for i in range(len(batch["target"]["index"])):
        #     idx = batch["target"]["index"][i].item()  
        #     single_color = output.color[0][i] 
        #     res_path = path / f"{psnr}_{scene}" / "color" / f"{int(idx):0>6}.png"
        #     save_image(single_color, res_path)

        if self.test_cfg.save_compare:
            context_img = inverse_normalize(
                batch["context"]["image"][0, :ctx_img_num]
            )
            model_depth_pred = vis_depth_map(depth_map.squeeze(-1)[0])
            accumulated_rendered_depth = output.depth[0]
            rendered_alpha = output.alpha[0]
            if accumulated_rendered_depth.ndim == 2:
                accumulated_rendered_depth = accumulated_rendered_depth.unsqueeze(0)
                rendered_alpha = rendered_alpha.unsqueeze(0)
            rendered_depth, rendered_depth_valid = (
                _expected_render_depth_for_visualization(
                    accumulated_rendered_depth, rendered_alpha
                )
            )
            depth_pred = vis_depth_map(rendered_depth)
            depth_pred = depth_pred * rendered_depth_valid.unsqueeze(-3)
            comparison_columns = [
                add_label(vcat(*context_img), "Context"),
            ]
            if test_append_masks is not None:
                append_masks = test_append_masks[0].float()
                if append_masks.shape[0] != ctx_img_num:
                    raise RuntimeError(
                        "Test append-mask count must match source views: "
                        f"masks={append_masks.shape[0]}, views={ctx_img_num}."
                    )
                append_masks = F.interpolate(
                    append_masks,
                    size=(h, w),
                    mode="nearest",
                )
                append_masks = append_masks.repeat(1, 3, 1, 1).to(
                    device=context_img.device,
                    dtype=context_img.dtype,
                )
                comparison_columns.append(
                    add_label(vcat(*append_masks), "GS Append Mask")
                )
            comparison_columns.extend(
                (
                    add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                    add_label(vcat(*rgb_pred), "Rendered Target"),
                    add_label(vcat(*model_depth_pred), "GS Depth"),
                    add_label(vcat(*depth_pred), "Rendered Depth"),
                )
            )
            comparison = hcat(*comparison_columns)
            save_image(comparison, path / f"{psnr}_{scene}.png")
        
#         if self.test_cfg.save_compare:
#             context_img = inverse_normalize(batch["context"]["image"][0][:num_context_view])  # [Vc, 3, H, W]

#             compare_items = [
#                 add_label(vcat(*context_img), "Context"),
#                 add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
#                 add_label(vcat(*rgb_pred), "Rendered Target"),
#                 add_label(vcat(*model_depth_pred), "GS Depth"),
#                 add_label(vcat(*depth_pred), "Rendered Depth"),
#             ]

#             # --------------------------------------------------
#             # Sampling heatmap visualization
#             # --------------------------------------------------
#             if sampling_vis is not None:
#                 u_v_pixel = sampling_vis["u_v_pixel"]
#                 primitive_mask_hard = sampling_vis.get("primitive_mask_hard", None)

#                 b_vis = sampling_vis.get("b", batch["context"]["image"].shape[0])
#                 v_vis = sampling_vis.get("v", batch["context"]["image"].shape[1])

#                 # 注意：这里用 context_img，因为 sampling 是发生在输入/context views 上的
#                 # context_img: [Vc, 3, H, W] -> [1, Vc, 3, H, W]
#                 sampling_heatmaps = make_sampling_heatmap_overlay_tensors(
#                     image=context_img.unsqueeze(0),
#                     u_v_pixel=u_v_pixel,
#                     b=b_vis,
#                     v=v_vis,
#                     primitive_mask_hard=primitive_mask_hard,
#                     batch_idx=0,
#                     alpha=0.55,
#                     blur_ksize=21,
#                     blur_sigma=7.0,
#                 )

#                 overlay_all = sampling_heatmaps["overlay_all"]      # [Vc, 3, H, W]
#                 overlay_kept = sampling_heatmaps["overlay_kept"]    # [Vc, 3, H, W] or None
#                 heat_all = sampling_heatmaps["heat_all"]            # [Vc, 3, H, W]
#                 heat_kept = sampling_heatmaps["heat_kept"]          # [Vc, 3, H, W] or None

#                 compare_items.append(
#                     add_label(vcat(*overlay_all), "Sampled Positions")
#                 )

#                 compare_items.append(
#                     add_label(vcat(*heat_all), "Sample Heatmap")
#                 )

#                 if overlay_kept is not None:
#                     compare_items.append(
#                         add_label(vcat(*overlay_kept), "Kept After Mask")
#                     )

#                 if heat_kept is not None:
#                     compare_items.append(
#                         add_label(vcat(*heat_kept), "Kept Heatmap")
#                     )

#             comparison = hcat(*compare_items)
#             save_image(comparison, path / f"{psnr}_{scene}.png")

    def on_test_epoch_end(self) -> None:
        if not self.test_cfg.compute_scores:
            return
        from omegaconf import OmegaConf

        rank_rows = [self._test_scene_rows]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_rows = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rank_rows, self._test_scene_rows)
        loader = self.trainer.test_dataloaders
        if isinstance(loader, (list, tuple)):
            (loader,) = loader
        rows, summary = summarize_scenes(rank_rows, len(loader.dataset))
        summary["checkpoint"] = str(Path(get_cfg().checkpointing.load).resolve())
        summary["world_size"] = self.trainer.world_size
        summary["config"] = OmegaConf.to_container(get_cfg(), resolve=True)
        self._test_result_summary = summary
        # All ranks have the same de-duplicated aggregate; do not average it again.
        self.log_dict(summary["metrics"], on_epoch=True, sync_dist=False)
        self.log_dict({f"test/gir_{k}": v for k, v in summary["gir"].items()},
                      on_epoch=True, sync_dist=False)
        if self.global_rank == 0:
            if self.test_cfg.save_metrics:
                write_results(self.test_cfg.output_path, rows, summary)
            print(f"\n[BENCHMARK RESULT] scenes={summary['scene_count']} "
                  f"duplicates_removed={summary['duplicate_rows_removed']}")
            for key, value in summary["metrics"].items():
                print(f"{key}={value:.6f}")
            if self.test_cfg.save_metrics:
                print(f"Results: {self.test_cfg.output_path / 'metrics.json'}")

    def on_test_end(self) -> None:
        self.benchmarker.summarize()
        if self._test_result_summary is None:
            self._print_test_gir_diagnostics()
        elif self.global_rank == 0:
            print("\n[GIR TEST SUMMARY]")
            for key, value in self._test_result_summary["gir"].items():
                print(f"{key}={value:.6f}")

    def _record_test_gir_diagnostics(
        self,
        diagnostics: dict[str, float],
    ) -> None:
        if not hasattr(self, "_test_gir_diagnostic_sums"):
            self._test_gir_diagnostic_sums = {
                key: 0.0 for key in diagnostics
            }
            self._test_gir_diagnostic_count = 0
        for key, value in diagnostics.items():
            self._test_gir_diagnostic_sums[key] += value
        self._test_gir_diagnostic_count += 1

    def _print_test_gir_diagnostics(self) -> None:
        if not hasattr(self, "_test_gir_diagnostic_sums"):
            return

        keys = sorted(self._test_gir_diagnostic_sums)
        values = [self._test_gir_diagnostic_sums[key] for key in keys]
        values.append(float(self._test_gir_diagnostic_count))
        totals = torch.tensor(values, device=self.device, dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)

        if self.global_rank != 0:
            return

        count = totals[-1].item()
        averages = {
            key: totals[index].item() / max(count, 1.0)
            for index, key in enumerate(keys)
        }
        print("\n[GIR TEST SUMMARY]")
        print(f"scenes={int(count)}")
        for key in (
            "map_gaussians",
            "map_opacity_mean",
            "map_opacity_mass",
            "test_prune_threshold",
            "test_pruned_new_ratio",
            "test_map_reduction_ratio",
            "top1_ownership_mean",
            "top1_ownership_above_0_1_ratio",
            "top1_ownership_above_0_25_ratio",
            "top1_ownership_above_0_5_ratio",
            "top1_confidence_mean",
            "top1_confidence_mode_code",
            "top1_confidence_floor",
            "old_delete_target_ratio",
            "old_delete_threshold",
            "old_delete_candidate_probability",
            "old_delete_candidate_count",
            "old_delete_candidate_rate",
            "old_delete_hard_ratio",
            "old_delete_removed_opacity_mass_ratio",
            "old_delete_map_count_before",
            "old_delete_map_count_after",
            "new_opacity_mass_ratio",
            "add_gate",
            "add_target",
            "add_gate_supported",
            "add_gate_unsupported",
            "add_gate_below_0_1_ratio",
            "add_gate_below_0_2_ratio",
            "effective_new_above_0_001_ratio",
            "effective_new_above_0_003_ratio",
            "effective_new_above_0_005_ratio",
            "effective_new_above_0_01_ratio",
            "map_above_0_001_ratio",
            "map_above_0_003_ratio",
            "map_above_0_005_ratio",
            "map_above_0_01_ratio",
        ):
            if key in averages:
                if key == "map_gaussians":
                    print(f"{key}={averages[key]:.1f}")
                else:
                    print(f"{key}={averages[key]:.6f}")

    def print_preview_metrics(
        self,
        metrics: dict[str, float | Tensor],
        methods: list[str] | None = None,
        overlap_tag: str | None = None,
    ) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {
                    k: ((s * v) + metrics[k]) / (s + 1)
                    for k, v in self.running_metrics_sub[overlap_tag].items()
                }
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ["ours"]

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        
        
#     def configure_optimizers(self):

#             # 1. 参数分组
#             backbone_params = []
#             head_params = []

#             for name, param in self.named_parameters():
#                 if not param.requires_grad:
#                     continue
#                 # 根据你的模块命名习惯区分
#                 if "gs_head" in name or "output_conv2" in name or "input_merger" in name:
#                     head_params.append(param)
#                 else:
#                     backbone_params.append(param)

#             # 初始学习率设置
#             param_dicts = [
#                 {"params": backbone_params, "lr": 1e-4, "name": "backbone"},
#                 {"params": head_params, "lr": 2e-4, "name": "head"}
#             ]

#             optimizer = torch.optim.AdamW(
#                 param_dicts, weight_decay=0.1, betas=(0.9, 0.95)
#             )

#             # 2. 🌟 绝对解耦的学习率逻辑 🌟
#             phase_1_steps = 50 
#             phase_2_steps = 10000 
#             eta_min_ratio = 0.1

#             # Backbone 逻辑：前 1w 步 Cosine 衰减，后 1w 步彻底消失 (0.0)
#             def lr_lambda_backbone(current_step):
#                 if current_step < phase_1_steps:
#                     # 0 -> 10k: 从 1.0 衰减到 0.1
#                     progress = float(current_step) / float(phase_1_steps)
#                     return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))
#                 else:
#                     # 🌟 10k 之后彻底关闭 Backbone 训练
#                     return 0.0

#             # Head 逻辑：前 1w 步彻底冻结 (0.0)，后 1w 步重新开启 Cosine 衰减
#             def lr_lambda_head(current_step):
#                 if current_step < phase_1_steps:
#                     # 🌟 0 -> 10k: 彻底不练 Head
#                     return 0.0
#                 else:
#                     # 10k -> 20k: 重新算进度 (从 0.0 衰减到 0.1)
#                     phase_2_progress = float(current_step - phase_1_steps) / float(phase_2_steps)
#                     phase_2_progress = min(1.0, phase_2_progress)
#                     return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * phase_2_progress))

#             lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#                 optimizer,
#                 lr_lambda=[lr_lambda_backbone, lr_lambda_head] 
#             )

#             return {
#                 "optimizer": optimizer,
#                 "lr_scheduler": {
#                     "scheduler": lr_scheduler,
#                     "interval": "step",
#                     "frequency": 1,
#                 },
#             }
    def configure_optimizers(self):
        pretrained_lr = (
            self.optimizer_cfg.pretrained_lr
            if self.optimizer_cfg.pretrained_lr is not None
            else self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier
        )
        scratch_lr = (
            self.optimizer_cfg.scratch_lr
            if self.optimizer_cfg.scratch_lr is not None
            else self.optimizer_cfg.lr
        )

        base_weight_decay = float(self.optimizer_cfg.base_weight_decay)
        gir_weight_decay = (
            base_weight_decay
            if self.optimizer_cfg.gir_weight_decay is None
            else float(self.optimizer_cfg.gir_weight_decay)
        )
        group_specs = {
            "gaussian_param_head": (
                self.optimizer_cfg.gaussian_param_head_lr or pretrained_lr,
                base_weight_decay,
            ),
            "gs_head": (
                self.optimizer_cfg.gs_head_lr or pretrained_lr,
                base_weight_decay,
            ),
            "camera": (
                self.optimizer_cfg.camera_lr or pretrained_lr,
                base_weight_decay,
            ),
            "depth": (
                self.optimizer_cfg.depth_lr or pretrained_lr,
                base_weight_decay,
            ),
            "gir": (
                self.optimizer_cfg.gir_lr or scratch_lr,
                gir_weight_decay,
            ),
            "scratch": (scratch_lr, base_weight_decay),
            "other": (self.optimizer_cfg.lr, base_weight_decay),
        }
        grouped_params = {name: [] for name in group_specs}
        grouped_names = {name: [] for name in group_specs}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "model.gir_" in name:
                group = "gir"
            elif "model.encoder.gs_head" in name:
                group = "gs_head"
            elif "model.encoder.camera_head" in name:
                group = "camera"
            elif "model.encoder.gaussian_param_head.scratch.output_conv2" in name:
                group = "depth"
            elif "model.encoder.gaussian_param_head" in name:
                group = "gaussian_param_head"
            elif (
                "model.encoder.depth_head" in name
            ):
                group = "depth"
            elif "model.encoder.depth_refiner" in name:
                group = "scratch"
            else:
                group = "other"
            grouped_params[group].append(param)
            grouped_names[group].append(name)

        param_dicts = []
        for name, (learning_rate, weight_decay) in group_specs.items():
            if grouped_params[name]:
                param_dicts.append(
                    {
                        "params": grouped_params[name],
                        "lr": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "name": name,
                    }
                )
        if not param_dicts:
            raise RuntimeError("No trainable parameters were selected.")

        if self.global_rank == 0:
            summary = ", ".join(
                f"{group['name']}={len(grouped_names[group['name']])} params "
                f"@ {group['lr']} wd={group['weight_decay']}"
                for group in param_dicts
            )
            print(f"Optimizer parameter groups: {summary}")

        optimizer = torch.optim.AdamW(
            param_dicts,
            betas=(float(self.optimizer_cfg.beta1), float(self.optimizer_cfg.beta2)),
        )

        scheduler_mode = str(self.optimizer_cfg.scheduler_mode).lower()
        if scheduler_mode == "constant":
            return optimizer
        if scheduler_mode != "cosine":
            raise ValueError(
                "optimizer.scheduler_mode must be cosine or constant, got "
                f"{scheduler_mode!r}."
            )
        
        max_steps = get_cfg()["trainer"]["max_steps"]
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        eta_min_ratio = 0.1 

        def lr_lambda_main(current_step):
            # 1. Warmup 阶段
            if warm_up_steps > 0 and current_step < warm_up_steps:
                return float(current_step) / float(max(1, warm_up_steps))
            
            # 2. Cosine 衰减阶段
            decay_steps = max_steps - warm_up_steps
            if decay_steps <= 0:
                return 1.0 # 防止总步数小于预热步数的异常情况
                
            current_decay_step = current_step - warm_up_steps
            progress = float(current_decay_step) / float(decay_steps)
            progress = min(1.0, max(0.0, progress)) # 确保在 0~1 之间
            
            return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lr_lambda_main] * len(param_dicts)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

#     def configure_optimizers(self):
#         new_params, new_param_names = [], []
#         delayed_params, delayed_param_names = [], [] 
        
#         for name, param in self.named_parameters():
#             if not param.requires_grad:
#                 continue

#             if "gaussian_param_head.scratch.output_conv2" in name:
#                 new_params.append(param)
#                 new_param_names.append(name)
#             elif "gaussian_param_head" in name:
#                 delayed_params.append(param)
#                 delayed_param_names.append(name)
#             else:
#                 new_params.append(param)
#                 new_param_names.append(name)
                
#         param_dicts = [
#             {
#                 "params": new_params,
#                 "lr": self.optimizer_cfg.lr, 
#                 "name": "new_params"
#             },
#             {
#                 "params": delayed_params,
#                 "lr": self.optimizer_cfg.lr,  
#                 "name": "delayed_params"
#             },
#         ]
        
#         optimizer = torch.optim.AdamW(
#             param_dicts, weight_decay=0.1, betas=(0.9, 0.95)
#         )
        
#         max_steps = get_cfg()["trainer"]["max_steps"]
#         warm_up_steps = self.optimizer_cfg.warm_up_steps
#         delay_steps = 5000  
#         eta_min_ratio = 0.1 

#         def lr_lambda_main(current_step):
#             # 1. Warmup 阶段
#             if warm_up_steps > 0 and current_step < warm_up_steps:
#                 return float(current_step) / float(max(1, warm_up_steps))
            
#             # 2. Cosine 衰减阶段
#             decay_steps = max_steps - warm_up_steps
#             if decay_steps <= 0:
#                 return 1.0 # 防止总步数小于预热步数的异常情况
                
#             current_decay_step = current_step - warm_up_steps
#             progress = float(current_decay_step) / float(decay_steps)
#             progress = min(1.0, max(0.0, progress)) # 确保在 0~1 之间
            
#             # 余弦退火公式
#             return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

#         # ⚠️ 关键点 3：定义延迟参数的学习率曲线
#         def lr_lambda_delayed(current_step):
#             if current_step < delay_steps:
#                 return 0.0  # 前 5000 步，强制乘子为 0，学习率绝对为 0
#             else:
#                 # 5000 步之后，直接调用主参数的函数！
#                 # 这样就能保证苏醒时的 LR 和衰减轨迹与主参数一模一样
#                 return lr_lambda_main(current_step)

#         # ⚠️ 关键点 4：将两个规则分别应用到两个参数组
#         lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#             optimizer,
#             lr_lambda=[lr_lambda_main, lr_lambda_delayed] # 顺序必须和 param_dicts 一一对应
#         )

#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": lr_scheduler,
#                 "interval": "step",
#                 "frequency": 1,
#             },
#         }
