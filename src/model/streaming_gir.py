import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder import sh_utils
from src.model.encoder.common.gaussians import (
    build_covariance,
    quaternion_to_matrix,
)
from src.model.types import Gaussians


def _group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def _flatten_harmonics(harmonics: torch.Tensor) -> torch.Tensor:
    return harmonics.reshape(*harmonics.shape[:2], -1)


def _restore_harmonics(flat: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return flat.reshape(reference.shape)


def _quat_multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.cross(left_xyz, right_xyz, dim=-1)
    )
    w = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    return torch.cat([xyz, w], dim=-1)


def _axis_angle_to_quaternion_xyzw(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > 1e-6,
        torch.sin(half_angle) / angle.clamp_min(1e-8),
        0.5 - angle.square() / 48.0,
    )
    return torch.cat([axis_angle * scale, torch.cos(half_angle)], dim=-1)


@dataclass
class DominantGIR:
    indices: torch.Tensor
    stable_ids: torch.Tensor
    valid: torch.Tensor
    rgb: torch.Tensor
    depth: torch.Tensor
    opacity: torch.Tensor
    scale: torch.Tensor
    observation_count: torch.Tensor
    raster_depth: torch.Tensor
    raster_alpha: torch.Tensor
    dominant_weight: torch.Tensor
    contributor_ids: torch.Tensor | None = None
    contributor_weights: torch.Tensor | None = None
    contributor_depth: torch.Tensor | None = None
    contributor_opacity: torch.Tensor | None = None
    contributor_scale: torch.Tensor | None = None
    contributor_observation_count: torch.Tensor | None = None

    @classmethod
    def empty(
        cls,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "DominantGIR":
        image_shape = (batch_size, 1, height, width)
        return cls(
            indices=torch.full(
                (batch_size, height, width), -1, device=device, dtype=torch.long
            ),
            stable_ids=torch.full(
                (batch_size, height, width), -1, device=device, dtype=torch.long
            ),
            valid=torch.zeros(image_shape, device=device, dtype=torch.bool),
            rgb=torch.zeros((batch_size, 3, height, width), device=device, dtype=dtype),
            depth=torch.zeros(image_shape, device=device, dtype=dtype),
            opacity=torch.zeros(image_shape, device=device, dtype=dtype),
            scale=torch.zeros(image_shape, device=device, dtype=dtype),
            observation_count=torch.zeros(image_shape, device=device, dtype=dtype),
            raster_depth=torch.zeros(image_shape, device=device, dtype=dtype),
            raster_alpha=torch.zeros(image_shape, device=device, dtype=dtype),
            dominant_weight=torch.zeros(image_shape, device=device, dtype=dtype),
        )


@dataclass
class GIRPrediction:
    delta_mean_camera: torch.Tensor
    delta_rotation: torch.Tensor
    delta_log_scale: torch.Tensor
    delta_opacity_logit: torch.Tensor
    delta_harmonics: torch.Tensor
    historical_gate: torch.Tensor
    add_logit: torch.Tensor
    current_delta_mean_camera: torch.Tensor
    current_delta_rotation: torch.Tensor
    current_delta_log_scale: torch.Tensor
    current_delta_opacity_logit: torch.Tensor
    current_delta_harmonics: torch.Tensor
    current_residual_gate: torch.Tensor
    delete_logit: torch.Tensor
    rank_delta_mean_camera: torch.Tensor | None = None
    rank_delta_rotation: torch.Tensor | None = None
    rank_delta_log_scale: torch.Tensor | None = None
    rank_delta_opacity_logit: torch.Tensor | None = None
    rank_delta_harmonics: torch.Tensor | None = None
    rank_historical_gate: torch.Tensor | None = None
    rank_decay_logits: torch.Tensor | None = None


@dataclass
class ProjectedGSEvidence:
    descriptor: torch.Tensor
    flat_candidate_indices: torch.Tensor
    candidate_mask: torch.Tensor
    surface_consistent: torch.Tensor
    old_in_front: torch.Tensor
    projected_radius: torch.Tensor
    relative_depth: torch.Tensor
    camera_depth: torch.Tensor
    opacity_gradient_features: torch.Tensor | None = None
    opacity_gradient_signal: torch.Tensor | None = None
    opacity_gradient_current_signal: torch.Tensor | None = None
    opacity_gradient_replay_signal: torch.Tensor | None = None
    mean_gradient_features: torch.Tensor | None = None
    mean_gradient_signal: torch.Tensor | None = None
    mean_gradient_current_signal: torch.Tensor | None = None
    mean_gradient_replay_signal: torch.Tensor | None = None


@dataclass
class ProjectedGSResidualPrediction:
    delta_mean_camera: torch.Tensor
    delta_rotation: torch.Tensor
    delta_log_scale: torch.Tensor
    delta_opacity_logit: torch.Tensor
    delta_harmonics: torch.Tensor
    front_decay_logit: torch.Tensor
    update_gate_logit: torch.Tensor


@dataclass
class ProjectedNewGSEvidence:
    descriptor: torch.Tensor
    flat_candidate_indices: torch.Tensor
    candidate_mask: torch.Tensor
    history_valid: torch.Tensor
    history_consistent: torch.Tensor
    history_in_front: torch.Tensor
    camera_depth: torch.Tensor
    opacity_gradient_features: torch.Tensor | None = None
    opacity_gradient_signal: torch.Tensor | None = None
    opacity_gradient_current_signal: torch.Tensor | None = None
    opacity_gradient_replay_signal: torch.Tensor | None = None
    mean_gradient_features: torch.Tensor | None = None
    mean_gradient_signal: torch.Tensor | None = None
    mean_gradient_current_signal: torch.Tensor | None = None
    mean_gradient_replay_signal: torch.Tensor | None = None


@dataclass
class ProjectedNewGSResidualPrediction:
    delta_mean_camera: torch.Tensor
    delta_rotation: torch.Tensor
    delta_log_scale: torch.Tensor
    delta_opacity_logit: torch.Tensor
    delta_harmonics: torch.Tensor
    update_gate_logit: torch.Tensor


PROJECTED_GS_STATE_DIM = 30
OPACITY_GRADIENT_FEATURE_DIM = 8
MEAN_GRADIENT_FEATURE_DIM = 10
OPACITY_GRADIENT_STAT_KEYS = (
    "gradient_current_positive_mean",
    "gradient_current_negative_mean",
    "gradient_current_nonzero_ratio",
    "gradient_replay_positive_mean",
    "gradient_replay_negative_mean",
    "gradient_replay_nonzero_mean",
    "gradient_replay_positive_ratio",
    "gradient_replay_negative_ratio",
    "gradient_signal_abs_mean",
    "gradient_signal_positive_ratio",
    "gradient_update_alignment",
)
OLD_OPACITY_GRADIENT_EFFECTIVE_STAT_KEYS = (
    "gradient_consistent_alignment",
    "gradient_front_alignment",
    "gradient_effective_alignment",
    "gradient_current_effective_alignment",
    "gradient_replay_effective_alignment",
    "gradient_current_replay_conflict_ratio",
)
MEAN_GRADIENT_STAT_KEYS = (
    "mean_gradient_current_norm_mean",
    "mean_gradient_replay_norm_mean",
    "mean_gradient_signal_norm_mean",
    "mean_gradient_current_replay_cosine",
    "mean_gradient_current_replay_conflict_ratio",
    "mean_gradient_raw_update_alignment",
    "mean_gradient_effective_update_alignment",
)


@dataclass
class OpacityGradientEvidence:
    features: torch.Tensor
    signal: torch.Tensor
    current_signal: torch.Tensor
    replay_signal: torch.Tensor


@dataclass
class MeanGradientEvidence:
    features: torch.Tensor
    signal: torch.Tensor
    current_signal: torch.Tensor
    replay_signal: torch.Tensor


@dataclass
class HarmonicsGradientEvidence:
    signal: torch.Tensor
    current_signal: torch.Tensor
    replay_signal: torch.Tensor


@dataclass
class SymmetricMultiviewGradientEvidence:
    """Permutation-invariant per-GS statistics over equally weighted views."""

    features: torch.Tensor
    active_mask: torch.Tensor
    visible_view_count: torch.Tensor
    mean_signal: torch.Tensor
    opacity_signal: torch.Tensor
    harmonics_signal: torch.Tensor
    mean_coherence: torch.Tensor
    mean_conflict_ratio: torch.Tensor
    visual_features: torch.Tensor | None = None
    visual_view_count: torch.Tensor | None = None
    visual_weight_sum: torch.Tensor | None = None


@dataclass
class SymmetricGSResidualPrediction:
    delta_mean_reference: torch.Tensor
    delta_rotation: torch.Tensor
    delta_log_scale: torch.Tensor
    delta_opacity_logit: torch.Tensor
    delta_harmonics: torch.Tensor
    update_gate_logits: torch.Tensor


SYMMETRIC_GRADIENT_SCALAR_FEATURE_DIM = 8


class SymmetricVisualEvidenceEncoder(nn.Module):
    """DeltaSplat-style image-space mixer for symmetric GS updates."""

    input_dim = 13  # RGB residual, rendered RGB, Plucker ray, rendered depth.

    def __init__(self, feature_dim: int = 24, hidden_dim: int = 32) -> None:
        super().__init__()
        feature_dim = int(feature_dim)
        hidden_dim = int(hidden_dim)
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Symmetric visual feature dimensions must be positive.")

        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        self.feature_dim = feature_dim
        self.high_resolution = block(self.input_dim, hidden_dim)
        self.low_resolution = block(self.input_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, feature_dim, 1),
        )

    def forward(self, image_context: torch.Tensor) -> torch.Tensor:
        if image_context.ndim != 4 or image_context.shape[1] != self.input_dim:
            raise ValueError(
                "Symmetric visual context must be [B,13,H,W], got "
                f"{tuple(image_context.shape)}."
            )
        high = self.high_resolution(image_context)
        low = self.low_resolution(
            F.avg_pool2d(
                image_context,
                kernel_size=4,
                stride=4,
                ceil_mode=True,
            )
        )
        low = F.interpolate(
            low,
            size=high.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fusion(torch.cat([high, low], dim=1))


def _as_bv1hw(
    value: torch.Tensor,
    *,
    batch: int,
    views: int,
    name: str,
) -> torch.Tensor:
    """Normalize scalar decoder images despite legacy singleton squeezing."""
    if value.ndim == 5 and value.shape[:3] == (batch, views, 1):
        return value
    if value.ndim == 4 and value.shape[:2] == (batch, views):
        return value.unsqueeze(2)
    if views == 1 and value.ndim == 3 and value.shape[0] == batch:
        return value[:, None, None]
    raise ValueError(
        f"Symmetric {name} must represent [B,V,1,H,W], got {tuple(value.shape)}."
    )


def _build_plucker_image_context(
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return bounded world-space Plucker rays as [B,V,6,H,W]."""
    batch, views = camera_to_world.shape[:2]
    dtype = torch.float32
    device = camera_to_world.device
    pixel_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    pixel_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    y, x = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
    x = x.view(1, 1, height, width)
    y = y.view(1, 1, height, width)

    intrinsics = intrinsics.detach().float()
    fx = intrinsics[..., 0, 0, None, None].clamp_min(1e-6)
    fy = intrinsics[..., 1, 1, None, None].clamp_min(1e-6)
    cx = intrinsics[..., 0, 2, None, None]
    cy = intrinsics[..., 1, 2, None, None]
    camera_directions = torch.stack(
        [
            (x.expand(batch, views, -1, -1) - cx) / fx,
            (y.expand(batch, views, -1, -1) - cy) / fy,
            torch.ones((batch, views, height, width), device=device, dtype=dtype),
        ],
        dim=2,
    )
    rotation = camera_to_world.detach().float()[..., :3, :3]
    directions = torch.einsum("bvij,bvjhw->bvihw", rotation, camera_directions)
    directions = F.normalize(directions, dim=2, eps=1e-6)
    origins = camera_to_world.detach().float()[..., :3, 3, None, None]
    origins = origins.expand(-1, -1, -1, height, width)
    moments = torch.tanh(torch.cross(origins, directions, dim=2))
    return torch.cat([directions, moments], dim=2)


def build_symmetric_multiview_visual_evidence(
    gaussians: Gaussians,
    target_rgb: torch.Tensor,
    rendered_rgb: torch.Tensor,
    rendered_depth: torch.Tensor,
    rendered_alpha: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    encoder: SymmetricVisualEvidenceEncoder,
    *,
    active_mask: torch.Tensor | None = None,
    chunk_size: int = 65536,
    image_scale: float = 0.5,
    depth_temperature: float = 0.10,
    min_alpha: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate causal image context onto GSs without temporal view roles."""
    if target_rgb.ndim != 5 or rendered_rgb.ndim != 5:
        raise ValueError("Symmetric RGB context must be [B,V,3,H,W].")
    batch, views, channels, height, width = rendered_rgb.shape
    if channels != 3 or target_rgb.shape[:3] != (batch, views, 3):
        raise ValueError(
            "Symmetric target/rendered RGB shapes do not match: "
            f"target={tuple(target_rgb.shape)}, rendered={tuple(rendered_rgb.shape)}."
        )
    if target_rgb.shape[-2:] != (height, width):
        target_rgb = F.interpolate(
            target_rgb.reshape(batch * views, 3, *target_rgb.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, views, 3, height, width)
    depth = _as_bv1hw(
        rendered_depth, batch=batch, views=views, name="rendered depth"
    )
    alpha = _as_bv1hw(
        rendered_alpha, batch=batch, views=views, name="rendered alpha"
    )
    if camera_to_world.shape != (batch, views, 4, 4):
        raise ValueError("Symmetric visual cameras must be [B,V,4,4].")
    if intrinsics.shape != (batch, views, 3, 3):
        raise ValueError("Symmetric visual intrinsics must be [B,V,3,3].")
    count = gaussians.means.shape[1]
    if active_mask is None:
        active_mask = gaussians.opacities.detach() > 0
    if active_mask.shape != (batch, count):
        raise ValueError("Symmetric visual active mask must be [B,N].")
    if not math.isfinite(float(depth_temperature)) or depth_temperature <= 0:
        raise ValueError("Symmetric visual depth_temperature must be positive.")
    if not math.isfinite(float(image_scale)) or not 0 < image_scale <= 1:
        raise ValueError("Symmetric visual image_scale must be in (0,1].")
    if not math.isfinite(float(min_alpha)) or not 0 <= min_alpha < 1:
        raise ValueError("Symmetric visual min_alpha must be in [0,1).")

    rendered = rendered_rgb.detach().float().clamp(0.0, 1.0)
    target = target_rgb.detach().float().clamp(0.0, 1.0)
    depth = torch.nan_to_num(
        depth.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    alpha = torch.nan_to_num(
        alpha.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0.0, 1.0)
    # DecoderSplattingCUDA uses gsplat's RGB+D mode, whose D channel is
    # alpha-accumulated depth. Convert it to expected depth for visibility.
    depth = torch.where(
        alpha > 1e-6,
        depth / alpha.clamp_min(1e-6),
        torch.zeros_like(depth),
    )
    if float(image_scale) < 1.0:
        scaled_height = max(8, int(round(height * float(image_scale))))
        scaled_width = max(8, int(round(width * float(image_scale))))

        def resize_views(value: torch.Tensor) -> torch.Tensor:
            channels = value.shape[2]
            return F.interpolate(
                value.reshape(batch * views, channels, height, width),
                size=(scaled_height, scaled_width),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, views, channels, scaled_height, scaled_width)

        target = resize_views(target)
        rendered = resize_views(rendered)
        depth = resize_views(depth)
        alpha = resize_views(alpha).clamp(0.0, 1.0)
        height, width = scaled_height, scaled_width
    plucker = _build_plucker_image_context(
        camera_to_world, intrinsics, height, width
    )
    valid_depth = depth > 1e-6
    flat_depth = depth.reshape(batch, views, -1)
    flat_valid = valid_depth.reshape(batch, views, -1)
    inf = torch.full_like(flat_depth, float("inf"))
    sorted_depth = torch.where(flat_valid, flat_depth, inf).sort(dim=-1).values
    valid_count = flat_valid.sum(dim=-1).clamp_min(1)
    median_index = ((valid_count - 1) // 2).unsqueeze(-1)
    depth_median = sorted_depth.gather(-1, median_index).squeeze(-1)
    depth_median = torch.where(
        torch.isfinite(depth_median),
        depth_median.clamp_min(1e-6),
        torch.ones_like(depth_median),
    )
    depth_context = torch.log(
        depth.clamp_min(1e-6) / depth_median[..., None, None, None]
    ).clamp(-6.0, 6.0)
    depth_context = torch.where(
        valid_depth, depth_context, torch.zeros_like(depth_context)
    )
    image_context = torch.cat(
        [target - rendered, rendered, plucker, depth_context], dim=2
    )
    with torch.autocast(device_type=image_context.device.type, enabled=False):
        encoded = encoder(
            image_context.reshape(
                batch * views, encoder.input_dim, height, width
            ).float()
        )

    world_to_camera = torch.linalg.inv(camera_to_world.detach().float())
    intrinsics_detached = intrinsics.detach().float()
    opacity = gaussians.opacities.detach().float().clamp(0.0, 1.0)
    chunk_size = max(1, int(chunk_size))
    candidate_indices = torch.nonzero(
        active_mask.any(dim=0), as_tuple=False
    ).squeeze(1)
    candidate_count = int(candidate_indices.numel())
    feature_chunks = []
    view_count_chunks = []
    weight_sum_chunks = []
    encoded_dim = encoded.shape[1]
    for start in range(0, candidate_count, chunk_size):
        end = min(start + chunk_size, candidate_count)
        chunk_count = end - start
        indices = candidate_indices[start:end]
        means = (
            gaussians.means.index_select(1, indices).detach().float()
        )
        ones = torch.ones((batch, chunk_count, 1), device=means.device)
        homogeneous = torch.cat([means, ones], dim=-1)
        camera_points = torch.einsum(
            "bvij,bnj->bvni", world_to_camera, homogeneous
        )[..., :3]
        z = camera_points[..., 2]
        z_safe = z.clamp_min(1e-6)
        u = (
            intrinsics_detached[..., 0, 0, None] * camera_points[..., 0] / z_safe
            + intrinsics_detached[..., 0, 2, None]
        )
        v = (
            intrinsics_detached[..., 1, 1, None] * camera_points[..., 1] / z_safe
            + intrinsics_detached[..., 1, 2, None]
        )
        grid = torch.stack([2.0 * u - 1.0, 2.0 * v - 1.0], dim=-1)
        flat_grid = grid.reshape(batch * views, chunk_count, 1, 2)
        sampled_features = F.grid_sample(
            encoded,
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(-1).transpose(1, 2).reshape(
            batch, views, chunk_count, encoded_dim
        )
        sampled_depth = F.grid_sample(
            depth.reshape(batch * views, 1, height, width),
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(batch, views, chunk_count)
        sampled_alpha = F.grid_sample(
            alpha.reshape(batch * views, 1, height, width),
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(batch, views, chunk_count)
        relative_depth_error = (
            (z - sampled_depth).abs() / sampled_depth.clamp_min(1e-4)
        )
        in_frustum = (
            (z > 1e-5)
            & (u >= 0.0)
            & (u <= 1.0)
            & (v >= 0.0)
            & (v <= 1.0)
            & (sampled_depth > 1e-5)
            & (sampled_alpha >= float(min_alpha))
            & active_mask.index_select(1, indices)[:, None]
        )
        weights = (
            opacity.index_select(1, indices)[:, None]
            * sampled_alpha
            * torch.exp(-relative_depth_error / float(depth_temperature))
            * in_frustum.to(sampled_alpha.dtype)
        ).detach()
        weight_sum = weights.sum(dim=1)
        aggregated = (
            (sampled_features * weights.unsqueeze(-1)).sum(dim=1)
            / weight_sum.clamp_min(1e-8).unsqueeze(-1)
        )
        visible_count = (weights > 1e-6).sum(dim=1).to(aggregated.dtype)
        confidence = torch.stack(
            [
                torch.log1p(weight_sum),
                visible_count / float(views),
                weights.amax(dim=1),
            ],
            dim=-1,
        )
        feature_chunks.append(torch.cat([aggregated, confidence], dim=-1))
        view_count_chunks.append(visible_count)
        weight_sum_chunks.append(weight_sum)

    if candidate_count == 0:
        features = encoded.new_zeros((batch, 0, encoder.feature_dim + 3))
        view_count = encoded.new_zeros((batch, 0))
        weight_sum = encoded.new_zeros((batch, 0))
    else:
        features = torch.cat(feature_chunks, dim=1)
        view_count = torch.cat(view_count_chunks, dim=1)
        weight_sum = torch.cat(weight_sum_chunks, dim=1)
    if candidate_count != count:
        features = features.new_zeros(
            (batch, count, encoder.feature_dim + 3)
        ).index_copy(1, candidate_indices, features)
        view_count = view_count.new_zeros((batch, count)).index_copy(
            1, candidate_indices, view_count
        )
        weight_sum = weight_sum.new_zeros((batch, count)).index_copy(
            1, candidate_indices, weight_sum
        )
    return features, view_count, weight_sum


def attach_symmetric_multiview_visual_evidence(
    evidence: SymmetricMultiviewGradientEvidence,
    visual_features: torch.Tensor,
    visual_view_count: torch.Tensor,
    visual_weight_sum: torch.Tensor,
) -> SymmetricMultiviewGradientEvidence:
    """Attach visual context while preserving permutation-invariant GS routing."""
    batch, count = evidence.active_mask.shape
    if visual_features.shape[:2] != (batch, count):
        raise ValueError("Symmetric visual features must start with [B,N].")
    for name, value in (
        ("visual_view_count", visual_view_count),
        ("visual_weight_sum", visual_weight_sum),
    ):
        if value.shape != (batch, count):
            raise ValueError(f"Symmetric {name} must be [B,N].")
    visual_active = visual_view_count > 0
    return SymmetricMultiviewGradientEvidence(
        features=evidence.features,
        active_mask=evidence.active_mask | visual_active,
        visible_view_count=torch.maximum(
            evidence.visible_view_count, visual_view_count
        ),
        mean_signal=evidence.mean_signal,
        opacity_signal=evidence.opacity_signal,
        harmonics_signal=evidence.harmonics_signal,
        mean_coherence=evidence.mean_coherence,
        mean_conflict_ratio=evidence.mean_conflict_ratio,
        visual_features=visual_features,
        visual_view_count=visual_view_count,
        visual_weight_sum=visual_weight_sum,
    )


def _symmetric_gradient_statistics(
    gradients: torch.Tensor,
    *,
    active_mask: torch.Tensor,
    gradient_scale: float,
    gradient_clip: float,
    gradient_threshold: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Summarize [B,V,N,...] gradients without assigning temporal roles."""
    if gradients.ndim < 4:
        raise ValueError(
            "Symmetric gradient statistics require [B,V,N,...], got "
            f"{tuple(gradients.shape)}."
        )
    batch, views, count = gradients.shape[:3]
    if views < 1:
        raise ValueError("Symmetric gradient evidence requires at least one view.")
    if active_mask.shape != (batch, count):
        raise ValueError(
            "Symmetric gradient active mask must be [B,N], got "
            f"{tuple(active_mask.shape)}."
        )
    for name, value in (
        ("gradient_scale", gradient_scale),
        ("gradient_clip", gradient_clip),
        ("gradient_threshold", gradient_threshold),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"Symmetric {name} must be finite and non-negative, got {value}."
            )

    feature_size = math.prod(gradients.shape[3:])
    with torch.no_grad():
        clip = abs(float(gradient_clip))
        scaled = torch.nan_to_num(
            gradients.detach().float().reshape(batch, views, count, feature_size)
            * float(gradient_scale),
            nan=0.0,
            posinf=clip,
            neginf=-clip,
        ).clamp(-clip, clip)
        norms = scaled.norm(dim=-1)
        visible = (
            (norms > float(gradient_threshold))
            & active_mask[:, None]
        )
        visible_float = visible.to(scaled.dtype)
        denominator = visible_float.sum(dim=1).clamp_min(1.0)
        masked = scaled * visible_float.unsqueeze(-1)
        mean = masked.sum(dim=1) / denominator.unsqueeze(-1)
        rms = (
            masked.square().sum(dim=1) / denominator.unsqueeze(-1)
        ).clamp_min(0.0).sqrt()
        std = (rms.square() - mean.square()).clamp_min(0.0).sqrt()

        unit = scaled / norms.clamp_min(1e-12).unsqueeze(-1)
        coherence = (
            (unit * visible_float.unsqueeze(-1)).sum(dim=1).norm(dim=-1)
            / denominator
        ).clamp(0.0, 1.0)
        consensus_norm = mean.norm(dim=-1)
        dot = (scaled * mean.unsqueeze(1)).sum(dim=-1)
        conflict = (
            ((dot < 0.0) & visible).to(scaled.dtype).sum(dim=1)
            / denominator
        )
        conflict = torch.where(
            consensus_norm > 1e-12,
            conflict,
            torch.zeros_like(conflict),
        )
        visible_count = visible_float.sum(dim=1)
        visible_ratio = visible_count / float(views)
        mean_norm = (
            norms.mul(visible_float).sum(dim=1) / denominator
        )
        max_norm = norms.masked_fill(~visible, 0.0).amax(dim=1)
        features = torch.cat(
            [
                mean,
                rms,
                std,
                torch.log1p(mean_norm).unsqueeze(-1),
                torch.log1p(max_norm).unsqueeze(-1),
                coherence.unsqueeze(-1),
                conflict.unsqueeze(-1),
                visible_ratio.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = features * active_mask.unsqueeze(-1).to(features.dtype)
        mean = mean * active_mask.unsqueeze(-1).to(mean.dtype)
    return features, mean, visible_count, coherence, conflict


def build_symmetric_multiview_gradient_evidence(
    opacity_logit_gradients: torch.Tensor,
    mean_reference_gradients: torch.Tensor,
    harmonics_gradients: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    opacity_gradient_scale: float = 1000.0,
    opacity_gradient_clip: float = 20.0,
    opacity_gradient_threshold: float = 1e-3,
    mean_gradient_scale: float = 1.0,
    mean_gradient_clip: float = 20.0,
    mean_gradient_threshold: float = 1e-6,
    harmonics_gradient_scale: float = 1.0,
    harmonics_gradient_clip: float = 20.0,
    harmonics_gradient_threshold: float = 1e-6,
) -> SymmetricMultiviewGradientEvidence:
    """Build equal-view evidence for a unified old/new Gaussian update."""
    if opacity_logit_gradients.ndim != 3:
        raise ValueError(
            "Opacity gradients must be [B,V,N], got "
            f"{tuple(opacity_logit_gradients.shape)}."
        )
    batch, _, count = opacity_logit_gradients.shape
    if mean_reference_gradients.shape[:3] != opacity_logit_gradients.shape or (
        mean_reference_gradients.ndim != 4
        or mean_reference_gradients.shape[-1] != 3
    ):
        raise ValueError(
            "Mean gradients must be [B,V,N,3] and match opacity gradients; got "
            f"{tuple(mean_reference_gradients.shape)}."
        )
    if harmonics_gradients.shape[:3] != opacity_logit_gradients.shape:
        raise ValueError(
            "Harmonics gradients must start with [B,V,N], got "
            f"{tuple(harmonics_gradients.shape)}."
        )
    if active_mask is None:
        active_mask = torch.ones(
            (batch, count),
            device=opacity_logit_gradients.device,
            dtype=torch.bool,
        )
    else:
        active_mask = active_mask.bool()

    opacity_features, opacity_signal, opacity_count, _, _ = (
        _symmetric_gradient_statistics(
            opacity_logit_gradients.unsqueeze(-1),
            active_mask=active_mask,
            gradient_scale=opacity_gradient_scale,
            gradient_clip=opacity_gradient_clip,
            gradient_threshold=opacity_gradient_threshold,
        )
    )
    mean_features, mean_signal, mean_count, mean_coherence, mean_conflict = (
        _symmetric_gradient_statistics(
            mean_reference_gradients,
            active_mask=active_mask,
            gradient_scale=mean_gradient_scale,
            gradient_clip=mean_gradient_clip,
            gradient_threshold=mean_gradient_threshold,
        )
    )
    harmonics_features, harmonics_signal_flat, harmonics_count, _, _ = (
        _symmetric_gradient_statistics(
            harmonics_gradients,
            active_mask=active_mask,
            gradient_scale=harmonics_gradient_scale,
            gradient_clip=harmonics_gradient_clip,
            gradient_threshold=harmonics_gradient_threshold,
        )
    )
    visible_count = torch.maximum(
        torch.maximum(opacity_count, mean_count), harmonics_count
    )
    evidence_active = active_mask & (visible_count > 0)
    features = torch.cat(
        [opacity_features, mean_features, harmonics_features], dim=-1
    )
    features = features * evidence_active.unsqueeze(-1).to(features.dtype)
    return SymmetricMultiviewGradientEvidence(
        features=features,
        active_mask=evidence_active,
        visible_view_count=visible_count,
        mean_signal=mean_signal,
        opacity_signal=opacity_signal.squeeze(-1),
        harmonics_signal=harmonics_signal_flat.reshape(
            *harmonics_gradients.shape[:1],
            harmonics_gradients.shape[2],
            *harmonics_gradients.shape[3:],
        ),
        mean_coherence=mean_coherence,
        mean_conflict_ratio=mean_conflict,
    )


FUNCTIONAL_GS_UPDATE_STAT_KEYS = (
    "selected_count",
    "selected_ratio",
    "score_mean",
    "mean_gradient_norm",
    "mean_step_norm",
    "opacity_gradient_abs",
    "opacity_logit_step_abs",
    "harmonics_gradient_norm",
    "harmonics_step_norm",
)


def gate_front_conflict_decay(
    front_conflict_mask: torch.Tensor,
    opacity_gradient_evidence: OpacityGradientEvidence | None,
    *,
    enabled: bool,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate deterministic front-conflict decay with the combined RGB gradient."""
    if front_conflict_mask.dtype != torch.bool:
        raise TypeError(
            "Front-conflict decay mask must be boolean, got "
            f"{front_conflict_mask.dtype}."
        )
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "Front-conflict opacity-gradient threshold must be finite and "
            "non-negative, got "
            f"{threshold}."
        )

    gated_mask = front_conflict_mask
    if enabled and opacity_gradient_evidence is not None:
        signal = opacity_gradient_evidence.signal
        if signal.shape != front_conflict_mask.shape:
            raise ValueError(
                "Front-conflict mask and opacity-gradient signal must have the "
                "same shape, got "
                f"{tuple(front_conflict_mask.shape)} and {tuple(signal.shape)}."
            )
        # Positive dL/d(logit(opacity)) means gradient descent lowers opacity.
        gated_mask = front_conflict_mask & (signal.detach() > threshold)

    conflict_count = front_conflict_mask.float().sum()
    gate_ratio = torch.where(
        conflict_count > 0,
        gated_mask.float().sum() / conflict_count.clamp_min(1.0),
        conflict_count,
    )
    return gated_mask, gate_ratio


def build_opacity_gradient_evidence(
    opacity_logit_gradients: torch.Tensor,
    *,
    gradient_scale: float = 1000.0,
    gradient_clip: float = 20.0,
    gradient_threshold: float = 1e-3,
    current_weight: float = 1.0,
    replay_weight: float = 0.5,
    active_mask: torch.Tensor | None = None,
) -> OpacityGradientEvidence:
    """Convert current/replay opacity gradients into detached RetroSplat evidence."""
    if opacity_logit_gradients.ndim != 3:
        raise ValueError(
            "Opacity gradients must be [B,V,N], got "
            f"{tuple(opacity_logit_gradients.shape)}."
        )
    batch, views, count = opacity_logit_gradients.shape
    if views < 1:
        raise ValueError("Opacity-gradient evidence requires a current view.")
    if active_mask is None:
        active_mask = torch.ones(
            (batch, count),
            device=opacity_logit_gradients.device,
            dtype=torch.bool,
        )
    if active_mask.shape != (batch, count):
        raise ValueError(
            "Opacity-gradient active mask must be [B,N], got "
            f"{tuple(active_mask.shape)}."
        )

    current_weight = float(current_weight)
    replay_weight = float(replay_weight)
    if not math.isfinite(current_weight) or current_weight < 0.0:
        raise ValueError(
            "Opacity-gradient current_weight must be finite and non-negative, "
            f"got {current_weight}."
        )
    if not math.isfinite(replay_weight) or replay_weight < 0.0:
        raise ValueError(
            "Opacity-gradient replay_weight must be finite and non-negative, "
            f"got {replay_weight}."
        )

    with torch.no_grad():
        scaled = torch.nan_to_num(
            opacity_logit_gradients.detach().float()
            * max(0.0, float(gradient_scale)),
            nan=0.0,
            posinf=float(gradient_clip),
            neginf=-float(gradient_clip),
        ).clamp(-abs(float(gradient_clip)), abs(float(gradient_clip)))
        scaled = torch.where(
            scaled.abs() > max(0.0, float(gradient_threshold)),
            scaled,
            torch.zeros_like(scaled),
        )
        scaled = scaled * active_mask[:, None].to(scaled.dtype)
        current = scaled[:, 0]
        replay = scaled[:, 1:]
        current_signal = current * current_weight
        current_positive = current_signal.clamp_min(0.0)
        current_negative = (-current_signal).clamp_min(0.0)
        if replay.shape[1]:
            weighted_replay = replay * replay_weight
            replay_positive = weighted_replay.clamp_min(0.0).mean(dim=1)
            replay_negative = (-weighted_replay).clamp_min(0.0).mean(dim=1)
            replay_nonzero = replay != 0
            replay_nonzero_count = replay_nonzero.sum(dim=1).float()
            denominator = replay_nonzero_count.clamp_min(1.0)
            replay_positive_ratio = (replay > 0).sum(dim=1).float() / denominator
            replay_negative_ratio = (replay < 0).sum(dim=1).float() / denominator
            replay_signal = weighted_replay.mean(dim=1)
        else:
            replay_positive = torch.zeros_like(current)
            replay_negative = torch.zeros_like(current)
            replay_nonzero_count = torch.zeros_like(current)
            replay_positive_ratio = torch.zeros_like(current)
            replay_negative_ratio = torch.zeros_like(current)
            replay_signal = torch.zeros_like(current)

        transform = lambda value: torch.log1p(value.clamp_min(0.0)) / 5.0
        features = torch.stack(
            [
                transform(current_positive),
                transform(current_negative),
                (current != 0).float(),
                transform(replay_positive),
                transform(replay_negative),
                transform(replay_nonzero_count),
                replay_positive_ratio,
                replay_negative_ratio,
            ],
            dim=-1,
        )
        features = features * active_mask.unsqueeze(-1).to(features.dtype)
        signal = (current_signal + replay_signal) * active_mask.to(current.dtype)
    return OpacityGradientEvidence(
        features=features,
        signal=signal,
        current_signal=current_signal,
        replay_signal=replay_signal,
    )


def build_mean_gradient_evidence(
    mean_camera_gradients: torch.Tensor,
    *,
    gradient_scale: float = 1.0,
    gradient_clip: float = 20.0,
    gradient_threshold: float = 1e-6,
    current_weight: float = 1.0,
    replay_weight: float = 0.5,
    active_mask: torch.Tensor | None = None,
) -> MeanGradientEvidence:
    """Pack current/replay RGB gradients for camera-space mean residuals."""
    if mean_camera_gradients.ndim != 4 or mean_camera_gradients.shape[-1] != 3:
        raise ValueError(
            "Mean gradients must be [B,V,N,3], got "
            f"{tuple(mean_camera_gradients.shape)}."
        )
    batch, views, count, _ = mean_camera_gradients.shape
    if views < 1:
        raise ValueError("Mean-gradient evidence requires a current view.")
    if active_mask is None:
        active_mask = torch.ones(
            (batch, count),
            device=mean_camera_gradients.device,
            dtype=torch.bool,
        )
    if active_mask.shape != (batch, count):
        raise ValueError(
            "Mean-gradient active mask must be [B,N], got "
            f"{tuple(active_mask.shape)}."
        )
    for name, value in (
        ("current_weight", current_weight),
        ("replay_weight", replay_weight),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"Mean-gradient {name} must be finite and non-negative, got "
                f"{value}."
            )

    with torch.no_grad():
        clip = abs(float(gradient_clip))
        scaled = torch.nan_to_num(
            mean_camera_gradients.detach().float()
            * max(0.0, float(gradient_scale)),
            nan=0.0,
            posinf=clip,
            neginf=-clip,
        ).clamp(-clip, clip)
        norm = scaled.norm(dim=-1)
        scaled = torch.where(
            (norm > max(0.0, float(gradient_threshold))).unsqueeze(-1),
            scaled,
            torch.zeros_like(scaled),
        )
        scaled = scaled * active_mask[:, None, :, None].to(scaled.dtype)
        current_signal = scaled[:, 0] * float(current_weight)
        replay = scaled[:, 1:]
        if replay.shape[1]:
            weighted_replay = replay * float(replay_weight)
            replay_signal = weighted_replay.mean(dim=1)
            replay_nonzero_ratio = (replay.norm(dim=-1) > 0).float().mean(dim=1)
        else:
            replay_signal = torch.zeros_like(current_signal)
            replay_nonzero_ratio = torch.zeros_like(current_signal[..., 0])
        signal = current_signal + replay_signal

        current_norm = current_signal.norm(dim=-1)
        replay_norm = replay_signal.norm(dim=-1)
        current_direction = F.normalize(current_signal, dim=-1, eps=1e-8)
        replay_direction = F.normalize(replay_signal, dim=-1, eps=1e-8)
        both = (current_norm > 0) & (replay_norm > 0)
        cosine = (current_direction * replay_direction).sum(dim=-1)
        cosine = torch.where(both, cosine, torch.zeros_like(cosine))
        transform = lambda value: torch.log1p(value.clamp_min(0.0)) / 5.0
        features = torch.cat(
            [
                current_direction,
                transform(current_norm).unsqueeze(-1),
                replay_direction,
                transform(replay_norm).unsqueeze(-1),
                cosine.unsqueeze(-1),
                replay_nonzero_ratio.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = features * active_mask.unsqueeze(-1).to(features.dtype)
    return MeanGradientEvidence(
        features=features,
        signal=signal,
        current_signal=current_signal,
        replay_signal=replay_signal,
    )


def build_harmonics_gradient_evidence(
    harmonics_gradients: torch.Tensor,
    *,
    gradient_scale: float = 1.0,
    gradient_clip: float = 20.0,
    gradient_threshold: float = 1e-6,
    current_weight: float = 1.0,
    replay_weight: float = 0.5,
    active_mask: torch.Tensor | None = None,
) -> HarmonicsGradientEvidence:
    """Combine current/replay RGB gradients for raw SH coefficients."""
    if harmonics_gradients.ndim < 5:
        raise ValueError(
            "Harmonics gradients must be [B,V,N,...], got "
            f"{tuple(harmonics_gradients.shape)}."
        )
    batch, views, count = harmonics_gradients.shape[:3]
    if views < 1:
        raise ValueError("Harmonics-gradient evidence requires a current view.")
    if active_mask is None:
        active_mask = torch.ones(
            (batch, count),
            device=harmonics_gradients.device,
            dtype=torch.bool,
        )
    if active_mask.shape != (batch, count):
        raise ValueError(
            "Harmonics-gradient active mask must be [B,N], got "
            f"{tuple(active_mask.shape)}."
        )
    for name, value in (
        ("current_weight", current_weight),
        ("replay_weight", replay_weight),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"Harmonics-gradient {name} must be finite and non-negative, "
                f"got {value}."
            )

    with torch.no_grad():
        clip = abs(float(gradient_clip))
        scaled = torch.nan_to_num(
            harmonics_gradients.detach().float()
            * max(0.0, float(gradient_scale)),
            nan=0.0,
            posinf=clip,
            neginf=-clip,
        ).clamp(-clip, clip)
        flat = scaled.flatten(3)
        norm = flat.norm(dim=-1)
        nonzero = norm > max(0.0, float(gradient_threshold))
        for _ in range(scaled.ndim - nonzero.ndim):
            nonzero = nonzero.unsqueeze(-1)
        scaled = torch.where(nonzero, scaled, torch.zeros_like(scaled))
        active = active_mask[:, None]
        for _ in range(scaled.ndim - active.ndim):
            active = active.unsqueeze(-1)
        scaled = scaled * active.to(scaled.dtype)
        current_signal = scaled[:, 0] * float(current_weight)
        replay = scaled[:, 1:]
        replay_signal = (
            replay.mean(dim=1) * float(replay_weight)
            if replay.shape[1]
            else torch.zeros_like(current_signal)
        )
        signal = current_signal + replay_signal
    return HarmonicsGradientEvidence(
        signal=signal,
        current_signal=current_signal,
        replay_signal=replay_signal,
    )


def select_current_and_recent_replay_gradients(
    gradients: torch.Tensor,
    replay_views: int,
) -> torch.Tensor:
    """Select the current view and a causal suffix from a joint gradient probe."""
    if gradients.ndim < 2 or gradients.shape[1] < 1:
        raise ValueError(
            "Gradient probes must have a non-empty view dimension at index 1, "
            f"got {tuple(gradients.shape)}."
        )
    replay_views = int(replay_views)
    if replay_views < -1:
        raise ValueError(
            "replay_views must be -1, 0, or positive; "
            f"got {replay_views}."
        )
    available = gradients.shape[1] - 1
    selected = available if replay_views == -1 else min(replay_views, available)
    if selected == available:
        return gradients
    if selected == 0:
        return gradients[:, :1]
    return torch.cat((gradients[:, :1], gradients[:, -selected:]), dim=1)


def build_provisional_gaussians(
    history: Gaussians,
    current: Gaussians,
    current_opacity_weight: torch.Tensor,
    current_mask: torch.Tensor,
) -> tuple[Gaussians, torch.Tensor]:
    """Compose a detached old+new map while retaining dense new-anchor indexing."""
    batch, current_count = current.opacities.shape
    if history.opacities.shape[0] != batch:
        raise ValueError("Historical and current Gaussians must share a batch size.")
    weight = current_opacity_weight.reshape(batch, current_count).detach().float()
    mask = current_mask.reshape(batch, current_count).detach().bool()
    current_opacity = current.opacities.detach().float() * weight * mask.float()
    provisional = Gaussians(
        means=torch.cat([history.means.detach(), current.means.detach()], dim=1),
        harmonics=torch.cat(
            [history.harmonics.detach(), current.harmonics.detach()], dim=1
        ),
        opacities=torch.cat(
            [history.opacities.detach(), current_opacity], dim=1
        ),
        scales=torch.cat([history.scales.detach(), current.scales.detach()], dim=1),
        rotations=torch.cat(
            [history.rotations.detach(), current.rotations.detach()], dim=1
        ),
    )
    history_active = history.opacities.detach() > 0
    active_mask = torch.cat([history_active, mask & (weight > 0)], dim=1)
    return provisional, active_mask


def _opacity_gradient_update_stats(
    features: torch.Tensor | None,
    signal: torch.Tensor | None,
    delta_opacity_logit: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if features is None or signal is None:
        zero = delta_opacity_logit.new_zeros(())
        return {key: zero for key in OPACITY_GRADIENT_STAT_KEYS}
    if features.shape[-1] != OPACITY_GRADIENT_FEATURE_DIM:
        raise ValueError(
            "Packed opacity-gradient features must have 8 channels, got "
            f"{tuple(features.shape)}."
        )
    if features.shape[0] == 0:
        zero = features.new_zeros(())
        return {key: zero for key in OPACITY_GRADIENT_STAT_KEYS}
    signal = signal.float()
    delta = delta_opacity_logit.float()
    nonzero = signal != 0
    nonzero_count = nonzero.float().sum().clamp_min(1.0)
    return {
        "gradient_current_positive_mean": features[:, 0].float().mean(),
        "gradient_current_negative_mean": features[:, 1].float().mean(),
        "gradient_current_nonzero_ratio": features[:, 2].float().mean(),
        "gradient_replay_positive_mean": features[:, 3].float().mean(),
        "gradient_replay_negative_mean": features[:, 4].float().mean(),
        "gradient_replay_nonzero_mean": features[:, 5].float().mean(),
        "gradient_replay_positive_ratio": features[:, 6].float().mean(),
        "gradient_replay_negative_ratio": features[:, 7].float().mean(),
        "gradient_signal_abs_mean": signal.abs().mean(),
        "gradient_signal_positive_ratio": (
            ((signal > 0).float() * nonzero.float()).sum() / nonzero_count
        ),
        "gradient_update_alignment": (
            (((signal * delta) < 0).float() * nonzero.float()).sum()
            / nonzero_count
        ),
    }


def _mean_gradient_update_stats(
    signal: torch.Tensor | None,
    current_signal: torch.Tensor | None,
    replay_signal: torch.Tensor | None,
    raw_delta_mean_camera: torch.Tensor,
    effective_delta_mean_camera: torch.Tensor,
    update_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    zero = raw_delta_mean_camera.new_zeros(())
    if signal is None or current_signal is None or replay_signal is None:
        return {key: zero for key in MEAN_GRADIENT_STAT_KEYS}
    if signal.shape != raw_delta_mean_camera.shape:
        raise ValueError(
            "Mean-gradient signal and residual must have the same shape, got "
            f"{tuple(signal.shape)} and {tuple(raw_delta_mean_camera.shape)}."
        )
    if signal.shape[0] == 0:
        return {key: zero for key in MEAN_GRADIENT_STAT_KEYS}
    if update_mask is not None and update_mask.shape != signal.shape[:-1]:
        raise ValueError(
            "Mean-gradient update mask must match the packed signal shape, got "
            f"{tuple(update_mask.shape)} and {tuple(signal.shape[:-1])}."
        )

    with torch.no_grad():
        signal = signal.detach().float()
        current_signal = current_signal.detach().float()
        replay_signal = replay_signal.detach().float()
        current_norm = current_signal.norm(dim=-1)
        replay_norm = replay_signal.norm(dim=-1)
        signal_norm = signal.norm(dim=-1)
        both = (current_norm > 0) & (replay_norm > 0)
        both_count = both.float().sum()
        cosine = F.cosine_similarity(
            current_signal, replay_signal, dim=-1, eps=1e-8
        )
        cosine_mean = torch.where(
            both_count > 0,
            (cosine * both.float()).sum() / both_count.clamp_min(1.0),
            zero.float(),
        )
        conflict_ratio = torch.where(
            both_count > 0,
            ((cosine < 0) & both).float().sum() / both_count.clamp_min(1.0),
            zero.float(),
        )
        valid = signal_norm > 0
        if update_mask is not None:
            valid = valid & update_mask.detach().bool()
        valid_count = valid.float().sum()

        def alignment(delta: torch.Tensor) -> torch.Tensor:
            dot = (signal * delta.detach().float()).sum(dim=-1)
            return torch.where(
                valid_count > 0,
                ((dot < 0) & valid).float().sum()
                / valid_count.clamp_min(1.0),
                zero.float(),
            )

        return {
            "mean_gradient_current_norm_mean": current_norm.mean(),
            "mean_gradient_replay_norm_mean": replay_norm.mean(),
            "mean_gradient_signal_norm_mean": signal_norm.mean(),
            "mean_gradient_current_replay_cosine": cosine_mean,
            "mean_gradient_current_replay_conflict_ratio": conflict_ratio,
            "mean_gradient_raw_update_alignment": alignment(
                raw_delta_mean_camera
            ),
            "mean_gradient_effective_update_alignment": alignment(
                effective_delta_mean_camera
            ),
        }


def compute_old_opacity_gradient_effective_stats(
    evidence: ProjectedGSEvidence,
    original_opacities: torch.Tensor,
    updated_opacities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure old-GS gradient alignment from the final opacity-logit change."""
    zero = original_opacities.new_zeros(())
    if (
        evidence.opacity_gradient_signal is None
        or evidence.opacity_gradient_current_signal is None
        or evidence.opacity_gradient_replay_signal is None
        or evidence.flat_candidate_indices.numel() == 0
    ):
        return {
            key: zero for key in OLD_OPACITY_GRADIENT_EFFECTIVE_STAT_KEYS
        }
    if original_opacities.shape != updated_opacities.shape:
        raise ValueError(
            "Old-GS effective alignment requires matching opacity tensors, got "
            f"{tuple(original_opacities.shape)} and "
            f"{tuple(updated_opacities.shape)}."
        )

    flat_indices = evidence.flat_candidate_indices.long()
    with torch.no_grad():
        original = original_opacities.detach().float().reshape(-1).index_select(
            0, flat_indices
        )
        updated = updated_opacities.detach().float().reshape(-1).index_select(
            0, flat_indices
        )
        effective_delta = torch.logit(updated.clamp(1e-5, 1.0 - 1e-5)) - torch.logit(
            original.clamp(1e-5, 1.0 - 1e-5)
        )
        signal = evidence.opacity_gradient_signal.detach().float()
        current_signal = evidence.opacity_gradient_current_signal.detach().float()
        replay_signal = evidence.opacity_gradient_replay_signal.detach().float()
        consistent = evidence.surface_consistent.reshape(-1).index_select(
            0, flat_indices
        )
        front = evidence.old_in_front.reshape(-1).index_select(0, flat_indices)

        def alignment(
            selected_signal: torch.Tensor,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            valid = selected_signal != 0
            if mask is not None:
                valid = valid & mask.bool()
            count = valid.float().sum()
            return torch.where(
                count > 0,
                (((selected_signal * effective_delta) < 0) & valid)
                .float()
                .sum()
                / count.clamp_min(1.0),
                zero.float(),
            )

        both_signals = (current_signal != 0) & (replay_signal != 0)
        conflict_count = both_signals.float().sum()
        conflict_ratio = torch.where(
            conflict_count > 0,
            (
                ((current_signal * replay_signal) < 0) & both_signals
            ).float().sum()
            / conflict_count.clamp_min(1.0),
            zero.float(),
        )
        return {
            "gradient_consistent_alignment": alignment(signal, consistent),
            "gradient_front_alignment": alignment(signal, front),
            "gradient_effective_alignment": alignment(signal),
            "gradient_current_effective_alignment": alignment(current_signal),
            "gradient_replay_effective_alignment": alignment(replay_signal),
            "gradient_current_replay_conflict_ratio": conflict_ratio,
        }


def align_gaussians_to_camera_depth_scale(
    gaussians: Gaussians,
    camera_to_world: torch.Tensor,
    depth_scale: torch.Tensor,
    point_map: torch.Tensor | None = None,
) -> tuple[Gaussians, torch.Tensor | None]:
    """Align current-view Gaussian geometry to the streaming map scale."""
    batch, count = gaussians.means.shape[:2]
    if camera_to_world.shape != (batch, 4, 4):
        raise ValueError(
            "Depth-scale alignment expects camera_to_world [B,4,4], got "
            f"{tuple(camera_to_world.shape)}."
        )
    if depth_scale.numel() != batch:
        raise ValueError(
            "Depth-scale alignment expects one scale per batch item, got "
            f"shape={tuple(depth_scale.shape)} for batch={batch}."
        )
    if point_map is not None and point_map.shape != (batch, count, 3):
        raise ValueError(
            "Depth-scale alignment point_map must match Gaussian means, got "
            f"{tuple(point_map.shape)} and {tuple(gaussians.means.shape)}."
        )

    scale_float = depth_scale.detach().float().reshape(batch)
    if not bool(torch.isfinite(scale_float).all()) or bool(
        (scale_float <= 0).any()
    ):
        raise ValueError(
            "Depth-scale alignment requires finite positive scales, got "
            f"{scale_float.tolist()}."
        )
    mean_scale = scale_float.to(gaussians.means.dtype)[:, None, None]
    camera_center = camera_to_world[:, None, :3, 3].to(
        gaussians.means.dtype
    )
    aligned_means = camera_center + (
        gaussians.means - camera_center
    ) * mean_scale
    aligned_scales = gaussians.scales * scale_float.to(
        gaussians.scales.dtype
    )[:, None, None]
    aligned_point_map = None
    if point_map is not None:
        point_scale = scale_float.to(point_map.dtype)[:, None, None]
        point_center = camera_to_world[:, None, :3, 3].to(point_map.dtype)
        aligned_point_map = point_center + (
            point_map - point_center
        ) * point_scale

    return (
        Gaussians(
            means=aligned_means,
            harmonics=gaussians.harmonics,
            opacities=gaussians.opacities,
            scales=aligned_scales,
            rotations=gaussians.rotations,
        ),
        aligned_point_map,
    )


@torch.no_grad()
def build_point_map_append_mask(
    state: "StreamingGaussianState",
    current_depth: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    current_rgb: torch.Tensor | None = None,
    old_rgb: torch.Tensor | None = None,
    old_depth: torch.Tensor | None = None,
    old_alpha: torch.Tensor | None = None,
    current_depth_confidence: torch.Tensor | None = None,
    depth_relative_tolerance: float = 0.15,
    rgb_bad_threshold: float = 0.08,
    old_alpha_threshold: float = 0.10,
    min_depth_confidence: float = 0.0,
    dilation_radius: int = 2,
    chunk_size: int = 65536,
    depth_check_enabled: bool = True,
    hybrid_enabled: bool = False,
    append_current_in_front_enabled: bool = True,
    append_history_in_front_enabled: bool = True,
    front_conflict_confirm_views: int = 1,
    max_append_ratio: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Find current pixels not covered by projected historical points.

    Historical Gaussian centers are treated strictly as a point map. Gaussian
    scale and covariance do not affect coverage; effectively transparent
    entries are ignored when deciding whether a front conflict is reliable.
    Hybrid mode aligns current depth to the rendered historical depth per
    scene and performs immediate layered fusion. Any historical point near the
    current depth is a match and suppresses duplicate append. Historical points
    in front of the current surface are opacity-decay candidates. If no match
    exists, the current Gaussian is appended regardless of whether history is
    in front, behind, or absent. The current-in-front and history-in-front
    cases can be disabled independently as ablations. RGB does not decide
    geometry in this mode.
    """
    if current_depth.ndim == 3:
        current_depth = current_depth.unsqueeze(1)
    b, channels, h, w = current_depth.shape
    if channels != 1 or state.batch_size != b:
        raise ValueError(
            "Point-map append expects depth [B,1,H,W] matching the state; "
            f"got depth={tuple(current_depth.shape)}, batch={state.batch_size}."
        )
    if camera_to_world.shape != (b, 4, 4):
        raise ValueError(
            "Point-map append expects camera_to_world [B,4,4], got "
            f"{tuple(camera_to_world.shape)}."
        )
    if intrinsics.shape != (b, 3, 3):
        raise ValueError(
            "Point-map append expects intrinsics [B,3,3], got "
            f"{tuple(intrinsics.shape)}."
        )

    device = current_depth.device
    depth = current_depth.detach().float()
    valid_current = torch.isfinite(depth) & (depth > 1e-5)
    reliable_current = valid_current.clone()
    if current_depth_confidence is not None:
        confidence = current_depth_confidence.detach().float()
        if confidence.ndim == 3:
            confidence = confidence.unsqueeze(1)
        if confidence.shape[-2:] != (h, w):
            confidence = F.interpolate(
                confidence, size=(h, w), mode="bilinear", align_corners=False
            )
        if confidence.shape != depth.shape:
            raise ValueError(
                "Point-map append depth confidence must match depth; got "
                f"{tuple(confidence.shape)} and {tuple(depth.shape)}."
            )
        reliable_current &= torch.isfinite(confidence) & (
            confidence >= float(min_depth_confidence)
        )
    confidence_score = torch.ones_like(depth)
    if current_depth_confidence is not None:
        confidence_score = torch.nan_to_num(
            confidence, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)

    def prepare_single_channel(
        name: str,
        value: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        value = value.detach().float()
        if value.ndim == 3:
            value = value.unsqueeze(1)
        if value.ndim != 4 or value.shape[:2] != (b, 1):
            raise ValueError(
                f"{name} must have shape [B,1,H,W], got {tuple(value.shape)}."
            )
        if value.shape[-2:] != (h, w):
            value = F.interpolate(
                value, size=(h, w), mode="nearest"
            )
        return value

    prepared_old_depth = prepare_single_channel("old_depth", old_depth)
    prepared_old_alpha = prepare_single_channel("old_alpha", old_alpha)
    depth_scale = torch.ones((b,), device=device, dtype=torch.float32)
    if prepared_old_depth is not None and prepared_old_alpha is not None:
        trusted = (
            reliable_current
            & torch.isfinite(prepared_old_depth)
            & (prepared_old_depth > 1e-5)
            & torch.isfinite(prepared_old_alpha)
            & (prepared_old_alpha >= max(0.0, float(old_alpha_threshold)))
        )
        for batch_idx in range(b):
            ratios = (
                prepared_old_depth[batch_idx][trusted[batch_idx]]
                / depth[batch_idx][trusted[batch_idx]].clamp_min(1e-5)
            )
            if ratios.numel() >= 32:
                log_scale = ratios.log().median().clamp(-2.3, 2.3)
                depth_scale[batch_idx] = log_scale.exp()

    aligned_depth = depth * depth_scale[:, None, None, None]
    depth_flat = aligned_depth[:, 0].reshape(b, h * w)
    reliable_flat = reliable_current[:, 0].reshape(b, h * w)
    projection_count = torch.zeros(
        (b, h * w), device=device, dtype=torch.float32
    )
    consistent_count = torch.zeros_like(projection_count)
    nearest_depth = torch.full_like(projection_count, float("inf"))
    world_to_camera = torch.linalg.inv(camera_to_world.detach().float())
    fx = intrinsics[:, 0, 0].detach().float().unsqueeze(1) * w
    fy = intrinsics[:, 1, 1].detach().float().unsqueeze(1) * h
    cx = intrinsics[:, 0, 2].detach().float().unsqueeze(1) * w
    cy = intrinsics[:, 1, 2].detach().float().unsqueeze(1) * h
    tolerance = max(0.0, float(depth_relative_tolerance))
    chunk_size = max(1, int(chunk_size))
    point_count = state.num_gaussians
    point_map = (
        state.point_map
        if state.point_map is not None
        else state.gaussians.means.detach()
    )

    for start in range(0, point_count, chunk_size):
        end = min(start + chunk_size, point_count)
        points = point_map[:, start:end].detach().float()
        ones = torch.ones(
            (*points.shape[:2], 1), device=device, dtype=torch.float32
        )
        camera_points = torch.einsum(
            "bij,bnj->bni",
            world_to_camera,
            torch.cat([points, ones], dim=-1),
        )[..., :3].float()
        z = camera_points[..., 2]
        z_safe = z.clamp_min(1e-6)
        u = fx * camera_points[..., 0] / z_safe + cx
        v = fy * camera_points[..., 1] / z_safe + cy
        pixel_x = u.round().long()
        pixel_y = v.round().long()
        in_image = (
            torch.isfinite(camera_points).all(dim=-1)
            & (z > 1e-5)
            & (pixel_x >= 0)
            & (pixel_x < w)
            & (pixel_y >= 0)
            & (pixel_y < h)
            & (state.stable_ids[:, start:end] >= 0)
            & (state.gaussians.opacities[:, start:end].detach().float() > 1e-4)
        )
        linear_index = (pixel_y.clamp(0, h - 1) * w + pixel_x.clamp(0, w - 1))
        sampled_depth = depth_flat.gather(1, linear_index)
        sampled_reliable = reliable_flat.gather(1, linear_index)
        depth_tolerance = tolerance * sampled_depth.clamp_min(1e-4)
        consistent = (
            in_image
            & sampled_reliable
            & ((z - sampled_depth).abs() <= depth_tolerance)
        )
        projection_count.scatter_add_(
            1, linear_index, in_image.to(projection_count.dtype)
        )
        consistent_count.scatter_add_(
            1, linear_index, consistent.to(consistent_count.dtype)
        )
        projected_depth = torch.where(
            in_image, z, torch.full_like(z, float("inf"))
        ).to(dtype=nearest_depth.dtype)
        nearest_depth.scatter_reduce_(
            1,
            linear_index,
            projected_depth,
            reduce="amin",
            include_self=True,
        )

    has_projection = projection_count > 0
    same_surface = consistent_count > 0
    tolerance_flat = tolerance * depth_flat.clamp_min(1e-4)
    history_in_front = (
        has_projection
        & reliable_flat
        & (nearest_depth < depth_flat - tolerance_flat)
    )
    current_in_front = (
        has_projection
        & reliable_flat
        & (nearest_depth > depth_flat + tolerance_flat)
    )
    raw_front_conflict_gaussians = torch.zeros(
        (b, point_count), device=device, dtype=torch.bool
    )
    observed_front_gaussians = torch.zeros_like(raw_front_conflict_gaussians)
    if hybrid_enabled:
        # Every projected historical point is classified independently. This
        # preserves a matching deeper layer while fading all floaters in front.
        for start in range(0, point_count, chunk_size):
            end = min(start + chunk_size, point_count)
            points = point_map[:, start:end].detach().float()
            ones = torch.ones(
                (*points.shape[:2], 1), device=device, dtype=torch.float32
            )
            camera_points = torch.einsum(
                "bij,bnj->bni",
                world_to_camera,
                torch.cat([points, ones], dim=-1),
            )[..., :3].float()
            z = camera_points[..., 2]
            z_safe = z.clamp_min(1e-6)
            u = fx * camera_points[..., 0] / z_safe + cx
            v = fy * camera_points[..., 1] / z_safe + cy
            pixel_x = u.round().long()
            pixel_y = v.round().long()
            in_image = (
                torch.isfinite(camera_points).all(dim=-1)
                & (z > 1e-5)
                & (pixel_x >= 0)
                & (pixel_x < w)
                & (pixel_y >= 0)
                & (pixel_y < h)
                & (state.stable_ids[:, start:end] >= 0)
                & (
                    state.gaussians.opacities[:, start:end].detach().float()
                    > 1e-4
                )
            )
            linear_index = (
                pixel_y.clamp(0, h - 1) * w + pixel_x.clamp(0, w - 1)
            )
            sampled_depth = depth_flat.gather(1, linear_index)
            sampled_tolerance = tolerance * sampled_depth.clamp_min(1e-4)
            observed = (
                in_image
                & reliable_flat.gather(1, linear_index)
            )
            observed_front_gaussians[:, start:end] = observed
            raw_front_conflict_gaussians[:, start:end] = (
                observed
                & (z < sampled_depth - sampled_tolerance)
            )

    previous_conflict_count = state.front_conflict_count
    if previous_conflict_count is None:
        previous_conflict_count = torch.zeros(
            (b, point_count), device=device, dtype=torch.uint8
        )
    elif previous_conflict_count.shape != (b, point_count):
        raise ValueError(
            "Historical front-conflict counts must match the streaming map; "
            f"got {tuple(previous_conflict_count.shape)} and {(b, point_count)}."
        )
    else:
        previous_conflict_count = previous_conflict_count.detach().to(torch.uint8)

    confirm_views = min(255, max(1, int(front_conflict_confirm_views)))
    next_conflict_count = previous_conflict_count.clone()
    next_conflict_count = torch.where(
        observed_front_gaussians & ~raw_front_conflict_gaussians,
        torch.zeros_like(next_conflict_count),
        next_conflict_count,
    )
    next_conflict_count = torch.where(
        raw_front_conflict_gaussians,
        (previous_conflict_count.to(torch.int16) + 1)
        .clamp_max(confirm_views)
        .to(torch.uint8),
        next_conflict_count,
    )
    confirmed_front_conflicts = (
        raw_front_conflict_gaussians
        & (next_conflict_count >= confirm_views)
    )

    if hybrid_enabled:
        # Matching is checked across every projected depth layer, not just the
        # nearest point. Neighboring matches may cover small point-projection
        # gaps below, provided the current depth stays on the same surface.
        coverage_flat = torch.where(reliable_flat, same_surface, has_projection)
    elif depth_check_enabled:
        coverage_flat = same_surface
    else:
        coverage_flat = has_projection

    radius = max(0, int(dilation_radius))
    projection_coverage = has_projection.reshape(b, 1, h, w)
    coverage = coverage_flat.reshape(b, 1, h, w)
    if radius > 0:
        kernel_size = 2 * radius + 1
        projection_coverage = F.max_pool2d(
            projection_coverage.float(), kernel_size, stride=1, padding=radius
        ).bool()
        if hybrid_enabled:
            coverage = _dilate_depth_consistent_coverage(
                coverage,
                aligned_depth,
                reliable_current,
                radius=radius,
                depth_relative_tolerance=tolerance,
            )
        else:
            coverage = F.max_pool2d(
                coverage.float(), kernel_size, stride=1, padding=radius
            ).bool()
    append_valid = reliable_current if hybrid_enabled else valid_current
    current_in_front_image = current_in_front.reshape(b, 1, h, w)
    front_without_match = (history_in_front & ~same_surface).reshape(
        b, 1, h, w
    )
    force_append = torch.zeros_like(coverage)
    if hybrid_enabled and append_current_in_front_enabled:
        force_append |= current_in_front_image
    if hybrid_enabled and append_history_in_front_enabled:
        force_append |= front_without_match
    coverage &= ~force_append
    append_candidates = append_valid & ~coverage
    if hybrid_enabled and not append_current_in_front_enabled:
        append_candidates &= ~current_in_front_image
    if hybrid_enabled and not append_history_in_front_enabled:
        append_candidates &= ~front_without_match

    append_mask = torch.zeros_like(append_candidates)
    max_ratio = max(0.0, min(1.0, float(max_append_ratio)))
    no_projection = ~projection_coverage
    priority = confidence_score * 0.01
    priority = priority + no_projection.float() * 3.0
    priority = priority + current_in_front_image.float() * 2.0
    priority = priority + front_without_match.float()
    pixel_index = torch.arange(h * w, device=device, dtype=torch.long)
    tie_breaker = (
        (pixel_index * 1103515245 + 12345).remainder(2147483647).float()
        / 2147483647.0
    ).reshape(1, 1, h, w)
    priority = priority + tie_breaker * 1e-4
    for batch_idx in range(b):
        candidate_indices = torch.nonzero(
            append_candidates[batch_idx].reshape(-1), as_tuple=False
        ).squeeze(1)
        valid_count = int(append_valid[batch_idx].sum().item())
        max_count = min(
            int(candidate_indices.numel()),
            int(valid_count * max_ratio + 0.999999),
        )
        if max_count <= 0:
            continue
        candidate_priority = priority[batch_idx].reshape(-1)[candidate_indices]
        selected = candidate_indices[
            candidate_priority.topk(max_count, sorted=False).indices
        ]
        append_mask[batch_idx].view(-1)[selected] = True

    return (
        append_mask,
        coverage,
        confirmed_front_conflicts,
        next_conflict_count,
        depth_scale,
    )


def _dilate_depth_consistent_coverage(
    coverage: torch.Tensor,
    current_depth: torch.Tensor,
    reliable_current: torch.Tensor,
    *,
    radius: int,
    depth_relative_tolerance: float,
) -> torch.Tensor:
    """Expand coverage along the same current-depth surface."""
    if coverage.dtype != torch.bool or reliable_current.dtype != torch.bool:
        raise TypeError("Coverage dilation masks must be boolean.")
    if (
        coverage.shape != current_depth.shape
        or coverage.shape != reliable_current.shape
    ):
        raise ValueError(
            "Coverage, current depth, and reliability must have matching "
            f"shapes, got {tuple(coverage.shape)}, {tuple(current_depth.shape)}, "
            f"and {tuple(reliable_current.shape)}."
        )
    radius = max(0, int(radius))
    if radius == 0:
        return coverage

    depth = current_depth.detach().float()
    tolerance = max(0.0, float(depth_relative_tolerance))
    expanded = coverage
    height, width = coverage.shape[-2:]
    for _ in range(radius):
        padded_coverage = F.pad(expanded.float(), (1, 1, 1, 1)).bool()
        padded_depth = F.pad(depth, (1, 1, 1, 1), value=float("nan"))
        next_coverage = expanded.clone()
        depth_margin = tolerance * depth.clamp_min(1e-4)
        for offset_y in range(3):
            for offset_x in range(3):
                neighbor_coverage = padded_coverage[
                    :, :, offset_y : offset_y + height, offset_x : offset_x + width
                ]
                neighbor_depth = padded_depth[
                    :, :, offset_y : offset_y + height, offset_x : offset_x + width
                ]
                depth_consistent = (
                    reliable_current
                    & torch.isfinite(neighbor_depth)
                    & ((neighbor_depth - depth).abs() <= depth_margin)
                )
                next_coverage |= neighbor_coverage & depth_consistent
        expanded = next_coverage
    return expanded


@torch.no_grad()
def build_confirmed_current_in_front_mask(
    current_depth: torch.Tensor,
    old_depth: torch.Tensor,
    old_alpha: torch.Tensor,
    *,
    depth_scale: torch.Tensor,
    depth_relative_tolerance: float,
    old_alpha_threshold: float,
    current_depth_confidence: torch.Tensor | None = None,
    min_depth_confidence: float = 0.0,
) -> torch.Tensor:
    """Identify reliable pixels where the current surface occludes old geometry."""
    if current_depth.ndim == 3:
        current_depth = current_depth.unsqueeze(1)
    if current_depth.ndim != 4 or current_depth.shape[1] != 1:
        raise ValueError(
            "Current-in-front detection expects current_depth [B,1,H,W], got "
            f"{tuple(current_depth.shape)}."
        )
    batch, _, height, width = current_depth.shape

    def prepare(name: str, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 3:
            value = value.unsqueeze(1)
        if value.ndim != 4 or value.shape[:2] != (batch, 1):
            raise ValueError(
                f"{name} must have shape [B,1,H,W], got {tuple(value.shape)}."
            )
        value = value.detach().float()
        if value.shape[-2:] != (height, width):
            value = F.interpolate(value, size=(height, width), mode="nearest")
        return value

    depth = current_depth.detach().float()
    historical_depth = prepare("old_depth", old_depth)
    historical_alpha = prepare("old_alpha", old_alpha)
    if depth_scale.shape != (batch,):
        raise ValueError(
            "Current-in-front depth scale must have shape [B], got "
            f"{tuple(depth_scale.shape)}."
        )
    scale = depth_scale.detach().float()[:, None, None, None]
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("Current-in-front depth scales must be finite and positive.")

    reliable = torch.isfinite(depth) & (depth > 1e-5)
    if current_depth_confidence is not None:
        confidence = prepare(
            "current_depth_confidence", current_depth_confidence
        )
        reliable &= torch.isfinite(confidence) & (
            confidence >= float(min_depth_confidence)
        )
    aligned_depth = depth * scale
    tolerance = max(0.0, float(depth_relative_tolerance))
    margin = tolerance * aligned_depth.clamp_min(1e-4)
    return (
        reliable
        & torch.isfinite(historical_depth)
        & (historical_depth > 1e-5)
        & torch.isfinite(historical_alpha)
        & (historical_alpha >= max(0.0, float(old_alpha_threshold)))
        & (historical_depth > aligned_depth + margin)
    )


def combine_tracker_point_map_append_mask(
    valid_current: torch.Tensor,
    tracker_coverage: torch.Tensor,
    point_depth_consistent_coverage: torch.Tensor,
    confirmed_current_in_front: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append only pixels missing from both memories, plus new foreground."""
    tensors = {
        "valid_current": valid_current,
        "tracker_coverage": tracker_coverage,
        "point_depth_consistent_coverage": point_depth_consistent_coverage,
        "confirmed_current_in_front": confirmed_current_in_front,
    }
    shape = valid_current.shape
    for name, value in tensors.items():
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must be boolean, got {value.dtype}.")
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}."
            )
    known = tracker_coverage | point_depth_consistent_coverage
    append = valid_current & ~known
    append |= valid_current & confirmed_current_in_front
    return append, known


def _sample_at_projected_centers(
    image: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    sampled = F.grid_sample(
        image,
        grid.unsqueeze(2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(-1).transpose(1, 2)


def build_projected_gaussian_evidence(
    state: "StreamingGaussianState",
    current_feature: torch.Tensor,
    current_rgb: torch.Tensor,
    current_depth: torch.Tensor,
    current_depth_confidence: torch.Tensor,
    old_rgb: torch.Tensor,
    old_depth: torch.Tensor,
    old_alpha: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    alpha_threshold: float,
    depth_relative_tolerance: float,
    rgb_bad_threshold: float,
    min_projected_radius: float,
    covariance_extent_sigma: float,
    chunk_size: int = 65536,
    opacity_gradient_evidence: OpacityGradientEvidence | None = None,
    mean_gradient_evidence: MeanGradientEvidence | None = None,
) -> ProjectedGSEvidence:
    """Build packed descriptors for revisable historical Gaussians."""
    b, n = state.gaussians.means.shape[:2]
    h, w = current_depth.shape[-2:]
    if current_depth.shape[:2] != (b, 1):
        raise ValueError(
            "Projected GS evidence expects current_depth [B,1,H,W], got "
            f"{tuple(current_depth.shape)}."
        )

    expected_dim = current_feature.shape[1] + PROJECTED_GS_STATE_DIM
    chunk_size = max(1, int(chunk_size))
    device = state.gaussians.means.device
    candidate_mask = torch.zeros((b, n), device=device, dtype=torch.bool)
    surface_consistent = torch.zeros_like(candidate_mask)
    old_in_front = torch.zeros_like(candidate_mask)
    projected_radius = torch.zeros((b, n), device=device, dtype=torch.float32)
    relative_depth = torch.zeros_like(projected_radius)
    camera_depth = torch.zeros_like(projected_radius)
    packed_descriptors = []
    packed_indices = []
    packed_gradient_features = []
    packed_gradient_signals = []
    packed_gradient_current_signals = []
    packed_gradient_replay_signals = []
    packed_mean_gradient_features = []
    packed_mean_gradient_signals = []
    packed_mean_gradient_current_signals = []
    packed_mean_gradient_replay_signals = []
    has_opacity_gradient_evidence = opacity_gradient_evidence is not None
    if opacity_gradient_evidence is None:
        gradient_features = torch.zeros(
            (b, n, OPACITY_GRADIENT_FEATURE_DIM),
            device=device,
            dtype=torch.float32,
        )
        gradient_signal = torch.zeros((b, n), device=device, dtype=torch.float32)
        gradient_current_signal = torch.zeros_like(gradient_signal)
        gradient_replay_signal = torch.zeros_like(gradient_signal)
    else:
        gradient_features = opacity_gradient_evidence.features
        gradient_signal = opacity_gradient_evidence.signal
        gradient_current_signal = opacity_gradient_evidence.current_signal
        gradient_replay_signal = opacity_gradient_evidence.replay_signal
        if gradient_features.shape != (b, n, OPACITY_GRADIENT_FEATURE_DIM):
            raise ValueError(
                "Historical opacity-gradient features must be [B,N,8], got "
                f"{tuple(gradient_features.shape)}."
            )
        if gradient_signal.shape != (b, n):
            raise ValueError(
                "Historical opacity-gradient signal must be [B,N], got "
                f"{tuple(gradient_signal.shape)}."
            )
        if gradient_current_signal.shape != (b, n):
            raise ValueError(
                "Historical current opacity-gradient signal must be [B,N], got "
                f"{tuple(gradient_current_signal.shape)}."
            )
        if gradient_replay_signal.shape != (b, n):
            raise ValueError(
                "Historical replay opacity-gradient signal must be [B,N], got "
                f"{tuple(gradient_replay_signal.shape)}."
            )
    if mean_gradient_evidence is None:
        mean_gradient_features = None
        mean_gradient_signal = None
        mean_gradient_current_signal = None
        mean_gradient_replay_signal = None
    else:
        mean_gradient_features = mean_gradient_evidence.features
        mean_gradient_signal = mean_gradient_evidence.signal
        mean_gradient_current_signal = mean_gradient_evidence.current_signal
        mean_gradient_replay_signal = mean_gradient_evidence.replay_signal
        if mean_gradient_features.shape != (b, n, MEAN_GRADIENT_FEATURE_DIM):
            raise ValueError(
                "Historical mean-gradient features must be [B,N,10], got "
                f"{tuple(mean_gradient_features.shape)}."
            )
        for name, value in (
            ("combined", mean_gradient_signal),
            ("current", mean_gradient_current_signal),
            ("replay", mean_gradient_replay_signal),
        ):
            if value.shape != (b, n, 3):
                raise ValueError(
                    f"Historical {name} mean-gradient signal must be [B,N,3], "
                    f"got {tuple(value.shape)}."
                )

    with torch.no_grad():
        world_to_camera = torch.linalg.inv(camera_to_world.detach().float())
        rotation_w2c = world_to_camera[:, None, :3, :3]
        camera_center = camera_to_world.detach().float()[:, None, :3, 3]
        fx = intrinsics[:, 0, 0].detach().float().unsqueeze(1) * w
        fy = intrinsics[:, 1, 1].detach().float().unsqueeze(1) * h
        cx = intrinsics[:, 0, 2].detach().float().unsqueeze(1) * w
        cy = intrinsics[:, 1, 2].detach().float().unsqueeze(1) * h
        sigma = max(float(covariance_extent_sigma), 0.0)
        harmonic_shape = state.gaussians.harmonics.shape
        if harmonic_shape[-1] == 3:
            # gsplat stores SH as [..., coefficient, RGB].
            sh_coefficient_count = harmonic_shape[-2]
            transpose_harmonics = True
        elif harmonic_shape[-2] == 3:
            # Also accept the annotated [..., RGB, coefficient] layout.
            sh_coefficient_count = harmonic_shape[-1]
            transpose_harmonics = False
        else:
            raise RuntimeError(
                "Historical Gaussian SH must have an RGB dimension of size 3; "
                f"got shape={tuple(harmonic_shape)}."
            )
        sh_degree = int(sh_coefficient_count**0.5) - 1
        if (sh_degree + 1) ** 2 != sh_coefficient_count or not 0 <= sh_degree <= 4:
            raise RuntimeError(
                "Historical Gaussian SH coefficient count must be a supported "
                f"square number, got {sh_coefficient_count}."
            )

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        count = end - start
        with torch.no_grad():
            means = state.gaussians.means[:, start:end].detach().float()
            scales = (
                state.gaussians.scales[:, start:end]
                .detach()
                .float()
                .clamp_min(1e-8)
            )
            rotations = state.gaussians.rotations[:, start:end].detach().float()
            ones = torch.ones((b, count, 1), device=device, dtype=torch.float32)
            camera_points = torch.einsum(
                "bij,bnj->bni",
                world_to_camera,
                torch.cat([means, ones], dim=-1),
            )[..., :3]
            z = camera_points[..., 2]
            z_safe = z.clamp_min(1e-6)
            u = fx * camera_points[..., 0] / z_safe + cx
            v = fy * camera_points[..., 1] / z_safe + cy
            grid = torch.stack(
                [
                    2.0 * u / max(w - 1, 1) - 1.0,
                    2.0 * v / max(h - 1, 1) - 1.0,
                ],
                dim=-1,
            )

            covariance_world = build_covariance(scales, rotations)
            covariance_camera = (
                rotation_w2c
                @ covariance_world
                @ rotation_w2c.transpose(-1, -2)
            )
            jacobian = torch.zeros(
                (b, count, 2, 3), device=device, dtype=torch.float32
            )
            jacobian[..., 0, 0] = fx / z_safe
            jacobian[..., 0, 2] = -fx * camera_points[..., 0] / z_safe.square()
            jacobian[..., 1, 1] = fy / z_safe
            jacobian[..., 1, 2] = -fy * camera_points[..., 1] / z_safe.square()
            covariance_2d = (
                jacobian @ covariance_camera @ jacobian.transpose(-1, -2)
            )
            cov_xx = covariance_2d[..., 0, 0].clamp_min(0.0)
            cov_xy = covariance_2d[..., 0, 1]
            cov_yy = covariance_2d[..., 1, 1].clamp_min(0.0)
            max_eigenvalue = 0.5 * (
                cov_xx
                + cov_yy
                + torch.sqrt(
                    (cov_xx - cov_yy).square() + 4.0 * cov_xy.square()
                )
            ).clamp_min(0.0)
            radius = sigma * torch.sqrt(max_eigenvalue + 1e-12)
            depth_extent = sigma * torch.sqrt(
                covariance_camera[..., 2, 2].clamp_min(0.0) + 1e-12
            )
            in_frustum = (
                (z > 1e-5)
                & (u >= 0.0)
                & (u <= float(w - 1))
                & (v >= 0.0)
                & (v <= float(h - 1))
                & (radius >= float(min_projected_radius))
                & (
                    state.gaussians.opacities[:, start:end].detach().float()
                    > 1e-4
                )
            )

        sampled_feature = _sample_at_projected_centers(current_feature, grid)
        sampled_current_rgb = _sample_at_projected_centers(
            current_rgb.float(), grid
        )
        sampled_current_depth = _sample_at_projected_centers(
            current_depth.float(), grid
        ).squeeze(-1)
        sampled_depth_confidence = _sample_at_projected_centers(
            current_depth_confidence.float(), grid
        ).squeeze(-1)
        sampled_old_rgb = _sample_at_projected_centers(old_rgb.float(), grid)
        sampled_old_depth = _sample_at_projected_centers(
            old_depth.float(), grid
        ).squeeze(-1)
        sampled_old_alpha = _sample_at_projected_centers(
            old_alpha.float(), grid
        ).squeeze(-1)

        with torch.no_grad():
            current_depth_detached = sampled_current_depth.detach()
            old_depth_detached = sampled_old_depth.detach()
            old_alpha_detached = sampled_old_alpha.detach()
            valid_current_depth = torch.isfinite(current_depth_detached) & (
                current_depth_detached > 1e-5
            )
            valid_old_depth = torch.isfinite(old_depth_detached) & (
                old_depth_detached > 1e-5
            )
            current_tolerance = (
                float(depth_relative_tolerance)
                * current_depth_detached.clamp_min(1e-4)
            )
            old_tolerance = (
                float(depth_relative_tolerance)
                * old_depth_detached.clamp_min(1e-4)
            )
            consistent_chunk = (
                (z - current_depth_detached).abs()
                <= depth_extent + current_tolerance
            )
            front_chunk = (
                z + depth_extent < current_depth_detached - current_tolerance
            )
            near_old_front = (
                z - depth_extent <= old_depth_detached + old_tolerance
            )
            rgb_error = (
                sampled_current_rgb.detach() - sampled_old_rgb.detach()
            ).abs().mean(dim=-1)
            candidate_chunk = (
                in_frustum
                & valid_current_depth
                & valid_old_depth
                & (old_alpha_detached >= float(alpha_threshold))
                & near_old_front
                & (
                    front_chunk
                    | (
                        consistent_chunk
                        & (rgb_error >= float(rgb_bad_threshold))
                    )
                )
            )
            consistent_chunk = consistent_chunk & candidate_chunk
            front_chunk = front_chunk & candidate_chunk

        current_depth_safe = sampled_current_depth.clamp_min(1e-4)
        old_depth_safe = sampled_old_depth.clamp_min(1e-4)
        relative_depth_chunk = (
            z.to(current_depth_safe.dtype) - current_depth_safe
        ) / current_depth_safe
        relative_old_depth = (
            z.to(old_depth_safe.dtype) - old_depth_safe
        ) / old_depth_safe
        camera_direction = F.normalize(camera_points, dim=-1, eps=1e-8)
        world_direction = F.normalize(means - camera_center, dim=-1, eps=1e-8)
        harmonics = (
            state.gaussians.harmonics[:, start:end].detach().float()
        )
        if transpose_harmonics:
            harmonics = harmonics.transpose(-1, -2)
        sh_color = (
            sh_utils.eval_sh(sh_degree, harmonics, world_direction) + 0.5
        ).clamp(0.0, 1.0)
        descriptor_chunk = torch.cat(
            [
                sampled_feature.float(),
                sampled_current_rgb,
                sampled_old_rgb,
                sampled_current_rgb - sampled_old_rgb,
                current_depth_safe.log().unsqueeze(-1).clamp(-8.0, 8.0),
                z.clamp_min(1e-4).log().unsqueeze(-1).clamp(-8.0, 8.0),
                relative_depth_chunk.unsqueeze(-1).clamp(-2.0, 2.0),
                relative_old_depth.unsqueeze(-1).clamp(-2.0, 2.0),
                sampled_depth_confidence.unsqueeze(-1).clamp(0.0, 1.0),
                sampled_old_alpha.unsqueeze(-1).clamp(0.0, 1.0),
                state.gaussians.opacities[:, start:end]
                .detach()
                .float()
                .unsqueeze(-1),
                scales.log().clamp(-12.0, 2.0),
                camera_direction,
                sh_color,
                state.observation_count[:, start:end]
                .detach()
                .float()
                .log1p()
                .unsqueeze(-1)
                .clamp_max(8.0),
                radius.log1p().unsqueeze(-1),
                (depth_extent / z_safe).unsqueeze(-1).clamp(0.0, 2.0),
                consistent_chunk.float().unsqueeze(-1),
                front_chunk.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        descriptor_chunk = torch.nan_to_num(
            descriptor_chunk, nan=0.0, posinf=8.0, neginf=-8.0
        )
        if descriptor_chunk.shape != (b, count, expected_dim):
            raise RuntimeError(
                "Projected GS descriptor shape mismatch: "
                f"got={tuple(descriptor_chunk.shape)}, "
                f"expected={(b, count, expected_dim)}."
            )
        flat_chunk_mask = candidate_chunk.reshape(-1)
        packed_descriptors.append(
            descriptor_chunk.reshape(-1, expected_dim)[flat_chunk_mask]
        )
        packed_gradient_features.append(
            gradient_features[:, start:end]
            .reshape(-1, OPACITY_GRADIENT_FEATURE_DIM)[flat_chunk_mask]
        )
        packed_gradient_signals.append(
            gradient_signal[:, start:end].reshape(-1)[flat_chunk_mask]
        )
        packed_gradient_current_signals.append(
            gradient_current_signal[:, start:end].reshape(-1)[flat_chunk_mask]
        )
        packed_gradient_replay_signals.append(
            gradient_replay_signal[:, start:end].reshape(-1)[flat_chunk_mask]
        )
        if mean_gradient_features is not None:
            packed_mean_gradient_features.append(
                mean_gradient_features[:, start:end]
                .reshape(-1, MEAN_GRADIENT_FEATURE_DIM)[flat_chunk_mask]
            )
            packed_mean_gradient_signals.append(
                mean_gradient_signal[:, start:end]
                .reshape(-1, 3)[flat_chunk_mask]
            )
            packed_mean_gradient_current_signals.append(
                mean_gradient_current_signal[:, start:end]
                .reshape(-1, 3)[flat_chunk_mask]
            )
            packed_mean_gradient_replay_signals.append(
                mean_gradient_replay_signal[:, start:end]
                .reshape(-1, 3)[flat_chunk_mask]
            )
        global_indices = (
            torch.arange(b, device=device)[:, None] * n
            + torch.arange(start, end, device=device)[None]
        )
        packed_indices.append(global_indices.reshape(-1)[flat_chunk_mask])
        candidate_mask[:, start:end] = candidate_chunk
        surface_consistent[:, start:end] = consistent_chunk
        old_in_front[:, start:end] = front_chunk
        projected_radius[:, start:end] = radius
        relative_depth[:, start:end] = relative_depth_chunk.detach().float()
        camera_depth[:, start:end] = z.detach().float()

    descriptor = torch.cat(packed_descriptors, dim=0).to(current_feature.dtype)
    flat_candidate_indices = torch.cat(packed_indices, dim=0)
    return ProjectedGSEvidence(
        descriptor=descriptor,
        flat_candidate_indices=flat_candidate_indices,
        candidate_mask=candidate_mask,
        surface_consistent=surface_consistent,
        old_in_front=old_in_front,
        projected_radius=projected_radius,
        relative_depth=relative_depth.detach(),
        camera_depth=camera_depth,
        opacity_gradient_features=(
            torch.cat(packed_gradient_features, dim=0).to(current_feature.dtype)
            if has_opacity_gradient_evidence
            else None
        ),
        opacity_gradient_signal=(
            torch.cat(packed_gradient_signals, dim=0)
            if has_opacity_gradient_evidence
            else None
        ),
        opacity_gradient_current_signal=(
            torch.cat(packed_gradient_current_signals, dim=0)
            if has_opacity_gradient_evidence
            else None
        ),
        opacity_gradient_replay_signal=(
            torch.cat(packed_gradient_replay_signals, dim=0)
            if has_opacity_gradient_evidence
            else None
        ),
        mean_gradient_features=(
            torch.cat(packed_mean_gradient_features, dim=0).to(
                current_feature.dtype
            )
            if packed_mean_gradient_features
            else None
        ),
        mean_gradient_signal=(
            torch.cat(packed_mean_gradient_signals, dim=0)
            if packed_mean_gradient_signals
            else None
        ),
        mean_gradient_current_signal=(
            torch.cat(packed_mean_gradient_current_signals, dim=0)
            if packed_mean_gradient_current_signals
            else None
        ),
        mean_gradient_replay_signal=(
            torch.cat(packed_mean_gradient_replay_signals, dim=0)
            if packed_mean_gradient_replay_signals
            else None
        ),
    )


def build_projected_new_gaussian_evidence(
    current: Gaussians,
    append_mask: torch.Tensor,
    current_feature: torch.Tensor,
    current_rgb: torch.Tensor,
    current_depth: torch.Tensor,
    current_depth_confidence: torch.Tensor,
    old_rgb: torch.Tensor,
    old_depth: torch.Tensor,
    old_alpha: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    alpha_threshold: float,
    depth_relative_tolerance: float,
    covariance_extent_sigma: float,
    opacity_gradient_evidence: OpacityGradientEvidence | None = None,
    mean_gradient_evidence: MeanGradientEvidence | None = None,
) -> ProjectedNewGSEvidence:
    """Pack append-selected anchors for the shared pointwise decoder."""
    b, n = current.means.shape[:2]
    h, w = current_depth.shape[-2:]
    if n != h * w:
        raise RuntimeError(
            "Projected new-GS evidence expects one anchor per image pixel: "
            f"gaussians={n}, image_shape={(h, w)}."
        )
    if current_depth.shape[:2] != (b, 1):
        raise ValueError(
            "Projected new-GS evidence expects current_depth [B,1,H,W], got "
            f"{tuple(current_depth.shape)}."
        )
    if append_mask.shape == (b, h, w):
        append_mask = append_mask.unsqueeze(1)
    elif append_mask.shape == (b, n):
        append_mask = append_mask.reshape(b, 1, h, w)
    if append_mask.shape != (b, 1, h, w):
        raise ValueError(
            "Projected new-GS append mask must be [B,1,H,W], [B,H,W], or "
            f"[B,H*W], got {tuple(append_mask.shape)}."
        )

    def resize_image(value: torch.Tensor, mode: str) -> torch.Tensor:
        if value.shape[-2:] == (h, w):
            return value
        kwargs = {"size": (h, w), "mode": mode}
        if mode != "nearest":
            kwargs["align_corners"] = False
        return F.interpolate(value.float(), **kwargs).to(value.dtype)

    old_rgb = resize_image(old_rgb, "bilinear")
    old_depth = resize_image(old_depth, "bilinear")
    old_alpha = resize_image(old_alpha, "bilinear")
    current_depth_confidence = resize_image(
        current_depth_confidence, "bilinear"
    )

    candidate_mask = append_mask[:, 0].bool()
    flat_candidate_indices = torch.nonzero(
        candidate_mask.reshape(-1), as_tuple=False
    ).squeeze(1)
    candidate_count = flat_candidate_indices.numel()
    feature_dim = current_feature.shape[1]
    expected_dim = feature_dim + PROJECTED_GS_STATE_DIM
    device = current.means.device
    has_opacity_gradient_evidence = opacity_gradient_evidence is not None
    if opacity_gradient_evidence is None:
        gradient_features = torch.zeros(
            (b, n, OPACITY_GRADIENT_FEATURE_DIM),
            device=device,
            dtype=torch.float32,
        )
        gradient_signal = torch.zeros((b, n), device=device, dtype=torch.float32)
        gradient_current_signal = torch.zeros_like(gradient_signal)
        gradient_replay_signal = torch.zeros_like(gradient_signal)
    else:
        gradient_features = opacity_gradient_evidence.features
        gradient_signal = opacity_gradient_evidence.signal
        gradient_current_signal = opacity_gradient_evidence.current_signal
        gradient_replay_signal = opacity_gradient_evidence.replay_signal
        if gradient_features.shape != (b, n, OPACITY_GRADIENT_FEATURE_DIM):
            raise ValueError(
                "New-GS opacity-gradient features must be [B,N,8], got "
                f"{tuple(gradient_features.shape)}."
            )
        if gradient_signal.shape != (b, n):
            raise ValueError(
                "New-GS opacity-gradient signal must be [B,N], got "
                f"{tuple(gradient_signal.shape)}."
            )
        if gradient_current_signal.shape != (b, n):
            raise ValueError(
                "New-GS current opacity-gradient signal must be [B,N], got "
                f"{tuple(gradient_current_signal.shape)}."
            )
        if gradient_replay_signal.shape != (b, n):
            raise ValueError(
                "New-GS replay opacity-gradient signal must be [B,N], got "
                f"{tuple(gradient_replay_signal.shape)}."
            )
    if mean_gradient_evidence is None:
        mean_gradient_features = None
        mean_gradient_signal = None
        mean_gradient_current_signal = None
        mean_gradient_replay_signal = None
    else:
        mean_gradient_features = mean_gradient_evidence.features
        mean_gradient_signal = mean_gradient_evidence.signal
        mean_gradient_current_signal = mean_gradient_evidence.current_signal
        mean_gradient_replay_signal = mean_gradient_evidence.replay_signal
        if mean_gradient_features.shape != (b, n, MEAN_GRADIENT_FEATURE_DIM):
            raise ValueError(
                "New-GS mean-gradient features must be [B,N,10], got "
                f"{tuple(mean_gradient_features.shape)}."
            )
        for name, value in (
            ("combined", mean_gradient_signal),
            ("current", mean_gradient_current_signal),
            ("replay", mean_gradient_replay_signal),
        ):
            if value.shape != (b, n, 3):
                raise ValueError(
                    f"New-GS {name} mean-gradient signal must be [B,N,3], "
                    f"got {tuple(value.shape)}."
                )

    if candidate_count:
        batch_indices = torch.div(
            flat_candidate_indices, n, rounding_mode="floor"
        )
        local_indices = flat_candidate_indices.remainder(n)
        pixel_y = torch.div(local_indices, w, rounding_mode="floor")
        pixel_x = local_indices.remainder(w)
        feature_chunks = []
        for batch_idx in range(b):
            selected = batch_indices == batch_idx
            if not bool(selected.any()):
                continue
            x = pixel_x[selected].to(torch.float32)
            y = pixel_y[selected].to(torch.float32)
            grid = torch.stack(
                [
                    2.0 * x / max(w - 1, 1) - 1.0,
                    2.0 * y / max(h - 1, 1) - 1.0,
                ],
                dim=-1,
            ).to(current_feature.dtype)
            feature_chunks.append(
                _sample_at_projected_centers(
                    current_feature[batch_idx : batch_idx + 1],
                    grid.unsqueeze(0),
                )[0]
            )
        sampled_feature = torch.cat(feature_chunks, dim=0)
    else:
        batch_indices = flat_candidate_indices
        sampled_feature = current_feature.new_empty((0, feature_dim))

    def select_image(value: torch.Tensor) -> torch.Tensor:
        channels = value.shape[1]
        flat = value.permute(0, 2, 3, 1).reshape(b * n, channels)
        return flat.index_select(0, flat_candidate_indices)

    selected_means = current.means.reshape(b * n, 3).index_select(
        0, flat_candidate_indices
    )
    selected_scales = current.scales.reshape(b * n, 3).index_select(
        0, flat_candidate_indices
    )
    selected_rotations = current.rotations.reshape(b * n, 4).index_select(
        0, flat_candidate_indices
    )
    harmonics_shape = current.harmonics.shape
    selected_harmonics = current.harmonics.reshape(
        b * n, *harmonics_shape[2:]
    ).index_select(0, flat_candidate_indices)
    selected_opacity = current.opacities.reshape(-1).index_select(
        0, flat_candidate_indices
    )

    # Evidence geometry is a conditioning signal, not a camera objective.
    # Detaching here also avoids the ill-conditioned inverse gradient from a
    # newly initialized/predicted camera entering the camera head.
    with torch.autocast(device_type=device.type, enabled=False):
        world_to_camera = torch.linalg.inv(camera_to_world.detach().float())
        selected_w2c = world_to_camera.index_select(0, batch_indices)
        ones = torch.ones_like(selected_means[..., :1], dtype=torch.float32)
        camera_points = torch.bmm(
            selected_w2c,
            torch.cat([selected_means.float(), ones], dim=-1).unsqueeze(-1),
        ).squeeze(-1)[..., :3]
        z = camera_points[..., 2].clamp_min(1e-6)

        selected_rotation_w2c = selected_w2c[:, :3, :3]
        covariance_world = build_covariance(
            selected_scales.float().clamp_min(1e-8),
            selected_rotations.float(),
        )
        covariance_camera = (
            selected_rotation_w2c
            @ covariance_world
            @ selected_rotation_w2c.transpose(-1, -2)
        )
        fx = (
            intrinsics[:, 0, 0]
            .detach()
            .float()
            .index_select(0, batch_indices)
            * w
        )
        fy = (
            intrinsics[:, 1, 1]
            .detach()
            .float()
            .index_select(0, batch_indices)
            * h
        )
        jacobian = torch.zeros(
            (candidate_count, 2, 3), device=device, dtype=torch.float32
        )
        jacobian[:, 0, 0] = fx / z
        jacobian[:, 0, 2] = -fx * camera_points[:, 0] / z.square()
        jacobian[:, 1, 1] = fy / z
        jacobian[:, 1, 2] = -fy * camera_points[:, 1] / z.square()
        covariance_2d = (
            jacobian @ covariance_camera @ jacobian.transpose(-1, -2)
        )
        cov_xx = covariance_2d[:, 0, 0].clamp_min(0.0)
        cov_xy = covariance_2d[:, 0, 1]
        cov_yy = covariance_2d[:, 1, 1].clamp_min(0.0)
        max_eigenvalue = 0.5 * (
            cov_xx
            + cov_yy
            + torch.sqrt(
                (cov_xx - cov_yy).square() + 4.0 * cov_xy.square()
            )
        ).clamp_min(0.0)
        sigma = max(float(covariance_extent_sigma), 0.0)
        projected_radius = sigma * torch.sqrt(max_eigenvalue + 1e-12)
        depth_extent = sigma * torch.sqrt(
            covariance_camera[:, 2, 2].clamp_min(0.0) + 1e-12
        )

    sampled_current_rgb = select_image(current_rgb.float())
    sampled_current_depth = select_image(current_depth.float()).squeeze(-1)
    sampled_confidence = select_image(
        current_depth_confidence.float()
    ).squeeze(-1)
    sampled_old_rgb = select_image(old_rgb.float())
    sampled_old_depth = select_image(old_depth.float()).squeeze(-1)
    sampled_old_alpha = select_image(old_alpha.float()).squeeze(-1)
    current_depth_safe = sampled_current_depth.clamp_min(1e-4)
    old_depth_safe = sampled_old_depth.clamp_min(1e-4)

    with torch.no_grad():
        history_valid_selected = (
            torch.isfinite(sampled_old_depth.detach())
            & (sampled_old_depth.detach() > 1e-5)
            & (sampled_old_alpha.detach() >= float(alpha_threshold))
        )
        tolerance = (
            float(depth_relative_tolerance) * current_depth_safe.detach()
        )
        history_consistent_selected = history_valid_selected & (
            (sampled_old_depth.detach() - z.detach()).abs()
            <= tolerance + depth_extent.detach()
        )
        history_in_front_selected = history_valid_selected & (
            sampled_old_depth.detach() + depth_extent.detach()
            < z.detach() - tolerance
        )

    if harmonics_shape[-1] == 3:
        sh_coefficient_count = harmonics_shape[-2]
        harmonics_for_eval = selected_harmonics.float().transpose(-1, -2)
    elif harmonics_shape[-2] == 3:
        sh_coefficient_count = harmonics_shape[-1]
        harmonics_for_eval = selected_harmonics.float()
    else:
        raise RuntimeError(
            "New Gaussian SH must have an RGB dimension of size 3; "
            f"got shape={tuple(harmonics_shape)}."
        )
    sh_degree = int(sh_coefficient_count**0.5) - 1
    if (sh_degree + 1) ** 2 != sh_coefficient_count or not 0 <= sh_degree <= 4:
        raise RuntimeError(
            "New Gaussian SH coefficient count must be a supported square "
            f"number, got {sh_coefficient_count}."
        )
    camera_centers = camera_to_world.detach().float()[:, :3, 3].index_select(
        0, batch_indices
    )
    world_direction = F.normalize(
        selected_means.float() - camera_centers, dim=-1, eps=1e-8
    )
    sh_color = (
        sh_utils.eval_sh(sh_degree, harmonics_for_eval, world_direction) + 0.5
    ).clamp(0.0, 1.0)
    relative_depth = (z - current_depth_safe) / current_depth_safe
    relative_old_depth = torch.where(
        history_valid_selected,
        (z - old_depth_safe) / old_depth_safe,
        torch.zeros_like(z),
    )
    descriptor = torch.cat(
        [
            sampled_feature.float(),
            sampled_current_rgb,
            sampled_old_rgb,
            sampled_current_rgb - sampled_old_rgb,
            current_depth_safe.log().unsqueeze(-1).clamp(-8.0, 8.0),
            z.log().unsqueeze(-1).clamp(-8.0, 8.0),
            relative_depth.unsqueeze(-1).clamp(-2.0, 2.0),
            relative_old_depth.unsqueeze(-1).clamp(-2.0, 2.0),
            sampled_confidence.unsqueeze(-1).clamp(0.0, 1.0),
            sampled_old_alpha.unsqueeze(-1).clamp(0.0, 1.0),
            selected_opacity.float().unsqueeze(-1),
            selected_scales.float().clamp_min(1e-8).log().clamp(-12.0, 2.0),
            F.normalize(camera_points, dim=-1, eps=1e-8),
            sh_color,
            history_valid_selected.float().unsqueeze(-1),
            projected_radius.log1p().unsqueeze(-1),
            (depth_extent / z).unsqueeze(-1).clamp(0.0, 2.0),
            history_consistent_selected.float().unsqueeze(-1),
            history_in_front_selected.float().unsqueeze(-1),
        ],
        dim=-1,
    )
    descriptor = torch.nan_to_num(
        descriptor, nan=0.0, posinf=8.0, neginf=-8.0
    ).to(current_feature.dtype)
    if descriptor.shape != (candidate_count, expected_dim):
        raise RuntimeError(
            "Projected new-GS descriptor shape mismatch: "
            f"got={tuple(descriptor.shape)}, "
            f"expected={(candidate_count, expected_dim)}."
        )

    history_valid = torch.zeros((b, n), device=device, dtype=torch.bool)
    history_consistent = torch.zeros_like(history_valid)
    history_in_front = torch.zeros_like(history_valid)
    camera_depth = torch.zeros((b, n), device=device, dtype=torch.float32)
    history_valid.reshape(-1).index_copy_(
        0, flat_candidate_indices, history_valid_selected
    )
    history_consistent.reshape(-1).index_copy_(
        0, flat_candidate_indices, history_consistent_selected
    )
    history_in_front.reshape(-1).index_copy_(
        0, flat_candidate_indices, history_in_front_selected
    )
    camera_depth.reshape(-1).index_copy_(
        0,
        flat_candidate_indices,
        z.detach().to(camera_depth.dtype),
    )
    return ProjectedNewGSEvidence(
        descriptor=descriptor,
        flat_candidate_indices=flat_candidate_indices,
        candidate_mask=candidate_mask.reshape(b, n),
        history_valid=history_valid,
        history_consistent=history_consistent,
        history_in_front=history_in_front,
        camera_depth=camera_depth,
        opacity_gradient_features=(
            gradient_features.reshape(b * n, OPACITY_GRADIENT_FEATURE_DIM)
            .index_select(0, flat_candidate_indices)
            .to(current_feature.dtype)
            if has_opacity_gradient_evidence
            else None
        ),
        opacity_gradient_signal=(
            gradient_signal.reshape(-1).index_select(0, flat_candidate_indices)
            if has_opacity_gradient_evidence
            else None
        ),
        opacity_gradient_current_signal=(
            gradient_current_signal.reshape(-1).index_select(
                0, flat_candidate_indices
            )
            if has_opacity_gradient_evidence
            else None
        ),
        opacity_gradient_replay_signal=(
            gradient_replay_signal.reshape(-1).index_select(
                0, flat_candidate_indices
            )
            if has_opacity_gradient_evidence
            else None
        ),
        mean_gradient_features=(
            mean_gradient_features.reshape(
                b * n, MEAN_GRADIENT_FEATURE_DIM
            ).index_select(0, flat_candidate_indices).to(current_feature.dtype)
            if mean_gradient_features is not None
            else None
        ),
        mean_gradient_signal=(
            mean_gradient_signal.reshape(b * n, 3).index_select(
                0, flat_candidate_indices
            )
            if mean_gradient_signal is not None
            else None
        ),
        mean_gradient_current_signal=(
            mean_gradient_current_signal.reshape(b * n, 3).index_select(
                0, flat_candidate_indices
            )
            if mean_gradient_current_signal is not None
            else None
        ),
        mean_gradient_replay_signal=(
            mean_gradient_replay_signal.reshape(b * n, 3).index_select(
                0, flat_candidate_indices
            )
            if mean_gradient_replay_signal is not None
            else None
        ),
    )


class ProjectedGSResidualHead(nn.Module):
    """Shared pointwise decoder for GS-centric historical parameter updates."""

    def __init__(
        self,
        feature_dim: int,
        harmonic_dim: int,
        hidden_dim: int = 128,
        chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        input_dim = int(feature_dim) + PROJECTED_GS_STATE_DIM
        self.input_dim = input_dim
        self.harmonic_dim = int(harmonic_dim)
        if self.harmonic_dim <= 0:
            raise ValueError("Projected GS harmonic_dim must be positive.")
        self.chunk_size = max(1, int(chunk_size))
        self.decoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.opacity_gradient_adapter = nn.Sequential(
            nn.LayerNorm(OPACITY_GRADIENT_FEATURE_DIM),
            nn.Linear(OPACITY_GRADIENT_FEATURE_DIM, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mean_gradient_adapter = nn.Sequential(
            nn.LayerNorm(MEAN_GRADIENT_FEATURE_DIM),
            nn.Linear(MEAN_GRADIENT_FEATURE_DIM, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.new_mean_gradient_adapter = nn.Sequential(
            nn.LayerNorm(MEAN_GRADIENT_FEATURE_DIM),
            nn.Linear(MEAN_GRADIENT_FEATURE_DIM, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = 12 + self.harmonic_dim
        self.output = nn.Linear(hidden_dim, self.output_dim)
        # Match the historical branch's parameter residuals, excluding only
        # front-surface decay, which does not apply to newly appended GS.
        self.new_output_dim = 11 + self.harmonic_dim
        self.new_output = nn.Linear(hidden_dim, self.new_output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.new_output.weight)
        nn.init.zeros_(self.new_output.bias)
        nn.init.zeros_(self.opacity_gradient_adapter[-1].weight)
        nn.init.zeros_(self.opacity_gradient_adapter[-1].bias)
        nn.init.zeros_(self.mean_gradient_adapter[-1].weight)
        nn.init.zeros_(self.mean_gradient_adapter[-1].bias)
        nn.init.zeros_(self.new_mean_gradient_adapter[-1].weight)
        nn.init.zeros_(self.new_mean_gradient_adapter[-1].bias)
        with torch.no_grad():
            self.output.bias[10 + self.harmonic_dim] = -4.0
            self.output.bias[11 + self.harmonic_dim] = 2.0
            self.new_output.bias[10 + self.harmonic_dim] = 2.0

    def _decode(
        self,
        descriptor: torch.Tensor,
        opacity_gradient_features: torch.Tensor | None,
        output_head: nn.Linear,
        mean_gradient_features: torch.Tensor | None = None,
        mean_gradient_adapter: nn.Module | None = None,
        inactive_mean_gradient_adapter: nn.Module | None = None,
        mean_gradient_output_channels: int | None = None,
    ) -> torch.Tensor:
        selected = descriptor
        selected_gradient = opacity_gradient_features
        if selected_gradient is not None and selected_gradient.shape != (
            descriptor.shape[0], OPACITY_GRADIENT_FEATURE_DIM
        ):
            raise RuntimeError(
                "Opacity-gradient feature mismatch: "
                f"got={tuple(selected_gradient.shape)}, "
                f"expected={(descriptor.shape[0], OPACITY_GRADIENT_FEATURE_DIM)}."
            )
        no_candidates = selected.shape[0] == 0
        selected_mean_gradient = mean_gradient_features
        if selected_mean_gradient is not None and selected_mean_gradient.shape != (
            descriptor.shape[0],
            MEAN_GRADIENT_FEATURE_DIM,
        ):
            raise RuntimeError(
                "Mean-gradient feature mismatch: "
                f"got={tuple(selected_mean_gradient.shape)}, "
                f"expected={(descriptor.shape[0], MEAN_GRADIENT_FEATURE_DIM)}."
            )
        if selected_mean_gradient is not None and mean_gradient_adapter is None:
            raise RuntimeError(
                "Mean-gradient features require a branch-specific adapter."
            )
        if mean_gradient_output_channels is not None and not (
            0 < mean_gradient_output_channels <= output_head.out_features
        ):
            raise ValueError(
                "mean_gradient_output_channels must select a non-empty output "
                f"prefix, got {mean_gradient_output_channels}."
            )
        if no_candidates:
            # Keep both the shared trunk and selected output head in the DDP
            # graph for batches where this branch has no append candidates.
            selected = descriptor.new_zeros((1, self.input_dim))
            if selected_gradient is not None:
                selected_gradient = descriptor.new_zeros(
                    (1, OPACITY_GRADIENT_FEATURE_DIM)
                )
            if selected_mean_gradient is not None:
                selected_mean_gradient = descriptor.new_zeros(
                    (1, MEAN_GRADIENT_FEATURE_DIM)
                )
        selected_outputs = []
        for start in range(0, selected.shape[0], self.chunk_size):
            chunk = selected[start : start + self.chunk_size]
            hidden = self.decoder(chunk)
            if selected_gradient is not None:
                gradient_chunk = selected_gradient[
                    start : start + self.chunk_size
                ]
                hidden = hidden + self.opacity_gradient_adapter(gradient_chunk)
            else:
                # Keep shared adapter parameters in the DDP graph while fully
                # bypassing their learned weights and biases for this branch.
                hidden = hidden + sum(
                    parameter.sum() * 0.0
                    for parameter in self.opacity_gradient_adapter.parameters()
                )
            mean_gradient_hidden = None
            if selected_mean_gradient is not None:
                mean_gradient_hidden = mean_gradient_adapter(
                    selected_mean_gradient[start : start + self.chunk_size]
                )
                if mean_gradient_output_channels is None:
                    hidden = hidden + mean_gradient_hidden
            elif mean_gradient_adapter is not None:
                hidden = hidden + sum(
                    parameter.sum() * 0.0
                    for parameter in mean_gradient_adapter.parameters()
                )
            if inactive_mean_gradient_adapter is not None:
                hidden = hidden + sum(
                    parameter.sum() * 0.0
                    for parameter in inactive_mean_gradient_adapter.parameters()
                )
            chunk_output = output_head(hidden)
            if (
                mean_gradient_hidden is not None
                and mean_gradient_output_channels is not None
            ):
                channels = mean_gradient_output_channels
                mean_gradient_output = F.linear(
                    mean_gradient_hidden,
                    output_head.weight[:channels],
                    bias=None,
                )
                chunk_output = torch.cat(
                    (
                        chunk_output[:, :channels] + mean_gradient_output,
                        chunk_output[:, channels:],
                    ),
                    dim=-1,
                )
            selected_outputs.append(chunk_output)
        selected_output = torch.cat(selected_outputs, dim=0)
        if no_candidates:
            selected_output = selected_output[:0] + selected_output.sum() * 0.0
        return selected_output

    def forward(self, evidence: ProjectedGSEvidence) -> ProjectedGSResidualPrediction:
        descriptor = evidence.descriptor
        if descriptor.ndim != 2 or descriptor.shape[-1] != self.input_dim:
            raise RuntimeError(
                "Projected GS head input mismatch: "
                f"got={tuple(descriptor.shape)}, expected=[candidates,{self.input_dim}]."
            )
        batch_size, gaussian_count = evidence.candidate_mask.shape
        candidate_indices = evidence.flat_candidate_indices
        if descriptor.shape[0] != candidate_indices.numel():
            raise RuntimeError(
                "Projected GS packed evidence mismatch: "
                f"descriptors={descriptor.shape[0]}, indices={candidate_indices.numel()}."
            )
        selected_output = self._decode(
            descriptor,
            evidence.opacity_gradient_features,
            self.output,
            mean_gradient_features=evidence.mean_gradient_features,
            mean_gradient_adapter=self.mean_gradient_adapter,
            inactive_mean_gradient_adapter=self.new_mean_gradient_adapter,
        )
        harmonic_end = 10 + self.harmonic_dim
        return ProjectedGSResidualPrediction(
            delta_mean_camera=selected_output[:, 0:3],
            delta_rotation=selected_output[:, 3:6],
            delta_log_scale=selected_output[:, 6:9],
            delta_opacity_logit=selected_output[:, 9],
            delta_harmonics=selected_output[:, 10:harmonic_end],
            front_decay_logit=selected_output[:, harmonic_end],
            update_gate_logit=selected_output[:, harmonic_end + 1],
        )

    def forward_new(
        self,
        evidence: ProjectedNewGSEvidence,
    ) -> ProjectedNewGSResidualPrediction:
        descriptor = evidence.descriptor
        if descriptor.ndim != 2 or descriptor.shape[-1] != self.input_dim:
            raise RuntimeError(
                "Projected new-GS head input mismatch: "
                f"got={tuple(descriptor.shape)}, "
                f"expected=[candidates,{self.input_dim}]."
            )
        candidate_indices = evidence.flat_candidate_indices
        if descriptor.shape[0] != candidate_indices.numel():
            raise RuntimeError(
                "Projected new-GS packed evidence mismatch: "
                f"descriptors={descriptor.shape[0]}, "
                f"indices={candidate_indices.numel()}."
            )
        selected_output = self._decode(
            descriptor,
            evidence.opacity_gradient_features,
            self.new_output,
            mean_gradient_features=evidence.mean_gradient_features,
            mean_gradient_adapter=self.new_mean_gradient_adapter,
            inactive_mean_gradient_adapter=self.mean_gradient_adapter,
            mean_gradient_output_channels=3,
        )
        harmonic_end = 10 + self.harmonic_dim
        return ProjectedNewGSResidualPrediction(
            delta_mean_camera=selected_output[:, 0:3],
            delta_rotation=selected_output[:, 3:6],
            delta_log_scale=selected_output[:, 6:9],
            delta_opacity_logit=selected_output[:, 9],
            delta_harmonics=selected_output[:, 10:harmonic_end],
            update_gate_logit=selected_output[:, harmonic_end],
        )


SYMMETRIC_GS_UPDATE_STAT_KEYS = (
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

ANCHOR_CHILD_STAT_KEYS = (
    "eligible_anchor_count",
    "expanded_anchor_count",
    "spawned_child_count",
    "active_anchor_count",
    "active_child_count",
    "selected_gradient_score_mean",
    "selected_coherence_mean",
    "map_count_before",
    "map_count_after",
)


class SymmetricGSResidualHead(nn.Module):
    """Shared learned optimizer for an appended map and equal-view gradients."""

    def __init__(
        self,
        harmonic_dim: int,
        hidden_dim: int = 128,
        chunk_size: int = 65536,
        visual_enabled: bool = False,
        visual_feature_dim: int = 24,
        visual_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.harmonic_dim = int(harmonic_dim)
        if self.harmonic_dim <= 0:
            raise ValueError("Symmetric GS harmonic_dim must be positive.")
        self.chunk_size = max(1, int(chunk_size))
        self.visual_enabled = bool(visual_enabled)
        self.visual_encoder = (
            SymmetricVisualEvidenceEncoder(
                feature_dim=visual_feature_dim,
                hidden_dim=visual_hidden_dim,
            )
            if self.visual_enabled
            else None
        )
        self.visual_dim = int(visual_feature_dim) + 3 if self.visual_enabled else 0
        # opacity: 3*1+5, mean: 3*3+5, harmonics: 3*D+5
        gradient_dim = 27 + 3 * self.harmonic_dim
        # opacity, log-scale, quaternion, SH, observations, generation, is_new
        state_dim = 11 + self.harmonic_dim
        self.input_dim = state_dim + gradient_dim + self.visual_dim
        self.output_dim = 15 + self.harmonic_dim

        def make_attribute_decoder(output_dim: int) -> nn.Sequential:
            decoder = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )
            nn.init.zeros_(decoder[-1].weight)
            nn.init.zeros_(decoder[-1].bias)
            return decoder

        self.attribute_decoders = nn.ModuleDict(
            {
                "mean": make_attribute_decoder(3),
                "shape": make_attribute_decoder(6),
                "opacity": make_attribute_decoder(1),
                "harmonics": make_attribute_decoder(self.harmonic_dim),
                "gates": make_attribute_decoder(5),
            }
        )
        with torch.no_grad():
            self.attribute_decoders["gates"][-1].bias.fill_(2.0)

    def forward(
        self,
        gaussians: Gaussians,
        observation_count: torch.Tensor,
        split_generation: torch.Tensor,
        evidence: SymmetricMultiviewGradientEvidence,
        is_new: torch.Tensor,
    ) -> SymmetricGSResidualPrediction:
        batch, count = gaussians.opacities.shape
        expected = (batch, count)
        for name, value in (
            ("observation_count", observation_count),
            ("split_generation", split_generation),
            ("active_mask", evidence.active_mask),
            ("is_new", is_new),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"Symmetric GS {name} must be {expected}, got "
                    f"{tuple(value.shape)}."
                )
        harmonic_flat = _flatten_harmonics(gaussians.harmonics).float()
        if harmonic_flat.shape[-1] != self.harmonic_dim:
            raise ValueError(
                "Symmetric GS harmonic dimension mismatch: "
                f"got={harmonic_flat.shape[-1]}, expected={self.harmonic_dim}."
            )
        expected_gradient_dim = 27 + 3 * self.harmonic_dim
        if evidence.features.shape != (batch, count, expected_gradient_dim):
            raise ValueError(
                "Symmetric gradient feature mismatch: "
                f"got={tuple(evidence.features.shape)}, "
                f"expected={(batch, count, expected_gradient_dim)}."
            )
        if self.visual_enabled:
            expected_visual_shape = (batch, count, self.visual_dim)
            if evidence.visual_features is None:
                raise ValueError(
                    "Symmetric visual evidence is enabled but was not provided."
                )
            if evidence.visual_features.shape != expected_visual_shape:
                raise ValueError(
                    "Symmetric visual feature mismatch: "
                    f"got={tuple(evidence.visual_features.shape)}, "
                    f"expected={expected_visual_shape}."
                )
        elif evidence.visual_features is not None:
            raise ValueError(
                "Symmetric visual evidence was provided to a gradient-only head."
            )

        opacity_logit = torch.logit(
            gaussians.opacities.float().clamp(1e-4, 1.0 - 1e-4)
        ).unsqueeze(-1)
        state_features = torch.cat(
            [
                opacity_logit,
                gaussians.scales.float().clamp_min(1e-8).log().clamp(-12.0, 4.0),
                F.normalize(gaussians.rotations.float(), dim=-1, eps=1e-8),
                harmonic_flat.clamp(-8.0, 8.0),
                observation_count.float().clamp_min(0.0).log1p().unsqueeze(-1),
                # 255 marks non-recursive anchor children. Keep the state
                # descriptor bounded while preserving root child counts.
                split_generation.float().clamp_max(8.0).unsqueeze(-1),
                is_new.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        descriptor_parts = [
            state_features,
            evidence.features.to(state_features.dtype),
        ]
        if evidence.visual_features is not None:
            descriptor_parts.append(
                evidence.visual_features.to(state_features.dtype)
            )
        descriptor = torch.cat(descriptor_parts, dim=-1)
        descriptor = torch.nan_to_num(
            descriptor, nan=0.0, posinf=8.0, neginf=-8.0
        )
        flat_mask = evidence.active_mask.reshape(-1)
        packed = descriptor.reshape(-1, self.input_dim)[flat_mask]
        no_candidates = packed.shape[0] == 0
        if no_candidates:
            packed = descriptor.new_zeros((1, self.input_dim))
        outputs = {name: [] for name in self.attribute_decoders}
        for start in range(0, packed.shape[0], self.chunk_size):
            chunk = packed[start : start + self.chunk_size]
            for name, decoder in self.attribute_decoders.items():
                outputs[name].append(decoder(chunk))
        outputs = {
            name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()
        }
        if no_candidates:
            outputs = {
                name: output[:0] + output.sum() * 0.0
                for name, output in outputs.items()
            }
        shape = outputs["shape"]
        return SymmetricGSResidualPrediction(
            delta_mean_reference=outputs["mean"],
            delta_rotation=shape[:, :3],
            delta_log_scale=shape[:, 3:6],
            delta_opacity_logit=outputs["opacity"][:, 0],
            delta_harmonics=outputs["harmonics"],
            update_gate_logits=outputs["gates"],
        )


def apply_symmetric_multiview_residual(
    state: "StreamingGaussianState",
    evidence: SymmetricMultiviewGradientEvidence,
    prediction: SymmetricGSResidualPrediction,
    reference_camera_to_world: torch.Tensor,
    *,
    max_mean_step_scale: float = 0.10,
    max_rotation_step: float = 0.01,
    max_log_scale_step: float = 0.02,
    max_opacity_logit_step: float = 0.10,
    max_harmonics_step: float = 0.05,
    unbounded_updates: bool = False,
) -> tuple["StreamingGaussianState", dict[str, torch.Tensor]]:
    """Apply bounded or direct residuals to GS with symmetric evidence."""
    batch, count = state.gaussians.opacities.shape
    if reference_camera_to_world.shape != (batch, 4, 4):
        raise ValueError(
            "Symmetric update reference cameras must be [B,4,4], got "
            f"{tuple(reference_camera_to_world.shape)}."
        )
    flat_mask = evidence.active_mask.reshape(-1)
    flat_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
    candidate_count = flat_indices.numel()
    expected_shapes = {
        "delta_mean_reference": (candidate_count, 3),
        "delta_rotation": (candidate_count, 3),
        "delta_log_scale": (candidate_count, 3),
        "delta_opacity_logit": (candidate_count,),
        "delta_harmonics": (candidate_count, prediction.delta_harmonics.shape[-1]),
        "update_gate_logits": (candidate_count, 5),
    }
    for name, expected in expected_shapes.items():
        value = getattr(prediction, name)
        if value.shape != expected:
            raise ValueError(
                f"Symmetric prediction {name} shape mismatch: "
                f"got={tuple(value.shape)}, expected={expected}."
            )
    if not unbounded_updates:
        for name, value in (
            ("max_mean_step_scale", max_mean_step_scale),
            ("max_rotation_step", max_rotation_step),
            ("max_log_scale_step", max_log_scale_step),
            ("max_opacity_logit_step", max_opacity_logit_step),
            ("max_harmonics_step", max_harmonics_step),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"Symmetric {name} must be finite and non-negative."
                )

    gaussians = state.gaussians
    if candidate_count == 0:
        # Preserve a zero-gradient graph from the rendered map to every
        # attribute decoder. Statistics are detached from the training loss,
        # so returning the original state here would leave the whole head
        # unused on batches where no GS receives visible gradient evidence.
        zero = sum(
            value.sum() * 0.0
            for value in (
                prediction.delta_mean_reference,
                prediction.delta_rotation,
                prediction.delta_log_scale,
                prediction.delta_opacity_logit,
                prediction.delta_harmonics,
                prediction.update_gate_logits,
            )
        )
        state = StreamingGaussianState(
            gaussians=Gaussians(
                means=gaussians.means + zero,
                harmonics=gaussians.harmonics + zero,
                opacities=gaussians.opacities + zero,
                scales=gaussians.scales + zero,
                rotations=gaussians.rotations + zero,
            ),
            stable_ids=state.stable_ids,
            observation_count=state.observation_count,
            point_map=state.point_map,
            front_conflict_count=state.front_conflict_count,
            split_generation=state.split_generation,
        )
        stats = {key: zero for key in SYMMETRIC_GS_UPDATE_STAT_KEYS}
        return state, stats

    batch_indices = torch.div(flat_indices, count, rounding_mode="floor")
    local_indices = flat_indices.remainder(count)
    selected_scales = gaussians.scales[batch_indices, local_indices].float()
    if unbounded_updates:
        gates = torch.ones_like(prediction.update_gate_logits)
        delta_mean_reference = prediction.delta_mean_reference
        delta_rotation = prediction.delta_rotation
        delta_log_scale = prediction.delta_log_scale
        delta_opacity_logit = prediction.delta_opacity_logit
        delta_harmonics = prediction.delta_harmonics
    else:
        gates = prediction.update_gate_logits.sigmoid()
        consistency = (
            evidence.mean_coherence.reshape(-1).index_select(0, flat_indices)
            * (
                1.0
                - evidence.mean_conflict_ratio.reshape(-1).index_select(
                    0, flat_indices
                )
            )
        ).clamp(0.0, 1.0)
        geometry_gates = gates[:, :3] * consistency.unsqueeze(-1)
        mean_bound = (
            selected_scales.amax(dim=-1, keepdim=True)
            * float(max_mean_step_scale)
        )
        delta_mean_reference = (
            prediction.delta_mean_reference.tanh()
            * mean_bound
            * geometry_gates[:, 0:1]
        )
        delta_rotation = (
            prediction.delta_rotation.tanh()
            * float(max_rotation_step)
            * geometry_gates[:, 1:2]
        )
        delta_log_scale = (
            prediction.delta_log_scale.tanh()
            * float(max_log_scale_step)
            * geometry_gates[:, 2:3]
        )
        delta_opacity_logit = (
            prediction.delta_opacity_logit.tanh()
            * float(max_opacity_logit_step)
            * gates[:, 3]
        )
        delta_harmonics = (
            prediction.delta_harmonics.tanh()
            * float(max_harmonics_step)
            * gates[:, 4:5]
        )

    reference_rotation = reference_camera_to_world[
        batch_indices, :3, :3
    ].float()
    delta_mean_world = torch.bmm(
        reference_rotation, delta_mean_reference.unsqueeze(-1)
    ).squeeze(-1)

    def scatter_selected(base: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        flat = base.reshape(batch * count, *base.shape[2:]).clone()
        flat.index_copy_(0, flat_indices, selected.to(flat.dtype))
        return flat.reshape_as(base)

    selected_means = gaussians.means[batch_indices, local_indices].float()
    selected_rotations = gaussians.rotations[batch_indices, local_indices].float()
    selected_opacity = gaussians.opacities[batch_indices, local_indices].float()
    selected_harmonics = _flatten_harmonics(gaussians.harmonics)[
        batch_indices, local_indices
    ].float()
    updated_means = scatter_selected(
        gaussians.means, selected_means + delta_mean_world
    )
    updated_rotations = scatter_selected(
        gaussians.rotations,
        F.normalize(
            _quat_multiply_xyzw(
                _axis_angle_to_quaternion_xyzw(delta_rotation),
                selected_rotations,
            ),
            dim=-1,
            eps=1e-8,
        ),
    )
    updated_scales = scatter_selected(
        gaussians.scales,
        selected_scales * torch.exp(delta_log_scale),
    )
    updated_opacities = scatter_selected(
        gaussians.opacities,
        torch.sigmoid(
            torch.logit(selected_opacity.clamp(1e-4, 1.0 - 1e-4))
            + delta_opacity_logit
        ),
    )
    updated_harmonics_flat = scatter_selected(
        _flatten_harmonics(gaussians.harmonics),
        selected_harmonics + delta_harmonics,
    )
    observed_views = evidence.visible_view_count.to(
        device=state.observation_count.device,
        dtype=state.observation_count.dtype,
    )
    updated_observation_count = torch.maximum(
        state.observation_count,
        observed_views,
    )
    updated_state = StreamingGaussianState(
        gaussians=Gaussians(
            means=updated_means,
            rotations=updated_rotations,
            scales=updated_scales,
            opacities=updated_opacities,
            harmonics=_restore_harmonics(
                updated_harmonics_flat, gaussians.harmonics
            ),
        ),
        stable_ids=state.stable_ids,
        observation_count=updated_observation_count,
        point_map=state.point_map,
        front_conflict_count=state.front_conflict_count,
        split_generation=state.split_generation,
    )

    # Counts can exceed FP16's finite range for dense maps.
    selected_count = torch.tensor(
        float(candidate_count), device=gates.device, dtype=torch.float32
    )
    total_active = (state.stable_ids >= 0).float().sum().clamp_min(1.0)
    opacity_shift = (
        updated_opacities.reshape(-1).index_select(0, flat_indices)
        - selected_opacity.to(updated_opacities.dtype)
    ).abs()
    scale_relative_shift = (torch.exp(delta_log_scale) - 1.0).abs().mean(dim=-1)
    residual_energy = (
        delta_mean_world.square().sum(dim=-1)
        + delta_rotation.square().sum(dim=-1)
        + delta_log_scale.square().sum(dim=-1)
        + delta_opacity_logit.square()
        + delta_harmonics.square().mean(dim=-1)
    ).mean()
    stats = {
        "active_count": selected_count,
        "active_ratio": selected_count / total_active,
        "visible_views_mean": evidence.visible_view_count.reshape(-1)
        .index_select(0, flat_indices)
        .float()
        .mean(),
        "mean_coherence": evidence.mean_coherence.reshape(-1)
        .index_select(0, flat_indices)
        .float()
        .mean(),
        "mean_conflict_ratio": evidence.mean_conflict_ratio.reshape(-1)
        .index_select(0, flat_indices)
        .float()
        .mean(),
        "gate_mean": gates.mean(),
        "mean_shift_mean": delta_mean_world.norm(dim=-1).mean(),
        "rotation_shift_mean": delta_rotation.norm(dim=-1).mean(),
        "scale_relative_shift_mean": scale_relative_shift.mean(),
        "opacity_shift_mean": opacity_shift.mean(),
        "harmonics_shift_mean": delta_harmonics.norm(dim=-1).mean(),
        "visual_visible_views_mean": (
            evidence.visual_view_count.reshape(-1)
            .index_select(0, flat_indices)
            .float()
            .mean()
            if evidence.visual_view_count is not None
            else selected_count * 0.0
        ),
        "visual_weight_mean": (
            evidence.visual_weight_sum.reshape(-1)
            .index_select(0, flat_indices)
            .float()
            .mean()
            if evidence.visual_weight_sum is not None
            else selected_count * 0.0
        ),
        "visual_feature_norm": (
            evidence.visual_features.reshape(
                batch * count, evidence.visual_features.shape[-1]
            )
            .index_select(0, flat_indices)
            .float()
            .norm(dim=-1)
            .mean()
            if evidence.visual_features is not None
            else selected_count * 0.0
        ),
        "regularization": residual_energy,
    }
    return updated_state, stats


def apply_projected_new_gaussian_residual(
    current: Gaussians,
    evidence: ProjectedNewGSEvidence,
    prediction: ProjectedNewGSResidualPrediction,
    camera_to_world: torch.Tensor,
    opacity_logit_scale: float = 1.0,
    mean_update_mode: str = "relative_depth",
    raw_scale_residual: bool = False,
    raw_opacity_residual: bool = False,
    raw_rotation_residual: bool = False,
    raw_harmonics_residual: bool = False,
    mean_relative_scale: float = 0.02,
    rotation_scale: float = 0.05,
    harmonics_scale: float = 0.10,
) -> tuple[Gaussians, dict[str, torch.Tensor]]:
    """Refine append-selected anchors with the historical-GS parameterization."""
    b, n = current.means.shape[:2]
    mean_update_mode = str(mean_update_mode).lower()
    if mean_update_mode not in {"absolute", "relative_depth"}:
        raise ValueError(
            "mean_update_mode must be absolute or relative_depth; "
            f"got {mean_update_mode!r}."
        )
    flat_indices = evidence.flat_candidate_indices.long()
    candidate_count_int = flat_indices.numel()
    harmonic_dim = current.harmonics.shape[-2] * current.harmonics.shape[-1]
    expected_vectors = (
        ("delta_mean_camera", prediction.delta_mean_camera, 3),
        ("delta_rotation", prediction.delta_rotation, 3),
        ("delta_log_scale", prediction.delta_log_scale, 3),
        ("delta_harmonics", prediction.delta_harmonics, harmonic_dim),
    )
    for name, value, channels in expected_vectors:
        if value.shape != (candidate_count_int, channels):
            raise RuntimeError(
                f"Projected new-GS {name} shape mismatch: "
                f"got={tuple(value.shape)}, "
                f"expected={(candidate_count_int, channels)}."
            )
    for name, value in (
        ("delta_opacity_logit", prediction.delta_opacity_logit),
        ("update_gate_logit", prediction.update_gate_logit),
    ):
        if value.shape != (candidate_count_int,):
            raise RuntimeError(
                f"Projected new-GS {name} shape mismatch: "
                f"got={tuple(value.shape)}, "
                f"expected={(candidate_count_int,)}."
            )
    if candidate_count_int and (
        int(flat_indices.min()) < 0 or int(flat_indices.max()) >= b * n
    ):
        raise RuntimeError("Projected new-GS candidate indices are out of range.")

    gate = prediction.update_gate_logit.float().sigmoid()
    batch_indices = torch.div(flat_indices, n, rounding_mode="floor")
    means_flat = current.means.reshape(b * n, 3)
    selected_means = means_flat.index_select(0, flat_indices).float()
    raw_mean = prediction.delta_mean_camera.float()
    if mean_update_mode == "relative_depth":
        camera_depth = (
            evidence.camera_depth.reshape(-1)
            .index_select(0, flat_indices)
            .float()
            .clamp_min(1e-4)
        )
        delta_mean_camera = (
            raw_mean.tanh()
            * min(max(float(mean_relative_scale), 0.0), 0.999)
            * camera_depth.unsqueeze(-1)
        )
    else:
        delta_mean_camera = raw_mean
    delta_mean_camera = delta_mean_camera * gate.unsqueeze(-1)
    camera_rotation = camera_to_world.detach().float()[batch_indices, :3, :3]
    delta_mean_world = torch.bmm(
        delta_mean_camera.unsqueeze(1), camera_rotation.transpose(1, 2)
    ).squeeze(1)
    updated_means_selected = selected_means + delta_mean_world

    raw_rotation = prediction.delta_rotation.float()
    delta_rotation = (
        raw_rotation
        if raw_rotation_residual
        else raw_rotation.tanh()
        * min(max(float(rotation_scale), 0.0), 0.999)
    )
    delta_rotation = delta_rotation * gate.unsqueeze(-1)
    rotations_flat = current.rotations.reshape(b * n, 4)
    selected_rotations = rotations_flat.index_select(0, flat_indices).float()
    updated_rotations_selected = F.normalize(
        _quat_multiply_xyzw(
            _axis_angle_to_quaternion_xyzw(delta_rotation),
            selected_rotations,
        ),
        dim=-1,
        eps=1e-8,
    )

    raw_scale = prediction.delta_log_scale.float()
    delta_log_scale = (
        raw_scale if raw_scale_residual else raw_scale.tanh()
    ) * gate.unsqueeze(-1)
    scales_flat = current.scales.reshape(b * n, 3)
    selected_scales = scales_flat.index_select(0, flat_indices).float()
    updated_scales_selected = (
        selected_scales * torch.exp(delta_log_scale.clamp(-2.0, 2.0))
    ).clamp(1e-6, 0.1)

    raw_opacity = prediction.delta_opacity_logit.float()
    delta_opacity = (
        raw_opacity if raw_opacity_residual else raw_opacity.tanh()
    )
    delta_opacity = (
        delta_opacity * max(0.0, float(opacity_logit_scale)) * gate
    )
    opacities_flat = current.opacities.reshape(-1)
    selected_opacities = opacities_flat.index_select(0, flat_indices).float()
    updated_opacities_selected = torch.sigmoid(
        torch.logit(selected_opacities.clamp(1e-5, 1.0 - 1e-5))
        + delta_opacity
    )

    raw_harmonics = prediction.delta_harmonics.float()
    delta_harmonics = (
        raw_harmonics
        if raw_harmonics_residual
        else raw_harmonics.tanh()
        * min(max(float(harmonics_scale), 0.0), 0.999)
    )
    delta_harmonics = delta_harmonics * gate.unsqueeze(-1)
    harmonics_flat = _flatten_harmonics(current.harmonics).reshape(
        b * n, harmonic_dim
    )
    selected_harmonics = harmonics_flat.index_select(0, flat_indices).float()
    updated_harmonics_selected = selected_harmonics + delta_harmonics

    means = means_flat.index_copy(
        0, flat_indices, updated_means_selected.to(means_flat.dtype)
    ).reshape_as(current.means)
    rotations = rotations_flat.index_copy(
        0,
        flat_indices,
        updated_rotations_selected.to(rotations_flat.dtype),
    ).reshape_as(current.rotations)
    scales = scales_flat.index_copy(
        0, flat_indices, updated_scales_selected.to(scales_flat.dtype)
    ).reshape_as(current.scales)
    opacities = opacities_flat.index_copy(
        0,
        flat_indices,
        updated_opacities_selected.to(opacities_flat.dtype),
    ).reshape_as(current.opacities)
    harmonics = harmonics_flat.index_copy(
        0,
        flat_indices,
        updated_harmonics_selected.to(harmonics_flat.dtype),
    ).reshape_as(current.harmonics)

    raw_energy = (
        raw_mean.square().sum(dim=-1)
        + raw_rotation.square().sum(dim=-1)
        + raw_scale.square().sum(dim=-1)
        + raw_opacity.square()
        + raw_harmonics.square().mean(dim=-1)
    )
    candidate_count = prediction.update_gate_logit.new_tensor(
        float(candidate_count_int)
    ).clamp_min(1.0)
    regularization = raw_energy.sum() / candidate_count
    mean_shift = (updated_means_selected - selected_means).norm(dim=-1)
    rotation_shift = delta_rotation.norm(dim=-1)
    scale_relative_shift = (
        updated_scales_selected / selected_scales.clamp_min(1e-8) - 1.0
    ).abs().mean(dim=-1)
    opacity_shift = (updated_opacities_selected - selected_opacities).abs()
    harmonics_shift = delta_harmonics.abs().mean(dim=-1)
    return (
        Gaussians(
            means=means,
            harmonics=harmonics,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        ),
        {
            "regularization": regularization,
            "candidate_ratio": evidence.candidate_mask.float().mean(),
            "gate_mean": gate.sum() / candidate_count,
            "raw_mean": raw_mean.norm(dim=-1).sum() / candidate_count,
            "mean_shift_mean": mean_shift.sum() / candidate_count,
            "raw_rotation": raw_rotation.norm(dim=-1).sum() / candidate_count,
            "rotation_shift_mean": rotation_shift.sum() / candidate_count,
            "raw_log_scale": raw_scale.abs().mean(dim=-1).sum()
            / candidate_count,
            "scale_relative_shift_mean": scale_relative_shift.sum()
            / candidate_count,
            "raw_opacity_logit": raw_opacity.abs().sum() / candidate_count,
            "opacity_shift_mean": opacity_shift.sum() / candidate_count,
            "raw_harmonics": raw_harmonics.abs().mean(dim=-1).sum()
            / candidate_count,
            "harmonics_shift_mean": harmonics_shift.sum() / candidate_count,
            **_opacity_gradient_update_stats(
                evidence.opacity_gradient_features,
                evidence.opacity_gradient_signal,
                delta_opacity,
            ),
            **_mean_gradient_update_stats(
                evidence.mean_gradient_signal,
                evidence.mean_gradient_current_signal,
                evidence.mean_gradient_replay_signal,
                raw_mean,
                delta_mean_camera,
            ),
        },
    )


def apply_current_gaussian_residual(
    current: Gaussians,
    prediction: GIRPrediction,
    camera_to_world: torch.Tensor,
    current_depth: torch.Tensor,
) -> Gaussians:
    """Apply history-conditioned residuals to the current per-pixel Gaussians."""
    b, n = current.means.shape[:2]
    h, w = current_depth.shape[-2:]
    if n != h * w:
        raise RuntimeError(
            "Current GS residual expects one Gaussian per image pixel: "
            f"gaussians={n}, image_shape={(h, w)}."
        )

    def upsample_flat(values: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(
            values,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        return values.permute(0, 2, 3, 1).reshape(b, n, values.shape[1])

    gate = upsample_flat(prediction.current_residual_gate).sigmoid()
    depth = current_depth[:, 0].reshape(b, n, 1).to(current.means.dtype)

    delta_mean_camera = upsample_flat(
        prediction.current_delta_mean_camera
    ).tanh()
    delta_mean_camera = delta_mean_camera * depth * gate
    rotation_c2w = camera_to_world[:, :3, :3].to(delta_mean_camera.dtype)
    delta_mean_world = torch.einsum(
        "bij,bnj->bni", rotation_c2w, delta_mean_camera
    )

    delta_rotation = upsample_flat(prediction.current_delta_rotation).tanh()
    delta_rotation = delta_rotation * gate
    delta_quaternion = _axis_angle_to_quaternion_xyzw(delta_rotation)
    rotations = _quat_multiply_xyzw(delta_quaternion, current.rotations)
    rotations = F.normalize(rotations, dim=-1, eps=1e-8)

    delta_log_scale = upsample_flat(prediction.current_delta_log_scale).tanh()
    delta_log_scale = delta_log_scale * gate
    scales = current.scales * torch.exp(delta_log_scale.clamp(-2.0, 2.0))

    delta_opacity = upsample_flat(
        prediction.current_delta_opacity_logit
    ).squeeze(-1).tanh()
    delta_opacity = delta_opacity * gate.squeeze(-1)
    opacity_logit = torch.logit(current.opacities.clamp(1e-5, 1.0 - 1e-5))
    opacities = torch.sigmoid(opacity_logit + delta_opacity)

    harmonics_flat = _flatten_harmonics(current.harmonics)
    delta_harmonics = upsample_flat(prediction.current_delta_harmonics).tanh()
    delta_harmonics = delta_harmonics * gate
    harmonics = _restore_harmonics(
        harmonics_flat + delta_harmonics,
        current.harmonics,
    )

    return Gaussians(
        means=current.means + delta_mean_world,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales.clamp(1e-6, 0.1),
        rotations=rotations,
    )


@dataclass
class StreamingGaussianState:
    gaussians: Gaussians
    stable_ids: torch.Tensor
    observation_count: torch.Tensor
    point_map: torch.Tensor | None = None
    front_conflict_count: torch.Tensor | None = None
    split_generation: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.split_generation is None:
            self.split_generation = torch.zeros_like(
                self.stable_ids, dtype=torch.uint8
            )
        elif self.split_generation.shape != self.stable_ids.shape:
            raise ValueError(
                "Split generations must match stable IDs; got "
                f"{tuple(self.split_generation.shape)} and "
                f"{tuple(self.stable_ids.shape)}."
            )

    @property
    def batch_size(self) -> int:
        return self.gaussians.means.shape[0]

    @property
    def num_gaussians(self) -> int:
        return self.gaussians.means.shape[1]

    def detach(self) -> "StreamingGaussianState":
        return StreamingGaussianState(
            gaussians=Gaussians(
                means=self.gaussians.means.detach(),
                harmonics=self.gaussians.harmonics.detach(),
                opacities=self.gaussians.opacities.detach(),
                scales=self.gaussians.scales.detach(),
                rotations=self.gaussians.rotations.detach(),
            ),
            stable_ids=self.stable_ids,
            observation_count=self.observation_count.detach(),
            point_map=(
                None if self.point_map is None else self.point_map.detach()
            ),
            front_conflict_count=(
                None
                if self.front_conflict_count is None
                else self.front_conflict_count.detach()
            ),
            split_generation=self.split_generation,
        )

    def detach_means(self) -> "StreamingGaussianState":
        """Detach historical positions while preserving appearance gradients.

        The overfit experiment uses this boundary before each newly arriving
        view.  Historical geometry must not accumulate gradients through the
        whole stream, while SH, opacity, scale, and rotation can still carry
        the remaining cross-view graph when the corresponding mode is used.
        """
        return StreamingGaussianState(
            gaussians=Gaussians(
                means=self.gaussians.means.detach(),
                harmonics=self.gaussians.harmonics,
                opacities=self.gaussians.opacities,
                scales=self.gaussians.scales,
                rotations=self.gaussians.rotations,
            ),
            stable_ids=self.stable_ids,
            observation_count=self.observation_count,
            point_map=self.point_map,
            front_conflict_count=self.front_conflict_count,
            split_generation=self.split_generation,
        )

    def sync_point_map_to_means(self) -> "StreamingGaussianState":
        """Use residual-updated means for subsequent point-map coverage."""
        if self.point_map is None:
            return self
        return StreamingGaussianState(
            gaussians=self.gaussians,
            stable_ids=self.stable_ids,
            observation_count=self.observation_count,
            point_map=self.gaussians.means.detach(),
            front_conflict_count=self.front_conflict_count,
            split_generation=self.split_generation,
        )

    def with_front_conflict_count(
        self,
        front_conflict_count: torch.Tensor,
    ) -> "StreamingGaussianState":
        if front_conflict_count.shape != self.stable_ids.shape:
            raise ValueError(
                "Front-conflict counts must match stable IDs; got "
                f"{tuple(front_conflict_count.shape)} and "
                f"{tuple(self.stable_ids.shape)}."
            )
        return StreamingGaussianState(
            gaussians=self.gaussians,
            stable_ids=self.stable_ids,
            observation_count=self.observation_count,
            point_map=self.point_map,
            front_conflict_count=front_conflict_count.detach().to(torch.uint8),
            split_generation=self.split_generation,
        )

    def decay_masked_opacity(
        self,
        mask: torch.Tensor,
        decay_fraction: float,
    ) -> "StreamingGaussianState":
        """Apply a fixed multiplicative opacity decay to selected old GS."""
        if mask.shape != self.gaussians.opacities.shape:
            raise ValueError(
                "Opacity decay mask must match historical opacities; got "
                f"{tuple(mask.shape)} and {tuple(self.gaussians.opacities.shape)}."
            )
        fraction = max(0.0, min(0.95, float(decay_fraction)))
        if fraction == 0.0 or not bool(mask.any()):
            return self
        multiplier = torch.where(
            mask,
            self.gaussians.opacities.new_full((), 1.0 - fraction),
            self.gaussians.opacities.new_ones(()),
        )
        return StreamingGaussianState(
            gaussians=Gaussians(
                means=self.gaussians.means,
                harmonics=self.gaussians.harmonics,
                opacities=self.gaussians.opacities * multiplier,
                scales=self.gaussians.scales,
                rotations=self.gaussians.rotations,
            ),
            stable_ids=self.stable_ids,
            observation_count=self.observation_count,
            point_map=self.point_map,
            front_conflict_count=self.front_conflict_count,
            split_generation=self.split_generation,
        )

    @classmethod
    def from_current(
        cls,
        current: Gaussians,
        add_gate: torch.Tensor,
        point_map: torch.Tensor | None = None,
    ) -> "StreamingGaussianState":
        b, n = current.means.shape[:2]
        if point_map is not None and point_map.shape != current.means.shape:
            raise ValueError(
                "Initial point map must match Gaussian means; got "
                f"{tuple(point_map.shape)} and {tuple(current.means.shape)}."
            )
        gate = add_gate.reshape(b, n).to(current.opacities.dtype)
        ids = torch.arange(n, device=current.means.device, dtype=torch.long)
        ids = ids.unsqueeze(0).expand(b, -1)
        return cls(
            gaussians=Gaussians(
                means=current.means,
                harmonics=current.harmonics,
                opacities=current.opacities * gate,
                scales=current.scales,
                rotations=current.rotations,
            ),
            stable_ids=ids,
            observation_count=torch.ones(
                (b, n), device=current.means.device, dtype=current.means.dtype
            ),
            point_map=(
                current.means.detach()
                if point_map is None
                else point_map.detach()
            ),
            front_conflict_count=torch.zeros(
                (b, n), device=current.means.device, dtype=torch.uint8
            ),
        )

    def append(
        self,
        current: Gaussians,
        add_gate: torch.Tensor,
        prune_threshold: float = 0.0,
        append_mask: torch.Tensor | None = None,
        point_map: torch.Tensor | None = None,
    ) -> "StreamingGaussianState":
        b, n = current.means.shape[:2]
        if point_map is not None and point_map.shape != current.means.shape:
            raise ValueError(
                "Appended point map must match Gaussian means; got "
                f"{tuple(point_map.shape)} and {tuple(current.means.shape)}."
            )
        new_point_map = (
            current.means.detach()
            if point_map is None
            else point_map.detach()
        )
        gate = add_gate.reshape(b, n).to(current.opacities.dtype)
        selection = None
        if append_mask is not None:
            selection = append_mask.reshape(b, n).bool()
        if prune_threshold > 0.0:
            gate_selection = gate >= float(prune_threshold)
            selection = (
                gate_selection
                if selection is None
                else selection & gate_selection
            )

        valid_new = None
        if selection is not None:
            counts = selection.sum(dim=1)
            max_count = int(counts.max().item())
            if max_count == 0:
                return self
            gather_indices = torch.zeros(
                (b, max_count), device=current.means.device, dtype=torch.long
            )
            valid_new = torch.zeros(
                (b, max_count), device=current.means.device, dtype=torch.bool
            )
            for batch_idx in range(b):
                selected = torch.nonzero(
                    selection[batch_idx], as_tuple=False
                ).squeeze(1)
                count = int(selected.numel())
                if count:
                    gather_indices[batch_idx, :count] = selected
                    valid_new[batch_idx, :count] = True

            def gather(values: torch.Tensor) -> torch.Tensor:
                index = gather_indices
                for _ in range(values.ndim - 2):
                    index = index.unsqueeze(-1)
                index = index.expand(*gather_indices.shape, *values.shape[2:])
                return values.gather(1, index)

            current = Gaussians(
                means=gather(current.means),
                harmonics=gather(current.harmonics),
                opacities=gather(current.opacities),
                scales=gather(current.scales),
                rotations=gather(current.rotations),
            )
            new_point_map = gather(new_point_map)
            gate = gate.gather(1, gather_indices)
            n = max_count
            valid_scalar = valid_new.to(current.means.dtype)
            valid_vector = valid_scalar.unsqueeze(-1)
            valid_harmonics = valid_vector.unsqueeze(-1)
            padding_rotation = torch.zeros_like(current.rotations)
            padding_rotation[..., 3] = 1.0
            current = Gaussians(
                means=current.means * valid_vector,
                harmonics=current.harmonics * valid_harmonics,
                opacities=current.opacities * valid_scalar,
                scales=torch.where(
                    valid_vector.bool(),
                    current.scales,
                    torch.full_like(current.scales, 1e-6),
                ),
                rotations=torch.where(
                    valid_vector.bool(), current.rotations, padding_rotation
                ),
            )
            new_point_map = new_point_map * valid_vector

        first_new_id = self.stable_ids.max(dim=1, keepdim=True).values + 1
        offsets = torch.arange(n, device=current.means.device, dtype=torch.long)
        new_ids = first_new_id + offsets.unsqueeze(0)
        new_observation_count = torch.ones(
            (b, n),
            device=current.means.device,
            dtype=self.observation_count.dtype,
        )
        old_front_conflict_count = self.front_conflict_count
        if old_front_conflict_count is None:
            old_front_conflict_count = torch.zeros_like(
                self.stable_ids, dtype=torch.uint8
            )
        new_front_conflict_count = torch.zeros(
            (b, n), device=current.means.device, dtype=torch.uint8
        )
        new_split_generation = torch.zeros(
            (b, n), device=current.means.device, dtype=torch.uint8
        )
        if valid_new is not None:
            gate = gate * valid_new.to(gate.dtype)
            new_ids = torch.where(valid_new, new_ids, torch.full_like(new_ids, -1))
            new_observation_count = new_observation_count * valid_new.to(
                new_observation_count.dtype
            )
        old_point_map = (
            self.point_map
            if self.point_map is not None
            else self.gaussians.means.detach()
        )
        return StreamingGaussianState(
            gaussians=Gaussians(
                means=torch.cat([self.gaussians.means, current.means], dim=1),
                harmonics=torch.cat(
                    [self.gaussians.harmonics, current.harmonics], dim=1
                ),
                opacities=torch.cat(
                    [self.gaussians.opacities, current.opacities * gate], dim=1
                ),
                scales=torch.cat([self.gaussians.scales, current.scales], dim=1),
                rotations=torch.cat(
                    [self.gaussians.rotations, current.rotations], dim=1
                ),
            ),
            stable_ids=torch.cat([self.stable_ids, new_ids], dim=1),
            observation_count=torch.cat(
                [self.observation_count, new_observation_count],
                dim=1,
            ),
            point_map=torch.cat(
                [old_point_map, new_point_map], dim=1
            ),
            front_conflict_count=torch.cat(
                [old_front_conflict_count, new_front_conflict_count], dim=1
            ),
            split_generation=torch.cat(
                [self.split_generation, new_split_generation], dim=1
            ),
        )

    def select(self, keep: torch.Tensor) -> "StreamingGaussianState":
        """Physically filter a batch-one streaming map during inference."""
        if self.batch_size != 1 or keep.dim() != 1:
            raise RuntimeError(
                "Physical historical GS pruning requires a 1D mask and batch size 1."
            )
        if keep.numel() != self.num_gaussians:
            raise RuntimeError(
                "Historical GS pruning mask size does not match the map: "
                f"mask={keep.numel()}, map={self.num_gaussians}."
            )
        return StreamingGaussianState(
            gaussians=Gaussians(
                means=self.gaussians.means[:, keep],
                harmonics=self.gaussians.harmonics[:, keep],
                opacities=self.gaussians.opacities[:, keep],
                scales=self.gaussians.scales[:, keep],
                rotations=self.gaussians.rotations[:, keep],
            ),
            stable_ids=self.stable_ids[:, keep],
            observation_count=self.observation_count[:, keep],
            point_map=(
                None if self.point_map is None else self.point_map[:, keep]
            ),
            front_conflict_count=(
                None
                if self.front_conflict_count is None
                else self.front_conflict_count[:, keep]
            ),
            split_generation=self.split_generation[:, keep],
        )

    def expand_anchor_children(
        self,
        evidence: SymmetricMultiviewGradientEvidence,
        *,
        max_child_count: int,
        add_child_count: int,
        min_observations: int,
        gradient_threshold: float,
        coherence_threshold: float,
        max_conflict_ratio: float,
        cooldown_views: int,
        max_ratio: float,
        offset_scale: float,
        scale_divisor: float,
        min_opacity: float,
        max_gaussians: int,
    ) -> tuple["StreamingGaussianState", dict[str, torch.Tensor]]:
        """Progressively add persistent, non-recursive children to anchors.

        The root Gaussian doubles as child zero. Its split_generation stores
        the number of children it has spawned; generated children use 255 and
        can therefore never become anchors themselves.
        """
        batch, count = self.gaussians.opacities.shape
        expected_shape = (batch, count)
        for name, value in (
            ("active_mask", evidence.active_mask),
            ("visible_view_count", evidence.visible_view_count),
            ("mean_coherence", evidence.mean_coherence),
            ("mean_conflict_ratio", evidence.mean_conflict_ratio),
            ("opacity_signal", evidence.opacity_signal),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"Anchor-child evidence {name} must be {expected_shape}, "
                    f"got {tuple(value.shape)}."
                )
        if evidence.mean_signal.shape != (batch, count, 3):
            raise ValueError("Anchor-child mean signal must be [B,N,3].")
        if evidence.harmonics_signal.shape[:2] != expected_shape:
            raise ValueError(
                "Anchor-child harmonics signal must start with [B,N]."
            )

        # uint8 value 255 is reserved for non-recursive children.
        max_child_count = min(255, max(1, int(max_child_count)))
        max_spawned = max_child_count - 1
        add_child_count = max(1, int(add_child_count))
        min_observations = max(1, int(min_observations))
        cooldown_views = max(1, int(cooldown_views))
        max_ratio = max(0.0, min(1.0, float(max_ratio)))
        offset_scale = max(0.0, float(offset_scale))
        scale_divisor = max(1.0 + 1e-6, float(scale_divisor))
        min_opacity = max(0.0, min(1.0, float(min_opacity)))
        max_gaussians = max(0, int(max_gaussians))
        for name, value in (
            ("gradient_threshold", gradient_threshold),
            ("coherence_threshold", coherence_threshold),
            ("max_conflict_ratio", max_conflict_ratio),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"Anchor-child {name} must be finite.")
        gradient_threshold = max(0.0, float(gradient_threshold))
        coherence_threshold = max(0.0, min(1.0, float(coherence_threshold)))
        max_conflict_ratio = max(0.0, min(1.0, float(max_conflict_ratio)))

        zero = self.gaussians.means.new_zeros((), dtype=torch.float32)
        active = self.stable_ids >= 0
        # Values 0..max_spawned identify roots and encode their persistent
        # child count. Value 255 is reserved for non-recursive children.
        spawned_count = self.split_generation.to(torch.int16)
        roots = active & (spawned_count <= max_spawned)
        active_children = active & (self.split_generation == 255)
        active_count_by_batch = active.sum(dim=1)
        root_count = roots.sum().float()
        child_count = active_children.sum().float()

        if max_spawned == 0 or max_ratio == 0.0:
            return self, {
                "eligible_anchor_count": zero,
                "expanded_anchor_count": zero,
                "spawned_child_count": zero,
                "active_anchor_count": root_count,
                "active_child_count": child_count,
                "selected_gradient_score_mean": zero,
                "selected_coherence_mean": zero,
                "map_count_before": active.sum().float() / max(batch, 1),
                "map_count_after": active.sum().float() / max(batch, 1),
            }

        mean_score = evidence.mean_signal.detach().float().norm(dim=-1)
        opacity_score = evidence.opacity_signal.detach().float().abs()
        harmonics_score = (
            evidence.harmonics_signal.detach().float().flatten(2).norm(dim=-1)
        )
        gradient_score = mean_score + opacity_score + harmonics_score
        required_observations = min_observations + (
            spawned_count.clamp(min=0, max=max_spawned) * cooldown_views
        )
        eligible = (
            roots
            & evidence.active_mask.detach()
            & (spawned_count < max_spawned)
            & (evidence.visible_view_count.detach() >= required_observations)
            & (gradient_score >= gradient_threshold)
            & (evidence.mean_coherence.detach() >= coherence_threshold)
            & (evidence.mean_conflict_ratio.detach() <= max_conflict_ratio)
            & (self.gaussians.opacities.detach().float() >= min_opacity)
        )
        score = gradient_score * evidence.mean_coherence.detach().float() * (
            1.0 - evidence.mean_conflict_ratio.detach().float()
        )
        score = torch.where(eligible, score, torch.zeros_like(score))

        selected_by_batch: list[torch.Tensor] = []
        children_per_parent: list[int] = []
        for batch_idx in range(batch):
            candidates = torch.nonzero(
                eligible[batch_idx], as_tuple=False
            ).squeeze(1)
            active_count = int(active_count_by_batch[batch_idx].item())
            budget = (
                max(1, int(math.floor(active_count * max_ratio)))
                if active_count > 0
                else 0
            )
            if max_gaussians > 0:
                budget = min(budget, max(0, max_gaussians - active_count))
            # Each selected root adds the same number in this streaming pass.
            per_parent = min(
                add_child_count,
                max_spawned,
            )
            parent_budget = budget // max(per_parent, 1)
            parent_budget = min(parent_budget, int(candidates.numel()))
            if parent_budget:
                top = torch.topk(
                    score[batch_idx].index_select(0, candidates),
                    parent_budget,
                    sorted=False,
                ).indices
                candidates = candidates.index_select(0, top)
            else:
                candidates = candidates[:0]
            selected_by_batch.append(candidates)
            children_per_parent.append(per_parent)

        # A root near its limit may add fewer children than the common upper
        # bound. Build each batch independently, then pad like append().
        appended_by_batch: list[dict[str, torch.Tensor]] = []
        selected_scores: list[torch.Tensor] = []
        selected_coherences: list[torch.Tensor] = []
        expanded_count = 0
        spawned_total = 0
        means = self.gaussians.means.clone()
        opacities = self.gaussians.opacities.clone()
        generations = self.split_generation.clone()

        offset_pattern = self.gaussians.means.new_tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [0.70710678, 0.70710678],
                [-0.70710678, 0.70710678],
                [0.70710678, -0.70710678],
                [-0.70710678, -0.70710678],
            ]
        )
        for batch_idx, selected in enumerate(selected_by_batch):
            batch_values: dict[str, list[torch.Tensor]] = {
                key: []
                for key in (
                    "means",
                    "harmonics",
                    "opacities",
                    "scales",
                    "rotations",
                    "observations",
                    "point_maps",
                    "generations",
                )
            }
            next_id = int(self.stable_ids[batch_idx].max().item()) + 1
            batch_ids: list[torch.Tensor] = []
            batch_spawned = 0
            for parent_idx_tensor in selected:
                parent_idx = int(parent_idx_tensor.item())
                already_spawned = int(generations[batch_idx, parent_idx].item())
                add_count = min(
                    children_per_parent[batch_idx],
                    max_spawned - already_spawned,
                )
                if max_gaussians > 0:
                    remaining = max_gaussians - (
                        int(active_count_by_batch[batch_idx].item())
                        + batch_spawned
                    )
                    add_count = min(add_count, max(0, remaining))
                if add_count <= 0:
                    continue

                parent_scale = self.gaussians.scales[
                    batch_idx, parent_idx
                ].float()
                parent_rotation = self.gaussians.rotations[
                    batch_idx, parent_idx
                ].float()
                rotation_matrix = quaternion_to_matrix(parent_rotation[None])[0]
                tangent_axes = torch.topk(parent_scale, 2).indices
                axis_world = rotation_matrix.index_select(1, tangent_axes)
                axis_scale = parent_scale.index_select(0, tangent_axes)
                pattern_indices = (
                    torch.arange(add_count, device=parent_scale.device)
                    + already_spawned
                ).remainder(offset_pattern.shape[0])
                local_offsets = offset_pattern.index_select(0, pattern_indices)
                offsets = torch.matmul(
                    axis_world,
                    (local_offsets * axis_scale * offset_scale).T,
                ).T.to(self.gaussians.means.dtype)
                child_means = self.gaussians.means[
                    batch_idx, parent_idx
                ].unsqueeze(0) + offsets

                # Read from the immutable source tensor. Reading through the
                # clone and then writing back into it would invalidate the
                # saved view needed by autograd.
                parent_opacity = self.gaussians.opacities[
                    batch_idx, parent_idx
                ].float()
                branch_count = add_count + 1
                branch_opacity = 1.0 - (
                    1.0 - parent_opacity.clamp(0.0, 1.0)
                ).clamp_min(0.0).pow(1.0 / branch_count)
                opacities[batch_idx, parent_idx] = branch_opacity.to(
                    opacities.dtype
                )
                generations[batch_idx, parent_idx] = already_spawned + add_count

                batch_values["means"].append(child_means)
                batch_values["harmonics"].append(
                    self.gaussians.harmonics[batch_idx, parent_idx]
                    .unsqueeze(0)
                    .expand(add_count, -1, -1)
                )
                batch_values["opacities"].append(
                    branch_opacity.to(opacities.dtype).expand(add_count)
                )
                batch_values["scales"].append(
                    (self.gaussians.scales[batch_idx, parent_idx] / scale_divisor)
                    .clamp_min(1e-6)
                    .unsqueeze(0)
                    .expand(add_count, -1)
                )
                batch_values["rotations"].append(
                    self.gaussians.rotations[batch_idx, parent_idx]
                    .unsqueeze(0)
                    .expand(add_count, -1)
                )
                batch_values["observations"].append(
                    self.observation_count[batch_idx, parent_idx]
                    .clamp_min(1.0)
                    .expand(add_count)
                )
                batch_values["point_maps"].append(child_means.detach())
                batch_values["generations"].append(
                    torch.full(
                        (add_count,),
                        255,
                        device=self.split_generation.device,
                        dtype=torch.uint8,
                    )
                )
                batch_ids.append(
                    torch.arange(
                        next_id,
                        next_id + add_count,
                        device=self.stable_ids.device,
                        dtype=torch.long,
                    )
                )
                next_id += add_count
                expanded_count += 1
                spawned_total += add_count
                batch_spawned += add_count
                selected_scores.append(score[batch_idx, parent_idx][None])
                selected_coherences.append(
                    evidence.mean_coherence[batch_idx, parent_idx].detach()[None]
                )

            concatenated: dict[str, torch.Tensor] = {}
            for key, values in batch_values.items():
                if values:
                    concatenated[key] = torch.cat(values, dim=0)
            concatenated["ids"] = (
                torch.cat(batch_ids)
                if batch_ids
                else self.stable_ids.new_empty((0,))
            )
            appended_by_batch.append(concatenated)

        appended_counts = [int(values["ids"].numel()) for values in appended_by_batch]
        max_appended = max(appended_counts, default=0)
        active_total = active.sum().float()
        if max_appended == 0:
            return self, {
                "eligible_anchor_count": eligible.sum().float(),
                "expanded_anchor_count": zero,
                "spawned_child_count": zero,
                "active_anchor_count": root_count,
                "active_child_count": child_count,
                "selected_gradient_score_mean": zero,
                "selected_coherence_mean": zero,
                "map_count_before": active_total / max(batch, 1),
                "map_count_after": active_total / max(batch, 1),
            }

        def padded_stack(
            key: str,
            reference: torch.Tensor,
            fill_value: float | int = 0,
        ) -> torch.Tensor:
            rows = []
            for values, appended_count in zip(appended_by_batch, appended_counts):
                value = values.get(
                    key,
                    reference.new_empty((0, *reference.shape[2:])),
                )
                if appended_count < max_appended:
                    padding = torch.full(
                        (max_appended - appended_count, *value.shape[1:]),
                        fill_value,
                        device=value.device,
                        dtype=value.dtype,
                    )
                    value = torch.cat([value, padding], dim=0)
                rows.append(value)
            return torch.stack(rows)

        appended_rotations = padded_stack(
            "rotations", self.gaussians.rotations
        )
        for batch_idx, appended_count in enumerate(appended_counts):
            if appended_count < max_appended:
                appended_rotations[batch_idx, appended_count:, 3] = 1.0
        appended_ids = padded_stack("ids", self.stable_ids, -1)
        appended_scales = padded_stack("scales", self.gaussians.scales, 1e-6)
        old_point_map = (
            self.gaussians.means.detach()
            if self.point_map is None
            else self.point_map
        )
        old_front_conflicts = (
            torch.zeros_like(self.stable_ids, dtype=torch.uint8)
            if self.front_conflict_count is None
            else self.front_conflict_count
        )
        updated = StreamingGaussianState(
            gaussians=Gaussians(
                means=torch.cat(
                    [means, padded_stack("means", self.gaussians.means)], dim=1
                ),
                harmonics=torch.cat(
                    [
                        self.gaussians.harmonics,
                        padded_stack("harmonics", self.gaussians.harmonics),
                    ],
                    dim=1,
                ),
                opacities=torch.cat(
                    [opacities, padded_stack("opacities", self.gaussians.opacities)],
                    dim=1,
                ),
                scales=torch.cat([self.gaussians.scales, appended_scales], dim=1),
                rotations=torch.cat(
                    [self.gaussians.rotations, appended_rotations], dim=1
                ),
            ),
            stable_ids=torch.cat([self.stable_ids, appended_ids], dim=1),
            observation_count=torch.cat(
                [
                    self.observation_count,
                    padded_stack("observations", self.observation_count),
                ],
                dim=1,
            ),
            point_map=torch.cat(
                [old_point_map, padded_stack("point_maps", self.gaussians.means)],
                dim=1,
            ),
            front_conflict_count=torch.cat(
                [
                    old_front_conflicts,
                    torch.zeros(
                        (batch, max_appended),
                        device=self.stable_ids.device,
                        dtype=torch.uint8,
                    ),
                ],
                dim=1,
            ),
            split_generation=torch.cat(
                [
                    generations,
                    padded_stack("generations", self.split_generation, 255),
                ],
                dim=1,
            ),
        )
        spawned_tensor = zero.new_tensor(float(spawned_total))
        return updated, {
            "eligible_anchor_count": eligible.sum().float(),
            "expanded_anchor_count": zero.new_tensor(float(expanded_count)),
            "spawned_child_count": spawned_tensor,
            "active_anchor_count": root_count,
            "active_child_count": child_count + spawned_tensor,
            "selected_gradient_score_mean": torch.cat(selected_scores).mean(),
            "selected_coherence_mean": torch.cat(selected_coherences).mean(),
            "map_count_before": active_total / max(batch, 1),
            "map_count_after": (active_total + spawned_tensor) / max(batch, 1),
        }

    def split_projected_historical(
        self,
        evidence: ProjectedGSEvidence,
        *,
        projected_radius_threshold: float,
        mean_gradient_threshold: float,
        min_observations: int,
        max_ratio: float,
        max_generation: int,
        offset_scale: float,
        scale_divisor: float,
        min_opacity: float,
        max_gaussians: int = 0,
    ) -> tuple["StreamingGaussianState", dict[str, torch.Tensor]]:
        """Split a budgeted set of large, high-error historical Gaussians."""
        b, n = self.gaussians.opacities.shape
        expected_shape = (b, n)
        for name, value in (
            ("surface_consistent", evidence.surface_consistent),
            ("old_in_front", evidence.old_in_front),
            ("projected_radius", evidence.projected_radius),
        ):
            if value.shape != expected_shape:
                raise RuntimeError(
                    f"Split evidence {name} shape mismatch: "
                    f"got={tuple(value.shape)}, expected={expected_shape}."
                )
        if evidence.mean_gradient_signal is None:
            raise RuntimeError(
                "Old-GS splitting requires camera-space mean-gradient evidence."
            )
        flat_indices = evidence.flat_candidate_indices.long()
        gradient_signal = evidence.mean_gradient_signal.detach().float()
        if gradient_signal.shape != (flat_indices.numel(), 3):
            raise RuntimeError(
                "Split mean-gradient signal must match projected candidates; "
                f"got={tuple(gradient_signal.shape)}, "
                f"candidates={flat_indices.numel()}."
            )

        radius_threshold = max(float(projected_radius_threshold), 1e-6)
        gradient_threshold = max(float(mean_gradient_threshold), 0.0)
        min_observations = max(1, int(min_observations))
        max_ratio = max(0.0, min(1.0, float(max_ratio)))
        max_generation = max(0, int(max_generation))
        offset_scale = max(0.0, float(offset_scale))
        scale_divisor = max(1.0 + 1e-6, float(scale_divisor))
        min_opacity = max(0.0, min(1.0, float(min_opacity)))
        max_gaussians = max(0, int(max_gaussians))

        gradient_norm_flat = torch.zeros(
            b * n,
            device=self.gaussians.means.device,
            dtype=torch.float32,
        )
        if flat_indices.numel():
            gradient_norm_flat.index_copy_(
                0, flat_indices, gradient_signal.norm(dim=-1)
            )
        gradient_norm = gradient_norm_flat.reshape(b, n)
        radius = evidence.projected_radius.detach().float()
        active = self.stable_ids >= 0
        # surface_consistent is set only for projected candidates whose
        # center RGB residual already exceeds the configured GIR threshold.
        eligible = (
            active
            & evidence.surface_consistent.detach()
            & ~evidence.old_in_front.detach()
            & (radius >= radius_threshold)
            & (gradient_norm >= gradient_threshold)
            & (self.observation_count.detach() >= min_observations)
            & (self.gaussians.opacities.detach().float() >= min_opacity)
            & (self.split_generation < max_generation)
        )
        score = gradient_norm * (radius / radius_threshold)
        score = torch.where(eligible, score, torch.zeros_like(score))

        selected_by_batch: list[torch.Tensor] = []
        active_counts = active.sum(dim=1)
        for batch_idx in range(b):
            candidates = torch.nonzero(
                eligible[batch_idx], as_tuple=False
            ).squeeze(1)
            active_count = int(active_counts[batch_idx].item())
            budget = (
                max(1, int(math.floor(active_count * max_ratio)))
                if max_ratio > 0.0 and active_count > 0
                else 0
            )
            if max_gaussians > 0:
                budget = min(budget, max(0, max_gaussians - active_count))
            budget = min(budget, int(candidates.numel()))
            if budget:
                candidate_scores = score[batch_idx].index_select(0, candidates)
                top = torch.topk(
                    candidate_scores, budget, sorted=False
                ).indices
                candidates = candidates.index_select(0, top)
            else:
                candidates = candidates[:0]
            selected_by_batch.append(candidates)

        split_counts = torch.tensor(
            [indices.numel() for indices in selected_by_batch],
            device=self.gaussians.means.device,
            dtype=torch.long,
        )
        max_split_count = int(split_counts.max().item())
        stats_device = self.gaussians.means.device
        eligible_count = eligible.sum().float()
        split_count = split_counts.sum().float()
        active_count = active_counts.sum().float()
        zero = torch.zeros((), device=stats_device, dtype=torch.float32)
        if max_split_count == 0:
            return self, {
                "eligible_count": eligible_count,
                "split_count": split_count,
                "split_ratio": zero,
                "selected_radius_mean": zero,
                "selected_gradient_norm_mean": zero,
                "map_count_before": active_count / max(b, 1),
                "map_count_after": active_count / max(b, 1),
            }

        means = self.gaussians.means.clone()
        harmonics = self.gaussians.harmonics.clone()
        opacities = self.gaussians.opacities.clone()
        scales = self.gaussians.scales.clone()
        rotations = self.gaussians.rotations.clone()
        observation_count = self.observation_count.clone()
        point_map = (
            self.gaussians.means.detach().clone()
            if self.point_map is None
            else self.point_map.clone()
        )
        front_conflict_count = (
            torch.zeros_like(self.stable_ids, dtype=torch.uint8)
            if self.front_conflict_count is None
            else self.front_conflict_count.clone()
        )
        split_generation = self.split_generation.clone()

        appended_means = []
        appended_harmonics = []
        appended_opacities = []
        appended_scales = []
        appended_rotations = []
        appended_ids = []
        appended_observations = []
        appended_point_maps = []
        appended_front_conflicts = []
        appended_generations = []
        selected_radii = []
        selected_gradients = []

        def pad_first_dim(
            value: torch.Tensor,
            size: int,
            fill_value: float | int = 0,
        ) -> torch.Tensor:
            if value.shape[0] == size:
                return value
            padding = torch.full(
                (size - value.shape[0], *value.shape[1:]),
                fill_value,
                device=value.device,
                dtype=value.dtype,
            )
            return torch.cat([value, padding], dim=0)

        for batch_idx, selected in enumerate(selected_by_batch):
            count = selected.numel()
            if count:
                parent_means = self.gaussians.means[batch_idx].index_select(
                    0, selected
                )
                parent_scales = self.gaussians.scales[batch_idx].index_select(
                    0, selected
                )
                parent_rotations = self.gaussians.rotations[
                    batch_idx
                ].index_select(0, selected)
                local_axis = parent_scales.detach().argmax(dim=-1)
                rotation_matrix = quaternion_to_matrix(parent_rotations.float())
                axis_world = rotation_matrix.gather(
                    2, local_axis[:, None, None].expand(-1, 3, 1)
                ).squeeze(-1)
                axis_scale = parent_scales.gather(
                    1, local_axis[:, None]
                ).float()
                offset = (
                    axis_world
                    * axis_scale
                    * offset_scale
                ).to(parent_means.dtype)
                first_means = parent_means - offset
                second_means = parent_means + offset
                child_scales = (parent_scales / scale_divisor).clamp_min(1e-6)
                parent_opacity = self.gaussians.opacities[
                    batch_idx
                ].index_select(0, selected)
                remaining_transmittance = (
                    1.0 - parent_opacity.float().clamp(0.0, 1.0)
                ).clamp_min(0.0)
                child_opacity = 1.0 - torch.sqrt(remaining_transmittance)
                child_opacity = child_opacity.to(parent_opacity.dtype)
                child_observations = (
                    self.observation_count[batch_idx]
                    .index_select(0, selected)
                    .float()
                    .div(2.0)
                    .clamp_min(1.0)
                    .to(self.observation_count.dtype)
                )
                child_generation = (
                    self.split_generation[batch_idx]
                    .index_select(0, selected)
                    .to(torch.int16)
                    .add(1)
                    .clamp_max(255)
                    .to(torch.uint8)
                )

                means[batch_idx, selected] = first_means
                opacities[batch_idx, selected] = child_opacity
                scales[batch_idx, selected] = child_scales
                observation_count[batch_idx, selected] = child_observations
                point_map[batch_idx, selected] = first_means.detach()
                front_conflict_count[batch_idx, selected] = 0
                split_generation[batch_idx, selected] = child_generation

                child_harmonics = self.gaussians.harmonics[
                    batch_idx
                ].index_select(0, selected)
                first_new_id = int(
                    self.stable_ids[batch_idx].max().item()
                ) + 1
                new_ids = torch.arange(
                    first_new_id,
                    first_new_id + count,
                    device=self.stable_ids.device,
                    dtype=torch.long,
                )
                selected_radii.append(
                    radius[batch_idx].index_select(0, selected)
                )
                selected_gradients.append(
                    gradient_norm[batch_idx].index_select(0, selected)
                )
            else:
                second_means = self.gaussians.means.new_empty((0, 3))
                child_harmonics = self.gaussians.harmonics.new_empty(
                    (0, *self.gaussians.harmonics.shape[2:])
                )
                child_opacity = self.gaussians.opacities.new_empty((0,))
                child_scales = self.gaussians.scales.new_empty((0, 3))
                parent_rotations = self.gaussians.rotations.new_empty((0, 4))
                child_observations = self.observation_count.new_empty((0,))
                child_generation = self.split_generation.new_empty((0,))
                new_ids = self.stable_ids.new_empty((0,))

            appended_means.append(pad_first_dim(second_means, max_split_count))
            appended_harmonics.append(
                pad_first_dim(child_harmonics, max_split_count)
            )
            appended_opacities.append(
                pad_first_dim(child_opacity, max_split_count)
            )
            appended_scales.append(
                pad_first_dim(child_scales, max_split_count, 1e-6)
            )
            padded_rotation = pad_first_dim(parent_rotations, max_split_count)
            if count < max_split_count:
                padded_rotation[count:, 3] = 1.0
            appended_rotations.append(padded_rotation)
            appended_ids.append(pad_first_dim(new_ids, max_split_count, -1))
            appended_observations.append(
                pad_first_dim(child_observations, max_split_count)
            )
            appended_point_maps.append(
                pad_first_dim(second_means.detach(), max_split_count)
            )
            appended_front_conflicts.append(
                torch.zeros(
                    max_split_count,
                    device=self.stable_ids.device,
                    dtype=torch.uint8,
                )
            )
            appended_generations.append(
                pad_first_dim(child_generation, max_split_count)
            )

        state = StreamingGaussianState(
            gaussians=Gaussians(
                means=torch.cat([means, torch.stack(appended_means)], dim=1),
                harmonics=torch.cat(
                    [harmonics, torch.stack(appended_harmonics)], dim=1
                ),
                opacities=torch.cat(
                    [opacities, torch.stack(appended_opacities)], dim=1
                ),
                scales=torch.cat(
                    [scales, torch.stack(appended_scales)], dim=1
                ),
                rotations=torch.cat(
                    [rotations, torch.stack(appended_rotations)], dim=1
                ),
            ),
            stable_ids=torch.cat(
                [self.stable_ids, torch.stack(appended_ids)], dim=1
            ),
            observation_count=torch.cat(
                [observation_count, torch.stack(appended_observations)], dim=1
            ),
            point_map=torch.cat(
                [point_map, torch.stack(appended_point_maps)], dim=1
            ),
            front_conflict_count=torch.cat(
                [front_conflict_count, torch.stack(appended_front_conflicts)],
                dim=1,
            ),
            split_generation=torch.cat(
                [split_generation, torch.stack(appended_generations)], dim=1
            ),
        )
        selected_radius = torch.cat(selected_radii)
        selected_gradient = torch.cat(selected_gradients)
        map_count_after = active_count + split_count
        return state, {
            "eligible_count": eligible_count,
            "split_count": split_count,
            "split_ratio": split_count / active_count.clamp_min(1.0),
            "selected_radius_mean": selected_radius.mean(),
            "selected_gradient_norm_mean": selected_gradient.mean(),
            "map_count_before": active_count / max(b, 1),
            "map_count_after": map_count_after / max(b, 1),
        }

    def functional_gradient_step(
        self,
        evidence: ProjectedGSEvidence,
        camera_to_world: torch.Tensor,
        *,
        opacity_gradient_evidence: OpacityGradientEvidence | None,
        mean_gradient_evidence: MeanGradientEvidence | None,
        harmonics_gradient_evidence: HarmonicsGradientEvidence | None,
        top_ratio: float,
        min_observations: int,
        mean_learning_rate: float,
        opacity_learning_rate: float,
        harmonics_learning_rate: float,
        max_mean_step_scale: float,
        max_opacity_logit_step: float,
        max_harmonics_step: float,
    ) -> tuple["StreamingGaussianState", dict[str, torch.Tensor]]:
        """Apply one first-order scene-local SGD step to selected old GS."""
        b, n = self.gaussians.opacities.shape
        expected = (b, n)
        for name, value in (
            ("candidate_mask", evidence.candidate_mask),
            ("surface_consistent", evidence.surface_consistent),
            ("old_in_front", evidence.old_in_front),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"Functional update {name} must be {expected}, got "
                    f"{tuple(value.shape)}."
                )
        if not 0.0 <= float(top_ratio) <= 1.0:
            raise ValueError(
                f"Functional update top_ratio must be in [0,1], got {top_ratio}."
            )
        for name, value in (
            ("mean_learning_rate", mean_learning_rate),
            ("opacity_learning_rate", opacity_learning_rate),
            ("harmonics_learning_rate", harmonics_learning_rate),
            ("max_mean_step_scale", max_mean_step_scale),
            ("max_opacity_logit_step", max_opacity_logit_step),
            ("max_harmonics_step", max_harmonics_step),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"Functional update {name} must be finite and non-negative, "
                    f"got {value}."
                )

        zero = self.gaussians.means.new_zeros((), dtype=torch.float32)
        empty_stats = {key: zero for key in FUNCTIONAL_GS_UPDATE_STAT_KEYS}
        if float(top_ratio) == 0.0:
            return self, empty_stats

        mean_signal = (
            None
            if mean_gradient_evidence is None
            else mean_gradient_evidence.signal.detach().float()
        )
        opacity_signal = (
            None
            if opacity_gradient_evidence is None
            else opacity_gradient_evidence.signal.detach().float()
        )
        harmonics_signal = (
            None
            if harmonics_gradient_evidence is None
            else harmonics_gradient_evidence.signal.detach().float()
        )
        if mean_signal is not None and mean_signal.shape != (b, n, 3):
            raise ValueError("Functional mean-gradient signal has the wrong shape.")
        if opacity_signal is not None and opacity_signal.shape != expected:
            raise ValueError("Functional opacity-gradient signal has the wrong shape.")
        if harmonics_signal is not None:
            if harmonics_signal.shape != self.gaussians.harmonics.shape:
                raise ValueError(
                    "Functional harmonics-gradient signal must match harmonics; "
                    f"got {tuple(harmonics_signal.shape)} and "
                    f"{tuple(self.gaussians.harmonics.shape)}."
                )
            score = harmonics_signal.flatten(2).norm(dim=-1)
        else:
            score = torch.zeros(expected, device=self.gaussians.means.device)
            if mean_signal is not None:
                score = score + mean_signal.norm(dim=-1)
            if opacity_signal is not None:
                score = score + opacity_signal.abs()

        eligible = (
            (self.stable_ids >= 0)
            & evidence.candidate_mask.detach()
            & evidence.surface_consistent.detach()
            & ~evidence.old_in_front.detach()
            & (self.observation_count.detach() >= max(1, int(min_observations)))
            & (score > 0)
        )
        selected = torch.zeros_like(eligible)
        for batch_idx in range(b):
            candidates = torch.nonzero(
                eligible[batch_idx], as_tuple=False
            ).squeeze(1)
            if not candidates.numel():
                continue
            count = max(1, int(math.ceil(candidates.numel() * float(top_ratio))))
            count = min(count, int(candidates.numel()))
            top = torch.topk(
                score[batch_idx].index_select(0, candidates),
                count,
                sorted=False,
            ).indices
            selected[batch_idx, candidates.index_select(0, top)] = True

        selected_float = selected.to(torch.float32)
        selected_count = selected_float.sum()
        if not bool(selected.any()):
            return self, empty_stats

        means = self.gaussians.means
        mean_step_world = torch.zeros_like(means, dtype=torch.float32)
        mean_gradient_norm = zero
        if mean_signal is not None and float(mean_learning_rate) > 0.0:
            mean_step_camera = -float(mean_learning_rate) * mean_signal
            step_norm = mean_step_camera.norm(dim=-1, keepdim=True)
            step_limit = (
                self.gaussians.scales.detach().float().amax(dim=-1, keepdim=True)
                * float(max_mean_step_scale)
            )
            mean_step_camera = mean_step_camera * torch.minimum(
                torch.ones_like(step_norm),
                step_limit / step_norm.clamp_min(1e-8),
            )
            mean_step_camera = mean_step_camera * selected_float.unsqueeze(-1)
            rotation = camera_to_world.detach().float()[:, :3, :3]
            mean_step_world = torch.einsum(
                "bij,bnj->bni", rotation, mean_step_camera
            )
            mean_gradient_norm = (
                mean_signal.norm(dim=-1) * selected_float
            ).sum() / selected_count

        opacities = self.gaussians.opacities
        opacity_logit_step = torch.zeros_like(opacities, dtype=torch.float32)
        opacity_gradient_abs = zero
        if opacity_signal is not None and float(opacity_learning_rate) > 0.0:
            opacity_logit_step = (
                -float(opacity_learning_rate) * opacity_signal
            ).clamp(
                -float(max_opacity_logit_step),
                float(max_opacity_logit_step),
            )
            opacity_logit_step = opacity_logit_step * selected_float
            opacity_gradient_abs = (
                opacity_signal.abs() * selected_float
            ).sum() / selected_count
            opacity_logits = torch.logit(
                opacities.float().clamp(1e-5, 1.0 - 1e-5)
            )
            opacities = torch.sigmoid(opacity_logits + opacity_logit_step).to(
                opacities.dtype
            )

        harmonics = self.gaussians.harmonics
        harmonics_step = torch.zeros_like(harmonics, dtype=torch.float32)
        harmonics_gradient_norm = zero
        if (
            harmonics_signal is not None
            and float(harmonics_learning_rate) > 0.0
        ):
            harmonics_step = (
                -float(harmonics_learning_rate) * harmonics_signal
            ).clamp(-float(max_harmonics_step), float(max_harmonics_step))
            harmonics_mask = selected_float
            for _ in range(harmonics_step.ndim - selected_float.ndim):
                harmonics_mask = harmonics_mask.unsqueeze(-1)
            harmonics_step = harmonics_step * harmonics_mask
            harmonics_gradient_norm = (
                harmonics_signal.flatten(2).norm(dim=-1) * selected_float
            ).sum() / selected_count
            harmonics = (harmonics.float() + harmonics_step).to(harmonics.dtype)

        updated = StreamingGaussianState(
            gaussians=Gaussians(
                means=(means.float() + mean_step_world).to(means.dtype),
                harmonics=harmonics,
                opacities=opacities,
                scales=self.gaussians.scales,
                rotations=self.gaussians.rotations,
            ),
            stable_ids=self.stable_ids,
            observation_count=self.observation_count,
            point_map=self.point_map,
            front_conflict_count=self.front_conflict_count,
            split_generation=self.split_generation,
        )
        active_count = (self.stable_ids >= 0).float().sum().clamp_min(1.0)
        return updated, {
            "selected_count": selected_count,
            "selected_ratio": selected_count / active_count,
            "score_mean": (score * selected_float).sum() / selected_count,
            "mean_gradient_norm": mean_gradient_norm,
            "mean_step_norm": (
                mean_step_world.norm(dim=-1) * selected_float
            ).sum()
            / selected_count,
            "opacity_gradient_abs": opacity_gradient_abs,
            "opacity_logit_step_abs": (
                opacity_logit_step.abs() * selected_float
            ).sum()
            / selected_count,
            "harmonics_gradient_norm": harmonics_gradient_norm,
            "harmonics_step_norm": (
                harmonics_step.flatten(2).norm(dim=-1) * selected_float
            ).sum()
            / selected_count,
        }

    def delete_historical(
        self,
        gir: DominantGIR,
        delete_logit: torch.Tensor,
        min_observations: int,
        threshold: float,
        temperature: float,
        physical_prune: bool,
    ) -> tuple["StreamingGaussianState", dict[str, torch.Tensor]]:
        """Aggregate top-1 pixel decisions and suppress matched historical GS."""
        b, n = self.gaussians.opacities.shape
        delete_probabilities = []
        candidate_masks = []

        # Keep scatter reductions in float32. Half-precision 0/0 on unsupported
        # Gaussians was the source of unstable gradients in earlier experiments.
        pixel_probability = torch.sigmoid(
            delete_logit.float() / max(float(temperature), 1e-4)
        )
        for batch_idx in range(b):
            point_indices = gir.indices[batch_idx].reshape(-1)
            valid = point_indices >= 0
            safe_indices = point_indices.clamp_min(0)
            support = (
                gir.dominant_weight[batch_idx].reshape(-1).detach().float()
                * valid.float()
            )
            weighted_probability = (
                pixel_probability[batch_idx].reshape(-1) * support
            )
            numerator = torch.zeros(
                n, device=delete_logit.device, dtype=torch.float32
            ).scatter_add(0, safe_indices, weighted_probability)
            denominator = torch.zeros_like(numerator).scatter_add(
                0, safe_indices, support
            )
            has_support = denominator > 0
            probability = torch.where(
                has_support,
                numerator / denominator.clamp_min(1e-8),
                torch.zeros_like(numerator),
            )
            candidate = has_support & (
                self.observation_count[batch_idx].detach()
                >= float(max(1, min_observations))
            )
            delete_probabilities.append(probability)
            candidate_masks.append(candidate)

        delete_soft = torch.stack(delete_probabilities)
        candidate_mask = torch.stack(candidate_masks)
        delete_soft = delete_soft * candidate_mask.to(delete_soft.dtype)
        delete_hard = candidate_mask & (delete_soft >= float(threshold))

        physical_keep = None
        if physical_prune:
            if b != 1:
                raise RuntimeError(
                    "Physical historical GS pruning is test-only and requires batch size 1."
                )
            if bool(delete_hard.any()):
                physical_keep = ~delete_hard[0]
                if not bool(physical_keep.any()):
                    safeguard_index = (
                        self.gaussians.opacities[0].detach().float().argmax()
                    )
                    delete_hard = delete_hard.clone()
                    delete_hard[0, safeguard_index] = False
                    physical_keep = ~delete_hard[0]

        if self.gaussians.opacities.requires_grad or delete_logit.requires_grad:
            delete_gate = (
                delete_hard.to(delete_soft.dtype).detach()
                - delete_soft.detach()
                + delete_soft
            )
        else:
            delete_gate = delete_hard.to(delete_soft.dtype)

        opacity_before = self.gaussians.opacities
        opacity_after = opacity_before * (1.0 - delete_gate).to(
            opacity_before.dtype
        )
        state = StreamingGaussianState(
            gaussians=Gaussians(
                means=self.gaussians.means,
                harmonics=self.gaussians.harmonics,
                opacities=opacity_after,
                scales=self.gaussians.scales,
                rotations=self.gaussians.rotations,
            ),
            stable_ids=self.stable_ids,
            observation_count=self.observation_count,
            point_map=self.point_map,
            front_conflict_count=self.front_conflict_count,
            split_generation=self.split_generation,
        )

        pruned_count = delete_hard.sum()
        removed_opacity = (
            opacity_before.float() * delete_hard.to(opacity_before.dtype)
        ).sum()
        map_count_before = torch.tensor(
            float(n), device=delete_logit.device, dtype=torch.float32
        )
        if physical_keep is not None:
            state = state.select(physical_keep)
            pruned_count = (~physical_keep).sum()

        candidate_count = candidate_mask.sum()
        candidate_probability_sum = (
            delete_soft * candidate_mask.to(delete_soft.dtype)
        ).sum()
        candidate_delete_rate = (
            delete_gate * candidate_mask.to(delete_gate.dtype)
        ).sum() / candidate_count.clamp_min(1).float()
        stats = {
            "candidate_count": candidate_count.float(),
            "candidate_probability_sum": candidate_probability_sum,
            "candidate_probability_mean": candidate_probability_sum
            / candidate_count.clamp_min(1).float(),
            "candidate_delete_rate": candidate_delete_rate,
            "hard_deleted_count": pruned_count.float(),
            "hard_deleted_ratio": pruned_count.float()
            / map_count_before.clamp_min(1.0),
            "removed_opacity_mass_ratio": removed_opacity
            / opacity_before.detach().float().sum().clamp_min(1e-8),
            "map_count_before": map_count_before,
            "map_count_after": torch.tensor(
                float(state.num_gaussians),
                device=delete_logit.device,
                dtype=torch.float32,
            ),
        }
        return state, stats

    def decay_historical_opacity(
        self,
        gir: DominantGIR,
        delete_logit: torch.Tensor,
        min_observations: int,
        temperature: float,
        decay_strength: float,
        decay_schedule: float,
        min_contributor_weight: float,
        physical_prune: bool,
        prune_threshold: float,
    ) -> tuple["StreamingGaussianState", dict[str, torch.Tensor]]:
        """Apply contributor-conditioned soft opacity decay to historical GS."""
        b, n = self.gaussians.opacities.shape
        pixel_probability = torch.sigmoid(
            delete_logit.float() / max(float(temperature), 1e-4)
        )
        if pixel_probability.dim() != 4 or pixel_probability.shape[0] != b:
            raise RuntimeError(
                "Old-GS decay expects logits shaped [B, K, H, W], "
                f"got {tuple(pixel_probability.shape)}."
            )

        if gir.contributor_ids is not None and gir.contributor_weights is not None:
            contributor_ids = gir.contributor_ids
            contributor_weights = gir.contributor_weights
        else:
            contributor_ids = gir.indices.unsqueeze(1)
            contributor_weights = gir.dominant_weight.unsqueeze(1)

        requested_count = pixel_probability.shape[1]
        contributor_count = contributor_ids.shape[1]
        if requested_count not in {1, contributor_count}:
            if requested_count > contributor_count:
                raise RuntimeError(
                    "Old-GS decay received fewer contributors than decay maps: "
                    f"maps={requested_count}, contributors={contributor_count}."
                )
            contributor_count = requested_count
            contributor_ids = contributor_ids[:, :contributor_count]
            contributor_weights = contributor_weights[:, :contributor_count]
        if contributor_weights.shape[:2] != contributor_ids.shape[:2]:
            raise RuntimeError(
                "Old-GS decay contributor ID/weight shape mismatch: "
                f"ids={tuple(contributor_ids.shape)}, "
                f"weights={tuple(contributor_weights.shape)}."
            )

        decay_probabilities = []
        candidate_masks = []
        decay_mass_ratios = []
        pruned_counts = []
        map_counts_before = []
        map_counts_after = []

        decay_scale = max(0.0, min(1.0, float(decay_schedule)))
        decay_strength = max(0.0, float(decay_strength)) * decay_scale
        prune_threshold = max(0.0, float(prune_threshold))

        for batch_idx in range(b):
            point_indices = contributor_ids[batch_idx].reshape(
                contributor_count, -1
            ).long()
            weights = torch.nan_to_num(
                contributor_weights[batch_idx].float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0).reshape(contributor_count, -1)
            pixel_valid = gir.valid[batch_idx, 0].reshape(1, -1)
            valid = (
                (point_indices >= 0)
                & (point_indices < n)
                & pixel_valid
                & (weights >= float(min_contributor_weight))
            )
            support = torch.where(valid, weights, torch.zeros_like(weights))
            safe_indices = point_indices.clamp_min(0).clamp_max(max(n - 1, 0))
            if requested_count == 1:
                pixel_probability_flat = pixel_probability[
                    batch_idx, 0
                ].reshape(1, -1).expand_as(support)
            else:
                pixel_probability_flat = pixel_probability[
                    batch_idx, :contributor_count
                ].reshape(contributor_count, -1)

            flat_indices = safe_indices.reshape(-1)
            flat_support = support.reshape(-1)
            flat_probability = pixel_probability_flat.reshape(-1)
            numerator = torch.zeros(
                n, device=delete_logit.device, dtype=torch.float32
            ).scatter_add(0, flat_indices, flat_support * flat_probability)
            denominator = torch.zeros_like(numerator).scatter_add(
                0, flat_indices, flat_support
            )
            has_support = denominator > 0
            probability = torch.where(
                has_support,
                numerator / denominator.clamp_min(1e-8),
                torch.zeros_like(numerator),
            )
            candidate = has_support & (
                self.observation_count[batch_idx].detach()
                >= float(max(1, min_observations))
            )
            candidate = candidate.detach()
            probability = probability * candidate.to(probability.dtype)

            opacity_before = self.gaussians.opacities[batch_idx]
            decay_amount = (decay_strength * probability).clamp_min(0.0)
            opacity_after = opacity_before * torch.exp(-decay_amount).to(
                opacity_before.dtype
            )
            decay_mass = (opacity_before.float() - opacity_after.float()).sum()
            decay_mass_ratios.append(
                decay_mass / opacity_before.detach().float().sum().clamp_min(1e-8)
            )

            physical_keep = None
            if physical_prune:
                if b != 1:
                    raise RuntimeError(
                        "Physical old-GS decay pruning is test-only and requires "
                        "batch size 1."
                    )
                physical_keep = opacity_after.detach().float() > prune_threshold
                if not bool(physical_keep.any()):
                    safeguard_index = opacity_after.detach().float().argmax()
                    physical_keep = physical_keep.clone()
                    physical_keep[safeguard_index] = True

            state = StreamingGaussianState(
                gaussians=Gaussians(
                    means=self.gaussians.means,
                    harmonics=self.gaussians.harmonics,
                    opacities=torch.cat(
                        [
                            self.gaussians.opacities[:batch_idx],
                            opacity_after.unsqueeze(0),
                            self.gaussians.opacities[batch_idx + 1 :],
                        ],
                        dim=0,
                    ),
                    scales=self.gaussians.scales,
                    rotations=self.gaussians.rotations,
                ),
                stable_ids=self.stable_ids,
                observation_count=self.observation_count,
                point_map=self.point_map,
                front_conflict_count=self.front_conflict_count,
                split_generation=self.split_generation,
            )
            if physical_keep is not None:
                state = state.select(physical_keep)
                pruned_count = (~physical_keep).sum().float()
            else:
                pruned_count = torch.zeros(
                    (), device=delete_logit.device, dtype=torch.float32
                )

            decay_probabilities.append(probability)
            candidate_masks.append(candidate)
            pruned_counts.append(pruned_count)
            map_counts_before.append(
                torch.tensor(float(n), device=delete_logit.device)
            )
            map_counts_after.append(
                torch.tensor(
                    float(state.num_gaussians),
                    device=delete_logit.device,
                )
            )

        if b != 1:
            # Training normally uses batch one for streaming state. For a
            # general batch, preserve the differentiable opacity updates from
            # every sample without physical filtering.
            opacity_after_all = []
            for batch_idx, probability in enumerate(decay_probabilities):
                opacity_after_all.append(
                    self.gaussians.opacities[batch_idx]
                    * torch.exp(-decay_strength * probability).to(
                        self.gaussians.opacities.dtype
                    )
                )
            state = StreamingGaussianState(
                gaussians=Gaussians(
                    means=self.gaussians.means,
                    harmonics=self.gaussians.harmonics,
                    opacities=torch.stack(opacity_after_all),
                    scales=self.gaussians.scales,
                    rotations=self.gaussians.rotations,
                ),
                stable_ids=self.stable_ids,
                observation_count=self.observation_count,
                point_map=self.point_map,
                front_conflict_count=self.front_conflict_count,
                split_generation=self.split_generation,
            )

        decay_soft = torch.stack(decay_probabilities)
        candidate_mask = torch.stack(candidate_masks)
        candidate_count = candidate_mask.sum().float()
        candidate_probability_sum = (
            decay_soft * candidate_mask.to(decay_soft.dtype)
        ).sum()
        return state, {
            "candidate_count": candidate_count,
            "candidate_probability_mean": candidate_probability_sum
            / candidate_count.clamp_min(1.0),
            "decay_opacity_mass_ratio": torch.stack(decay_mass_ratios).mean(),
            "hard_pruned_count": torch.stack(pruned_counts).mean(),
            "hard_pruned_ratio": torch.stack(pruned_counts).mean()
            / torch.stack(map_counts_before).mean().clamp_min(1.0),
            "map_count_before": torch.stack(map_counts_before).mean(),
            "map_count_after": torch.stack(map_counts_after).mean(),
        }

    def update_projected_historical(
        self,
        evidence: ProjectedGSEvidence,
        prediction: ProjectedGSResidualPrediction,
        camera_to_world: torch.Tensor,
        opacity_logit_scale: float,
        front_decay_strength: float,
        decay_schedule: float = 1.0,
        mean_update_mode: str = "relative_depth",
        raw_scale_residual: bool = False,
        raw_opacity_residual: bool = False,
        raw_rotation_residual: bool = False,
        raw_harmonics_residual: bool = False,
        mean_relative_scale: float = 0.02,
        rotation_scale: float = 0.05,
        harmonics_scale: float = 0.10,
    ) -> tuple["StreamingGaussianState", dict[str, torch.Tensor]]:
        """Apply shared-decoder residuals directly to selected historical GS."""
        b, n = self.gaussians.opacities.shape
        expected_shape = (b, n)
        for name, value in (
            ("candidate_mask", evidence.candidate_mask),
            ("surface_consistent", evidence.surface_consistent),
            ("old_in_front", evidence.old_in_front),
            ("camera_depth", evidence.camera_depth),
        ):
            if value.shape != expected_shape:
                raise RuntimeError(
                    f"Projected GS {name} shape mismatch: "
                    f"got={tuple(value.shape)}, expected={expected_shape}."
                )
        mean_update_mode = str(mean_update_mode).lower()
        if mean_update_mode not in {"absolute", "relative_depth"}:
            raise ValueError(
                "mean_update_mode must be absolute or relative_depth; "
                f"got {mean_update_mode!r}."
            )

        flat_indices = evidence.flat_candidate_indices.long()
        candidate_count_int = flat_indices.numel()
        harmonic_dim = self.gaussians.harmonics.shape[-2] * (
            self.gaussians.harmonics.shape[-1]
        )
        vector_shapes = (
            ("delta_mean_camera", prediction.delta_mean_camera, 3),
            ("delta_rotation", prediction.delta_rotation, 3),
            ("delta_log_scale", prediction.delta_log_scale, 3),
            ("delta_harmonics", prediction.delta_harmonics, harmonic_dim),
        )
        for name, value, channels in vector_shapes:
            if value.shape != (candidate_count_int, channels):
                raise RuntimeError(
                    f"Projected GS {name} shape mismatch: got={tuple(value.shape)}, "
                    f"expected={(candidate_count_int, channels)}."
                )
        for name, value in (
            ("delta_opacity_logit", prediction.delta_opacity_logit),
            ("front_decay_logit", prediction.front_decay_logit),
            ("update_gate_logit", prediction.update_gate_logit),
        ):
            if value.shape != (candidate_count_int,):
                raise RuntimeError(
                    f"Projected GS {name} shape mismatch: got={tuple(value.shape)}, "
                    f"expected={(candidate_count_int,)}."
                )
        if candidate_count_int and (
            int(flat_indices.min()) < 0 or int(flat_indices.max()) >= b * n
        ):
            raise RuntimeError("Projected GS candidate indices are out of range.")

        flat_consistent = evidence.surface_consistent.reshape(-1)
        flat_old_in_front = evidence.old_in_front.reshape(-1)
        consistent = flat_consistent.index_select(0, flat_indices).float()
        old_in_front = flat_old_in_front.index_select(0, flat_indices).float()
        gate = prediction.update_gate_logit.float().sigmoid()
        flat_observation_count = self.observation_count.reshape(-1)
        damping = (
            flat_observation_count.detach()
            .float()
            .index_select(0, flat_indices)
            .add(1.0)
            .rsqrt()
        )
        update_weight = gate * damping
        consistent_weight = update_weight * consistent

        raw_mean = prediction.delta_mean_camera.float()
        if mean_update_mode == "relative_depth":
            camera_depth = (
                evidence.camera_depth.reshape(-1)
                .index_select(0, flat_indices)
                .float()
                .clamp_min(1e-4)
            )
            delta_mean_camera = (
                raw_mean.tanh()
                * min(max(float(mean_relative_scale), 0.0), 0.999)
                * camera_depth.unsqueeze(-1)
            )
        else:
            delta_mean_camera = raw_mean
        delta_mean_camera = delta_mean_camera * consistent_weight.unsqueeze(-1)
        batch_indices = torch.div(flat_indices, n, rounding_mode="floor")
        camera_rotation = camera_to_world.detach().float()[
            batch_indices, :3, :3
        ]
        delta_mean_world = torch.bmm(
            delta_mean_camera.unsqueeze(1),
            camera_rotation.transpose(1, 2),
        ).squeeze(1)

        raw_rotation = prediction.delta_rotation.float()
        delta_rotation = (
            raw_rotation
            if raw_rotation_residual
            else raw_rotation.tanh()
            * min(max(float(rotation_scale), 0.0), 0.999)
        )
        delta_rotation = delta_rotation * consistent_weight.unsqueeze(-1)

        raw_scale = prediction.delta_log_scale.float()
        delta_log_scale = (
            raw_scale if raw_scale_residual else raw_scale.tanh()
        ) * consistent_weight.unsqueeze(-1)

        raw_opacity = prediction.delta_opacity_logit.float()
        delta_opacity = (
            raw_opacity if raw_opacity_residual else raw_opacity.tanh()
        )
        delta_opacity = (
            delta_opacity
            * max(0.0, float(opacity_logit_scale))
            * consistent_weight
        )

        raw_harmonics = prediction.delta_harmonics.float()
        delta_harmonics = (
            raw_harmonics
            if raw_harmonics_residual
            else raw_harmonics.tanh()
            * min(max(float(harmonics_scale), 0.0), 0.999)
        )
        delta_harmonics = delta_harmonics * consistent_weight.unsqueeze(-1)

        means_flat = self.gaussians.means.reshape(-1, 3)
        selected_means = means_flat.index_select(0, flat_indices)
        updated_means_selected = selected_means.float() + delta_mean_world

        rotations_flat = self.gaussians.rotations.reshape(-1, 4)
        selected_rotations = rotations_flat.index_select(0, flat_indices)
        delta_quaternion = _axis_angle_to_quaternion_xyzw(delta_rotation)
        updated_rotations_selected = _quat_multiply_xyzw(
            delta_quaternion, selected_rotations.float()
        )
        updated_rotations_selected = F.normalize(
            updated_rotations_selected, dim=-1, eps=1e-8
        )

        scales_flat = self.gaussians.scales.reshape(-1, 3)
        selected_scales = scales_flat.index_select(0, flat_indices)
        updated_scales_selected = selected_scales.float() * torch.exp(
            delta_log_scale.clamp(-2.0, 2.0)
        )
        updated_scales_selected = updated_scales_selected.clamp(1e-6, 0.1)

        harmonics_flat = _flatten_harmonics(self.gaussians.harmonics).reshape(
            b * n, harmonic_dim
        )
        selected_harmonics = harmonics_flat.index_select(0, flat_indices)
        updated_harmonics_selected = selected_harmonics.float() + delta_harmonics

        opacities_flat = self.gaussians.opacities.reshape(-1)
        selected_opacities = opacities_flat.index_select(0, flat_indices)
        opacity_logit = torch.logit(
            selected_opacities.float().clamp(1e-5, 1.0 - 1e-5)
        )
        updated_opacities_selected = torch.sigmoid(opacity_logit + delta_opacity)

        front_decay_probability = prediction.front_decay_logit.float().sigmoid()
        decay_fraction = (
            max(0.0, float(front_decay_strength))
            * max(0.0, min(1.0, float(decay_schedule)))
            * front_decay_probability
            * update_weight
            * old_in_front
        ).clamp(0.0, 0.95)
        updated_opacities_selected = updated_opacities_selected * (
            1.0 - decay_fraction
        )

        residual_energy = (
            raw_mean.square().sum(dim=-1)
            + raw_rotation.square().sum(dim=-1)
            + raw_scale.square().sum(dim=-1)
            + raw_opacity.square()
            + raw_harmonics.square().mean(dim=-1)
        )
        candidate_count = prediction.update_gate_logit.new_tensor(
            float(candidate_count_int)
        ).clamp_min(1.0)
        regularization = (
            residual_energy * consistent
            + front_decay_probability.square() * old_in_front
        ).sum() / candidate_count

        means = means_flat.index_copy(
            0, flat_indices, updated_means_selected.to(means_flat.dtype)
        ).reshape_as(self.gaussians.means)
        rotations = rotations_flat.index_copy(
            0,
            flat_indices,
            updated_rotations_selected.to(rotations_flat.dtype),
        ).reshape_as(self.gaussians.rotations)
        scales = scales_flat.index_copy(
            0, flat_indices, updated_scales_selected.to(scales_flat.dtype)
        ).reshape_as(self.gaussians.scales)
        harmonics = harmonics_flat.index_copy(
            0,
            flat_indices,
            updated_harmonics_selected.to(harmonics_flat.dtype),
        ).reshape_as(self.gaussians.harmonics)
        opacities = opacities_flat.index_copy(
            0,
            flat_indices,
            updated_opacities_selected.to(opacities_flat.dtype),
        ).reshape_as(self.gaussians.opacities)

        observation_count = flat_observation_count.clone()
        observation_count.index_add_(
            0, flat_indices, consistent.to(observation_count.dtype)
        )

        mean_shift = delta_mean_world.norm(dim=-1)
        rotation_shift = delta_rotation.norm(dim=-1)
        scale_relative_shift = (
            updated_scales_selected
            / selected_scales.float().clamp_min(1e-8)
            - 1.0
        ).abs().mean(dim=-1)
        opacity_shift = (
            updated_opacities_selected - selected_opacities.float()
        ).abs()
        harmonics_shift = delta_harmonics.abs().mean(dim=-1)
        state = StreamingGaussianState(
            gaussians=Gaussians(
                means=means,
                harmonics=harmonics,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
            ),
            stable_ids=self.stable_ids,
            observation_count=observation_count.reshape_as(self.observation_count),
            point_map=self.point_map,
            front_conflict_count=self.front_conflict_count,
            split_generation=self.split_generation,
        )
        return state, {
            "regularization": regularization,
            "candidate_ratio": evidence.candidate_mask.float().mean(),
            "surface_consistent_ratio": evidence.surface_consistent.float().mean(),
            "old_in_front_ratio": evidence.old_in_front.float().mean(),
            "gate_mean": gate.sum() / candidate_count,
            "mean_shift_mean": mean_shift.sum() / candidate_count,
            "rotation_shift_mean": rotation_shift.sum() / candidate_count,
            "scale_relative_shift_mean": scale_relative_shift.sum()
            / candidate_count,
            "opacity_shift_mean": opacity_shift.sum() / candidate_count,
            "harmonics_shift_mean": harmonics_shift.sum() / candidate_count,
            "front_decay_probability_mean": (
                front_decay_probability * old_in_front
            ).sum()
            / old_in_front.sum().clamp_min(1.0),
            **_opacity_gradient_update_stats(
                evidence.opacity_gradient_features,
                evidence.opacity_gradient_signal,
                delta_opacity,
            ),
            **_mean_gradient_update_stats(
                evidence.mean_gradient_signal,
                evidence.mean_gradient_current_signal,
                evidence.mean_gradient_replay_signal,
                raw_mean,
                delta_mean_camera,
                update_mask=consistent.bool(),
            ),
            **compute_old_opacity_gradient_effective_stats(
                evidence,
                self.gaussians.opacities,
                state.gaussians.opacities,
            ),
        }

    def update_historical(
        self,
        gir: DominantGIR,
        prediction: GIRPrediction,
        camera_to_world: torch.Tensor,
        update_confidence: torch.Tensor | None = None,
        num_contributors: int = 1,
        mean_update_mode: str = "relative_depth",
        raw_scale_residual: bool = False,
        raw_opacity_residual: bool = False,
        raw_rotation_residual: bool = False,
        raw_harmonics_residual: bool = False,
    ) -> "StreamingGaussianState":
        b, n = self.gaussians.means.shape[:2]
        harmonics_flat = _flatten_harmonics(self.gaussians.harmonics)
        harmonic_dim = harmonics_flat.shape[-1]
        num_contributors = max(1, int(num_contributors))
        mean_update_mode = str(mean_update_mode).lower()
        if mean_update_mode not in {"absolute", "relative_depth"}:
            raise ValueError(
                "mean_update_mode must be absolute or relative_depth; "
                f"got {mean_update_mode!r}."
            )
        use_topk = num_contributors > 1
        if use_topk:
            if gir.contributor_ids is None or gir.contributor_weights is None:
                raise RuntimeError(
                    "Top-k historical update requires contributor IDs and weights."
                )
            if gir.contributor_ids.shape[1] < num_contributors:
                raise RuntimeError(
                    "Top-k historical update received fewer contributors than "
                    f"requested: requested={num_contributors}, "
                    f"available={gir.contributor_ids.shape[1]}."
                )

        mean_updates = []
        rotation_updates = []
        scale_updates = []
        opacity_updates = []
        harmonic_updates = []
        observation_increments = []

        for batch_idx in range(b):
            if use_topk:
                point_indices = gir.contributor_ids[
                    batch_idx, :num_contributors
                ].reshape(num_contributors, -1)
                contribution = gir.contributor_weights[
                    batch_idx, :num_contributors
                ].float().reshape(num_contributors, -1).clamp_min(0.0)
                valid = (point_indices >= 0) & (point_indices < n)
                safe_indices = point_indices.clamp(min=0, max=max(n - 1, 0))

                # Contributor depths are computed from the actual Gaussian
                # centers. This matters because lower-ranked splats do not
                # share the dominant splat's depth.
                means = self.gaussians.means[batch_idx].detach().float()
                ones = torch.ones(
                    (n, 1), device=means.device, dtype=means.dtype
                )
                means_h = torch.cat([means, ones], dim=-1)
                world_to_camera = torch.linalg.inv(
                    camera_to_world[batch_idx].detach().float()
                )
                camera_points = means_h @ world_to_camera.transpose(0, 1)
                contributor_depth = camera_points[:, 2].gather(
                    0, safe_indices.reshape(-1)
                ).reshape_as(safe_indices)
                valid = valid & (contributor_depth > 1e-5)

                dominant_weight = gir.dominant_weight[batch_idx].float().reshape(
                    1, -1
                ).clamp_min(1e-8)
                relative_slot_weight = (
                    contribution / dominant_weight
                ).clamp(0.0, 1.0)
            else:
                point_indices = gir.indices[batch_idx].reshape(-1)
                valid = point_indices >= 0
                safe_indices = point_indices.clamp_min(0)
                contribution = gir.dominant_weight[batch_idx].reshape(
                    -1
                ).clamp_min(0.0)
                relative_slot_weight = None
            visible = valid.to(self.gaussians.means.dtype)

            def flatten_prediction(
                primary: torch.Tensor,
                ranked: torch.Tensor | None,
            ) -> torch.Tensor:
                primary_value = primary[batch_idx].permute(1, 2, 0).reshape(
                    -1, primary.shape[1]
                )
                if not use_topk:
                    return primary_value
                if ranked is None:
                    return primary_value.unsqueeze(0).expand(
                        num_contributors, -1, -1
                    )
                if ranked.shape[1] < num_contributors:
                    raise RuntimeError(
                        "Historical prediction has fewer rank heads than "
                        f"requested: requested={num_contributors}, "
                        f"available={ranked.shape[1]}."
                    )
                return ranked[batch_idx, :num_contributors].permute(
                    0, 2, 3, 1
                ).reshape(num_contributors, -1, ranked.shape[2])

            gate = flatten_prediction(
                prediction.historical_gate,
                prediction.rank_historical_gate,
            ).squeeze(-1).sigmoid()
            old_count = self.observation_count[batch_idx].gather(
                0, safe_indices.reshape(-1)
            ).reshape_as(safe_indices)
            damping = old_count.add(1.0).rsqrt()
            support_weight = visible * contribution
            update_weight = support_weight * gate * damping
            if relative_slot_weight is not None:
                update_weight = update_weight * relative_slot_weight
            if update_confidence is not None:
                confidence = update_confidence[batch_idx].reshape(-1).to(
                    update_weight.dtype
                )
                if use_topk:
                    confidence = confidence.unsqueeze(0).expand(
                        num_contributors, -1
                    )
                update_weight = update_weight * confidence

            delta_mean_camera = flatten_prediction(
                prediction.delta_mean_camera,
                prediction.rank_delta_mean_camera,
            )
            if mean_update_mode == "relative_depth":
                if use_topk:
                    depth = contributor_depth.clamp_min(1e-4)
                else:
                    depth = gir.depth[batch_idx].reshape(-1).clamp_min(1e-4)
                delta_mean_camera = (
                    delta_mean_camera.tanh() * depth.unsqueeze(-1)
                )
            rotation_c2w = camera_to_world[batch_idx, :3, :3].to(
                delta_mean_camera.dtype
            )
            delta_mean_world = delta_mean_camera @ rotation_c2w.transpose(0, 1)

            def aggregate(values: torch.Tensor) -> torch.Tensor:
                # GIR normally runs under mixed precision.  In float16,
                # 1e-8 used by the denominator clamp underflows to zero, so
                # pixels with no valid contributor become 0/0 and poison the
                # entire streaming state even when the predicted residual is
                # exactly zero.  Accumulate and normalize in float32, then
                # convert the finite update back to the Gaussian dtype.
                output_dtype = values.dtype
                values = values.reshape(-1, values.shape[-1]).float()
                flat_update_weight = update_weight.reshape(-1).float()
                flat_support_weight = support_weight.reshape(-1).float()
                flat_indices = safe_indices.reshape(-1)
                weighted = values * flat_update_weight.unsqueeze(-1)
                index = flat_indices.unsqueeze(-1).expand(-1, values.shape[-1])
                numerator = torch.zeros(
                    (n, values.shape[-1]),
                    device=values.device,
                    dtype=torch.float32,
                ).scatter_add(0, index, weighted)
                denominator = torch.zeros(
                    n,
                    device=values.device,
                    dtype=torch.float32,
                ).scatter_add(0, flat_indices, flat_support_weight)
                return (
                    numerator / denominator.clamp_min(1e-8).unsqueeze(-1)
                ).to(output_dtype)

            mean_updates.append(aggregate(delta_mean_world))

            delta_rotation = flatten_prediction(
                prediction.delta_rotation,
                prediction.rank_delta_rotation,
            )
            if not raw_rotation_residual:
                delta_rotation = delta_rotation.tanh()
            rotation_updates.append(aggregate(delta_rotation))

            delta_scale = flatten_prediction(
                prediction.delta_log_scale,
                prediction.rank_delta_log_scale,
            )
            if not raw_scale_residual:
                delta_scale = delta_scale.tanh()
            scale_updates.append(aggregate(delta_scale))

            delta_opacity = flatten_prediction(
                prediction.delta_opacity_logit,
                prediction.rank_delta_opacity_logit,
            )
            if not raw_opacity_residual:
                delta_opacity = delta_opacity.tanh()
            opacity_updates.append(aggregate(delta_opacity))

            delta_harmonics = flatten_prediction(
                prediction.delta_harmonics,
                prediction.rank_delta_harmonics,
            )
            if not raw_harmonics_residual:
                delta_harmonics = delta_harmonics.tanh()
            harmonic_updates.append(aggregate(delta_harmonics))

            observation_increments.append(
                torch.zeros(
                    n,
                    device=visible.device,
                    dtype=self.observation_count.dtype,
                )
                .scatter_add(0, safe_indices.reshape(-1), visible.reshape(-1))
                .clamp_max(1.0)
            )

        delta_mean = torch.stack(mean_updates)
        delta_rotation = torch.stack(rotation_updates)
        delta_log_scale = torch.stack(scale_updates)
        delta_opacity = torch.stack(opacity_updates).squeeze(-1)
        delta_harmonics = torch.stack(harmonic_updates)

        delta_quaternion = _axis_angle_to_quaternion_xyzw(delta_rotation)
        rotations = _quat_multiply_xyzw(delta_quaternion, self.gaussians.rotations)
        rotations = F.normalize(rotations, dim=-1, eps=1e-8)

        opacity_logit = torch.logit(
            self.gaussians.opacities.clamp(1e-5, 1.0 - 1e-5)
        )
        updated_opacity = torch.sigmoid(opacity_logit + delta_opacity)
        updated_scales = self.gaussians.scales * torch.exp(
            delta_log_scale.clamp(-2.0, 2.0)
        )
        updated_harmonics = _restore_harmonics(
            harmonics_flat + delta_harmonics,
            self.gaussians.harmonics,
        )

        return StreamingGaussianState(
            gaussians=Gaussians(
                means=self.gaussians.means + delta_mean,
                harmonics=updated_harmonics,
                opacities=updated_opacity,
                scales=updated_scales.clamp(1e-6, 0.1),
                rotations=rotations,
            ),
            stable_ids=self.stable_ids,
            observation_count=self.observation_count
            + torch.stack(observation_increments),
            point_map=self.point_map,
            front_conflict_count=self.front_conflict_count,
            split_generation=self.split_generation,
        )


class DominantGIRRenderer(nn.Module):
    """Projects one front-most historical Gaussian center into each GIR pixel."""

    def __init__(self, min_opacity: float = 1e-4) -> None:
        super().__init__()
        self.min_opacity = min_opacity

    @torch.no_grad()
    def forward(
        self,
        state: StreamingGaussianState,
        camera_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> DominantGIR:
        height, width = image_shape
        means = state.gaussians.means.detach().float()
        b, n = means.shape[:2]
        result = DominantGIR.empty(
            b, height, width, means.device, state.gaussians.means.dtype
        )

        world_to_camera = torch.linalg.inv(camera_to_world.detach().float())
        ones = torch.ones((b, n, 1), device=means.device, dtype=means.dtype)
        means_h = torch.cat([means, ones], dim=-1)
        camera_points = torch.einsum("bij,bnj->bni", world_to_camera, means_h)
        z = camera_points[..., 2]

        fx = intrinsics[:, 0, 0].detach().float().unsqueeze(1) * width
        fy = intrinsics[:, 1, 1].detach().float().unsqueeze(1) * height
        cx = intrinsics[:, 0, 2].detach().float().unsqueeze(1) * width
        cy = intrinsics[:, 1, 2].detach().float().unsqueeze(1) * height
        z_safe = z.clamp_min(1e-6)
        u = fx * camera_points[..., 0] / z_safe + cx
        v = fy * camera_points[..., 1] / z_safe + cy
        pixel_x = torch.floor(u).long()
        pixel_y = torch.floor(v).long()

        valid_point = (
            (z > 1e-5)
            & (pixel_x >= 0)
            & (pixel_x < width)
            & (pixel_y >= 0)
            & (pixel_y < height)
            & (state.gaussians.opacities.detach() > self.min_opacity)
        )

        harmonics = _flatten_harmonics(state.gaussians.harmonics.detach())
        rgb = sh_utils.SH2RGB(harmonics[..., :3]).clamp(0.0, 1.0)
        scale = state.gaussians.scales.detach().norm(dim=-1)
        pixel_count = height * width

        for batch_idx in range(b):
            point_ids = torch.arange(n, device=means.device, dtype=torch.long)
            valid_ids = point_ids[valid_point[batch_idx]]
            if valid_ids.numel() == 0:
                continue

            flat_pixels = (
                pixel_y[batch_idx, valid_ids] * width
                + pixel_x[batch_idx, valid_ids]
            )
            valid_depth = z[batch_idx, valid_ids]
            min_depth = torch.full(
                (pixel_count,), float("inf"), device=means.device, dtype=valid_depth.dtype
            )
            min_depth.scatter_reduce_(
                0, flat_pixels, valid_depth, reduce="amin", include_self=True
            )
            is_front = valid_depth <= min_depth.gather(0, flat_pixels) + 1e-6
            candidates = torch.where(
                is_front,
                valid_ids,
                torch.full_like(valid_ids, n),
            )
            winners = torch.full(
                (pixel_count,), n, device=means.device, dtype=torch.long
            )
            winners.scatter_reduce_(
                0, flat_pixels, candidates, reduce="amin", include_self=True
            )
            valid_pixel = winners < n
            safe_winners = winners.clamp_max(max(n - 1, 0))

            result.indices[batch_idx].view(-1)[valid_pixel] = winners[valid_pixel]
            result.stable_ids[batch_idx].view(-1)[valid_pixel] = state.stable_ids[
                batch_idx
            ].gather(0, safe_winners[valid_pixel])
            result.valid[batch_idx].view(-1)[valid_pixel] = True
            result.depth[batch_idx].view(-1)[valid_pixel] = z[batch_idx].gather(
                0, safe_winners[valid_pixel]
            ).to(result.depth.dtype)
            result.opacity[batch_idx].view(-1)[valid_pixel] = (
                state.gaussians.opacities[batch_idx]
                .detach()
                .gather(0, safe_winners[valid_pixel])
                .to(result.opacity.dtype)
            )
            result.scale[batch_idx].view(-1)[valid_pixel] = scale[batch_idx].gather(
                0, safe_winners[valid_pixel]
            ).to(result.scale.dtype)
            result.observation_count[batch_idx].view(-1)[valid_pixel] = (
                state.observation_count[batch_idx]
                .detach()
                .gather(0, safe_winners[valid_pixel])
                .to(result.observation_count.dtype)
            )
            result.dominant_weight[batch_idx].view(-1)[valid_pixel] = 1.0
            result.rgb[batch_idx].reshape(3, -1)[:, valid_pixel] = rgb[
                batch_idx
            ].gather(
                0, safe_winners[valid_pixel, None].expand(-1, 3)
            ).transpose(0, 1).to(result.rgb.dtype)

        return result


class GIRUpdateHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        harmonic_dim: int,
        hidden_dim: int = 64,
        use_raster_evidence: bool = False,
    ) -> None:
        super().__init__()
        self.harmonic_dim = harmonic_dim
        self.use_raster_evidence = use_raster_evidence
        evidence_dim = feature_dim + 15 + int(use_raster_evidence)
        output_dim = 3 + 3 + 3 + 1 + harmonic_dim + 1 + 1
        current_output_dim = 3 + 3 + 3 + 1 + harmonic_dim + 1
        groups = _group_count(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(evidence_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.prediction = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        self.additional_historical_predictions = nn.ModuleList()
        self.current_prediction = nn.Conv2d(
            hidden_dim + 1,
            current_output_dim,
            kernel_size=1,
        )
        self.delete_prediction = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        contributor_evidence_dim = hidden_dim + 7
        self.contributor_decay_prediction = nn.Sequential(
            nn.Conv2d(
                contributor_evidence_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)
        with torch.no_grad():
            # The final prediction channel is add_logit. Start near identity so
            # a newly initialized GIR does not suppress the stage-1 GS map.
            self.prediction.bias[-1] = 4.0
        nn.init.zeros_(self.current_prediction.weight)
        nn.init.zeros_(self.current_prediction.bias)
        nn.init.zeros_(self.delete_prediction.weight)
        nn.init.constant_(self.delete_prediction.bias, -4.0)
        nn.init.zeros_(self.contributor_decay_prediction[-1].weight)
        nn.init.constant_(self.contributor_decay_prediction[-1].bias, -4.0)

    def configure_historical_prediction_heads(self, num_heads: int) -> None:
        """Create independent historical heads for contributor ranks."""
        required_additional = max(1, int(num_heads)) - 1
        while len(self.additional_historical_predictions) < required_additional:
            self.additional_historical_predictions.append(
                copy.deepcopy(self.prediction)
            )

    def forward(
        self,
        current_feature: torch.Tensor,
        current_rgb: torch.Tensor,
        current_depth: torch.Tensor,
        current_depth_confidence: torch.Tensor,
        gir: DominantGIR,
        num_historical_predictions: int = 1,
        decay_gir: DominantGIR | None = None,
        num_decay_predictions: int = 0,
    ) -> GIRPrediction:
        size = gir.depth.shape[-2:]
        current_feature = F.interpolate(
            current_feature, size=size, mode="bilinear", align_corners=False
        )
        current_rgb = F.interpolate(
            current_rgb.float(), size=size, mode="bilinear", align_corners=False
        ).to(current_feature.dtype)
        current_depth = F.interpolate(
            current_depth.float(), size=size, mode="bilinear", align_corners=False
        ).to(current_feature.dtype)
        current_depth_confidence = F.interpolate(
            current_depth_confidence.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).to(current_feature.dtype)

        log_depth = current_depth.clamp_min(1e-4).log().clamp(-8.0, 8.0)

        def encode_gir(evidence_gir: DominantGIR) -> torch.Tensor:
            valid = evidence_gir.valid.to(current_feature.dtype)
            historical_rgb = evidence_gir.rgb.to(current_feature.dtype)
            historical_depth = (
                evidence_gir.raster_depth
                if self.use_raster_evidence
                else evidence_gir.depth
            ).to(current_feature.dtype)
            relative_depth = (
                (historical_depth - current_depth)
                / current_depth.clamp_min(1e-4)
            ).clamp(-2.0, 2.0)
            evidence_parts = [
                current_feature,
                current_rgb,
                historical_rgb,
                current_rgb - historical_rgb,
                log_depth,
                relative_depth,
                evidence_gir.opacity.to(current_feature.dtype),
                evidence_gir.scale.to(current_feature.dtype),
                current_depth_confidence,
                valid,
            ]
            if self.use_raster_evidence:
                evidence_parts.append(
                    evidence_gir.raster_alpha.to(current_feature.dtype)
                )
            return self.encoder(torch.cat(evidence_parts, dim=1))

        encoded = encode_gir(gir)
        base_prediction = self.prediction(encoded)
        base_splits = torch.split(
            base_prediction,
            [3, 3, 3, 1, self.harmonic_dim, 1, 1],
            dim=1,
        )
        num_historical_predictions = max(1, int(num_historical_predictions))
        available_predictions = 1 + len(self.additional_historical_predictions)
        if num_historical_predictions > available_predictions:
            raise RuntimeError(
                "GIRUpdateHead has fewer historical prediction heads than "
                f"requested: requested={num_historical_predictions}, "
                f"available={available_predictions}."
            )
        historical_splits = base_splits[:6]
        rank_predictions = [None] * 6
        if num_historical_predictions > 1:
            rank_splits = [base_splits]
            for head in self.additional_historical_predictions[
                : num_historical_predictions - 1
            ]:
                rank_splits.append(
                    torch.split(
                        head(encoded),
                        [3, 3, 3, 1, self.harmonic_dim, 1, 1],
                        dim=1,
                    )
                )
            rank_predictions = [
                torch.stack(
                    [rank_split[field_idx] for rank_split in rank_splits],
                    dim=1,
                )
                for field_idx in range(6)
            ]
        dominant_weight = gir.dominant_weight.to(encoded.dtype)
        current_prediction = self.current_prediction(
            torch.cat([encoded, dominant_weight], dim=1)
        )
        current_splits = torch.split(
            current_prediction,
            [3, 3, 3, 1, self.harmonic_dim, 1],
            dim=1,
        )
        delete_logit = self.delete_prediction(encoded)
        rank_decay_logits = None
        num_decay_predictions = max(0, int(num_decay_predictions))
        if num_decay_predictions > 0:
            decay_gir = gir if decay_gir is None else decay_gir
            contributor_tensors = (
                decay_gir.contributor_ids,
                decay_gir.contributor_weights,
                decay_gir.contributor_depth,
                decay_gir.contributor_opacity,
                decay_gir.contributor_scale,
                decay_gir.contributor_observation_count,
            )
            if any(value is None for value in contributor_tensors):
                raise RuntimeError(
                    "Contributor-conditioned opacity decay requires per-rank "
                    "IDs, weights, depth, opacity, scale, and observation count."
                )
            available_contributors = decay_gir.contributor_ids.shape[1]
            if available_contributors < num_decay_predictions:
                raise RuntimeError(
                    "Opacity decay received fewer contributors than requested: "
                    f"requested={num_decay_predictions}, "
                    f"available={available_contributors}."
                )

            contributor_depth = decay_gir.contributor_depth[
                :, :num_decay_predictions
            ].to(current_feature.dtype)
            contributor_opacity = decay_gir.contributor_opacity[
                :, :num_decay_predictions
            ].to(current_feature.dtype)
            contributor_scale = decay_gir.contributor_scale[
                :, :num_decay_predictions
            ].to(current_feature.dtype)
            contributor_weight = decay_gir.contributor_weights[
                :, :num_decay_predictions
            ].to(current_feature.dtype)
            contributor_observations = decay_gir.contributor_observation_count[
                :, :num_decay_predictions
            ].to(current_feature.dtype)
            contributor_valid = (
                decay_gir.contributor_ids[:, :num_decay_predictions] >= 0
            ).to(current_feature.dtype)
            current_depth_rank = current_depth[:, 0].unsqueeze(1)
            contributor_relative_depth = (
                (contributor_depth - current_depth_rank)
                / current_depth_rank.clamp_min(1e-4)
            ).clamp(-2.0, 2.0)
            contributor_features = torch.stack(
                [
                    contributor_depth.clamp_min(1e-4).log().clamp(-8.0, 8.0),
                    contributor_relative_depth,
                    contributor_opacity.clamp(0.0, 1.0),
                    contributor_scale.clamp_min(1e-8).log().clamp(-12.0, 2.0),
                    contributor_weight.clamp(0.0, 1.0),
                    contributor_observations.clamp_min(0.0).log1p().clamp_max(8.0),
                    contributor_valid,
                ],
                dim=2,
            )
            decay_encoded = (
                encoded if decay_gir is gir else encode_gir(decay_gir)
            )
            batch_size, _, height, width = decay_encoded.shape
            decay_encoded = decay_encoded[:, None].expand(
                -1, num_decay_predictions, -1, -1, -1
            )
            decay_evidence = torch.cat(
                [decay_encoded, contributor_features], dim=2
            ).reshape(
                batch_size * num_decay_predictions,
                -1,
                height,
                width,
            )
            rank_decay_logits = self.contributor_decay_prediction(
                decay_evidence
            ).reshape(batch_size, num_decay_predictions, height, width)

        return GIRPrediction(
            delta_mean_camera=historical_splits[0],
            delta_rotation=historical_splits[1],
            delta_log_scale=historical_splits[2],
            delta_opacity_logit=historical_splits[3],
            delta_harmonics=historical_splits[4],
            historical_gate=historical_splits[5],
            add_logit=base_splits[6],
            current_delta_mean_camera=current_splits[0],
            current_delta_rotation=current_splits[1],
            current_delta_log_scale=current_splits[2],
            current_delta_opacity_logit=current_splits[3],
            current_delta_harmonics=current_splits[4],
            current_residual_gate=current_splits[5],
            delete_logit=delete_logit,
            rank_delta_mean_camera=rank_predictions[0],
            rank_delta_rotation=rank_predictions[1],
            rank_delta_log_scale=rank_predictions[2],
            rank_delta_opacity_logit=rank_predictions[3],
            rank_delta_harmonics=rank_predictions[4],
            rank_historical_gate=rank_predictions[5],
            rank_decay_logits=rank_decay_logits,
        )
