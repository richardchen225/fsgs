import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder import sh_utils
from src.model.encoder.common.gaussians import build_covariance
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


@dataclass
class ProjectedGSResidualPrediction:
    delta_mean_camera: torch.Tensor
    delta_rotation: torch.Tensor
    delta_log_scale: torch.Tensor
    delta_opacity_logit: torch.Tensor
    delta_harmonics: torch.Tensor
    front_decay_logit: torch.Tensor
    update_gate_logit: torch.Tensor


PROJECTED_GS_STATE_DIM = 30


@torch.no_grad()
def build_point_map_append_mask(
    state: "StreamingGaussianState",
    current_depth: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_relative_tolerance: float = 0.08,
    dilation_radius: int = 1,
    chunk_size: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find current pixels without a depth-consistent historical point.

    Historical Gaussian centers are treated strictly as a point map. Gaussian
    scale, opacity, covariance, and rasterized alpha do not affect coverage.
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
    depth_flat = depth[:, 0].reshape(b, h * w)
    coverage_count = torch.zeros(
        (b, h * w), device=device, dtype=torch.float32
    )
    world_to_camera = torch.linalg.inv(camera_to_world.detach().float())
    fx = intrinsics[:, 0, 0].detach().float().unsqueeze(1) * w
    fy = intrinsics[:, 1, 1].detach().float().unsqueeze(1) * h
    cx = intrinsics[:, 0, 2].detach().float().unsqueeze(1) * w
    cy = intrinsics[:, 1, 2].detach().float().unsqueeze(1) * h
    tolerance = max(0.0, float(depth_relative_tolerance))
    chunk_size = max(1, int(chunk_size))
    point_count = state.num_gaussians

    for start in range(0, point_count, chunk_size):
        end = min(start + chunk_size, point_count)
        point_map = (
            state.point_map
            if state.point_map is not None
            else state.gaussians.means.detach()
        )
        points = point_map[:, start:end].detach().float()
        ones = torch.ones(
            (*points.shape[:2], 1), device=device, dtype=torch.float32
        )
        camera_points = torch.einsum(
            "bij,bnj->bni",
            world_to_camera,
            torch.cat([points, ones], dim=-1),
        )[..., :3]
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
        )
        linear_index = (pixel_y.clamp(0, h - 1) * w + pixel_x.clamp(0, w - 1))
        sampled_depth = depth_flat.gather(1, linear_index)
        depth_consistent = (
            in_image
            & torch.isfinite(sampled_depth)
            & (sampled_depth > 1e-5)
            & (
                (z - sampled_depth).abs()
                <= tolerance * sampled_depth.clamp_min(1e-4)
            )
        )
        coverage_count.scatter_add_(
            1, linear_index, depth_consistent.to(coverage_count.dtype)
        )

    coverage = (coverage_count > 0).reshape(b, 1, h, w)
    radius = max(0, int(dilation_radius))
    if radius > 0:
        kernel_size = 2 * radius + 1
        coverage = F.max_pool2d(
            coverage.float(),
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        ).bool()
    append_mask = valid_current & ~coverage
    return append_mask, coverage


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
        self.output_dim = 12 + self.harmonic_dim
        self.output = nn.Linear(hidden_dim, self.output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[10 + self.harmonic_dim] = -4.0
            self.output.bias[11 + self.harmonic_dim] = 2.0

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
        selected = descriptor
        no_candidates = selected.shape[0] == 0
        if no_candidates:
            # Keep the shared head in the DDP graph even for views with no
            # eligible old GS, while preserving an exact identity update.
            selected = descriptor.new_zeros((1, self.input_dim))
        selected_outputs = []
        for start in range(0, selected.shape[0], self.chunk_size):
            chunk = selected[start : start + self.chunk_size]
            selected_outputs.append(self.output(self.decoder(chunk)))
        selected_output = torch.cat(selected_outputs, dim=0)
        if no_candidates:
            selected_output = selected_output[:0] + selected_output.sum() * 0.0
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
        )

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
