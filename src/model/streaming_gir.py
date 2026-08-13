from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder import sh_utils
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
    contributor_rgb: torch.Tensor | None = None
    contributor_depth: torch.Tensor | None = None
    contributor_camera_points: torch.Tensor | None = None
    contributor_opacity: torch.Tensor | None = None
    contributor_scales: torch.Tensor | None = None
    contributor_rotations: torch.Tensor | None = None
    contributor_harmonics: torch.Tensor | None = None
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
        )

    def select(self, keep: torch.Tensor) -> "StreamingGaussianState":
        if self.batch_size != 1:
            raise RuntimeError(
                "Test-only GIR map selection currently requires batch size 1."
            )
        keep = keep.reshape(-1).to(device=self.gaussians.means.device, dtype=torch.bool)
        if keep.numel() != self.num_gaussians:
            raise ValueError(
                "GIR map selection mask has the wrong size: "
                f"mask={keep.numel()}, gaussians={self.num_gaussians}."
            )
        if not keep.any():
            raise RuntimeError("GIR map selection cannot remove every Gaussian.")
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
        )

    @classmethod
    def from_current(
        cls,
        current: Gaussians,
        add_gate: torch.Tensor,
    ) -> "StreamingGaussianState":
        b, n = current.means.shape[:2]
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
        )

    def append(
        self,
        current: Gaussians,
        add_gate: torch.Tensor,
        prune_threshold: float = 0.0,
    ) -> "StreamingGaussianState":
        b, n = current.means.shape[:2]
        gate = add_gate.reshape(b, n).to(current.opacities.dtype)
        if prune_threshold > 0.0:
            if b != 1:
                raise RuntimeError(
                    "Test-only GIR gate pruning currently requires batch size 1."
                )
            keep = gate[0] >= prune_threshold
            current = Gaussians(
                means=current.means[:, keep],
                harmonics=current.harmonics[:, keep],
                opacities=current.opacities[:, keep],
                scales=current.scales[:, keep],
                rotations=current.rotations[:, keep],
            )
            gate = gate[:, keep]
            n = int(keep.sum().item())

        first_new_id = self.stable_ids.max(dim=1, keepdim=True).values + 1
        offsets = torch.arange(n, device=current.means.device, dtype=torch.long)
        new_ids = first_new_id + offsets.unsqueeze(0)
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
                [
                    self.observation_count,
                    torch.ones(
                        (b, n),
                        device=current.means.device,
                        dtype=self.observation_count.dtype,
                    ),
                ],
                dim=1,
            ),
        )

    def update_historical(
        self,
        gir: DominantGIR,
        prediction: GIRPrediction,
        camera_to_world: torch.Tensor,
        update_confidence: torch.Tensor | None = None,
        num_contributors: int = 1,
    ) -> "StreamingGaussianState":
        b, n = self.gaussians.means.shape[:2]
        harmonics_flat = _flatten_harmonics(self.gaussians.harmonics)
        harmonic_dim = harmonics_flat.shape[-1]

        num_contributors = max(1, int(num_contributors))
        if (
            num_contributors > 1
            and gir.contributor_ids is not None
            and gir.contributor_weights is not None
        ):
            contributor_count = min(
                num_contributors, gir.contributor_ids.shape[1]
            )
            contributor_ids = gir.contributor_ids[:, :contributor_count]
            contributor_weights = gir.contributor_weights[:, :contributor_count]
        else:
            contributor_count = 1
            contributor_ids = gir.indices.unsqueeze(1)
            contributor_weights = gir.dominant_weight.unsqueeze(1)

        # The rasterizer is detached, but each contributor still uses its own
        # camera-space depth when the shared pixel residual updates geometry.
        with torch.no_grad():
            means = self.gaussians.means.detach().float()
            ones = torch.ones((b, n, 1), device=means.device, dtype=means.dtype)
            means_h = torch.cat([means, ones], dim=-1)
            world_to_camera = torch.linalg.inv(
                camera_to_world.detach().float()
            )
            contributor_depths = torch.einsum(
                "bij,bnj->bni", world_to_camera, means_h
            )[..., 2]

        mean_updates = []
        rotation_updates = []
        scale_updates = []
        opacity_updates = []
        harmonic_updates = []
        observation_increments = []

        for batch_idx in range(b):
            point_indices = contributor_ids[batch_idx].reshape(
                contributor_count, -1
            )
            contribution = contributor_weights[batch_idx].reshape(
                contributor_count, -1
            )
            pixel_valid = gir.valid[batch_idx].reshape(1, -1)
            valid = (
                (point_indices >= 0)
                & (point_indices < n)
                & pixel_valid
                & (contribution > 0)
            )
            safe_indices = point_indices.clamp_min(0).clamp_max(max(n - 1, 0))
            depth = contributor_depths[batch_idx].gather(
                0, safe_indices.reshape(-1)
            ).reshape_as(safe_indices)
            valid = valid & (depth > 1e-5)
            depth = depth.clamp_min(1e-4)

            gate = prediction.historical_gate[
                batch_idx, :contributor_count, 0
            ].reshape(contributor_count, -1).sigmoid()
            old_count = self.observation_count[batch_idx].gather(
                0, safe_indices.reshape(-1)
            ).reshape_as(safe_indices)
            damping = old_count.add(1.0).rsqrt()
            support_weight = torch.where(
                valid,
                contribution.clamp_min(0.0),
                torch.zeros_like(contribution),
            )
            assignment_weight = support_weight / support_weight.sum(
                dim=0, keepdim=True
            ).clamp_min(1e-8)
            update_weight = support_weight * gate * damping
            if update_confidence is not None:
                confidence = update_confidence[batch_idx].to(update_weight.dtype)
                if confidence.dim() == 4:
                    confidence = confidence[:, 0]
                confidence = confidence.reshape(contributor_count, -1)
                update_weight = update_weight * confidence

            def flatten_contributors(values: torch.Tensor) -> torch.Tensor:
                values = values[:contributor_count]
                channels = values.shape[1]
                return values.permute(0, 2, 3, 1).reshape(-1, channels)

            delta_mean_camera = prediction.delta_mean_camera[batch_idx]
            delta_mean_camera = flatten_contributors(delta_mean_camera).tanh()
            delta_mean_camera = delta_mean_camera * depth.reshape(-1, 1).to(
                delta_mean_camera.dtype
            )
            rotation_c2w = camera_to_world[batch_idx, :3, :3].to(
                delta_mean_camera.dtype
            )
            delta_mean_world = delta_mean_camera @ rotation_c2w.transpose(0, 1)

            def aggregate(values: torch.Tensor) -> torch.Tensor:
                flat_update_weight = update_weight.reshape(-1).to(values.dtype)
                flat_support_weight = support_weight.reshape(-1).to(values.dtype)
                flat_indices = safe_indices.reshape(-1)
                weighted = values * flat_update_weight.unsqueeze(-1)
                index = flat_indices.unsqueeze(-1).expand(-1, values.shape[-1])
                numerator = torch.zeros(
                    (n, values.shape[-1]),
                    device=values.device,
                    dtype=values.dtype,
                ).scatter_add(0, index, weighted)
                denominator = torch.zeros(
                    n,
                    device=values.device,
                    dtype=values.dtype,
                ).scatter_add(0, flat_indices, flat_support_weight)
                return numerator / denominator.clamp_min(1e-8).unsqueeze(-1)

            mean_updates.append(aggregate(delta_mean_world))

            delta_rotation = prediction.delta_rotation[batch_idx]
            delta_rotation = flatten_contributors(delta_rotation).tanh()
            rotation_updates.append(aggregate(delta_rotation))

            delta_scale = prediction.delta_log_scale[batch_idx]
            delta_scale = flatten_contributors(delta_scale).tanh()
            scale_updates.append(aggregate(delta_scale))

            delta_opacity = prediction.delta_opacity_logit[batch_idx]
            delta_opacity = flatten_contributors(delta_opacity).tanh()
            opacity_updates.append(aggregate(delta_opacity))

            delta_harmonics = prediction.delta_harmonics[batch_idx]
            delta_harmonics = flatten_contributors(delta_harmonics).tanh()
            harmonic_updates.append(aggregate(delta_harmonics))

            observation_increments.append(
                torch.zeros(
                    n,
                    device=valid.device,
                    dtype=self.observation_count.dtype,
                )
                .scatter_add(
                    0,
                    safe_indices.reshape(-1),
                    assignment_weight.reshape(-1).to(
                        self.observation_count.dtype
                    ),
                )
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
        num_contributors: int = 1,
    ) -> None:
        super().__init__()
        self.harmonic_dim = harmonic_dim
        self.use_raster_evidence = use_raster_evidence
        self.num_contributors = max(1, int(num_contributors))
        base_evidence_dim = feature_dim + 11 + int(use_raster_evidence)
        contributor_evidence_dim = 23 + harmonic_dim
        evidence_dim = (
            base_evidence_dim
            + self.num_contributors * contributor_evidence_dim
        )
        self.historical_output_dim = 3 + 3 + 3 + 1 + harmonic_dim + 1
        output_dim = self.num_contributors * self.historical_output_dim + 1
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
        self.current_prediction = nn.Conv2d(
            hidden_dim + 1,
            current_output_dim,
            kernel_size=1,
        )
        nn.init.zeros_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)
        nn.init.zeros_(self.current_prediction.weight)
        nn.init.zeros_(self.current_prediction.bias)

    def forward(
        self,
        current_feature: torch.Tensor,
        current_rgb: torch.Tensor,
        current_depth: torch.Tensor,
        current_depth_confidence: torch.Tensor,
        gir: DominantGIR,
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

        valid = gir.valid.to(current_feature.dtype)
        historical_rgb = gir.rgb.to(current_feature.dtype)
        log_depth = current_depth.clamp_min(1e-4).log().clamp(-8.0, 8.0)
        evidence_parts = [
            current_feature,
            current_rgb,
            historical_rgb,
            current_rgb - historical_rgb,
            log_depth,
            current_depth_confidence,
        ]
        if self.use_raster_evidence:
            evidence_parts.append(gir.raster_alpha.to(current_feature.dtype))

        contributor_shape = (
            current_feature.shape[0],
            self.num_contributors,
            1,
            *size,
        )

        def contributor_or_zeros(
            values: torch.Tensor | None,
            channels: int,
        ) -> torch.Tensor:
            if values is None:
                return current_feature.new_zeros(
                    contributor_shape[:2] + (channels,) + contributor_shape[3:]
                )
            if values.shape[1] < self.num_contributors:
                raise RuntimeError(
                    "GIR has fewer contributor evidence maps than the head: "
                    f"head={self.num_contributors}, evidence={values.shape[1]}."
                )
            return values[:, : self.num_contributors].to(current_feature.dtype)

        contributor_rgb = contributor_or_zeros(gir.contributor_rgb, 3)
        contributor_depth = contributor_or_zeros(gir.contributor_depth, 1)
        contributor_camera_points = contributor_or_zeros(
            gir.contributor_camera_points, 3
        )
        contributor_opacity = contributor_or_zeros(gir.contributor_opacity, 1)
        contributor_scales = contributor_or_zeros(gir.contributor_scales, 3)
        contributor_rotations = contributor_or_zeros(
            gir.contributor_rotations, 4
        )
        contributor_harmonics = contributor_or_zeros(
            gir.contributor_harmonics, self.harmonic_dim
        )
        contributor_count = contributor_or_zeros(
            gir.contributor_observation_count, 1
        )
        contributor_weight = contributor_or_zeros(
            None
            if gir.contributor_weights is None
            else gir.contributor_weights.unsqueeze(2),
            1,
        )
        contributor_valid = (
            contributor_weight > 0
        ).to(current_feature.dtype)
        current_rgb_per_contributor = current_rgb.unsqueeze(1)
        current_depth_per_contributor = current_depth.unsqueeze(1)
        relative_depth = (
            (contributor_depth - current_depth_per_contributor)
            / current_depth_per_contributor.clamp_min(1e-4)
        ).clamp(-2.0, 2.0)
        normalized_log_scale = (
            contributor_scales.clamp_min(1e-8).log()
            - current_depth_per_contributor.clamp_min(1e-4).log()
        ).clamp(-12.0, 4.0)
        log_contributor_depth = contributor_depth.clamp_min(1e-4).log().clamp(
            -8.0, 8.0
        )
        normalized_camera_points = contributor_camera_points / (
            current_depth_per_contributor.clamp_min(1e-4)
        )
        raster_alpha = gir.raster_alpha.to(current_feature.dtype).unsqueeze(1)
        ownership = (
            contributor_weight / raster_alpha.clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        contributor_evidence = torch.cat(
            [
                contributor_rgb,
                current_rgb_per_contributor - contributor_rgb,
                relative_depth,
                log_contributor_depth,
                normalized_camera_points.clamp(-4.0, 4.0),
                contributor_opacity,
                normalized_log_scale,
                contributor_rotations,
                contributor_harmonics,
                contributor_count.clamp_min(0.0).log1p().clamp_max(8.0),
                contributor_weight,
                ownership,
                contributor_valid,
            ],
            dim=2,
        )
        evidence_parts.append(
            contributor_evidence.flatten(1, 2)
        )
        evidence = torch.cat(evidence_parts, dim=1)
        encoded = self.encoder(evidence)
        prediction = self.prediction(encoded)
        historical_prediction = prediction[:, :-1].reshape(
            prediction.shape[0],
            self.num_contributors,
            self.historical_output_dim,
            *prediction.shape[-2:],
        )
        splits = torch.split(
            historical_prediction,
            [3, 3, 3, 1, self.harmonic_dim, 1],
            dim=2,
        )
        add_logit = prediction[:, -1:]
        dominant_weight = gir.dominant_weight.to(encoded.dtype)
        current_prediction = self.current_prediction(
            torch.cat([encoded, dominant_weight], dim=1)
        )
        current_splits = torch.split(
            current_prediction,
            [3, 3, 3, 1, self.harmonic_dim, 1],
            dim=1,
        )
        return GIRPrediction(*splits, add_logit, *current_splits)
