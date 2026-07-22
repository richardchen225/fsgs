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

    def append(self, current: Gaussians, add_gate: torch.Tensor) -> "StreamingGaussianState":
        b, n = current.means.shape[:2]
        gate = add_gate.reshape(b, n).to(current.opacities.dtype)
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
    ) -> "StreamingGaussianState":
        b, n = self.gaussians.means.shape[:2]
        harmonics_flat = _flatten_harmonics(self.gaussians.harmonics)
        harmonic_dim = harmonics_flat.shape[-1]

        mean_updates = []
        rotation_updates = []
        scale_updates = []
        opacity_updates = []
        harmonic_updates = []
        observation_increments = []

        for batch_idx in range(b):
            point_indices = gir.indices[batch_idx].reshape(-1)
            valid = point_indices >= 0
            safe_indices = point_indices.clamp_min(0)
            visible = valid.to(self.gaussians.means.dtype)

            gate = prediction.historical_gate[batch_idx].reshape(-1).sigmoid()
            old_count = self.observation_count[batch_idx].gather(0, safe_indices)
            damping = old_count.add(1.0).rsqrt()
            pixel_weight = visible * gate * damping

            depth = gir.depth[batch_idx].reshape(-1).clamp_min(1e-4)
            delta_mean_camera = prediction.delta_mean_camera[batch_idx]
            delta_mean_camera = delta_mean_camera.permute(1, 2, 0).reshape(-1, 3)
            delta_mean_camera = delta_mean_camera.tanh() * depth.unsqueeze(-1)
            rotation_c2w = camera_to_world[batch_idx, :3, :3].to(
                delta_mean_camera.dtype
            )
            delta_mean_world = delta_mean_camera @ rotation_c2w.transpose(0, 1)

            def aggregate(values: torch.Tensor) -> torch.Tensor:
                weighted = values * pixel_weight.unsqueeze(-1)
                index = safe_indices.unsqueeze(-1).expand(-1, values.shape[-1])
                return torch.zeros(
                    (n, values.shape[-1]),
                    device=values.device,
                    dtype=values.dtype,
                ).scatter_add(0, index, weighted)

            mean_updates.append(aggregate(delta_mean_world))

            delta_rotation = prediction.delta_rotation[batch_idx]
            delta_rotation = delta_rotation.permute(1, 2, 0).reshape(-1, 3).tanh()
            rotation_updates.append(aggregate(delta_rotation))

            delta_scale = prediction.delta_log_scale[batch_idx]
            delta_scale = delta_scale.permute(1, 2, 0).reshape(-1, 3).tanh()
            scale_updates.append(aggregate(delta_scale))

            delta_opacity = prediction.delta_opacity_logit[batch_idx]
            delta_opacity = delta_opacity.permute(1, 2, 0).reshape(-1, 1).tanh()
            opacity_updates.append(aggregate(delta_opacity))

            delta_harmonics = prediction.delta_harmonics[batch_idx]
            delta_harmonics = delta_harmonics.permute(1, 2, 0).reshape(
                -1, harmonic_dim
            ).tanh()
            harmonic_updates.append(aggregate(delta_harmonics))

            observation_increments.append(
                torch.zeros(
                    n,
                    device=visible.device,
                    dtype=self.observation_count.dtype,
                )
                .scatter_add(0, safe_indices, visible)
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
    ) -> None:
        super().__init__()
        self.harmonic_dim = harmonic_dim
        evidence_dim = feature_dim + 15
        output_dim = 3 + 3 + 3 + 1 + harmonic_dim + 1 + 1
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
        nn.init.zeros_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)

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
        historical_depth = gir.depth.to(current_feature.dtype)
        relative_depth = (
            (historical_depth - current_depth) / current_depth.clamp_min(1e-4)
        ).clamp(-2.0, 2.0)
        log_depth = current_depth.clamp_min(1e-4).log().clamp(-8.0, 8.0)
        evidence = torch.cat(
            [
                current_feature,
                current_rgb,
                historical_rgb,
                current_rgb - historical_rgb,
                log_depth,
                relative_depth,
                gir.opacity.to(current_feature.dtype),
                gir.scale.to(current_feature.dtype),
                current_depth_confidence,
                valid,
            ],
            dim=1,
        )
        prediction = self.prediction(self.encoder(evidence))
        splits = torch.split(
            prediction,
            [3, 3, 3, 1, self.harmonic_dim, 1, 1],
            dim=1,
        )
        return GIRPrediction(*splits)
