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
        """Apply a soft, contributor-weighted opacity decay to old GS.

        The pixel-level delete prediction is shared by all contributors of that
        pixel. Contributor weights decide how much evidence each historical GS
        receives; no historical geometry or appearance residual is written here.
        """
        b, n = self.gaussians.opacities.shape
        pixel_probability = torch.sigmoid(
            delete_logit.float() / max(float(temperature), 1e-4)
        )
        if pixel_probability.shape[1] != 1:
            raise RuntimeError(
                "Old-GS decay expects one pixel-level delete logit channel, "
                f"got {tuple(pixel_probability.shape)}."
            )

        if gir.contributor_ids is not None and gir.contributor_weights is not None:
            contributor_ids = gir.contributor_ids
            contributor_weights = gir.contributor_weights
        else:
            contributor_ids = gir.indices.unsqueeze(1)
            contributor_weights = gir.dominant_weight.unsqueeze(1)

        contributor_count = contributor_ids.shape[1]
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
            pixel_probability_flat = pixel_probability[batch_idx, 0].reshape(1, -1)
            pixel_probability_flat = pixel_probability_flat.expand_as(support)

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

    def update_historical(
        self,
        gir: DominantGIR,
        prediction: GIRPrediction,
        camera_to_world: torch.Tensor,
        update_confidence: torch.Tensor | None = None,
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
            contribution = gir.dominant_weight[batch_idx].reshape(-1).clamp_min(0.0)
            support_weight = visible * contribution
            update_weight = support_weight * gate * damping
            if update_confidence is not None:
                confidence = update_confidence[batch_idx].reshape(-1).to(
                    update_weight.dtype
                )
                update_weight = update_weight * confidence

            depth = gir.depth[batch_idx].reshape(-1).clamp_min(1e-4)
            delta_mean_camera = prediction.delta_mean_camera[batch_idx]
            delta_mean_camera = delta_mean_camera.permute(1, 2, 0).reshape(-1, 3)
            delta_mean_camera = delta_mean_camera.tanh() * depth.unsqueeze(-1)
            rotation_c2w = camera_to_world[batch_idx, :3, :3].to(
                delta_mean_camera.dtype
            )
            delta_mean_world = delta_mean_camera @ rotation_c2w.transpose(0, 1)

            def aggregate(values: torch.Tensor) -> torch.Tensor:
                weighted = values * update_weight.unsqueeze(-1)
                index = safe_indices.unsqueeze(-1).expand(-1, values.shape[-1])
                numerator = torch.zeros(
                    (n, values.shape[-1]),
                    device=values.device,
                    dtype=values.dtype,
                ).scatter_add(0, index, weighted)
                denominator = torch.zeros(
                    n,
                    device=values.device,
                    dtype=values.dtype,
                ).scatter_add(0, safe_indices, support_weight)
                return numerator / denominator.clamp_min(1e-8).unsqueeze(-1)

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
        self.current_prediction = nn.Conv2d(
            hidden_dim + 1,
            current_output_dim,
            kernel_size=1,
        )
        self.delete_prediction = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)
        nn.init.zeros_(self.current_prediction.weight)
        nn.init.zeros_(self.current_prediction.bias)
        nn.init.zeros_(self.delete_prediction.weight)
        nn.init.constant_(self.delete_prediction.bias, -4.0)

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
        historical_depth = (
            gir.raster_depth if self.use_raster_evidence else gir.depth
        ).to(current_feature.dtype)
        relative_depth = (
            (historical_depth - current_depth) / current_depth.clamp_min(1e-4)
        ).clamp(-2.0, 2.0)
        log_depth = current_depth.clamp_min(1e-4).log().clamp(-8.0, 8.0)
        evidence_parts = [
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
        ]
        if self.use_raster_evidence:
            evidence_parts.append(gir.raster_alpha.to(current_feature.dtype))
        evidence = torch.cat(evidence_parts, dim=1)
        encoded = self.encoder(evidence)
        prediction = self.prediction(encoded)
        splits = torch.split(
            prediction,
            [3, 3, 3, 1, self.harmonic_dim, 1, 1],
            dim=1,
        )
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
        return GIRPrediction(*splits, *current_splits, delete_logit)
