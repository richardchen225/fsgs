from __future__ import annotations

import torch


def estimate_camera_sim3(
    source_c2w: torch.Tensor,
    target_c2w: torch.Tensor,
    *,
    eps: float = 1e-8,
    orientation_weight: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate a source-to-target Sim(3) from corresponding camera poses."""
    if source_c2w.shape != target_c2w.shape or source_c2w.ndim != 4:
        raise ValueError(
            "Sim(3) camera tensors must have matching [B,V,4,4] shapes, got "
            f"{tuple(source_c2w.shape)} and {tuple(target_c2w.shape)}."
        )
    if source_c2w.shape[1] < 3:
        raise ValueError("Sim(3) alignment requires at least three source cameras.")

    source_centers = source_c2w[..., :3, 3].double()
    target_centers = target_c2w[..., :3, 3].double()
    source_centered = source_centers - source_centers.mean(dim=1, keepdim=True)
    target_centered = target_centers - target_centers.mean(dim=1, keepdim=True)
    source_radius = source_centered.square().sum(dim=-1).mean(dim=1).sqrt()
    target_radius = target_centered.square().sum(dim=-1).mean(dim=1).sqrt()
    if (source_radius <= eps).any() or (target_radius <= eps).any():
        raise ValueError("Camera centers are degenerate for Sim(3) alignment.")

    source_axes = source_c2w[..., :3, :3].double().transpose(-1, -2)
    target_axes = target_c2w[..., :3, :3].double().transpose(-1, -2)
    source_axis_points = source_centers[:, :, None] + (
        float(orientation_weight) * source_radius[:, None, None, None] * source_axes
    )
    target_axis_points = target_centers[:, :, None] + (
        float(orientation_weight) * target_radius[:, None, None, None] * target_axes
    )
    source = torch.cat(
        [source_centers, source_axis_points.reshape(source_c2w.shape[0], -1, 3)],
        dim=1,
    )
    target = torch.cat(
        [target_centers, target_axis_points.reshape(target_c2w.shape[0], -1, 3)],
        dim=1,
    )
    source_mean = source.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = source_centered.square().sum(dim=(1, 2)) / source.shape[1]
    if (source_variance <= eps).any():
        raise ValueError("Predicted camera poses are degenerate for Sim(3) alignment.")

    covariance = torch.einsum(
        "bvi,bvj->bij", target_centered, source_centered
    ) / source.shape[1]
    left, singular_values, right_h = torch.linalg.svd(covariance)
    correction = torch.ones(
        (source.shape[0], 3), device=source.device, dtype=source.dtype
    )
    correction[:, -1] = torch.where(
        torch.det(left @ right_h) < 0.0, -1.0, 1.0
    )
    rotation = left @ torch.diag_embed(correction) @ right_h
    scale = (singular_values * correction).sum(dim=-1) / source_variance
    if (scale <= eps).any() or not torch.isfinite(scale).all():
        raise ValueError(f"Invalid Sim(3) scale: {scale.tolist()}.")
    translation = target_mean[:, 0] - scale[:, None] * torch.einsum(
        "bij,bj->bi", rotation, source_mean[:, 0]
    )
    dtype = source_c2w.dtype
    return rotation.to(dtype), translation.to(dtype), scale.to(dtype)


def estimate_camera_sim3_with_fallback(
    source_c2w: torch.Tensor,
    target_c2w: torch.Tensor,
    *,
    eps: float = 1e-8,
    orientation_weight: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate Sim(3) per sample and keep degenerate trajectories finite.

    A collapsed camera trajectory has no observable translation scale. For
    those samples, align the first camera orientation and center with unit
    scale. The returned validity mask lets the caller replace the unobservable
    geometry-camera trajectory while retaining the finite scale/rotation used
    by the remaining geometry branches.
    """
    if source_c2w.shape != target_c2w.shape or source_c2w.ndim != 4:
        raise ValueError(
            "Sim(3) camera tensors must have matching [B,V,4,4] shapes, got "
            f"{tuple(source_c2w.shape)} and {tuple(target_c2w.shape)}."
        )
    if source_c2w.shape[1] < 3:
        raise ValueError("Sim(3) alignment requires at least three source cameras.")

    rotations = []
    translations = []
    scales = []
    valid = []
    for batch_index in range(source_c2w.shape[0]):
        source = source_c2w[batch_index : batch_index + 1]
        target = target_c2w[batch_index : batch_index + 1]
        source_centers = source[..., :3, 3].double()
        target_centers = target[..., :3, 3].double()
        source_radius = (
            source_centers - source_centers.mean(dim=1, keepdim=True)
        ).square().sum(dim=-1).mean().sqrt()
        target_radius = (
            target_centers - target_centers.mean(dim=1, keepdim=True)
        ).square().sum(dim=-1).mean().sqrt()
        inputs_finite = bool(
            torch.isfinite(source).all() and torch.isfinite(target).all()
        )
        trajectory_valid = (
            inputs_finite
            and bool(source_radius > eps)
            and bool(target_radius > eps)
        )

        if trajectory_valid:
            try:
                rotation, translation, scale = estimate_camera_sim3(
                    source,
                    target,
                    eps=eps,
                    orientation_weight=orientation_weight,
                )
            except (RuntimeError, ValueError):
                trajectory_valid = False

        if not trajectory_valid:
            dtype = source.dtype
            device = source.device
            source_rotation = source[0, 0, :3, :3]
            target_rotation = target[0, 0, :3, :3]
            rotations_finite = bool(
                torch.isfinite(source_rotation).all()
                and torch.isfinite(target_rotation).all()
            )
            if rotations_finite:
                rotation = (
                    target_rotation.float() @ source_rotation.float().transpose(-1, -2)
                ).to(dtype).unsqueeze(0)
            else:
                rotation = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            scale = torch.ones(1, device=device, dtype=dtype)
            source_center = torch.nan_to_num(
                source[0, 0, :3, 3], nan=0.0, posinf=0.0, neginf=0.0
            )
            target_center = torch.nan_to_num(
                target[0, 0, :3, 3], nan=0.0, posinf=0.0, neginf=0.0
            )
            translation = (
                target_center - torch.einsum("ij,j->i", rotation[0], source_center)
            ).unsqueeze(0)

        rotations.append(rotation)
        translations.append(translation)
        scales.append(scale)
        valid.append(trajectory_valid)

    return (
        torch.cat(rotations, dim=0),
        torch.cat(translations, dim=0),
        torch.cat(scales, dim=0),
        torch.tensor(valid, device=source_c2w.device, dtype=torch.bool),
    )


def apply_camera_sim3(
    camera_to_world: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply a batched source-to-target Sim(3) to camera-to-world poses."""
    aligned = camera_to_world.clone()
    aligned[..., :3, :3] = torch.einsum(
        "bij,bvjk->bvik", rotation, camera_to_world[..., :3, :3]
    )
    aligned[..., :3, 3] = (
        scale[:, None, None]
        * torch.einsum("bij,bvj->bvi", rotation, camera_to_world[..., :3, 3])
        + translation[:, None]
    )
    return aligned


def apply_point_sim3(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply a batched source-to-target Sim(3) to points shaped [B,...,3]."""
    flat = points.reshape(points.shape[0], -1, 3)
    transformed = (
        scale[:, None, None] * torch.einsum("bij,bnj->bni", rotation, flat)
        + translation[:, None]
    )
    return transformed.reshape_as(points)
