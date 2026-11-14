import math

import torch


def screen_radius_guarded_scales(
    scales: torch.Tensor,
    means: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics_px: torch.Tensor,
    max_screen_radius: float,
    sigma: float = 3.0,
    near_plane: float | torch.Tensor = 1e-10,
    chunk_size: int = 131072,
) -> torch.Tensor:
    """Shrink GS scales to a conservative screen-radius bound per camera.

    The bound uses the spectral norm of the perspective-projection Jacobian
    and the largest 3D standard deviation, so it is valid for any GS rotation.
    The scale factor is detached: this is a rasterization safety guard, not an
    additional camera or geometry supervision path.
    """
    if scales.ndim != 2 or scales.shape[-1] != 3:
        raise ValueError(
            f"Scales must have shape [N,3], got {tuple(scales.shape)}."
        )
    if means.shape != scales.shape:
        raise ValueError(
            "Means must have the same [N,3] shape as scales, got "
            f"means={tuple(means.shape)}, scales={tuple(scales.shape)}."
        )
    if world_to_camera.ndim != 3 or world_to_camera.shape[-2:] != (4, 4):
        raise ValueError(
            "World-to-camera poses must have shape [V,4,4], got "
            f"{tuple(world_to_camera.shape)}."
        )
    view_count = world_to_camera.shape[0]
    if view_count == 0:
        raise ValueError("At least one render camera is required.")
    if intrinsics_px.shape != (view_count, 3, 3):
        raise ValueError(
            "Pixel intrinsics must have shape [V,3,3], got "
            f"{tuple(intrinsics_px.shape)}."
        )

    max_radius = float(max_screen_radius)
    sigma_value = float(sigma)
    chunk_size = int(chunk_size)
    if not math.isfinite(max_radius) or max_radius <= 0.0:
        raise ValueError(
            "Maximum Gaussian screen radius must be finite and positive, "
            f"got {max_radius}."
        )
    if not math.isfinite(sigma_value) or sigma_value <= 0.0:
        raise ValueError(
            f"Gaussian screen-radius sigma must be finite and positive, got {sigma_value}."
        )
    if chunk_size <= 0:
        raise ValueError(
            f"Gaussian screen guard chunk size must be positive, got {chunk_size}."
        )

    near = torch.as_tensor(
        near_plane,
        device=means.device,
        dtype=torch.float32,
    )
    if near.ndim == 0:
        near = near.expand(view_count)
    elif near.shape != (view_count,):
        raise ValueError(
            "Near plane must be scalar or have shape [V], got "
            f"{tuple(near.shape)}."
        )
    guarded_chunks = []
    with torch.no_grad():
        w2c = world_to_camera.detach().float()
        intrinsics = intrinsics_px.detach().float()
        rotation = w2c[:, :3, :3]
        translation = w2c[:, :3, 3]
        fx = intrinsics[:, 0, 0].abs()
        fy = intrinsics[:, 1, 1].abs()
        camera_valid = (
            torch.isfinite(w2c).all(dim=(-2, -1))
            & torch.isfinite(intrinsics).all(dim=(-2, -1))
            & torch.isfinite(near)
            & (near > 0.0)
            & (fx > 0.0)
            & (fy > 0.0)
        )

        for start in range(0, means.shape[0], chunk_size):
            end = min(start + chunk_size, means.shape[0])
            means_chunk = means[start:end].detach().float()
            scale_bound = scales[start:end].detach().float().abs().amax(dim=-1)
            camera_means = torch.einsum(
                "vij,nj->vni", rotation, means_chunk
            ) + translation[:, None]
            x, y, z = camera_means.unbind(dim=-1)
            valid = (
                camera_valid[:, None]
                & torch.isfinite(camera_means).all(dim=-1)
                & torch.isfinite(scale_bound)[None]
                & (z > near[:, None])
            )
            safe_z = torch.where(valid, z, torch.ones_like(z))
            inv_z = safe_z.reciprocal()
            x_over_z = x * inv_z
            y_over_z = y * inv_z
            fx_over_z = fx[:, None] * inv_z
            fy_over_z = fy[:, None] * inv_z

            jacobian_a = fx_over_z.square() * (1.0 + x_over_z.square())
            jacobian_d = fy_over_z.square() * (1.0 + y_over_z.square())
            jacobian_b = (
                fx_over_z * fy_over_z * x_over_z * y_over_z
            )
            discriminant = (
                (jacobian_a - jacobian_d).square()
                + 4.0 * jacobian_b.square()
            ).clamp_min(0.0)
            max_eigenvalue = 0.5 * (
                jacobian_a + jacobian_d + discriminant.sqrt()
            )
            radius = (
                sigma_value
                * scale_bound[None]
                * max_eigenvalue.clamp_min(0.0).sqrt()
            )
            valid = valid & torch.isfinite(radius) & (radius > 0.0)
            per_view_factor = torch.where(
                valid,
                (max_radius / radius.clamp_min(1e-12)).clamp(max=1.0),
                torch.ones_like(radius),
            )
            factor = per_view_factor.amin(dim=0).to(scales.dtype)
            guarded_chunks.append(factor)

    if not guarded_chunks:
        return scales
    factors = torch.cat(guarded_chunks, dim=0).detach()
    return scales * factors[:, None]


def pixel_footprint_scales(
    raw_scales: torch.Tensor,
    depths: torch.Tensor,
    intrinsics_px: torch.Tensor,
    min_projected_radius: float,
    max_projected_radius: float,
) -> torch.Tensor:
    """Map raw scale logits to depth-aware world-space standard deviations."""
    if raw_scales.ndim != 4 or raw_scales.shape[-1] != 3:
        raise ValueError(
            "Raw scales must have shape [B,V,N,3], got "
            f"{tuple(raw_scales.shape)}."
        )
    batch, views, gaussian_count, _ = raw_scales.shape
    if depths.numel() != batch * views * gaussian_count:
        raise ValueError(
            "Depth count must match raw scales: "
            f"depths={tuple(depths.shape)}, raw_scales={tuple(raw_scales.shape)}."
        )
    if intrinsics_px.shape != (batch, views, 3, 3):
        raise ValueError(
            "Pixel intrinsics must have shape [B,V,3,3], got "
            f"{tuple(intrinsics_px.shape)}."
        )
    min_radius = float(min_projected_radius)
    max_radius = float(max_projected_radius)
    if (
        not math.isfinite(min_radius)
        or not math.isfinite(max_radius)
        or not 0.0 < min_radius < max_radius
    ):
        raise ValueError(
            "Projected Gaussian radius bounds must be finite and satisfy "
            f"0 < min < max, got min={min_radius}, max={max_radius}."
        )

    depth = depths.reshape(batch, views, gaussian_count).float()
    depth = torch.nan_to_num(depth, nan=1e-6, posinf=1e6, neginf=1e-6)
    depth = depth.clamp_min(1e-6)
    fx = intrinsics_px[..., 0, 0].float().abs().clamp_min(1e-6)
    fy = intrinsics_px[..., 1, 1].float().abs().clamp_min(1e-6)
    focal = (fx * fy).sqrt()
    pixel_footprint = depth / focal[..., None]
    projected_radius = min_radius + (
        max_radius - min_radius
    ) * raw_scales.float().sigmoid()
    return (pixel_footprint[..., None] * projected_radius).to(raw_scales.dtype)
