import math

import torch

from src.model.types import Gaussians


SH_C0 = 0.28209479177387814


def build_analytic_gaussian_anchors(
    point_map: torch.Tensor,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    intrinsic_px: torch.Tensor,
    *,
    base_opacity: float = 0.5,
    scale_multiplier: float = 1.0,
    max_scale: float = 0.1,
) -> Gaussians:
    """Build one camera-independent isotropic Gaussian anchor per pixel."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"RGB must have shape [B,3,H,W], got {tuple(rgb.shape)}.")
    batch, _, height, width = rgb.shape
    if depth.shape == (batch, 1, height, width):
        depth_hw = depth[:, 0]
    elif depth.shape == (batch, height, width):
        depth_hw = depth
    else:
        raise ValueError(
            "Depth must have shape [B,H,W] or [B,1,H,W], got "
            f"{tuple(depth.shape)}."
        )
    if point_map.shape == (batch, height, width, 3):
        means = point_map.reshape(batch, height * width, 3)
    elif point_map.shape == (batch, height * width, 3):
        means = point_map
    else:
        raise ValueError(
            "Point map must have shape [B,H,W,3] or [B,H*W,3], got "
            f"{tuple(point_map.shape)}."
        )
    if intrinsic_px.shape != (batch, 3, 3):
        raise ValueError(
            "Pixel intrinsics must have shape [B,3,3], got "
            f"{tuple(intrinsic_px.shape)}."
        )
    if not 0.0 < float(base_opacity) < 1.0:
        raise ValueError("Analytic anchor base_opacity must be between 0 and 1.")
    if not math.isfinite(float(scale_multiplier)) or scale_multiplier <= 0.0:
        raise ValueError("Analytic anchor scale_multiplier must be finite and positive.")
    if not math.isfinite(float(max_scale)) or max_scale <= 0.0:
        raise ValueError("Analytic anchor max_scale must be finite and positive.")

    geometry_depth = torch.nan_to_num(
        depth_hw.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    valid = geometry_depth > 0.0
    fx = intrinsic_px[:, 0, 0].float().abs().clamp_min(1e-6)
    fy = intrinsic_px[:, 1, 1].float().abs().clamp_min(1e-6)
    focal = (fx * fy).sqrt()
    footprint = (
        geometry_depth / focal[:, None, None] * float(scale_multiplier)
    ).clamp(1e-6, float(max_scale))
    scales = footprint.reshape(batch, height * width, 1).expand(-1, -1, 3)

    rotations = torch.zeros(
        (batch, height * width, 4), device=rgb.device, dtype=torch.float32
    )
    rotations[..., 3] = 1.0
    opacities = torch.full(
        (batch, height * width),
        float(base_opacity),
        device=rgb.device,
        dtype=torch.float32,
    )
    opacities = opacities * valid.reshape(batch, height * width)
    rgb_flat = rgb.float().flatten(2).transpose(1, 2)
    harmonics = ((rgb_flat - 0.5) / SH_C0).unsqueeze(-2)

    return Gaussians(
        means=means.float(),
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )
