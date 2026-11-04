import torch

from src.model.analytic_gaussians import (
    SH_C0,
    build_analytic_gaussian_anchors,
)


def test_analytic_anchors_use_pixel_footprint_and_rgb_sh() -> None:
    depth = torch.full((1, 2, 3), 2.0)
    rgb = torch.full((1, 3, 2, 3), 0.75)
    point_map = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)
    intrinsic = torch.eye(3).unsqueeze(0)
    intrinsic[:, 0, 0] = 100.0
    intrinsic[:, 1, 1] = 400.0

    anchors = build_analytic_gaussian_anchors(
        point_map,
        rgb,
        depth,
        intrinsic,
        base_opacity=0.4,
        scale_multiplier=2.0,
    )

    assert anchors.means.shape == (1, 6, 3)
    assert anchors.harmonics.shape == (1, 6, 1, 3)
    assert torch.allclose(anchors.scales, torch.full_like(anchors.scales, 0.02))
    assert torch.allclose(anchors.opacities, torch.full_like(anchors.opacities, 0.4))
    assert torch.allclose(
        anchors.harmonics,
        torch.full_like(anchors.harmonics, 0.25 / SH_C0),
    )
    assert torch.all(anchors.rotations[..., :3] == 0)
    assert torch.all(anchors.rotations[..., 3] == 1)


def test_analytic_anchors_mask_invalid_depth_opacity() -> None:
    depth = torch.tensor([[[1.0, 0.0]]])
    rgb = torch.zeros((1, 3, 1, 2))
    point_map = torch.zeros((1, 1, 2, 3))
    intrinsic = torch.eye(3).unsqueeze(0)

    anchors = build_analytic_gaussian_anchors(
        point_map, rgb, depth, intrinsic
    )

    assert anchors.opacities.tolist() == [[0.5, 0.0]]
    assert torch.isfinite(anchors.scales).all()
