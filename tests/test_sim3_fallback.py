import torch

from src.geometry.sim3 import (
    apply_camera_sim3,
    estimate_camera_sim3_with_fallback,
)


def _camera_path() -> torch.Tensor:
    cameras = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 4, 1, 1)
    cameras[0, :, :3, 3] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return cameras


def test_valid_trajectory_keeps_exact_sim3_estimate() -> None:
    source = _camera_path()
    rotation = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    translation = torch.tensor([[2.0, -1.0, 0.5]])
    scale = torch.tensor([3.0])
    target = apply_camera_sim3(source, rotation, translation, scale)

    estimated = estimate_camera_sim3_with_fallback(source, target)
    estimated_rotation, estimated_translation, estimated_scale, valid = estimated

    assert bool(valid.all())
    torch.testing.assert_close(estimated_rotation, rotation)
    torch.testing.assert_close(estimated_translation, translation)
    torch.testing.assert_close(estimated_scale, scale)


def test_collapsed_source_uses_finite_first_camera_fallback() -> None:
    source = _camera_path()
    source[..., :3, 3] = 0.0
    target = _camera_path()
    target[:, 0, :3, 3] = torch.tensor([2.0, -1.0, 0.5])

    rotation, translation, scale, valid = estimate_camera_sim3_with_fallback(
        source,
        target,
    )
    aligned = apply_camera_sim3(source, rotation, translation, scale)

    assert not bool(valid.any())
    assert bool(torch.isfinite(rotation).all())
    assert bool(torch.isfinite(translation).all())
    assert bool(torch.isfinite(scale).all())
    torch.testing.assert_close(scale, torch.ones_like(scale))
    torch.testing.assert_close(
        aligned[:, 0, :3, 3],
        target[:, 0, :3, 3],
    )


def test_mixed_batch_only_falls_back_for_degenerate_sample() -> None:
    valid_source = _camera_path()
    collapsed_source = valid_source.clone()
    collapsed_source[..., :3, 3] = 0.0
    source = torch.cat([valid_source, collapsed_source], dim=0)
    target = torch.cat([_camera_path(), _camera_path()], dim=0)

    rotation, translation, scale, valid = estimate_camera_sim3_with_fallback(
        source,
        target,
    )

    torch.testing.assert_close(valid, torch.tensor([True, False]))
    assert bool(torch.isfinite(rotation).all())
    assert bool(torch.isfinite(translation).all())
    assert bool(torch.isfinite(scale).all())


def test_nonfinite_source_returns_finite_fallback_parameters() -> None:
    source = _camera_path()
    source[0, 0, 0, 3] = float("nan")
    target = _camera_path()

    rotation, translation, scale, valid = estimate_camera_sim3_with_fallback(
        source,
        target,
    )

    assert not bool(valid.any())
    assert bool(torch.isfinite(rotation).all())
    assert bool(torch.isfinite(translation).all())
    assert bool(torch.isfinite(scale).all())
