import numpy as np

from src.loss.dav3.src.depth_anything_3.utils.pose_align import (
    estimate_scale_from_camera_baselines,
)


def _w2c_from_centers(centers: np.ndarray) -> np.ndarray:
    extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None], len(centers), axis=0)
    extrinsics[:, :3, 3] = -centers
    return extrinsics


def test_camera_baseline_scale_handles_collinear_trajectory() -> None:
    gt_centers = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    predicted_centers = gt_centers * 2.5 + np.array([4.0, -2.0, 7.0])

    scale = estimate_scale_from_camera_baselines(
        _w2c_from_centers(predicted_centers)[:, :3],
        _w2c_from_centers(gt_centers),
    )

    assert scale is not None
    assert np.isclose(scale, 2.5)


def test_camera_baseline_scale_handles_rotated_3d_trajectory() -> None:
    gt_centers = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [-1.0, 1.0, 3.0]]
    )
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    predicted_centers = (gt_centers @ rotation.T) * 0.4

    scale = estimate_scale_from_camera_baselines(
        _w2c_from_centers(predicted_centers),
        _w2c_from_centers(gt_centers),
    )

    assert scale is not None
    assert np.isclose(scale, 0.4)


def test_camera_baseline_scale_rejects_static_trajectory() -> None:
    static = _w2c_from_centers(np.zeros((3, 3), dtype=np.float64))

    assert estimate_scale_from_camera_baselines(static, static) is None
