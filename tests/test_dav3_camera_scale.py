import numpy as np

from src.loss.dav3.src.depth_anything_3.api import (
    _normalize_checkpoint_state_dict,
)
from src.loss.dav3.src.depth_anything_3.utils.pose_align import (
    camera_trajectory_span,
    estimate_scale_from_camera_baselines,
)


def test_checkpoint_key_normalization_loads_legacy_camera_weights() -> None:
    base_token = object()
    trained_token = object()
    normalized = _normalize_checkpoint_state_dict(
        {
            "module.model.backbone.pretrained.camera_token": base_token,
            "module.model.backbone.pretrained.camera_token_extra": trained_token,
            "model.all_heads.camera_head.fc_rot.weight": object(),
            "model.all_heads.camera_head.more_mlps.0.weight": object(),
        }
    )

    assert normalized["backbone.pretrained.camera_token"] is trained_token
    assert "cam_dec.fc_qvec.weight" in normalized
    assert "cam_dec.backbone.0.weight" in normalized


def _w2c_from_centers(
    centers: np.ndarray,
    rotations: np.ndarray | None = None,
) -> np.ndarray:
    extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None], len(centers), axis=0)
    if rotations is not None:
        extrinsics[:, :3, :3] = rotations
    extrinsics[:, :3, 3] = -np.einsum(
        "nij,nj->ni", extrinsics[:, :3, :3], centers
    )
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
    gt_rotations = np.repeat(rotation[None], len(gt_centers), axis=0)
    predicted_rotations = np.repeat(rotation.T[None], len(gt_centers), axis=0)

    scale = estimate_scale_from_camera_baselines(
        _w2c_from_centers(predicted_centers, predicted_rotations),
        _w2c_from_centers(gt_centers, gt_rotations),
    )

    assert scale is not None
    assert np.isclose(scale, 0.4)
    expected_span = np.linalg.norm(
        predicted_centers[:, None] - predicted_centers[None, :], axis=-1
    ).max()
    assert np.isclose(
        camera_trajectory_span(
            _w2c_from_centers(predicted_centers, predicted_rotations)
        ),
        expected_span,
    )


def test_camera_baseline_scale_rejects_static_trajectory() -> None:
    static = _w2c_from_centers(np.zeros((3, 3), dtype=np.float64))

    assert estimate_scale_from_camera_baselines(static, static) is None
    assert camera_trajectory_span(static) == 0.0
