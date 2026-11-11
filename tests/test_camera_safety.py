import ast
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.geometry.camera_safety import sanitize_pose_encoding_for_geometry


ROOT = Path(__file__).resolve().parents[1]


def _load_function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


QUAT_TO_MAT = _load_function(
    ROOT / "src/model/encoder/vggt/utils/rotation.py",
    "quat_to_mat",
    {"torch": torch},
)
CAMERA_LOSS = _load_function(
    ROOT / "src/model/model_wrapper.py",
    "_camera_supervision_losses",
    {
        "torch": torch,
        "F": F,
        "math": math,
        "quat_to_mat": QUAT_TO_MAT,
    },
)


def _valid_camera_inputs(view_count: int = 4):
    pose = torch.zeros(1, view_count, 9)
    pose[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    pose[..., 7:9] = math.pi / 2.0
    pose[0, :, 0] = torch.arange(view_count, dtype=torch.float32)
    target_c2w = torch.eye(4).reshape(1, 1, 4, 4).repeat(
        1, view_count, 1, 1
    )
    target_c2w[0, :, 0, 3] = -pose[0, :, 0]
    target_intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(
        1, view_count, 1, 1
    )
    target_intrinsics[..., 0, 0] = 0.5
    target_intrinsics[..., 1, 1] = 0.5
    target_intrinsics[..., 0, 2] = 0.5
    target_intrinsics[..., 1, 2] = 0.5
    return pose, target_c2w, target_intrinsics


def _camera_loss(
    pose,
    target_c2w,
    target_intrinsics,
    *,
    pose_loss_mode="adjacent_pair",
):
    return CAMERA_LOSS(
        pose,
        target_c2w,
        target_intrinsics,
        translation_weight=1.0,
        rotation_weight=1.0,
        focal_weight=1.0,
        pose_loss_mode=pose_loss_mode,
        translation_direction_weight=0.1,
        translation_direction_min_baseline_ratio=0.05,
        translation_direction_baseline_weighting=True,
        global_trajectory_weight=0.1,
    )


def test_geometry_pose_sanitization_is_finite_and_bounded() -> None:
    pose = torch.tensor(
        [[[float("nan"), float("inf"), float("-inf"),
           float("nan"), 0.0, 0.0, 0.0,
           float("nan"), float("inf")]]]
    )

    safe = sanitize_pose_encoding_for_geometry(pose)

    assert bool(torch.isfinite(safe).all())
    torch.testing.assert_close(safe[..., :3], torch.zeros_like(safe[..., :3]))
    torch.testing.assert_close(
        safe[..., 3:7],
        torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4),
    )
    assert bool((safe[..., 7:9] > 0.0).all())
    assert bool((safe[..., 7:9] < math.pi).all())


def test_valid_camera_supervision_is_unchanged() -> None:
    pose, target_c2w, target_intrinsics = _valid_camera_inputs()
    pose.requires_grad_()

    losses = _camera_loss(pose, target_c2w, target_intrinsics)
    losses["total"].backward()

    torch.testing.assert_close(losses["total"], torch.tensor(0.0))
    torch.testing.assert_close(losses["invalid_ratio"], torch.tensor(0.0))
    torch.testing.assert_close(losses["valid_view_count"], torch.tensor(4.0))
    assert pose.grad is not None
    assert bool(torch.isfinite(pose.grad).all())


def test_camera_loss_masks_one_invalid_view() -> None:
    pose, target_c2w, target_intrinsics = _valid_camera_inputs()
    pose[0, 1, 7] = float("nan")
    pose.requires_grad_()

    losses = _camera_loss(pose, target_c2w, target_intrinsics)
    losses["total"].backward()

    torch.testing.assert_close(losses["invalid_ratio"], torch.tensor(0.25))
    torch.testing.assert_close(losses["valid_view_count"], torch.tensor(3.0))
    assert all(bool(torch.isfinite(value)) for value in losses.values())
    assert pose.grad is not None
    assert bool(torch.isfinite(pose.grad).all())


def test_all_invalid_cameras_produce_connected_zero_loss() -> None:
    pose, target_c2w, target_intrinsics = _valid_camera_inputs()
    pose.fill_(float("nan"))
    pose.requires_grad_()

    adjacent_losses = _camera_loss(pose, target_c2w, target_intrinsics)
    first_relative_losses = _camera_loss(
        pose,
        target_c2w,
        target_intrinsics,
        pose_loss_mode="first_relative",
    )

    for losses in (adjacent_losses, first_relative_losses):
        assert losses["total"].requires_grad
        torch.testing.assert_close(losses["total"], torch.tensor(0.0))
        torch.testing.assert_close(losses["invalid_ratio"], torch.tensor(1.0))
        torch.testing.assert_close(
            losses["valid_view_count"], torch.tensor(0.0)
        )
        assert all(bool(torch.isfinite(value)) for value in losses.values())
    (adjacent_losses["total"] + first_relative_losses["total"]).backward()
    assert pose.grad is not None
    assert bool(torch.isfinite(pose.grad).all())
    torch.testing.assert_close(pose.grad, torch.zeros_like(pose.grad))


def test_invalid_gt_intrinsics_are_masked() -> None:
    pose, target_c2w, target_intrinsics = _valid_camera_inputs()
    target_intrinsics[0, 2, 0, 0] = 0.0
    pose.requires_grad_()

    losses = _camera_loss(pose, target_c2w, target_intrinsics)
    losses["total"].backward()

    torch.testing.assert_close(losses["invalid_ratio"], torch.tensor(0.25))
    assert bool(torch.isfinite(losses["total"]))
    assert pose.grad is not None
    assert bool(torch.isfinite(pose.grad).all())


def test_invalid_camera_head_output_keeps_parameter_gradients_connected() -> None:
    camera_head = torch.nn.Linear(5, 9)
    camera_head.weight.data.fill_(float("nan"))
    pose = camera_head(torch.ones(1, 4, 5))
    _, target_c2w, target_intrinsics = _valid_camera_inputs()

    losses = _camera_loss(pose, target_c2w, target_intrinsics)
    losses["total"].backward()

    torch.testing.assert_close(losses["total"], torch.tensor(0.0))
    torch.testing.assert_close(losses["invalid_ratio"], torch.tensor(1.0))
    for parameter in camera_head.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
