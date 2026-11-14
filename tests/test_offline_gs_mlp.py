from pathlib import Path

import torch
from omegaconf import OmegaConf

import src.experiments.overfit_offline_gs_mlp as offline_experiment
from src.experiments.overfit_offline_gs_mlp import (
    _append_view,
    _build_full_graph_incremental_loss,
    _camera_error_metrics,
    _make_state,
    _requires_sim3_alignment,
    _select_replay_indices,
    _slice_gaussians,
    apply_camera_sim3,
    apply_point_sim3,
    estimate_camera_sim3,
)
from src.experiments.overfit_old_gs_residual import _camera_supervision_losses
from src.model.types import Gaussians


def test_offline_experiment_separates_training_and_evaluation_cameras() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "experiment"
        / "offline_gs_mlp_overfit.yaml"
    )
    offline_cfg = OmegaConf.load(config_path).offline

    assert offline_cfg.training_geometry_camera_source == "predicted"
    assert offline_cfg.training_source_render_camera == "gt"
    assert offline_cfg.training_heldout_loss_camera == "gt"
    assert offline_cfg.training_sim3_alignment_mode == "always"
    assert offline_cfg.evaluation_geometry_camera_source == "predicted"
    assert offline_cfg.evaluation_source_render_camera == "predicted"
    assert offline_cfg.evaluation_heldout_loss_camera == "predicted"
    assert offline_cfg.evaluation_heldout_test_camera == "predicted"
    assert offline_cfg.evaluation_sim3_alignment_mode == "never"
    assert offline_cfg.train_camera_head
    assert offline_cfg.loss_camera > 0.0


def _rotation_z(angle: float) -> torch.Tensor:
    cosine = torch.cos(torch.tensor(angle))
    sine = torch.sin(torch.tensor(angle))
    zero = torch.zeros_like(cosine)
    one = torch.ones_like(cosine)
    return torch.stack(
        [
            torch.stack([cosine, -sine, zero]),
            torch.stack([sine, cosine, zero]),
            torch.stack([zero, zero, one]),
        ]
    )


def test_camera_sim3_recovers_synthetic_gauge() -> None:
    source = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 8, 1, 1)
    source[0, :, :3, 3] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [0.3, 1.2, -0.2],
            [-0.5, 0.2, 0.7],
            [1.1, 0.8, 0.4],
            [0.2, -0.7, 1.0],
            [-0.8, 1.1, 0.3],
            [0.6, 0.4, 1.4],
        ]
    )
    rotation = _rotation_z(0.4).unsqueeze(0)
    translation = torch.tensor([[1.5, -0.8, 0.4]])
    scale = torch.tensor([2.3])
    target = apply_camera_sim3(source, rotation, translation, scale)

    estimated_rotation, estimated_translation, estimated_scale = (
        estimate_camera_sim3(source, target)
    )
    recovered = apply_camera_sim3(
        source, estimated_rotation, estimated_translation, estimated_scale
    )

    torch.testing.assert_close(estimated_rotation, rotation, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(estimated_translation, translation, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(estimated_scale, scale, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(recovered, target, atol=1e-5, rtol=1e-5)


def test_point_sim3_matches_camera_center_transform() -> None:
    points = torch.tensor([[[0.0, 1.0, 2.0], [2.0, -1.0, 0.5]]])
    rotation = _rotation_z(-0.2).unsqueeze(0)
    translation = torch.tensor([[-0.3, 0.9, 1.2]])
    scale = torch.tensor([0.7])
    transformed = apply_point_sim3(points, rotation, translation, scale)
    expected = scale[:, None, None] * torch.einsum(
        "bij,bnj->bni", rotation, points
    ) + translation[:, None]
    torch.testing.assert_close(transformed, expected)


def test_camera_error_metrics_report_pose_and_focal_errors() -> None:
    gt_c2w = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    predicted_c2w = gt_c2w.clone()
    predicted_c2w[0, 1, 0, 3] = 2.0
    gt_intrinsic = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    predicted_intrinsic = gt_intrinsic.clone()
    predicted_intrinsic[..., 0, 0] = 1.5
    predicted_intrinsic[..., 1, 1] = 0.5

    metrics = _camera_error_metrics(
        predicted_c2w,
        predicted_intrinsic,
        gt_c2w,
        gt_intrinsic,
        "test",
    )

    torch.testing.assert_close(
        torch.tensor(metrics["test_center_rmse"]), torch.sqrt(torch.tensor(2.0))
    )
    assert metrics["test_rotation_deg_mean"] == 0.0
    assert metrics["test_focal_abs_rel"] == 0.5


def test_sim3_alignment_is_skipped_only_for_all_predicted_renders() -> None:
    assert not _requires_sim3_alignment("predicted", "predicted", False)
    assert _requires_sim3_alignment("gt", "predicted", False)
    assert _requires_sim3_alignment("predicted", "gt", False)
    assert _requires_sim3_alignment("predicted", "predicted", True)
    assert not _requires_sim3_alignment(
        "predicted",
        "predicted",
        True,
        heldout_loss_camera="predicted",
    )
    assert _requires_sim3_alignment(
        "predicted",
        "predicted",
        False,
        geometry_camera_source="gt",
    )


def test_sim3_alignment_mode_can_override_predicted_camera_auto_mode() -> None:
    assert _requires_sim3_alignment(
        "predicted", "predicted", False, "always"
    )
    assert not _requires_sim3_alignment(
        "predicted", "predicted", False, "never"
    )
    assert not _requires_sim3_alignment(
        "gt",
        "predicted",
        True,
        "never",
        geometry_camera_source="gt",
    )


def test_adjacent_camera_loss_uses_consecutive_pose_pairs() -> None:
    class Config(dict):
        __getattr__ = dict.__getitem__

    predicted = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    predicted[0, :, 0, 3] = torch.tensor([0.0, 1.0, 3.0])
    predicted.requires_grad_(True)
    gt = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    gt[0, :, 0, 3] = torch.tensor([0.0, 2.0, 4.0])
    camera_prediction = {
        "relative_c2w": predicted,
        "gt_relative_c2w": gt,
        "pose_encoding": torch.zeros(1, 3, 9),
    }
    intrinsic_px = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1)
    common = {
        "camera_translation_weight": 1.0,
        "camera_translation_direction_weight": 0.1,
        "camera_rotation_weight": 0.0,
        "camera_focal_weight": 0.0,
    }
    adjacent = _camera_supervision_losses(
        camera_prediction,
        intrinsic_px,
        (8, 8),
        Config(common | {"camera_pose_loss_mode": "adjacent_pair"}),
    )
    first_relative = _camera_supervision_losses(
        camera_prediction,
        intrinsic_px,
        (8, 8),
        Config(common | {"camera_pose_loss_mode": "first_relative"}),
    )

    assert adjacent["translation"] < first_relative["translation"]
    adjacent["total"].backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()


def test_adjacent_camera_translation_loss_detaches_global_scale() -> None:
    class Config(dict):
        __getattr__ = dict.__getitem__

    predicted = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    predicted[0, :, 0, 3] = torch.tensor([0.0, 1.0, 3.0])
    predicted.requires_grad_(True)
    gt = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    gt[0, :, 0, 3] = torch.tensor([0.0, 2.0, 6.0])
    losses = _camera_supervision_losses(
        {
            "relative_c2w": predicted,
            "gt_relative_c2w": gt,
            "pose_encoding": torch.zeros(1, 3, 9),
        },
        torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1),
        (8, 8),
        Config(
            {
                "camera_pose_loss_mode": "adjacent_pair",
                "camera_translation_weight": 1.0,
                "camera_translation_direction_weight": 0.1,
                "camera_rotation_weight": 0.0,
                "camera_focal_weight": 0.0,
            }
        ),
    )

    torch.testing.assert_close(
        losses["translation_alignment_scale"], torch.tensor(2.0)
    )
    torch.testing.assert_close(
        losses["translation_shape"], torch.tensor(0.0), atol=1e-7, rtol=0.0
    )
    torch.testing.assert_close(
        losses["translation_direction"],
        torch.tensor(0.0),
        atol=1e-7,
        rtol=0.0,
    )


def test_slice_gaussians_preserves_per_view_anchor_order() -> None:
    values = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    gaussians = Gaussians(
        means=values,
        harmonics=torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3),
        opacities=torch.arange(4, dtype=torch.float32).reshape(1, 4),
        scales=values + 1.0,
        rotations=torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
    )

    sliced = _slice_gaussians(gaussians, 1, 3)

    torch.testing.assert_close(sliced.means, gaussians.means[:, 1:3])
    torch.testing.assert_close(sliced.harmonics, gaussians.harmonics[:, 1:3])
    torch.testing.assert_close(sliced.opacities, gaussians.opacities[:, 1:3])
    torch.testing.assert_close(sliced.scales, gaussians.scales[:, 1:3])
    torch.testing.assert_close(sliced.rotations, gaussians.rotations[:, 1:3])


def test_select_replay_indices_is_causal() -> None:
    assert _select_replay_indices(0, -1) == []
    assert _select_replay_indices(4, -1) == [0, 1, 2, 3]
    assert _select_replay_indices(4, 2) == [2, 3]
    assert _select_replay_indices(4, 0) == []


def test_append_view_all_adds_every_valid_depth_anchor() -> None:
    initial = Gaussians(
        means=torch.zeros(1, 1, 3),
        harmonics=torch.zeros(1, 1, 1, 3),
        opacities=torch.ones(1, 1),
        scales=torch.ones(1, 1, 3),
        rotations=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
    )
    current = Gaussians(
        means=torch.arange(12, dtype=torch.float32).reshape(1, 4, 3),
        harmonics=torch.zeros(1, 4, 1, 3),
        opacities=torch.ones(1, 4),
        scales=torch.ones(1, 4, 3),
        rotations=torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]] * 4], dtype=torch.float32
        ),
    )

    updated, stats = _append_view(
        None,
        {"append_mode": "all"},
        _make_state(initial),
        current,
        torch.tensor([[[1.0, 2.0], [float("nan"), -1.0]]]),
        torch.eye(4).reshape(1, 1, 4, 4),
        torch.eye(3).reshape(1, 1, 3, 3),
        (2, 2),
    )

    assert updated.num_gaussians == 3
    assert stats["appended_count"] == 2.0
    assert stats["append_ratio"] == 0.5
    torch.testing.assert_close(updated.gaussians.means[:, 1:], current.means[:, :2])


def test_full_graph_incremental_loss_preserves_cross_view_gradient(
    monkeypatch,
) -> None:
    class Config(dict):
        __getattr__ = dict.__getitem__

    class ScalarHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def anchor() -> Gaussians:
        return Gaussians(
            means=torch.zeros(1, 1, 3),
            harmonics=torch.zeros(1, 1, 1, 3),
            opacities=torch.ones(1, 1),
            scales=torch.ones(1, 1, 3),
            rotations=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
        )

    def fake_update(model, head, cfg, state, render_data, view_idx):
        gaussians = state.gaussians
        updated = type(state)(
            gaussians=Gaussians(
                means=gaussians.means + head.weight,
                harmonics=gaussians.harmonics,
                opacities=gaussians.opacities,
                scales=gaussians.scales,
                rotations=gaussians.rotations,
            ),
            stable_ids=state.stable_ids,
            observation_count=state.observation_count,
            point_map=state.point_map,
            front_conflict_count=state.front_conflict_count,
            split_generation=state.split_generation,
        )
        return updated, {"regularization": head.weight * 0.0}

    def fake_losses(model, cfg, state, render_data, view_idx):
        current = state.gaussians.means[..., 0].sum()
        return current, current.new_zeros(())

    monkeypatch.setattr(
        offline_experiment, "_apply_incremental_residual", fake_update
    )
    monkeypatch.setattr(
        offline_experiment, "_incremental_rgb_losses", fake_losses
    )
    cfg = Config(
        append_mode="all",
        incremental_replay_views=0,
        incremental_current_loss_weight=1.0,
        incremental_replay_loss_weight=0.0,
        residual_regularization_weight=0.0,
    )
    head = ScalarHead()
    geometry = {
        "depth": torch.ones(1, 2, 1, 1),
        "c2w": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
        "intrinsic": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
    }
    render_data = {"image_shape": (1, 1)}

    first_anchor = anchor()
    first_anchor.means.requires_grad_(True)
    _, loss, *_ = _build_full_graph_incremental_loss(
        None, head, cfg, [first_anchor, anchor()], geometry, render_data
    )
    loss.backward()

    # View 2 sees both its own update and the non-detached update from view 1.
    torch.testing.assert_close(head.weight.grad, torch.tensor(2.0))
    assert first_anchor.means.grad is not None
    assert first_anchor.means.grad.abs().sum() > 0
