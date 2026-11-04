from types import SimpleNamespace

import torch

from src.experiments.overfit_offline_gs_mlp import (
    _direct_camera_supervision_losses as offline_camera_loss,
    _predict_corrected_state,
)
from src.loss.loss_lpips import LossLpips, LossLpipsCfg
from src.model.decoder.decoder import DecoderOutput
from src.model.model.anysplat import final_map_auxiliary_enabled
from src.model.model_wrapper import _camera_supervision_losses as main_camera_loss
from src.model.streaming_gir import (
    StreamingGaussianState,
    SymmetricGSResidualHead,
    apply_symmetric_multiview_residual,
    build_symmetric_multiview_gradient_evidence,
)
from src.model.types import Gaussians


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _state() -> StreamingGaussianState:
    rotations = torch.zeros(1, 2, 4)
    rotations[..., 3] = 1.0
    means = torch.tensor([[[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]]])
    return StreamingGaussianState(
        gaussians=Gaussians(
            means=means,
            harmonics=torch.zeros(1, 2, 1, 3),
            opacities=torch.full((1, 2), 0.5),
            scales=torch.full((1, 2, 3), 0.02),
            rotations=rotations,
        ),
        stable_ids=torch.tensor([[0, 1]]),
        observation_count=torch.ones(1, 2),
        point_map=means.clone(),
    )


def _evidence():
    return build_symmetric_multiview_gradient_evidence(
        torch.tensor([[[1.0, -0.5], [0.2, 0.4]]]),
        torch.ones(1, 2, 2, 3) * 0.1,
        torch.ones(1, 2, 2, 1, 3) * 0.2,
        active_mask=torch.ones(1, 2, dtype=torch.bool),
    )


def test_offline_and_main_apply_identical_symmetric_mlp_update() -> None:
    torch.manual_seed(4)
    state = _state()
    evidence = _evidence()
    head = SymmetricGSResidualHead(harmonic_dim=3, hidden_dim=16, chunk_size=1)
    for decoder in head.attribute_decoders.values():
        torch.nn.init.normal_(decoder[-1].weight, std=0.01)
        torch.nn.init.normal_(decoder[-1].bias, std=0.01)
    cfg = AttrDict(
        max_mean_step_scale=1.0,
        max_rotation_step=0.25,
        max_log_scale_step=0.5,
        max_opacity_logit_step=4.0,
        max_harmonics_step=0.5,
        unbounded_residual_updates=True,
    )
    reference = torch.eye(4).unsqueeze(0)
    offline_state, _ = _predict_corrected_state(
        head, cfg, state, evidence, reference
    )
    prediction = head(
        state.gaussians,
        evidence.visible_view_count,
        state.split_generation,
        evidence,
        torch.zeros_like(state.stable_ids, dtype=torch.bool),
    )
    main_state, _ = apply_symmetric_multiview_residual(
        state,
        evidence,
        prediction,
        reference,
        max_mean_step_scale=cfg.max_mean_step_scale,
        max_rotation_step=cfg.max_rotation_step,
        max_log_scale_step=cfg.max_log_scale_step,
        max_opacity_logit_step=cfg.max_opacity_logit_step,
        max_harmonics_step=cfg.max_harmonics_step,
        unbounded_updates=True,
    )
    for field in ("means", "scales", "rotations", "opacities", "harmonics"):
        torch.testing.assert_close(
            getattr(offline_state.gaussians, field),
            getattr(main_state.gaussians, field),
        )


def test_incremental_final_map_never_adds_intermediate_rgb_objective() -> None:
    cfg = SimpleNamespace(
        gir_training_mode="incremental_final_map",
        gir_aux_loss_weight=1.0,
    )
    assert not final_map_auxiliary_enabled(cfg, training=True)
    cfg.gir_training_mode = "streaming_aux"
    assert final_map_auxiliary_enabled(cfg, training=True)
    assert not final_map_auxiliary_enabled(cfg, training=False)


def test_source_lpips_excludes_heldout_views_and_backpropagates() -> None:
    class SquaredErrorLpips(torch.nn.Module):
        def forward(self, prediction, target, normalize=True):
            return (prediction - target).square().mean(dim=(1, 2, 3), keepdim=True)

    loss = object.__new__(LossLpips)
    torch.nn.Module.__init__(loss)
    loss.cfg = LossLpipsCfg(
        weight=0.05,
        apply_after_step=0,
        view_group="source",
        chunk_size=1,
        gradient_checkpointing=True,
    )
    loss.lpips = SquaredErrorLpips()
    color = torch.ones(1, 4, 3, 2, 2, requires_grad=True)
    output = DecoderOutput(color=color, depth=None, alpha=None, lod_rendering=None)
    batch = {"context": {"image": torch.full_like(color, -1.0)}}
    value = loss.forward(
        output,
        batch,
        None,
        {"depth": torch.ones(1, 2, 2, 2, 1)},
        0,
    )
    value.backward()
    assert color.grad[:, :2].abs().sum() > 0
    assert color.grad[:, 2:].abs().sum() == 0


def test_final_map_gradient_reaches_every_anchor_and_shared_residual() -> None:
    torch.manual_seed(9)
    head = SymmetricGSResidualHead(harmonic_dim=3, hidden_dim=16, chunk_size=8)
    depth_head = torch.nn.Linear(1, 1)
    gs_head = torch.nn.Linear(1, 7)
    anchors = []
    residual_predictions = []
    for view_idx in range(3):
        view_feature = torch.tensor([[[1.0 + 0.25 * view_idx]]])
        depth = depth_head(view_feature) + 2.0
        raw_gs = gs_head(view_feature)
        means = torch.cat([raw_gs[..., :2], depth], dim=-1)
        means.retain_grad()
        rotations = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]]
        )
        anchors.append(
            Gaussians(
                means=means,
                harmonics=raw_gs[..., 2:5].unsqueeze(-2),
                opacities=raw_gs[..., 5].sigmoid(),
                scales=torch.nn.functional.softplus(raw_gs[..., 6:7]).expand(
                    -1, -1, 3
                ),
                rotations=rotations,
            )
        )

    state = StreamingGaussianState.from_current(
        anchors[0],
        torch.ones_like(anchors[0].opacities),
        point_map=anchors[0].means,
    )
    for view_idx, anchor in enumerate(anchors):
        if view_idx > 0:
            state = state.append(
                anchor,
                torch.ones_like(anchor.opacities),
                point_map=anchor.means,
            )
        count = state.num_gaussians
        evidence = build_symmetric_multiview_gradient_evidence(
            torch.ones(1, view_idx + 1, count),
            torch.ones(1, view_idx + 1, count, 3) * 0.1,
            torch.ones(1, view_idx + 1, count, 1, 3) * 0.1,
            active_mask=torch.ones(1, count, dtype=torch.bool),
        )
        prediction = head(
            state.gaussians,
            evidence.visible_view_count,
            state.split_generation,
            evidence,
            torch.zeros_like(state.stable_ids, dtype=torch.bool),
        )
        prediction.delta_mean_reference.retain_grad()
        residual_predictions.append(prediction)
        state, _ = apply_symmetric_multiview_residual(
            state,
            evidence,
            prediction,
            torch.eye(4).unsqueeze(0),
            unbounded_updates=True,
        )

    final_loss = sum(
        value.square().mean()
        for value in (
            state.gaussians.means,
            state.gaussians.harmonics,
            state.gaussians.opacities,
            state.gaussians.scales,
        )
    )
    final_loss.backward()
    assert all(anchor.means.grad is not None for anchor in anchors)
    assert all(anchor.means.grad.abs().sum() > 0 for anchor in anchors)
    assert all(
        prediction.delta_mean_reference.grad is not None
        and prediction.delta_mean_reference.grad.abs().sum() > 0
        for prediction in residual_predictions
    )
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in depth_head.parameters()
    )
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in gs_head.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in head.parameters()
    )


def test_offline_uses_main_camera_objective() -> None:
    assert offline_camera_loss is main_camera_loss


def test_camera_direction_loss_masks_tiny_baseline_pair() -> None:
    pose = torch.zeros(1, 3, 9)
    pose[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    # Identity rotations mean c2w translation is the negative w2c encoding.
    pose[0, :, 0] = torch.tensor([0.0, -0.001, -1.001])
    pose.requires_grad_(True)
    target = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    target[0, :, 0, 3] = torch.tensor([0.0, 0.001, 1.001])
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1)
    intrinsics[..., 0, 0] = 0.8
    intrinsics[..., 1, 1] = 0.7

    losses = main_camera_loss(
        pose,
        target,
        intrinsics,
        translation_weight=1.0,
        rotation_weight=0.0,
        focal_weight=0.0,
        pose_loss_mode="adjacent_pair",
        translation_direction_weight=0.1,
        translation_direction_min_baseline_ratio=0.05,
        translation_direction_baseline_weighting=True,
    )

    torch.testing.assert_close(
        losses["translation_direction_valid_ratio"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        losses["translation_direction"], torch.tensor(0.0), atol=1e-6, rtol=0.0
    )
    losses["total"].backward()
    assert pose.grad is not None
    assert torch.isfinite(pose.grad).all()


def test_camera_global_trajectory_penalizes_accumulated_rotation_drift() -> None:
    view_count = 6
    pose = torch.zeros(1, view_count, 9)
    angles = torch.arange(view_count, dtype=torch.float32) * 0.01
    pose[0, :, 5] = torch.sin(0.5 * angles)
    pose[0, :, 6] = torch.cos(0.5 * angles)
    pose.requires_grad_(True)
    target = torch.eye(4).reshape(1, 1, 4, 4).repeat(
        1, view_count, 1, 1
    )
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(
        1, view_count, 1, 1
    )
    intrinsics[..., 0, 0] = 0.8
    intrinsics[..., 1, 1] = 0.7

    losses = main_camera_loss(
        pose,
        target,
        intrinsics,
        translation_weight=1.0,
        rotation_weight=1.0,
        focal_weight=0.0,
        pose_loss_mode="adjacent_pair",
        translation_direction_weight=0.0,
        global_trajectory_weight=0.1,
    )

    assert losses["global_rotation"] > 5.0 * losses["rotation"]
    torch.testing.assert_close(
        losses["total"],
        losses["rotation"] + 0.1 * losses["global_trajectory"],
    )
    losses["total"].backward()
    assert pose.grad is not None
    assert torch.isfinite(pose.grad).all()
