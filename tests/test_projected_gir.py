from dataclasses import replace

import torch

from src.model.streaming_gir import (
    ProjectedGSEvidence,
    ProjectedNewGSResidualPrediction,
    ProjectedGSResidualHead,
    ProjectedGSResidualPrediction,
    StreamingGaussianState,
    apply_projected_new_gaussian_residual,
    build_mean_gradient_evidence,
    build_opacity_gradient_evidence,
    build_point_map_append_mask,
    build_projected_gaussian_evidence,
    build_projected_new_gaussian_evidence,
    build_provisional_gaussians,
    gate_front_conflict_decay,
    select_current_and_recent_replay_gradients,
)
from src.model.types import Gaussians


def _make_state() -> StreamingGaussianState:
    batch_size, gaussian_count = 2, 3
    means = torch.tensor(
        [
            [[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [-0.2, 0.0, 2.0]],
            [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [-0.2, 0.0, 1.0]],
        ]
    )
    rotations = torch.zeros(batch_size, gaussian_count, 4)
    rotations[..., 3] = 1.0
    return StreamingGaussianState(
        gaussians=Gaussians(
            means=means,
            # Match the gsplat layout used by the encoder: [B, N, K, RGB].
            harmonics=torch.zeros(batch_size, gaussian_count, 1, 3),
            opacities=torch.full((batch_size, gaussian_count), 0.5),
            scales=torch.full((batch_size, gaussian_count, 3), 0.02),
            rotations=rotations,
        ),
        stable_ids=torch.arange(gaussian_count)
        .expand(batch_size, -1)
        .clone(),
        observation_count=torch.zeros(
            batch_size, gaussian_count, dtype=torch.long
        ),
        point_map=means.detach().clone(),
    )


def test_projected_shared_update_is_batch_safe() -> None:
    batch_size, gaussian_count, height, width, feature_dim = 2, 3, 8, 10, 4
    state = _make_state()
    feature = torch.stack(
        [
            torch.ones(feature_dim, height, width),
            torch.full((feature_dim, height, width), 5.0),
        ]
    ).requires_grad_()
    current_rgb = torch.ones(batch_size, 3, height, width)
    current_depth = torch.full((batch_size, 1, height, width), 2.0)
    old_rgb = torch.zeros_like(current_rgb)
    old_depth = torch.stack(
        [
            torch.full((1, height, width), 2.0),
            torch.full((1, height, width), 1.0),
        ]
    )
    camera_to_world = torch.eye(4).expand(batch_size, -1, -1).clone()
    intrinsics = torch.tensor(
        [
            [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
        ]
    )

    evidence = build_projected_gaussian_evidence(
        state=state,
        current_feature=feature,
        current_rgb=current_rgb,
        current_depth=current_depth,
        current_depth_confidence=torch.ones_like(current_depth),
        old_rgb=old_rgb,
        old_depth=old_depth,
        old_alpha=torch.ones(batch_size, 1, height, width),
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
        alpha_threshold=0.02,
        depth_relative_tolerance=0.08,
        rgb_bad_threshold=0.08,
        min_projected_radius=0.0,
        covariance_extent_sigma=3.0,
        chunk_size=2,
    )

    assert evidence.candidate_mask.all()
    assert evidence.surface_consistent[0].all()
    assert not evidence.surface_consistent[1].any()
    assert not evidence.old_in_front[0].any()
    assert evidence.old_in_front[1].all()
    assert evidence.descriptor.shape == (
        batch_size * gaussian_count,
        feature_dim + 30,
    )
    assert evidence.opacity_gradient_features is None
    for descriptor, flat_index in zip(
        evidence.descriptor, evidence.flat_candidate_indices
    ):
        expected = 1.0 if int(flat_index) < gaussian_count else 5.0
        assert torch.allclose(
            descriptor[:feature_dim], torch.full((feature_dim,), expected)
        )

    head = ProjectedGSResidualHead(
        feature_dim, harmonic_dim=3, hidden_dim=16, chunk_size=2
    )
    prediction = head(evidence)
    updated, stats = state.update_projected_historical(
        evidence,
        prediction,
        camera_to_world,
        opacity_logit_scale=1.0,
        front_decay_strength=0.2,
        decay_schedule=1.0,
    )

    assert torch.equal(updated.gaussians.means, state.gaussians.means)
    assert torch.equal(updated.gaussians.scales, state.gaussians.scales)
    assert torch.equal(updated.gaussians.rotations, state.gaussians.rotations)
    assert torch.equal(updated.gaussians.harmonics, state.gaussians.harmonics)
    assert torch.allclose(
        updated.gaussians.opacities[0], state.gaussians.opacities[0]
    )
    assert torch.all(
        updated.gaussians.opacities[1] < state.gaussians.opacities[1]
    )

    loss = updated.gaussians.opacities.sum() + stats["regularization"]
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in [*head.decoder.parameters(), *head.output.parameters()]
    )


def test_projected_shared_head_stays_in_graph_without_candidates() -> None:
    head = ProjectedGSResidualHead(
        feature_dim=4, harmonic_dim=3, hidden_dim=16, chunk_size=2
    )
    evidence = ProjectedGSEvidence(
        descriptor=torch.empty(0, 34),
        flat_candidate_indices=torch.empty(0, dtype=torch.long),
        candidate_mask=torch.zeros(2, 3, dtype=torch.bool),
        surface_consistent=torch.zeros(2, 3, dtype=torch.bool),
        old_in_front=torch.zeros(2, 3, dtype=torch.bool),
        projected_radius=torch.zeros(2, 3),
        relative_depth=torch.zeros(2, 3),
        camera_depth=torch.zeros(2, 3),
        mean_gradient_features=torch.empty(0, 10),
    )

    prediction = head(evidence)
    loss = (
        prediction.delta_opacity_logit.sum()
        + prediction.front_decay_logit.sum()
        + prediction.update_gate_logit.sum()
    )
    loss.backward()

    assert prediction.delta_opacity_logit.shape == (0,)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in [
            *head.decoder.parameters(),
            *head.mean_gradient_adapter.parameters(),
            *head.new_mean_gradient_adapter.parameters(),
            *head.output.parameters(),
        ]
    )


def test_projected_shared_update_policy_and_bounds() -> None:
    state = _make_state()
    camera_to_world = torch.eye(4).expand(2, -1, -1).clone()
    candidate_mask = torch.tensor(
        [[True, True, False], [False, False, False]]
    )
    surface_consistent = torch.tensor(
        [[True, False, False], [False, False, False]]
    )
    old_in_front = torch.tensor(
        [[False, True, False], [False, False, False]]
    )
    evidence = ProjectedGSEvidence(
        descriptor=torch.zeros(2, 34),
        flat_candidate_indices=torch.tensor([0, 1]),
        candidate_mask=candidate_mask,
        surface_consistent=surface_consistent,
        old_in_front=old_in_front,
        projected_radius=torch.ones(2, 3),
        relative_depth=torch.zeros(2, 3),
        camera_depth=torch.full((2, 3), 2.0),
        opacity_gradient_features=torch.zeros(2, 8),
        opacity_gradient_signal=torch.tensor([-1.0, 1.0]),
        opacity_gradient_current_signal=torch.tensor([-1.0, 1.0]),
        opacity_gradient_replay_signal=torch.zeros(2),
    )
    prediction = ProjectedGSResidualPrediction(
        delta_mean_camera=torch.full((2, 3), 100.0, requires_grad=True),
        delta_rotation=torch.full((2, 3), 100.0, requires_grad=True),
        delta_log_scale=torch.full((2, 3), 100.0, requires_grad=True),
        delta_opacity_logit=torch.full((2,), 100.0, requires_grad=True),
        delta_harmonics=torch.full((2, 3), 100.0, requires_grad=True),
        front_decay_logit=torch.full((2,), 100.0, requires_grad=True),
        update_gate_logit=torch.full((2,), 100.0, requires_grad=True),
    )

    updated, stats = state.update_projected_historical(
        evidence,
        prediction,
        camera_to_world,
        opacity_logit_scale=1.0,
        front_decay_strength=0.2,
        decay_schedule=1.0,
        mean_update_mode="relative_depth",
        raw_scale_residual=False,
        raw_opacity_residual=False,
        raw_rotation_residual=False,
        raw_harmonics_residual=False,
        mean_relative_scale=0.02,
        rotation_scale=0.05,
        harmonics_scale=0.10,
    )

    surface = (0, 0)
    front = (0, 1)
    behind = (0, 2)
    mean_delta = updated.gaussians.means[surface] - state.gaussians.means[surface]
    assert torch.all(mean_delta.abs() <= 0.04 + 1e-6)
    assert not torch.equal(
        updated.gaussians.rotations[surface], state.gaussians.rotations[surface]
    )
    assert torch.all(updated.gaussians.scales[surface] > state.gaussians.scales[surface])
    assert updated.gaussians.opacities[surface] > state.gaussians.opacities[surface]
    harmonic_delta = (
        updated.gaussians.harmonics[surface]
        - state.gaussians.harmonics[surface]
    )
    assert torch.all(harmonic_delta.abs() <= 0.10 + 1e-6)
    assert torch.equal(updated.point_map, state.point_map)

    for parameter in ("means", "rotations", "scales", "harmonics"):
        assert torch.equal(
            getattr(updated.gaussians, parameter)[front],
            getattr(state.gaussians, parameter)[front],
        )
    assert updated.gaussians.opacities[front] < state.gaussians.opacities[front]
    assert stats["gradient_consistent_alignment"] == torch.tensor(1.0)
    assert stats["gradient_front_alignment"] == torch.tensor(1.0)
    assert stats["gradient_effective_alignment"] == torch.tensor(1.0)
    assert stats["gradient_update_alignment"] == torch.tensor(0.5)

    for parameter in ("means", "rotations", "scales", "opacities", "harmonics"):
        assert torch.equal(
            getattr(updated.gaussians, parameter)[behind],
            getattr(state.gaussians, parameter)[behind],
        )

    loss = (
        updated.gaussians.means.sum()
        + updated.gaussians.rotations.sum()
        + updated.gaussians.scales.sum()
        + updated.gaussians.opacities.sum()
        + updated.gaussians.harmonics.sum()
        + stats["regularization"]
    )
    loss.backward()
    for value in prediction.__dict__.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()


def test_projected_absolute_mean_uses_raw_output() -> None:
    state = _make_state()
    evidence = ProjectedGSEvidence(
        descriptor=torch.zeros(1, 34),
        flat_candidate_indices=torch.tensor([0]),
        candidate_mask=torch.tensor(
            [[True, False, False], [False, False, False]]
        ),
        surface_consistent=torch.tensor(
            [[True, False, False], [False, False, False]]
        ),
        old_in_front=torch.zeros(2, 3, dtype=torch.bool),
        projected_radius=torch.ones(2, 3),
        relative_depth=torch.zeros(2, 3),
        camera_depth=torch.full((2, 3), 2.0),
    )
    prediction = ProjectedGSResidualPrediction(
        delta_mean_camera=torch.full((1, 3), 100.0),
        delta_rotation=torch.zeros(1, 3),
        delta_log_scale=torch.zeros(1, 3),
        delta_opacity_logit=torch.zeros(1),
        delta_harmonics=torch.zeros(1, 3),
        front_decay_logit=torch.zeros(1),
        update_gate_logit=torch.full((1,), 100.0),
    )
    updated, _ = state.update_projected_historical(
        evidence,
        prediction,
        torch.eye(4).expand(2, -1, -1).clone(),
        opacity_logit_scale=1.0,
        front_decay_strength=0.2,
        mean_update_mode="absolute",
        mean_relative_scale=0.02,
    )
    delta = updated.gaussians.means[0, 0] - state.gaussians.means[0, 0]
    assert torch.allclose(delta, torch.full((3,), 100.0))


def test_point_map_append_ignores_depth_inconsistent_floaters() -> None:
    state = _make_state()
    depth = torch.full((2, 1, 5, 5), 2.0)
    camera_to_world = torch.eye(4).expand(2, -1, -1).clone()
    intrinsics = torch.tensor(
        [
            [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
        ]
    )

    append_mask, coverage, conflicts, _, _ = build_point_map_append_mask(
        state,
        depth,
        camera_to_world,
        intrinsics,
        depth_relative_tolerance=0.08,
        dilation_radius=0,
        chunk_size=2,
    )

    # Batch zero points lie on the current surface. Batch one points are
    # front floaters at z=1 and must not turn any current pixel into coverage.
    assert coverage[0].any()
    assert not coverage[1].any()
    assert torch.equal(append_mask, ~coverage)
    assert not conflicts.any()


def test_projection_only_append_is_batch_safe_and_ignores_depth_mismatch() -> None:
    state = _make_state()
    # Different covered pixels in each scene; one point is in front of the
    # current surface, the other behind it. Neither matches the current depth.
    state.point_map = torch.tensor([
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [float("nan"), 0.0, 2.0]],
        [[0.4, 0.0, 4.0], [100.0, 0.0, 2.0], [-0.4, 0.0, 2.0]],
    ])
    state.stable_ids[1, 2] = -1
    depth = torch.full((2, 1, 10, 10), 2.0)
    depth[0, 0, 0, 0] = 0.0
    depth[1, 0, 0, 0] = float("nan")
    depth.requires_grad_()
    c2w = torch.eye(4).expand(2, -1, -1).clone().requires_grad_()
    intrinsics = torch.tensor([
        [1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0],
    ]).expand(2, -1, -1).clone()
    valid = torch.isfinite(depth) & (depth > 0)

    for radius in (0, 1):
        expected = torch.zeros_like(depth, dtype=torch.bool)
        expected[0, 0, 5-radius:6+radius, 5-radius:6+radius] = True
        expected[1, 0, 5-radius:6+radius, 6-radius:7+radius] = True
        for chunk_size in (1, 65536):
            append_mask, coverage, conflicts, _, _ = build_point_map_append_mask(
                state, depth, c2w, intrinsics,
                dilation_radius=radius, chunk_size=chunk_size,
                depth_check_enabled=False,
            )
            assert torch.equal(coverage, expected)
            assert torch.equal(append_mask, valid & ~expected)
            assert not conflicts.any()
            assert not coverage.requires_grad and not append_mask.requires_grad

    # The original depth-consistent mode still treats these surfaces as holes.
    append_mask, coverage, conflicts, _, _ = build_point_map_append_mask(
        state, depth, c2w, intrinsics, depth_check_enabled=True,
    )
    assert not coverage.any()
    assert torch.equal(append_mask, valid)
    assert not conflicts.any()


def test_projection_only_coverage_does_not_depend_on_current_depth_values() -> None:
    state = _make_state()
    c2w = torch.eye(4).expand(2, -1, -1).clone()
    intrinsics = torch.eye(3).expand(2, -1, -1).clone()
    intrinsics[:, :2, 2] = 0.5
    depth = torch.full((2, 1, 10, 10), 2.0)
    _, reference, _, _, _ = build_point_map_append_mask(
        state, depth, c2w, intrinsics, depth_check_enabled=False,
    )
    depth.fill_(20.0)
    # Even an invalid current sample must not erase historical coverage or
    # create valid neighboring appends through a change in the dilation mask.
    depth[:, :, 5, 5] = float("nan")
    append_mask, coverage, conflicts, _, _ = build_point_map_append_mask(
        state, depth, c2w, intrinsics,
        depth_relative_tolerance=0.0, depth_check_enabled=False,
    )
    assert torch.equal(coverage, reference)
    assert torch.equal(append_mask, torch.isfinite(depth) & ~reference)
    assert not conflicts.any()


def test_hybrid_append_uses_all_projected_depth_layers() -> None:
    state = _make_state()
    state.point_map = torch.tensor([
        # Batch zero has both a front floater and a matching layer at the same
        # pixel. Batch one has only a front floater at that pixel.
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [100.0, 0.0, 2.0]],
        [[0.0, 0.0, 1.0], [0.4, 0.0, 2.0], [100.0, 0.0, 2.0]],
    ])
    depth = torch.full((2, 1, 5, 5), 2.0)
    camera_to_world = torch.eye(4).expand(2, -1, -1).clone()
    intrinsics = torch.tensor([
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
    ]).expand(2, -1, -1).clone()

    append_mask, coverage, conflicts, counts, _ = build_point_map_append_mask(
        state=state,
        current_depth=depth,
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
        current_depth_confidence=torch.ones_like(depth),
        depth_relative_tolerance=0.08,
        dilation_radius=0,
        hybrid_enabled=True,
        front_conflict_confirm_views=1,
    )

    # The matching deeper layer suppresses duplicate append, while the front
    # floater is still selected for immediate opacity decay.
    assert not append_mask[0, 0, 2, 2]
    assert coverage[0, 0, 2, 2]
    assert conflicts[0, 0]
    assert not conflicts[0, 1:].any()
    assert counts[0, 0] == 1

    # Without a matching layer, the current surface is appended immediately
    # and the old front floater is decayed in the same update.
    assert append_mask[1, 0, 2, 2]
    assert not coverage[1, 0, 2, 2]
    assert conflicts[1, 0]
    assert counts[1, 0] == 1

    disabled_append_mask, _, disabled_conflicts, _, _ = (
        build_point_map_append_mask(
            state=state,
            current_depth=depth,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            current_depth_confidence=torch.ones_like(depth),
            depth_relative_tolerance=0.08,
            dilation_radius=0,
            hybrid_enabled=True,
            append_history_in_front_enabled=False,
            front_conflict_confirm_views=1,
        )
    )
    assert not disabled_append_mask[1, 0, 2, 2]
    assert disabled_conflicts[1, 0]

    decayed = state.decay_masked_opacity(conflicts, 0.30)
    assert torch.allclose(
        decayed.gaussians.opacities[0, 0],
        state.gaussians.opacities[0, 0] * 0.70,
    )
    assert torch.allclose(
        decayed.gaussians.opacities[1, 0],
        state.gaussians.opacities[1, 0] * 0.70,
    )


def test_hybrid_append_keeps_old_layers_behind_current_surface() -> None:
    state = _make_state()
    state.point_map = torch.tensor([
        [[0.0, 0.0, 3.0], [0.4, 0.0, 3.0], [100.0, 0.0, 3.0]],
        [[0.0, 0.0, 4.0], [0.4, 0.0, 4.0], [100.0, 0.0, 4.0]],
    ])
    depth = torch.full((2, 1, 5, 5), 2.0)
    c2w = torch.eye(4).expand(2, -1, -1).clone()
    intrinsics = torch.tensor([
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
    ]).expand(2, -1, -1).clone()

    append_mask, coverage, conflicts, _, _ = build_point_map_append_mask(
        state=state,
        current_depth=depth,
        camera_to_world=c2w,
        intrinsics=intrinsics,
        dilation_radius=0,
        hybrid_enabled=True,
    )

    assert append_mask[:, 0, 2, 2].all()
    assert not coverage[:, 0, 2, 2].any()
    assert not conflicts.any()

    disabled_append_mask, _, disabled_conflicts, _, _ = (
        build_point_map_append_mask(
            state=state,
            current_depth=depth,
            camera_to_world=c2w,
            intrinsics=intrinsics,
            dilation_radius=0,
            hybrid_enabled=True,
            append_current_in_front_enabled=False,
        )
    )
    assert not disabled_append_mask[:, 0, 2, 2].any()
    assert not disabled_conflicts.any()


def test_hybrid_append_aligns_depth_scale_per_batch() -> None:
    state = _make_state()
    depth = torch.stack([
        torch.full((1, 10, 10), 4.0),
        torch.full((1, 10, 10), 0.5),
    ])
    old_depth = torch.stack([
        torch.full((1, 10, 10), 2.0),
        torch.full((1, 10, 10), 1.0),
    ])
    old_alpha = torch.ones_like(old_depth)
    c2w = torch.eye(4).expand(2, -1, -1).clone()
    intrinsics = torch.eye(3).expand(2, -1, -1).clone()
    intrinsics[:, :2, 2] = 0.5

    _, coverage, conflicts, _, scales = build_point_map_append_mask(
        state=state,
        current_depth=depth,
        camera_to_world=c2w,
        intrinsics=intrinsics,
        current_rgb=torch.zeros(2, 3, 10, 10),
        old_rgb=torch.zeros(2, 3, 10, 10),
        old_depth=old_depth,
        old_alpha=old_alpha,
        hybrid_enabled=True,
        dilation_radius=0,
    )

    assert torch.allclose(scales, torch.tensor([0.5, 2.0]))
    assert coverage[0].any() and coverage[1].any()
    assert not conflicts.any()


def test_point_map_append_is_safe_under_mixed_precision() -> None:
    state = _make_state()
    state.point_map[..., 2] = 2.0
    depth = torch.full((2, 1, 5, 5), 2.0)
    camera_to_world = torch.eye(4).expand(2, -1, -1).clone()
    intrinsics = torch.tensor([
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
    ]).expand(2, -1, -1).clone()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        append_mask, coverage, conflicts, _, _ = build_point_map_append_mask(
            state=state,
            current_depth=depth,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            hybrid_enabled=True,
            dilation_radius=0,
        )

    assert coverage[0].any()
    assert not conflicts.any()
    assert append_mask.dtype == torch.bool


def test_point_map_append_budget_is_per_batch() -> None:
    state = _make_state()
    state.point_map = torch.full_like(state.point_map, 1000.0)
    depth = torch.ones(2, 1, 10, 10)
    c2w = torch.eye(4).expand(2, -1, -1).clone()
    intrinsics = torch.eye(3).expand(2, -1, -1).clone()
    intrinsics[:, :2, 2] = 0.5

    append_mask, coverage, _, _, _ = build_point_map_append_mask(
        state=state,
        current_depth=depth,
        camera_to_world=c2w,
        intrinsics=intrinsics,
        depth_check_enabled=False,
        dilation_radius=0,
        max_append_ratio=0.20,
    )

    assert not coverage.any()
    assert torch.equal(
        append_mask.flatten(1).sum(dim=1),
        torch.tensor([20, 20]),
    )


def test_masked_append_is_physical_and_batch_safe() -> None:
    state = _make_state()
    means = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    means.requires_grad_()
    rotations = torch.zeros(2, 4, 4)
    rotations[..., 3] = 1.0
    current = Gaussians(
        means=means,
        harmonics=torch.zeros(2, 4, 1, 3),
        opacities=torch.full((2, 4), 0.5),
        scales=torch.full((2, 4, 3), 0.02),
        rotations=rotations,
    )
    append_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, False]]
    )

    updated = state.append(
        current,
        torch.ones(2, 4),
        append_mask=append_mask,
    )

    assert updated.num_gaussians == state.num_gaussians + 2
    assert torch.equal(updated.gaussians.means[0, -2:], means[0, [0, 2]])
    assert torch.equal(updated.gaussians.means[1, -2], means[1, 1])
    assert updated.gaussians.opacities[1, -1] == 0
    assert updated.stable_ids[1, -1] == -1
    assert updated.observation_count[1, -1] == 0

    updated.gaussians.means.sum().backward()
    assert means.grad is not None
    assert torch.equal(means.grad[0, :, 0], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert torch.equal(means.grad[1, :, 0], torch.tensor([0.0, 1.0, 0.0, 0.0]))


def _make_new_anchor_evidence(
    append_mask: torch.Tensor,
    mean_gradient_evidence=None,
    *,
    mean_requires_grad: bool = False,
    camera_requires_grad: bool = False,
):
    batch_size, height, width, feature_dim = 2, 2, 2, 4
    depth = torch.tensor(
        [[[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]
    )
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    x = (xx - 0.5 * width) * depth / (0.5 * width)
    y = (yy - 0.5 * height) * depth / (0.5 * height)
    means = torch.stack([x, y, depth], dim=-1).reshape(batch_size, -1, 3)
    means.requires_grad_(mean_requires_grad)
    rotations = torch.zeros(batch_size, height * width, 4)
    rotations[..., 3] = 1.0
    current = Gaussians(
        means=means,
        harmonics=torch.zeros(batch_size, height * width, 1, 3),
        opacities=torch.full((batch_size, height * width), 0.5),
        scales=torch.full((batch_size, height * width, 3), 0.02),
        rotations=rotations,
    )
    camera_to_world = torch.eye(4).expand(batch_size, -1, -1).clone()
    camera_to_world.requires_grad_(camera_requires_grad)
    intrinsics = torch.tensor(
        [[[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]]
    ).expand(batch_size, -1, -1).clone()
    current_depth = depth[:, None]
    evidence = build_projected_new_gaussian_evidence(
        current=current,
        append_mask=append_mask,
        current_feature=torch.randn(
            batch_size, feature_dim, height, width, requires_grad=True
        ),
        current_rgb=torch.rand(batch_size, 3, height, width),
        current_depth=current_depth,
        current_depth_confidence=torch.ones_like(current_depth),
        old_rgb=torch.zeros(batch_size, 3, height, width),
        old_depth=torch.zeros(batch_size, 1, height, width),
        old_alpha=torch.zeros(batch_size, 1, height, width),
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
        alpha_threshold=0.1,
        depth_relative_tolerance=0.1,
        covariance_extent_sigma=3.0,
        mean_gradient_evidence=mean_gradient_evidence,
    )
    return current, evidence, camera_to_world


def test_projected_new_head_is_identity_at_initialization_and_batch_safe() -> None:
    append_mask = torch.tensor(
        [
            [[[True, False], [True, False]]],
            [[[False, True], [False, False]]],
        ]
    )
    current, evidence, camera_to_world = _make_new_anchor_evidence(append_mask)
    assert evidence.descriptor.shape == (3, 34)
    assert torch.equal(
        evidence.candidate_mask.sum(dim=1), torch.tensor([2, 1])
    )
    head = ProjectedGSResidualHead(
        feature_dim=4, harmonic_dim=3, hidden_dim=16, chunk_size=2
    )
    prediction = head.forward_new(evidence)
    updated, stats = apply_projected_new_gaussian_residual(
        current, evidence, prediction, camera_to_world
    )
    for name in ("means", "rotations", "scales", "opacities", "harmonics"):
        assert torch.equal(getattr(updated, name), getattr(current, name))
    assert stats["regularization"] == 0
    assert torch.allclose(stats["gate_mean"], torch.sigmoid(torch.tensor(2.0)))


def test_projected_new_evidence_supports_autocast_geometry() -> None:
    append_mask = torch.tensor(
        [
            [[[True, False], [True, False]]],
            [[[False, True], [False, False]]],
        ]
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, evidence, _ = _make_new_anchor_evidence(append_mask)

    assert evidence.camera_depth.dtype == torch.float32
    assert torch.isfinite(evidence.camera_depth).all()
    assert torch.equal(
        evidence.camera_depth[evidence.candidate_mask],
        torch.tensor([2.0, 2.0, 3.0]),
    )


def test_projected_new_evidence_detaches_camera_but_not_gaussians() -> None:
    append_mask = torch.tensor(
        [
            [[[True, False], [True, False]]],
            [[[False, True], [False, False]]],
        ]
    )
    current, evidence, camera_to_world = _make_new_anchor_evidence(
        append_mask,
        mean_requires_grad=True,
        camera_requires_grad=True,
    )

    evidence.descriptor.float().sum().backward()

    assert current.means.grad is not None
    assert torch.isfinite(current.means.grad).all()
    assert camera_to_world.grad is None


def test_projected_new_head_bypasses_absent_opacity_gradient() -> None:
    append_mask = torch.tensor(
        [
            [[[True, False], [True, False]]],
            [[[False, True], [False, False]]],
        ]
    )
    _, evidence, _ = _make_new_anchor_evidence(append_mask)
    assert evidence.opacity_gradient_features is None
    head = ProjectedGSResidualHead(
        feature_dim=4, harmonic_dim=3, hidden_dim=16, chunk_size=2
    )
    with torch.no_grad():
        head.new_output.weight.fill_(0.1)
        head.new_output.bias.zero_()
        head.opacity_gradient_adapter[-1].bias.fill_(1.0)

    bypassed = head.forward_new(evidence)
    conditioned = head.forward_new(
        replace(
            evidence,
            opacity_gradient_features=torch.zeros(
                evidence.descriptor.shape[0], 8
            ),
        )
    )

    assert not torch.allclose(
        bypassed.delta_opacity_logit, conditioned.delta_opacity_logit
    )


def test_projected_new_residual_uses_absolute_old_gs_mean_parameterization() -> None:
    append_mask = torch.tensor(
        [
            [[[True, False], [False, False]]],
            [[[False, True], [False, False]]],
        ]
    )
    current, evidence, camera_to_world = _make_new_anchor_evidence(append_mask)
    count = evidence.flat_candidate_indices.numel()
    prediction = ProjectedNewGSResidualPrediction(
        delta_mean_camera=torch.tensor([[1.0, 0.0, 0.0]]).expand(count, -1),
        delta_rotation=torch.zeros(count, 3),
        delta_log_scale=torch.zeros(count, 3),
        delta_opacity_logit=torch.zeros(count),
        delta_harmonics=torch.zeros(count, 3),
        update_gate_logit=torch.full((count,), 20.0),
    )
    updated, _ = apply_projected_new_gaussian_residual(
        current,
        evidence,
        prediction,
        camera_to_world,
        mean_update_mode="absolute",
    )
    selected = evidence.candidate_mask
    assert torch.equal(updated.means[~selected], current.means[~selected])
    original_selected = current.means[selected]
    updated_selected = updated.means[selected]
    assert torch.allclose(
        updated_selected - original_selected,
        torch.tensor([[1.0, 0.0, 0.0]]).expand(count, -1),
        atol=1e-5,
    )

    state = _make_state()
    appended = state.append(
        updated,
        torch.ones_like(updated.opacities),
        append_mask=evidence.candidate_mask,
        point_map=current.means,
    )
    selected_anchor_means = torch.stack(
        [current.means[0, 0], current.means[1, 1]], dim=0
    )
    assert torch.equal(appended.point_map[:, -1], selected_anchor_means)
    assert not torch.equal(
        appended.gaussians.means[:, -1], appended.point_map[:, -1]
    )


def test_projected_new_residual_uses_old_gs_raw_channel_controls() -> None:
    append_mask = torch.tensor(
        [
            [[[True, False], [False, False]]],
            [[[False, True], [False, False]]],
        ]
    )
    current, evidence, camera_to_world = _make_new_anchor_evidence(append_mask)
    count = evidence.flat_candidate_indices.numel()
    raw_rotation = torch.tensor([[0.0, 0.0, 0.2]]).expand(count, -1)
    raw_scale = torch.tensor([[0.5, -0.5, 0.25]]).expand(count, -1)
    raw_harmonics = torch.full((count, 3), 0.3)
    prediction = ProjectedNewGSResidualPrediction(
        delta_mean_camera=torch.zeros(count, 3),
        delta_rotation=raw_rotation,
        delta_log_scale=raw_scale,
        delta_opacity_logit=torch.full((count,), 2.0),
        delta_harmonics=raw_harmonics,
        update_gate_logit=torch.full((count,), 20.0),
    )
    updated, stats = apply_projected_new_gaussian_residual(
        current,
        evidence,
        prediction,
        camera_to_world,
        mean_update_mode="absolute",
        raw_scale_residual=True,
        raw_opacity_residual=True,
        raw_rotation_residual=True,
        raw_harmonics_residual=True,
    )
    selected = evidence.candidate_mask
    assert torch.allclose(
        updated.scales[selected] / current.scales[selected],
        raw_scale.exp(),
        atol=1e-5,
    )
    assert torch.allclose(
        updated.opacities[selected],
        torch.sigmoid(torch.full((count,), 2.0)),
        atol=1e-5,
    )
    assert torch.allclose(
        updated.harmonics[selected] - current.harmonics[selected],
        raw_harmonics.reshape(count, 1, 3),
        atol=1e-5,
    )
    assert torch.allclose(stats["raw_rotation"], torch.tensor(0.2))
    assert torch.allclose(stats["rotation_shift_mean"], torch.tensor(0.2))
    assert stats["scale_relative_shift_mean"] > 0
    assert stats["opacity_shift_mean"] > 0
    assert stats["harmonics_shift_mean"] > 0


def test_projected_new_head_stays_in_graph_without_candidates() -> None:
    append_mask = torch.zeros(2, 1, 2, 2, dtype=torch.bool)
    mean_evidence = build_mean_gradient_evidence(
        torch.ones(2, 1, 4, 3),
        active_mask=append_mask.reshape(2, 4),
    )
    current, evidence, camera_to_world = _make_new_anchor_evidence(
        append_mask,
        mean_gradient_evidence=mean_evidence,
    )
    head = ProjectedGSResidualHead(
        feature_dim=4, harmonic_dim=3, hidden_dim=16, chunk_size=2
    )
    head.output.requires_grad_(False)
    prediction = head.forward_new(evidence)
    updated, stats = apply_projected_new_gaussian_residual(
        current, evidence, prediction, camera_to_world
    )
    (updated.opacities.sum() + stats["regularization"]).backward()
    assert all(
        torch.isfinite(stats[key])
        for key in (
            "mean_gradient_current_norm_mean",
            "mean_gradient_replay_norm_mean",
            "mean_gradient_signal_norm_mean",
            "mean_gradient_raw_update_alignment",
            "mean_gradient_effective_update_alignment",
        )
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in [
            *head.decoder.parameters(),
            *head.opacity_gradient_adapter.parameters(),
            *head.mean_gradient_adapter.parameters(),
            *head.new_mean_gradient_adapter.parameters(),
            *head.new_output.parameters(),
        ]
    )


def test_opacity_gradient_evidence_weights_current_and_replay_mean() -> None:
    gradients = torch.tensor(
        [[[2.0, -3.0, 4.0], [-1.0, -2.0, 5.0], [3.0, 1.0, -6.0]]]
    )
    active = torch.tensor([[True, True, False]])
    evidence = build_opacity_gradient_evidence(
        gradients,
        gradient_scale=1.0,
        gradient_clip=20.0,
        gradient_threshold=0.1,
        active_mask=active,
    )

    expected_first = torch.tensor(
        [
            torch.log1p(torch.tensor(2.0)) / 5.0,
            0.0,
            1.0,
            torch.log1p(torch.tensor(0.75)) / 5.0,
            torch.log1p(torch.tensor(0.25)) / 5.0,
            torch.log1p(torch.tensor(2.0)) / 5.0,
            0.5,
            0.5,
        ]
    )
    assert torch.allclose(evidence.features[0, 0], expected_first)
    assert evidence.current_signal[0, 0] == 2.0
    assert evidence.replay_signal[0, 0] == 0.5
    assert evidence.signal[0, 0] == 2.5
    assert evidence.current_signal[0, 1] == -3.0
    assert evidence.replay_signal[0, 1] == -0.25
    assert evidence.signal[0, 1] == -3.25
    assert torch.equal(evidence.features[0, 2], torch.zeros(8))
    assert evidence.signal[0, 2] == 0.0


def test_mean_gradient_evidence_uses_common_camera_vector_space() -> None:
    gradients = torch.tensor(
        [
            [
                [[2.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
                [[0.0, 2.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.0, 4.0, 0.0], [5.0, 0.0, 0.0]],
            ]
        ]
    )
    active = torch.tensor([[True, False]])
    evidence = build_mean_gradient_evidence(
        gradients,
        gradient_scale=1.0,
        gradient_clip=20.0,
        gradient_threshold=0.0,
        current_weight=1.0,
        replay_weight=0.5,
        active_mask=active,
    )

    assert torch.allclose(
        evidence.current_signal[0, 0], torch.tensor([2.0, 0.0, 0.0])
    )
    assert torch.allclose(
        evidence.replay_signal[0, 0], torch.tensor([0.0, 1.5, 0.0])
    )
    assert torch.allclose(
        evidence.signal[0, 0], torch.tensor([2.0, 1.5, 0.0])
    )
    assert torch.allclose(
        evidence.features[0, 0, :3], torch.tensor([1.0, 0.0, 0.0])
    )
    assert torch.allclose(
        evidence.features[0, 0, 4:7], torch.tensor([0.0, 1.0, 0.0])
    )
    assert evidence.features[0, 0, 8] == 0.0
    assert evidence.features[0, 0, 9] == 1.0
    assert torch.equal(evidence.features[0, 1], torch.zeros(10))
    assert torch.equal(evidence.signal[0, 1], torch.zeros(3))

    replay_only = build_mean_gradient_evidence(
        gradients,
        gradient_scale=1.0,
        gradient_clip=20.0,
        gradient_threshold=0.0,
        current_weight=0.0,
        replay_weight=1.0,
        active_mask=active,
    )
    assert torch.equal(
        replay_only.current_signal[0, 0], torch.zeros(3)
    )
    assert torch.allclose(
        replay_only.replay_signal[0, 0], torch.tensor([0.0, 3.0, 0.0])
    )
    assert torch.equal(
        replay_only.signal[0, 0], replay_only.replay_signal[0, 0]
    )


def test_joint_gradient_probe_can_select_independent_replay_suffixes() -> None:
    gradients = torch.arange(6).reshape(1, 6, 1)

    assert torch.equal(
        select_current_and_recent_replay_gradients(gradients, -1), gradients
    )
    assert torch.equal(
        select_current_and_recent_replay_gradients(gradients, 2),
        torch.tensor([[[0], [4], [5]]]),
    )
    assert torch.equal(
        select_current_and_recent_replay_gradients(gradients, 0),
        torch.tensor([[[0]]]),
    )


def test_projected_new_mean_gradient_is_packed_and_reports_alignment() -> None:
    append_mask = torch.tensor(
        [
            [[[True, False], [True, False]]],
            [[[False, True], [False, False]]],
        ]
    )
    gradients = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0]] * 4,
                [[0.0, 2.0, 0.0]] * 4,
            ],
            [
                [[0.0, 0.0, 3.0]] * 4,
                [[1.0, 0.0, 0.0]] * 4,
            ],
        ]
    )
    dense_evidence = build_mean_gradient_evidence(
        gradients,
        gradient_scale=1.0,
        gradient_clip=20.0,
        gradient_threshold=0.0,
        active_mask=append_mask.reshape(2, 4),
    )
    current, evidence, camera_to_world = _make_new_anchor_evidence(
        append_mask,
        mean_gradient_evidence=dense_evidence,
    )
    flat_indices = evidence.flat_candidate_indices
    expected_features = dense_evidence.features.reshape(8, 10).index_select(
        0, flat_indices
    )
    expected_signal = dense_evidence.signal.reshape(8, 3).index_select(
        0, flat_indices
    )
    assert torch.equal(evidence.mean_gradient_features, expected_features)
    assert torch.equal(evidence.mean_gradient_signal, expected_signal)

    count = flat_indices.numel()
    prediction = ProjectedNewGSResidualPrediction(
        delta_mean_camera=-expected_signal,
        delta_rotation=torch.zeros(count, 3),
        delta_log_scale=torch.zeros(count, 3),
        delta_opacity_logit=torch.zeros(count),
        delta_harmonics=torch.zeros(count, 3),
        update_gate_logit=torch.full((count,), 20.0),
    )
    _, stats = apply_projected_new_gaussian_residual(
        current,
        evidence,
        prediction,
        camera_to_world,
        mean_update_mode="absolute",
    )
    assert stats["mean_gradient_raw_update_alignment"] == 1.0
    assert stats["mean_gradient_effective_update_alignment"] == 1.0


def test_projected_new_mean_gradient_adapter_only_changes_mean_output() -> None:
    append_mask = torch.tensor(
        [
            [[[True, False], [True, False]]],
            [[[False, True], [False, False]]],
        ]
    )
    dense_evidence = build_mean_gradient_evidence(
        torch.ones(2, 1, 4, 3),
        active_mask=append_mask.reshape(2, 4),
    )
    _, evidence, _ = _make_new_anchor_evidence(
        append_mask,
        mean_gradient_evidence=dense_evidence,
    )
    evidence_without_mean = replace(
        evidence,
        mean_gradient_features=None,
        mean_gradient_signal=None,
        mean_gradient_current_signal=None,
        mean_gradient_replay_signal=None,
    )
    head = ProjectedGSResidualHead(
        feature_dim=4, harmonic_dim=3, hidden_dim=16, chunk_size=2
    )
    with torch.no_grad():
        head.new_output.weight.fill_(0.1)
        head.new_output.bias.zero_()
        head.new_mean_gradient_adapter[-1].bias.fill_(1.0)

    conditioned = head.forward_new(evidence)
    baseline = head.forward_new(evidence_without_mean)

    assert not torch.allclose(
        conditioned.delta_mean_camera, baseline.delta_mean_camera
    )
    for name in (
        "delta_rotation",
        "delta_log_scale",
        "delta_opacity_logit",
        "delta_harmonics",
        "update_gate_logit",
    ):
        assert torch.equal(getattr(conditioned, name), getattr(baseline, name))


def test_front_conflict_decay_gradient_gate() -> None:
    conflicts = torch.tensor([[True, True, False], [True, False, False]])
    evidence = build_opacity_gradient_evidence(
        torch.tensor([[[2.0, -1.0, 5.0]], [[0.25, 3.0, -4.0]]]),
        gradient_scale=1.0,
        gradient_clip=20.0,
        gradient_threshold=0.0,
    )

    gated, ratio = gate_front_conflict_decay(
        conflicts,
        evidence,
        enabled=True,
        threshold=0.5,
    )
    assert torch.equal(
        gated,
        torch.tensor([[True, False, False], [False, False, False]]),
    )
    assert torch.allclose(ratio, torch.tensor(1.0 / 3.0))

    disabled, disabled_ratio = gate_front_conflict_decay(
        conflicts,
        evidence,
        enabled=False,
    )
    assert torch.equal(disabled, conflicts)
    assert disabled_ratio == 1.0

    missing, missing_ratio = gate_front_conflict_decay(
        conflicts,
        None,
        enabled=True,
    )
    assert torch.equal(missing, conflicts)
    assert missing_ratio == 1.0


def test_front_conflict_decay_gradient_gate_validates_shape() -> None:
    evidence = build_opacity_gradient_evidence(
        torch.ones(1, 1, 2), gradient_scale=1.0
    )
    try:
        gate_front_conflict_decay(
            torch.ones(1, 3, dtype=torch.bool),
            evidence,
            enabled=True,
        )
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected a shape mismatch to raise ValueError.")


def test_provisional_gaussians_keep_dense_new_anchor_indexing() -> None:
    history = _make_state().gaussians
    current = Gaussians(
        means=torch.zeros(2, 4, 3),
        harmonics=torch.zeros(2, 4, 1, 3),
        opacities=torch.full((2, 4), 0.8),
        scales=torch.full((2, 4, 3), 0.02),
        rotations=torch.nn.functional.pad(torch.zeros(2, 4, 3), (0, 1), value=1.0),
    )
    mask = torch.tensor(
        [[True, False, True, False], [False, True, False, False]]
    )
    weight = torch.full((2, 4), 0.5)
    provisional, active = build_provisional_gaussians(
        history, current, weight, mask
    )

    assert provisional.opacities.shape == (2, 7)
    assert torch.allclose(
        provisional.opacities[:, -4:], 0.4 * mask.to(torch.float32)
    )
    assert torch.equal(active[:, -4:], mask)


def test_new_residual_reports_opacity_gradient_alignment() -> None:
    append_mask = torch.tensor(
        [
            [[[True, False], [True, False]]],
            [[[False, True], [False, False]]],
        ]
    )
    current, evidence, camera_to_world = _make_new_anchor_evidence(append_mask)
    count = evidence.flat_candidate_indices.numel()
    evidence.opacity_gradient_features = torch.zeros(count, 8)
    evidence.opacity_gradient_signal = torch.tensor([1.0, -1.0, 0.0])
    prediction = ProjectedNewGSResidualPrediction(
        delta_mean_camera=torch.zeros(count, 3),
        delta_rotation=torch.zeros(count, 3),
        delta_log_scale=torch.zeros(count, 3),
        delta_opacity_logit=torch.tensor([-1.0, 1.0, 1.0]),
        delta_harmonics=torch.zeros(count, 3),
        update_gate_logit=torch.full((count,), 20.0),
    )
    _, stats = apply_projected_new_gaussian_residual(
        current,
        evidence,
        prediction,
        camera_to_world,
        raw_opacity_residual=True,
    )
    assert stats["gradient_signal_abs_mean"] == torch.tensor(2.0 / 3.0)
    assert stats["gradient_signal_positive_ratio"] == torch.tensor(0.5)
    assert stats["gradient_update_alignment"] == torch.tensor(1.0)
