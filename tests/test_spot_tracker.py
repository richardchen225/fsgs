import pytest
import torch

from src.model.tracking import (
    RollingSPOTTracker,
    forward_backward_tracker_consistency,
    forward_splat_tracker_coverage,
)


class _FakeSPOT:
    def __init__(self, *, nonfinite: bool = False) -> None:
        self.nonfinite = nonfinite
        self.previous_input = None
        self.current_input = None

    def encode_context(self, image: torch.Tensor):
        self.previous_input = image
        shape = (image.shape[0], 128, *image.shape[-2:])
        return image.new_zeros(shape), image.new_zeros(shape)

    def encode_features(self, image: torch.Tensor):
        batch, _, height, width = image.shape
        coordinates = image.new_zeros(batch, 2, height, width)
        features = image.new_zeros(batch, 256, height, width)
        alpha = image.new_zeros(batch, 1, height, width)
        return coordinates, coordinates, features, alpha, None

    def fnet(self, image: torch.Tensor) -> torch.Tensor:
        self.current_input = image
        return image.new_zeros(image.shape[0], 256, *image.shape[-2:])

    def predict_flow(self, *args, **kwargs):
        source_features = args[4]
        batch, _, height, width = source_features.shape
        value = float("nan") if self.nonfinite else 0.0
        return {
            "flow": source_features.new_full((batch, height, width, 2), value),
            "alpha": source_features.new_zeros(batch, height, width),
        }


def _tracker_with_fake_model(model: _FakeSPOT) -> RollingSPOTTracker:
    tracker = RollingSPOTTracker("unused", "unused", mixed_precision=False)
    tracker._model = model
    tracker._device = torch.device("cpu")
    return tracker


def test_tracker_coverage_preserves_supported_pixels_under_zero_flow() -> None:
    support = torch.tensor(
        [[[[True, False, True], [True, True, False]]]]
    )
    flow = torch.zeros(1, 2, 3, 2)
    visibility = torch.ones(1, 2, 3)

    coverage = forward_splat_tracker_coverage(
        flow,
        visibility,
        support,
        dilation_radius=0,
    )

    assert torch.equal(coverage, support)


def test_tracker_coverage_forward_splats_horizontal_translation() -> None:
    support = torch.ones(1, 1, 2, 4, dtype=torch.bool)
    flow = torch.zeros(1, 2, 4, 2)
    flow[..., 0] = 1.0
    visibility = torch.ones(1, 2, 4)

    coverage = forward_splat_tracker_coverage(
        flow,
        visibility,
        support,
        dilation_radius=0,
    )

    expected = torch.ones_like(support)
    expected[..., 0] = False
    assert torch.equal(coverage, expected)


def test_tracker_coverage_respects_visibility_threshold() -> None:
    support = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    flow = torch.zeros(1, 2, 2, 2)
    visibility = torch.tensor([[[0.9, 0.7], [0.8, 0.1]]])

    coverage = forward_splat_tracker_coverage(
        flow,
        visibility,
        support,
        visibility_threshold=0.8,
        dilation_radius=0,
    )

    expected = torch.tensor([[[[True, False], [True, False]]]])
    assert torch.equal(coverage, expected)


def test_forward_backward_consistency_accepts_inverse_translation() -> None:
    forward = torch.zeros(1, 2, 4, 2)
    backward = torch.zeros_like(forward)
    forward[..., 0] = 1.0
    backward[..., 0] = -1.0
    visibility = torch.ones(1, 2, 4)

    valid, error = forward_backward_tracker_consistency(
        forward,
        visibility,
        backward,
        visibility,
        absolute_threshold=0.1,
        relative_threshold=0.0,
    )

    expected = torch.ones(1, 1, 2, 4, dtype=torch.bool)
    expected[..., -1] = False
    assert torch.equal(valid, expected)
    assert torch.all(error[valid] == 0.0)


def test_forward_backward_consistency_rejects_cycle_error() -> None:
    forward = torch.zeros(1, 2, 4, 2)
    forward[..., 0] = 1.0
    backward = torch.zeros_like(forward)
    visibility = torch.ones(1, 2, 4)

    valid, error = forward_backward_tracker_consistency(
        forward,
        visibility,
        backward,
        visibility,
        absolute_threshold=0.1,
        relative_threshold=0.0,
    )

    assert not valid.any()
    assert torch.all(error[..., :-1] == 1.0)


def test_forward_backward_consistency_requires_backward_visibility() -> None:
    flow = torch.zeros(1, 2, 3, 2)
    forward_visibility = torch.ones(1, 2, 3)
    backward_visibility = torch.ones(1, 2, 3)
    backward_visibility[..., 1] = 0.0

    valid, _ = forward_backward_tracker_consistency(
        flow,
        forward_visibility,
        flow,
        backward_visibility,
    )

    expected = torch.ones(1, 1, 2, 3, dtype=torch.bool)
    expected[..., 1] = False
    assert torch.equal(valid, expected)


def test_tracker_maps_rgb_to_official_spot_input_range() -> None:
    model = _FakeSPOT()
    tracker = _tracker_with_fake_model(model)

    flow, visibility = tracker.predict_pair(
        torch.zeros(1, 3, 8, 8),
        torch.ones(1, 3, 8, 8),
    )

    assert torch.equal(
        model.previous_input,
        torch.full_like(model.previous_input, -1.0),
    )
    assert torch.equal(model.current_input, torch.ones_like(model.current_input))
    assert flow.shape == (1, 8, 8, 2)
    assert torch.all(visibility == 0.5)


def test_tracker_rejects_nonfinite_predictions() -> None:
    tracker = _tracker_with_fake_model(_FakeSPOT(nonfinite=True))

    with pytest.raises(FloatingPointError, match="non-finite"):
        tracker.predict_pair(
            torch.zeros(1, 3, 8, 8),
            torch.zeros(1, 3, 8, 8),
        )
