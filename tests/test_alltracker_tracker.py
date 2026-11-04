import pytest
import torch

from src.model.tracking import RollingAllTracker


class _FakeAllTracker:
    def __init__(self) -> None:
        self.images = None

    def __call__(self, images, *, iters, sw, is_training):
        self.images = images
        batch, _, _, height, width = images.shape
        flow = images.new_zeros(batch, 2, height, width)
        flow[:, 0] = 2.0
        visibility_confidence = images.new_zeros(batch, 2, height, width)
        visibility_confidence[:, 0] = 0.8
        visibility_confidence[:, 1] = 0.5
        return flow, visibility_confidence, None, None


def test_alltracker_adapter_uses_official_visibility_confidence_product() -> None:
    model = _FakeAllTracker()
    tracker = RollingAllTracker("unused", "unused", mixed_precision=False)
    tracker._model = model
    tracker._device = torch.device("cpu")

    flow, visibility = tracker.predict_pair(
        torch.zeros(1, 3, 8, 8),
        torch.ones(1, 3, 8, 8),
    )

    assert model.images.shape == (1, 2, 3, 8, 8)
    assert torch.all(model.images[:, 0] == 0.0)
    assert torch.all(model.images[:, 1] == 255.0)
    assert flow.shape == (1, 8, 8, 2)
    assert torch.all(flow[..., 0] == 2.0)
    assert torch.all(flow[..., 1] == 0.0)
    assert torch.allclose(visibility, torch.full_like(visibility, 0.4))


def test_alltracker_adapter_rejects_shape_mismatch() -> None:
    tracker = RollingAllTracker("unused", "unused", mixed_precision=False)

    with pytest.raises(ValueError, match="shapes must match"):
        tracker.predict_pair(
            torch.zeros(1, 3, 8, 8),
            torch.zeros(1, 3, 8, 9),
        )
