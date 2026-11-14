from types import MethodType, SimpleNamespace

import torch
from torch import nn

from src.loss.loss_depth import LossDepth


def _depth_loss_without_teacher() -> LossDepth:
    loss = LossDepth.__new__(LossDepth)
    nn.Module.__init__(loss)
    loss.cfg = SimpleNamespace(
        weight=1.0,
        alignment="sequence_scale_only_log",
        absolute_log_scale_weight=0.1,
    )
    loss.name = "depth"
    loss.last_teacher_gt_metrics = {}
    loss.last_depth_components = {}
    return loss


def test_prepared_context_target_skips_teacher_in_loss() -> None:
    loss = _depth_loss_without_teacher()

    def fail_if_called(self, batch, ctx_num, device):
        raise AssertionError("teacher target was recomputed")

    loss._context_depth_target = MethodType(fail_if_called, loss)
    prediction = torch.full((1, 2, 3, 4, 1), 2.0, requires_grad=True)
    target = torch.full((1, 2, 3, 4), 2.5)
    prepared_target = (
        target,
        torch.tensor([True]),
        torch.tensor([1.0]),
    )

    value = loss.ctx_depth_loss(
        prediction,
        {"context": {}},
        cxt_depth_weight=1.0,
        prepared_target=prepared_target,
    )

    assert torch.isfinite(value)
    value.backward()
    assert prediction.grad is not None


def test_disabled_depth_supervision_does_not_prepare_teacher_target() -> None:
    loss = _depth_loss_without_teacher()

    def fail_if_called(self, batch, ctx_num, device):
        raise AssertionError("disabled supervision ran the teacher")

    loss._context_depth_target = MethodType(fail_if_called, loss)
    batch = {"context": {"image": torch.zeros(1, 2, 3, 3, 4)}}

    assert loss.prepare_context_depth_target(
        batch,
        ctx_num=1,
        cxt_depth_weight=0.0,
    ) is None
