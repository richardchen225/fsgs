from pathlib import Path

import torch

from src.loss.loss_lpips import LossLpips, LossLpipsCfg
from src.loss.loss_mse import LossMse, LossMseCfg
from src.model.decoder.decoder import DecoderOutput


def _batch(target: torch.Tensor) -> dict:
    return {"context": {"image": target * 2.0 - 1.0}}


def test_decoder_preserves_single_view_alpha_dimension() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/model/decoder/decoder_splatting_cuda.py"
    ).read_text(encoding="utf-8")

    assert "torch.cat(rendering_alpha_list, dim=0).squeeze(-1)" in source


def test_mse_ignores_uncovered_source_and_heldout_pixels() -> None:
    loss = object.__new__(LossMse)
    torch.nn.Module.__init__(loss)
    loss.cfg = LossMseCfg(
        weight_ctx=1.0,
        weight_novel=1.0,
        alpha=True,
        alpha_threshold=1e-4,
    )
    color = torch.zeros(1, 2, 3, 1, 2, requires_grad=True)
    alpha = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    target = torch.zeros_like(color)
    target[..., 0] = 1.0
    target[..., 1] = 0.25
    output = DecoderOutput(color, None, alpha, None)

    value = loss.forward(
        output,
        _batch(target),
        None,
        {"depth": torch.ones(1, 1, 1, 2, 1)},
        0,
    )

    torch.testing.assert_close(value, torch.tensor(2.0))
    value.backward()
    assert color.grad is not None
    assert color.grad[..., 0].abs().sum() > 0
    assert color.grad[..., 1].abs().sum() == 0


def test_lpips_masks_uncovered_pixels_with_detached_alpha() -> None:
    class SquaredErrorLpips(torch.nn.Module):
        def forward(self, prediction, target, normalize=True):
            return (prediction - target).square().mean(
                dim=(1, 2, 3), keepdim=True
            )

    loss = object.__new__(LossLpips)
    torch.nn.Module.__init__(loss)
    loss.cfg = LossLpipsCfg(
        weight=1.0,
        apply_after_step=0,
        view_group="all",
        alpha=True,
        alpha_threshold=1e-4,
    )
    loss.lpips = SquaredErrorLpips()
    color = torch.zeros(1, 1, 3, 1, 2, requires_grad=True)
    alpha = torch.tensor([[[[1.0, 0.0]]]], requires_grad=True)
    target = torch.ones_like(color)
    output = DecoderOutput(color, None, alpha, None)

    value = loss.forward(
        output,
        _batch(target),
        None,
        {"depth": torch.ones(1, 1, 1, 2, 1)},
        0,
    )

    torch.testing.assert_close(value, torch.tensor(0.5))
    value.backward()
    assert color.grad is not None
    assert color.grad[..., 0].abs().sum() > 0
    assert color.grad[..., 1].abs().sum() == 0
    assert alpha.grad is None
