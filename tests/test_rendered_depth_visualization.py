"""Test rendered-depth display conversion without importing the model stack."""

import ast
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_helper():
    tree = ast.parse(
        (ROOT / "src/model/model_wrapper.py").read_text(encoding="utf-8")
    )
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_expected_render_depth_for_visualization"
    )
    namespace = {"torch": torch, "math": math}
    exec(
        compile(
            ast.Module(body=[helper], type_ignores=[]),
            "rendered_depth_visualization",
            "exec",
        ),
        namespace,
    )
    return namespace[helper.name]


CONVERT = load_helper()


def test_converts_accumulated_depth_to_expected_depth() -> None:
    expected = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])
    alpha = torch.tensor([[[0.5, 0.25], [0.8, 1.0]]])

    display, valid = CONVERT(expected * alpha, alpha)

    assert valid.all()
    torch.testing.assert_close(display, expected)


def test_masks_low_alpha_without_polluting_display_depth() -> None:
    accumulated = torch.tensor([[[1.0, 0.0], [2.0, float("nan")]]])
    alpha = torch.tensor([[[0.5, 0.0], [0.5, 1.0]]])

    display, valid = CONVERT(accumulated, alpha, min_alpha=0.01)

    torch.testing.assert_close(
        valid,
        torch.tensor([[[True, False], [True, False]]]),
    )
    assert torch.isfinite(display).all()
    torch.testing.assert_close(display[valid], torch.tensor([2.0, 4.0]))


def test_rejects_mismatched_shapes_and_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        CONVERT(torch.ones(2, 2), torch.ones(2, 3))
    with pytest.raises(ValueError, match="min_alpha"):
        CONVERT(torch.ones(2, 2), torch.ones(2, 2), min_alpha=1.0)


def test_model_keeps_all_alpha_views_for_rgb_loss_masking() -> None:
    source = (ROOT / "src/model/model/anysplat.py").read_text(encoding="utf-8")

    assert "output.depth = output.depth[:, :ctx_img_num, ...]" in source
    assert "output.alpha = output.alpha[:, :ctx_img_num, ...]" not in source
