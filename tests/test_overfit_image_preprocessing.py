import ast
from pathlib import Path

import torch
import torch.nn.functional as F


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "experiments"
    / "overfit_old_gs_residual.py"
)


def _load_resize_and_center_crop():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resize_and_center_crop"
    )
    namespace = {"torch": torch, "F": F}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_resize_and_center_crop"]


_resize_and_center_crop = _load_resize_and_center_crop()


def test_resize_and_center_crop_preserves_aspect_ratio_and_updates_intrinsics():
    image = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8)
    intrinsic = torch.tensor(
        [[4.0, 0.0, 4.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]
    )

    resized, resized_intrinsic = _resize_and_center_crop(
        image,
        intrinsic,
        (4, 4),
    )

    torch.testing.assert_close(resized, image[:, :, 2:6])
    torch.testing.assert_close(
        resized_intrinsic,
        torch.tensor(
            [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]
        ),
    )


def test_resize_and_center_crop_scales_before_cropping():
    image = torch.zeros((3, 4, 8), dtype=torch.float32)
    intrinsic = torch.tensor(
        [[4.0, 0.0, 4.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]
    )

    resized, resized_intrinsic = _resize_and_center_crop(
        image,
        intrinsic,
        (8, 8),
    )

    assert resized.shape == (3, 8, 8)
    torch.testing.assert_close(
        resized_intrinsic,
        torch.tensor(
            [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]
        ),
    )
