import ast
from pathlib import Path
from types import SimpleNamespace


MAIN_PATH = Path(__file__).parents[1] / "src" / "main.py"


def _load_prefix_helper():
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_required_test_checkpoint_prefixes"
    )
    namespace = {}
    module = ast.Module(body=[helper], type_ignores=[])
    exec(compile(module, MAIN_PATH, "exec"), namespace)
    return namespace[helper.name]


REQUIRED_PREFIXES = _load_prefix_helper()


def test_october_baseline_does_not_require_new_projected_parameters() -> None:
    prefixes = REQUIRED_PREFIXES(
        SimpleNamespace(
            gir_enabled=True,
            gir_old_residual_enabled=True,
            gir_old_update_mode="projected_shared",
            gir_new_residual_enabled=False,
            gir_old_opacity_gradient_enabled=False,
            gir_new_opacity_gradient_enabled=False,
            gir_mean_gradient_enabled=False,
            gir_new_mean_gradient_enabled=False,
        )
    )

    assert "gir_projected_update_head.decoder." in prefixes
    assert "gir_projected_update_head.output." in prefixes
    assert "gir_projected_update_head.new_output." not in prefixes
    assert "gir_projected_update_head.opacity_gradient_adapter." not in prefixes
    assert "gir_projected_update_head.mean_gradient_adapter." not in prefixes
    assert "gir_projected_update_head.new_mean_gradient_adapter." not in prefixes


def test_enabled_projected_branches_require_their_checkpoint_parameters() -> None:
    prefixes = REQUIRED_PREFIXES(
        SimpleNamespace(
            gir_enabled=True,
            gir_old_residual_enabled=True,
            gir_old_update_mode="projected_shared",
            gir_new_residual_enabled=True,
            gir_old_opacity_gradient_enabled=True,
            gir_new_opacity_gradient_enabled=True,
            gir_mean_gradient_enabled=True,
            gir_new_mean_gradient_enabled=True,
        )
    )

    assert "gir_projected_update_head.decoder." in prefixes
    assert "gir_projected_update_head.output." in prefixes
    assert "gir_projected_update_head.new_output." in prefixes
    assert "gir_projected_update_head.opacity_gradient_adapter." in prefixes
    assert "gir_projected_update_head.mean_gradient_adapter." in prefixes
    assert "gir_projected_update_head.new_mean_gradient_adapter." in prefixes
