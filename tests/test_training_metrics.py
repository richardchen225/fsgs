"""Check logging selection without importing the GPU model stack."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_selector():
    namespace = {}
    constants = ast.parse((ROOT / "src/model/streaming_gir.py").read_text(encoding="utf-8"))
    assignments = [
        node for node in constants.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {"MEAN_GRADIENT_STAT_KEYS", "OLD_OPACITY_GRADIENT_EFFECTIVE_STAT_KEYS"}
            for target in node.targets
        )
    ]
    wrapper = ast.parse((ROOT / "src/model/model_wrapper.py").read_text(encoding="utf-8"))
    helper = next(
        node for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == "_gir_training_metric_names"
    )
    exec(compile(ast.Module(body=[*assignments, helper], type_ignores=[]),
                 "training_metrics", "exec"), namespace)
    return namespace[helper.name]


SELECT = load_selector()


def experiment(**overrides):
    values = dict(
        gir_enabled=True,
        gir_append_mode="point_map",
        gir_old_residual_enabled=True,
        gir_old_update_mode="projected_shared",
        gir_projected_front_decay_strength=0.0,
        gir_new_residual_enabled=True,
        gir_old_opacity_gradient_enabled=True,
        gir_new_opacity_gradient_enabled=True,
        gir_opacity_gradient_front_decay_gate_enabled=True,
        gir_mean_gradient_enabled=False,
        gir_new_mean_gradient_enabled=False,
        gir_old_delete_enabled=False,
        gir_old_decay_enabled=False,
    )
    return SimpleNamespace(**(values | overrides))


class TrainingMetricTests(unittest.TestCase):
    def test_current_experiment_retains_opacity_diagnostics_without_dead_curves(self):
        names = SELECT(experiment())
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("gir_projected_gradient_current_replay_conflict_ratio", names)
        self.assertIn("gir_new_residual_gradient_update_alignment", names)
        self.assertIn("gir_point_front_gradient_decay_gate_ratio", names)
        self.assertFalse(any("mean_gradient_" in name for name in names))
        self.assertFalse(any(name.endswith(("_enabled", "_mode_code", "_warmup_progress")) for name in names))
        for removed in (
            "gir_residual_independent_heads", "gir_residual_topk",
            "gir_historical_gate", "gir_residual_magnitude",
            "gir_new_residual_magnitude", "gir_new_residual_gate",
            "gir_projected_front_decay_probability", "gir_add_rate", "gir_add_gate_learned",
        ):
            self.assertNotIn(removed, names)

    def test_disabled_opacity_paths_do_not_emit_placeholder_statistics(self):
        names = SELECT(experiment(
            gir_old_opacity_gradient_enabled=False,
            gir_new_opacity_gradient_enabled=False,
            gir_opacity_gradient_front_decay_gate_enabled=False,
        ))
        self.assertFalse(any("gradient_" in name for name in names))

    def test_optional_mean_and_decay_diagnostics_return_when_enabled(self):
        names = SELECT(experiment(
            gir_mean_gradient_enabled=True,
            gir_new_mean_gradient_enabled=True,
            gir_projected_front_decay_strength=0.02,
            gir_old_delete_enabled=True,
        ))
        self.assertEqual(sum("mean_gradient_" in name for name in names), 14)
        self.assertIn("gir_projected_front_decay_probability", names)
        self.assertIn("gir_old_delete_candidate_count", names)
        self.assertNotIn("gir_old_delete_target_ratio", names)

    def test_pixel_and_learned_paths_keep_their_own_diagnostics(self):
        names = SELECT(experiment(
            gir_old_update_mode="pixel_topk", gir_append_mode="learned",
            gir_top1_confidence_mode="none", gir_new_residual_enabled=False,
        ))
        self.assertIn("gir_historical_gate", names)
        self.assertIn("gir_add_rate", names)
        self.assertNotIn("gir_top1_confidence_mean", names)
        self.assertFalse(any(name.startswith(("gir_projected_", "gir_new_residual_", "gir_point_")) for name in names))

    def test_disabled_gir_has_no_gir_metrics(self):
        self.assertEqual(SELECT(experiment(gir_enabled=False)), ())


if __name__ == "__main__":
    unittest.main()
