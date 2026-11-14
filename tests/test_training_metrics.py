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
        gir_split_enabled=False,
        gir_functional_update_enabled=False,
        gir_symmetric_multiview_update_enabled=False,
        gir_new_mean_gradient_enabled=False,
        gir_add_gate_enabled=False,
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
        self.assertNotIn("gir_add_gate", names)
        self.assertFalse(any("mean_gradient_" in name for name in names))
        self.assertFalse(any(name.endswith(("_enabled", "_mode_code", "_warmup_progress")) for name in names))
        for removed in (
            "gir_residual_independent_heads", "gir_residual_topk",
            "gir_historical_gate", "gir_residual_magnitude",
            "gir_new_residual_magnitude", "gir_new_residual_gate",
            "gir_projected_front_decay_probability", "gir_add_rate", "gir_add_gate_learned",
            "gir_visible_ratio", "gir_raster_alpha", "gir_effective_new_ratio",
            "gir_new_residual_candidate_ratio", "gir_new_residual_raw_mean",
            "gir_new_residual_regularization",
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
        self.assertEqual(sum("mean_gradient_" in name for name in names), 10)
        self.assertIn("gir_projected_front_decay_probability", names)
        self.assertIn("gir_old_delete_candidate_rate", names)
        self.assertNotIn("gir_old_delete_candidate_count", names)
        self.assertNotIn("gir_old_delete_target_ratio", names)

    def test_pixel_and_learned_paths_keep_their_own_diagnostics(self):
        names = SELECT(experiment(
            gir_old_update_mode="pixel_topk", gir_append_mode="learned",
            gir_top1_confidence_mode="none", gir_new_residual_enabled=False,
        ))
        self.assertIn("gir_historical_gate", names)
        self.assertIn("gir_add_rate", names)
        self.assertIn("gir_add_gate_learned", names)
        self.assertNotIn("gir_top1_confidence_mean", names)
        self.assertFalse(any(name.startswith(("gir_projected_", "gir_new_residual_", "gir_point_")) for name in names))

    def test_tracker_path_keeps_only_tracker_coverage_diagnostics(self):
        names = SELECT(experiment(gir_append_mode="tracker"))
        self.assertIn("gir_point_coverage_ratio", names)
        self.assertIn("gir_point_append_ratio", names)
        self.assertIn("gir_tracker_visibility_ratio", names)
        self.assertIn("gir_tracker_fb_consistency_ratio", names)
        self.assertIn("gir_tracker_cycle_error_mean", names)
        self.assertNotIn("gir_point_depth_scale", names)
        self.assertNotIn("gir_point_front_conflict_ratio", names)

    def test_tracker_point_map_path_keeps_both_coverage_diagnostics(self):
        names = SELECT(experiment(gir_append_mode="tracker_point_map"))
        for name in (
            "gir_point_coverage_ratio",
            "gir_point_append_ratio",
            "gir_point_front_conflict_ratio",
            "gir_point_depth_scale",
            "gir_tracker_coverage_ratio",
            "gir_tracker_visibility_ratio",
            "gir_tracker_fb_consistency_ratio",
            "gir_tracker_cycle_error_mean",
            "gir_tracker_point_union_coverage_ratio",
            "gir_confirmed_current_in_front_ratio",
        ):
            self.assertIn(name, names)

    def test_split_path_keeps_only_actionable_diagnostics(self):
        names = SELECT(experiment(
            gir_mean_gradient_enabled=True,
            gir_split_enabled=True,
        ))
        self.assertIn("gir_split_count", names)
        self.assertIn("gir_split_ratio", names)
        self.assertIn("gir_split_selected_radius", names)
        self.assertIn("gir_split_selected_gradient_norm", names)
        self.assertIn("gir_split_map_count_after", names)
        self.assertNotIn("gir_split_eligible_count", names)
        self.assertNotIn("gir_split_selected_score", names)

    def test_functional_update_keeps_only_actionable_diagnostics(self):
        names = SELECT(experiment(gir_functional_update_enabled=True))
        self.assertIn("gir_functional_update_selected_count", names)
        self.assertIn("gir_functional_update_selected_ratio", names)
        self.assertIn("gir_functional_update_mean_step_norm", names)
        self.assertIn("gir_functional_update_opacity_logit_step_abs", names)
        self.assertIn("gir_functional_update_harmonics_step_norm", names)
        self.assertNotIn("gir_functional_update_score_mean", names)
        self.assertNotIn("gir_functional_update_mean_gradient_norm", names)

    def test_symmetric_update_keeps_view_balance_and_update_diagnostics(self):
        names = SELECT(experiment(
            gir_old_residual_enabled=False,
            gir_new_residual_enabled=False,
            gir_symmetric_multiview_update_enabled=True,
        ))
        for name in (
            "gir_symmetric_multiview_active_ratio",
            "gir_symmetric_multiview_visible_views_mean",
            "gir_symmetric_multiview_mean_coherence",
            "gir_symmetric_multiview_mean_conflict_ratio",
            "gir_symmetric_multiview_mean_shift_mean",
            "gir_symmetric_multiview_opacity_shift_mean",
            "gir_symmetric_multiview_harmonics_shift_mean",
        ):
            self.assertIn(name, names)
        self.assertNotIn("gir_symmetric_multiview_active_count", names)
        self.assertNotIn("gir_symmetric_multiview_regularization", names)

    def test_compact_mode_keeps_only_dashboard_gir_metrics(self):
        names = SELECT(
            experiment(
                gir_old_residual_enabled=False,
                gir_new_residual_enabled=False,
                gir_symmetric_multiview_update_enabled=True,
            ),
            compact=True,
        )
        self.assertEqual(names, (
            "gir_symmetric_multiview_active_ratio",
            "gir_symmetric_multiview_visible_views_mean",
            "gir_symmetric_multiview_mean_conflict_ratio",
            "gir_symmetric_multiview_mean_shift_mean",
            "gir_symmetric_multiview_opacity_shift_mean",
            "gir_symmetric_multiview_harmonics_shift_mean",
        ))

    def test_disabled_gir_has_no_gir_metrics(self):
        self.assertEqual(SELECT(experiment(gir_enabled=False)), ())

    def test_checkpoint_step_metric_survives_compact_logging(self):
        wrapper = ast.parse(
            (ROOT / "src/model/model_wrapper.py").read_text(encoding="utf-8")
        )
        training_step = next(
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.FunctionDef)
            and node.name == "training_step"
        )
        calls = [
            node
            for node in ast.walk(training_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "log"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "info/global_step"
        ]

        self.assertEqual(len(calls), 1)
        logger_keyword = next(
            keyword for keyword in calls[0].keywords if keyword.arg == "logger"
        )
        self.assertIsInstance(logger_keyword.value, ast.UnaryOp)
        self.assertIsInstance(logger_keyword.value.op, ast.Not)
        self.assertIsInstance(logger_keyword.value.operand, ast.Name)
        self.assertEqual(logger_keyword.value.operand.id, "compact_logging")


if __name__ == "__main__":
    unittest.main()
