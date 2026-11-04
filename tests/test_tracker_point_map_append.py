"""Unit tests for the tracker/point-map append policy."""

import ast
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def load_helpers():
    module = ast.parse(
        (ROOT / "src/model/streaming_gir.py").read_text(encoding="utf-8")
    )
    helpers = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_dilate_depth_consistent_coverage",
            "build_confirmed_current_in_front_mask",
            "combine_tracker_point_map_append_mask",
        }
    ]
    namespace = {"torch": torch, "F": F}
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), "append_mask", "exec"),
        namespace,
    )
    return (
        namespace["combine_tracker_point_map_append_mask"],
        namespace["build_confirmed_current_in_front_mask"],
        namespace["_dilate_depth_consistent_coverage"],
    )


COMBINE, CURRENT_IN_FRONT, DILATE_COVERAGE = load_helpers()


class TrackerPointMapAppendTests(unittest.TestCase):
    def test_coverage_dilation_expands_across_a_continuous_surface(self):
        coverage = torch.zeros((1, 1, 5, 5), dtype=torch.bool)
        coverage[..., 2, 2] = True
        depth = torch.ones((1, 1, 5, 5))
        reliable = torch.ones_like(coverage)

        expanded = DILATE_COVERAGE(
            coverage,
            depth,
            reliable,
            radius=2,
            depth_relative_tolerance=0.1,
        )

        self.assertTrue(bool(expanded.all()))

    def test_coverage_dilation_stops_at_a_depth_discontinuity(self):
        coverage = torch.zeros((1, 1, 3, 5), dtype=torch.bool)
        coverage[..., 1, 0] = True
        depth = torch.ones((1, 1, 3, 5))
        depth[..., 2:] = 3.0
        reliable = torch.ones_like(coverage)

        expanded = DILATE_COVERAGE(
            coverage,
            depth,
            reliable,
            radius=4,
            depth_relative_tolerance=0.1,
        )

        self.assertTrue(bool(expanded[..., :2].all()))
        self.assertFalse(bool(expanded[..., 2:].any()))

    def test_union_coverage_suppresses_duplicate_append(self):
        valid = torch.ones((1, 1, 1, 4), dtype=torch.bool)
        tracker = torch.tensor([[[[True, False, False, False]]]])
        point = torch.tensor([[[[False, True, False, False]]]])
        foreground = torch.zeros_like(valid)

        append, known = COMBINE(valid, tracker, point, foreground)

        self.assertTrue(torch.equal(known, tracker | point))
        self.assertTrue(
            torch.equal(
                append,
                torch.tensor([[[[False, False, True, True]]]]),
            )
        )

    def test_confirmed_foreground_forces_append_over_known_coverage(self):
        valid = torch.ones((1, 1, 1, 2), dtype=torch.bool)
        covered = torch.ones_like(valid)
        foreground = torch.tensor([[[[True, False]]]])

        append, _ = COMBINE(valid, covered, covered, foreground)

        self.assertTrue(
            torch.equal(append, torch.tensor([[[[True, False]]]]))
        )

    def test_invalid_current_pixel_is_never_appended(self):
        valid = torch.zeros((1, 1, 1, 1), dtype=torch.bool)
        missing = torch.zeros_like(valid)
        foreground = torch.ones_like(valid)

        append, _ = COMBINE(valid, missing, missing, foreground)

        self.assertFalse(bool(append.any()))

    def test_rejects_shape_and_dtype_mismatches(self):
        valid = torch.ones((1, 1, 1, 1), dtype=torch.bool)
        with self.assertRaises(ValueError):
            COMBINE(valid, valid[..., :0], valid, valid)
        with self.assertRaises(TypeError):
            COMBINE(valid, valid.float(), valid, valid)

    def test_current_foreground_uses_aligned_depth_and_old_alpha(self):
        current_depth = torch.ones((1, 1, 1, 3))
        old_depth = torch.tensor([[[[2.5, 2.5, 2.5]]]])
        old_alpha = torch.tensor([[[[1.0, 0.01, 1.0]]]])
        confidence = torch.tensor([[[[1.0, 1.0, 0.1]]]])

        foreground = CURRENT_IN_FRONT(
            current_depth,
            old_depth,
            old_alpha,
            depth_scale=torch.tensor([2.0]),
            depth_relative_tolerance=0.1,
            old_alpha_threshold=0.1,
            current_depth_confidence=confidence,
            min_depth_confidence=0.5,
        )

        self.assertTrue(
            torch.equal(foreground, torch.tensor([[[[True, False, False]]]]))
        )


if __name__ == "__main__":
    unittest.main()
