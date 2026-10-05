"""Depth supervision contracts without importing CUDA or external teachers."""

import ast
from dataclasses import dataclass, fields
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F


class LossBase(nn.Module):
    def __class_getitem__(cls, args):
        return cls

    def __init__(self, cfg):
        super().__init__()


def load_depth_loss():
    path = Path(__file__).resolve().parents[1] / "src/loss/loss_depth.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *classes], type_ignores=[]))
    namespace = {"__name__": __name__, "Loss": LossBase, "torch": torch, "F": F,
                 "dataclass": dataclass, "fields": fields}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["LossDepth"], namespace["LossDepthCfg"], namespace["LossDepthCfgWrapper"]


LossDepth, Config, Wrapper = load_depth_loss()


class ABotDepthLossTests(unittest.TestCase):
    def make_loss(self):
        return LossDepth(Wrapper(Config(weight=1., sigma_image=None, use_second_derivative=False, teacher="abot")))

    def test_abot_constructor_has_no_dav3_import_or_network(self):
        with patch("builtins.__import__", wraps=__import__) as imports:
            loss = self.make_loss()
        self.assertFalse(any("dav3" in str(call) for call in imports.call_args_list))
        self.assertFalse(hasattr(loss, "depth_anything"))
        self.assertEqual(list(loss.parameters()), [])

    def test_log_l1_weight_and_student_gradient_with_detached_teacher(self):
        torch.manual_seed(4)
        prediction = (torch.rand(2, 3, 4, 5, 1) + 1).requires_grad_()
        reference_prediction = prediction.detach().clone().requires_grad_()
        teacher = (prediction.detach() * 1.7 + torch.rand_like(prediction) + 0.3).requires_grad_()
        loss = self.make_loss().ctx_depth_loss(prediction, None, 0.7, teacher)
        p = reference_prediction.flatten(0, 1).squeeze(-1)
        t = teacher.detach().flatten(0, 1).squeeze(-1)
        expected = 0.7 * F.l1_loss(p.log(), t.log())
        torch.testing.assert_close(loss, expected)
        loss.backward()
        expected.backward()
        torch.testing.assert_close(prediction.grad, reference_prediction.grad)
        self.assertIsNone(teacher.grad)

    def test_scale_and_shift_errors_are_penalized_without_alignment(self):
        prediction = torch.linspace(1, 4, 20).reshape(1, 1, 4, 5, 1).requires_grad_()
        loss = self.make_loss()
        scale_error = loss.ctx_depth_loss(prediction, None, 1., prediction.detach() * 3.)
        shift_error = loss.ctx_depth_loss(prediction, None, 1., prediction.detach() + 7.)
        torch.testing.assert_close(scale_error, torch.tensor(3.).log())
        self.assertGreater(shift_error.item(), 0.5)
        scale_error.backward()
        # Positive teacher scale error must push student depth upwards.
        self.assertTrue((prediction.grad < 0).all())

    def test_invalid_teacher_is_excluded_from_log_and_reduction(self):
        prediction = torch.tensor([1., 2., 4., 500., 600., 700., 800.], requires_grad=True)
        teacher = torch.tensor([1., 3., 2., float("nan"), float("inf"), 0., -1.])
        loss = LossDepth._masked_log_depth_l1(prediction.reshape(1, 1, -1), teacher.reshape(1, 1, -1))
        valid_p = prediction[:3].reshape(1, 1, 3)
        valid_t = teacher[:3].reshape(1, 1, 3)
        expected = F.l1_loss(valid_p.log(), valid_t.log())
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        torch.testing.assert_close(prediction.grad[3:], torch.zeros(4))

    def test_tiny_depth_is_numerically_safe_and_equal_depth_has_zero_loss(self):
        prediction = torch.tensor([0., 1e-12, 1., 4.], requires_grad=True)
        target = torch.tensor([1e-12, 1e-12, 1., 4.])
        loss = LossDepth._masked_log_depth_l1(prediction, target)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_empty_teacher_keeps_student_in_backward_graph(self):
        student = nn.Conv2d(3, 1, 1)
        prediction = student(torch.ones(2, 3, 4, 4)).permute(0, 2, 3, 1).unsqueeze(1)
        teacher = torch.full_like(prediction, float("nan"))
        loss = self.make_loss().ctx_depth_loss(prediction, None, teacher_depth=teacher)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        for parameter in student.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_missing_or_misordered_shape_fails_without_falling_back_to_dav3(self):
        loss = self.make_loss()
        prediction = torch.ones(2, 3, 4, 5, 1)
        with self.assertRaisesRegex(RuntimeError, "requires source teacher_depth"):
            loss.ctx_depth_loss(prediction, None)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            loss.ctx_depth_loss(prediction, None, teacher_depth=prediction[:, :2])

    def test_dav3_uses_sequence_scale_only_log_alignment(self):
        loss = self.make_loss()
        loss.cfg.teacher = "dav3"
        loss.cfg.alignment = "sequence_scale_only_log"
        prediction = torch.rand(2, 3, 4, 5, 1) + 1.0
        target = torch.rand(6, 4, 5) + 1.0
        with patch.object(loss, "_context_depth_target", return_value=target) as teacher:
            actual = loss.ctx_depth_loss(prediction, None, 0.6)
        teacher.assert_called_once()
        expected = 0.6 * LossDepth._sequence_scale_only_log_l1(
            prediction.squeeze(-1), target.reshape(2, 3, 4, 5)
        )
        torch.testing.assert_close(actual, expected)

    def test_dav3_alignment_is_sequence_level_not_per_view(self):
        target = torch.ones(1, 2, 2, 2)
        globally_scaled = target * 3.0
        per_view_scaled = target.clone()
        per_view_scaled[:, 0] *= 2.0
        per_view_scaled[:, 1] *= 3.0

        global_loss = LossDepth._sequence_scale_only_log_l1(
            globally_scaled, target
        )
        per_view_loss = LossDepth._sequence_scale_only_log_l1(
            per_view_scaled, target
        )

        self.assertLess(global_loss.item(), 1e-6)
        self.assertGreater(per_view_loss.item(), 0.1)

    def test_dav3_gt_metrics_remove_one_sequence_scale(self):
        teacher = torch.ones(1, 2, 2, 2) * 3.0
        gt = torch.ones_like(teacher)
        metrics = LossDepth._teacher_gt_metrics(teacher, gt)

        self.assertAlmostEqual(metrics["dav3_gt_scale_aligned"].item(), 1.0 / 3.0, places=5)
        self.assertLess(metrics["dav3_gt_abs_rel_aligned"].item(), 1e-6)
        self.assertLess(metrics["dav3_gt_rmse_aligned"].item(), 1e-6)
        self.assertLess(metrics["dav3_gt_log_l1_aligned"].item(), 1e-6)
        self.assertGreater(metrics["dav3_gt_abs_rel_raw"].item(), 1.9)

    def test_dav3_gt_metrics_keep_cross_view_scale_error(self):
        gt = torch.ones(1, 2, 2, 2)
        teacher = gt.clone()
        teacher[:, 0] *= 2.0
        teacher[:, 1] *= 4.0
        metrics = LossDepth._teacher_gt_metrics(teacher, gt)

        self.assertGreater(metrics["dav3_gt_abs_rel_aligned"].item(), 0.1)
        self.assertGreater(metrics["dav3_gt_log_l1_aligned"].item(), 0.1)

    def test_dav3_gt_metrics_ignore_invalid_pixels(self):
        teacher = torch.tensor([[[[2.0, float("nan")], [0.0, 4.0]]]])
        gt = torch.tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        metrics = LossDepth._teacher_gt_metrics(teacher, gt)

        self.assertAlmostEqual(metrics["dav3_gt_valid_ratio"].item(), 0.5, places=6)
        self.assertTrue(torch.isfinite(torch.stack(list(metrics.values()))).all())


if __name__ == "__main__":
    unittest.main()
