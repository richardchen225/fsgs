"""CPU contracts for ABot integration, without importing CUDA model packages."""

import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.misc.backbone_checkpoint import backbone_signature, validate_backbone_checkpoint


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "abot_adapter_test", ROOT / "src/model/encoder/abot_adapter.py"
)
adapter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_module)
ABotBackboneAdapter = adapter_module.ABotBackboneAdapter


class LocalBlock(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(float(offset)))

    def forward(self, x):
        return x + self.offset


class CameraDecoder(nn.Module):
    def forward(self, x, xpos=None):
        return x[..., :1024]


class FakeNetwork(nn.Module):
    """Global blocks bypass hooks, just like official packed SDPA decode."""

    patch_size = 14
    patch_start_idx = 5
    dec_embed_dim = 1024

    def __init__(self):
        super().__init__()
        self.decoder = nn.ModuleList([LocalBlock(i + 1) for i in range(36)])
        self.camera_decoder = CameraDecoder()
        self.register_buffer("image_mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("image_std", torch.ones(1, 3, 1, 1))
        self.expected_pairs = []

    def encoder(self, frame, is_training=True):
        batch, _, h, w = frame.shape
        value = frame.mean(dim=(1, 2, 3)).reshape(batch, 1, 1)
        return {"x_norm_patchtokens": value.expand(batch, (h // 14) * (w // 14), 1024)}

    def decode(self, hidden, n, height, width, past_key_values=None, **kwargs):
        hidden = torch.cat([hidden[:, :1].expand(-1, 5, -1), hidden], dim=1)
        signal = hidden.mean(dim=(1, 2)).reshape(-1, 1, 1)
        history = torch.zeros_like(signal) if past_key_values is None else past_key_values
        expected = {}
        for index, block in enumerate(self.decoder):
            if index % 2 == 0:
                hidden = block(hidden)
                local = hidden
            else:
                hidden = hidden + block.offset + history
                expected[index] = torch.cat([local, hidden], dim=-1)
        self.expected_pairs.append(expected)
        return expected[35], None, history + signal

    def _postprocess_stream_carry(self, carry, spatial_hw):
        return carry

    def _predict_camera_poses(self, hidden, B, N, camera_state=None, **kwargs):
        x = hidden.mean(dim=(1, 2))
        x = x if camera_state is None else x + camera_state
        camera = torch.eye(4).expand(B, N, 4, 4).clone()
        camera[:, 0, 0, 3] = x
        return camera, x


def make_adapter():
    model = ABotBackboneAdapter.__new__(ABotBackboneAdapter)
    nn.Module.__init__(model)
    model.network = FakeNetwork()
    model.feature_layers = (7, 17, 25, 35)
    model._validate_layers()
    model.requires_grad_(False)
    model.eval()
    return model


class PointHead(nn.Module):
    def forward(self, features, image_shape):
        height, width = image_shape
        log_z = features[0][..., 0].mean(1) / 1000
        xyz = log_z[:, None, None, None].expand(-1, height, width, 3).clone()
        xyz[..., :2] = -100
        return xyz


def make_teacher_adapter():
    model = make_adapter()
    model.network.point_decoder = CameraDecoder()
    model.network.point_head = PointHead()
    model.network.point_z_log_max = 10.0
    model._depth_from_log_z = lambda z, cap: torch.exp(z.clamp(max=cap))
    model.requires_grad_(False)
    return model


class ABotDepthTeacherTests(unittest.TestCase):
    def test_source_only_z_export_does_not_change_features_or_cameras(self):
        model = make_teacher_adapter()
        images = torch.rand(2, 5, 3, 14, 14, requires_grad=True)
        plain = model(images, 3)
        with patch.object(model.network.point_decoder, "forward", wraps=model.network.point_decoder.forward) as decoder:
            features, start, cameras, depth = model(images, 3, return_depth=True)
        self.assertEqual(decoder.call_count, 3)
        self.assertEqual(depth.shape, (2, 3, 14, 14, 1))
        self.assertFalse(depth.requires_grad)
        self.assertTrue((depth > 0).all())
        self.assertEqual(start, plain[1])
        torch.testing.assert_close(cameras, plain[2], rtol=0, atol=0)
        for feature, original in zip(features, plain[0]):
            torch.testing.assert_close(feature, original, rtol=0, atol=0)
        expected = model.network.expected_pairs[-5][35][..., :1024][:, 5:, 0].mean(1) / 1000
        torch.testing.assert_close(depth[:, 0, 0, 0, 0], expected.exp())
        with patch.object(model.network.point_head, "forward", side_effect=AssertionError("teacher ran at test")):
            model(images, 3)
        with self.assertRaisesRegex(ValueError, "enable_depth_teacher"):
            make_adapter()(images, 3, return_depth=True)

    def test_two_pass_teacher_restores_source_order_and_ignores_targets(self):
        model = make_teacher_adapter()
        images = torch.rand(2, 5, 3, 14, 14)
        ids = torch.tensor([[6, 0, 4, 1, 3], [8, 2, 5, 7, 1]])
        with patch.object(model.network.point_decoder, "forward", wraps=model.network.point_decoder.forward) as decoder:
            features, _, cameras, _, depth = model.forward_two_pass(images, 3, ids, return_depth=True)
        self.assertEqual(decoder.call_count, 3)  # No teacher in the full-clip pass.
        changed = images.clone()
        changed[:, 3:] = 50
        other = model.forward_two_pass(changed, 3, ids, return_depth=True)
        torch.testing.assert_close(depth, other[-1], rtol=0, atol=0)
        for b in range(2):
            order = ids[b, :3].argsort()
            source = model(images[b:b+1, :3][:, order], 3, return_depth=True)
            torch.testing.assert_close(depth[b:b+1], source[-1][:, order.argsort()])
            single = model.forward_two_pass(images[b:b+1], 3, ids[b:b+1], return_depth=True)
            torch.testing.assert_close(single[-1], depth[b:b+1])
            torch.testing.assert_close(single[2], cameras[b:b+1])
        # With no targets, the same source Z is returned without a second pass.
        source_only = model.forward_two_pass(images[:, :3], 3, ids[:, :3], return_depth=True)
        torch.testing.assert_close(source_only[-1], depth)
        head = nn.Linear(2048, 1)
        sum(head(feature).square().mean() for feature in features).backward()
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))


class ABotAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = make_adapter()
        self.images = torch.rand(2, 5, 3, 28, 28)

    def test_four_pairs_equal_actual_intermediate_computation(self):
        features, start, cameras = self.model(self.images, 3)
        self.assertEqual(start, 5)
        self.assertEqual(cameras.shape, (2, 5, 4, 4))
        for index, layer in enumerate(self.model.feature_layers):
            self.assertEqual(features[index].shape, (2, 3, 9, 2048))
            expected = torch.stack([
                value[layer] for value in self.model.network.expected_pairs[:3]
            ], dim=1)
            torch.testing.assert_close(features[index], expected)
        for block in self.model.network.decoder:
            self.assertFalse(block._forward_hooks)
            self.assertFalse(block._forward_pre_hooks)

    def test_target_append_cannot_change_source(self):
        full_features, _, full_camera = self.model(self.images, 3)
        source_features, _, source_camera = self.model(self.images[:, :3], 3)
        changed = self.images.clone()
        changed[:, 3:] = 20
        changed_features, _, _ = self.model(changed, 3)
        torch.testing.assert_close(full_camera[:, :3], source_camera)
        for full, source, other in zip(full_features, source_features, changed_features):
            torch.testing.assert_close(full, source)
            torch.testing.assert_close(full, other)

    def test_batch_and_scene_state_are_independent(self):
        features, _, cameras = self.model(self.images, 3)
        for batch_index in range(2):
            single_features, _, single_camera = self.model(self.images[batch_index:batch_index + 1], 3)
            torch.testing.assert_close(cameras[batch_index:batch_index + 1], single_camera)
            for full, single in zip(features, single_features):
                torch.testing.assert_close(full[batch_index:batch_index + 1], single)

    def test_downstream_backward_and_frozen_eval(self):
        self.model.train()
        self.assertFalse(self.model.training)
        self.assertFalse(self.model.network.training)
        features, _, cameras = self.model(self.images, 3)
        heads = nn.ModuleList([nn.Linear(2048, 2) for _ in features])
        loss = sum(head(feature).square().mean() for head, feature in zip(heads, features))
        loss.backward()
        for parameter in heads.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertFalse(cameras.requires_grad)

    def test_existing_dpt_depth_and_gs_feature_backward(self):
        # Load the real DPT head without encoder/__init__ importing CUDA-only
        # rasterizers. Only its package import is substituted, not its code.
        package = types.ModuleType("_abot_test_dpt")
        package.__path__ = [str(ROOT / "src/model/encoder/vggt/heads")]
        sys.modules[package.__name__] = package
        dpt = importlib.import_module("_abot_test_dpt.dpt_head")
        source = ROOT / "src/model/encoder/heads/vggt_dpt_gs_head.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        head_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        from typing import List
        namespace = {"DPTHead": dpt.DPTHead, "torch": torch, "nn": nn,
                     "F": torch.nn.functional, "List": List}
        exec(compile(ast.Module(body=[head_class], type_ignores=[]), str(source), "exec"), namespace)
        head = namespace["VGGT_DPT_GS_Head"](
            dim_in=2048, patch_size=(14, 14), output_dim=2, features=256,
        )
        features, start, _ = self.model(self.images, 3)
        # Fake blocks intentionally generate large values; the real head's
        # LayerNorm makes this a useful check of the actual DPT input contract.
        out, depth, confidence, _ = head(
            features, self.images[:, :3], patch_start_idx=start, image_size=(28, 28)
        )
        self.assertEqual(out.shape, (2, 3, 128, 28, 28))
        self.assertEqual(depth.shape, (2, 3, 28, 28, 1))
        loss = out.square().mean() + depth.square().mean() + confidence.square().mean()
        loss.backward()
        for name, parameter in head.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertIsNotNone(head.projects[0].weight.grad)
        self.assertIsNotNone(head.scratch.output_conv2[-1].weight.grad)

    def test_invalid_layers_fail_and_hook_cleanup_on_error(self):
        self.model.feature_layers = (4, 11, 17, 23)
        with self.assertRaises(ValueError):
            self.model._validate_layers()
        self.model.feature_layers = (7, 17, 25, 35)
        def fail(*args, **kwargs):
            raise RuntimeError("decode failed")
        self.model.network.decode = fail
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            self.model(self.images, 3)
        self.assertTrue(all(not block._forward_hooks and not block._forward_pre_hooks
                            for block in self.model.network.decoder))


class ABotTwoPassTests(unittest.TestCase):
    def test_known_sim3_noncollinear_collinear_and_stationary(self):
        align = adapter_module.align_source_cameras
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        moving = torch.eye(4).repeat(3, 5, 1, 1)
        moving[0, :, :3, 3] = torch.tensor([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [2., 3., 4.]
        ])
        moving[1, :, 0, 3] = torch.arange(5.)
        scales = torch.tensor([2.5, 0.2, 1.])
        reference = moving.clone()
        reference[..., :3, :3] = rotation @ moving[..., :3, :3]
        reference[..., :3, 3] = (
            scales[:, None, None] * (moving[..., :3, 3] @ rotation.T)
            + torch.tensor([5., -2., 3.])
        )
        # The last camera is held out of the fit.
        with torch.autocast("cpu", dtype=torch.bfloat16):
            aligned, diagnostics = align(reference[:, :4], moving[:, :4], moving)
        torch.testing.assert_close(aligned, reference, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(diagnostics["scale"], scales)
        torch.testing.assert_close(diagnostics["orientation_fallback"], torch.tensor([0., 1., 1.]))
        torch.testing.assert_close(diagnostics["scale_fallback"], torch.tensor([0., 0., 1.]))
        torch.testing.assert_close(torch.linalg.det(aligned[..., :3, :3]), torch.ones(3, 5))
        # One source has no baseline: scale=1, orientation and translation work.
        aligned, diagnostics = align(reference[2:, :1], moving[2:, :1], moving[2:])
        torch.testing.assert_close(aligned, reference[2:])
        self.assertEqual(diagnostics["scale_fallback"].item(), 1)
        bad = moving.clone()
        bad[0, 0, 0, 3] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            align(reference, bad, moving)

    def test_sort_restore_target_alignment_and_camera_only_second_pass(self):
        model = make_adapter()
        ids = torch.tensor([[6, 0, 4, 1, 3], [8, 2, 5, 7, 1]])
        images = ids[:, :, None, None, None].float().expand(-1, -1, 3, 14, 14)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        scale = torch.tensor([2., 3.])
        translation = torch.tensor([[2., 3., 1.], [-1., 2., 5.]])

        def canonical(frame_ids):
            poses = torch.eye(4).repeat(2, frame_ids.shape[1], 1, 1)
            poses[..., 0, 3] = frame_ids
            poses[..., 1, 3] = frame_ids.square()
            return poses

        calls = []
        def stream(frames, num_feature_views):
            frame_ids = frames[:, :, 0, 0, 0]
            calls.append((frame_ids, num_feature_views))
            poses = canonical(frame_ids)
            if num_feature_views:
                features = [frame_ids[:, :, None, None].clone() for _ in range(4)]
            else:
                features = []
                poses[..., :3, :3] = rotation.T
                poses[..., :3, 3] = (
                    (poses[..., :3, 3] - translation[:, None]) @ rotation
                ) / scale[:, None, None]
            return features, 5, poses

        with patch.object(model, "forward", side_effect=stream):
            features, start, poses, diagnostics = model.forward_two_pass(images, 3, ids)
        self.assertEqual([call[1] for call in calls], [3, 0])
        torch.testing.assert_close(calls[0][0], ids[:, :3].sort(1).values.float())
        torch.testing.assert_close(calls[1][0], ids.sort(1).values.float())
        self.assertEqual(start, 5)
        for feature in features:
            torch.testing.assert_close(feature[:, :, 0, 0], ids[:, :3].float())
        torch.testing.assert_close(poses, canonical(ids.float()), atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(diagnostics["scale"], scale)

    def test_targets_cannot_change_source_and_batch_state_is_independent(self):
        model = make_adapter()
        torch.manual_seed(11)
        images = torch.rand(2, 5, 3, 14, 14, requires_grad=True)
        ids = torch.tensor([[6, 0, 4, 1, 3], [8, 2, 5, 7, 1]])
        features, _, cameras, _ = model.forward_two_pass(images, 3, ids)
        changed = images.detach().clone()
        changed[:, 3:] = 20
        other_features, _, other_cameras, _ = model.forward_two_pass(changed, 3, ids)
        torch.testing.assert_close(cameras[:, :3], other_cameras[:, :3], rtol=0, atol=0)
        for feature, other in zip(features, other_features):
            torch.testing.assert_close(feature, other, rtol=0, atol=0)
        for b in range(2):
            single_features, _, single_cameras, _ = model.forward_two_pass(
                images[b:b+1], 3, ids[b:b+1]
            )
            torch.testing.assert_close(cameras[b:b+1], single_cameras)
            for feature, single in zip(features, single_features):
                torch.testing.assert_close(feature[b:b+1], single)
        head = nn.Linear(2048, 2)
        sum(head(feature).square().mean() for feature in features).backward()
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        self.assertIsNone(images.grad)
        self.assertFalse(cameras.requires_grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_no_targets_missing_indices_and_zero_feature_pass(self):
        model = make_adapter()
        images = torch.rand(1, 3, 3, 14, 14)
        ids = torch.tensor([0, 2, 4])
        with patch.object(model, "forward", wraps=model.forward) as stream:
            features, _, cameras, diagnostics = model.forward_two_pass(images, 3, ids)
        self.assertEqual(stream.call_count, 1)
        self.assertEqual(diagnostics, {})
        plain_features, _, plain_cameras = model(images, 3)
        torch.testing.assert_close(cameras, plain_cameras)
        for feature, plain in zip(features, plain_features):
            torch.testing.assert_close(feature, plain)
        empty, _, camera_only = model(images, 0)
        self.assertEqual(empty, [])
        torch.testing.assert_close(cameras, camera_only)
        with self.assertRaisesRegex(ValueError, "real frame indices"):
            model.forward_two_pass(images, 2, None)
        with self.assertRaisesRegex(ValueError, "frame indices"):
            model.forward_two_pass(images, 2, ids[:2])


class BackboneCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.abot = SimpleNamespace(reconstruction_backbone="abot", abot_feature_layers=[7, 17, 25, 35])
        self.zipmap = SimpleNamespace(reconstruction_backbone="zipmap")

    def test_matching_and_legacy_checkpoints(self):
        validate_backbone_checkpoint({"reconstruction_backbone": backbone_signature(self.abot)}, self.abot)
        validate_backbone_checkpoint({"state_dict": {"model.encoder.aggregator.aggregator.weight": 1}}, self.zipmap)
        validate_backbone_checkpoint({"gs_head.weight": 1}, self.abot, allow_transfer=True)

    def test_test_and_resume_reject_wrong_backbone_or_feature_layers(self):
        checkpoint = {"reconstruction_backbone": backbone_signature(self.zipmap)}
        with self.assertRaises(RuntimeError):
            validate_backbone_checkpoint(checkpoint, self.abot)
        checkpoint = {"reconstruction_backbone": {"name": "abot", "feature_layers": [1, 3, 5, 35]}}
        with self.assertRaises(RuntimeError):
            validate_backbone_checkpoint(checkpoint, self.abot)
        with self.assertRaises(RuntimeError):
            validate_backbone_checkpoint({"encoder.gs_head.weight": 1}, self.abot)

    def test_pose_protocol_metadata_and_legacy_warning(self):
        self.assertEqual(backbone_signature(self.abot)["pose_mode"], "two_pass")
        checkpoint = {"reconstruction_backbone": {
            "name": "abot", "feature_layers": [7, 17, 25, 35]
        }}
        with patch("builtins.print") as output:
            validate_backbone_checkpoint(checkpoint, self.abot)
        self.assertIn("camera protocol changed", output.call_args[0][0])


@unittest.skipUnless(os.environ.get("ABOT_RECON_CHECKPOINT") and torch.cuda.is_available(),
                     "requires official ABot weights, package, and CUDA")
class ABotRealNetworkTests(unittest.TestCase):
    def test_official_camera_parity_prefix_batch_and_head_backward(self):
        # Avoid constructing a second 1B model: official camera-only inference
        # uses the exact same pretrained network retained by our adapter.
        from abot_recon.modeling.streaming.core.cache_utils import detach_carry

        adapter = ABotBackboneAdapter(os.environ["ABOT_RECON_CHECKPOINT"], enable_depth_teacher=True).cuda()
        torch.manual_seed(3)
        images = torch.rand(2, 4, 3, 56, 56, device="cuda")
        features, _, cameras = adapter(images, 2)
        prefix_features, _, prefix_cameras = adapter(images[:, :2], 2)
        torch.testing.assert_close(cameras[:, :2], prefix_cameras)
        for full, prefix in zip(features, prefix_features):
            torch.testing.assert_close(full, prefix)
        for index in range(2):
            _, _, single = adapter(images[index:index + 1], 2)
            torch.testing.assert_close(cameras[index:index + 1], single, atol=5e-3, rtol=5e-3)

        # The official camera-only helper supports B=1 even on its SDPA path.
        carry = camera_state = None
        reference = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(images.shape[1]):
                pred = adapter.network._forward_frame_camera_only(
                    images[:1, i:i + 1], past_key_values=carry,
                    camera_state=camera_state, frame_idx=i, ref_hidden=None,
                    causal_global_attn=True, use_paged=False,
                )
                carry = adapter.network._postprocess_stream_carry(
                    detach_carry(pred["past_key_values"]), spatial_hw=(56, 56)
                )
                camera_state = pred["camera_state"]
                reference.append(pred["camera_poses"])
        torch.testing.assert_close(cameras[:1], torch.cat(reference, dim=1), atol=5e-3, rtol=5e-3)
        indices = torch.tensor([[0, 4, 1, 3], [0, 6, 2, 4]], device="cuda")
        two_features, _, two_cameras, diagnostics = adapter.forward_two_pass(images, 2, indices)
        torch.testing.assert_close(two_cameras[:, :2], prefix_cameras)
        for feature, prefix in zip(two_features, prefix_features):
            torch.testing.assert_close(feature, prefix)
        self.assertTrue(torch.isfinite(two_cameras).all())
        self.assertTrue(all(torch.isfinite(value).all() for value in diagnostics.values()))
        teacher_output = adapter.forward_two_pass(images, 2, indices, return_depth=True)
        torch.testing.assert_close(teacher_output[2], two_cameras)
        self.assertEqual(teacher_output[-1].shape, (2, 2, 56, 56, 1))
        self.assertFalse(teacher_output[-1].requires_grad)
        source_teacher = adapter(images[:, :2], 2, return_depth=True)[-1]
        torch.testing.assert_close(teacher_output[-1], source_teacher)
        self.assertTrue(torch.isfinite(source_teacher).all())
        self.assertTrue((source_teacher > 0).all())
        head = nn.Linear(2048, 2).cuda()
        sum(head(feature.float()).square().mean() for feature in two_features).backward()
        self.assertTrue(torch.isfinite(head.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
