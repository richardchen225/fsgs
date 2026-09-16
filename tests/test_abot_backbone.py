"""CPU contracts for ABot integration, without importing CUDA model packages."""

import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import types
import unittest

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


@unittest.skipUnless(os.environ.get("ABOT_RECON_CHECKPOINT") and torch.cuda.is_available(),
                     "requires official ABot weights, package, and CUDA")
class ABotRealNetworkTests(unittest.TestCase):
    def test_official_camera_parity_prefix_batch_and_head_backward(self):
        # Avoid constructing a second 1B model: official camera-only inference
        # uses the exact same pretrained network retained by our adapter.
        from abot_recon.modeling.streaming.core.cache_utils import detach_carry

        adapter = ABotBackboneAdapter(os.environ["ABOT_RECON_CHECKPOINT"]).cuda()
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
        head = nn.Linear(2048, 2).cuda()
        sum(head(feature.float()).square().mean() for feature in features).backward()
        self.assertTrue(torch.isfinite(head.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
