"""Exercise the StreamVGGT integration without allocating its full backbone."""

import ast
from contextlib import contextmanager
import importlib
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from src.misc.backbone_checkpoint import validate_backbone_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def load_initializer():
    source = ROOT / "src/model/encoder/anysplat.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    encoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EncoderAnySplat")
    method = next(n for n in encoder.body if isinstance(n, ast.FunctionDef) and n.name == "_init_streamvggt")
    namespace = {
        "EncoderAnySplatCfg": SimpleNamespace,
        "Path": Path,
        "torch": torch,
        "load_file": load_file,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


INITIALIZE = load_initializer()


@contextmanager
def stream_modules():
    # Bypass the encoder package's eager imports of unrelated CUDA extensions.
    with patch.dict(sys.modules):
        if "src.model.encoder" not in sys.modules:
            encoder = ModuleType("src.model.encoder")
            encoder.__path__ = [str(ROOT / "src/model/encoder")]
            sys.modules[encoder.__name__] = encoder
        aggregator = importlib.import_module("src.model.encoder.streamvggt.models.aggregator")
        camera = importlib.import_module("src.model.encoder.streamvggt.heads.camera_head")
        depth = importlib.import_module("src.model.encoder.streamvggt.heads.dpt_head")
        yield aggregator, camera, depth


class StreamVGGTBackboneTests(unittest.TestCase):
    def test_overfit_uses_native_depth_and_world_mirror_features(self):
        source = ROOT / "src/experiments/overfit_old_gs_residual.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        method = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_extract_features"
        )
        namespace = {"torch": torch}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            namespace,
        )

        class GaussianFeatures(nn.Module):
            def forward(self, tokens, images, **kwargs):
                self.predict_depth = kwargs["predict_depth"]
                shape = (images.shape[0], images.shape[1], 4, *images.shape[-2:])
                features = images.new_full(shape, 3.0)
                wm_depth = images.new_full((*images.shape[:2], *images.shape[-2:], 1), 99.0)
                return features, wm_depth, wm_depth, features

        class NativeDepth(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, tokens, images, patch_start_idx):
                self.calls += 1
                depth = images.new_full((*images.shape[:2], *images.shape[-2:], 1), 7.0)
                confidence = images.new_full((*images.shape[:2], *images.shape[-2:]), 2.0)
                return depth, confidence

        gaussian = GaussianFeatures()
        native_depth = NativeDepth()
        model = SimpleNamespace(
            encoder=SimpleNamespace(
                cfg=SimpleNamespace(depth_head_source="native"),
                gaussian_param_head=gaussian,
                depth_head=native_depth,
            )
        )
        images = torch.zeros(1, 2, 3, 4, 6)
        features, depth, confidence = namespace[method.name](
            model, [], 5, images, (4, 6), False
        )
        self.assertFalse(gaussian.predict_depth)
        torch.testing.assert_close(features, torch.full_like(features, 3.0))
        torch.testing.assert_close(depth, torch.full_like(depth, 7.0))
        torch.testing.assert_close(confidence, torch.full_like(confidence, 2.0))
        self.assertEqual(native_depth.calls, 1)

        model.encoder.cfg.depth_head_source = "world_mirror"
        _, depth, confidence = namespace[method.name](
            model, [], 5, images, (4, 6), False
        )
        self.assertTrue(gaussian.predict_depth)
        torch.testing.assert_close(depth, torch.full_like(depth, 99.0))
        torch.testing.assert_close(confidence, torch.full_like(confidence, 99.0))
        self.assertEqual(native_depth.calls, 1)

    def test_overfit_shares_intrinsics_and_routes_focal_gradients_to_first_frame(self):
        source = ROOT / "src/experiments/overfit_old_gs_residual.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        methods = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {
            "_as_homogeneous_transform", "_predict_camera_geometry",
        }]
        with stream_modules():
            pose_utils = importlib.import_module("src.model.encoder.vggt.utils.pose_enc")
        namespace = {
            "torch": torch, "math": math,
            "pose_encoding_to_extri_intri": pose_utils.pose_encoding_to_extri_intri,
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
        predict = namespace["_predict_camera_geometry"]

        class Camera(nn.Module):
            def __init__(self):
                super().__init__()
                poses = torch.zeros(2, 3, 9)
                poses[..., 6] = 1.0
                poses[..., 0] = torch.tensor([0.0, 0.1, 0.2])
                poses[..., 7:9] = torch.tensor([1.0, 1.2, 1.4])[None, :, None]
                poses[1, :, 7:9] += 0.1
                self.poses = nn.Parameter(poses)

            def forward(self, tokens):
                if tokens[-1].dtype != torch.float32:
                    raise AssertionError("Camera features must be float32")
                return [self.poses]

        for backbone in ("zipmap", "streamvggt"):
            with self.subTest(backbone=backbone):
                cfg = SimpleNamespace(reconstruction_backbone=backbone, share_first_frame_intrinsics=False)
                camera = Camera()
                model = SimpleNamespace(encoder=SimpleNamespace(cfg=cfg, camera_head=camera))
                dtype = torch.float16 if backbone == "streamvggt" else torch.float32
                tokens = [torch.zeros(2, 3, 5, 64, dtype=dtype)]
                gt_c2w = torch.eye(4).expand(2, 3, 4, 4)
                independent = predict(model, tokens, (28, 56), gt_c2w)
                self.assertFalse(torch.equal(
                    independent["intrinsic_px"][:, 0], independent["intrinsic_px"][:, 1]
                ))
                cfg.share_first_frame_intrinsics = True
                shared = predict(model, tokens, (28, 56), gt_c2w)
                torch.testing.assert_close(shared["c2w"], independent["c2w"])
                torch.testing.assert_close(
                    shared["pose_encoding"][..., :7],
                    independent["pose_encoding"][..., :7],
                )
                torch.testing.assert_close(
                    shared["pose_encoding"][..., 7:9],
                    independent["pose_encoding"][:, :1, 7:9].expand(2, 3, 2),
                )
                torch.testing.assert_close(
                    shared["raw_pose_encoding"], independent["pose_encoding"]
                )
                torch.testing.assert_close(
                    shared["intrinsic_px"], independent["intrinsic_px"][:, :1].expand(2, 3, 3, 3)
                )
                self.assertFalse(torch.equal(shared["intrinsic_px"][0], shared["intrinsic_px"][1]))
                shared["intrinsic_px"].sum().backward()
                gradient = camera.poses.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue((gradient[:, 0, 7:9].abs() > 0).all())
                torch.testing.assert_close(gradient[:, 1:, 7:9], torch.zeros_like(gradient[:, 1:, 7:9]))

    def test_aggregator_is_causal_and_camera_head_has_finite_gradients(self):
        with stream_modules() as (aggregator_module, camera_module, _):
            torch.manual_seed(42)
            aggregator = aggregator_module.Aggregator(
                img_size=28, patch_size=14, embed_dim=32,
                depth=2, num_heads=4, patch_embed="conv",
            ).eval()
            camera = camera_module.CameraHead(dim_in=64, trunk_depth=1, num_heads=4)
            images = torch.rand(2, 3, 3, 28, 28)
            changed = images.clone()
            changed[:, -1] = 1.0 - changed[:, -1]
            with torch.no_grad():
                tokens, patch_start = aggregator(images)
                other_tokens, _ = aggregator(changed)
            self.assertEqual(tokens[-1].shape, (2, 3, 9, 64))
            self.assertEqual(patch_start, 5)
            torch.testing.assert_close(tokens[-1][:, :2], other_tokens[-1][:, :2])
            self.assertFalse(torch.allclose(tokens[-1][:, -1], other_tokens[-1][:, -1]))
            poses = camera([tokens[-1].float()])
            other_poses = camera([other_tokens[-1].float()])
            self.assertEqual(poses[-1].shape, (2, 3, 9))
            torch.testing.assert_close(poses[-1][:, :2], other_poses[-1][:, :2])
            poses[-1].square().mean().backward()
            gradient = camera.pose_branch.fc2.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_pretrained_weights_load_from_raw_and_wrapped_checkpoints(self):
        for extension, wrapper, prefix in (
            (".safetensors", None, ""),
            (".pt", "model", "module."),
            (".pt", "state_dict", "model.encoder."),
        ):
            with self.subTest(extension=extension, wrapper=wrapper):
                with tempfile.TemporaryDirectory() as folder, stream_modules() as (agg_module, cam_module, depth_module):
                    weights = {
                        prefix + name + "." + key: value.detach().clone()
                        for name in ("aggregator", "camera_head", "depth_head")
                        for key, value in nn.Linear(2, 2).state_dict().items()
                    }
                    path = Path(folder) / ("pretrained" + extension)
                    if extension == ".safetensors":
                        save_file(weights, path)
                    else:
                        torch.save({wrapper: weights}, path)
                    model = nn.Module()
                    with patch.object(agg_module, "Aggregator", side_effect=lambda: nn.Linear(2, 2)), \
                         patch.object(cam_module, "CameraHead", side_effect=lambda **kw: nn.Linear(2, 2)), \
                         patch.object(depth_module, "DPTHead", side_effect=lambda **kw: nn.Linear(2, 2)):
                        INITIALIZE(model, SimpleNamespace(streamvggt_weights_path=str(path)))
                    for name in ("aggregator", "camera_head", "depth_head"):
                        for key, value in getattr(model, name).state_dict().items():
                            torch.testing.assert_close(value, weights[prefix + name + "." + key])

    def test_missing_camera_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder, stream_modules() as (agg_module, cam_module, depth_module):
            path = Path(folder) / "incomplete.pt"
            torch.save({"aggregator.weight": torch.ones(2, 2), "aggregator.bias": torch.zeros(2)}, path)
            with patch.object(agg_module, "Aggregator", side_effect=lambda: nn.Linear(2, 2)), \
                 patch.object(cam_module, "CameraHead", side_effect=lambda **kw: nn.Linear(2, 2)), \
                 patch.object(depth_module, "DPTHead", side_effect=lambda **kw: nn.Linear(2, 2)):
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    INITIALIZE(nn.Module(), SimpleNamespace(streamvggt_weights_path=str(path)))

    def test_missing_native_depth_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder, stream_modules() as (agg_module, cam_module, depth_module):
            path = Path(folder) / "no_depth.pt"
            weights = {
                name + "." + key: value
                for name in ("aggregator", "camera_head")
                for key, value in nn.Linear(2, 2).state_dict().items()
            }
            torch.save(weights, path)
            with patch.object(agg_module, "Aggregator", side_effect=lambda: nn.Linear(2, 2)), \
                 patch.object(cam_module, "CameraHead", side_effect=lambda **kw: nn.Linear(2, 2)), \
                 patch.object(depth_module, "DPTHead", side_effect=lambda **kw: nn.Linear(2, 2)):
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    INITIALIZE(nn.Module(), SimpleNamespace(streamvggt_weights_path=str(path)))

    def test_camera_is_trainable_while_feature_backbone_stays_frozen(self):
        source = ROOT / "src/model/model_wrapper.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelWrapper")
        method = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == "_configure_training_stage")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        for backbone in ("zipmap", "streamvggt"):
            with self.subTest(backbone=backbone):
                model = nn.Module()
                model.encoder = nn.Module()
                model.encoder.cfg = SimpleNamespace(reconstruction_backbone=backbone, gir_enabled=False)
                for name in ("aggregator", "camera_head", "gaussian_param_head", "gs_head", "depth_head"):
                    setattr(model.encoder, name, nn.Linear(2, 2))
                model.requires_grad_(False)
                stage = SimpleNamespace(
                    model=model,
                    optimizer_cfg=SimpleNamespace(train_base_heads=True, train_gir=False),
                    train_cfg=SimpleNamespace(weight_pose=1.0),
                )
                namespace[method.name](stage)
                self.assertTrue(all(p.requires_grad for p in model.encoder.camera_head.parameters()))
                self.assertTrue(all(p.requires_grad for p in model.encoder.depth_head.parameters()))
                self.assertFalse(any(p.requires_grad for p in model.encoder.aggregator.parameters()))

    def test_backbone_mismatch_is_rejected_for_test_and_allowed_for_transfer(self):
        stream_cfg = SimpleNamespace(reconstruction_backbone="streamvggt")
        zip_cfg = SimpleNamespace(reconstruction_backbone="zipmap")
        stream_checkpoint = {"state_dict": {
            "model.encoder.aggregator.camera_token": torch.zeros(1),
            "model.encoder.camera_head.trunk.0.weight": torch.zeros(1),
        }}
        validate_backbone_checkpoint(stream_checkpoint, stream_cfg)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_backbone_checkpoint(stream_checkpoint, zip_cfg)
        zip_checkpoint = {"reconstruction_backbone": {"name": "zipmap"}}
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_backbone_checkpoint(zip_checkpoint, stream_cfg)
        validate_backbone_checkpoint(zip_checkpoint, stream_cfg, allow_transfer=True)


if __name__ == "__main__":
    unittest.main()
