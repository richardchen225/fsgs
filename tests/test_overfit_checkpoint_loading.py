"""Check checkpoint ownership without importing CUDA-only experiment modules."""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from src.misc.backbone_checkpoint import backbone_signature, validate_backbone_checkpoint


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/experiments/overfit_old_gs_residual.py"


def load_helpers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    methods = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
        "_rename_wm_key", "_load_checkpoint", "_save_checkpoint",
    }]
    namespace = {
        "torch": torch, "Path": Path, "Any": Any, "load_file": load_file,
        "to_absolute_path": lambda path: str(Path(path).resolve()),
        "backbone_signature": backbone_signature,
        "validate_backbone_checkpoint": validate_backbone_checkpoint,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def make_model():
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.cfg = SimpleNamespace(reconstruction_backbone="streamvggt")
    model.encoder.aggregator = nn.Module()
    model.encoder.aggregator.frame_blocks = nn.ModuleList([nn.Linear(2, 2)])
    model.encoder.camera_head = nn.Module()
    model.encoder.camera_head.trunk = nn.ModuleList([nn.Linear(2, 2)])
    model.encoder.depth_head = nn.Linear(2, 2)
    model.encoder.gs_head = nn.Linear(2, 2)
    model.encoder.gaussian_param_head = nn.Linear(2, 2)
    model.gir_update_head = nn.Module()
    model.gir_update_head.prediction = nn.Linear(2, 2)
    model.gir_update_head.additional_historical_predictions = nn.ModuleList()
    model.gir_projected_update_head = None
    return model


class OverfitCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.helpers = load_helpers()

    def test_world_mirror_initialization_preserves_streamvggt_backbone_and_camera(self):
        model = make_model()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        state = {
            "visual_geometry_transformer.frame_blocks.0.weight": torch.full((2, 2), 7.0),
            "cam_head.refine_net.0.weight": torch.full((2, 2), 8.0),
            "gs_renderer.gs_head.weight": torch.full((2, 2), 9.0),
            "gs_head.weight": torch.full((2, 2), 10.0),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pre_wm.safetensors"
            save_file(state, path)
            result = self.helpers["_load_checkpoint"](model, str(path))
        self.assertIsNone(result)
        for key in before:
            if key.startswith((
                "encoder.aggregator.",
                "encoder.camera_head.",
                "encoder.depth_head.",
            )):
                torch.testing.assert_close(model.state_dict()[key], before[key])
        torch.testing.assert_close(model.encoder.gs_head.weight, state["gs_renderer.gs_head.weight"])
        torch.testing.assert_close(model.encoder.gaussian_param_head.weight, state["gs_head.weight"])

    def test_same_backbone_resume_restores_camera_and_keeps_pretrained_aggregator(self):
        model = make_model()
        before = model.encoder.aggregator.frame_blocks[0].weight.detach().clone()
        state = {
            "model.encoder.aggregator.frame_blocks.0.weight": torch.full((2, 2), 7.0),
            "model.encoder.camera_head.trunk.0.weight": torch.full((2, 2), 8.0),
            "model.encoder.depth_head.weight": torch.full((2, 2), 6.0),
            "model.encoder.gs_head.weight": torch.full((2, 2), 9.0),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "resume.ckpt"
            torch.save({"state_dict": state, "optimizer": {}, "step": 12,
                        "reconstruction_backbone": {"name": "streamvggt"}}, path)
            result = self.helpers["_load_checkpoint"](model, str(path))
        self.assertEqual(result["step"], 12)
        torch.testing.assert_close(model.encoder.aggregator.frame_blocks[0].weight, before)
        torch.testing.assert_close(model.encoder.camera_head.trunk[0].weight, torch.full((2, 2), 8.0))
        torch.testing.assert_close(model.encoder.depth_head.weight, torch.full((2, 2), 6.0))

    def test_cross_backbone_checkpoint_is_rejected_before_mutating_weights(self):
        for metadata in (True, False):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as folder:
                model = make_model()
                before = model.encoder.gs_head.weight.detach().clone()
                payload = {"state_dict": {
                    "encoder.gs_head.weight": torch.full((2, 2), 9.0),
                    "encoder.camera_head.camera_mlp_head.weight": torch.ones(2, 2),
                }}
                if metadata:
                    payload["reconstruction_backbone"] = {"name": "zipmap"}
                path = Path(folder) / "zipmap.ckpt"
                torch.save(payload, path)
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    self.helpers["_load_checkpoint"](model, str(path))
                torch.testing.assert_close(model.encoder.gs_head.weight, before)

    def test_empty_head_initialization_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrong.safetensors"
            save_file({"aggregator.weight": torch.ones(2, 2)}, path)
            with self.assertRaisesRegex(RuntimeError, "No compatible"):
                self.helpers["_load_checkpoint"](make_model(), str(path))

    def test_saved_overfit_checkpoint_records_backbone(self):
        stateful = SimpleNamespace(state_dict=lambda: {})
        with tempfile.TemporaryDirectory() as folder:
            self.helpers["_save_checkpoint"](
                make_model(), stateful, stateful, stateful, 5, Path(folder)
            )
            payload = torch.load(Path(folder) / "checkpoints/step_000005.ckpt", weights_only=True)
        self.assertEqual(payload["reconstruction_backbone"], {"name": "streamvggt"})
        self.assertIn("encoder.depth_head.weight", payload["state_dict"])


if __name__ == "__main__":
    unittest.main()
