import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest

import torch
from torch import nn

from src.misc.backbone_checkpoint import validate_backbone_checkpoint
ROOT = Path(__file__).resolve().parents[1]

# Avoid the encoder package's eager imports of optional CUDA extensions. The
# adapter itself only depends on the lightweight VGGT pose utilities.
if "src.model.encoder" not in sys.modules:
    encoder_package = ModuleType("src.model.encoder")
    encoder_package.__path__ = [str(ROOT / "src/model/encoder")]
    sys.modules[encoder_package.__name__] = encoder_package
abot_adapter = importlib.import_module("src.model.encoder.abot_adapter")
ABotBackboneAdapter = abot_adapter.ABotBackboneAdapter
ABotCameraHeadAdapter = abot_adapter.ABotCameraHeadAdapter
ABotDepthHeadAdapter = abot_adapter.ABotDepthHeadAdapter
rescale_local_points_to_depth = abot_adapter.rescale_local_points_to_depth
transform_local_points_to_world = abot_adapter.transform_local_points_to_world


class _FakeBlock(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.offset


class _FakeABotNetwork(nn.Module):
    patch_start_idx = 5

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.ModuleList(_FakeBlock(float(index)) for index in range(36))
        self.last_images = None

    def forward(self, images: torch.Tensor, causal_global_attn: bool = True):
        self.last_images = images.detach().clone()
        assert causal_global_attn
        batch, views, _channels, height, width = images.shape
        token_count = 5 + (height // 14) * (width // 14)
        local = images.new_zeros((batch * views, token_count, 1024))
        global_ = images.new_zeros((batch, views * token_count, 1024))
        for index, block in enumerate(self.decoder):
            if index % 2:
                global_ = block(global_)
            else:
                local = block(local)

        fx = 0.8
        fy = 0.9
        u = (
            (torch.arange(width, device=images.device, dtype=torch.float32) + 0.5)
            / width
        ).view(1, width).expand(height, width)
        v = (
            (torch.arange(height, device=images.device, dtype=torch.float32) + 0.5)
            / height
        ).view(height, 1).expand(height, width)
        z = torch.full((height, width), 2.0, device=images.device)
        points = torch.stack([(u - 0.5) * z / fx, (v - 0.5) * z / fy, z], dim=-1)
        points = points.view(1, 1, height, width, 3).expand(batch, views, -1, -1, -1)
        c2w = torch.eye(4, device=images.device).view(1, 1, 4, 4).repeat(batch, views, 1, 1)
        c2w[:, :, 0, 3] = torch.arange(views, device=images.device)
        confidence = torch.zeros((batch, views, height, width, 1), device=images.device)
        return {
            "camera_poses": c2w,
            "local_points": points,
            "conf": confidence,
        }


class ABotBackboneTests(unittest.TestCase):
    def test_adapter_contract_and_geometry(self):
        network = _FakeABotNetwork()
        backbone = ABotBackboneAdapter(network)
        camera = ABotCameraHeadAdapter(backbone)
        depth = ABotDepthHeadAdapter(backbone)

        normalized_images = torch.zeros(1, 2, 3, 14, 14)
        tokens, patch_start = backbone(normalized_images)
        self.assertEqual(patch_start, 5)
        self.assertEqual(len(tokens), 4)
        for token in tokens:
            self.assertEqual(token.shape, (1, 2, 6, 2048))
        torch.testing.assert_close(
            network.last_images,
            torch.full_like(network.last_images, 0.5),
        )

        pose = camera(tokens)[-1]
        self.assertEqual(pose.shape, (1, 2, 9))
        self.assertTrue(torch.isfinite(pose).all())
        self.assertAlmostEqual(pose[0, 1, 0].item(), -1.0, places=5)
        self.assertAlmostEqual(
            pose[0, 0, 8].item(),
            2.0 * torch.atan(torch.tensor(0.5 / 0.8)).item(),
            places=4,
        )

        predicted_depth, confidence = depth(
            tokens, images=normalized_images, patch_start_idx=patch_start
        )
        self.assertEqual(predicted_depth.shape, (1, 2, 14, 14, 1))
        self.assertEqual(confidence.shape, (1, 2, 14, 14))
        torch.testing.assert_close(predicted_depth, torch.full_like(predicted_depth, 2.0))
        torch.testing.assert_close(confidence, torch.full_like(confidence, 0.5))

    def test_native_local_points_are_preserved_for_gaussian_means(self):
        local = torch.tensor([[[[[1.0, 2.0, 2.0]]]]])
        refined_depth = torch.tensor([[[[[4.0]]]]])
        rescaled = rescale_local_points_to_depth(local, refined_depth)
        torch.testing.assert_close(
            rescaled,
            torch.tensor([[[[[2.0, 4.0, 4.0]]]]]),
        )
        c2w = torch.eye(4).view(1, 1, 4, 4)
        c2w[..., 0, 3] = 3.0
        world = transform_local_points_to_world(rescaled, c2w)
        torch.testing.assert_close(
            world,
            torch.tensor([[[[[5.0, 4.0, 4.0]]]]]),
        )

        encoder_source = (
            ROOT / "src/model/encoder/anysplat.py"
        ).read_text(encoding="utf-8")
        self.assertIn("transform_local_points_to_world", encoder_source)
        self.assertIn("self.aggregator.cached_geometry(ctx_img_num)", encoder_source)

    def test_config_declares_abot_and_checkpoint_detection_rejects_mismatch(self):
        source = (ROOT / "src/model/encoder/anysplat.py").read_text(encoding="utf-8")
        self.assertIn('Literal["zipmap", "streamvggt", "abot"]', source)
        checkpoint = {
            "state_dict": {
                "model.encoder.aggregator.network.encoder.weight": torch.zeros(1)
            }
        }
        cfg = type("Cfg", (), {"reconstruction_backbone": "abot"})()
        validate_backbone_checkpoint(checkpoint, cfg)
        cfg.reconstruction_backbone = "zipmap"
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_backbone_checkpoint(checkpoint, cfg)


if __name__ == "__main__":
    unittest.main()
