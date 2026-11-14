import unittest

import torch

from src.model.gaussian_scales import (
    pixel_footprint_scales,
    screen_radius_guarded_scales,
)


class PixelFootprintScaleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
        self.intrinsics[..., 0, 0] = 100.0
        self.intrinsics[..., 1, 1] = 100.0

    def test_maps_logits_to_bounded_projected_radius(self) -> None:
        raw = torch.tensor(
            [[[[0.0, -100.0, 100.0], [0.0, -100.0, 100.0]]]]
        )
        depth = torch.tensor([[[[[2.0], [4.0]]]]])

        scales = pixel_footprint_scales(
            raw, depth, self.intrinsics, 0.5, 2.0
        )
        projected_radius = scales * 100.0 / depth.reshape(1, 1, 2, 1)

        torch.testing.assert_close(
            projected_radius[0, 0, :, 0], torch.full((2,), 1.25)
        )
        torch.testing.assert_close(
            projected_radius[0, 0, :, 1], torch.full((2,), 0.5)
        )
        torch.testing.assert_close(
            projected_radius[0, 0, :, 2], torch.full((2,), 2.0)
        )

    def test_is_equivariant_to_scene_scale(self) -> None:
        raw = torch.zeros(1, 1, 2, 3)
        depth = torch.tensor([[[[[2.0], [4.0]]]]])
        base = pixel_footprint_scales(
            raw, depth, self.intrinsics, 0.5, 2.0
        )
        scaled = pixel_footprint_scales(
            raw, depth * 7.0, self.intrinsics, 0.5, 2.0
        )
        torch.testing.assert_close(scaled, base * 7.0)

    def test_keeps_scale_logits_trainable(self) -> None:
        raw = torch.zeros(1, 1, 1, 3, requires_grad=True)
        depth = torch.ones(1, 1, 1, 1, 1)
        scales = pixel_footprint_scales(
            raw, depth, self.intrinsics, 0.5, 2.0
        )
        scales.sum().backward()

        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue((raw.grad > 0.0).all())


class ScreenRadiusGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world_to_camera = torch.eye(4).reshape(1, 4, 4)
        self.intrinsics = torch.eye(3).reshape(1, 3, 3)
        self.intrinsics[:, 0, 0] = 100.0
        self.intrinsics[:, 1, 1] = 100.0

    def guard(
        self,
        scales: torch.Tensor,
        means: torch.Tensor,
        max_radius: float = 10.0,
        **kwargs,
    ) -> torch.Tensor:
        return screen_radius_guarded_scales(
            scales,
            means,
            self.world_to_camera,
            self.intrinsics,
            max_radius,
            sigma=3.0,
            **kwargs,
        )

    def test_shrinks_centered_gaussian_to_radius_limit(self) -> None:
        scales = torch.ones(1, 3)
        means = torch.tensor([[0.0, 0.0, 10.0]])

        guarded = self.guard(scales, means)

        torch.testing.assert_close(guarded, scales / 3.0)

    def test_keeps_small_and_behind_camera_gaussians_unchanged(self) -> None:
        scales = torch.tensor([[0.1, 0.1, 0.1], [100.0, 100.0, 100.0]])
        means = torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, -1.0]])

        guarded = self.guard(scales, means)

        torch.testing.assert_close(guarded, scales)

    def test_uses_strictest_camera(self) -> None:
        scales = torch.ones(1, 3)
        means = torch.tensor([[0.0, 0.0, 10.0]])
        cameras = self.world_to_camera.repeat(2, 1, 1)
        intrinsics = self.intrinsics.repeat(2, 1, 1)
        intrinsics[1, 0, 0] = 200.0
        intrinsics[1, 1, 1] = 200.0

        guarded = screen_radius_guarded_scales(
            scales,
            means,
            cameras,
            intrinsics,
            max_screen_radius=10.0,
            sigma=3.0,
        )

        torch.testing.assert_close(guarded, scales / 6.0)

    def test_detached_guard_preserves_scale_gradient(self) -> None:
        scales = torch.ones(1, 3, requires_grad=True)
        means = torch.tensor([[0.0, 0.0, 10.0]], requires_grad=True)

        guarded = self.guard(scales, means)
        guarded.sum().backward()

        self.assertIsNotNone(scales.grad)
        self.assertTrue(torch.isfinite(scales.grad).all())
        self.assertTrue((scales.grad > 0.0).all())
        self.assertIsNone(means.grad)

    def test_chunking_matches_single_chunk(self) -> None:
        scales = torch.tensor(
            [[1.0, 0.5, 0.25], [0.2, 0.2, 0.2], [2.0, 1.0, 0.5]]
        )
        means = torch.tensor(
            [[0.0, 0.0, 10.0], [1.0, 0.0, 2.0], [-2.0, 1.0, 20.0]]
        )

        chunked = self.guard(scales, means, chunk_size=1)
        whole = self.guard(scales, means, chunk_size=100)

        torch.testing.assert_close(chunked, whole)

    def test_rejects_invalid_configuration(self) -> None:
        scales = torch.ones(1, 3)
        means = torch.tensor([[0.0, 0.0, 10.0]])

        with self.assertRaisesRegex(ValueError, "screen radius"):
            self.guard(scales, means, max_radius=0.0)
        with self.assertRaisesRegex(ValueError, "chunk size"):
            self.guard(scales, means, chunk_size=0)


if __name__ == "__main__":
    unittest.main()
