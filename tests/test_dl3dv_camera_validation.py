"""Camera validation tests without importing the complete dataset stack."""

import ast
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_camera_validator():
    tree = ast.parse(
        (ROOT / "src/dataset/dataset_dl3dv.py").read_text(encoding="utf-8")
    )
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_camera_parameters"
    )
    namespace = {"torch": torch, "Tensor": torch.Tensor}
    exec(
        compile(
            ast.Module(body=[helper], type_ignores=[]),
            "dataset_dl3dv_camera_validation",
            "exec",
        ),
        namespace,
    )
    return namespace[helper.name]


def load_rigid_camera_inverter():
    tree = ast.parse(
        (
            ROOT / "src/model/decoder/decoder_splatting_cuda.py"
        ).read_text(encoding="utf-8")
    )
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_invert_rigid_camera_poses"
    )
    namespace = {"torch": torch, "Tensor": torch.Tensor}
    exec(
        compile(
            ast.Module(body=[helper], type_ignores=[]),
            "decoder_rigid_camera_inversion",
            "exec",
        ),
        namespace,
    )
    return namespace[helper.name]


VALIDATE = load_camera_validator()
INVERT_RIGID = load_rigid_camera_inverter()


def valid_cameras(count=2):
    extrinsics = torch.eye(4).repeat(count, 1, 1)
    extrinsics[:, 0, 3] = torch.arange(count, dtype=torch.float32)
    intrinsics = torch.eye(3).repeat(count, 1, 1)
    intrinsics[:, 0, 0] = 0.8
    intrinsics[:, 1, 1] = 0.9
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    return extrinsics, intrinsics


class DL3DVCameraValidationTests(unittest.TestCase):
    def test_accepts_finite_rigid_cameras(self):
        extrinsics, intrinsics = valid_cameras()
        VALIDATE(extrinsics, intrinsics, label="valid")

    def test_rejects_singular_rotation_with_frame_index(self):
        extrinsics, intrinsics = valid_cameras()
        extrinsics[1, 2, :3] = 0.0
        with self.assertRaisesRegex(ValueError, r"frame=17, det=0"):
            VALIDATE(
                extrinsics,
                intrinsics,
                label="singular",
                frame_indices=torch.tensor([3, 17]),
            )

    def test_rejects_invalid_homogeneous_row(self):
        extrinsics, intrinsics = valid_cameras()
        extrinsics[0, 3, 3] = 0.0
        with self.assertRaisesRegex(ValueError, "bottom_err=1"):
            VALIDATE(extrinsics, intrinsics, label="bottom")

    def test_rejects_non_positive_focal_length(self):
        extrinsics, intrinsics = valid_cameras()
        intrinsics[0, 0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "fx=0"):
            VALIDATE(extrinsics, intrinsics, label="intrinsics")


class RigidCameraInversionTests(unittest.TestCase):
    def test_inverts_valid_rigid_camera(self):
        camera_to_world = torch.eye(4).repeat(2, 1, 1)
        camera_to_world[1, :3, 3] = torch.tensor([1.0, -2.0, 3.0])

        world_to_camera = INVERT_RIGID(camera_to_world)

        torch.testing.assert_close(
            world_to_camera,
            torch.linalg.inv(camera_to_world),
        )

    def test_preserves_finite_camera_gradients(self):
        camera_to_world = torch.eye(4).repeat(2, 1, 1).requires_grad_()

        world_to_camera = INVERT_RIGID(camera_to_world)
        loss = (
            world_to_camera[:, :3, :3].sum()
            + world_to_camera[:, :3, 3].square().sum()
        )
        loss.backward()

        self.assertIsNotNone(camera_to_world.grad)
        self.assertTrue(bool(torch.isfinite(camera_to_world.grad).all()))
        self.assertGreater(
            camera_to_world.grad[:, :3, :3].abs().sum().item(),
            0.0,
        )

    def test_rejects_singular_camera_with_batch_view_index(self):
        camera_to_world = torch.eye(4).repeat(2, 3, 1, 1)
        camera_to_world[1, 2, 0, :3] = 0.0

        with self.assertRaisesRegex(
            ValueError,
            r"index=\(1, 2\), det=0",
        ):
            INVERT_RIGID(camera_to_world)


if __name__ == "__main__":
    unittest.main()
