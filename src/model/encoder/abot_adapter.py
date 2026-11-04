"""Adapters exposing ABot-Recon through the VGGT-style encoder interface."""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence

import torch
from torch import nn

from src.model.encoder.vggt.utils.pose_enc import extri_intri_to_pose_encoding


DEFAULT_FEATURE_LAYER_PAIRS = ((8, 9), (16, 17), (26, 27), (34, 35))


def _hook_tensor(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not torch.is_tensor(output):
        raise RuntimeError(
            "ABot decoder hook expected a tensor output, got "
            f"{type(output).__name__}."
        )
    return output


def rescale_local_points_to_depth(
    local_points: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    """Preserve ABot rays while replacing their Z with the selected depth."""
    if local_points.ndim != 5 or local_points.shape[-1] != 3:
        raise ValueError(
            "ABot local points must have shape [B,V,H,W,3], got "
            f"{tuple(local_points.shape)}."
        )
    if depth.shape != local_points.shape[:-1] + (1,):
        raise ValueError(
            "ABot depth must match the local point map, got "
            f"points={tuple(local_points.shape)}, depth={tuple(depth.shape)}."
        )
    points = torch.nan_to_num(
        local_points.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    target_z = torch.nan_to_num(
        depth.float(), nan=1e-4, posinf=100.0, neginf=1e-4
    ).clamp(1e-4, 100.0)
    source_z = local_points[..., 2:3].float()
    valid_z = torch.isfinite(source_z) & (source_z > 1e-6)
    scale = torch.where(
        valid_z,
        target_z / source_z.clamp_min(1e-6),
        torch.zeros_like(target_z),
    )
    return torch.cat([points[..., :2] * scale, target_z], dim=-1)


def transform_local_points_to_world(
    local_points: torch.Tensor,
    camera_to_world: torch.Tensor,
) -> torch.Tensor:
    """Transform dense ABot camera-space XYZ maps into world coordinates."""
    if camera_to_world.shape != local_points.shape[:2] + (4, 4):
        raise ValueError(
            "ABot camera transforms must be [B,V,4,4], got "
            f"points={tuple(local_points.shape)}, c2w={tuple(camera_to_world.shape)}."
        )
    points = local_points.to(dtype=camera_to_world.dtype)
    homogeneous = torch.cat(
        [points, torch.ones_like(points[..., :1])], dim=-1
    )
    return torch.einsum(
        "bvij,bvhwj->bvhwi", camera_to_world, homogeneous
    )[..., :3]


class ABotBackboneAdapter(nn.Module):
    """Run ABot and expose four 2048-D intermediate token tensors."""

    def __init__(
        self,
        network: nn.Module,
        *,
        input_mean: Sequence[float] = (0.5, 0.5, 0.5),
        input_std: Sequence[float] = (0.5, 0.5, 0.5),
        feature_layer_pairs: Sequence[tuple[int, int]] = DEFAULT_FEATURE_LAYER_PAIRS,
    ) -> None:
        super().__init__()
        self.network = network
        self.feature_layer_pairs = tuple(
            (int(local), int(global_)) for local, global_ in feature_layer_pairs
        )
        decoder = getattr(network, "decoder", None)
        if decoder is None:
            raise RuntimeError("ABot network has no decoder blocks.")
        required_layer = max(index for pair in self.feature_layer_pairs for index in pair)
        if len(decoder) <= required_layer:
            raise RuntimeError(
                "ABot decoder is too shallow for the selected feature layers: "
                f"layers={len(decoder)}, required_index={required_layer}."
            )

        self.patch_start_idx = int(getattr(network, "patch_start_idx", 5))
        self.depth = len(decoder) // 2
        self.register_buffer(
            "input_mean",
            torch.tensor(tuple(input_mean), dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(tuple(input_std), dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self._captured: dict[int, torch.Tensor] = {}
        self._cached_output: dict[str, torch.Tensor | None] | None = None
        self._cached_image_shape: tuple[int, int] | None = None
        self._hook_handles = []
        for index in {index for pair in self.feature_layer_pairs for index in pair}:
            self._hook_handles.append(
                decoder[index].register_forward_hook(self._capture_hook(index))
            )

    def _capture_hook(self, index: int):
        def capture(_module, _inputs, output) -> None:
            self._captured[index] = _hook_tensor(output)

        return capture

    def _reshape_layer(
        self,
        tensor: torch.Tensor,
        *,
        batch_size: int,
        view_count: int,
        token_count: int,
        global_layer: bool,
    ) -> torch.Tensor:
        if global_layer:
            expected = (batch_size, view_count * token_count)
            if tensor.shape[:2] != expected:
                raise RuntimeError(
                    "Unexpected ABot global decoder shape: "
                    f"got={tuple(tensor.shape)}, expected_prefix={expected}."
                )
            return tensor.reshape(batch_size, view_count, token_count, -1)
        expected = (batch_size * view_count, token_count)
        if tensor.shape[:2] != expected:
            raise RuntimeError(
                "Unexpected ABot local decoder shape: "
                f"got={tuple(tensor.shape)}, expected_prefix={expected}."
            )
        return tensor.reshape(batch_size, view_count, token_count, -1)

    def forward(
        self,
        images: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
        past_frame_idx: int = 0,
    ):
        del past_key_values, past_frame_idx
        if use_cache:
            raise NotImplementedError(
                "ABot's AnySplat adapter currently evaluates the full causal sequence; "
                "external KV-cache calls are not supported."
            )
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(
                "ABot images must have shape [B,V,3,H,W], got "
                f"{tuple(images.shape)}."
            )
        height, width = images.shape[-2:]
        if height % 14 or width % 14:
            raise ValueError(
                "ABot input height and width must be divisible by 14, got "
                f"{height}x{width}."
            )

        self._captured.clear()
        self.network.eval()
        mean = self.input_mean.to(device=images.device, dtype=images.dtype)
        std = self.input_std.to(device=images.device, dtype=images.dtype)
        images_01 = (images * std + mean).clamp(0.0, 1.0)
        output = self.network(images_01, causal_global_attn=True)
        required = {"camera_poses", "local_points"}
        missing = required.difference(output)
        if missing:
            raise RuntimeError(f"ABot output is missing required fields: {sorted(missing)}")
        self._cached_output = {
            "camera_poses": output["camera_poses"],
            "local_points": output["local_points"],
            "conf": output.get("conf"),
        }
        self._cached_image_shape = (height, width)

        batch_size, view_count = images.shape[:2]
        token_count = self.patch_start_idx + (height // 14) * (width // 14)
        features = []
        for local_index, global_index in self.feature_layer_pairs:
            if local_index not in self._captured or global_index not in self._captured:
                raise RuntimeError(
                    "ABot decoder hooks did not observe all requested feature layers."
                )
            local = self._reshape_layer(
                self._captured[local_index],
                batch_size=batch_size,
                view_count=view_count,
                token_count=token_count,
                global_layer=False,
            )
            global_ = self._reshape_layer(
                self._captured[global_index],
                batch_size=batch_size,
                view_count=view_count,
                token_count=token_count,
                global_layer=True,
            )
            fused = torch.cat([local, global_], dim=-1)
            if fused.shape[-1] != 2048:
                raise RuntimeError(
                    "ABot local/global decoder features must concatenate to 2048 "
                    f"channels, got {fused.shape[-1]}."
                )
            features.append(fused)
        return features, self.patch_start_idx

    def cached_geometry(
        self, view_count: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, int]]:
        if self._cached_output is None or self._cached_image_shape is None:
            raise RuntimeError("ABot geometry was requested before running the backbone.")
        camera_poses = self._cached_output["camera_poses"]
        local_points = self._cached_output["local_points"]
        confidence = self._cached_output["conf"]
        if view_count is not None:
            camera_poses = camera_poses[:, :view_count]
            local_points = local_points[:, :view_count]
            if confidence is not None:
                confidence = confidence[:, :view_count]
        return camera_poses, local_points, confidence, self._cached_image_shape


def _fit_normalized_intrinsics(local_points: torch.Tensor) -> torch.Tensor:
    """Fit one pinhole focal pair per frame from ABot's local point map."""
    batch_size, view_count, height, width, channels = local_points.shape
    if channels != 3:
        raise ValueError(
            f"ABot local point maps must end in XYZ, got {tuple(local_points.shape)}."
        )
    points = local_points.detach().float()
    z = points[..., 2]
    valid = torch.isfinite(points).all(dim=-1) & (z > 1e-6)
    z_safe = z.clamp_min(1e-6)
    ray_x = points[..., 0] / z_safe
    ray_y = points[..., 1] / z_safe
    u = (
        (torch.arange(width, device=points.device, dtype=points.dtype) + 0.5)
        / float(width)
    ).view(1, 1, 1, width).expand(batch_size, view_count, height, width)
    v = (
        (torch.arange(height, device=points.device, dtype=points.dtype) + 0.5)
        / float(height)
    ).view(1, 1, height, 1).expand(batch_size, view_count, height, width)

    def fit_focal(pixel: torch.Tensor, ray: torch.Tensor) -> torch.Tensor:
        mask = valid & torch.isfinite(ray)
        weight = mask.to(points.dtype)
        count = weight.sum(dim=(-2, -1)).clamp_min(1.0)
        pixel_mean = (pixel * weight).sum(dim=(-2, -1)) / count
        ray_mean = (ray * weight).sum(dim=(-2, -1)) / count
        pixel_centered = pixel - pixel_mean[..., None, None]
        ray_centered = ray - ray_mean[..., None, None]
        covariance = (pixel_centered * ray_centered * weight).sum(dim=(-2, -1))
        variance = (pixel_centered.square() * weight).sum(dim=(-2, -1))
        slope = covariance / variance.clamp_min(1e-8)
        default_focal = 0.5 / math.tan(math.radians(60.0) / 2.0)
        focal = torch.where(
            (count >= 16) & torch.isfinite(slope) & (slope > 1e-4),
            slope.reciprocal(),
            torch.full_like(slope, default_focal),
        )
        return focal.clamp(0.05, 20.0)

    fx = fit_focal(u, ray_x)
    fy = fit_focal(v, ray_y)
    intrinsics = points.new_zeros((batch_size, view_count, 3, 3))
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = 0.5
    intrinsics[..., 1, 2] = 0.5
    intrinsics[..., 2, 2] = 1.0
    return intrinsics


class ABotCameraHeadAdapter(nn.Module):
    """Convert ABot c2w poses and fitted focal lengths to pose encodings."""

    def __init__(self, backbone: ABotBackboneAdapter) -> None:
        super().__init__()
        object.__setattr__(self, "_backbone_ref", weakref.ref(backbone))
        self.trunk_depth = 0

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor],
        past_key_values_camera=None,
        use_cache: bool = False,
    ):
        backbone = self._backbone_ref()
        if backbone is None:
            raise RuntimeError("ABot backbone was released before its camera adapter.")
        view_count = aggregated_tokens_list[-1].shape[1]
        c2w, local_points, _confidence, image_shape = backbone.cached_geometry(view_count)
        w2c = torch.linalg.inv(c2w.float())[..., :3, :]
        normalized_intrinsics = _fit_normalized_intrinsics(local_points)
        pose_encoding = extri_intri_to_pose_encoding(
            w2c,
            normalized_intrinsics,
            image_size_hw=image_shape,
        )
        result = [pose_encoding]
        if use_cache:
            return result, past_key_values_camera
        return result


class ABotDepthHeadAdapter(nn.Module):
    """Expose local-point Z and optional ABot confidence as native depth."""

    def __init__(self, backbone: ABotBackboneAdapter) -> None:
        super().__init__()
        object.__setattr__(self, "_backbone_ref", weakref.ref(backbone))

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del images, patch_start_idx
        backbone = self._backbone_ref()
        if backbone is None:
            raise RuntimeError("ABot backbone was released before its depth adapter.")
        view_count = aggregated_tokens_list[-1].shape[1]
        _c2w, local_points, confidence, _image_shape = backbone.cached_geometry(view_count)
        depth = torch.nan_to_num(
            local_points[..., 2:3].float(), nan=1e-4, posinf=100.0, neginf=1e-4
        ).clamp(1e-4, 100.0)
        if confidence is None:
            depth_confidence = torch.ones_like(depth[..., 0])
        else:
            if confidence.ndim == 5 and confidence.shape[-1] == 1:
                confidence = confidence[..., 0]
            if confidence.ndim != 4:
                raise RuntimeError(
                    "ABot confidence must have shape [B,V,H,W] or [B,V,H,W,1], "
                    f"got {tuple(confidence.shape)}."
                )
            depth_confidence = torch.sigmoid(confidence.float())
        return depth, depth_confidence
