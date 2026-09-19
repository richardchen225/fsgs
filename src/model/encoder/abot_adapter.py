"""Frozen ABot geometry features and cameras for the existing DPT/GS heads.

Uses the official SDPA streaming decode, including its KV pruning and camera
state. Intermediate pairs are read without changing the pretrained computation.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn


ABOT_REVISION = "edbe7e153bc2a5e35e2d4cb76fb02c90d14fade8"
DEFAULT_FEATURE_LAYERS = (7, 17, 25, 35)


def _gather_views(tensor, indices):
    batch = torch.arange(tensor.shape[0], device=tensor.device)[:, None]
    return tensor[batch, indices]


def _proper_rotation(matrix):
    u, singular, vh = torch.linalg.svd(matrix)
    correction = torch.ones_like(singular)
    correction[..., -1] = torch.linalg.det(u @ vh)
    return (u * correction.unsqueeze(-2)) @ vh, singular


@torch.no_grad()
def align_source_cameras(reference, moving, cameras):
    """Fit moving -> reference Sim(3) on matched source c2w poses only.

    Non-collinear camera centers use Umeyama. For a line/point trajectory,
    camera orientations resolve the otherwise unobservable rotation. A single
    source or stationary trajectory cannot determine scale; use 1 in that case.
    """
    if (reference.ndim != 4 or reference.shape != moving.shape
            or reference.shape[1] < 1 or reference.shape[-2:] != (4, 4)
            or cameras.ndim != 4 or cameras.shape[0] != reference.shape[0]
            or cameras.shape[-2:] != (4, 4)):
        raise ValueError("Alignment expects matched [B,S,4,4] and full [B,V,4,4] c2w poses.")
    if not all(torch.isfinite(x).all() for x in (reference, moving, cameras)):
        raise ValueError("ABot produced non-finite camera poses before alignment.")
    dtype = cameras.dtype
    # Small 3x3 fits in float64 avoid half-precision SVD and baseline cancellation.
    with torch.autocast(cameras.device.type, enabled=False):
        reference, moving, cameras = (x.double() for x in (reference, moving, cameras))
        x, y = moving[..., :3, 3], reference[..., :3, 3]
        x_mean, y_mean = x.mean(1), y.mean(1)
        xc, yc = x - x_mean[:, None], y - y_mean[:, None]
        covariance = yc.transpose(1, 2) @ xc / x.shape[1]
        center_rotation, singular = _proper_rotation(covariance)
        orientation_rotation, _ = _proper_rotation(
            (reference[..., :3, :3] @ moving[..., :3, :3].transpose(-1, -2)).mean(1)
        )
        use_orientation = (singular[:, 0] <= 1e-12) | (singular[:, 1] <= singular[:, 0] * 1e-4)
        rotation = torch.where(use_orientation[:, None, None], orientation_rotation, center_rotation)
        rotated_xc = xc @ rotation.transpose(-1, -2)
        variance = xc.square().sum((1, 2))
        scale = (rotated_xc * yc).sum((1, 2)) / variance.clamp_min(1e-12)
        scale_fallback = (variance <= 1e-12) | (yc.square().sum((1, 2)) <= 1e-12) | (scale <= 0)
        scale = torch.where(scale_fallback, torch.ones_like(scale), scale)
        translation = y_mean - scale[:, None] * (rotation @ x_mean.unsqueeze(-1)).squeeze(-1)
        aligned = cameras.clone()
        # Scale camera centers, NEVER the camera rotation basis.
        aligned[..., :3, :3] = rotation[:, None] @ cameras[..., :3, :3]
        aligned[..., :3, 3] = (
            scale[:, None, None] * (cameras[..., :3, 3] @ rotation.transpose(-1, -2))
            + translation[:, None]
        )
        center_error = scale[:, None, None] * rotated_xc - yc
        rmse = center_error.square().sum(-1).mean(1).sqrt()
        radius = yc.square().sum(-1).mean(1).sqrt()
        relative_rotation = (
            (rotation[:, None] @ moving[..., :3, :3])
            @ reference[..., :3, :3].transpose(-1, -2)
        )
        angle = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos()
        diagnostics = {
            "scale": scale,
            "source_center_rmse": rmse,
            "source_center_relative_rmse": rmse / radius.clamp_min(1e-6),
            "source_rotation_error_deg": angle.mean(1) * (180 / torch.pi),
            "orientation_fallback": use_orientation,
            "scale_fallback": scale_fallback,
        }
    return aligned.to(dtype), {key: value.float() for key, value in diagnostics.items()}


class ABotBackboneAdapter(nn.Module):
    def __init__(
        self,
        weights_path: str,
        feature_layers: tuple[int, ...] = DEFAULT_FEATURE_LAYERS,
        enable_depth_teacher: bool = False,
    ) -> None:
        super().__init__()
        path = Path(weights_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ABot checkpoint not found: {path}")
        try:
            from abot_recon.config import InferenceConfig
            from abot_recon.model import build_model
        except ImportError as exc:
            raise ImportError(
                "ABot is optional. Install the pinned package from "
                "requirements_abot.txt; see scripts/ABOT_BACKBONE.md."
            ) from exc

        # Load the complete official checkpoint strictly before discarding the
        # unused branches. The optional point branch is a frozen depth teacher.
        runtime = build_model(InferenceConfig(
            checkpoint=path.resolve(), device="cpu", attention_backend="sdpa",
            loop_closure=False, output_confidence=False,
        ))
        self.network = runtime.network
        unused = ["conf_decoder", "conf_head", "global_points_decoder", "global_point_head"]
        if enable_depth_teacher:
            from abot_recon.modeling.pi3.models.depth_utils import depth_from_log_z

            self._depth_from_log_z = depth_from_log_z
        else:
            unused.extend(["point_decoder", "point_head"])
        for name in unused:
            if hasattr(self.network, name):
                delattr(self.network, name)
        self.feature_layers = tuple(feature_layers)
        self._validate_layers()
        self.requires_grad_(False)
        self.eval()

    def _validate_layers(self) -> None:
        layers = self.feature_layers
        count = len(self.network.decoder)
        if (
            len(layers) != 4 or tuple(sorted(set(layers))) != layers
            or any(i < 1 or i >= count or i % 2 != 1 for i in layers)
        ):
            raise ValueError(
                "abot_feature_layers must contain four increasing odd block "
                f"indices in [1, {count - 1}]; got {layers}"
            )
        if 2 * self.network.dec_embed_dim != 2048:
            raise ValueError("The existing DPT requires ABot large (2048 fused channels).")

    def train(self, mode: bool = True):
        # Lightning's parent .train() must not enable dropout/checkpointing in
        # this frozen pretrained branch.
        return super().train(False)

    def _decode(self, hidden, height, width, carry, collect_features):
        captured = {}
        handles = []
        if collect_features:
            for end in self.feature_layers:
                if end == len(self.network.decoder) - 1:
                    continue

                def save_local(module, args, output, index=end):
                    captured[(index, "local")] = output

                def save_global(module, args, index=end):
                    # Official packed SDPA calls global attention directly,
                    # bypassing the global block's forward hooks. The next
                    # local block receives exactly that global block's output.
                    local = captured.pop((index, "local"))
                    captured[index] = torch.cat(
                        [local, args[0].reshape_as(local)], dim=-1
                    )

                handles.append(self.network.decoder[end - 1].register_forward_hook(save_local))
                handles.append(self.network.decoder[end + 1].register_forward_pre_hook(save_global))
        try:
            fused, positions, carry = self.network.decode(
                hidden, 1, height, width, causal_global=True,
                long_sequence_parallel=True, streaming_inference=True,
                past_key_values=carry,
            )
        finally:
            for handle in handles:
                handle.remove()
        if collect_features:
            last = len(self.network.decoder) - 1
            if last in self.feature_layers:
                captured[last] = fused
            features = [captured[index] for index in self.feature_layers]
        else:
            features = None
        return fused, positions, carry, features

    @torch.no_grad()
    def forward_two_pass(self, images, num_source_views, frame_indices, return_depth: bool = False):
        """Source-only geometry, then chronological full-clip target localization.

        Return features/cameras in the caller's original source-prefix order.
        Each forward invocation owns a fresh KV/camera state, including pass 2.
        Duplicate frame IDs retain their input-slot ordering and correspondence.
        """
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("ABot expects images [B,V,3,H,W].")
        batch, views = images.shape[:2]
        if not 0 < num_source_views <= views:
            raise ValueError("num_source_views must be in [1, V].")
        if frame_indices is None:
            raise ValueError("ABot two_pass requires real frame indices to sort source + target.")
        indices = torch.as_tensor(frame_indices, device=images.device)
        if indices.ndim == 1 and batch == 1:
            indices = indices.unsqueeze(0)
        if indices.shape != (batch, views) or not torch.isfinite(indices).all():
            raise ValueError("ABot frame indices must be finite [B,V] values.")

        source_order = indices[:, :num_source_views].argsort(dim=1, stable=True)
        source_images = _gather_views(images[:, :num_source_views], source_order)
        source_output = self(source_images, num_source_views, return_depth=True) if return_depth else self(
            source_images, num_source_views
        )
        features, start, source_cameras = source_output[:3]
        source_inverse = source_order.argsort(dim=1)
        features = [_gather_views(feature, source_inverse) for feature in features]
        source_cameras = _gather_views(source_cameras, source_inverse)
        teacher_depth = _gather_views(source_output[3], source_inverse) if return_depth else None
        del source_output, source_images
        if views == num_source_views:
            result = (features, start, source_cameras, {})
            return (*result, teacher_depth) if return_depth else result

        full_order = indices.argsort(dim=1, stable=True)
        _, _, full_cameras = self(_gather_views(images, full_order), num_feature_views=0)
        full_cameras = _gather_views(full_cameras, full_order.argsort(dim=1))
        aligned, diagnostics = align_source_cameras(
            source_cameras, full_cameras[:, :num_source_views], full_cameras
        )
        cameras = torch.cat([source_cameras, aligned[:, num_source_views:]], dim=1)
        result = (features, start, cameras, diagnostics)
        return (*result, teacher_depth) if return_depth else result

    def _predict_local_depth(self, fused, positions, height, width):
        # Same point branch and depth activation as official ABot. Do not use
        # point confidence, project with MoGe, or run a second geometry stream.
        hidden = self.network.point_decoder(fused, xpos=positions)
        with torch.autocast(fused.device.type, enabled=False):
            raw_points = self.network.point_head(
                [hidden.float()[:, self.network.patch_start_idx:]], (height, width)
            )
            depth = self._depth_from_log_z(raw_points[..., 2:3], self.network.point_z_log_max)
        return depth.float().detach()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, num_feature_views: int, return_depth: bool = False):
        """One fresh stream in supplied order; zero feature views is camera-only.

        State is local to this invocation and batched along B, so scenes and
        separate train/validation/test calls cannot share historical state.
        """
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("ABot expects images [B,V,3,H,W].")
        batch, views, _, height, width = images.shape
        if views < 1 or not 0 <= num_feature_views <= views:
            raise ValueError("num_feature_views must be in [0, V], with V >= 1.")
        if return_depth and (num_feature_views == 0 or not hasattr(self.network, "point_head")):
            raise ValueError("Depth export requires source views and enable_depth_teacher=True.")
        patch_size = self.network.patch_size
        if height % patch_size or width % patch_size:
            raise ValueError("ABot/DPT image height and width must be divisible by 14.")

        carry, camera_state = None, None
        feature_frames = [[] for _ in self.feature_layers]
        camera_frames = []
        depth_frames = []
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if images.is_cuda else nullcontext()
        )
        with amp:
            for frame_index in range(views):
                frame = images[:, frame_index].float()
                frame = (frame - self.network.image_mean) / self.network.image_std
                hidden = self.network.encoder(frame, is_training=True)
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]
                fused, positions, carry, features = self._decode(
                    hidden, height, width, carry,
                    collect_features=frame_index < num_feature_views,
                )
                carry = self.network._postprocess_stream_carry(
                    carry, spatial_hw=(height, width)
                )
                if features is not None:
                    for destination, feature in zip(feature_frames, features):
                        destination.append(feature)
                    if return_depth:
                        depth_frames.append(self._predict_local_depth(fused, positions, height, width))

                camera_hidden = self.network.camera_decoder(fused, xpos=positions)
                # Match the official forward: pose prediction is float32,
                # includes all camera tokens, and carries the rotation refiner.
                with torch.autocast(images.device.type, enabled=False):
                    cameras, camera_state = self.network._predict_camera_poses(
                        camera_hidden.float(), B=batch, N=1,
                        patch_h=height // patch_size, patch_w=width // patch_size,
                        camera_state=camera_state, return_state=True,
                    )
                camera_frames.append(cameras.float())

        # These are ordinary no_grad tensors, not inference_mode tensors, so
        # the trainable DPT can save them for its parameter-gradient backward.
        features = [torch.stack(frames, dim=1) for frames in feature_frames] if num_feature_views else []
        result = (features, self.network.patch_start_idx, torch.cat(camera_frames, dim=1))
        return (*result, torch.stack(depth_frames, dim=1)) if return_depth else result
