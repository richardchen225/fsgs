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


class ABotBackboneAdapter(nn.Module):
    def __init__(
        self,
        weights_path: str,
        feature_layers: tuple[int, ...] = DEFAULT_FEATURE_LAYERS,
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
        # unused point/confidence branches. No loop closure or image resizing.
        runtime = build_model(InferenceConfig(
            checkpoint=path.resolve(), device="cpu", attention_backend="sdpa",
            loop_closure=False, output_confidence=False,
        ))
        self.network = runtime.network
        for name in (
            "point_decoder", "point_head", "conf_decoder", "conf_head",
            "global_points_decoder", "global_point_head",
        ):
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
    def forward(self, images: torch.Tensor, num_feature_views: int):
        """Input is RGB [0, 1], in source-then-target order; no sorting/reset.

        State is local to this invocation and batched along B, so scenes and
        separate train/validation/test calls cannot share historical state.
        """
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("ABot expects images [B,V,3,H,W].")
        batch, views, _, height, width = images.shape
        if not 0 < num_feature_views <= views:
            raise ValueError("num_feature_views must be in [1, V].")
        patch_size = self.network.patch_size
        if height % patch_size or width % patch_size:
            raise ValueError("ABot/DPT image height and width must be divisible by 14.")

        carry, camera_state = None, None
        feature_frames = [[] for _ in self.feature_layers]
        camera_frames = []
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
        features = [torch.stack(frames, dim=1) for frames in feature_frames]
        return features, self.network.patch_start_idx, torch.cat(camera_frames, dim=1)
