from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file

from src.config import load_typed_root_config
from src.model.encoder import act_gs, sh_utils
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.model.model import get_model
from src.model.ply_export import export_ply
from src.model.streaming_gir import DominantGIR, StreamingGaussianState
from src.model.types import Gaussians
from src.experiments.plot_replica_overfit_metrics import (
    plot_metrics,
    plot_residual_update_metrics,
)


MAX_LOGGED_RESIDUAL_RANKS = 8
RANK_UPDATE_LOG_KEYS = (
    "valid_ratio",
    "contribution_weight",
    "relative_weight",
    "update_multiplier",
    "raw_mean_shift",
    "effective_mean_shift",
    "raw_rotation",
    "effective_rotation",
    "raw_log_scale",
    "effective_log_scale",
    "raw_opacity_logit",
    "effective_opacity_logit",
    "raw_harmonics",
    "effective_harmonics",
)
MAP_UPDATE_LOG_KEYS = (
    "map_updated_ratio",
    "map_mean_shift_mean",
    "map_mean_shift_max",
    "map_rotation_mean",
    "map_rotation_max",
    "map_scale_relative_mean",
    "map_scale_relative_max",
    "map_opacity_mean",
    "map_opacity_max",
    "map_harmonics_mean",
    "map_harmonics_max",
)


TRAIN_CSV_FIELDS = [
    "step",
    "loss",
    "loss_base_first",
    "loss_updated_current",
    "loss_new_holes",
    "loss_replay_history",
    "loss_depth",
    "loss_depth_structure",
    "loss_depth_absolute_log_scale",
    "loss_camera",
    "loss_camera_translation",
    "loss_camera_rotation",
    "loss_camera_focal",
    "loss_heldout",
    "loss_regularization",
    "loss_opacity_decay_budget",
    "grad_norm",
    "overlap_ratio",
    "old_coverage_ratio",
    "depth_consistent_ratio",
    "current_in_front_ratio",
    "old_in_front_ratio",
    "bad_ratio",
    "append_ratio",
    "appended_gs",
    "unique_updated_gs",
    "opacity_decay_candidate_probability",
    "opacity_decay_candidate_count",
    "opacity_decay_mass_ratio",
    *[
        f"rank{rank_idx}_{key}"
        for rank_idx in range(1, MAX_LOGGED_RESIDUAL_RANKS + 1)
        for key in RANK_UPDATE_LOG_KEYS
    ],
    *MAP_UPDATE_LOG_KEYS,
]

# Evaluation-only diagnostics for the frozen DAV3 teacher against Replica GT.
# These are written to eval_metrics.csv, not mixed into per-step train losses.
DAV3_GT_METRIC_FIELDS = (
    "depth_dav3_gt_abs_rel_aligned",
    "depth_dav3_gt_rmse_aligned",
    "depth_dav3_gt_log_l1_aligned",
    "depth_dav3_gt_scale",
    "depth_dav3_gt_abs_rel_raw",
    "depth_dav3_gt_rmse_raw",
    "depth_dav3_gt_valid_ratio",
)


def _load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing RGB image: {path}")
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).to(device)


def _load_depth(path: Path, device: torch.device) -> torch.Tensor:
    depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if depth is None:
        raise RuntimeError(
            f"Failed to read EXR depth {path}. Check OPENCV_IO_ENABLE_OPENEXR and "
            "the server OpenCV build."
        )
    if depth.ndim == 3:
        if depth.shape[-1] != 1:
            channel_delta = np.nanmax(np.abs(depth - depth[..., :1]))
            if channel_delta > 1e-5:
                raise ValueError(
                    f"Depth EXR has non-identical channels: {path}, max delta={channel_delta}"
                )
        depth = depth[..., 0]
    depth = torch.from_numpy(depth.astype(np.float32, copy=False)).to(device)
    # Replica EXRs use very large finite values (commonly 1e10) as invalid
    # background depth. Do not let those pixels enter geometry or GT metrics.
    valid = torch.isfinite(depth) & (depth > 0) & (depth < 1e9)
    return torch.where(valid, depth, torch.zeros_like(depth))


def _ray_distance_to_z(depth: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    height, width = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = (xx - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) / intrinsic[1, 1]
    return depth / torch.sqrt(1.0 + x.square() + y.square())


def _normalized_intrinsic(intrinsic: torch.Tensor, height: int, width: int) -> torch.Tensor:
    result = intrinsic.clone()
    result[..., 0, :] /= width
    result[..., 1, :] /= height
    return result


def _find_camera_json(data_root: Path) -> Path:
    canonical = data_root / "cameras.json"
    if canonical.is_file():
        return canonical
    candidates = sorted(data_root.glob("*camera*.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Could not identify the camera JSON under {data_root}. Expected "
            f"cameras.json or one *camera*.json file, found: {candidates}"
        )
    return candidates[0]


def _find_rgb(data_root: Path, frame: dict[str, Any], frame_id: int) -> Path:
    metadata_path = frame.get("rgb_path")
    if metadata_path:
        candidate = data_root / metadata_path
        if candidate.is_file():
            return candidate
    candidates = []
    for suffix in ("png", "jpg", "jpeg"):
        candidates.extend(sorted(data_root.glob(f"rgb_{frame_id:03d}*.{suffix}")))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one RGB file for frame {frame_id} under {data_root}, "
            f"found: {candidates}"
        )
    return candidates[0]


def _find_frame_depth(data_root: Path, frame: dict[str, Any], frame_id: int) -> Path:
    patterns = []
    if frame.get("depth_pattern"):
        patterns.append(frame["depth_pattern"])
    patterns.extend(
        [
            f"depth_{frame_id:03d}_*.exr",
            f"depth_{frame_id:03d}.exr",
        ]
    )
    matches = []
    for pattern in patterns:
        matches.extend(Path(path) for path in sorted(glob.glob(str(data_root / pattern))))
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one depth EXR for frame {frame_id} under {data_root}, "
            f"found: {matches}"
        )
    return matches[0]


def _dl3dv_frame_id(frame: dict[str, Any], fallback_index: int) -> int:
    """Use DL3DV's image ID, with a filename/index fallback."""
    if frame.get("colmap_im_id") is not None:
        return int(frame["colmap_im_id"])
    file_path = Path(str(frame.get("file_path", "")))
    match = re.search(r"(\d+)", file_path.stem)
    return int(match.group(1)) if match else fallback_index + 1


def _resolve_dl3dv_image(
    data_root: Path,
    frame: dict[str, Any],
    image_dir: str,
) -> Path:
    """Resolve transforms.json paths to the downsampled image directory."""
    raw_path = Path(str(frame["file_path"]))
    candidates = []
    candidates.append(data_root / raw_path)
    candidates.append(data_root / image_dir / raw_path.name)
    parts = list(raw_path.parts)
    if parts and parts[0].lower() in {"images", "images_2", "images_4", "images_8"}:
        candidates.append(data_root / image_dir / Path(*parts[1:]))
    candidates.append(data_root / image_dir / raw_path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not resolve DL3DV frame image. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_dl3dv_data(cfg: DictConfig, device: torch.device) -> dict[str, Any]:
    """Load an RGB-only DL3DV/NeRF scene for the overfit experiment."""
    data_root = Path(to_absolute_path(str(cfg.data_root)))
    if not data_root.is_dir():
        raise NotADirectoryError(f"DL3DV data root is not a directory: {data_root}")
    transforms_path = cfg.get("transforms_path")
    if transforms_path:
        transforms_path = Path(to_absolute_path(str(transforms_path)))
    else:
        transforms_path = data_root / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"DL3DV transforms.json not found: {transforms_path}")
    with transforms_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in {transforms_path}")

    image_dir = str(cfg.get("image_dir", "images_8"))
    resize_shape_value = cfg.get("resize_shape", [252, 518])
    if resize_shape_value is None:
        resize_shape = None
    else:
        resize_shape = tuple(int(value) for value in resize_shape_value)
        if len(resize_shape) != 2 or min(resize_shape) <= 0:
            raise ValueError(
                "toy.resize_shape must be [height, width] with positive values."
            )
        if any(value % 14 != 0 for value in resize_shape):
            raise ValueError(
                "toy.resize_shape must be divisible by the backbone patch size 14, "
                f"got {resize_shape}."
            )
    frame_table = {}
    for index, frame in enumerate(frames):
        frame_id = _dl3dv_frame_id(frame, index)
        if frame_id in frame_table:
            raise ValueError(f"Duplicate DL3DV frame id {frame_id} in {transforms_path}")
        frame_table[frame_id] = frame

    train_ids = _frame_id_list(cfg.train_frames, "toy.train_frames", minimum=2)
    heldout_loss_ids = _frame_id_list(
        cfg.heldout_loss_frame, "toy.heldout_loss_frame"
    )
    heldout_test_ids = _frame_id_list(
        cfg.heldout_test_frame, "toy.heldout_test_frame"
    )
    requested_ids = train_ids + heldout_loss_ids + heldout_test_ids
    missing_ids = sorted(set(requested_ids) - set(frame_table))
    if missing_ids:
        raise KeyError(
            f"DL3DV frame IDs {missing_ids} are absent from {transforms_path}. "
            f"Available range/count: {min(frame_table)}..{max(frame_table)}, "
            f"{len(frame_table)} frames."
        )
    overlap_ids = sorted(
        set(train_ids) & (set(heldout_loss_ids) | set(heldout_test_ids))
    )
    if overlap_ids:
        raise ValueError(
            "DL3DV held-out frames must be different from training frames; "
            f"overlap={overlap_ids}"
        )

    pose_flip = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    meta_h = float(metadata.get("h", 0))
    meta_w = float(metadata.get("w", 0))
    if meta_h <= 0 or meta_w <= 0:
        raise ValueError("DL3DV transforms.json must contain positive h and w.")
    required_intrinsics = ("fl_x", "fl_y", "cx", "cy")
    if any(key not in metadata for key in required_intrinsics):
        raise ValueError(
            "DL3DV transforms.json must contain fl_x, fl_y, cx and cy."
        )

    image_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def load_view(frame_id: int, role: str):
        if frame_id in image_cache:
            return image_cache[frame_id]
        frame = frame_table[frame_id]
        image_path = _resolve_dl3dv_image(data_root, frame, image_dir)
        image = _load_rgb(image_path, device)
        height, width = image.shape[-2:]
        intrinsic = torch.eye(3, device=device, dtype=torch.float32)
        intrinsic[0, 0] = float(metadata["fl_x"]) * width / meta_w
        intrinsic[1, 1] = float(metadata["fl_y"]) * height / meta_h
        intrinsic[0, 2] = float(metadata["cx"]) * width / meta_w
        intrinsic[1, 2] = float(metadata["cy"]) * height / meta_h
        if resize_shape is not None and (height, width) != resize_shape:
            target_height, target_width = resize_shape
            image = F.interpolate(
                image.unsqueeze(0),
                size=resize_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            intrinsic[0, 0] *= target_width / width
            intrinsic[0, 2] *= target_width / width
            intrinsic[1, 1] *= target_height / height
            intrinsic[1, 2] *= target_height / height
        c2w_nerf = torch.tensor(
            frame["transform_matrix"], device=device, dtype=torch.float32
        )
        if c2w_nerf.shape != (4, 4):
            raise ValueError(
                f"DL3DV frame {frame_id} transform_matrix must be 4x4, "
                f"got {tuple(c2w_nerf.shape)}"
            )
        c2w_opencv = c2w_nerf @ pose_flip
        image_cache[frame_id] = (image, c2w_opencv, intrinsic)
        print(f"DL3DV {role} frame {frame_id}: RGB={image_path.name}")
        return image_cache[frame_id]

    def stack_views(frame_ids: list[int], role: str) -> dict[str, Any]:
        views = [load_view(frame_id, role) for frame_id in frame_ids]
        shapes = {tuple(view[0].shape) for view in views}
        if len(shapes) != 1:
            raise ValueError(f"DL3DV {role} images do not have the same shape: {shapes}")
        return {
            "ids": frame_ids,
            "rgb": torch.stack([view[0] for view in views], dim=0).unsqueeze(0),
            "c2w": torch.stack([view[1] for view in views], dim=0).unsqueeze(0),
            "intrinsic_px": torch.stack([view[2] for view in views], dim=0).unsqueeze(0),
            "intrinsic": _normalized_intrinsic(
                torch.stack([view[2] for view in views], dim=0).unsqueeze(0),
                views[0][0].shape[-2],
                views[0][0].shape[-1],
            ),
        }

    train = stack_views(train_ids, "training")
    heldout_loss = stack_views(heldout_loss_ids, "held-out loss")
    heldout_test = stack_views(heldout_test_ids, "held-out test")
    images = train["rgb"]
    if heldout_loss["rgb"].shape[-2:] != images.shape[-2:]:
        raise ValueError("DL3DV held-out loss images must match training resolution.")
    if heldout_test["rgb"].shape[-2:] != images.shape[-2:]:
        raise ValueError("DL3DV held-out test images must match training resolution.")
    c2w = train["c2w"]
    intrinsic_px = train["intrinsic_px"]
    height, width = images.shape[-2:]
    depth = torch.zeros(
        (1, len(train_ids), height, width), device=device, dtype=torch.float32
    )
    valid_depth = torch.zeros_like(depth, dtype=torch.bool)
    print(
        f"DL3DV RGB-only scene: {data_root}; frames={len(frame_table)}, "
        f"resolution={width}x{height}; GT depth=unavailable"
    )
    return {
        "root": data_root,
        "ids": train_ids,
        "images": images,
        "depth": depth,
        "valid_depth": valid_depth,
        "has_gt_depth": False,
        "c2w": c2w,
        "intrinsic_px": intrinsic_px,
        "intrinsic": train["intrinsic"],
        "heldout_loss": {
            "ids": heldout_loss["ids"],
            "rgb": heldout_loss["rgb"],
            "c2w": heldout_loss["c2w"],
            "intrinsic": heldout_loss["intrinsic"],
        },
        "heldout_test": {
            "ids": heldout_test["ids"],
            "rgb": heldout_test["rgb"],
            "c2w": heldout_test["c2w"],
            "intrinsic": heldout_test["intrinsic"],
        },
        "image_shape": (height, width),
    }


def _frame_id_list(value: Any, name: str, minimum: int = 1) -> list[int]:
    if isinstance(value, (int, np.integer, str)):
        frame_ids = [int(value)]
    else:
        try:
            frame_ids = [int(frame_id) for frame_id in value]
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer or a list of integers.") from exc
    if len(frame_ids) < minimum:
        raise ValueError(
            f"{name} requires at least {minimum} frame(s), got {frame_ids}."
        )
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"{name} contains duplicate frame IDs: {frame_ids}.")
    return frame_ids


def _load_toy_data(cfg: DictConfig, device: torch.device) -> dict[str, Any]:
    dataset_type = str(cfg.get("dataset_type", "replica")).lower()
    if dataset_type == "dl3dv":
        return _load_dl3dv_data(cfg, device)
    if dataset_type not in {"replica", "toy"}:
        raise ValueError(
            "toy.dataset_type must be 'replica' or 'dl3dv', "
            f"got {dataset_type!r}."
        )
    data_root = Path(to_absolute_path(str(cfg.data_root)))
    if not data_root.is_dir():
        raise NotADirectoryError(f"toy.data_root is not a directory: {data_root}")
    camera_path = _find_camera_json(data_root)
    print(f"Toy data root: {data_root}")
    print(f"Camera metadata: {camera_path}")
    with camera_path.open("r", encoding="utf-8") as handle:
        camera_data = json.load(handle)
    frame_table = {int(frame["frame_id"]): frame for frame in camera_data["frames"]}

    train_ids = _frame_id_list(cfg.train_frames, "toy.train_frames", minimum=2)
    missing_ids = [frame_id for frame_id in train_ids if frame_id not in frame_table]
    if missing_ids:
        raise KeyError(
            f"Training frames {missing_ids} are absent from {camera_path}. "
            f"Requested temporal sequence: {train_ids}."
        )

    rgbs, depths, c2ws, intrinsics = [], [], [], []
    for frame_id in train_ids:
        frame = frame_table[frame_id]
        rgb_path = _find_rgb(data_root, frame, frame_id)
        depth_path = _find_frame_depth(data_root, frame, frame_id)
        print(f"Frame {frame_id}: RGB={rgb_path.name}, depth={depth_path.name}")
        rgb = _load_rgb(rgb_path, device)
        depth = _load_depth(depth_path, device)
        intrinsic = torch.tensor(frame["K"], device=device, dtype=torch.float32)
        c2w = torch.tensor(frame["c2w_opencv"], device=device, dtype=torch.float32)
        w2c = torch.tensor(frame["w2c_opencv"], device=device, dtype=torch.float32)
        inverse_error = (torch.linalg.inv(c2w) - w2c).abs().max().item()
        if inverse_error > 1e-4:
            raise ValueError(
                f"Frame {frame_id} c2w/w2c mismatch: max error={inverse_error:.3e}"
            )
        if tuple(depth.shape) != tuple(rgb.shape[-2:]):
            raise ValueError(
                f"Frame {frame_id} RGB/depth shape mismatch: {rgb.shape[-2:]} vs {depth.shape}"
            )
        if str(cfg.depth_type).lower() == "ray":
            depth = _ray_distance_to_z(depth, intrinsic)
        elif str(cfg.depth_type).lower() != "z":
            raise ValueError(f"toy.depth_type must be 'z' or 'ray', got {cfg.depth_type!r}")
        rgbs.append(rgb)
        depths.append(depth)
        c2ws.append(c2w)
        intrinsics.append(intrinsic)

    images = torch.stack(rgbs).unsqueeze(0)
    depth = torch.stack(depths).unsqueeze(0)
    c2w = torch.stack(c2ws).unsqueeze(0)
    intrinsic_px = torch.stack(intrinsics).unsqueeze(0)
    height, width = images.shape[-2:]

    auxiliary_cache = {}

    def load_auxiliary_view(frame_id: int, role: str) -> dict[str, Any]:
        if frame_id in train_ids:
            raise ValueError(
                f"{role} frame {frame_id} is also one of the training frames {train_ids}."
            )
        if frame_id in auxiliary_cache:
            return auxiliary_cache[frame_id]
        frame = frame_table.get(frame_id)
        if frame is None:
            raise KeyError(f"{role} frame {frame_id} is absent from {camera_path}.")
        rgb_path = _find_rgb(data_root, frame, frame_id)
        rgb = _load_rgb(rgb_path, device)
        if tuple(rgb.shape[-2:]) != (height, width):
            raise ValueError(
                f"{role} frame {frame_id} has shape {rgb.shape[-2:]}, "
                f"expected {(height, width)}."
            )
        intrinsic = torch.tensor(frame["K"], device=device, dtype=torch.float32)
        view = {
            "id": frame_id,
            "rgb": rgb.unsqueeze(0).unsqueeze(0),
            "c2w": torch.tensor(
                frame["c2w_opencv"], device=device, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0),
            "intrinsic": _normalized_intrinsic(
                intrinsic.unsqueeze(0).unsqueeze(0), height, width
            ),
            "rgb_path": rgb_path,
        }
        auxiliary_cache[frame_id] = view
        return view

    def stack_auxiliary_views(frame_ids: list[int], role: str) -> dict[str, Any]:
        views = [load_auxiliary_view(frame_id, role) for frame_id in frame_ids]
        return {
            "ids": frame_ids,
            "rgb": torch.cat([view["rgb"] for view in views], dim=1),
            "c2w": torch.cat([view["c2w"] for view in views], dim=1),
            "intrinsic": torch.cat([view["intrinsic"] for view in views], dim=1),
            "rgb_paths": [view["rgb_path"] for view in views],
        }

    heldout_loss_ids = _frame_id_list(
        cfg.heldout_loss_frame, "toy.heldout_loss_frame"
    )
    heldout_test_ids = _frame_id_list(
        cfg.heldout_test_frame, "toy.heldout_test_frame"
    )
    heldout_loss = stack_auxiliary_views(heldout_loss_ids, "Held-out loss")
    heldout_test = stack_auxiliary_views(heldout_test_ids, "Held-out test")
    print(f"Training sequence: {train_ids}")
    print(f"Held-out loss frames: {heldout_loss_ids}")
    print(f"Held-out test frames: {heldout_test_ids}")

    valid_depth = depth > 0
    for view_idx, frame_id in enumerate(train_ids):
        values = depth[0, view_idx][valid_depth[0, view_idx]]
        if values.numel() == 0:
            raise ValueError(f"Frame {frame_id} has no valid depth values.")
        print(
            f"Frame {frame_id}: depth min/median/max="
            f"{values.min().item():.4f}/{values.median().item():.4f}/{values.max().item():.4f}, "
            f"valid={values.numel() / depth[0, view_idx].numel():.2%}"
        )

    return {
        "root": data_root,
        "ids": train_ids,
        "images": images,
        "depth": depth,
        "valid_depth": valid_depth,
        "has_gt_depth": True,
        "c2w": c2w,
        "intrinsic_px": intrinsic_px,
        "intrinsic": _normalized_intrinsic(intrinsic_px, height, width),
        "heldout_loss": heldout_loss,
        "heldout_test": heldout_test,
        "image_shape": (height, width),
    }


def _rename_wm_key(key: str) -> str:
    if key.startswith("gs_head"):
        key = key.replace("gs_head", "encoder.gaussian_param_head", 1)
    elif key.startswith("gs_renderer"):
        key = key.replace("gs_renderer", "encoder", 1)
    elif key.startswith("depth_head"):
        key = key.replace("depth_head", "encoder.depth_head", 1)
    elif key.startswith("visual_geometry_transformer"):
        key = key.replace("visual_geometry_transformer", "encoder.aggregator", 1)
    elif key.startswith("cam_head"):
        key = key.replace("cam_head", "encoder.camera_head", 1)
    replacements = {
        "reg_token": "register_token",
        "cam_token": "camera_token",
        "refine_net": "trunk",
        "init_token": "empty_pose_tokens",
        "out_norm": "trunk_norm",
        "param_predictor": "pose_branch",
        "adapt_norm_gen": "poseLN_modulation",
        "param_embed": "embed_pose",
    }
    for source, target in replacements.items():
        key = key.replace(source, target)
    return key


def _load_checkpoint(model: torch.nn.Module, checkpoint: str | None) -> dict[str, Any] | None:
    if checkpoint is None or str(checkpoint).lower() in {"", "null", "none"}:
        print("No toy.checkpoint supplied; trainable heads keep their configured initialization.")
        return None
    path = Path(to_absolute_path(str(checkpoint)))
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    payload = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu")
    resume_state = (
        dict(payload)
        if isinstance(payload, dict)
        and "state_dict" in payload
        and "optimizer" in payload
        else None
    )
    model_payload = payload
    if isinstance(model_payload, dict) and "state_dict" in model_payload:
        model_payload = model_payload["state_dict"]
    if (
        isinstance(model_payload, dict)
        and "model" in model_payload
        and isinstance(model_payload["model"], dict)
    ):
        model_payload = model_payload["model"]
    if not isinstance(model_payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(model_payload)}")

    model_state = model.state_dict()
    compatible = {}
    for original_key, value in model_payload.items():
        if not torch.is_tensor(value):
            continue
        stripped = original_key
        for prefix in ("module.", "model."):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
        candidates = (stripped, _rename_wm_key(stripped))
        for candidate in candidates:
            if candidate in model_state and model_state[candidate].shape == value.shape:
                compatible[candidate] = value
                break
    model.load_state_dict(compatible, strict=False)
    copied_rank_heads = 0
    additional_heads = getattr(
        model.gir_update_head, "additional_historical_predictions", []
    )
    for rank_idx, head in enumerate(additional_heads, start=2):
        prefix = (
            "gir_update_head.additional_historical_predictions."
            f"{rank_idx - 2}."
        )
        expected_keys = {f"{prefix}{key}" for key in head.state_dict()}
        if not expected_keys.issubset(compatible):
            head.load_state_dict(model.gir_update_head.prediction.state_dict())
            copied_rank_heads += 1
    prefixes = (
        "encoder.gaussian_param_head.",
        "encoder.gs_head.",
        "encoder.camera_head.",
        "gir_update_head.",
    )
    loaded_heads = sum(key.startswith(prefixes) for key in compatible)
    print(
        f"Loaded {len(compatible)} compatible tensors from {path}; "
        f"{loaded_heads} belong to the trainable toy heads."
    )
    if copied_rank_heads:
        print(
            f"Initialized {copied_rank_heads} missing contributor heads from "
            "the loaded rank-1 residual head."
        )
    if resume_state is not None:
        resume_state["_checkpoint_path"] = str(path)
    return resume_state


def _configure_trainable_modules(
    model: torch.nn.Module,
    residual_enabled: bool,
    train_gaussian_param_head: bool,
    train_gs_head: bool,
    train_depth_head: bool,
    train_camera_head: bool,
    opacity_decay_enabled: bool,
) -> list[dict[str, Any]]:
    model.requires_grad_(False)
    model.encoder.gaussian_param_head.requires_grad_(train_gaussian_param_head)
    model.encoder.gaussian_param_head.scratch.output_conv2.requires_grad_(
        train_gaussian_param_head and train_depth_head
    )
    model.encoder.camera_head.requires_grad_(train_camera_head)
    model.encoder.gs_head.requires_grad_(train_gs_head)
    gir_trainable = residual_enabled or opacity_decay_enabled
    model.gir_update_head.encoder.requires_grad_(gir_trainable)
    model.gir_update_head.prediction.requires_grad_(residual_enabled)
    model.gir_update_head.additional_historical_predictions.requires_grad_(
        residual_enabled
    )
    model.gir_update_head.current_prediction.requires_grad_(False)
    model.gir_update_head.delete_prediction.requires_grad_(False)
    model.gir_update_head.contributor_decay_prediction.requires_grad_(
        opacity_decay_enabled
    )

    groups = []
    if train_gaussian_param_head:
        groups.append({
            "name": "gaussian_param_head",
            "params": [p for p in model.encoder.gaussian_param_head.parameters() if p.requires_grad],
        })
    if train_gs_head:
        groups.append(
            {"name": "gs_head", "params": list(model.encoder.gs_head.parameters())}
        )
    if gir_trainable:
        groups.append(
            {
                "name": "old_residual_head",
                "params": [
                    *model.gir_update_head.encoder.parameters(),
                    *(
                        model.gir_update_head.prediction.parameters()
                        if residual_enabled
                        else []
                    ),
                    *(
                        model.gir_update_head.additional_historical_predictions.parameters()
                        if residual_enabled
                        else []
                    ),
                    *(
                        model.gir_update_head.contributor_decay_prediction.parameters()
                        if opacity_decay_enabled
                        else []
                    ),
                ],
            }
        )
    if train_camera_head:
        groups.append(
            {
                "name": "camera_head",
                "params": list(model.encoder.camera_head.parameters()),
            }
        )
    for group in groups:
        count = sum(parameter.numel() for parameter in group["params"])
        print(f"Trainable {group['name']}: {count:,} parameters")
    if not groups:
        raise ValueError("No trainable modules are enabled for this experiment.")
    return groups


@torch.no_grad()
def _cache_backbone_tokens(model: torch.nn.Module, images: torch.Tensor, use_amp: bool):
    model.encoder.aggregator.eval()
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        tokens, patch_start_idx = model.encoder.aggregator(images.to(torch.float16 if use_amp else torch.float32))
    # The trainable DPT/GS path runs in float32 below. Cache float32 tokens once
    # so every optimization step does not repeat a half-to-float allocation.
    tokens = [token.detach().float() for token in tokens]
    return tokens, patch_start_idx


def _build_base_gaussians(
    model: torch.nn.Module,
    feature: torch.Tensor,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    c2w: torch.Tensor,
    intrinsic_px: torch.Tensor,
) -> Gaussians:
    batch, _, height, width = feature.shape
    # Keep parameter activation in float32. In fp16, exp(raw_scale) can become
    # inf before clamp; its backward then evaluates 0 * inf and produces NaN.
    with torch.autocast("cuda", enabled=False):
        raw = model.encoder.gs_head(feature.float())
    raw = raw.permute(0, 2, 3, 1)
    quats, scales, opacities, residual_sh, _ = torch.split(raw, [4, 3, 1, 3, 1], dim=-1)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(batch, -1, -1)
    yy = yy.unsqueeze(0).expand(batch, -1, -1)
    x_camera = (xx - intrinsic_px[:, 0, 2, None, None]) * depth / intrinsic_px[
        :, 0, 0, None, None
    ]
    y_camera = (yy - intrinsic_px[:, 1, 2, None, None]) * depth / intrinsic_px[
        :, 1, 1, None, None
    ]
    camera_points = torch.stack([x_camera, y_camera, depth], dim=-1)
    homogeneous = torch.cat(
        [camera_points, torch.ones_like(camera_points[..., :1])], dim=-1
    ).reshape(batch, height * width, 4)
    means = torch.bmm(homogeneous, c2w.transpose(1, 2))[..., :3]
    rgb_flat = rgb.flatten(2).transpose(1, 2)
    base_sh = sh_utils.RGB2SH(rgb_flat)
    valid = (depth > 0).reshape(batch, height * width)
    rotations = act_gs.reg_dense_rotation(quats.float()).reshape(batch, height * width, 4)
    identity = torch.zeros_like(rotations)
    identity[..., 3] = 1.0
    rotations = torch.where(
        (rotations.norm(dim=-1, keepdim=True) > 1e-6), rotations, identity
    )
    return Gaussians(
        means=means,
        harmonics=(base_sh + residual_sh.reshape(batch, height * width, 3)).unsqueeze(-2),
        opacities=act_gs.reg_dense_opacities(opacities.float()).reshape(batch, height * width)
        * valid.to(opacities.dtype),
        scales=scales.float()
        .clamp(math.log(1e-6), math.log(0.1))
        .exp()
        .reshape(batch, height * width, 3),
        rotations=rotations,
    )


def _make_state(gaussians: Gaussians) -> StreamingGaussianState:
    batch, count = gaussians.means.shape[:2]
    ids = torch.arange(count, device=gaussians.means.device).unsqueeze(0).expand(batch, -1)
    return StreamingGaussianState(
        gaussians=gaussians,
        stable_ids=ids,
        observation_count=torch.zeros(
            (batch, count), device=gaussians.means.device, dtype=gaussians.means.dtype
        ),
    )


def _historical_detach_mode(cfg: DictConfig) -> str:
    mode = str(
        cfg.get(
            "historical_detach_mode",
            "all" if bool(cfg.get("detach_historical_base", True)) else "none",
        )
    ).lower()
    if mode not in {"all", "means", "none"}:
        raise ValueError(
            "toy.historical_detach_mode must be 'all', 'means', or 'none', "
            f"got {mode!r}."
        )
    return mode


def _render(model, gaussians, c2w, intrinsic, image_shape):
    batch, views = c2w.shape[:2]
    near = torch.full((batch, views), 1e-4, device=c2w.device)
    far = torch.full((batch, views), 100.0, device=c2w.device)
    return model.decoder(gaussians, c2w, intrinsic, near, far, image_shape)


def _bound_tanh_input(value: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        return value * 0.0
    bounded = (torch.tanh(value) * min(float(scale), 0.999)).clamp(-0.999, 0.999)
    return torch.atanh(bounded)


def _mask_gir(gir: DominantGIR, mask: torch.Tensor) -> DominantGIR:
    mask_bool = mask.bool()
    mask_hw = mask_bool[:, 0]
    contributor_ids = gir.contributor_ids
    contributor_weights = gir.contributor_weights
    if contributor_ids is not None:
        contributor_ids = torch.where(
            mask_bool,
            contributor_ids,
            torch.full_like(contributor_ids, -1),
        )
    if contributor_weights is not None:
        contributor_weights = torch.where(
            mask_bool,
            contributor_weights,
            torch.zeros_like(contributor_weights),
        )

    def mask_contributor_evidence(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return torch.where(
            mask_bool,
            value,
            torch.zeros_like(value),
        )

    return replace(
        gir,
        indices=torch.where(mask_hw, gir.indices, torch.full_like(gir.indices, -1)),
        stable_ids=torch.where(mask_hw, gir.stable_ids, torch.full_like(gir.stable_ids, -1)),
        valid=gir.valid & mask_bool,
        dominant_weight=torch.where(mask_bool, gir.dominant_weight, torch.zeros_like(gir.dominant_weight)),
        depth=torch.where(mask_bool, gir.depth, torch.zeros_like(gir.depth)),
        opacity=torch.where(mask_bool, gir.opacity, torch.zeros_like(gir.opacity)),
        scale=torch.where(mask_bool, gir.scale, torch.zeros_like(gir.scale)),
        observation_count=torch.where(
            mask_bool,
            gir.observation_count,
            torch.zeros_like(gir.observation_count),
        ),
        raster_depth=torch.where(
            mask_bool, gir.raster_depth, torch.zeros_like(gir.raster_depth)
        ),
        raster_alpha=torch.where(
            mask_bool, gir.raster_alpha, torch.zeros_like(gir.raster_alpha)
        ),
        contributor_ids=contributor_ids,
        contributor_weights=contributor_weights,
        contributor_depth=mask_contributor_evidence(gir.contributor_depth),
        contributor_opacity=mask_contributor_evidence(gir.contributor_opacity),
        contributor_scale=mask_contributor_evidence(gir.contributor_scale),
        contributor_observation_count=mask_contributor_evidence(
            gir.contributor_observation_count
        ),
    )


def _empty_residual_update_stats(
    reference: torch.Tensor,
    rank_count: int,
) -> dict[str, torch.Tensor]:
    rank_zeros = reference.new_zeros(max(1, int(rank_count)), dtype=torch.float32)
    scalar_zero = reference.new_zeros((), dtype=torch.float32)
    return {
        **{f"rank_{key}": rank_zeros.clone() for key in RANK_UPDATE_LOG_KEYS},
        **{key: scalar_zero.clone() for key in MAP_UPDATE_LOG_KEYS},
    }


@torch.no_grad()
def _residual_update_stats(
    before: StreamingGaussianState,
    after: StreamingGaussianState,
    gir: DominantGIR,
    prediction,
    rank_count: int,
) -> dict[str, torch.Tensor]:
    """Collect the overfit diagnostics without changing core GIR write-back."""
    stats = _empty_residual_update_stats(before.gaussians.means, rank_count)
    if gir.contributor_ids is None or gir.contributor_weights is None:
        return stats
    rank_count = min(rank_count, gir.contributor_ids.shape[1])
    ids = gir.contributor_ids[:, :rank_count]
    weights = gir.contributor_weights[:, :rank_count].float().clamp_min(0.0)
    valid = (ids >= 0) & (ids < before.num_gaussians)
    safe_ids = ids.clamp(min=0, max=max(before.num_gaussians - 1, 0))
    dominant = gir.dominant_weight.float().clamp_min(1e-8)
    relative_weight = (weights / dominant).clamp(0.0, 1.0)
    old_count = before.observation_count.gather(
        1, safe_ids.reshape(safe_ids.shape[0], -1)
    ).reshape_as(safe_ids)

    def ranked(primary: torch.Tensor, value: torch.Tensor | None) -> torch.Tensor:
        if value is not None:
            return value[:, :rank_count]
        return primary[:, None].expand(-1, rank_count, -1, -1, -1)

    gate = ranked(
        prediction.historical_gate,
        prediction.rank_historical_gate,
    ).squeeze(2).sigmoid()
    multiplier = gate * old_count.add(1.0).rsqrt() * relative_weight
    valid_float = valid.float()
    count = valid_float.sum(dim=(0, 2, 3)).clamp_min(1.0)
    stats["rank_valid_ratio"][:rank_count] = valid_float.mean(dim=(0, 2, 3))
    stats["rank_contribution_weight"][:rank_count] = (
        weights * valid_float
    ).sum(dim=(0, 2, 3)) / count
    stats["rank_relative_weight"][:rank_count] = (
        relative_weight * valid_float
    ).sum(dim=(0, 2, 3)) / count
    stats["rank_update_multiplier"][:rank_count] = (
        multiplier * valid_float
    ).sum(dim=(0, 2, 3)) / count

    rank_values = {
        "mean_shift": ranked(
            prediction.delta_mean_camera,
            prediction.rank_delta_mean_camera,
        ),
        "rotation": ranked(
            prediction.delta_rotation,
            prediction.rank_delta_rotation,
        ),
        "log_scale": ranked(
            prediction.delta_log_scale,
            prediction.rank_delta_log_scale,
        ),
        "opacity_logit": ranked(
            prediction.delta_opacity_logit,
            prediction.rank_delta_opacity_logit,
        ),
        "harmonics": ranked(
            prediction.delta_harmonics,
            prediction.rank_delta_harmonics,
        ),
    }
    for name, value in rank_values.items():
        if name == "opacity_logit":
            magnitude = value[:, :, 0].abs()
        elif name == "harmonics":
            magnitude = value.square().mean(dim=2).sqrt()
        else:
            magnitude = value.norm(dim=2)
        stats[f"rank_raw_{name}"][:rank_count] = (
            magnitude * valid_float
        ).sum(dim=(0, 2, 3)) / count
        stats[f"rank_effective_{name}"][:rank_count] = (
            magnitude * multiplier * valid_float
        ).sum(dim=(0, 2, 3)) / count

    mean_shift = (after.gaussians.means - before.gaussians.means).norm(dim=-1)
    rotation_shift = (
        after.gaussians.rotations - before.gaussians.rotations
    ).norm(dim=-1)
    scale_relative = (
        after.gaussians.scales / before.gaussians.scales.clamp_min(1e-8) - 1.0
    ).abs().amax(dim=-1)
    opacity_shift = (
        after.gaussians.opacities - before.gaussians.opacities
    ).abs()
    harmonics_shift = (
        after.gaussians.harmonics - before.gaussians.harmonics
    ).flatten(2).square().mean(dim=-1).sqrt()
    updated = (
        (mean_shift > 1e-10)
        | (rotation_shift > 1e-10)
        | (scale_relative > 1e-10)
        | (opacity_shift > 1e-10)
        | (harmonics_shift > 1e-10)
    )
    updated_float = updated.float()
    updated_count = updated_float.sum().clamp_min(1.0)
    stats["map_updated_ratio"] = updated_float.mean()
    for key, value in (
        ("mean_shift", mean_shift),
        ("rotation", rotation_shift),
        ("scale_relative", scale_relative),
        ("opacity", opacity_shift),
        ("harmonics", harmonics_shift),
    ):
        stats[f"map_{key}_mean"] = (value * updated_float).sum() / updated_count
        stats[f"map_{key}_max"] = value.max()
    return stats


def _predict_updated_state(
    model,
    cfg: DictConfig,
    state: StreamingGaussianState,
    current_feature: torch.Tensor,
    current_rgb: torch.Tensor,
    current_depth: torch.Tensor,
    current_c2w: torch.Tensor,
    current_intrinsic: torch.Tensor,
    image_shape: tuple[int, int],
    global_step: int,
):
    batch = state.gaussians.means.shape[0]
    height, width = image_shape
    residual_topk = max(1, int(cfg.get("residual_topk", 1)))
    independent_heads = bool(cfg.get("residual_independent_heads", False))
    opacity_decay_enabled = bool(cfg.get("opacity_decay_enabled", False))
    opacity_decay_topk = max(1, int(cfg.get("opacity_decay_topk", 4)))
    raw_scale_residual = bool(cfg.get("raw_scale_residual", False))
    raw_opacity_residual = bool(cfg.get("raw_opacity_residual", False))
    raw_rotation_residual = bool(cfg.get("raw_rotation_residual", False))
    raw_harmonics_residual = bool(cfg.get("raw_harmonics_residual", False))
    mean_update_mode = str(cfg.get("mean_update_mode", "relative_depth")).lower()
    if mean_update_mode not in {"relative_depth", "absolute"}:
        raise ValueError(
            "toy.mean_update_mode must be 'relative_depth' or 'absolute', "
            f"got {mean_update_mode!r}."
        )
    append_mode = str(cfg.get("append_mode", "logic")).lower()
    if append_mode not in {"logic", "all"}:
        raise ValueError(
            "toy.append_mode must be 'logic' or 'all', "
            f"got {append_mode!r}."
        )
    gir = DominantGIR.empty(batch, height, width, current_feature.device, current_feature.dtype)
    gir = model._render_old_map_gir_evidence(
        gir,
        state,
        current_c2w,
        current_intrinsic,
        image_shape,
        use_dominant_ids=True,
        min_dominant_weight=float(cfg.dominant_min_weight),
        num_top_contributors=max(
            residual_topk if bool(cfg.residual_enabled) else 1,
            opacity_decay_topk if opacity_decay_enabled else 1,
        ),
    )
    with torch.no_grad():
        valid_depth = current_depth[:, None] > 0
        old_coverage = gir.valid & (
            gir.raster_alpha.float() >= float(cfg.alpha_threshold)
        )
        relative_depth_delta = (
            gir.depth.float() - current_depth[:, None]
        ) / current_depth[:, None].clamp_min(1e-4)
        depth_consistent = (
            old_coverage
            & valid_depth
            & (
                relative_depth_delta.abs()
                <= float(cfg.depth_relative_tolerance)
            )
        )
        current_in_front = (
            old_coverage
            & valid_depth
            & (relative_depth_delta > float(cfg.depth_relative_tolerance))
        )
        old_in_front = (
            old_coverage
            & valid_depth
            & (relative_depth_delta < -float(cfg.depth_relative_tolerance))
        )
        overlap = depth_consistent
        # Keep the geometric rule by default. The all mode is an explicit
        # ablation that appends every valid current-frame Gaussian.
        if append_mode == "all":
            append_mask = valid_depth
        else:
            append_mask = valid_depth & (~old_coverage | current_in_front)
        rgb_error = (gir.rgb.float() - current_rgb.float()).abs().mean(dim=1, keepdim=True)
        rgb_bad = rgb_error >= float(cfg.rgb_bad_threshold)
        bad_mask = (depth_consistent & rgb_bad) | old_in_front

    if not bool(cfg.residual_enabled) and not opacity_decay_enabled:
        stats = {
            "overlap": overlap,
            "old_coverage": old_coverage,
            "depth_consistent": depth_consistent,
            "current_in_front": current_in_front,
            "old_in_front": old_in_front,
            "bad_mask": bad_mask,
            "rgb_error": rgb_error,
            "bad_ratio": bad_mask.float().mean(),
            "overlap_ratio": overlap.float().mean(),
            "old_coverage_ratio": old_coverage.float().mean(),
            "depth_consistent_ratio": depth_consistent.float().mean(),
            "current_in_front_ratio": current_in_front.float().mean(),
            "old_in_front_ratio": old_in_front.float().mean(),
            "append_mask": append_mask,
            "unique_gs": 0,
            "updated_indices": torch.empty(
                0, device=current_feature.device, dtype=torch.long
            ),
            "residual_regularization": current_feature.new_zeros(()),
            "opacity_decay_candidate_probability": current_feature.new_zeros(()),
            "opacity_decay_candidate_count": current_feature.new_zeros(()),
            "opacity_decay_mass_ratio": current_feature.new_zeros(()),
            **_empty_residual_update_stats(current_feature, residual_topk),
        }
        return state, stats

    masked_gir = _mask_gir(gir, bad_mask)
    confidence = torch.ones_like(current_depth[:, None])
    with torch.autocast("cuda", enabled=False):
        prediction = model.gir_update_head(
            current_feature.float(),
            current_rgb.float(),
            current_depth[:, None].float(),
            confidence.float(),
            masked_gir,
            num_historical_predictions=(
                residual_topk
                if independent_heads and bool(cfg.residual_enabled)
                else 1
            ),
            decay_gir=gir,
            num_decay_predictions=(
                opacity_decay_topk if opacity_decay_enabled else 0
            ),
        )
    residual_channels = {
        "mean": bool(cfg.get("residual_mean_enabled", True)),
        "rotation": bool(cfg.get("residual_rotation_enabled", True)),
        "scale": bool(cfg.get("residual_scale_enabled", True)),
        "opacity": bool(cfg.get("residual_opacity_enabled", True)),
        "harmonics": bool(cfg.get("residual_harmonics_enabled", True)),
    }

    def disable_channel(primary_name: str, ranked_name: str) -> None:
        primary = getattr(prediction, primary_name)
        setattr(prediction, primary_name, torch.zeros_like(primary))
        ranked = getattr(prediction, ranked_name)
        if ranked is not None:
            setattr(prediction, ranked_name, torch.zeros_like(ranked))

    if not residual_channels["mean"]:
        disable_channel("delta_mean_camera", "rank_delta_mean_camera")
    if not residual_channels["rotation"]:
        disable_channel("delta_rotation", "rank_delta_rotation")
    if not residual_channels["scale"]:
        disable_channel("delta_log_scale", "rank_delta_log_scale")
    if not residual_channels["opacity"]:
        disable_channel("delta_opacity_logit", "rank_delta_opacity_logit")
    if not residual_channels["harmonics"]:
        disable_channel("delta_harmonics", "rank_delta_harmonics")

    if mean_update_mode == "relative_depth":
        prediction.delta_mean_camera = _bound_tanh_input(
            prediction.delta_mean_camera, float(cfg.mean_relative_scale)
        )
    if not raw_rotation_residual:
        prediction.delta_rotation = _bound_tanh_input(
            prediction.delta_rotation, float(cfg.rotation_scale)
        )
    if not raw_harmonics_residual:
        prediction.delta_harmonics = _bound_tanh_input(
            prediction.delta_harmonics, float(cfg.harmonics_scale)
        )
    if prediction.rank_delta_mean_camera is not None:
        if mean_update_mode == "relative_depth":
            prediction.rank_delta_mean_camera = _bound_tanh_input(
                prediction.rank_delta_mean_camera,
                float(cfg.mean_relative_scale),
            )
        if not raw_rotation_residual:
            prediction.rank_delta_rotation = _bound_tanh_input(
                prediction.rank_delta_rotation, float(cfg.rotation_scale)
            )
        if not raw_harmonics_residual:
            prediction.rank_delta_harmonics = _bound_tanh_input(
                prediction.rank_delta_harmonics, float(cfg.harmonics_scale)
            )
    # Selection is heuristic in this experiment. Do not let the legacy -4 gate
    # suppress an otherwise selected old-GS update.
    prediction.historical_gate = torch.full_like(prediction.historical_gate, 12.0)
    if prediction.rank_historical_gate is not None:
        prediction.rank_historical_gate = torch.full_like(
            prediction.rank_historical_gate, 12.0
        )
    if bool(cfg.residual_enabled):
        updated = state.update_historical(
            masked_gir,
            prediction,
            current_c2w,
            num_contributors=residual_topk,
            mean_update_mode=mean_update_mode,
            raw_scale_residual=raw_scale_residual,
            raw_opacity_residual=raw_opacity_residual,
            raw_rotation_residual=raw_rotation_residual,
            raw_harmonics_residual=raw_harmonics_residual,
        )
        update_stats = _residual_update_stats(
            state,
            updated,
            masked_gir,
            prediction,
            residual_topk,
        )
    else:
        updated = state
        update_stats = _empty_residual_update_stats(
            current_feature, residual_topk
        )

    decay_stats = {
        "candidate_probability_mean": current_feature.new_zeros(()),
        "candidate_count": current_feature.new_zeros(()),
        "decay_opacity_mass_ratio": current_feature.new_zeros(()),
    }
    if opacity_decay_enabled:
        if prediction.rank_decay_logits is None:
            raise RuntimeError(
                "Opacity decay is enabled, but no contributor-conditioned "
                "decay logits were produced."
            )
        warmup_steps = max(0, int(cfg.get("opacity_decay_warmup_steps", 0)))
        decay_schedule = (
            1.0
            if warmup_steps == 0
            else min(1.0, max(0.0, float(global_step) / warmup_steps))
        )
        updated, decay_stats = updated.decay_historical_opacity(
            gir,
            prediction.rank_decay_logits,
            min_observations=max(
                1, int(cfg.get("opacity_decay_min_observations", 2))
            ),
            temperature=float(cfg.get("opacity_decay_temperature", 1.0)),
            decay_strength=float(cfg.get("opacity_decay_strength", 0.02)),
            decay_schedule=decay_schedule,
            min_contributor_weight=float(cfg.dominant_min_weight),
            physical_prune=(
                not model.training
                and bool(cfg.get("opacity_decay_test_prune_enabled", False))
            ),
            prune_threshold=float(
                cfg.get("opacity_decay_prune_threshold", 0.005)
            ),
        )

    if bool(cfg.residual_enabled) and residual_topk > 1:
        if masked_gir.contributor_ids is None:
            raise RuntimeError(
                "Top-k residual update requires contributor IDs from the GIR renderer."
            )
        updated_indices = masked_gir.contributor_ids
        updated_indices = updated_indices[updated_indices >= 0].detach()
    elif bool(cfg.residual_enabled):
        updated_indices = masked_gir.indices[masked_gir.indices >= 0].detach()
    else:
        updated_indices = torch.empty(
            0, device=current_feature.device, dtype=torch.long
        )

    if bool(cfg.residual_enabled) and prediction.rank_delta_mean_camera is not None:
        residual_terms = [
            prediction.rank_delta_mean_camera,
            prediction.rank_delta_rotation,
            prediction.rank_delta_log_scale,
            prediction.rank_delta_opacity_logit,
            prediction.rank_delta_harmonics,
        ]
        mask_float = bad_mask[:, None].to(current_feature.dtype)
        residual_regularization = sum(
            (term.square() * mask_float).sum()
            / (
                mask_float.sum() * term.shape[1] * term.shape[2]
            ).clamp_min(1.0)
            for term in residual_terms
        )
    elif bool(cfg.residual_enabled):
        residual_terms = [
            prediction.delta_mean_camera,
            prediction.delta_rotation,
            prediction.delta_log_scale,
            prediction.delta_opacity_logit,
            prediction.delta_harmonics,
        ]
        mask_float = bad_mask.to(current_feature.dtype)
        residual_regularization = sum(
            (term.square() * mask_float).sum()
            / (mask_float.sum() * term.shape[1]).clamp_min(1.0)
            for term in residual_terms
        )
    else:
        residual_regularization = current_feature.new_zeros(())
    stats = {
        "overlap": overlap,
        "old_coverage": old_coverage,
        "depth_consistent": depth_consistent,
        "current_in_front": current_in_front,
        "old_in_front": old_in_front,
        "bad_mask": bad_mask,
        "rgb_error": rgb_error,
        "bad_ratio": bad_mask.float().mean(),
        "overlap_ratio": overlap.float().mean(),
        "old_coverage_ratio": old_coverage.float().mean(),
        "depth_consistent_ratio": depth_consistent.float().mean(),
        "current_in_front_ratio": current_in_front.float().mean(),
        "old_in_front_ratio": old_in_front.float().mean(),
        "append_mask": append_mask,
        "unique_gs": torch.unique(updated_indices).numel(),
        "updated_indices": updated_indices,
        "residual_regularization": residual_regularization,
        "opacity_decay_candidate_probability": decay_stats[
            "candidate_probability_mean"
        ],
        "opacity_decay_candidate_count": decay_stats["candidate_count"],
        "opacity_decay_mass_ratio": decay_stats[
            "decay_opacity_mass_ratio"
        ],
        **update_stats,
    }
    return updated, stats


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(prediction.dtype)
    if mask.dim() == prediction.dim() - 1:
        mask = mask.unsqueeze(2)
    if mask.shape[2] == 1 and prediction.shape[2] != 1:
        mask = mask.expand(-1, -1, prediction.shape[2], -1, -1)
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def _cache_dav3_depth_targets(
    images: torch.Tensor,
    image_shape: tuple[int, int],
    gt_c2w: torch.Tensor,
    gt_intrinsic_px: torch.Tensor,
    checkpoint_path: str,
    align_to_gt_camera_scale: bool,
) -> torch.Tensor:
    from src.loss.dav3.src.depth_anything_3.api import DepthAnything3

    device = images.device
    batch, views = images.shape[:2]
    checkpoint = Path(to_absolute_path(str(checkpoint_path)))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DAV3 checkpoint does not exist: {checkpoint}")
    print(
        f"Caching DAV3 teacher depth for {batch} scene(s), {views} views each..."
    )
    teacher = DepthAnything3(checkpoint_path=str(checkpoint)).to(
        device=device
    ).eval()
    scene_depths = []
    for batch_idx in range(batch):
        image_list = [
            (
                image.permute(1, 2, 0)
                .detach()
                .float()
                .clamp(0, 1)
                .cpu()
                .numpy()
                * 255.0
            )
            .round()
            .astype(np.uint8)
            for image in images[batch_idx]
        ]
        scene_w2c = None
        scene_intrinsic = None
        if align_to_gt_camera_scale:
            scene_w2c = torch.linalg.inv(
                gt_c2w[batch_idx].detach().float()
            ).cpu().numpy()
            scene_intrinsic = (
                gt_intrinsic_px[batch_idx].detach().float().cpu().numpy()
            )
        inference_output = teacher.inference(
            image_list,
            extrinsics=scene_w2c,
            intrinsics=scene_intrinsic,
            align_to_input_ext_scale=align_to_gt_camera_scale,
        )
        prediction = (
            inference_output[0]
            if isinstance(inference_output, tuple)
            else inference_output
        )
        scene_depth = torch.from_numpy(np.asarray(prediction.depth)).to(
            device=device, dtype=torch.float32
        )
        if scene_depth.ndim == 4 and scene_depth.shape[-1] == 1:
            scene_depth = scene_depth.squeeze(-1)
        if scene_depth.ndim != 3 or scene_depth.shape[0] != views:
            raise ValueError(
                "DAV3 depth must have shape [V,H,W] for each scene, got "
                f"{tuple(scene_depth.shape)} for batch item {batch_idx}."
            )
        scene_depth = F.interpolate(
            scene_depth[:, None],
            size=image_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        scene_depths.append(scene_depth)
    teacher_depth = torch.stack(scene_depths, dim=0)
    del prediction, teacher
    torch.cuda.empty_cache()
    valid = torch.isfinite(teacher_depth) & (teacher_depth > 0)
    if not bool(valid.any()):
        raise ValueError("DAV3 produced no finite positive teacher depth values.")
    print(
        "Cached DAV3 depth: "
        f"valid={valid.float().mean().item():.2%}, "
        f"min={teacher_depth[valid].min().item():.4f}, "
        f"median={teacher_depth[valid].median().item():.4f}, "
        f"max={teacher_depth[valid].max().item():.4f}"
    )
    return teacher_depth


@torch.no_grad()
def _dav3_gt_depth_metrics(
    dav3_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_gt: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compare DAV3 directly with GT using one scale per input sequence.

    DAV3 has arbitrary metric scale.  The aligned metrics fit one shared
    multiplicative scale over all views and pixels of each batch item.  This
    matches the sequence-level scale-only log-depth supervision used by the
    main pipeline and avoids hiding cross-view scale errors with per-view
    alignment.
    """
    teacher = dav3_depth.squeeze(-1).float()
    gt = gt_depth.squeeze(-1).float()
    if teacher.shape != gt.shape or teacher.dim() != 4:
        raise ValueError(
            "DAV3 and GT depth must both have shape [B,V,H,W], got "
            f"teacher={tuple(teacher.shape)}, gt={tuple(gt.shape)}."
        )
    valid = (
        torch.isfinite(teacher)
        & torch.isfinite(gt)
        & (teacher > 0)
        & (gt > 0)
    )
    if valid_gt is not None:
        valid = valid & valid_gt.bool()
    valid_float = valid.to(torch.float32)
    reduce_dims = (1, 2, 3)
    count = valid_float.sum(dim=reduce_dims).clamp_min(1.0)

    safe_teacher = torch.where(valid, teacher, torch.ones_like(teacher))
    safe_gt = torch.where(valid, gt, torch.ones_like(gt))
    log_scale = (
        (safe_gt.clamp_min(1e-6).log() - safe_teacher.clamp_min(1e-6).log())
        * valid_float
    ).sum(dim=reduce_dims) / count
    scale = log_scale.clamp(min=-20.0, max=20.0).exp()

    aligned = safe_teacher * scale[:, None, None, None]
    aligned = torch.where(valid, aligned, torch.zeros_like(aligned))
    safe_teacher = torch.where(valid, safe_teacher, torch.zeros_like(safe_teacher))
    safe_gt = torch.where(valid, safe_gt, torch.zeros_like(safe_gt))
    denominator = torch.where(valid, safe_gt, torch.ones_like(safe_gt))

    def masked_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * valid_float).sum() / valid_float.sum().clamp_min(1.0)

    aligned_error = aligned - safe_gt
    raw_error = safe_teacher - safe_gt
    aligned_abs_rel = masked_mean(aligned_error.abs() / denominator)
    raw_abs_rel = masked_mean(raw_error.abs() / denominator)
    aligned_rmse = torch.sqrt(
        (aligned_error.square() * valid_float).sum()
        / valid_float.sum().clamp_min(1.0)
    )
    raw_rmse = torch.sqrt(
        (raw_error.square() * valid_float).sum()
        / valid_float.sum().clamp_min(1.0)
    )
    aligned_log_l1 = masked_mean(
        (
            aligned.clamp_min(1e-6).log()
            - safe_gt.clamp_min(1e-6).log()
        ).abs()
    )
    usable = valid_float.sum(dim=reduce_dims) > 0
    return {
        "depth_dav3_gt_abs_rel_aligned": float(aligned_abs_rel.item()),
        "depth_dav3_gt_rmse_aligned": float(aligned_rmse.item()),
        "depth_dav3_gt_log_l1_aligned": float(aligned_log_l1.item()),
        "depth_dav3_gt_scale": float(
            (scale * usable.float()).sum() / usable.float().sum().clamp_min(1.0)
        ),
        "depth_dav3_gt_abs_rel_raw": float(raw_abs_rel.item()),
        "depth_dav3_gt_rmse_raw": float(raw_rmse.item()),
        "depth_dav3_gt_valid_ratio": float(valid_float.mean().item()),
    }


def _sequence_scale_only_log_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    absolute_log_scale_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequence-level scale-only log-depth L1 used by the main pipeline.

    One scale is fitted across every view and pixel of each sequence item.
    There is deliberately no additive shift and no per-view alignment, so
    relative depth between views remains supervised.
    """
    prediction = prediction.squeeze(-1).float()
    target = target.squeeze(-1).float()
    if prediction.shape != target.shape or prediction.dim() != 4:
        raise ValueError(
            "Expected prediction and target with shape [B,V,H,W], got "
            f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}."
        )
    if valid is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    valid = (
        valid
        & torch.isfinite(prediction)
        & torch.isfinite(target)
        & (prediction > 0)
        & (target > 0)
    )
    valid_float = valid.to(prediction.dtype)
    safe_prediction = torch.where(valid, prediction, torch.ones_like(prediction))
    safe_target = torch.where(valid, target, torch.ones_like(target))
    log_error = (
        safe_prediction.clamp_min(1e-6).log()
        - safe_target.clamp_min(1e-6).log()
    )
    reduce_dims = (1, 2, 3)
    count = valid_float.sum(dim=reduce_dims, keepdim=True).clamp_min(1.0)
    sequence_log_error = (
        (log_error * valid_float).sum(dim=reduce_dims, keepdim=True) / count
    )
    centered_error = log_error - sequence_log_error
    structure_loss = (
        (centered_error.abs() * valid_float).sum()
        / valid_float.sum().clamp_min(1.0)
    )
    valid_sequences = (
        valid_float.sum(dim=reduce_dims, keepdim=True) > 0
    ).to(prediction.dtype)
    absolute_scale_loss = F.smooth_l1_loss(
        sequence_log_error,
        torch.zeros_like(sequence_log_error),
        reduction="none",
    )
    absolute_scale_loss = (
        (absolute_scale_loss * valid_sequences).sum()
        / valid_sequences.sum().clamp_min(1.0)
    )
    loss = structure_loss + float(absolute_log_scale_weight) * absolute_scale_loss
    # This is the multiplicative factor that maps prediction to target. It is
    # diagnostic only and must not retain the training graph.
    alignment_scale = (-sequence_log_error).exp().flatten().detach()
    return (
        torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0),
        alignment_scale,
        structure_loss,
        absolute_scale_loss,
    )


def _absolute_gt_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.squeeze(-1).float()
    target = target.float()
    valid = (
        valid.bool()
        & torch.isfinite(prediction)
        & torch.isfinite(target)
        & (target > 0)
    )
    valid_float = valid.to(prediction.dtype)
    prediction = torch.nan_to_num(
        prediction, nan=1e-4, posinf=100.0, neginf=1e-4
    ).clamp(1e-4, 100.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        (prediction - target).abs() * valid_float
    ).sum() / valid_float.sum().clamp_min(1.0)


def _psnr(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mse = (prediction - target).square().mean()
    else:
        mse = _masked_mse(prediction, target, mask)
    return float((-10.0 * torch.log10(mse.clamp_min(1e-10))).item())


def _save_image(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().float().clamp(0, 1).cpu()
    if array.dim() == 4:
        array = array[0]
    array = (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def _save_mask(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().float().squeeze().clamp(0, 1).cpu().numpy()
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


def _extract_features(model, tokens, patch_start_idx, images, image_shape, use_amp):
    del use_amp
    with torch.autocast("cuda", enabled=False):
        features, predicted_depth, depth_confidence, _ = model.encoder.gaussian_param_head(
            tokens,
            images.float(),
            patch_start_idx=patch_start_idx,
            image_size=image_shape,
        )
    return features, predicted_depth, depth_confidence


def _as_homogeneous_transform(extrinsic: torch.Tensor) -> torch.Tensor:
    transform = torch.eye(
        4, device=extrinsic.device, dtype=extrinsic.dtype
    ).view(1, 1, 4, 4).expand(*extrinsic.shape[:2], -1, -1).clone()
    transform[..., :3, :4] = extrinsic
    return transform


def _predict_camera_geometry(model, tokens, image_shape, gt_c2w):
    pose_encoding = model.encoder.camera_head(tokens)[-1].float()
    # The decoder is scale-invariant for nonzero quaternions, but explicit
    # normalization avoids an undefined rotation if a trainable head approaches 0.
    raw_quaternion = pose_encoding[..., 3:7]
    quaternion_norm = raw_quaternion.norm(dim=-1, keepdim=True)
    normalized_quaternion = raw_quaternion / quaternion_norm.clamp_min(1e-6)
    identity_quaternion = torch.zeros_like(normalized_quaternion)
    identity_quaternion[..., 3] = 1.0
    safe_quaternion = torch.where(
        quaternion_norm > 1e-6,
        normalized_quaternion,
        identity_quaternion,
    )
    safe_pose_encoding = torch.cat(
        [
            pose_encoding[..., :3],
            safe_quaternion,
            pose_encoding[..., 7:9].clamp(1e-3, math.pi - 1e-3),
        ],
        dim=-1,
    )
    predicted_w2c, predicted_intrinsic_px = pose_encoding_to_extri_intri(
        safe_pose_encoding,
        image_size_hw=image_shape,
    )
    predicted_c2w = torch.linalg.inv(_as_homogeneous_transform(predicted_w2c))

    # Camera-head poses have an arbitrary world gauge. Anchor the predicted
    # sequence to the first GT input camera before using it for reconstruction.
    predicted_relative_c2w = (
        torch.linalg.inv(predicted_c2w[:, :1]) @ predicted_c2w
    )
    aligned_predicted_c2w = gt_c2w[:, :1] @ predicted_relative_c2w
    gt_relative_c2w = torch.linalg.inv(gt_c2w[:, :1]) @ gt_c2w
    return {
        "pose_encoding": pose_encoding,
        "c2w": aligned_predicted_c2w,
        "relative_c2w": predicted_relative_c2w,
        "gt_relative_c2w": gt_relative_c2w,
        "intrinsic_px": predicted_intrinsic_px,
    }


def _camera_supervision_losses(
    camera_prediction: dict[str, torch.Tensor],
    gt_intrinsic_px: torch.Tensor,
    image_shape: tuple[int, int],
    cfg: DictConfig,
) -> dict[str, torch.Tensor]:
    predicted_relative = camera_prediction["relative_c2w"]
    gt_relative = camera_prediction["gt_relative_c2w"]
    translation = F.smooth_l1_loss(
        predicted_relative[..., :3, 3], gt_relative[..., :3, 3]
    )
    rotation = F.mse_loss(
        predicted_relative[..., :3, :3], gt_relative[..., :3, :3]
    )
    height, width = image_shape
    gt_fov_h = 2.0 * torch.atan(
        (height / 2.0) / gt_intrinsic_px[..., 1, 1].clamp_min(1e-4)
    )
    gt_fov_w = 2.0 * torch.atan(
        (width / 2.0) / gt_intrinsic_px[..., 0, 0].clamp_min(1e-4)
    )
    gt_fov = torch.stack([gt_fov_h, gt_fov_w], dim=-1)
    # Supervise the raw FOV encoding. The bounded copy used by rendering may
    # saturate, but this loss must retain a gradient that can pull it back.
    focal = F.smooth_l1_loss(
        camera_prediction["pose_encoding"][..., 7:9],
        gt_fov,
    )
    total = (
        float(cfg.camera_translation_weight) * translation
        + float(cfg.camera_rotation_weight) * rotation
        + float(cfg.camera_focal_weight) * focal
    )
    return {
        "total": total,
        "translation": translation,
        "rotation": rotation,
        "focal": focal,
    }


def _forward_experiment(
    model,
    cfg,
    data,
    tokens,
    patch_start_idx,
    use_amp,
    global_step: int,
    render_all_inputs: bool = False,
    render_heldout_loss: bool = False,
):
    images = data["images"]
    gt_depth = data["depth"]
    gt_c2w = data["c2w"]
    gt_intrinsic_px = data["intrinsic_px"]
    image_shape = data["image_shape"]
    features, predicted_depth, depth_confidence = _extract_features(
        model, tokens, patch_start_idx, images, image_shape, use_amp
    )
    camera_prediction = _predict_camera_geometry(
        model, tokens, image_shape, gt_c2w
    )
    camera_source = str(cfg.camera_source).lower()
    depth_source = str(cfg.depth_source).lower()
    if camera_source not in {"gt", "predicted"}:
        raise ValueError(
            f"toy.camera_source must be 'gt' or 'predicted', got {cfg.camera_source!r}."
        )
    if depth_source not in {"gt", "predicted"}:
        raise ValueError(
            f"toy.depth_source must be 'gt' or 'predicted', got {cfg.depth_source!r}."
        )

    if camera_source == "predicted":
        c2w = camera_prediction["c2w"]
        intrinsic_px = camera_prediction["intrinsic_px"]
        if bool(cfg.detach_predicted_camera):
            c2w = c2w.detach()
            intrinsic_px = intrinsic_px.detach()
    else:
        c2w = gt_c2w
        intrinsic_px = gt_intrinsic_px
    intrinsic = _normalized_intrinsic(
        intrinsic_px, image_shape[0], image_shape[1]
    )

    predicted_depth_hw = predicted_depth.squeeze(-1).float()
    if depth_source == "predicted":
        depth = torch.nan_to_num(
            predicted_depth_hw, nan=1e-4, posinf=100.0, neginf=1e-4
        ).clamp(1e-4, 100.0)
        if bool(cfg.detach_predicted_depth):
            depth = depth.detach()
    else:
        depth = gt_depth
    geometry_valid_depth = torch.isfinite(depth) & (depth > 0)
    base = _build_base_gaussians(
        model,
        features[:, 0],
        images[:, 0],
        depth[:, 0],
        c2w[:, 0],
        intrinsic_px[:, 0],
    )
    state = _make_state(base)
    base_render = _render(model, base, c2w[:, :1], intrinsic[:, :1], image_shape)
    max_replay_views = int(cfg.max_replay_views)
    if max_replay_views == 0 or max_replay_views < -1:
        raise ValueError(
            f"toy.max_replay_views must be -1 or a positive integer, got {max_replay_views}."
        )
    historical_detach_mode = _historical_detach_mode(cfg)

    steps = []
    for view_idx in range(1, images.shape[1]):
        if historical_detach_mode == "all":
            state = state.detach()
        elif historical_detach_mode == "means":
            # Fix historical positions while preserving cross-view gradients
            # for SH, opacity, scale, and rotation.
            state = state.detach_means()
        state, stats = _predict_updated_state(
            model,
            cfg,
            state,
            features[:, view_idx],
            images[:, view_idx],
            depth[:, view_idx],
            c2w[:, view_idx],
            intrinsic[:, view_idx],
            image_shape,
            global_step,
        )
        append_enabled = bool(cfg.append_new_gs_enabled)
        if append_enabled:
            previous_gs_count = state.num_gaussians
            current_gaussians = _build_base_gaussians(
                model,
                features[:, view_idx],
                images[:, view_idx],
                depth[:, view_idx],
                c2w[:, view_idx],
                intrinsic_px[:, view_idx],
            )
            state = state.append(
                current_gaussians,
                stats["append_mask"].to(current_gaussians.opacities.dtype),
                prune_threshold=0.5,
            )
            appended_gs = state.num_gaussians - previous_gs_count
        else:
            appended_gs = 0
        stats["appended_gs"] = appended_gs
        stats["append_ratio"] = stats["append_mask"].float().mean()
        current_render = _render(
            model,
            state.gaussians,
            c2w[:, view_idx : view_idx + 1],
            intrinsic[:, view_idx : view_idx + 1],
            image_shape,
        )
        replay_start = 0 if max_replay_views < 0 else max(0, view_idx - max_replay_views)
        replay_indices = list(range(replay_start, view_idx))
        replay_render = _render(
            model,
            state.gaussians,
            c2w[:, replay_indices],
            intrinsic[:, replay_indices],
            image_shape,
        )
        steps.append(
            {
                "view_idx": view_idx,
                "current_render": current_render,
                "replay_indices": replay_indices,
                "replay_render": replay_render,
                "stats": stats,
            }
        )

    final_input_render = None
    if render_all_inputs:
        final_input_render = _render(
            model, state.gaussians, c2w, intrinsic, image_shape
        )
    heldout_loss_render = None
    if render_heldout_loss:
        heldout_loss = data["heldout_loss"]
        heldout_loss_render = _render(
            model,
            state.gaussians,
            heldout_loss["c2w"],
            heldout_loss["intrinsic"],
            image_shape,
        )
    return {
        "base": base,
        "final_state": state,
        "base_render": base_render,
        "steps": steps,
        "final_input_render": final_input_render,
        "heldout_loss_render": heldout_loss_render,
        "predicted_depth": predicted_depth,
        "depth_confidence": depth_confidence,
        "camera_prediction": camera_prediction,
        "camera_losses": _camera_supervision_losses(
            camera_prediction, gt_intrinsic_px, image_shape, cfg
        ),
        "geometry_c2w": c2w,
        "geometry_intrinsic_px": intrinsic_px,
        "geometry_intrinsic": intrinsic,
        "geometry_depth": depth,
        "geometry_valid_depth": geometry_valid_depth,
    }


def _summarize_step_stats(steps: list[dict[str, Any]]) -> dict[str, Any]:
    stats = [step["stats"] for step in steps]
    updated_indices = [item["updated_indices"] for item in stats if item["updated_indices"].numel()]
    unique_gs = (
        int(torch.unique(torch.cat(updated_indices)).numel()) if updated_indices else 0
    )
    summary = {
        "overlap_ratio": torch.stack([item["overlap_ratio"] for item in stats]).mean(),
        "old_coverage_ratio": torch.stack(
            [item["old_coverage_ratio"] for item in stats]
        ).mean(),
        "depth_consistent_ratio": torch.stack(
            [item["depth_consistent_ratio"] for item in stats]
        ).mean(),
        "current_in_front_ratio": torch.stack(
            [item["current_in_front_ratio"] for item in stats]
        ).mean(),
        "old_in_front_ratio": torch.stack(
            [item["old_in_front_ratio"] for item in stats]
        ).mean(),
        "bad_ratio": torch.stack([item["bad_ratio"] for item in stats]).mean(),
        "append_ratio": torch.stack([item["append_ratio"] for item in stats]).mean(),
        "appended_gs": sum(int(item["appended_gs"]) for item in stats),
        "unique_gs": unique_gs,
        "residual_regularization": torch.stack(
            [item["residual_regularization"] for item in stats]
        ).mean(),
        "opacity_decay_candidate_probability": torch.stack(
            [item["opacity_decay_candidate_probability"] for item in stats]
        ).mean(),
        "opacity_decay_candidate_count": torch.stack(
            [item["opacity_decay_candidate_count"] for item in stats]
        ).mean(),
        "opacity_decay_mass_ratio": torch.stack(
            [item["opacity_decay_mass_ratio"] for item in stats]
        ).mean(),
    }
    for key in RANK_UPDATE_LOG_KEYS:
        tensor_key = f"rank_{key}"
        summary[tensor_key] = torch.stack(
            [item[tensor_key] for item in stats]
        ).mean(dim=0)
    for key in MAP_UPDATE_LOG_KEYS:
        summary[key] = torch.stack([item[key] for item in stats]).mean()
    return summary


def _flatten_residual_update_stats(summary: dict[str, Any]) -> dict[str, float]:
    values = {}
    rank_count = min(
        MAX_LOGGED_RESIDUAL_RANKS,
        int(summary["rank_valid_ratio"].numel()),
    )
    for rank_idx in range(rank_count):
        for key in RANK_UPDATE_LOG_KEYS:
            values[f"rank{rank_idx + 1}_{key}"] = float(
                summary[f"rank_{key}"][rank_idx].item()
            )
    for key in MAP_UPDATE_LOG_KEYS:
        values[key] = float(summary[key].item())
    return values


@torch.no_grad()
def _evaluate(model, cfg, data, tokens, patch_start_idx, use_amp, step, output_dir):
    model.encoder.gaussian_param_head.eval()
    model.encoder.camera_head.eval()
    model.encoder.gs_head.eval()
    model.gir_update_head.eval()
    result = _forward_experiment(
        model,
        cfg,
        data,
        tokens,
        patch_start_idx,
        use_amp,
        global_step=step,
        render_all_inputs=True,
        render_heldout_loss=True,
    )
    base = result["base"]
    updated = result["final_state"].gaussians
    base_render = result["base_render"]
    global_render = result["final_input_render"]
    steps = result["steps"]

    ply_dir = output_dir / "gaussians"
    for batch_idx in range(updated.means.shape[0]):
        batch_suffix = "" if updated.means.shape[0] == 1 else f"_batch{batch_idx:02d}"
        ply_path = ply_dir / f"global_step_{step:06d}{batch_suffix}.ply"
        export_ply(
            updated.means[batch_idx],
            updated.scales[batch_idx],
            updated.rotations[batch_idx],
            updated.harmonics[batch_idx],
            updated.opacities[batch_idx],
            ply_path,
            shift_and_scale=False,
            filter_gaussians=False,
        )
        print(
            f"[eval] exported {updated.means.shape[1]} unfiltered GS to {ply_path}"
        )

    summary = _summarize_step_stats(steps)
    last_stats = steps[-1]["stats"]
    images = data["images"]
    input_psnr = [
        _psnr(global_render.color[:, view_idx], images[:, view_idx])
        for view_idx in range(images.shape[1])
    ]
    metrics = {
        "step": step,
        "residual_topk": max(1, int(cfg.get("residual_topk", 1))),
        "residual_independent_heads": bool(
            cfg.get("residual_independent_heads", False)
        ),
        "residual_contributor_conditioned": bool(
            cfg.get("residual_contributor_conditioned", False)
        ),
        "append_mode": str(cfg.get("append_mode", "logic")).lower(),
        "mean_update_mode": str(
            cfg.get("mean_update_mode", "relative_depth")
        ).lower(),
        "camera_source": str(cfg.camera_source).lower(),
        "depth_source": str(cfg.depth_source).lower(),
        "base_first_psnr": _psnr(base_render.color[:, 0], images[:, 0]),
        "global_first_psnr": input_psnr[0],
        "global_current_psnr": input_psnr[-1],
        "input_mean_psnr": float(sum(input_psnr) / len(input_psnr)),
        "overlap_ratio": float(summary["overlap_ratio"].item()),
        "old_coverage_ratio": float(summary["old_coverage_ratio"].item()),
        "depth_consistent_ratio": float(
            summary["depth_consistent_ratio"].item()
        ),
        "current_in_front_ratio": float(
            summary["current_in_front_ratio"].item()
        ),
        "old_in_front_ratio": float(summary["old_in_front_ratio"].item()),
        "bad_ratio": float(summary["bad_ratio"].item()),
        "opacity_decay_enabled": bool(
            cfg.get("opacity_decay_enabled", False)
        ),
        "opacity_decay_topk": int(cfg.get("opacity_decay_topk", 4)),
        "opacity_decay_candidate_probability": float(
            summary["opacity_decay_candidate_probability"].item()
        ),
        "opacity_decay_candidate_count": float(
            summary["opacity_decay_candidate_count"].item()
        ),
        "opacity_decay_mass_ratio": float(
            summary["opacity_decay_mass_ratio"].item()
        ),
        "append_ratio": float(summary["append_ratio"].item()),
        "appended_gs": int(summary["appended_gs"]),
        "final_gs_count": int(updated.means.shape[1]),
        "unique_updated_gs": int(summary["unique_gs"]),
        "base_scale_min": float(base.scales.min().item()),
        "base_scale_median": float(base.scales.median().item()),
        "updated_scale_min": float(updated.scales.min().item()),
        "updated_scale_median": float(updated.scales.median().item()),
    }
    metrics.update(_flatten_residual_update_stats(summary))
    overlap = last_stats["overlap"].unsqueeze(1)
    if bool(overlap.any()):
        metrics["global_current_overlap_psnr"] = _psnr(
            global_render.color[:, -1:], images[:, -1:], overlap
        )
    last_append_mask = last_stats["append_mask"].unsqueeze(1)
    if bool(last_append_mask.any()):
        metrics["global_current_hole_psnr"] = _psnr(
            global_render.color[:, -1:], images[:, -1:], last_append_mask
        )
    for frame_id, frame_psnr in zip(data["ids"], input_psnr):
        metrics[f"global_frame_{frame_id:03d}_psnr"] = frame_psnr
    predicted_depth = result["predicted_depth"].squeeze(-1).float()
    if bool(data.get("has_gt_depth", True)):
        target_depth = data["depth"].float()
        valid_depth = data["valid_depth"] & torch.isfinite(predicted_depth)
        predicted_depth = torch.nan_to_num(
            predicted_depth, nan=1e-4, posinf=100.0, neginf=1e-4
        ).clamp(1e-4, 100.0)
        valid_float = valid_depth.to(predicted_depth.dtype)
        metrics["depth_abs_rel"] = float(
            (
                (predicted_depth - target_depth).abs()
                / target_depth.clamp_min(1e-4)
                * valid_float
            ).sum().div(valid_float.sum().clamp_min(1.0)).item()
        )
        metrics["depth_rmse"] = float(
            torch.sqrt(
                (
                    (predicted_depth - target_depth).square() * valid_float
                ).sum().div(valid_float.sum().clamp_min(1.0))
            ).item()
        )
    else:
        metrics["gt_depth_available"] = False
    if "dav3_depth" in data:
        (
            dav3_loss,
            dav3_scale,
            dav3_structure_loss,
            dav3_absolute_scale_loss,
        ) = _sequence_scale_only_log_depth_loss(
            result["predicted_depth"],
            data["dav3_depth"],
            absolute_log_scale_weight=float(
                cfg.get("depth_absolute_log_scale_weight", 0.1)
            ),
        )
        metrics["depth_dav3_loss"] = float(dav3_loss.item())
        metrics["depth_dav3_structure_log_l1"] = float(
            dav3_structure_loss.item()
        )
        metrics["depth_dav3_absolute_log_scale"] = float(
            dav3_absolute_scale_loss.item()
        )
        metrics["depth_dav3_sequence_scale_median"] = float(dav3_scale.median().item())
        if bool(data.get("has_gt_depth", True)):
            metrics.update(
                _dav3_gt_depth_metrics(
                    data["dav3_depth"],
                    data["depth"],
                    data.get("valid_depth"),
                )
            )
    camera_losses = result["camera_losses"]
    metrics["camera_loss"] = float(camera_losses["total"].item())
    metrics["camera_translation_loss"] = float(
        camera_losses["translation"].item()
    )
    metrics["camera_rotation_loss"] = float(camera_losses["rotation"].item())
    metrics["camera_focal_loss"] = float(camera_losses["focal"].item())
    predicted_focal = result["camera_prediction"]["intrinsic_px"][
        ..., (0, 1), (0, 1)
    ]
    gt_focal = data["intrinsic_px"][..., (0, 1), (0, 1)]
    metrics["camera_focal_abs_rel"] = float(
        (
            (predicted_focal - gt_focal).abs()
            / gt_focal.clamp_min(1e-4)
        ).mean().item()
    )
    raw_fov = result["camera_prediction"]["pose_encoding"][..., 7:9]
    metrics["camera_raw_fov_min"] = float(raw_fov.min().item())
    metrics["camera_raw_fov_max"] = float(raw_fov.max().item())
    metrics["detach_predicted_camera"] = bool(cfg.detach_predicted_camera)
    metrics["detach_predicted_depth"] = bool(cfg.detach_predicted_depth)
    metrics["historical_detach_mode"] = _historical_detach_mode(cfg)
    heldout_loss = data["heldout_loss"]
    heldout_loss_render = result["heldout_loss_render"]
    heldout_loss_psnrs = [
        _psnr(heldout_loss_render.color[:, view_idx], heldout_loss["rgb"][:, view_idx])
        for view_idx in range(len(heldout_loss["ids"]))
    ]
    metrics["heldout_loss_psnr"] = float(
        sum(heldout_loss_psnrs) / len(heldout_loss_psnrs)
    )
    heldout_test = data["heldout_test"]
    if heldout_test["ids"] == heldout_loss["ids"]:
        heldout_test_render = heldout_loss_render
    else:
        heldout_test_render = _render(
            model,
            updated,
            heldout_test["c2w"],
            heldout_test["intrinsic"],
            data["image_shape"],
        )
    heldout_test_psnrs = [
        _psnr(heldout_test_render.color[:, view_idx], heldout_test["rgb"][:, view_idx])
        for view_idx in range(len(heldout_test["ids"]))
    ]
    metrics["heldout_test_psnr"] = float(
        sum(heldout_test_psnrs) / len(heldout_test_psnrs)
    )
    metrics["heldout_loss_frames"] = heldout_loss["ids"]
    metrics["heldout_test_frames"] = heldout_test["ids"]
    for frame_id, frame_psnr in zip(heldout_loss["ids"], heldout_loss_psnrs):
        metrics[f"heldout_loss_frame_{frame_id:03d}_psnr"] = frame_psnr
    for frame_id, frame_psnr in zip(heldout_test["ids"], heldout_test_psnrs):
        metrics[f"heldout_test_frame_{frame_id:03d}_psnr"] = frame_psnr

    if step == 0 or step % int(cfg.save_every) == 0 or step == int(cfg.max_steps):
        image_dir = output_dir / "renders" / f"step_{step:06d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        first_frame_id = data["ids"][0]
        _save_image(
            image_dir / f"base_frame{first_frame_id:03d}.png",
            base_render.color[:, 0],
        )
        for view_idx, frame_id in enumerate(data["ids"]):
            _save_image(
                image_dir / f"global_frame{frame_id:03d}.png",
                global_render.color[:, view_idx],
            )
        for update_step in steps:
            frame_id = data["ids"][update_step["view_idx"]]
            _save_mask(
                image_dir / f"overlap_frame{frame_id:03d}.png",
                update_step["stats"]["overlap"],
            )
            _save_mask(
                image_dir / f"coverage_frame{frame_id:03d}.png",
                update_step["stats"]["old_coverage"],
            )
            _save_mask(
                image_dir / f"depth_consistent_frame{frame_id:03d}.png",
                update_step["stats"]["depth_consistent"],
            )
            _save_mask(
                image_dir / f"current_in_front_frame{frame_id:03d}.png",
                update_step["stats"]["current_in_front"],
            )
            _save_mask(
                image_dir / f"old_in_front_frame{frame_id:03d}.png",
                update_step["stats"]["old_in_front"],
            )
            _save_mask(
                image_dir / f"bad_frame{frame_id:03d}.png",
                update_step["stats"]["bad_mask"],
            )
            _save_mask(
                image_dir / f"append_frame{frame_id:03d}.png",
                update_step["stats"]["append_mask"],
            )
        for view_idx, frame_id in enumerate(heldout_loss["ids"]):
            _save_image(
                image_dir / f"heldout_loss_frame{frame_id:03d}.png",
                heldout_loss_render.color[:, view_idx],
            )
        for view_idx, frame_id in enumerate(heldout_test["ids"]):
            _save_image(
                image_dir / f"heldout_test_frame{frame_id:03d}.png",
                heldout_test_render.color[:, view_idx],
            )

    model.encoder.gaussian_param_head.train(
        bool(cfg.get("train_gaussian_param_head", True))
    )
    model.encoder.camera_head.train(bool(cfg.train_camera_head))
    model.encoder.gs_head.train(bool(cfg.get("train_gs_head", True)))
    model.gir_update_head.train(
        bool(cfg.residual_enabled)
        or bool(cfg.get("opacity_decay_enabled", False))
    )
    return metrics


def _save_checkpoint(model, optimizer, scheduler, scaler, step: int, output_dir: Path) -> None:
    state = {}
    modules = [
        ("encoder.gaussian_param_head", model.encoder.gaussian_param_head),
        ("encoder.gs_head", model.encoder.gs_head),
        ("gir_update_head", model.gir_update_head),
    ]
    if any(parameter.requires_grad for parameter in model.encoder.camera_head.parameters()):
        modules.append(("encoder.camera_head", model.encoder.camera_head))
    for prefix, module in modules:
        state.update({f"{prefix}.{key}": value.cpu() for key, value in module.state_dict().items()})
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "state_dict": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint_dir / f"step_{step:06d}.ckpt",
    )


def _record_evaluation(metrics: dict[str, Any], output_dir: Path) -> None:
    with (output_dir / "eval_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")
    eval_csv_path = output_dir / "eval_metrics.csv"
    eval_csv_exists = eval_csv_path.is_file() and eval_csv_path.stat().st_size > 0
    with eval_csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", *DAV3_GT_METRIC_FIELDS],
            extrasaction="ignore",
        )
        if not eval_csv_exists:
            writer.writeheader()
        writer.writerow(
            {
                "step": metrics.get("step", ""),
                **{
                    field: metrics.get(field, "")
                    for field in DAV3_GT_METRIC_FIELDS
                },
            }
        )
    try:
        curve_path = plot_metrics(output_dir)
        print(f"Updated metric curves: {curve_path}")
        residual_curve_path = plot_residual_update_metrics(output_dir)
        if residual_curve_path is not None:
            print(f"Updated residual update curves: {residual_curve_path}")
    except Exception as exc:
        print(f"[warning] Could not update metric curves: {type(exc).__name__}: {exc}")


def _find_nonfinite_gradient(model: torch.nn.Module) -> tuple[str | None, float]:
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        if not bool(finite.all()):
            bad_ratio = 1.0 - float(finite.float().mean().item())
            return name, bad_ratio
    return None, 0.0


def _record_skipped_step(
    output_dir: Path,
    step: int,
    reason: str,
    bad_parameter: str | None,
    amp_scale: float,
) -> None:
    event = {
        "step": step,
        "reason": reason,
        "bad_parameter": bad_parameter,
        "amp_scale": amp_scale,
    }
    with (output_dir / "skipped_steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _inherit_resume_logs(resume_state: dict[str, Any] | None, output_dir: Path) -> None:
    if resume_state is None or "_checkpoint_path" not in resume_state:
        return
    checkpoint_path = Path(resume_state["_checkpoint_path"])
    if checkpoint_path.parent.name != "checkpoints":
        return
    previous_output_dir = checkpoint_path.parent.parent
    if previous_output_dir.resolve() == output_dir.resolve():
        return
    resume_step = int(resume_state.get("step", 0))
    inherited = []
    metrics_source = previous_output_dir / "metrics.csv"
    metrics_destination = output_dir / "metrics.csv"
    if metrics_source.is_file() and not metrics_destination.exists():
        with metrics_source.open("r", encoding="utf-8", newline="") as source_handle:
            reader = csv.DictReader(source_handle)
            rows = [row for row in reader if int(float(row["step"])) <= resume_step]
            fieldnames = reader.fieldnames
        if fieldnames is not None:
            if "loss_updated_first" in fieldnames and "loss_replay_history" not in fieldnames:
                for row in rows:
                    row["loss_replay_history"] = row.pop("loss_updated_first", "")
            rows = [
                {field: row.get(field, "0") for field in TRAIN_CSV_FIELDS}
                for row in rows
            ]
            with metrics_destination.open("w", encoding="utf-8", newline="") as destination_handle:
                writer = csv.DictWriter(
                    destination_handle, fieldnames=TRAIN_CSV_FIELDS
                )
                writer.writeheader()
                writer.writerows(rows)
            inherited.append("metrics.csv")

    for filename in ("eval_metrics.jsonl", "skipped_steps.jsonl"):
        source = previous_output_dir / filename
        destination = output_dir / filename
        if not source.is_file() or destination.exists():
            continue
        kept_lines = []
        with source.open("r", encoding="utf-8") as source_handle:
            for line in source_handle:
                event = json.loads(line)
                if int(event["step"]) <= resume_step:
                    kept_lines.append(json.dumps(event, sort_keys=True))
        with destination.open("w", encoding="utf-8") as destination_handle:
            if kept_lines:
                destination_handle.write("\n".join(kept_lines) + "\n")
        inherited.append(filename)
    if inherited:
        print(
            f"Inherited resume logs from {previous_output_dir}: "
            + ", ".join(inherited)
        )


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The Replica old-GS overfit experiment requires CUDA/gsplat.")
    if "toy" not in cfg_dict:
        raise KeyError("Use +experiment=replica_old_gs_overfit so the toy config is present.")
    torch.manual_seed(int(cfg_dict.seed))
    np.random.seed(int(cfg_dict.seed))
    device = torch.device("cuda:0")
    toy_cfg = cfg_dict.toy
    output_dir = Path(HydraConfig.get()["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg_dict, output_dir / "resolved_config.yaml", resolve=True)

    residual_enabled = bool(toy_cfg.residual_enabled)
    residual_topk = max(1, int(toy_cfg.get("residual_topk", 1)))
    residual_independent_heads = bool(
        toy_cfg.get("residual_independent_heads", False)
    )
    residual_contributor_conditioned = bool(
        toy_cfg.get("residual_contributor_conditioned", False)
    )
    if residual_contributor_conditioned:
        raise ValueError(
            "This migrated overfit uses the main pipeline's independent "
            "top-k residual heads. Set toy.residual_contributor_conditioned=false."
        )
    opacity_decay_enabled = bool(
        toy_cfg.get("opacity_decay_enabled", False)
    )
    raw_scale_residual = bool(toy_cfg.get("raw_scale_residual", False))
    raw_opacity_residual = bool(toy_cfg.get("raw_opacity_residual", False))
    raw_rotation_residual = bool(toy_cfg.get("raw_rotation_residual", False))
    raw_harmonics_residual = bool(
        toy_cfg.get("raw_harmonics_residual", False)
    )
    mean_update_mode = str(
        toy_cfg.get("mean_update_mode", "relative_depth")
    ).lower()
    if mean_update_mode not in {"relative_depth", "absolute"}:
        raise ValueError(
            "toy.mean_update_mode must be 'relative_depth' or 'absolute', "
            f"got {mean_update_mode!r}."
        )
    root_cfg = load_typed_root_config(cfg_dict)
    model = get_model(root_cfg.model.encoder, root_cfg.model.decoder)
    if residual_enabled and residual_independent_heads:
        model.gir_update_head.configure_historical_prediction_heads(
            residual_topk
        )
    model = model.to(device)
    resume_state = _load_checkpoint(model, toy_cfg.checkpoint)
    train_gaussian_param_head = bool(
        toy_cfg.get("train_gaussian_param_head", True)
    )
    train_gs_head = bool(toy_cfg.get("train_gs_head", True))
    train_depth_head = bool(toy_cfg.train_depth_head)
    depth_loss_enabled = bool(toy_cfg.depth_loss_enabled)
    depth_loss_target = str(toy_cfg.get("depth_loss_target", "dav3")).lower()
    if depth_loss_target not in {"dav3", "gt"}:
        raise ValueError(
            "toy.depth_loss_target must be 'dav3' or 'gt', "
            f"got {depth_loss_target!r}."
        )
    if (
        depth_loss_enabled
        and depth_loss_target == "dav3"
        and float(toy_cfg.get("depth_absolute_log_scale_weight", 0.1)) > 0
        and not bool(toy_cfg.get("dav3_align_to_gt_camera_scale", True))
    ):
        raise ValueError(
            "toy.depth_absolute_log_scale_weight requires "
            "toy.dav3_align_to_gt_camera_scale=true."
        )
    train_camera_head = bool(toy_cfg.train_camera_head)
    heldout_loss_enabled = bool(toy_cfg.heldout_loss_enabled)
    print(f"Old-GS residual enabled: {residual_enabled}")
    print(
        "Old-GS residual contributors per pixel: "
        f"{residual_topk}"
    )
    print(f"Independent contributor residual heads: {residual_independent_heads}")
    print(
        "Shared contributor-conditioned residual head: "
        f"{residual_contributor_conditioned}"
    )
    print(
        "Contributor-conditioned opacity decay: "
        f"enabled={opacity_decay_enabled}, "
        f"topk={int(toy_cfg.get('opacity_decay_topk', 4))}, "
        f"strength={float(toy_cfg.get('opacity_decay_strength', 0.02)):g}, "
        f"target={float(toy_cfg.get('opacity_decay_target_ratio', 0.01)):g}, "
        f"budget_weight={float(toy_cfg.get('opacity_decay_budget_weight', 0.0)):g}, "
        f"warmup_steps={int(toy_cfg.get('opacity_decay_warmup_steps', 0))}"
    )
    print(f"Historical mean update mode: {mean_update_mode}")
    if mean_update_mode == "absolute":
        print("Absolute mean mode: toy.mean_relative_scale is ignored.")
    print(
        "Raw historical residuals: "
        f"mean={mean_update_mode == 'absolute'}, "
        f"rotation={raw_rotation_residual}, "
        f"scale={raw_scale_residual}, "
        f"opacity={raw_opacity_residual}, "
        f"harmonics={raw_harmonics_residual}"
    )
    print(
        "Enabled historical residual channels: "
        f"mean={bool(toy_cfg.get('residual_mean_enabled', True))}, "
        f"rotation={bool(toy_cfg.get('residual_rotation_enabled', True))}, "
        f"scale={bool(toy_cfg.get('residual_scale_enabled', True))}, "
        f"opacity={bool(toy_cfg.get('residual_opacity_enabled', True))}, "
        f"harmonics={bool(toy_cfg.get('residual_harmonics_enabled', True))}"
    )
    print(f"Train Gaussian feature/depth head: {train_gaussian_param_head}")
    print(f"Train base GS parameter head: {train_gs_head}")
    print(f"Train depth output head: {train_depth_head}")
    print(
        f"Depth loss enabled: {depth_loss_enabled} "
        f"(target={depth_loss_target if depth_loss_enabled else 'none'})"
    )
    if depth_loss_enabled and depth_loss_target == "dav3":
        print(
            "DAV3 supervision: per-scene inference, "
            f"GT-camera-scale alignment={bool(toy_cfg.get('dav3_align_to_gt_camera_scale', True))}, "
            "absolute-log-scale-weight="
            f"{float(toy_cfg.get('depth_absolute_log_scale_weight', 0.1)):g}"
        )
    print(f"Train camera head: {train_camera_head}")
    print(
        f"Input geometry sources: camera={str(toy_cfg.camera_source).lower()}, "
        f"depth={str(toy_cfg.depth_source).lower()}"
    )
    print(
        "RGB geometry gradients: "
        f"camera={'detached' if bool(toy_cfg.detach_predicted_camera) else 'enabled'}, "
        f"depth={'detached' if bool(toy_cfg.detach_predicted_depth) else 'enabled'}"
    )
    print(
        "Historical state detach mode before each new view: "
        f"{_historical_detach_mode(toy_cfg)}"
    )
    print(
        "Append mode: "
        f"{str(toy_cfg.get('append_mode', 'logic')).lower()} "
        "(logic=uncovered/front surface, all=every valid current pixel)"
    )
    if depth_loss_enabled and not train_depth_head:
        raise ValueError(
            "toy.depth_loss_enabled=true requires toy.train_depth_head=true."
        )
    if train_depth_head and not train_gaussian_param_head:
        raise ValueError(
            "toy.train_depth_head=true requires "
            "toy.train_gaussian_param_head=true."
        )
    group_defs = _configure_trainable_modules(
        model,
        residual_enabled,
        train_gaussian_param_head,
        train_gs_head,
        train_depth_head,
        train_camera_head,
        opacity_decay_enabled,
    )
    data = _load_toy_data(toy_cfg, device)
    if not bool(data.get("has_gt_depth", True)):
        if str(toy_cfg.depth_source).lower() == "gt":
            raise ValueError(
                "This dataset does not provide GT depth; use "
                "toy.depth_source=predicted."
            )
        if depth_loss_enabled and depth_loss_target == "gt":
            raise ValueError(
                "This dataset does not provide GT depth; use "
                "toy.depth_loss_target=dav3 or disable toy.depth_loss_enabled."
            )
        print(
            "GT depth: unavailable; depth metrics and DAV3-vs-GT metrics are skipped."
        )
    print(
        f"Held-out RGB loss enabled: {heldout_loss_enabled} "
        f"(loss frames {data['heldout_loss']['ids']}, "
        f"test frames {data['heldout_test']['ids']})"
    )
    supervised_test_frames = sorted(
        set(data["heldout_loss"]["ids"]) & set(data["heldout_test"]["ids"])
    )
    if heldout_loss_enabled and supervised_test_frames:
        print(
            "[warning] Some held-out test frames are also loss frames and are "
            f"therefore supervised: {supervised_test_frames}."
        )
    use_amp = bool(toy_cfg.mixed_precision)
    tokens, patch_start_idx = _cache_backbone_tokens(
        model, data["images"], use_amp
    )

    # These frozen modules are no longer needed after token extraction.
    model.encoder.aggregator.to("cpu")
    torch.cuda.empty_cache()
    cache_dav3 = (
        depth_loss_enabled and depth_loss_target == "dav3"
    ) or (
        bool(toy_cfg.get("dav3_gt_metrics_enabled", False))
        and bool(data.get("has_gt_depth", True))
    )
    if cache_dav3:
        data["dav3_depth"] = _cache_dav3_depth_targets(
            data["images"],
            data["image_shape"],
            data["c2w"],
            data["intrinsic_px"],
            str(toy_cfg.dav3_weights_path),
            bool(toy_cfg.get("dav3_align_to_gt_camera_scale", True)),
        )

    learning_rates = {
        "gaussian_param_head": float(toy_cfg.gaussian_head_lr),
        "gs_head": float(toy_cfg.gs_head_lr),
        "old_residual_head": float(toy_cfg.residual_head_lr),
        "camera_head": float(toy_cfg.camera_head_lr),
    }
    optimizer_groups = [
        {"params": group["params"], "lr": learning_rates[group["name"]], "name": group["name"]}
        for group in group_defs
    ]
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=float(toy_cfg.weight_decay), betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(toy_cfg.max_steps),
        eta_min=min(learning_rates[group["name"]] for group in group_defs) * 0.1,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_step = 0
    if resume_state is not None:
        try:
            optimizer.load_state_dict(resume_state["optimizer"])
            if "scheduler" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler"])
            if "scaler" in resume_state:
                scaler.load_state_dict(resume_state["scaler"])
            start_step = int(resume_state.get("step", 0))
            print(f"Resuming toy optimizer state from step {start_step}.")
        except (KeyError, ValueError) as exc:
            print(
                "Checkpoint optimizer groups are incompatible with the current "
                "camera/depth configuration; loaded model weights only and "
                f"started a fresh optimizer. Details: {exc}"
            )
            resume_state = None
    model.encoder.gaussian_param_head.train(train_gaussian_param_head)
    model.encoder.camera_head.train(train_camera_head)
    model.encoder.gs_head.train(train_gs_head)
    model.gir_update_head.train()

    _inherit_resume_logs(resume_state, output_dir)
    csv_path = output_dir / "metrics.csv"
    csv_exists = csv_path.is_file() and csv_path.stat().st_size > 0
    csv_handle = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_handle, fieldnames=TRAIN_CSV_FIELDS)
    if not csv_exists:
        writer.writeheader()

    initial_metrics = _evaluate(
        model, toy_cfg, data, tokens, patch_start_idx, use_amp, start_step, output_dir
    )
    _record_evaluation(initial_metrics, output_dir)
    print("[eval] " + json.dumps(initial_metrics, sort_keys=True))

    try:
        for step in range(start_step + 1, int(toy_cfg.max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            result = _forward_experiment(
                model,
                toy_cfg,
                data,
                tokens,
                patch_start_idx,
                use_amp,
                global_step=step,
                render_heldout_loss=heldout_loss_enabled,
            )
            images = data["images"]
            geometry_valid_depth = result["geometry_valid_depth"]
            valid_first = geometry_valid_depth[:, :1, None]
            loss_base_first = _masked_mse(
                result["base_render"].color, images[:, :1], valid_first
            )
            current_losses = []
            new_hole_losses = []
            replay_losses = []
            for update_step in result["steps"]:
                view_idx = update_step["view_idx"]
                current_valid = geometry_valid_depth[
                    :, view_idx : view_idx + 1, None
                ]
                current_losses.append(
                    _masked_mse(
                        update_step["current_render"].color,
                        images[:, view_idx : view_idx + 1],
                        current_valid,
                    )
                )
                append_mask = update_step["stats"]["append_mask"].unsqueeze(1)
                if not bool(toy_cfg.append_new_gs_enabled):
                    append_mask = torch.zeros_like(append_mask)
                new_hole_losses.append(
                    _masked_mse(
                        update_step["current_render"].color,
                        images[:, view_idx : view_idx + 1],
                        append_mask,
                    )
                )
                replay_indices = update_step["replay_indices"]
                replay_losses.append(
                    _masked_mse(
                        update_step["replay_render"].color,
                        images[:, replay_indices],
                        geometry_valid_depth[:, replay_indices, None],
                    )
                )
            loss_updated_current = torch.stack(current_losses).mean()
            loss_new_holes = torch.stack(new_hole_losses).mean()
            loss_replay_history = torch.stack(replay_losses).mean()
            if depth_loss_enabled:
                if depth_loss_target == "dav3":
                    (
                        loss_depth,
                        depth_alignment_scale,
                        loss_depth_structure,
                        loss_depth_absolute_scale,
                    ) = (
                        _sequence_scale_only_log_depth_loss(
                            result["predicted_depth"],
                            data["dav3_depth"],
                            absolute_log_scale_weight=float(
                                toy_cfg.get(
                                    "depth_absolute_log_scale_weight", 0.1
                                )
                            ),
                        )
                    )
                else:
                    loss_depth = _absolute_gt_depth_loss(
                        result["predicted_depth"],
                        data["depth"],
                        data["valid_depth"],
                    )
                    loss_maps = result["predicted_depth"].shape[:2]
                    depth_alignment_scale = loss_depth.new_ones(loss_maps)
                    loss_depth_structure = loss_depth
                    loss_depth_absolute_scale = loss_depth.new_zeros(())
            else:
                loss_depth = loss_base_first.new_zeros(())
                depth_alignment_scale = loss_depth.reshape(1)
                loss_depth_structure = loss_depth
                loss_depth_absolute_scale = loss_depth
            camera_losses = result["camera_losses"]
            if train_camera_head:
                loss_camera = camera_losses["total"]
            else:
                loss_camera = loss_base_first.new_zeros(())
            if heldout_loss_enabled:
                loss_heldout = F.mse_loss(
                    result["heldout_loss_render"].color,
                    data["heldout_loss"]["rgb"],
                )
            else:
                loss_heldout = loss_base_first.new_zeros(())
            summary = _summarize_step_stats(result["steps"])
            loss_regularization = summary["residual_regularization"]
            decay_warmup_steps = max(
                0, int(toy_cfg.get("opacity_decay_warmup_steps", 0))
            )
            decay_schedule = (
                1.0
                if decay_warmup_steps == 0
                else min(1.0, float(step) / decay_warmup_steps)
            )
            decay_target = float(
                toy_cfg.get("opacity_decay_target_ratio", 0.01)
            ) * decay_schedule
            decay_candidate_gate = (
                summary["opacity_decay_candidate_count"] > 0
            ).to(loss_base_first.dtype).detach()
            loss_opacity_decay_budget = (
                summary["opacity_decay_mass_ratio"] - decay_target
            ).abs() * decay_candidate_gate
            loss = (
                float(toy_cfg.loss_base_first) * loss_base_first
                + float(toy_cfg.loss_updated_current) * loss_updated_current
                + float(toy_cfg.loss_new_holes) * loss_new_holes
                + float(toy_cfg.loss_replay_history) * loss_replay_history
                + float(toy_cfg.loss_depth) * loss_depth
                + float(toy_cfg.loss_camera) * loss_camera
                + float(toy_cfg.loss_heldout) * loss_heldout
                + float(toy_cfg.loss_residual_regularization) * loss_regularization
                + float(toy_cfg.get("opacity_decay_budget_weight", 0.0))
                * loss_opacity_decay_budget
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            bad_parameter, bad_gradient_ratio = _find_nonfinite_gradient(model)
            if bad_parameter is not None:
                previous_scale = float(scaler.get_scale())
                if use_amp:
                    # unscale_ has already recorded the non-finite gradients;
                    # GradScaler therefore skips this optimizer update.
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                current_scale = float(scaler.get_scale())
                _record_skipped_step(
                    output_dir,
                    step,
                    "nonfinite_gradient",
                    bad_parameter,
                    current_scale,
                )
                print(
                    f"[skip step] step={step} bad_param={bad_parameter} "
                    f"bad_values={bad_gradient_ratio:.2%} "
                    f"amp_scale={previous_scale:g}->{current_scale:g}"
                )
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                float(toy_cfg.grad_clip),
            )
            if not torch.isfinite(grad_norm):
                previous_scale = float(scaler.get_scale())
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.update(new_scale=max(previous_scale * 0.5, 1.0))
                current_scale = float(scaler.get_scale())
                _record_skipped_step(
                    output_dir,
                    step,
                    "nonfinite_gradient_norm",
                    None,
                    current_scale,
                )
                print(
                    f"[skip step] step={step} nonfinite gradient norm "
                    f"amp_scale={previous_scale:g}->{current_scale:g}"
                )
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_row = {
                "step": step,
                "loss": float(loss.item()),
                "loss_base_first": float(loss_base_first.item()),
                "loss_updated_current": float(loss_updated_current.item()),
                "loss_new_holes": float(loss_new_holes.item()),
                "loss_replay_history": float(loss_replay_history.item()),
                "loss_depth": float(loss_depth.item()),
                "loss_depth_structure": float(loss_depth_structure.item()),
                "loss_depth_absolute_log_scale": float(
                    loss_depth_absolute_scale.item()
                ),
                "loss_camera": float(loss_camera.item()),
                "loss_camera_translation": float(
                    camera_losses["translation"].item()
                ),
                "loss_camera_rotation": float(camera_losses["rotation"].item()),
                "loss_camera_focal": float(camera_losses["focal"].item()),
                "loss_heldout": float(loss_heldout.item()),
                "loss_regularization": float(loss_regularization.item()),
                "loss_opacity_decay_budget": float(
                    loss_opacity_decay_budget.item()
                ),
                "grad_norm": float(grad_norm.item()),
                "overlap_ratio": float(summary["overlap_ratio"].item()),
                "old_coverage_ratio": float(
                    summary["old_coverage_ratio"].item()
                ),
                "depth_consistent_ratio": float(
                    summary["depth_consistent_ratio"].item()
                ),
                "current_in_front_ratio": float(
                    summary["current_in_front_ratio"].item()
                ),
                "old_in_front_ratio": float(
                    summary["old_in_front_ratio"].item()
                ),
                "bad_ratio": float(summary["bad_ratio"].item()),
                "append_ratio": float(summary["append_ratio"].item()),
                "appended_gs": int(summary["appended_gs"]),
                "unique_updated_gs": int(summary["unique_gs"]),
                "opacity_decay_candidate_probability": float(
                    summary["opacity_decay_candidate_probability"].item()
                ),
                "opacity_decay_candidate_count": float(
                    summary["opacity_decay_candidate_count"].item()
                ),
                "opacity_decay_mass_ratio": float(
                    summary["opacity_decay_mass_ratio"].item()
                ),
            }
            train_row.update(_flatten_residual_update_stats(summary))
            writer.writerow(train_row)
            csv_handle.flush()

            if step == 1 or step % 20 == 0:
                rank_effective_mean = ",".join(
                    f"{value:.2e}"
                    for value in summary["rank_effective_mean_shift"].tolist()
                )
                rank_multiplier = ",".join(
                    f"{value:.2e}"
                    for value in summary["rank_update_multiplier"].tolist()
                )
                print(
                    f"step={step:04d} loss={loss.item():.6f} "
                    f"base{data['ids'][0]}={loss_base_first.item():.6f} "
                    f"current_full={loss_updated_current.item():.6f} "
                    f"new_holes={loss_new_holes.item():.6f} "
                    f"replay_mean={loss_replay_history.item():.6f} "
                    f"depth={loss_depth.item():.6f} "
                    f"depth_s={depth_alignment_scale.median().item():.4f} "
                    f"camera={loss_camera.item():.6f} "
                    f"heldout={loss_heldout.item():.6f} "
                    f"decay_p={summary['opacity_decay_candidate_probability'].item():.3f} "
                    f"decay_mass={summary['opacity_decay_mass_ratio'].item():.3%} "
                    f"coverage={summary['old_coverage_ratio'].item():.2%} "
                    f"consistent={summary['depth_consistent_ratio'].item():.2%} "
                    f"new_front={summary['current_in_front_ratio'].item():.2%} "
                    f"old_front={summary['old_in_front_ratio'].item():.2%} "
                    f"bad={summary['bad_ratio'].item():.2%} "
                    f"updated_gs={int(summary['unique_gs'])} "
                    f"appended_gs={int(summary['appended_gs'])} "
                    f"rank_mean_eff=[{rank_effective_mean}] "
                    f"rank_mult=[{rank_multiplier}] "
                    f"map_mean={summary['map_mean_shift_mean'].item():.2e}"
                )
            if step % int(toy_cfg.eval_every) == 0 or step == int(toy_cfg.max_steps):
                metrics = _evaluate(
                    model,
                    toy_cfg,
                    data,
                    tokens,
                    patch_start_idx,
                    use_amp,
                    step,
                    output_dir,
                )
                _record_evaluation(metrics, output_dir)
                print("[eval] " + json.dumps(metrics, sort_keys=True))
            if bool(toy_cfg.save_checkpoints) and (
                step % int(toy_cfg.save_every) == 0
                or step == int(toy_cfg.max_steps)
            ):
                _save_checkpoint(model, optimizer, scheduler, scaler, step, output_dir)
    finally:
        csv_handle.close()

    print(f"Experiment complete. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
