from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization
from ..evaluation.benchmark_io import discover_scenes, image_8_path
from .shims.load_shim import imread_cv2


def _validate_camera_parameters(
    extrinsics: Tensor,
    intrinsics: Tensor,
    *,
    label: str,
    frame_indices: Tensor | None = None,
) -> None:
    """Reject malformed cameras before they reach the GPU renderer."""
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(
            f"{label}: expected camera poses [V,4,4], got {tuple(extrinsics.shape)}."
        )
    if intrinsics.shape != (extrinsics.shape[0], 3, 3):
        raise ValueError(
            f"{label}: expected intrinsics {(extrinsics.shape[0], 3, 3)}, "
            f"got {tuple(intrinsics.shape)}."
        )

    poses = extrinsics.detach().float()
    cameras = intrinsics.detach().float()
    rotation = poses[:, :3, :3]
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    expected_bottom = poses.new_tensor([0.0, 0.0, 0.0, 1.0])

    pose_finite = torch.isfinite(poses).all(dim=(-2, -1))
    determinant = torch.linalg.det(rotation)
    orthogonality_error = (
        rotation.transpose(-1, -2) @ rotation - identity
    ).abs().amax(dim=(-2, -1))
    bottom_error = (poses[:, 3] - expected_bottom).abs().amax(dim=-1)
    pose_valid = (
        pose_finite
        & torch.isfinite(determinant)
        & torch.isfinite(orthogonality_error)
        & ((determinant - 1.0).abs() <= 5.0e-2)
        & (orthogonality_error <= 5.0e-2)
        & (bottom_error <= 1.0e-4)
    )

    intrinsic_finite = torch.isfinite(cameras).all(dim=(-2, -1))
    focal_valid = (cameras[:, 0, 0] > 1.0e-8) & (
        cameras[:, 1, 1] > 1.0e-8
    )
    intrinsic_bottom = cameras.new_tensor([0.0, 0.0, 1.0])
    intrinsic_bottom_error = (
        cameras[:, 2] - intrinsic_bottom
    ).abs().amax(dim=-1)
    intrinsic_valid = (
        intrinsic_finite
        & focal_valid
        & (intrinsic_bottom_error <= 1.0e-4)
    )

    valid = pose_valid & intrinsic_valid
    if bool(valid.all()):
        return

    bad_local_indices = torch.nonzero(~valid, as_tuple=False).flatten()
    if frame_indices is None:
        bad_frame_indices = bad_local_indices
    else:
        bad_frame_indices = frame_indices.detach().cpu()[bad_local_indices.cpu()]
    preview = []
    for local_idx, frame_idx in zip(
        bad_local_indices[:8].tolist(), bad_frame_indices[:8].tolist()
    ):
        preview.append(
            f"frame={frame_idx}, det={determinant[local_idx].item():.6g}, "
            f"orth_err={orthogonality_error[local_idx].item():.6g}, "
            f"bottom_err={bottom_error[local_idx].item():.6g}, "
            f"fx={cameras[local_idx, 0, 0].item():.6g}, "
            f"fy={cameras[local_idx, 1, 1].item():.6g}"
        )
    raise ValueError(
        f"{label}: invalid GT camera parameters at "
        + "; ".join(preview)
    )


@dataclass
class DatasetDl3dvCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool = False           
    relative_pose: bool = False     
    skip_bad_shape: bool = True     
    avg_pose: bool = False
    rescale_to_1cube: bool = False
    intr_augment: bool = False
    normalize_by_pts3d: bool = False
    mode: Optional[Literal["train", "test"]] = None
    ctx_list: list | None = None   
    tgt_list: list | None = None   
    test_scan_scenes: bool = False
    test_expected_scenes: int | None = None
    # Optional scene-local GT depth directory.  When unset, the dataset stays
    # RGB-only and DAV3-vs-GT diagnostics are simply not emitted.
    gt_depth_dir: str | None = None
    gt_depth_extension: str = "auto"
    gt_depth_scale: float = 1.0


@dataclass
class DatasetDL3DVCfgWrapper:
    dl3dv: DatasetDl3dvCfg


class DatasetDL3DV(Dataset):
    cfg: DatasetDl3dvCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    @staticmethod
    def resolve_scene_dir(data_root, item):
        item_path = os.path.join(data_root, item)
        scene_name = item.split("/")[-1].split(".")[0]
        candidates = [
            os.path.join(item_path, scene_name),
            os.path.join(item_path, "nerfstudio"),
            item_path,
        ]
        if os.path.isdir(item_path):
            for child in sorted(os.listdir(item_path)):
                candidates.append(os.path.join(item_path, child))

        seen = set()
        for scene_path in candidates:
            if scene_path in seen:
                continue
            seen.add(scene_path)
            images_8_path = os.path.join(scene_path, "images_8")
            transforms_path = os.path.join(scene_path, "transforms.json")
            if os.path.isdir(images_8_path) and os.path.exists(transforms_path):
                return scene_path, images_8_path, transforms_path
        return None, None, None

    def __init__(
        self,
        cfg: DatasetDl3dvCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # load data
        self.data_root = cfg.roots[0]
        self.data_list = []

        index_path = Path(self.data_root) / f"{self.data_stage}_index.json"
        scan_scenes = self.stage == "test" and cfg.test_scan_scenes
        if scan_scenes:
            self.data_list = discover_scenes(self.data_root, cfg.test_expected_scenes)
        elif not index_path.exists():
            raise FileNotFoundError(
                f"DL3DV {self.data_stage} index not found: {index_path}. "
                f"Expected train_index.json/test_index.json under dataset root."
            )

        if not scan_scenes:
            with index_path.open("r", encoding="utf-8") as file:
                data_index = json.load(file)
            if not isinstance(data_index, list):
                raise TypeError(f"{index_path} must be a JSON array.")

        def filter_data_list(data_index, data_root):
            data_list = []

            for item in data_index:
                item = str(item)
                item_path = os.path.join(data_root, item)
                scene_path, images_8_path, transforms_path = self.resolve_scene_dir(
                    data_root, item
                )
                error = None
                if not os.path.exists(item_path):
                    error = f"scene item does not exist: {item_path}"
                elif scene_path is None:
                    error = (
                        "could not find transforms.json and images_8 under "
                        f"scene item: {item_path}"
                    )
                else:
                    with open(transforms_path, "r", encoding="utf-8") as f:
                        transforms_data = json.load(f)
                    frames = transforms_data.get("frames")
                    if not isinstance(frames, list) or not frames:
                        error = f"no frames in {transforms_path}"
                    else:
                        missing_images = [
                            str(image_8_path(Path(scene_path) / frame["file_path"]))
                            for frame in frames
                            if not image_8_path(
                                Path(scene_path) / frame["file_path"]
                            ).is_file()
                        ]
                        if missing_images:
                            error = (
                                f"missing {len(missing_images)} referenced images_8 "
                                f"files; first missing file: {missing_images[0]}"
                            )

                if error is not None:
                    if self.stage == "test":
                        raise RuntimeError(
                            f"Invalid DL3DV test scene {item!r}: {error}"
                        )
                    continue

                scene_id = item.strip("/\\").replace("/", "_").replace("\\", "_")
                data_list.append((scene_path, scene_id))
            return data_list

        if not scan_scenes:
            self.data_list = filter_data_list(data_index, self.data_root)
        self.scene_ids = {}
        self.scenes = {}
        index = 0

        if cfg.mode == "train":
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = [
                    executor.submit(self.load_jsons, scene_info)
                    for scene_info in self.data_list
                ]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    scene_frames, scene_id = future.result()
                    self.scenes[scene_id] = scene_frames
                    self.scene_ids[index] = scene_id
                    index += 1
        else:
            futures = [self.load_jsons(scene_info) for scene_info in self.data_list]
            for future in tqdm(futures, total=len(futures)):
                scene_frames, scene_id = future
                self.scenes[scene_id] = scene_frames
                self.scene_ids[index] = scene_id
                index += 1

        unique_scene_count = len(set(self.scene_ids.values()))
        print(
            f"DL3DV: {self.stage}: loaded {len(self.scene_ids)} scenes "
            f"({unique_scene_count} unique scene ids)"
        )
        if (
            self.stage == "test"
            and cfg.test_expected_scenes is not None
            and unique_scene_count != int(cfg.test_expected_scenes)
        ):
            raise RuntimeError(
                "DL3DV test scene count mismatch: "
                f"expected={cfg.test_expected_scenes}, "
                f"loaded={len(self.scene_ids)}, unique={unique_scene_count}."
            )
        if unique_scene_count != len(self.scene_ids):
            print(
                f"WARNING: DL3DV {self.stage} has duplicated scene ids; "
                "check train_index.json/test_index.json."
            )

    def convert_intrinsics(self, meta_data):
        store_h, store_w = meta_data["h"], meta_data["w"]
        fx, fy, cx, cy = (
            meta_data["fl_x"],
            meta_data["fl_y"],
            meta_data["cx"],
            meta_data["cy"],
        )
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = float(fx) / float(store_w)
        intrinsics[1, 1] = float(fy) / float(store_h)
        intrinsics[0, 2] = float(cx) / float(store_w)
        intrinsics[1, 2] = float(cy) / float(store_h)
        return intrinsics

    def blender2opencv_c2w(self, pose):
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        opencv_c2w = np.array(pose) @ blender2opencv
        return opencv_c2w.tolist()

    def load_jsons(self, scene_info):
        if isinstance(scene_info, tuple):
            scene_path, scene_id = scene_info
        else:
            scene_path = scene_info
            scene_id = os.path.relpath(scene_path, self.data_root)
            scene_id = scene_id.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
        json_path = os.path.join(scene_path, "transforms.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        scene_frames = []
        for i, frame in enumerate(data["frames"]):
            frame_tmp = {}
            frame_tmp["file_path"] = os.path.join(scene_path, frame["file_path"])
            frame_tmp["intrinsics"] = self.convert_intrinsics(data).tolist()
            frame_tmp["extrinsics"] = self.blender2opencv_c2w(frame["transform_matrix"])
            scene_frames.append(frame_tmp)
        return scene_frames, scene_id

    def load_frames(self, frames):
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Create a list to store futures with their original indices
            futures_with_idx = []
            for idx, file_path in enumerate(frames):
                # file_path = file_path["file_path"]
                file_path = str(image_8_path(file_path["file_path"]))
                futures_with_idx.append(
                    (
                        idx,
                        executor.submit(
                            lambda p: self.to_tensor(Image.open(p).convert("RGB")),
                            file_path,
                        ),
                    )
                )

            # Pre-allocate list with correct size to maintain order
            torch_images = [None] * len(frames)
            for idx, future in futures_with_idx:
                torch_images[idx] = future.result()
            # Check if all images have the same size

            sizes = set(img.shape for img in torch_images)
            if len(sizes) == 1:
                torch_images = torch.stack(torch_images)
        return torch_images

    def _gt_depth_path(self, frame):
        if self.cfg.gt_depth_dir is None:
            return None
        image_path = image_8_path(frame["file_path"])
        depth_dir = Path(self.cfg.gt_depth_dir)
        if not depth_dir.is_absolute():
            depth_dir = image_path.parent.parent / depth_dir
        stem = image_path.stem
        extension = self.cfg.gt_depth_extension
        if extension == "auto":
            extensions = (".npy", ".exr", ".png", ".tiff", ".tif")
        else:
            extensions = (extension if extension.startswith(".") else "." + extension,)
        for suffix in extensions:
            candidate = depth_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        expected = ", ".join(str(depth_dir / f"{stem}{suffix}") for suffix in extensions)
        raise FileNotFoundError(
            "GT depth is enabled, but no depth file was found. Tried: " + expected
        )

    def load_depths(self, frames):
        if self.cfg.gt_depth_dir is None:
            return None

        depths = []
        for frame in frames:
            depth_path = self._gt_depth_path(frame)
            if depth_path.suffix.lower() == ".npy":
                depth = np.load(depth_path)
            else:
                depth = imread_cv2(str(depth_path))
            depth = np.asarray(depth)
            if depth.ndim == 3:
                if depth.shape[-1] == 1:
                    depth = depth[..., 0]
                else:
                    depth = depth[..., 0]
            if depth.ndim != 2:
                raise ValueError(
                    f"GT depth must be HxW, got {depth.shape} from {depth_path}"
                )
            depths.append(torch.from_numpy(depth.astype(np.float32, copy=False)))
        return torch.stack(depths) * float(self.cfg.gt_depth_scale)

    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:

        scene = self.scene_ids[index]

        example = self.scenes[scene]
        # load poses
        extrinsics = []
        intrinsics = []
        for frame in example:
            extrinsic = frame["extrinsics"]
            intrinsic = frame["intrinsics"]
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)

        extrinsics = np.array(extrinsics)
        intrinsics = np.array(intrinsics)
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

        if self.cfg.mode == "train":
            try:
                context_indices, target_indices, overlap = self.view_sampler.sample(
                    scene,
                    num_context_views,
                    extrinsics,
                    intrinsics,
                )
                context_indices = torch.sort(context_indices)[0]
                
                # 2. 提取偶数下标和奇数下标的元素
                # 比如原序为 [0, 1, 2, 3, 4]，偶数位置为 [0, 2, 4]，奇数位置为 [1, 3]
                even_indices = context_indices[0::2]
                odd_indices = context_indices[1::2]
                
                # 3. 拼接：将奇数位置的元素整体挪到最后面
                context_indices = torch.cat([even_indices, odd_indices], dim=0)
            except ValueError:
                # Skip because the example doesn't have enough frames.
                raise Exception("Not enough frames")
        else:
            context_indices = torch.as_tensor(self.cfg.ctx_list, dtype=torch.long)
            target_indices = torch.as_tensor(self.cfg.tgt_list, dtype=torch.long)

        selected_indices = torch.unique(
            torch.cat([context_indices, target_indices]), sorted=True
        )
        _validate_camera_parameters(
            extrinsics[selected_indices],
            intrinsics[selected_indices],
            label=f"DL3DV scene {scene} raw cameras",
            frame_indices=selected_indices,
        )
        
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise Exception("Field of view too wide")
        
        input_frames = [example[i] for i in context_indices]
        target_frame = [example[i] for i in target_indices]

        context_images = self.load_frames(input_frames)
        target_images = self.load_frames(target_frame)
        context_depth = self.load_depths(input_frames)
        target_depth = self.load_depths(target_frame)
        resize = tf.Resize((270, 480))
        context_images = resize(context_images)
        target_images = resize(target_images)
        if context_depth is not None:
            context_depth = F.interpolate(
                context_depth[:, None], size=(270, 480), mode="nearest"
            ).squeeze(1)
            target_depth = F.interpolate(
                target_depth[:, None], size=(270, 480), mode="nearest"
            ).squeeze(1)

        # Skip the example if the images don't have the right shape.
        context_image_invalid = context_images.shape[1:] != (
            3,
            *self.cfg.original_image_shape,
        )
        target_image_invalid = target_images.shape[1:] != (
            3,
            *self.cfg.original_image_shape,
        )
        if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
            print(
                f"Skipped bad example {scene}. Context shape was "
                f"{context_images.shape} and target shape was "
                f"{target_images.shape}."
            )

            raise Exception("Bad example image shape")

        context_extrinsics = extrinsics[context_indices]
        if self.cfg.make_baseline_1:
            a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                print(
                    f"Skipped {scene} because of baseline out of range: " f"{scale:.6f}"
                )
                raise Exception("baseline out of range")
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        if self.cfg.relative_pose:
            extrinsics = camera_normalization(
                extrinsics[context_indices][0:1], extrinsics
            )

        rescale_factor = 1
        if self.cfg.rescale_to_1cube:
            scene_scale = torch.max(
                torch.abs(extrinsics[context_indices][:, :3, 3])
            )  
            rescale_factor = 1 * scene_scale
            extrinsics[:, :3, 3] /= rescale_factor

        _validate_camera_parameters(
            extrinsics[selected_indices],
            intrinsics[selected_indices],
            label=f"DL3DV scene {scene} normalized cameras",
            frame_indices=selected_indices,
        )

        # Depth is expressed in the same camera/world unit as the camera
        # translations.  Apply the exact scene normalization used above.
        if context_depth is not None:
            context_depth = context_depth / scale / rescale_factor
            target_depth = target_depth / scale / rescale_factor

        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            raise Exception("encounter nan or inf in input poses")

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                **({"depth": context_depth} if context_depth is not None else {}),
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
                # "overlap": overlap,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                **({"depth": target_depth} if target_depth is not None else {}),
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
            },
            "scene": "dl3dv_" + scene,
        }
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        if self.stage == "train" and self.cfg.intr_augment:
            intr_aug = True
        else:
            intr_aug = False

        example = apply_crop_shim(
            example, (patchsize[0] * 14, patchsize[1] * 14), intr_aug=intr_aug
        )
        return example

    def __getitem__(self, index_tuple: tuple) -> dict:
        index, num_context_views, patchsize_h = index_tuple
        patchsize_w = self.cfg.input_image_shape[1] // 14
        max_attempts = min(max(len(self), 1), 64)
        attempted_scenes = []
        last_error = None
        current_index = int(index)

        for attempt in range(max_attempts):
            scene_id = self.scene_ids[current_index]
            try:
                return self.getitem(
                    current_index,
                    num_context_views,
                    (patchsize_h, patchsize_w),
                )
            except Exception as error:
                if self.stage == "test":
                    raise RuntimeError(
                        f"DL3DV test scene {scene_id} failed; test scenes must "
                        "not be replaced with random samples."
                    ) from error
                last_error = error
                attempted_scenes.append(str(scene_id))
                if attempt < 3:
                    print(
                        "[DL3DV retry] "
                        f"attempt={attempt + 1}/{max_attempts}, "
                        f"scene={scene_id}, "
                        f"error={type(error).__name__}: {error}"
                    )
                current_index = int(np.random.randint(len(self)))

        preview = ", ".join(attempted_scenes[:8])
        raise RuntimeError(
            f"DL3DV failed to produce a valid {self.stage} sample after "
            f"{max_attempts} attempts. Attempted scenes: {preview}."
        ) from last_error

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        if self.stage == "test" and self.cfg.test_scan_scenes:
            return {scene_id: Path(path) for path, scene_id in self.data_list}
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")

        for data_stage in data_stages:
            for root in self.cfg.roots:
                index_path = Path(root) / f"{data_stage}_index.json"
                if not index_path.exists():
                    continue
                with index_path.open("r", encoding="utf-8") as f:
                    index = json.load(f)
                if not isinstance(index, list):
                    raise TypeError(f"{index_path} must be a JSON array.")

                for item in index:
                    key = str(item)
                    path = Path(root) / key
                    assert key not in merged_index
                    merged_index[key] = path
        return merged_index

    def __len__(self) -> int:
        return len(self.scene_ids)
