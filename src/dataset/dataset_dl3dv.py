from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional
import os
import traceback
import numpy as np
import torch
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
        if not index_path.exists():
            raise FileNotFoundError(
                f"DL3DV {self.data_stage} index not found: {index_path}. "
                f"Expected train_index.json/test_index.json under dataset root."
            )

        with index_path.open("r", encoding="utf-8") as file:
            data_index = json.load(file)
        if not isinstance(data_index, list):
            raise TypeError(f"{index_path} must be a JSON array.")

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

        def filter_data_list(data_index, data_root):
            data_list = []

            for item in data_index:
                item = str(item)
                item_path = os.path.join(data_root, item)
                scene_path, images_8_path, transforms_path = resolve_scene_dir(data_root, item)
                if os.path.exists(item_path) and \
                   scene_path is not None and \
                   len(os.listdir(images_8_path)) > 0 and \
                   os.path.exists(transforms_path):
                    with open(transforms_path, 'r') as f:
                        transforms_data = json.load(f)
                    if 'frames' in transforms_data:
                        frames_length = len(transforms_data['frames'])
                        images_8_files_count = len(os.listdir(images_8_path))
                        if frames_length == images_8_files_count:
                            scene_id = item.strip("/\\").replace("/", "_").replace("\\", "_")
                            data_list.append((scene_path, scene_id))
            return data_list

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
                file_path = file_path["file_path"].replace("images", "images_8")
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
            context_indices = self.cfg.ctx_list
            target_indices = self.cfg.tgt_list
        
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise Exception("Field of view too wide")
        
        input_frames = [example[i] for i in context_indices]
        target_frame = [example[i] for i in target_indices]

        context_images = self.load_frames(input_frames)
        target_images = self.load_frames(target_frame)
        resize = tf.Resize((270, 480))
        context_images = resize(context_images)
        target_images = resize(target_images)

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

        if self.cfg.rescale_to_1cube:
            scene_scale = torch.max(
                torch.abs(extrinsics[context_indices][:, :3, 3])
            )  
            rescale_factor = 1 * scene_scale
            extrinsics[:, :3, 3] /= rescale_factor

        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            raise Exception("encounter nan or inf in input poses")

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
                # "overlap": overlap,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
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
        # patchsize_w = 37
        # patchsize_h = 18
        try:
            return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            index = np.random.randint(len(self))
            return self.__getitem__((index, num_context_views, patchsize_h))

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
