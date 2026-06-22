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
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
import gzip
import pickle


@dataclass
class Datasetre10kCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    avg_pose: bool
    rescale_to_1cube: bool
    intr_augment: bool
    normalize_by_pts3d: bool
    rescale_to_1cube: bool
    mode: Optional[Literal["train", "test"]] = None
    ctx_list: list | None = None   
    tgt_list: list | None = None   

@dataclass
class Datasetre10kCfgWrapper:
    re10k: Datasetre10kCfg


class Datasetre10k(Dataset):
    cfg: Datasetre10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: Datasetre10kCfg,
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
        self.index_root = cfg.roots[1]
        self.data_list = []
        with open(self.index_root, "r") as file:
            self.data_index = json.load(file)
        
        for item in self.data_index:
            self.data_list.append(os.path.join(self.data_root, f"train/{item}"))
        self.scene_ids = {}
        self.scenes = {}
        index = 0
        
        with gzip.open(os.path.join(self.data_root, "train.pickle.gz"), "rb") as f:
            self.seq_data = pickle.load(f)
        
        print("re10k loading data !!!")
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [
                executor.submit(self.load_jsons, scene_path)
                for scene_path in self.data_list
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                scene_frames, scene_id = future.result()
                self.scenes[scene_id] = scene_frames
                self.scene_ids[index] = scene_id
                index += 1
        print(f"RE10k: {self.stage}: loaded {len(self.scene_ids)} scenes")

    def load_jsons(self, scene_path):
        files = os.listdir(scene_path)
        files = [f for f in files if os.path.isfile(os.path.join(scene_path, f))]
        files.sort()
        scene_frames = []
        scene_id = scene_path.split("/")[-1]
        for i, frame in enumerate(files):
            frame_tmp = {}

            frame_tmp["file_path"] = os.path.join(scene_path, frame)
            frame_tmp["extrinsics"] = np.array(
                [[0.0, -0.5, 0, 0], [0.5, 0.866, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
            )
            itmp = self.seq_data[scene_id]["intrinsics"][i]
            intrinsics = np.eye(3, dtype=np.float32)
            intrinsics[0, 0] = float(itmp[0])
            intrinsics[1, 1] = float(itmp[1])
            intrinsics[0, 2] = float(itmp[2])
            intrinsics[1, 2] = float(itmp[3])
            frame_tmp["intrinsics"] = intrinsics

            scene_frames.append(frame_tmp)

        return scene_frames, scene_id

    def load_frames(self, frames):
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Create a list to store futures with their original indices
            futures_with_idx = []
            for idx, file_path in enumerate(frames):
                file_path = file_path["file_path"]
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
        # Return as list if images have different sizes
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

        input_frames = [example[i] for i in context_indices]
        target_frame = [example[i] for i in target_indices]

        context_images = self.load_frames(input_frames)
        target_images = self.load_frames(target_frame)

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)),
                "far": self.get_bound("far", len(context_indices)),
                "index": context_indices,
                "overlap": [],
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)),
                "far": self.get_bound("far", len(target_indices)),
                "index": target_indices,
            },
            "scene": "re10k_" + scene,
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
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return len(self.scene_ids)
