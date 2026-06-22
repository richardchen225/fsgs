import copy
import numpy as np
import os
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import List, Literal, Optional
from einops import rearrange
import torch
import math
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from src.dataset.shims.normalize_shim import apply_normalize_shim
from src.dataset.types import BatchedExample, DataShim

from src.model.encoder.heads.vggt_dpt_gs_head import VGGT_DPT_GS_Head
from src.model.encoder.vggt.utils.geometry import (
    batchify_unproject_depth_map_to_point_map,
    closed_form_inverse_se3,
    batchify_sparse_unproject_depth_to_point
)
import matplotlib.pyplot as plt
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from torch import nn, Tensor
from torch_scatter import scatter_add, scatter_max
from ..types import Gaussians
from .backbone import BackboneCfg
from .common.gaussian_adapter import (
    GaussianAdapter,
    GaussianAdapterCfg,
    UnifiedGaussianAdapter,
)
from .encoder import Encoder, EncoderOutput
from .heads import head_factory
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
import cv2
from safetensors.torch import load_file

root_path = os.path.abspath(".")
sys.path.append(root_path)
from src.model.encoder.vggt.models.vggt import VGGT
from moge.model.v2 import MoGeModel
from src.model.encoder.zipmap.models.ZipMap_AR import ZipMap
# from src.model.encoder.lingbot_map.models.gct_stream import GCTStream

inf = float("inf")


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class GSHeadParams:
    dec_depth: int = 23
    patch_size: tuple[int, int] = (14, 14)
    enc_embed_dim: int = 2048
    dec_embed_dim: int = 2048
    feature_dim: int = 256
    depth_mode = ("exp", -inf, inf)
    conf_mode = True


@dataclass
class EncoderAnySplatCfg:
    name: Literal["anysplat"]
    mode: Optional[Literal["train", "test"]]
    n_offsets: int
    d_feature: int
    add_view: bool
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    gt_pose_to_pts: bool = False
    opacity_threshold: float = 0.001
    gs_keep_ratio: float = 1.0
    opacity_conf: bool = False
    # Model weights paths
    streamvggt_weights_path: Optional[str] = None
    zipmap_weights_path: Optional[str] = None
    zipmap_use_ema: bool = False
    zipmap_ttt_window_size: int = 1
    moge_weights_path: Optional[str] = None
    depth_refine_enabled: bool = True
    depth_refine_iters: int = 4
    depth_refine_candidates: int = 9
    depth_refine_hidden_dim: int = 128
    depth_refine_max_log_radius: float = 0.20
    depth_refine_min_log_radius: float = 0.025
    depth_refine_downsample_factor: int = 4
    depth_refine_geometry_neighbors: int = 2
    depth_refine_geometry_weight: float = 0.1
    num_test_context_views: Optional[int] = None
    gs_refine_enabled: bool = True
    gs_refine_iters: int = 4
    gs_refine_render_scale: float = 0.25
    gs_refine_hidden_dim: int = 64
    gs_refine_step_opacity: float = 0.25
    gs_refine_step_scale: float = 0.10
    gs_refine_step_sh: float = 0.05
    gs_refine_detach_evidence: bool = True
    gs_refine_max_render_views: Optional[int] = None

class CameraDec(nn.Module):
    def __init__(self, dim_in=2048):
        super().__init__()
        output_dim = dim_in
        self.backbone = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_fov = nn.Sequential(nn.Linear(output_dim, 1), nn.ReLU())

    def forward(self, feat):
        # feat shape: (B, N, C)
        B, N, C = feat.shape

        feat_single = feat[:, 0, :]  # (B, C)

        combined_feat = torch.cat([feat_single], dim=0)

        x = self.backbone(combined_feat)
        out = self.fc_fov(x.float())  # (B * 2, 2)

        out_fov_single = out[:B].reshape(B, 1, 1)

        return out_fov_single


class CausalDepthProbabilityRefiner(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_iters: int = 4,
        num_candidates: int = 9,
        max_log_radius: float = 0.20,
        min_log_radius: float = 0.025,
        downsample_factor: int = 4,
        geometry_neighbors: int = 2,
        geometry_weight: float = 0.1,
        depth_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_iters = num_iters
        self.num_candidates = num_candidates
        self.downsample_factor = max(1, int(downsample_factor))
        self.geometry_neighbors = max(0, int(geometry_neighbors))
        self.depth_eps = depth_eps

        self.feature_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.context_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        norm_groups = min(8, hidden_dim)
        while hidden_dim % norm_groups != 0:
            norm_groups -= 1
        self.depth_proj = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.logit_head = nn.Conv2d(hidden_dim, num_candidates, kernel_size=1)
        self.iter_embed = nn.Parameter(torch.zeros(max(1, num_iters), hidden_dim, 1, 1))
        self.geometry_bias_weight = nn.Parameter(torch.tensor(float(geometry_weight)))

        candidate_offsets = torch.linspace(-1.0, 1.0, num_candidates)
        if num_iters > 1:
            log_depth_radii = torch.linspace(max_log_radius, min_log_radius, num_iters)
        else:
            log_depth_radii = torch.tensor([min_log_radius])
        self.register_buffer("candidate_offsets", candidate_offsets, persistent=False)
        self.register_buffer("log_depth_radii", log_depth_radii, persistent=False)

        self._init_as_identity()

    def _init_as_identity(self) -> None:
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)

    @staticmethod
    def _ensure_batched_c2w(
        extrinsics_c2w: torch.Tensor,
        b: int,
        s: int,
    ) -> torch.Tensor:
        if extrinsics_c2w.dim() == 3:
            extrinsics_c2w = extrinsics_c2w.view(b, s, *extrinsics_c2w.shape[-2:])
        elif extrinsics_c2w.dim() != 4:
            raise ValueError(f"Expected extrinsics with 3 or 4 dims, got {extrinsics_c2w.shape}.")

        if extrinsics_c2w.shape[-2:] == (3, 4):
            pad = torch.tensor(
                [0, 0, 0, 1],
                device=extrinsics_c2w.device,
                dtype=extrinsics_c2w.dtype,
            ).view(1, 1, 1, 4)
            pad = pad.expand(*extrinsics_c2w.shape[:-2], 1, 4)
            extrinsics_c2w = torch.cat([extrinsics_c2w, pad], dim=-2)

        return extrinsics_c2w

    @staticmethod
    def _ensure_batched_intrinsics(
        intrinsics: torch.Tensor,
        b: int,
        s: int,
    ) -> torch.Tensor:
        if intrinsics.dim() == 3:
            intrinsics = intrinsics.view(b, s, 3, 3)
        elif intrinsics.dim() != 4:
            raise ValueError(f"Expected intrinsics with 3 or 4 dims, got {intrinsics.shape}.")
        return intrinsics

    def _prepare_geometry_inputs(
        self,
        extrinsics_c2w: Optional[torch.Tensor],
        intrinsics: Optional[torch.Tensor],
        b: int,
        s: int,
        h: int,
        w: int,
        low_h: int,
        low_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if (
            self.geometry_neighbors <= 0
            or extrinsics_c2w is None
            or intrinsics is None
            or s <= 1
        ):
            return None

        c2w = self._ensure_batched_c2w(extrinsics_c2w, b, s).detach().float()
        intrinsics_low = self._ensure_batched_intrinsics(intrinsics, b, s).detach().float().clone()

        intrinsics_low[..., 0, :] *= float(low_w) / float(w)
        intrinsics_low[..., 1, :] *= float(low_h) / float(h)
        w2c = torch.linalg.inv(c2w)

        return c2w, w2c, intrinsics_low

    def _causal_geometry_bias(
        self,
        features_low: torch.Tensor,
        candidate_depth: torch.Tensor,
        c2w: torch.Tensor,
        w2c: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        b, s, c, h, w = features_low.shape
        _, _, m, _, _ = candidate_depth.shape
        device = features_low.device

        features_norm = F.normalize(features_low.float(), dim=2, eps=1e-6)
        candidate_depth = candidate_depth.float().clamp_min(self.depth_eps)

        ys = torch.arange(h, device=device, dtype=torch.float32).view(1, 1, h, 1)
        xs = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, 1, w)

        bias_sum = features_norm.new_zeros((b, s, m, h, w))
        bias_count = features_norm.new_zeros((b, s, m, h, w))

        for view_idx in range(s):
            first_neighbor = max(0, view_idx - self.geometry_neighbors)
            if first_neighbor == view_idx:
                continue

            cur_feat = features_norm[:, view_idx].unsqueeze(1)
            cur_depth = candidate_depth[:, view_idx]
            cur_k = intrinsics[:, view_idx]
            cur_fx = cur_k[:, 0, 0].view(b, 1, 1, 1).clamp_min(1e-6)
            cur_fy = cur_k[:, 1, 1].view(b, 1, 1, 1).clamp_min(1e-6)
            cur_cx = cur_k[:, 0, 2].view(b, 1, 1, 1)
            cur_cy = cur_k[:, 1, 2].view(b, 1, 1, 1)

            x_cam = (xs - cur_cx) * cur_depth / cur_fx
            y_cam = (ys - cur_cy) * cur_depth / cur_fy
            ones = torch.ones_like(cur_depth)
            cur_points = torch.stack([x_cam, y_cam, cur_depth, ones], dim=-1)
            world_points = torch.einsum("bij,bmhwj->bmhwi", c2w[:, view_idx], cur_points)[..., :3]
            world_points_h = torch.cat([world_points, ones.unsqueeze(-1)], dim=-1)

            for neighbor_idx in range(first_neighbor, view_idx):
                prev_points = torch.einsum("bij,bmhwj->bmhwi", w2c[:, neighbor_idx], world_points_h)[..., :3]
                z_prev = prev_points[..., 2]
                valid_z = z_prev > self.depth_eps
                z_safe = z_prev.clamp_min(self.depth_eps)

                prev_k = intrinsics[:, neighbor_idx]
                prev_fx = prev_k[:, 0, 0].view(b, 1, 1, 1)
                prev_fy = prev_k[:, 1, 1].view(b, 1, 1, 1)
                prev_cx = prev_k[:, 0, 2].view(b, 1, 1, 1)
                prev_cy = prev_k[:, 1, 2].view(b, 1, 1, 1)

                x_pix = prev_fx * (prev_points[..., 0] / z_safe) + prev_cx
                y_pix = prev_fy * (prev_points[..., 1] / z_safe) + prev_cy
                valid_xy = (
                    (x_pix >= 0)
                    & (x_pix <= w - 1)
                    & (y_pix >= 0)
                    & (y_pix <= h - 1)
                )
                valid = (valid_z & valid_xy).float()

                grid_x = 2.0 * (x_pix + 0.5) / float(w) - 1.0
                grid_y = 2.0 * (y_pix + 0.5) / float(h) - 1.0
                grid = torch.stack([grid_x, grid_y], dim=-1).view(b * m, h, w, 2)

                prev_feat = features_norm[:, neighbor_idx].repeat_interleave(m, dim=0)
                sampled_feat = F.grid_sample(
                    prev_feat,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampled_feat = sampled_feat.view(b, m, c, h, w)
                similarity = (sampled_feat * cur_feat).sum(dim=2)

                bias_sum[:, view_idx] = bias_sum[:, view_idx] + similarity * valid
                bias_count[:, view_idx] = bias_count[:, view_idx] + valid

        return bias_sum / bias_count.clamp_min(1.0)

    def forward(
        self,
        features: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: Optional[torch.Tensor] = None,
        extrinsics_c2w: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        if self.num_iters <= 0:
            return depth, [], None

        b, s, _, h, w = features.shape
        depth_base_full = depth.float().clamp_min(self.depth_eps)
        if self.downsample_factor > 1:
            low_h = max(1, h // self.downsample_factor)
            low_w = max(1, w // self.downsample_factor)

            features_low = rearrange(features.float(), "b s c h w -> (b s) c h w")
            features_low = F.interpolate(
                features_low,
                size=(low_h, low_w),
                mode="bilinear",
                align_corners=False,
            )
            features_low = rearrange(features_low, "(b s) c h w -> b s c h w", b=b, s=s)

            depth_base_low = rearrange(depth_base_full, "b s h w c -> (b s) c h w")
            depth_base_low = F.interpolate(
                depth_base_low,
                size=(low_h, low_w),
                mode="bilinear",
                align_corners=False,
            )
            depth_base_low = rearrange(depth_base_low, "(b s) c h w -> b s h w c", b=b, s=s)

            if depth_conf is not None:
                depth_conf_low = rearrange(depth_conf.float(), "b s h w c -> (b s) c h w")
                depth_conf_low = F.interpolate(
                    depth_conf_low,
                    size=(low_h, low_w),
                    mode="bilinear",
                    align_corners=False,
                )
                depth_conf_low = rearrange(depth_conf_low, "(b s) c h w -> b s h w c", b=b, s=s)
            else:
                depth_conf_low = None
        else:
            low_h, low_w = h, w
            features_low = features.float()
            depth_base_low = depth_base_full
            depth_conf_low = depth_conf.float() if depth_conf is not None else None

        depth_cur = depth_base_low.clamp_min(self.depth_eps)
        log_depth_base_low = torch.log(depth_base_low.clamp_min(self.depth_eps))

        causal_context = torch.cumsum(features_low, dim=1)
        counts = torch.arange(1, s + 1, device=features.device, dtype=features_low.dtype)
        causal_context = causal_context / counts.view(1, s, 1, 1, 1)

        feat_flat = rearrange(features_low, "b s c h w -> (b s) c h w")
        context_flat = rearrange(causal_context, "b s c h w -> (b s) c h w")
        feat_enc = self.feature_proj(feat_flat)
        context_enc = self.context_proj(context_flat)

        if depth_conf_low is None:
            log_conf = torch.zeros_like(depth_cur)
        else:
            log_conf = torch.log(depth_conf_low.clamp_min(1e-6))

        geometry_inputs = self._prepare_geometry_inputs(
            extrinsics_c2w,
            intrinsics,
            b,
            s,
            h,
            w,
            low_h,
            low_w,
        )

        refined_depths = []
        uncertainty = torch.zeros_like(depth_cur)
        offsets = self.candidate_offsets.to(device=features.device, dtype=features.dtype)
        log_num_candidates = math.log(float(self.num_candidates))

        def lift_low_depth_to_full(depth_low: torch.Tensor) -> torch.Tensor:
            if low_h == h and low_w == w:
                return depth_low
            delta_log_depth = torch.log(depth_low.clamp_min(self.depth_eps)) - log_depth_base_low
            delta_log_depth = rearrange(delta_log_depth, "b s h w c -> (b s) c h w")
            delta_log_depth = F.interpolate(
                delta_log_depth,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            delta_log_depth = rearrange(delta_log_depth, "(b s) c h w -> b s h w c", b=b, s=s)
            return (depth_base_full * delta_log_depth.exp()).clamp_min(self.depth_eps)

        for iter_idx in range(self.num_iters):
            log_depth = torch.log(depth_cur.clamp_min(self.depth_eps))
            depth_inputs = torch.cat([log_depth, log_conf, uncertainty], dim=-1)
            depth_inputs = rearrange(depth_inputs, "b s h w c -> (b s) c h w")
            depth_enc = self.depth_proj(depth_inputs)

            fused = self.fuse(torch.cat([feat_enc, context_enc, depth_enc], dim=1))
            fused = fused + self.iter_embed[iter_idx].to(dtype=fused.dtype)
            logits = self.logit_head(fused)
            logits = rearrange(logits, "(b s) m h w -> b s m h w", b=b, s=s)

            radius = self.log_depth_radii[iter_idx].to(device=features.device, dtype=features.dtype)
            candidate_log_depth = rearrange(log_depth, "b s h w c -> b s c h w")
            candidate_log_depth = candidate_log_depth + radius * offsets.view(1, 1, -1, 1, 1)

            if geometry_inputs is not None:
                with torch.no_grad():
                    geometry_bias = self._causal_geometry_bias(
                        features_low.detach(),
                        candidate_log_depth.detach().exp(),
                        *geometry_inputs,
                    )
                logits = logits + self.geometry_bias_weight.to(dtype=logits.dtype) * geometry_bias.to(dtype=logits.dtype)
            else:
                logits = logits + self.geometry_bias_weight.to(dtype=logits.dtype) * logits.new_zeros(())
            prob = torch.softmax(logits, dim=2)
            refined_log_depth = (prob * candidate_log_depth).sum(dim=2)
            depth_cur = refined_log_depth.exp().unsqueeze(-1).clamp_min(self.depth_eps)

            log_prob = torch.log(prob.clamp_min(1e-8))
            entropy = -(prob * log_prob).sum(dim=2) / log_num_candidates
            uncertainty = entropy.unsqueeze(-1)
            if return_intermediate:
                refined_depths.append(lift_low_depth_to_full(depth_cur))

        depth_full = lift_low_depth_to_full(depth_cur)
        if low_h != h or low_w != w:
            uncertainty_full = rearrange(uncertainty, "b s h w c -> (b s) c h w")
            uncertainty_full = F.interpolate(
                uncertainty_full,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            uncertainty_full = rearrange(uncertainty_full, "(b s) c h w -> b s h w c", b=b, s=s)
        else:
            uncertainty_full = uncertainty

        return depth_full, refined_depths, uncertainty_full


class ZipMapStreamingAggregatorAdapter(nn.Module):
    def __init__(self, aggregator: nn.Module, window_size: int = 1):
        super().__init__()
        self.aggregator = aggregator
        self.window_size = window_size
        self.depth = getattr(aggregator, "depth", 0)
        self._state_list = None

    def forward(
        self,
        images: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
        past_frame_idx: int = 0,
    ):
        del past_frame_idx
        info = {"store_state": bool(use_cache)}
        if self.window_size is not None:
            info["window_size"] = self.window_size

        cached_state = None
        if past_key_values is not None:
            if len(past_key_values) > 0 and past_key_values[0] is not None:
                cached_state = past_key_values
        elif self._state_list is not None and len(self._state_list) > 0 and self._state_list[0] is not None:
            cached_state = self._state_list
        if use_cache and cached_state is not None:
            info["state_list"] = cached_state

        aggregated_tokens_list, patch_start_idx, state_list = self.aggregator(
            images,
            target_query_conditions=None,
            info=info,
        )
        if use_cache:
            self._state_list = state_list
            return aggregated_tokens_list, patch_start_idx, state_list
        return aggregated_tokens_list, patch_start_idx


class ZipMapCameraHeadAdapter(nn.Module):
    def __init__(self, camera_head: Optional[nn.Module], camera_mlp_head: Optional[nn.Module]):
        super().__init__()
        self.camera_head = camera_head
        self.camera_mlp_head = camera_mlp_head
        self.trunk_depth = getattr(camera_head, "trunk_depth", 0)

    def forward(
        self,
        aggregated_tokens_list: list,
        past_key_values_camera=None,
        use_cache: bool = False,
    ):
        if self.camera_head is not None:
            pred_pose_enc_list = self.camera_head(aggregated_tokens_list)
        else:
            camera_tokens = aggregated_tokens_list[-1][:, :, 0]
            pred_pose_enc_list = [self.camera_mlp_head(camera_tokens)]
        if use_cache:
            return pred_pose_enc_list, past_key_values_camera
        return pred_pose_enc_list


def _bytes_to_mb(x):
    return x / 1024**2


def _bytes_to_gb(x):
    return x / 1024**3


def tensor_tree_nbytes(obj, only_cuda=True, seen=None):
    """
    递归统计一个嵌套结构里所有 tensor 的真实字节数。
    支持 tensor / list / tuple / dict。
    """
    if seen is None:
        seen = set()

    if obj is None:
        return 0

    if torch.is_tensor(obj):
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        if only_cuda and not obj.is_cuda:
            return 0

        return obj.numel() * obj.element_size()

    if isinstance(obj, dict):
        return sum(tensor_tree_nbytes(v, only_cuda=only_cuda, seen=seen) for v in obj.values())

    if isinstance(obj, (list, tuple)):
        return sum(tensor_tree_nbytes(v, only_cuda=only_cuda, seen=seen) for v in obj)

    return 0


def current_cuda_allocated():
    if not torch.cuda.is_available():
        return 0
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device)


def get_rank_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def log_cache_bank_mem(
    tag,
    past_key_values=None,
    past_key_values_camera=None,
    online_gs_bank=None,
):
    rank, world_size = get_rank_info()
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    total_alloc = current_cuda_allocated()

    agg_cache_bytes = tensor_tree_nbytes(past_key_values, only_cuda=True)
    cam_cache_bytes = tensor_tree_nbytes(past_key_values_camera, only_cuda=True)
    bank_bytes = tensor_tree_nbytes(online_gs_bank, only_cuda=True)

    print(
        f"[rank={rank}/{world_size} cuda:{device}] {tag} | "
        f"allocated={_bytes_to_gb(total_alloc):.3f} GB | "
        f"agg_cache={_bytes_to_mb(agg_cache_bytes):.2f} MB | "
        f"cam_cache={_bytes_to_mb(cam_cache_bytes):.2f} MB | "
        f"online_gs_bank={_bytes_to_mb(bank_bytes):.2f} MB",
        flush=True,
    )
    
class EncoderAnySplat(Encoder[EncoderAnySplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderAnySplatCfg) -> None:
        super().__init__(cfg)

        zipmap_config = {
            "img_size": 518,
            "patch_size": 14,
            "embed_dim": 1024,
            "enable_camera": False,
            "enable_camera_mlp": True,
            "enable_local_point": False,
            "enable_depth": False,
            "ttt_config": {
                "ttt_mode": True,
                "params": {
                    "bias": True,
                    "head_dim": 1024,
                    "inter_multi": 2,
                    "base_lr": 0.01,
                    "muon_update_steps": 5,
                    "use_gate_fn": True,
                },
                "window_size": cfg.zipmap_ttt_window_size,
            },
            "other_config": {
                "affine_invariant": True,
            },
        }

        print("Initializing and loading ZipMap streaming model...")
        model_full = ZipMap(**zipmap_config)
        if cfg.zipmap_weights_path is None and cfg.streamvggt_weights_path is not None:
            zipmap_ckpt_path = os.path.join(
                os.path.dirname(cfg.streamvggt_weights_path),
                "pre_zipmap",
            )
        elif cfg.zipmap_weights_path is not None:
            zipmap_ckpt_path = cfg.zipmap_weights_path
        else:
            zipmap_ckpt_path = None
        zipmap_ckpt_candidates = [zipmap_ckpt_path] if zipmap_ckpt_path else []
        if zipmap_ckpt_path and os.path.splitext(zipmap_ckpt_path)[1] == "":
            zipmap_ckpt_candidates = [
                f"{zipmap_ckpt_path}.pt",
                f"{zipmap_ckpt_path}.pth",
                f"{zipmap_ckpt_path}.safetensors",
                zipmap_ckpt_path,
            ]

        ckpt = None
        loaded_zipmap_ckpt_path = None
        zipmap_load_errors = []
        for candidate_path in zipmap_ckpt_candidates:
            if not candidate_path or not os.path.exists(candidate_path):
                continue
            try:
                if candidate_path.endswith(".safetensors"):
                    ckpt = load_file(candidate_path)
                else:
                    ckpt = torch.load(candidate_path, map_location="cpu", weights_only=True)
                loaded_zipmap_ckpt_path = candidate_path
                break
            except Exception as exc:
                zipmap_load_errors.append(f"{candidate_path}: {type(exc).__name__}: {exc}")

        if ckpt is not None:
            if isinstance(ckpt, dict) and cfg.zipmap_use_ema and "ema" in ckpt:
                ckpt = ckpt["ema"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                ckpt = ckpt["model"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            ckpt = {
                key.replace("module.", "", 1).replace("model.", "", 1): value
                for key, value in ckpt.items()
            }
            missing, unexpected = model_full.load_state_dict(ckpt, strict=False)
            print(
                f"Loaded ZipMap weights from {loaded_zipmap_ckpt_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            print(
                "ZipMap weights not found or failed to load. Tried: "
                f"{zipmap_ckpt_candidates}. Errors: {zipmap_load_errors}. "
                "Using initialized ZipMap weights."
            )

        self.aggregator = ZipMapStreamingAggregatorAdapter(
            model_full.aggregator,
            window_size=cfg.zipmap_ttt_window_size,
        )
        self.camera_head = ZipMapCameraHeadAdapter(
            model_full.camera_head,
            model_full.camera_mlp_head,
        )

        if cfg.moge_weights_path is None:
            raise ValueError("model.encoder.moge_weights_path must be set before using EncoderAnySplat.")
        print("Loading MoGe-2 model for intrinsics...")
        self.moge_model = MoGeModel.from_pretrained(cfg.moge_weights_path).to("cuda").eval()
        for param in self.moge_model.parameters():
            param.requires_grad = False

        for module in [self.aggregator, self.camera_head]:
            for param in module.parameters():
                param.requires_grad = False
        del model_full

        head_params = GSHeadParams()
        self.gaussian_param_head = VGGT_DPT_GS_Head(
            dim_in=2048,
            patch_size=head_params.patch_size,
            output_dim=2,
            activation="norm_exp",
            conf_activation="expp1",
            features=head_params.feature_dim,
        )
        
        self.feature_dim = 256
        self.p = 64
        
        self.sh_degree = 0
 
        self.nums_sh = (self.sh_degree + 1) ** 2
        # gaussian_raw_channels = 9 + self.p * 2 + self.nums_sh * 3
        gaussian_raw_channels = 9 + self.nums_sh * 3
        self.gs_head = nn.Sequential(
            nn.Conv2d(
                self.feature_dim // 2,
                self.feature_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(self.feature_dim, gaussian_raw_channels, kernel_size=1),
        )

        self.depth_refiner = None
        if cfg.depth_refine_enabled and cfg.depth_refine_iters > 0:
            self.depth_refiner = CausalDepthProbabilityRefiner(
                feature_dim=self.feature_dim // 2,
                hidden_dim=cfg.depth_refine_hidden_dim,
                num_iters=cfg.depth_refine_iters,
                num_candidates=cfg.depth_refine_candidates,
                max_log_radius=cfg.depth_refine_max_log_radius,
                min_log_radius=cfg.depth_refine_min_log_radius,
                downsample_factor=cfg.depth_refine_downsample_factor,
                geometry_neighbors=cfg.depth_refine_geometry_neighbors,
                geometry_weight=cfg.depth_refine_geometry_weight,
            )
    def forward(
        self,
        image: torch.Tensor,
        ctx_index: list = None,
        global_step: int = 0,
        name: str = None,
        target_view_count: Optional[int] = None,
    ) -> Gaussians:

        device = image.device
        b, v, _, h, w = image.shape
        if self.cfg.num_test_context_views is not None and self.cfg.mode != "train":
            ctx_img_num = min(int(self.cfg.num_test_context_views), v)
        elif self.cfg.mode != "train" and target_view_count is not None:
            ctx_img_num = v - int(target_view_count)
            if ctx_img_num <= 0:
                ctx_img_num = v
        else:
            if v == 1:
                ctx_img_num = 1
            else:
                ctx_img_num = int(v * 0.5)

        distill_infos = {}
        pred_all_extrinsic = None
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                aggregated_tokens_list, patch_start_idx = self.aggregator(
                    image.to(torch.float16)
                )

        with torch.amp.autocast("cuda", enabled=False):
            pred_pose_enc_list = self.camera_head(aggregated_tokens_list)
            last_pred_pose_enc = pred_pose_enc_list[-1]
            distill_infos["pred_pose_enc_list"] = last_pred_pose_enc
            
            # moge 
            moge_out = self.moge_model.infer(
                image[:,0],
                fov_x=None,                         # None = 让 MoGe 自己估计 FOV / focal
                use_fp16=(device.type == "cuda"),    # CUDA 上快一些
            )

            K_norm = moge_out["intrinsics"].float()
            if K_norm.dim() == 2:
                K_norm = K_norm.unsqueeze(0)
           
            
            pred_all_extrinsic, _ = pose_encoding_to_extri_intri(
                last_pred_pose_enc, image.shape[-2:]
            )

            gt_ex = closed_form_inverse_se3(
                pred_all_extrinsic[:, :ctx_img_num, ...].flatten(0, 1)
            )

            K_px = K_norm.to(device=device, dtype=last_pred_pose_enc.dtype).clone()

            K_px[:, 0, 0] = K_px[:, 0, 0] * w   # fx
            K_px[:, 1, 1] = K_px[:, 1, 1] * h   # fy
            K_px[:, 0, 2] = K_px[:, 0, 2] * w   # cx
            K_px[:, 1, 2] = K_px[:, 1, 2] * h   # cy
            # 把 fy 改成 fx
            K_px[:, 1, 1] = K_px[:, 0, 0]

            # 复制成 [b, ctx_img_num, 3, 3]
            K = K_px[:, None, :, :].expand(b, ctx_img_num, 3, 3).clone()
            intrinsic = K
            gt_ix = intrinsic

            extrinsic_padding = (
                torch.tensor(
                    [0, 0, 0, 1],
                    device=pred_all_extrinsic.device,
                    dtype=pred_all_extrinsic.dtype,
                )
                .view(1, 1, 1, 4)
                .repeat(b, image.shape[1], 1, 1)
            )
            pred_all_extrinsic = torch.cat(
                [pred_all_extrinsic, extrinsic_padding], dim=2
            ).inverse()

            ctx_agg_token_list = [
                token[:, :ctx_img_num, ...] for token in aggregated_tokens_list
            ]
            out, depth_map, depth_conf, out_tmp = self.gaussian_param_head(
                ctx_agg_token_list,
                image[:, :ctx_img_num, ...],
                patch_start_idx=patch_start_idx,
                image_size=(h, w),
            )
            del out_tmp
            depth_for_loss = depth_map
            depth_uncertainty = None
            if self.depth_refiner is not None:
                depth_base = depth_map
                depth_map, _, depth_uncertainty = self.depth_refiner(
                    out,
                    depth_base,
                    depth_conf,
                    extrinsics_c2w=gt_ex.detach(),
                    intrinsics=gt_ix.detach(),
                    return_intermediate=False,
                )

        del aggregated_tokens_list, patch_start_idx
        torch.cuda.empty_cache()
        # ====================================================================
        v = ctx_img_num
        image = image[:, :ctx_img_num, ...]
        from . import act_gs, sh_utils
        gs_feats_reshape = rearrange(out, "b s c h w -> (b s) c h w")
        with torch.amp.autocast("cuda", enabled=False):
            gs_raw = self.gs_head(gs_feats_reshape)
    
        gs_params = rearrange(gs_raw, "(b s) c h w -> b s h w c", b=b, s=v)
        quats, scales, opacities, res_sh, weights = torch.split(
            gs_params, [4, 3, 1, self.nums_sh * 3, 1], dim=-1
        )
        
        means = batchify_unproject_depth_map_to_point_map(
            depth_map, gt_ex.detach(), gt_ix.detach()
        )
        # means = means.reshape(b, v , h * w, 3).squeeze(0)
        # quats = quats.reshape(b, v , h * w, 4).squeeze(0)
        # scales = scales.reshape(b, v , h * w, 3).squeeze(0)
        # opacities = opacities.reshape(b, v , h * w, 1).squeeze(0)
        # res_sh = res_sh.reshape(b, v , h * w, self.nums_sh* 3).squeeze(0)
        
        means_raw = means.reshape(b, v, h * w, 3)
        quats_raw = quats.reshape(b, v, h * w, 4)
        scales_raw = scales.reshape(b, v, h * w, 3)
        opacities_raw = opacities.reshape(b, v, h * w, 1)
        res_sh_raw = res_sh.reshape(b, v, h * w, self.nums_sh * 3)

        means = means_raw.reshape(b, v * h * w, 3)
        quats = quats_raw.reshape(b, v * h * w, 4)
        scales = scales_raw.reshape(b, v * h * w, 3)
        opacities = opacities_raw.reshape(b, v * h * w, 1)
        res_sh = res_sh_raw.reshape(b, v * h * w, self.nums_sh * 3)
        
        sampled_rgb_raw = image.flatten(3, 4).permute(0, 1, 3, 2)
        # sampled_rgb = image.squeeze(0).flatten(2,3).permute(0,2,1)
        
        opacities_flat = act_gs.reg_dense_opacities(opacities).squeeze(-1)
    
        splats = {}
        splats["means"] = means
        splats["quats"] = act_gs.reg_dense_rotation(quats)
        
        # 依然保留你的 Scale 截断保护
        splats["scales"] = act_gs.reg_dense_scales(scales).clamp_max(0.1)
        splats["opacities"] = opacities_flat
        # new_sh = torch.zeros_like(res_sh).flatten(0, 1)
        # new_sh[:, :3] = sh_utils.RGB2SH(sampled_rgb).flatten(0, 1)
        # splats["sh"] = (new_sh + res_sh.flatten(0, 1)).unsqueeze(-2)
        
        base_sh_raw = torch.zeros_like(res_sh_raw)
        base_sh_raw[..., :3] = sh_utils.RGB2SH(sampled_rgb_raw)
        new_sh = base_sh_raw.reshape(b, v * h * w, self.nums_sh * 3)
        splats["sh"] = (new_sh + res_sh).unsqueeze(-2)
        
        key_mapping = {"quats": "rotations", "sh": "harmonics"}        
        gaussians = {key_mapping.get(k, k): v for k, v in splats.items()}
        gaussians = Gaussians(**gaussians)

        if self.cfg.mode != "train":
            intrinsic = intrinsic.clone()
            intrinsic = torch.stack(
                [intrinsic[:, :, 0] / w, intrinsic[:, :, 1] / h, intrinsic[:, :, 2]], dim=2
            )
            return gaussians, pred_all_extrinsic[:, ctx_img_num:], intrinsic, depth_map, ctx_img_num
    
        infos = {}
        if self.cfg.gs_refine_enabled:
            infos["gs_refine"] = {
                "features": out,
                "means": means_raw,
                "quats": quats_raw,
                "scales_raw": scales_raw,
                "opacities_raw": opacities_raw,
                "res_sh_raw": res_sh_raw,
                "base_sh": base_sh_raw,
                "depth": depth_map,
                "depth_conf": depth_conf,
                "depth_uncertainty": depth_uncertainty,
                "image_shape": (h, w),
                "num_context_views": v,
                "sh_degree": self.sh_degree,
            }
        depth_dict = dict(depth=depth_for_loss)
        if depth_uncertainty is not None:
            depth_dict["depth_uncertainty"] = depth_uncertainty

        # print("B:", b, "V:", v, "H:", h, "W:", w)
        extrinsic_padding = (
            torch.tensor([0, 0, 0, 1], device=device, dtype=pred_all_extrinsic.dtype)
            .view(1, 1, 1, 4)
            .repeat(b, v, 1, 1)
        )
        intrinsic = intrinsic.clone()
        intrinsic = torch.stack(
            [intrinsic[:, :, 0] / w, intrinsic[:, :, 1] / h, intrinsic[:, :, 2]], dim=2
        )
        return (
            EncoderOutput(
                gaussians=gaussians,
                pred_pose_enc_list=pred_pose_enc_list,
                pred_context_pose=dict(
                    extrinsic=torch.cat(
                        [extrinsic_padding], dim=2
                    ),
                    intrinsic=intrinsic,
                ),
                depth_dict=depth_dict,
                infos=infos,
                distill_infos=distill_infos,
            ),
            pred_all_extrinsic,
            ctx_img_num
        )        

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
