from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import torchvision

from ..types import Gaussians
# from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput
from math import prod, sqrt
from gsplat import rasterization

try:
    from gsplat import rasterize_top_contributing_gaussian_ids
except ImportError:
    rasterize_top_contributing_gaussian_ids = None

try:
    from gsplat import rasterize_to_indices_in_range
except ImportError:
    rasterize_to_indices_in_range = None

from ...misc.utils import vis_depth_map

DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]

@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool


@dataclass
class GIRRasterizationOutput:
    color: Tensor
    expected_depth: Tensor
    alpha: Tensor
    dominant_ids: Tensor
    dominant_weights: Tensor
    contributor_ids: Tensor | None = None
    contributor_weights: Tensor | None = None


@torch.no_grad()
def _rasterize_top_contributor_fallback(
    means2d: Tensor,
    conics: Tensor,
    opacities: Tensor,
    tile_offsets: Tensor,
    flatten_ids: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    num_depth_samples: int = 1,
) -> tuple[Tensor, Tensor]:
    """Reconstruct top alpha*T contributors with the gsplat 1.5.3 API."""
    if rasterize_to_indices_in_range is None:
        raise RuntimeError(
            "Dominant GIR requires gsplat 1.5.3 or a newer build with a "
            "contributor rasterization API."
        )

    image_dims = means2d.shape[:-2]
    num_images = prod(image_dims)
    num_gaussians = means2d.shape[-2]
    pixel_count = image_height * image_width
    if num_depth_samples <= 0:
        raise ValueError("num_depth_samples must be greater than zero.")
    output_shape = image_dims + (
        image_height,
        image_width,
        num_depth_samples,
    )

    dominant_ids = torch.full(
        (num_images * pixel_count, num_depth_samples),
        -1,
        device=means2d.device,
        dtype=torch.long,
    )
    dominant_weights = torch.zeros(
        (num_images * pixel_count, num_depth_samples),
        device=means2d.device,
        dtype=means2d.dtype,
    )
    if num_gaussians == 0:
        return dominant_ids.reshape(output_shape), dominant_weights.reshape(
            output_shape
        )

    transmittances = torch.ones(
        image_dims + (image_height, image_width),
        device=means2d.device,
        dtype=means2d.dtype,
    )
    gaussian_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
        range_start=0,
        range_end=2**31 - 1,
        transmittances=transmittances,
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=tile_offsets,
        flatten_ids=flatten_ids,
    )
    if gaussian_ids.numel() == 0:
        return dominant_ids.reshape(output_shape), dominant_weights.reshape(
            output_shape
        )

    gaussian_ids = gaussian_ids.long()
    pixel_ids = pixel_ids.long()
    image_ids = image_ids.long()
    flat_means2d = means2d.reshape(num_images, num_gaussians, 2)
    flat_conics = conics.reshape(num_images, num_gaussians, 3)
    flat_opacities = opacities.reshape(num_images, num_gaussians)

    pair_means = flat_means2d[image_ids, gaussian_ids]
    pair_conics = flat_conics[image_ids, gaussian_ids]
    pair_opacities = flat_opacities[image_ids, gaussian_ids]
    pixel_x = (pixel_ids % image_width).to(means2d.dtype) + 0.5
    pixel_y = torch.div(
        pixel_ids, image_width, rounding_mode="floor"
    ).to(means2d.dtype) + 0.5
    delta_x = pair_means[:, 0] - pixel_x
    delta_y = pair_means[:, 1] - pixel_y
    sigma = (
        0.5
        * (
            pair_conics[:, 0] * delta_x.square()
            + pair_conics[:, 2] * delta_y.square()
        )
        + pair_conics[:, 1] * delta_x * delta_y
    )
    pair_alpha = (pair_opacities * torch.exp(-sigma)).clamp_(0.0, 0.999)

    # gsplat 1.5.3 writes pairs grouped by pixel and front-to-back within
    # each group. A float64 prefix avoids precision loss when segment bases
    # are subtracted after a long flattened cumulative sum.
    global_pixel_ids = image_ids * pixel_count + pixel_ids
    segment_start = torch.ones_like(global_pixel_ids, dtype=torch.bool)
    segment_start[1:] = global_pixel_ids[1:] != global_pixel_ids[:-1]
    segment_ids = segment_start.long().cumsum(0) - 1
    log_survival = torch.log1p(-pair_alpha.double())
    inclusive_log_survival = log_survival.cumsum(0)
    exclusive_log_survival = inclusive_log_survival - log_survival
    segment_bases = exclusive_log_survival[segment_start]
    local_exclusive_log_survival = (
        exclusive_log_survival - segment_bases[segment_ids]
    )
    pair_weights = pair_alpha * local_exclusive_log_survival.exp().to(
        pair_alpha.dtype
    )

    pair_positions = torch.arange(
        gaussian_ids.numel(), device=gaussian_ids.device, dtype=torch.long
    )
    sentinel = gaussian_ids.numel()
    remaining_weights = pair_weights.clone()
    flat_pixel_count = num_images * pixel_count
    for rank in range(num_depth_samples):
        rank_weights = torch.zeros(
            flat_pixel_count,
            device=means2d.device,
            dtype=means2d.dtype,
        )
        rank_weights.scatter_reduce_(
            0,
            global_pixel_ids,
            remaining_weights,
            reduce="amax",
            include_self=True,
        )
        winning_positions = torch.full(
            (flat_pixel_count,),
            sentinel,
            device=gaussian_ids.device,
            dtype=torch.long,
        )
        is_winner = (
            remaining_weights == rank_weights[global_pixel_ids]
        ) & (remaining_weights > 0)
        winning_positions.scatter_reduce_(
            0,
            global_pixel_ids,
            torch.where(
                is_winner,
                pair_positions,
                torch.full_like(pair_positions, sentinel),
            ),
            reduce="amin",
            include_self=True,
        )
        valid = winning_positions < sentinel
        dominant_weights[:, rank] = torch.where(
            valid, rank_weights, torch.zeros_like(rank_weights)
        )
        dominant_ids[valid, rank] = gaussian_ids[winning_positions[valid]]
        if not valid.any():
            break
        remaining_weights[winning_positions[valid]] = -1
    return dominant_ids.reshape(output_shape), dominant_weights.reshape(output_shape)


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]
    
    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def render_gir_evidence(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        image_shape: tuple[int, int],
        num_top_contributors: int = 1,
    ) -> GIRRasterizationOutput:
        if (
            rasterize_top_contributing_gaussian_ids is None
            and rasterize_to_indices_in_range is None
        ):
            raise RuntimeError(
                "Dominant GIR requires gsplat 1.5.3 or a newer contributor "
                "rasterization API."
            )

        num_top_contributors = max(1, int(num_top_contributors))
        b = gaussians.means.shape[0]
        h, w = image_shape
        colors = []
        expected_depths = []
        alphas = []
        dominant_ids = []
        dominant_weights = []
        contributor_ids = []
        contributor_weights = []

        for batch_idx in range(b):
            xyz = gaussians.means[batch_idx].float()
            features = gaussians.harmonics[batch_idx].float()
            scales = gaussians.scales[batch_idx].float()
            rotations = gaussians.rotations[batch_idx].float()
            opacities = gaussians.opacities[batch_idx].float()
            world_to_camera = extrinsics[batch_idx : batch_idx + 1].float().inverse()
            intrinsics_px = intrinsics[batch_idx : batch_idx + 1].float().clone()
            intrinsics_px[:, 0] = intrinsics_px[:, 0] * w
            intrinsics_px[:, 1] = intrinsics_px[:, 1] * h
            sh_degree = int(sqrt(features.shape[-2])) - 1

            rendering, alpha, info = rasterization(
                xyz,
                rotations,
                scales,
                opacities,
                features,
                world_to_camera,
                intrinsics_px,
                w,
                h,
                sh_degree=sh_degree,
                render_mode="RGB+ED",
                packed=False,
                near_plane=1e-10,
                backgrounds=self.background_color.unsqueeze(0),
                radius_clip=0.1,
                rasterize_mode="classic",
            )
            required_info = {
                "means2d",
                "conics",
                "opacities",
                "isect_offsets",
                "flatten_ids",
                "tile_size",
            }
            missing_info = required_info.difference(info)
            if missing_info:
                missing = ", ".join(sorted(missing_info))
                raise RuntimeError(
                    f"The installed gsplat GIR API is incompatible; missing: {missing}."
                )

            contributor_args = {
                "means2d": info["means2d"],
                "conics": info["conics"],
                "opacities": info["opacities"],
                "tile_offsets": info["isect_offsets"],
                "flatten_ids": info["flatten_ids"],
                "image_width": w,
                "image_height": h,
                "tile_size": info["tile_size"],
            }
            def rasterize_contributors(count: int) -> tuple[Tensor, Tensor]:
                if rasterize_top_contributing_gaussian_ids is not None:
                    return rasterize_top_contributing_gaussian_ids(
                        **contributor_args,
                        num_depth_samples=count,
                    )
                return _rasterize_top_contributor_fallback(
                    **contributor_args,
                    num_depth_samples=count,
                )

            # Keep the legacy K=1 query as the canonical dominant match. This
            # makes K=1 checkpoint behavior independent of the top-k API.
            dominant_id, dominant_weight = rasterize_contributors(1)
            if num_top_contributors == 1:
                ids, weights = dominant_id, dominant_weight
            else:
                ids, weights = rasterize_contributors(num_top_contributors)
            expected_id_shape = (1, h, w, num_top_contributors)
            if ids.shape != expected_id_shape or weights.shape != expected_id_shape:
                raise RuntimeError(
                    "Unexpected contributor GIR output shapes: "
                    f"ids={tuple(ids.shape)}, weights={tuple(weights.shape)}, "
                    f"expected={expected_id_shape}."
                )
            weights = torch.nan_to_num(
                weights.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min_(0.0)
            dominant_match = ids == dominant_id
            ordering_score = weights + dominant_match.to(weights.dtype) * (
                weights.amax(dim=-1, keepdim=True) + 1.0
            )
            order = ordering_score.argsort(dim=-1, descending=True)
            ids = ids.gather(-1, order)
            weights = weights.gather(-1, order)
            ids[..., :1] = dominant_id
            weights[..., :1] = dominant_weight.to(weights.dtype)

            expected_dominant_shape = (1, h, w, 1)
            if (
                dominant_id.shape != expected_dominant_shape
                or dominant_weight.shape != expected_dominant_shape
            ):
                raise RuntimeError(
                    "Unexpected dominant GIR output shapes: "
                    f"ids={tuple(dominant_id.shape)}, "
                    f"weights={tuple(dominant_weight.shape)}, "
                    f"expected={expected_dominant_shape}."
                )

            rgb, depth = torch.split(rendering, [3, 1], dim=-1)
            colors.append(rgb[0].permute(2, 0, 1).clamp(0.0, 1.0))
            expected_depths.append(depth[0].permute(2, 0, 1))
            alphas.append(alpha[0].permute(2, 0, 1))
            dominant_ids.append(dominant_id[0, ..., 0].long())
            dominant_weights.append(dominant_weight[0, ..., 0].unsqueeze(0))
            contributor_ids.append(ids[0].permute(2, 0, 1).long())
            contributor_weights.append(weights[0].permute(2, 0, 1))

        return GIRRasterizationOutput(
            color=torch.stack(colors),
            expected_depth=torch.stack(expected_depths),
            alpha=torch.stack(alphas),
            dominant_ids=torch.stack(dominant_ids),
            dominant_weights=torch.stack(dominant_weights),
            contributor_ids=torch.stack(contributor_ids),
            contributor_weights=torch.stack(contributor_weights),
        )

    def rendering_fn(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:
        B, V, _, _  = intrinsics.shape
        H, W = image_shape
        rendered_imgs, rendered_depths, rendered_alphas = [], [], []
        # xyzs, opacitys, rotations, scales, features = gaussians.means, gaussians.opacities, gaussians.rotations, gaussians.scales, gaussians.harmonics.permute(0, 1, 3, 2).contiguous()
        xyzs, opacitys, rotations, scales, features = gaussians.means, gaussians.opacities, gaussians.rotations, gaussians.scales, gaussians.harmonics
        # covariances = gaussians.covariances
        for i in range(B):
            xyz_i = xyzs[i].float()
            feature_i = features[i].float()
            # covar_i = covariances[i].float()
            scale_i = scales[i].float()
            rotation_i = rotations[i].float()
            opacity_i = opacitys[i].float()
            test_w2c_i = extrinsics[i].float().inverse() # (V, 4, 4)
            test_intr_i_normalized = intrinsics[i].float()
            # Denormalize the intrinsics into standred format
            test_intr_i = test_intr_i_normalized.clone()
            test_intr_i[:, 0] = test_intr_i_normalized[:, 0] * W
            test_intr_i[:, 1] = test_intr_i_normalized[:, 1] * H
            sh_degree = (int(sqrt(feature_i.shape[-2])) - 1)

            rendering_list = []
            rendering_depth_list = []
            rendering_alpha_list = []
            for j in range(V):
                rendering, alpha, *ignored = rasterization(xyz_i, rotation_i, scale_i, opacity_i, feature_i,
                                                test_w2c_i[j:j+1], test_intr_i[j:j+1], W, H, 
                                                sh_degree=sh_degree, 
                                                # near_plane=near[i].mean(), far_plane=far[i].mean(),
                                                render_mode="RGB+D", packed=False,
                                                near_plane=1e-10,
                                                backgrounds=self.background_color.unsqueeze(0).repeat(1, 1),
                                                radius_clip=0.1,
                                                # covars=covar_i,
                                                rasterize_mode='classic'
                                                        ) # (V, H, W, 3) 
                rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1)
                rendering_img = rendering_img.clamp(0.0, 1.0)
                rendering_list.append(rendering_img.permute(0, 3, 1, 2))
                rendering_depth_list.append(rendering_depth)
                rendering_alpha_list.append(alpha)
            rendered_depths.append(torch.cat(rendering_depth_list, dim=0).squeeze())
            rendered_imgs.append(torch.cat(rendering_list, dim=0))
            rendered_alphas.append(torch.cat(rendering_alpha_list, dim=0).squeeze())
        return DecoderOutput(torch.stack(rendered_imgs), torch.stack(rendered_depths), torch.stack(rendered_alphas), lod_rendering=None)

    def render_gradient_selection_mask(
        self,
        gaussians: Gaussians,
        target_images: Float[Tensor, "batch view 3 height width"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        image_shape: tuple[int, int],
        k_num: int,
    ) -> torch.Tensor:
        B, V, _, _ = intrinsics.shape
        H, W = image_shape
        masks = []
        xyzs, opacitys, rotations, scales, features = (
            gaussians.means,
            gaussians.opacities,
            gaussians.rotations,
            gaussians.scales,
            gaussians.harmonics,
        )
        for i in range(B):
            xyz_i = xyzs[i].float()
            feature_i = features[i].float()
            scale_i = scales[i].float()
            rotation_i = rotations[i].float()
            opacity_i = opacitys[i].float()
            test_w2c_i = extrinsics[i].float().inverse()
            test_intr_i = intrinsics[i].float().clone()
            test_intr_i[:, 0] = test_intr_i[:, 0] * W
            test_intr_i[:, 1] = test_intr_i[:, 1] * H
            sh_degree = int(sqrt(feature_i.shape[-2])) - 1

            means2d_per_view = []
            losses = []
            for j in range(V):
                rendering, _, info = rasterization(
                    xyz_i,
                    rotation_i,
                    scale_i,
                    opacity_i,
                    feature_i,
                    test_w2c_i[j : j + 1],
                    test_intr_i[j : j + 1],
                    W,
                    H,
                    sh_degree=sh_degree,
                    render_mode="RGB",
                    packed=False,
                    near_plane=1e-10,
                    backgrounds=self.background_color.unsqueeze(0).repeat(1, 1),
                    radius_clip=0.1,
                    rasterize_mode="classic",
                    absgrad=True,
                )
                means2d = info.get("means2d")
                if means2d is not None:
                    means2d.retain_grad()
                    means2d_per_view.append(means2d)
                pred = rendering.permute(0, 3, 1, 2)
                losses.append((pred - target_images[i : i + 1, j]).pow(2).mean())

            if not means2d_per_view:
                masks.append(torch.ones_like(opacity_i, dtype=torch.bool))
                continue

            loss = torch.stack(losses).mean()
            grads = torch.autograd.grad(
                loss,
                means2d_per_view,
                retain_graph=True,
                allow_unused=True,
            )

            grad_score = xyz_i.new_zeros(xyz_i.shape[0])
            for grad in grads:
                if grad is None:
                    continue
                grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                if grad.shape[-1] >= 2:
                    grad_score = grad_score + torch.norm(grad[..., :2].mean(dim=0), dim=-1)

            valid = grad_score > 0
            if not valid.any():
                valid = opacity_i.detach() > 0.005
            if k_num > 0 and valid.sum() > k_num:
                score = grad_score.masked_fill(~valid, -1.0)
                topk = torch.topk(score, k_num, dim=0).indices
                mask = torch.zeros_like(valid)
                mask[topk] = True
            else:
                mask = valid
            masks.append(mask)

        return torch.stack(masks, dim=0)

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:
        
        return self.rendering_fn(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode, cam_rot_delta, cam_trans_delta)
