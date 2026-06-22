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
from math import sqrt 
from gsplat import rasterization, rasterization_2dgs

from ...misc.utils import vis_depth_map

DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]

@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool


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
