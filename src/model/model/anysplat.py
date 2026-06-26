import os
from contextlib import nullcontext
from copy import deepcopy
import time
from typing import Optional
from einops import rearrange
import huggingface_hub
from omegaconf import DictConfig, OmegaConf
import torch.distributed
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass

from src.model.types import Gaussians
from src.model.encoder import act_gs, sh_utils
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.decoder.decoder_splatting_cuda import (
    DecoderSplattingCUDA,
    DecoderSplattingCUDACfg,
)
from src.model.encoder.anysplat import (
    EncoderAnySplat,
    EncoderAnySplatCfg,
    OpacityMappingCfg,
)


class GaussianResidualRefiner(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        sh_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        evidence_dim = 9
        self.sh_dim = sh_dim
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim + evidence_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 3 + 4 + 1 + 3 + sh_dim + 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        rgb_residual: torch.Tensor,
        depth_residual: torch.Tensor,
        alpha: torch.Tensor,
        depth_conf: torch.Tensor,
        depth_uncertainty: torch.Tensor,
        opacity: torch.Tensor,
        scale_norm: torch.Tensor,
        view_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, c, h, w = features.shape
        x = torch.cat(
            [
                features,
                rgb_residual,
                depth_residual,
                alpha,
                depth_conf,
                depth_uncertainty,
                opacity,
                scale_norm,
            ],
            dim=2,
        )
        x = rearrange(x, "b s c h w -> (b s) c h w")
        delta = self.net(x)
        delta = rearrange(delta, "(b s) c h w -> b s c h w", b=b, s=s)

        delta_mean = delta[:, :, 0:3]
        delta_quat = delta[:, :, 3:7]
        delta_opacity = delta[:, :, 7:8]
        delta_scale = delta[:, :, 8:11]
        delta_sh = delta[:, :, 11 : 11 + self.sh_dim]
        learned_gate = torch.sigmoid(delta[:, :, 11 + self.sh_dim : 12 + self.sh_dim])
        gate = learned_gate * view_gate

        return (
            gate * delta_mean.tanh(),
            gate * delta_quat.tanh(),
            gate * delta_opacity.tanh(),
            gate * delta_scale.tanh(),
            gate * delta_sh.tanh(),
        )


class AnySplat(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(
        self,
        encoder_cfg: EncoderAnySplatCfg,
        decoder_cfg: DecoderSplattingCUDACfg,
    ):
        super(AnySplat, self).__init__()
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.build_encoder(encoder_cfg)
        self.build_decoder(decoder_cfg)
        self.build_gs_refiner()

    def convert_nested_config(self, cfg_dict: dict, target_class: type):
        """Convert nested dictionary config to dataclass instance

        Args:
            cfg_dict: Configuration dictionary or already converted object
            target_class: Target dataclass type to convert to

        Returns:
            Instance of target_class
        """
        if isinstance(cfg_dict, dict):
            # Convert dict to dataclass
            return target_class(**cfg_dict)
        elif isinstance(cfg_dict, target_class):
            # Already converted, return as is
            return cfg_dict
        elif hasattr(cfg_dict, "__dict__"):
            # Accept equivalent dataclasses from sibling encoder variants.
            return target_class(**cfg_dict.__dict__)
        elif cfg_dict is None:
            # Handle None case
            return None
        else:
            raise ValueError(f"Cannot convert {type(cfg_dict)} to {target_class}")

    def convert_config_recursively(self, cfg_obj, conversion_map: dict):
        """Convert nested configurations recursively using a conversion map

        Args:
            cfg_obj: Configuration object to convert
            conversion_map: Dict mapping field names to their target classes
                           e.g., {'gaussian_adapter': GaussianAdapterCfg}

        Returns:
            Converted configuration object
        """
        if not hasattr(cfg_obj, "__dict__"):
            return cfg_obj

        cfg_dict = cfg_obj.__dict__.copy()

        for field_name, target_class in conversion_map.items():
            if field_name in cfg_dict:
                cfg_dict[field_name] = self.convert_nested_config(
                    cfg_dict[field_name], target_class
                )

        # Return new instance of the same type
        return type(cfg_obj)(**cfg_dict)

    def convert_encoder_config(
        self, encoder_cfg: EncoderAnySplatCfg
    ) -> EncoderAnySplatCfg:
        """Convert all nested configurations in encoder_cfg"""
        conversion_map = {
            "gaussian_adapter": GaussianAdapterCfg,
            "opacity_mapping": OpacityMappingCfg,
        }

        return self.convert_config_recursively(encoder_cfg, conversion_map)

    def build_encoder(self, encoder_cfg: EncoderAnySplatCfg):
        # Convert nested configurations using the helper method
        encoder_cfg = self.convert_encoder_config(encoder_cfg)
        self.encoder = EncoderAnySplat(encoder_cfg)

    def build_decoder(self, decoder_cfg: DecoderSplattingCUDACfg):
        self.decoder = DecoderSplattingCUDA(decoder_cfg)

    def build_gs_refiner(self):
        cfg = self.encoder.cfg
        self.gs_residual_refiner = None
        if not getattr(cfg, "gs_refine_enabled", False):
            return

        feature_dim = self.encoder.feature_dim // 2
        sh_dim = self.encoder.nums_sh * 3
        self.gs_residual_refiner = GaussianResidualRefiner(
            feature_dim=feature_dim,
            sh_dim=sh_dim,
            hidden_dim=cfg.gs_refine_hidden_dim,
        )

    def _build_gaussians_from_refine_state(
        self,
        refine_info: dict,
        means_raw: torch.Tensor,
        quats_raw: torch.Tensor,
        scales_raw: torch.Tensor,
        opacities_raw: torch.Tensor,
        res_sh_raw: torch.Tensor,
    ) -> Gaussians:
        b, s, n, _ = means_raw.shape
        means = means_raw.reshape(b, s * n, 3)
        rotations = act_gs.reg_dense_rotation(quats_raw).reshape(b, s * n, 4)
        scales = act_gs.reg_dense_scales(scales_raw).clamp_max(0.1).reshape(b, s * n, 3)
        opacities = act_gs.reg_dense_opacities(opacities_raw).reshape(b, s * n)
        harmonics = (refine_info["base_sh"] + res_sh_raw).reshape(b, s * n, -1).unsqueeze(-2)
        return Gaussians(
            means=means,
            harmonics=harmonics,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )

    @staticmethod
    def _normalize_render_depth(depth: torch.Tensor, b: int, views: int, h: int, w: int) -> torch.Tensor:
        if depth.dim() == 2:
            depth = depth.view(1, 1, h, w)
        elif depth.dim() == 3:
            if depth.shape[0] == b * views:
                depth = depth.view(b, views, h, w)
            elif depth.shape[0] == views and b == 1:
                depth = depth.unsqueeze(0)
            else:
                depth = depth.view(b, views, h, w)
        elif depth.dim() != 4:
            depth = depth.reshape(b, views, h, w)
        return depth.unsqueeze(2)

    @staticmethod
    def _normalize_render_alpha(alpha: torch.Tensor, b: int, views: int, h: int, w: int) -> torch.Tensor:
        if alpha.dim() == 2:
            alpha = alpha.view(1, 1, h, w)
        elif alpha.dim() == 3:
            if alpha.shape[0] == b * views:
                alpha = alpha.view(b, views, h, w)
            elif alpha.shape[0] == views and b == 1:
                alpha = alpha.unsqueeze(0)
            else:
                alpha = alpha.view(b, views, h, w)
        elif alpha.dim() != 4:
            alpha = alpha.reshape(b, views, h, w)
        return alpha.unsqueeze(2)

    def _refine_gaussians(
        self,
        encoder_output,
        context_image: torch.Tensor,
        pred_all_extrinsic: torch.Tensor,
        pred_context_pose: dict,
        ctx_img_num: int,
        near: float,
        far: float,
    ) -> Gaussians:
        if self.gs_residual_refiner is None:
            return encoder_output.gaussians
        refine_info = None if encoder_output.infos is None else encoder_output.infos.get("gs_refine")
        if refine_info is None:
            return encoder_output.gaussians

        cfg = self.encoder.cfg
        features = refine_info["features"]
        b, s, _, h, w = features.shape
        device = features.device
        evidence_features = features.detach() if cfg.gs_refine_detach_evidence else features
        render_scale = float(max(0.0, min(1.0, cfg.gs_refine_render_scale)))
        low_h = max(8, int(round(h * render_scale)))
        low_w = max(8, int(round(w * render_scale)))
        render_views = ctx_img_num
        if cfg.gs_refine_max_render_views is not None:
            render_views = min(render_views, int(cfg.gs_refine_max_render_views))
        render_views = max(1, render_views)

        means_raw = refine_info["means"]
        quats_raw = refine_info["quats"]
        scales_raw = refine_info["scales_raw"]
        opacities_raw = refine_info["opacities_raw"]
        res_sh_raw = refine_info["res_sh_raw"]

        depth = refine_info["depth"]
        if cfg.gs_refine_detach_evidence:
            depth = depth.detach()
        depth_low = rearrange(depth, "b s h w c -> (b s) c h w")
        depth_low = F.interpolate(depth_low.float(), size=(low_h, low_w), mode="bilinear", align_corners=False)
        depth_low = rearrange(depth_low, "(b s) c h w -> b s c h w", b=b, s=s)

        depth_conf = refine_info["depth_conf"]
        if cfg.gs_refine_detach_evidence:
            depth_conf = depth_conf.detach()
        depth_conf = rearrange(depth_conf, "b s h w c -> (b s) c h w")
        depth_conf = F.interpolate(depth_conf.float(), size=(h, w), mode="bilinear", align_corners=False)
        depth_conf = rearrange(depth_conf, "(b s) c h w -> b s c h w", b=b, s=s)

        depth_uncertainty = refine_info["depth_uncertainty"]
        if depth_uncertainty is None:
            depth_uncertainty = torch.zeros((b, s, 1, h, w), device=device, dtype=features.dtype)
        else:
            if cfg.gs_refine_detach_evidence:
                depth_uncertainty = depth_uncertainty.detach()
            depth_uncertainty = rearrange(depth_uncertainty, "b s h w c -> (b s) c h w")
            depth_uncertainty = F.interpolate(
                depth_uncertainty.float(), size=(h, w), mode="bilinear", align_corners=False
            )
            depth_uncertainty = rearrange(depth_uncertainty, "(b s) c h w -> b s c h w", b=b, s=s)

        target_low = rearrange(context_image[:, :s], "b s c h w -> (b s) c h w")
        target_low = F.interpolate(target_low.float(), size=(low_h, low_w), mode="bilinear", align_corners=False)
        target_low = rearrange(target_low, "(b s) c h w -> b s c h w", b=b, s=s)

        render_intrinsics = pred_context_pose["intrinsic"][:, 0:1, ...].repeat(1, render_views, 1, 1).detach()
        render_extrinsics = pred_all_extrinsic[:, :render_views].detach()
        near_tensor = torch.ones(b, render_views, device=device) * near
        far_tensor = torch.ones(b, render_views, device=device) * far
        view_gate = torch.zeros((b, s, 1, h, w), device=device, dtype=features.dtype)
        view_gate[:, :render_views] = 1.0

        for _ in range(max(0, int(cfg.gs_refine_iters))):
            render_context = torch.no_grad() if cfg.gs_refine_detach_evidence else nullcontext()
            with render_context:
                current_gaussians = self._build_gaussians_from_refine_state(
                    refine_info,
                    means_raw,
                    quats_raw,
                    scales_raw,
                    opacities_raw,
                    res_sh_raw,
                )
                low_output = self.decoder.forward(
                    current_gaussians,
                    render_extrinsics,
                    render_intrinsics,
                    near_tensor,
                    far_tensor,
                    (low_h, low_w),
                    "depth",
                )
            render_color = low_output.color.detach() if cfg.gs_refine_detach_evidence else low_output.color
            render_depth = self._normalize_render_depth(
                low_output.depth.detach() if cfg.gs_refine_detach_evidence else low_output.depth,
                b,
                render_views,
                low_h,
                low_w,
            )
            render_alpha = self._normalize_render_alpha(
                low_output.alpha.detach() if cfg.gs_refine_detach_evidence else low_output.alpha,
                b,
                render_views,
                low_h,
                low_w,
            )

            rgb_residual_low = torch.zeros((b, s, 3, low_h, low_w), device=device, dtype=features.dtype)
            depth_residual_low = torch.zeros((b, s, 1, low_h, low_w), device=device, dtype=features.dtype)
            alpha_low = torch.zeros((b, s, 1, low_h, low_w), device=device, dtype=features.dtype)

            rgb_residual_low[:, :render_views] = (render_color - target_low[:, :render_views]).to(features.dtype)
            depth_target = depth_low[:, :render_views].clamp_min(1e-4)
            depth_residual_low[:, :render_views] = (
                (render_depth - depth_target) / depth_target
            ).clamp(-1.0, 1.0).to(features.dtype)
            alpha_low[:, :render_views] = render_alpha.to(features.dtype)

            rgb_residual = rearrange(rgb_residual_low, "b s c h w -> (b s) c h w")
            rgb_residual = F.interpolate(rgb_residual, size=(h, w), mode="bilinear", align_corners=False)
            rgb_residual = rearrange(rgb_residual, "(b s) c h w -> b s c h w", b=b, s=s)

            depth_residual = rearrange(depth_residual_low, "b s c h w -> (b s) c h w")
            depth_residual = F.interpolate(depth_residual, size=(h, w), mode="bilinear", align_corners=False)
            depth_residual = rearrange(depth_residual, "(b s) c h w -> b s c h w", b=b, s=s)

            alpha = rearrange(alpha_low, "b s c h w -> (b s) c h w")
            alpha = F.interpolate(alpha, size=(h, w), mode="bilinear", align_corners=False)
            alpha = rearrange(alpha, "(b s) c h w -> b s c h w", b=b, s=s)

            opacity_source = opacities_raw.detach() if cfg.gs_refine_detach_evidence else opacities_raw
            scale_source = scales_raw.detach() if cfg.gs_refine_detach_evidence else scales_raw
            opacity = act_gs.reg_dense_opacities(opacity_source)
            opacity = rearrange(opacity, "b s (h w) c -> b s c h w", h=h, w=w)
            scale_norm = act_gs.reg_dense_scales(scale_source).clamp_max(0.1).norm(dim=-1, keepdim=True)
            scale_norm = rearrange(scale_norm, "b s (h w) c -> b s c h w", h=h, w=w)

            delta_mean, delta_quat, delta_opacity, delta_scale, delta_sh = self.gs_residual_refiner(
                evidence_features.float(),
                rgb_residual.float(),
                depth_residual.float(),
                alpha.float(),
                depth_conf.float(),
                depth_uncertainty.float(),
                opacity.float(),
                scale_norm.float(),
                view_gate.float(),
            )

            delta_mean = rearrange(delta_mean, "b s c h w -> b s (h w) c")
            delta_quat = rearrange(delta_quat, "b s c h w -> b s (h w) c")
            delta_opacity = rearrange(delta_opacity, "b s c h w -> b s (h w) c")
            delta_scale = rearrange(delta_scale, "b s c h w -> b s (h w) c")
            delta_sh = rearrange(delta_sh, "b s c h w -> b s (h w) c")

            mean_step_source = depth.detach() if cfg.gs_refine_detach_evidence else depth
            mean_step = rearrange(mean_step_source.float(), "b s h w c -> b s (h w) c")
            means_raw = means_raw + mean_step.to(means_raw.dtype) * delta_mean.to(means_raw.dtype)
            quats_raw = quats_raw + delta_quat.to(quats_raw.dtype)
            opacities_raw = opacities_raw + cfg.gs_refine_step_opacity * delta_opacity.to(opacities_raw.dtype)
            scales_raw = scales_raw + cfg.gs_refine_step_scale * delta_scale.to(scales_raw.dtype)
            res_sh_raw = res_sh_raw + cfg.gs_refine_step_sh * delta_sh.to(res_sh_raw.dtype)

            del (
                current_gaussians,
                low_output,
                render_color,
                render_depth,
                render_alpha,
                rgb_residual_low,
                depth_residual_low,
                alpha_low,
                rgb_residual,
                depth_residual,
                alpha,
                opacity_source,
                scale_source,
                opacity,
                scale_norm,
                delta_mean,
                delta_quat,
                delta_opacity,
                delta_scale,
                delta_sh,
                mean_step_source,
                mean_step,
            )

        refined_gaussians = self._build_gaussians_from_refine_state(
            refine_info,
            means_raw,
            quats_raw,
            scales_raw,
            opacities_raw,
            res_sh_raw,
        )
        if encoder_output.infos is not None:
            encoder_output.infos.pop("gs_refine", None)
            encoder_output.infos["gs_refine_steps"] = torch.tensor(
                int(cfg.gs_refine_iters), device=device
            )
        return refined_gaussians

    @torch.no_grad()
    def inference(
        self,
        context_image: torch.Tensor,
    ):
        self.encoder.distill = False
        encoder_output = self.encoder(
            context_image, global_step=0, visualization_dump=None
        )
        gaussians, pred_context_pose = (
            encoder_output.gaussians,
            encoder_output.pred_context_pose,
        )
        return gaussians, pred_context_pose

    def forward(
        self,
        context_image: torch.Tensor,
        ctx_index: list = None, 
        global_step: int = 0,
        near: float = 0.01,
        far: float = 100.0,
    ):
        b, v, c, h, w = context_image.shape
        device = context_image.device

        encoder_output, pred_all_extrinsic, ctx_img_num = self.encoder(
            context_image, ctx_index, global_step=global_step
        )
        gaussians, pred_context_pose = (
            encoder_output.gaussians,
            encoder_output.pred_context_pose,
        )
        gaussians = self._refine_gaussians(
            encoder_output,
            context_image,
            pred_all_extrinsic,
            pred_context_pose,
            ctx_img_num,
            near,
            far,
        )
        encoder_output.gaussians = gaussians

        # num_context_view = ctx_img_num
        # pred_all_context_extrinsic, pred_all_target_extrinsic = (
        #     pred_all_extrinsic[:, :num_context_view],
        #     pred_all_extrinsic[:, num_context_view:],
        # )
        # scale_factor = (
        #     pred_context_pose["extrinsic"][:, :, :3, 3].mean()
        #     / pred_all_context_extrinsic[:, :, :3, 3].mean()
        # )
        # pred_all_target_extrinsic[..., :3, 3] = (
        #     pred_all_target_extrinsic[..., :3, 3] * scale_factor
        # )
        # pred_all_context_extrinsic[..., :3, 3] = (
        #     pred_all_context_extrinsic[..., :3, 3] * scale_factor
        # )
        # pred_context_ex = torch.cat(
        #     (pred_context_pose["extrinsic"], pred_all_target_extrinsic), dim=1
        # )

        output = self.decoder.forward(
            gaussians,
            pred_all_extrinsic.detach(),
            pred_context_pose["intrinsic"][:, 0:1, ...].repeat(1, v, 1, 1).detach(),
            torch.ones(b, v, device=device) * near,
            torch.ones(b, v, device=device) * far,
            (h, w),
            "depth",
        )
        output.depth = output.depth[:, :ctx_img_num, ...]

        return encoder_output, output
