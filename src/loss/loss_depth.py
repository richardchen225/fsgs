from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss
from typing import Literal, TypeVar
from dataclasses import fields
import torch.nn.functional as F
import sys
import os
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


@dataclass
class LossDepthCfg:
    weight: float
    sigma_image: float | None
    use_second_derivative: bool
    dav3_weights_path: str | None = None
    teacher: Literal["dav3", "abot"] = "dav3"


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

        if self.cfg.teacher not in ("dav3", "abot"):
            raise ValueError(f"Unknown depth teacher: {self.cfg.teacher}")
        loss_kind = "log-depth L1 (no scale/shift alignment)" if self.cfg.teacher == "abot" else "per-image scale+shift aligned MSE"
        print(f"Depth supervision: teacher={self.cfg.teacher}, {loss_kind}")
        if self.cfg.teacher == "abot":
            # Teacher Z arrives from the frozen source-only ABot forward.
            # In particular, do not import/load/run Depth Anything 3 here.
            return

        from .dav3.src.depth_anything_3.api import DepthAnything3

        if self.cfg.dav3_weights_path is None:
            raise ValueError(
                "loss.depth.dav3_weights_path must be set to the Depth Anything 3 checkpoint path."
            )
        dav3_weights_path = Path(self.cfg.dav3_weights_path)
        if not dav3_weights_path.exists():
            raise FileNotFoundError(f"Depth Anything 3 checkpoint not found: {dav3_weights_path}")

        device = torch.device("cuda")
        model = DepthAnything3(checkpoint_path=str(dav3_weights_path))
        model = model.to(device=device)
        self.depth_anything = model

    def _context_depth_target(self, depth_map: torch.Tensor, batch) -> torch.Tensor:
        B, V, _, H, W = batch["context"]["image"].shape
        ctx_num = depth_map.shape[1]
        ctx_imgs = (
            batch["context"]["image"][:, :ctx_num, ...]
            .reshape(B * ctx_num, 3, H, W)
            .float()
        )

        with torch.no_grad():
            ctx_imgs_tmp = ctx_imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            ctx_imgs_tmp = ((ctx_imgs_tmp + 1) / 2 * 255.0).astype(np.uint8)
            ctx_imgs_list = [ctx_imgs_tmp[i] for i in range(ctx_imgs_tmp.shape[0])]

            da_output, *ig = self.depth_anything.inference(ctx_imgs_list)
            da_output = torch.from_numpy(da_output.depth).to(depth_map.device)
            da_output = F.interpolate(
                da_output[:, None], (H, W), mode="bilinear", align_corners=True
            ).squeeze(1)

        return da_output

    @staticmethod
    def _align_depth(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        N = prediction.shape[0]
        p = prediction.reshape(N, -1)
        t = target.reshape(N, -1)

        p_mean = p.mean(dim=-1, keepdim=True)
        t_mean = t.mean(dim=-1, keepdim=True)

        p_centered = p - p_mean
        t_centered = t - t_mean

        upper = (p_centered * t_centered).sum(dim=-1, keepdim=True)
        lower = (p_centered**2).sum(dim=-1, keepdim=True) + 1e-8
        s = upper / lower
        s = torch.clamp(s, min=1e-4, max=100.0)
        shift = t_mean - s * p_mean

        return s.view(N, 1, 1) * prediction + shift.view(N, 1, 1)

    def ctx_depth_loss(
        self,
        depth_map: torch.Tensor,  # [B, V, H, W, C]
        batch,
        cxt_depth_weight: float = 0.01,
        teacher_depth: torch.Tensor | None = None,
    ):
        if self.cfg.teacher == "abot":
            if teacher_depth is None:
                raise RuntimeError("ABot depth supervision requires source teacher_depth from the encoder.")
            if teacher_depth.shape != depth_map.shape:
                raise ValueError(
                    "ABot teacher/student depth shapes must match [B,S,H,W,1]: "
                    f"teacher={tuple(teacher_depth.shape)}, student={tuple(depth_map.shape)}"
                )
            prediction = depth_map.float().flatten(0, 1).squeeze(-1)
            target = teacher_depth.detach().to(device=depth_map.device, dtype=torch.float32).flatten(0, 1).squeeze(-1)
            return cxt_depth_weight * self._masked_log_depth_l1(prediction, target)

        da_output = self._context_depth_target(depth_map, batch)
        pred_depth = depth_map.flatten(0, 1).squeeze(-1)
        aligned_pred1 = self._align_depth(pred_depth, da_output)
        loss_local = F.mse_loss(aligned_pred1, da_output, reduction="none").mean()

        return cxt_depth_weight * torch.nan_to_num(loss_local, nan=0.0)

    @staticmethod
    def _masked_log_depth_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Keep the native teacher scale; sanitize invalid targets BEFORE log
        # so masked NaNs cannot contaminate backward.
        valid = torch.isfinite(target) & (target > 0)
        p = torch.where(valid, prediction, 1).clamp_min(1e-6)
        t = torch.where(valid, target, 1).clamp_min(1e-6)
        error = (p.log() - t.log()).abs()
        # An all-invalid teacher returns graph-connected zero, not a detached
        # scalar; this keeps the student branch in DDP's backward graph.
        return error.sum() / valid.sum().clamp_min(1)

    def ctx_depth_sequence_loss(
        self,
        depth_iters: torch.Tensor,  # [B, R, V, H, W, C]
        batch,
        cxt_depth_weight: float = 0.01,
        iter_weights: torch.Tensor | None = None,
        aux_weight: float = 0.5,
        final_weight: float = 1.0,
    ):
        if self.cfg.teacher != "dav3":
            raise RuntimeError("ABot teacher supervises the initial DPT depth via ctx_depth_loss only.")
        if depth_iters.numel() == 0:
            return depth_iters.new_tensor(0.0)

        B, R, V, H, W, C = depth_iters.shape
        da_output = self._context_depth_target(depth_iters[:, -1], batch)
        pred_depth = depth_iters.permute(1, 0, 2, 3, 4, 5).reshape(R, B * V, H, W, C).squeeze(-1)

        losses = []
        for iter_idx in range(R):
            aligned_pred = self._align_depth(pred_depth[iter_idx], da_output)
            loss_iter = F.mse_loss(aligned_pred, da_output, reduction="none").mean()
            losses.append(torch.nan_to_num(loss_iter, nan=0.0))

        losses = torch.stack(losses)
        final_loss = losses[-1] * final_weight
        if R == 1 or aux_weight <= 0:
            return cxt_depth_weight * final_loss

        aux_losses = losses[:-1]
        if iter_weights is None:
            iter_weights = torch.linspace(
                1.0 / max(1, R - 1),
                1.0,
                R - 1,
                device=depth_iters.device,
                dtype=losses.dtype,
            )
        else:
            iter_weights = iter_weights.to(device=depth_iters.device, dtype=losses.dtype)
        iter_weights = iter_weights / iter_weights.sum().clamp_min(1e-8)
        aux_loss = torch.sum(aux_losses * iter_weights)

        return cxt_depth_weight * (final_loss + aux_weight * aux_loss)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        if self.cfg.teacher != "dav3":
            raise RuntimeError("Use ctx_depth_loss with encoder teacher_depth for ABot supervision.")
        # Scale the depth between the near and far planes.
        target_imgs = batch["target"]["image"]
        B, V, _, H, W = target_imgs.shape
        target_imgs = target_imgs.reshape(B * V, 3, H, W)
        da_output = self.depth_anything(target_imgs.float())
        da_output = self.disp_rescale(da_output)

        disp_gs = 1.0 / prediction.depth.flatten(0, 1).clamp(1e-3).float()
        gs_output = self.disp_rescale(disp_gs)

        return self.cfg.weight * torch.nan_to_num(
            F.smooth_l1_loss(da_output, gs_output), nan=0.0
        )
