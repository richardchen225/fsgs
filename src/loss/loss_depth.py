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
    teacher: Literal["dav3"] = "dav3"
    alignment: Literal["sequence_scale_only_log"] = "sequence_scale_only_log"
    align_to_gt_camera_scale: bool = True
    absolute_log_scale_weight: float = 0.1


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

        if self.cfg.teacher != "dav3":
            raise ValueError(f"Unknown depth teacher: {self.cfg.teacher}")
        loss_kind = str(self.cfg.alignment)
        if loss_kind != "sequence_scale_only_log":
            raise ValueError(
                "DAV3 depth supervision requires alignment="
                "'sequence_scale_only_log'."
            )
        if self.cfg.absolute_log_scale_weight < 0:
            raise ValueError("absolute_log_scale_weight must be non-negative.")
        if (
            self.cfg.absolute_log_scale_weight > 0
            and not self.cfg.align_to_gt_camera_scale
        ):
            raise ValueError(
                "absolute_log_scale_weight requires "
                "align_to_gt_camera_scale=true; otherwise DAV3 depth has no "
                "metric scale target."
            )
        print(
            f"Depth supervision: teacher={self.cfg.teacher}, {loss_kind}, "
            f"GT-camera-scale={self.cfg.align_to_gt_camera_scale}, "
            f"absolute-log-scale-weight={self.cfg.absolute_log_scale_weight}"
        )
        # Populated by ctx_depth_loss when the dataset provides context GT depth.
        # These detached values are diagnostics only and never enter the loss.
        self.last_teacher_gt_metrics: dict[str, torch.Tensor] = {}
        self.last_depth_components: dict[str, torch.Tensor] = {}
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
        if ctx_num > V:
            raise ValueError(
                f"Requested {ctx_num} DAV3 views, but batch only has {V}."
            )
        ctx_imgs = batch["context"]["image"][:, :ctx_num].float()

        gt_c2w = batch["context"].get("extrinsics")
        gt_intrinsics = batch["context"].get("intrinsics")
        if self.cfg.align_to_gt_camera_scale and (
            gt_c2w is None or gt_intrinsics is None
        ):
            raise ValueError(
                "GT-camera DAV3 alignment requires context extrinsics and "
                "intrinsics in the batch."
            )

        with torch.no_grad():
            # DAV3 interprets its image list as one multiview sequence. Never
            # flatten B and V here: with B=2 that would align two unrelated
            # scenes to one camera trajectory and corrupt both depth scales.
            scene_targets = []
            for batch_idx in range(B):
                scene_images = (
                    ctx_imgs[batch_idx]
                    .permute(0, 2, 3, 1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                scene_images = np.clip(
                    (scene_images + 1.0) * 127.5, 0.0, 255.0
                ).astype(np.uint8)
                scene_image_list = [
                    scene_images[view_idx] for view_idx in range(ctx_num)
                ]

                scene_w2c = None
                scene_intrinsics_px = None
                if self.cfg.align_to_gt_camera_scale:
                    scene_c2w = gt_c2w[batch_idx, :ctx_num].detach().float()
                    scene_w2c = torch.linalg.inv(scene_c2w).cpu().numpy()
                    scene_intrinsics_px_tensor = (
                        gt_intrinsics[batch_idx, :ctx_num]
                        .detach()
                        .float()
                        .clone()
                    )
                    # Dataset intrinsics are normalized by image width/height;
                    # DAV3's input processor expects pixel-space intrinsics.
                    scene_intrinsics_px_tensor[:, 0, :] *= float(W)
                    scene_intrinsics_px_tensor[:, 1, :] *= float(H)
                    scene_intrinsics_px = scene_intrinsics_px_tensor.cpu().numpy()

                inference_output = self.depth_anything.inference(
                    scene_image_list,
                    extrinsics=scene_w2c,
                    intrinsics=scene_intrinsics_px,
                    align_to_input_ext_scale=self.cfg.align_to_gt_camera_scale,
                )
                prediction = (
                    inference_output[0]
                    if isinstance(inference_output, tuple)
                    else inference_output
                )
                scene_depth = torch.from_numpy(prediction.depth).to(
                    device=depth_map.device,
                    dtype=torch.float32,
                )
                if scene_depth.shape[0] != ctx_num:
                    raise RuntimeError(
                        "DAV3 returned the wrong number of views for one "
                        f"scene: expected={ctx_num}, got={scene_depth.shape[0]}."
                    )
                scene_depth = F.interpolate(
                    scene_depth[:, None],
                    (H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                scene_targets.append(scene_depth)

        return torch.stack(scene_targets, dim=0)

    def ctx_depth_loss(
        self,
        depth_map: torch.Tensor,  # [B, V, H, W, C]
        batch,
        cxt_depth_weight: float = 0.01,
    ):
        self.last_teacher_gt_metrics = {}
        self.last_depth_components = {}
        effective_weight = float(self.cfg.weight) * float(cxt_depth_weight)

        # Do not build the DAV3 graph when this branch is disabled. In
        # particular, 0 * NaN is still NaN in the backward graph, so merely
        # multiplying the final loss by zero is not sufficient.
        if effective_weight == 0.0:
            safe_depth = torch.nan_to_num(
                depth_map.float(),
                nan=1e-4,
                posinf=100.0,
                neginf=1e-4,
            )
            return safe_depth.sum() * 0.0

        if self.cfg.alignment != "sequence_scale_only_log":
            raise RuntimeError(
                "Unsupported DAV3 depth alignment: "
                f"{self.cfg.alignment!r}."
            )

        da_output = self._context_depth_target(depth_map, batch)
        batch_size, view_count, height, width, channels = depth_map.shape
        if channels != 1:
            raise ValueError(
                "Expected depth_map with one channel, got "
                f"{channels} channels."
            )
        expected_shape = (batch_size, view_count, height, width)
        if tuple(da_output.shape) != expected_shape:
            raise ValueError(
                "DAV3 target shape does not match context depth shape: "
                f"target={tuple(da_output.shape)}, expected={expected_shape}."
            )

        # The student depth is produced by exp(raw) in the Gaussian parameter
        # head. Sanitize at the loss boundary as a second line of defense for
        # callers that do not use EncoderAnySplat's geometry path. This keeps
        # valid depths differentiable, while preventing an overflowing exp
        # value from creating NaN during backward through a masked log loss.
        prediction = torch.nan_to_num(
            depth_map.float(),
            nan=1e-4,
            posinf=100.0,
            neginf=1e-4,
        ).clamp(1e-4, 100.0).squeeze(-1)
        target = da_output.detach().to(
            device=prediction.device, dtype=prediction.dtype
        ).reshape(batch_size, view_count, height, width)
        prediction = prediction.reshape(batch_size, view_count, height, width)
        gt_depth = None
        if batch is not None and "context" in batch:
            gt_depth = batch["context"].get("depth")
        if gt_depth is not None:
            self.last_teacher_gt_metrics = self._teacher_gt_metrics(
                target.detach().float(), gt_depth.detach().float()
            )
        structure_loss, absolute_scale_loss = (
            self._sequence_scale_only_log_components(prediction, target)
        )
        absolute_weight = float(self.cfg.absolute_log_scale_weight)
        loss_local = structure_loss + absolute_weight * absolute_scale_loss
        self.last_depth_components = {
            "depth_structure": structure_loss.detach(),
            "depth_absolute_log_scale": absolute_scale_loss.detach(),
            "depth_absolute_log_scale_weighted": (
                absolute_weight * absolute_scale_loss
            ).detach(),
        }
        return effective_weight * torch.nan_to_num(loss_local, nan=0.0)

    @staticmethod
    @torch.no_grad()
    def _teacher_gt_metrics(
        teacher_depth: torch.Tensor,
        gt_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Measure DAV3 against GT without adding a second training target.

        DAV3 has an arbitrary global scale.  The aligned metrics therefore fit
        one multiplicative scale per sequence, exactly like the current DAV3
        supervision loss.  Raw metrics are retained as diagnostics only.
        """
        if teacher_depth.dim() != 4:
            raise ValueError(
                "Expected DAV3 depth with shape [B, V, H, W], got "
                f"{tuple(teacher_depth.shape)}."
            )
        if gt_depth.dim() == 5 and gt_depth.shape[2] == 1:
            gt_depth = gt_depth.squeeze(2)
        elif gt_depth.dim() == 4:
            pass
        else:
            raise ValueError(
                "Expected context GT depth with shape [B, V, H, W] or "
                f"[B, V, 1, H, W], got {tuple(gt_depth.shape)}."
            )
        if teacher_depth.shape != gt_depth.shape:
            raise ValueError(
                "DAV3 and GT depth shapes must match for diagnostics: "
                f"teacher={tuple(teacher_depth.shape)}, gt={tuple(gt_depth.shape)}."
            )

        teacher_depth = teacher_depth.float()
        gt_depth = gt_depth.to(device=teacher_depth.device, dtype=torch.float32)
        valid = (
            torch.isfinite(teacher_depth)
            & torch.isfinite(gt_depth)
            & (teacher_depth > 0)
            & (gt_depth > 0)
        )
        valid_float = valid.float()
        reduce_dims = (1, 2, 3)
        count = valid_float.sum(dim=reduce_dims).clamp_min(1.0)

        safe_teacher = torch.where(valid, teacher_depth, torch.ones_like(teacher_depth))
        safe_gt = torch.where(valid, gt_depth, torch.ones_like(gt_depth))
        log_ratio = safe_gt.clamp_min(1e-6).log() - safe_teacher.clamp_min(1e-6).log()
        log_scale = (
            (log_ratio * valid_float).sum(dim=reduce_dims) / count
        )
        # Avoid an invalid exponential if a malformed input contains very large
        # finite values.  Normal DAV3/GT depths never reach these limits.
        scale = log_scale.clamp(min=-20.0, max=20.0).exp()
        # Use sanitized tensors for arithmetic: NaN * 0 is still NaN, so a
        # later multiplication by the validity mask would not be sufficient.
        aligned_teacher = safe_teacher * scale[:, None, None, None]
        aligned_teacher = torch.where(valid, aligned_teacher, torch.zeros_like(aligned_teacher))
        safe_teacher = torch.where(valid, safe_teacher, torch.zeros_like(safe_teacher))
        safe_gt = torch.where(valid, safe_gt, torch.zeros_like(safe_gt))

        aligned_abs_error = (aligned_teacher - safe_gt).abs()
        raw_abs_error = (safe_teacher - safe_gt).abs()
        aligned_log_error = (
            aligned_teacher.clamp_min(1e-6).log()
            - safe_gt.clamp_min(1e-6).log()
        ).abs()

        denominator = torch.where(valid, safe_gt, torch.ones_like(safe_gt))
        aligned_abs_rel = aligned_abs_error / denominator
        raw_abs_rel = raw_abs_error / denominator

        def masked_mean(value: torch.Tensor) -> torch.Tensor:
            return (value * valid_float).sum() / valid_float.sum().clamp_min(1.0)

        aligned_mse = ((aligned_teacher - safe_gt) ** 2) * valid_float
        raw_mse = ((safe_teacher - safe_gt) ** 2) * valid_float
        return {
            "dav3_gt_abs_rel_aligned": masked_mean(aligned_abs_rel).detach(),
            "dav3_gt_rmse_aligned": torch.sqrt(
                aligned_mse.sum() / valid_float.sum().clamp_min(1.0)
            ).detach(),
            "dav3_gt_log_l1_aligned": masked_mean(aligned_log_error).detach(),
            "dav3_gt_scale_aligned": (scale * (valid_float.sum(dim=reduce_dims) > 0).float()).sum()
            / (valid_float.sum(dim=reduce_dims) > 0).float().sum().clamp_min(1.0),
            "dav3_gt_abs_rel_raw": masked_mean(raw_abs_rel).detach(),
            "dav3_gt_rmse_raw": torch.sqrt(
                raw_mse.sum() / valid_float.sum().clamp_min(1.0)
            ).detach(),
            "dav3_gt_valid_ratio": valid_float.mean().detach(),
        }

    @staticmethod
    def _sequence_scale_only_log_l1(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compare a whole view sequence up to one multiplicative scale.

        The scale is estimated across all views and valid pixels of each batch
        item. No per-view scale and no additive shift are fitted, so relative
        scale between views remains supervised.
        """
        if prediction.shape != target.shape or prediction.dim() != 4:
            raise ValueError(
                "Expected prediction and target with matching shape "
                "[B, V, H, W], got "
                f"prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}."
            )

        structure_loss, _ = LossDepth._sequence_scale_only_log_components(
            prediction, target
        )
        return structure_loss

    @staticmethod
    def _sequence_scale_only_log_components(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scale-free structure and absolute sequence-scale losses."""
        if prediction.shape != target.shape or prediction.dim() != 4:
            raise ValueError(
                "Expected prediction and target with matching shape "
                "[B, V, H, W], got "
                f"prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}."
            )

        valid = (
            torch.isfinite(prediction)
            & torch.isfinite(target)
            & (prediction > 0)
            & (target > 0)
        )
        safe_prediction = torch.where(valid, prediction, torch.ones_like(prediction))
        safe_target = torch.where(valid, target, torch.ones_like(target))
        log_error = safe_prediction.clamp_min(1e-6).log() - safe_target.clamp_min(1e-6).log()

        valid_float = valid.to(log_error.dtype)
        reduce_dims = (1, 2, 3)
        count = valid_float.sum(dim=reduce_dims, keepdim=True).clamp_min(1.0)
        sequence_log_scale = (
            (log_error * valid_float).sum(dim=reduce_dims, keepdim=True) / count
        )
        centered_error = log_error - sequence_log_scale
        structure_loss = (
            (centered_error.abs() * valid_float).sum()
            / valid_float.sum().clamp_min(1.0)
        )
        valid_sequences = (
            valid_float.sum(dim=reduce_dims, keepdim=True) > 0
        ).to(log_error.dtype)
        absolute_scale_loss = F.smooth_l1_loss(
            sequence_log_scale,
            torch.zeros_like(sequence_log_scale),
            reduction="none",
        )
        absolute_scale_loss = (
            (absolute_scale_loss * valid_sequences).sum()
            / valid_sequences.sum().clamp_min(1.0)
        )
        return structure_loss, absolute_scale_loss

    def ctx_depth_sequence_loss(
        self,
        depth_iters: torch.Tensor,  # [B, R, V, H, W, C]
        batch,
        cxt_depth_weight: float = 0.01,
        iter_weights: torch.Tensor | None = None,
        aux_weight: float = 0.5,
        final_weight: float = 1.0,
    ):
        if depth_iters.numel() == 0:
            return depth_iters.new_tensor(0.0)

        B, R, V, H, W, C = depth_iters.shape
        da_output = self._context_depth_target(depth_iters[:, -1], batch)
        pred_depth = depth_iters.permute(1, 0, 2, 3, 4, 5).reshape(R, B * V, H, W, C).squeeze(-1)

        losses = []
        for iter_idx in range(R):
            structure_loss, absolute_scale_loss = self._sequence_scale_only_log_components(
                pred_depth[iter_idx].reshape(B, V, H, W),
                da_output.reshape(B, V, H, W),
            )
            loss_iter = structure_loss + (
                float(self.cfg.absolute_log_scale_weight) * absolute_scale_loss
            )
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
