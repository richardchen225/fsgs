from dataclasses import dataclass
from pathlib import Path
import gc
import random
import sys
from typing import Literal, Optional, Protocol, runtime_checkable, Any
import json
import torch
import math
import torchvision
import wandb
from PIL import Image
import cv2
import numpy as np
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from tabulate import tabulate
from torch import Tensor, nn, optim
import torch.nn.functional as F
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
from loss.loss_lpips import LossLpips
from loss.loss_mse import LossMse
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import (
    compute_lpips,
    compute_psnr,
    compute_ssim,
)
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import (
    inverse_normalize,
    vis_depth_map,
    confidence_map,
    get_overlap_tag,
)
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat

from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .encoder.vggt.utils.rotation import quat_to_mat
from .ply_export import export_ply
from lightning.pytorch.loggers.wandb import WandbLogger

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float
    pretrained_lr: float | None = None
    scratch_lr: float | None = None


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    generate_video: bool
    mode: Literal["inference", "evaluation"]
    image_folder: str


@dataclass
class TrainCfg:
    output_path: Path
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    max_val_comparisons: int = 16
    pose_loss_alpha: float = 1.0
    pose_loss_delta: float = 1.0
    cxt_depth_weight: float = 0.01
    weight_pose: float = 1.0
    weight_depth: float = 1.0
    weight_normal: float = 1.0
    render_ba: bool = False
    render_ba_after_step: int = 0


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model: nn.Module
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        model: nn.Module,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder_visualizer = None
        self.model = model
        self.data_shim = get_data_shim(self.model.encoder)
        self.losses = nn.ModuleList(losses)

        self.benchmarker = Benchmarker()
        self._skip_optimizer_step = False
        self._bad_grad_name = None
        self._val_comparison_images: list[np.ndarray] = []
        self._val_comparison_captions: list[str] = []
    def on_train_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        print(f"Train epoch start on rank {self.trainer.global_rank}")
        if hasattr(self.trainer.datamodule.train_loader.dataset, "set_epoch"):
            self.trainer.datamodule.train_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.train_loader.sampler, "set_epoch"):
            self.trainer.datamodule.train_loader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        print(f"Validation epoch start on rank {self.trainer.global_rank}")
        self._val_comparison_images = []
        self._val_comparison_captions = []
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.datamodule.val_loader.dataset, "set_epoch"):
            self.trainer.datamodule.val_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.val_loader.sampler, "set_epoch"):
            self.trainer.datamodule.val_loader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        if self.global_rank != 0 or len(self._val_comparison_images) == 0:
            return

        self.logger.log_image(
            "comparison",
            self._val_comparison_images,
            step=self.global_step,
            caption=self._val_comparison_captions,
        )

    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        # if self.global_rank == 0:
        #     print(
        #         f"context = {batch['context']['index'].tolist()}"
        #     )
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat(
                                    [batch_combined[k][kk], batch_per_dl[k][kk]], dim=0
                                )
                        else:
                            raise NotImplementedError
            batch = batch_combined

        batch: BatchedExample = self.data_shim(batch)
        b, v, c, h, w = batch["context"]["image"].shape
        context_image = (batch["context"]["image"] + 1) / 2
        if (
            self.global_rank == 0
            and self.train_cfg.print_log_every_n_steps > 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"training step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Run the model
#         if self.global_step < 50:
#             loss_align = self.model.encoder(context_image, batch["context"]["index"], self.global_step)
#             total_loss = 0
#             total_loss = total_loss + torch.nan_to_num(
#                     loss_align, nan=0.0, posinf=0.0, neginf=0.0
#                 ) * 1e-2
#             print(f'total_loss:{total_loss}')
#             self.log("loss/total", total_loss.item())
#             self.log("info/global_step", self.global_step)
#             # print(f"total_loss: {total_loss}")
#             # print(f"scene = {[x[:20] for x in batch['scene']]}; " )

#             # Tell the data loader processes about the current step.
#             if self.step_tracker is not None:
#                 self.step_tracker.set_step(self.global_step)

#             del batch
#             if self.global_step % 50 == 0:
#                 gc.collect()
#                 torch.cuda.empty_cache()

#             return total_loss
        encoder_output, output = self.model(context_image, batch["context"]["index"], self.global_step)
        gaussians, pred_pose_enc_list, depth_dict = (
            encoder_output.gaussians,
            encoder_output.pred_pose_enc_list,
            encoder_output.depth_dict,
        )
        
        distill_infos = encoder_output.distill_infos
        if (
            encoder_output.infos is not None
            and "gs_refine_history_views" in encoder_output.infos
        ):
            self.log(
                "train/gs_refine_history_views",
                encoder_output.infos["gs_refine_history_views"].float(),
            )
        
        target_gt = (batch["context"]["image"] + 1) / 2
        num_context_views = target_gt.shape[1]

        using_index = torch.arange(num_context_views, device=gaussians.means.device)
        batch["using_index"] = using_index

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean().item())

        total_loss = 0

        with torch.amp.autocast("cuda", enabled=False):
            depth_loss_idx = list(get_cfg()["loss"].keys()).index("depth")
            depth_loss_module = self.losses[depth_loss_idx]
            loss_depth_ctx = depth_loss_module.ctx_depth_loss(
                depth_dict["depth"],
                batch,
                cxt_depth_weight=self.train_cfg.cxt_depth_weight,
            )

            self.log(
                "loss/loss_depth_ctx",
                loss_depth_ctx.item())
            total_loss = total_loss + loss_depth_ctx
            
            # depth_loss_idx = list(get_cfg()["loss"].keys()).index("depth")
            # depth_loss_fn = self.losses[depth_loss_idx].ctx_depth_loss
            # loss_depth_ctx = depth_loss_fn(
            #     depth_dict["depth"],
            #     batch,
            #     cxt_depth_weight=self.train_cfg.cxt_depth_weight,
            # )
            # # print(f'loss_depth_ctx :{loss_depth_ctx}')
            # self.log("loss/loss_depth_ctx", loss_depth_ctx.item())
            # total_loss = total_loss + loss_depth_ctx

            for loss_fn in self.losses:
                if loss_fn.name == "depth":
                    break
                loss = loss_fn.forward(output, batch, gaussians, depth_dict, self.global_step)
                self.log(f"loss/{loss_fn.name}", loss.item())
                total_loss = total_loss + loss

            # int loss
            # loss_ca1 = F.mse_loss(
            #     pred_pose_enc_list, distill_infos["pred_pose_enc_list"][:, 0:1, -1]
            # )
            # loss_ca = 10 * loss_ca1
            # self.log("loss/loss_ca", loss_ca.item())
            # total_loss = total_loss + loss_ca
            # print(f'loss_sparsity:{loss_sparsity}')
            # self.log("loss/loss_new", loss_sparsity.item())
            # total_loss = total_loss + loss_sparsity
            # gt ex loss
#             gt_pose = batch["context"]["extrinsics"]
#             gt_t = gt_pose[:, :, :3, 3].float()  # BxSx3
#             gt_R_c2w = gt_pose[:, :, :3, :3].float()  # BxSx3x3
            
#             weights=[0.25, 0.5, 0.75, 1.0]
   
#             delta_t_gt = gt_t - gt_t[:, 0:1, :]

#             for i, pred_9d in enumerate(pred_list):
#                 # 1. 拆分 W2C 定义的 9 维向量
#                 t_w2c = pred_9d[..., :3]     # [B, S, 3]
#                 q_w2c = pred_9d[..., 3:7]    # [B, S, 4]

#                 # 2. 🌟 核心转换：W2C -> C2W 🌟
#                 # 获取 W2C 旋转矩阵
#                 R_w2c = quat_to_mat(q_w2c) # [B, S, 3, 3]

#                 # 相机在世界坐标系的旋转 R_c2w = R_w2c^T
#                 R_c2w_pred = R_w2c.transpose(-1, -2)

#                 # 相机在世界坐标系的位置 t_c2w = -R_w2c^T @ t_w2c
#                 # 这一步算出的才是相机真正的运动轨迹！
#                 t_c2w_pred = -torch.matmul(R_c2w_pred, t_w2c.unsqueeze(-1)).squeeze(-1) # [B, S, 3]

#                 # 3. 🌟 尺度不变对齐 (Procrustes Alignment) 🌟
#                 # 只在 C2W 轨迹空间算相对位移
#                 delta_t_pred = t_c2w_pred - t_c2w_pred[:, 0:1, :]

#                 # 计算最优缩放因子 alpha
#                 num = torch.sum(delta_t_pred * delta_t_gt, dim=(1, 2))
#                 den = torch.sum(delta_t_pred * delta_t_pred, dim=(1, 2)) + 1e-6
#                 alpha = (num / den).view(-1, 1, 1)

#                 # 平移 Loss (尺度对齐后)
#                 loss_t = F.mse_loss(alpha * delta_t_pred, delta_t_gt)

#                 # 4. 旋转 Loss (直接在 C2W 矩阵空间对比)
#                 loss_R = F.mse_loss(R_c2w_pred, gt_R_c2w)

#                 # 6. 组合阶段 Loss
#                 stage_loss = 10.0 * loss_t + 10.0 * loss_R
#                 total_loss += weights[i] * stage_loss

        self.log("loss/total", total_loss.item())
        self.log("info/global_step", self.global_step)
        # print(f"total_loss: {total_loss}")
        # print(f"scene = {[x[:20] for x in batch['scene']]}; " )
        
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        del batch
        if self.global_step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        
        return total_loss

    def on_after_backward(self):
        # Keep DDP's strict unused-parameter check enabled. If a future graph
        # change disconnects a trainable branch, print exact names before DDP
        # raises on the next forward pass.
        if self.global_step > 1:
            return
        unused = [
            name
            for name, param in self.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        if unused:
            details = (
                f"[DDP unused parameters][rank={self.global_rank}]\n"
                + "\n".join(unused)
            )
            print(details, file=sys.stderr, flush=True)
            raise RuntimeError(
                "Trainable parameters are disconnected from the training loss; "
                "see the parameter list printed immediately above."
            )

    def on_before_optimizer_step(self, optimizer):
        # 这个 hook 里检查的是“真正要 step 的梯度”
        self._skip_optimizer_step = False
        self._bad_grad_name = None

        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                self._skip_optimizer_step = True
                self._bad_grad_name = name
                break

        if self._skip_optimizer_step:
            if self.global_rank == 0:
                print(
                    f"[skip step] global_step={self.global_step}, "
                    f"bad_param={self._bad_grad_name}"
                )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # 如果这一轮梯度坏了：不更新参数，只清梯度
        if self._skip_optimizer_step:
            optimizer.zero_grad(set_to_none=True)
            self._skip_optimizer_step = False
            self._bad_grad_name = None
            return

        # 正常情况才 step
        optimizer.step(closure=optimizer_closure)
        self._skip_optimizer_step = False
        self._bad_grad_name = None
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # if self.global_step < 20:
        #     return
        batch: BatchedExample = self.data_shim(batch)
        total_batches = len(self.trainer.datamodule.val_loader)
        if self.global_rank == 0:
            print(f"Rank {self.global_rank}, batch {batch_idx+1}/{total_batches}")
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        encoder_output, output = self.model(
            (batch["context"]["image"] + 1) / 2,
            batch["context"]["index"],
            self.global_step,
        )

        # Compute validation metrics over every sample/view in the batch.
        rgb_pred = rearrange(output.color.float(), "b v c h w -> (b v) c h w")
        rgb_gt = rearrange((batch["context"]["image"].float() + 1) / 2, "b v c h w -> (b v) c h w")
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr, sync_dist=True)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips, sync_dist=True)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim, sync_dist=True)

        if self.global_rank != 0:
            return

        depth_dict = encoder_output.depth_dict
        scenes = batch["scene"]
        for i in range(len(scenes)):
            if len(self._val_comparison_images) >= self.train_cfg.max_val_comparisons:
                break

            context_img = inverse_normalize(batch["context"]["image"][i])
            context = [context_img[j] for j in range(context_img.shape[0])]

            model_depth_pred = depth_dict["depth"].squeeze(-1)[i]
            model_depth_pred = vis_depth_map(model_depth_pred)

            depth_pred = vis_depth_map(output.depth[i])
            rgb_gt_i = (batch["context"]["image"][i].float() + 1) / 2
            rgb_pred_i = output.color[i].float()

            comparison = hcat(
                add_label(vcat(*context), "Context"),
                add_label(vcat(*rgb_gt_i), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred_i), "Rendered Target"),
                add_label(vcat(*model_depth_pred), "GS Depth"),
                add_label(vcat(*depth_pred), "Rendered Depth"),
            )

            comparison = torch.nn.functional.interpolate(
                comparison.unsqueeze(0),
                scale_factor=0.5,
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

            self._val_comparison_images.append(prep_image(add_border(comparison)))
            self._val_comparison_captions.append(
                f"step={self.global_step}, batch={batch_idx}, index={i}, scene={scenes[i]}"
            )

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")
        
        target_image = batch["target"]["image"]
        target_view_count = target_image.shape[1]
        
        with torch.no_grad():
            with self.benchmarker.time("encoder"):
                (
                    encoder_output,
                    pred_all_extrinsic,
                    intrinsic, 
                    depth_map, 
                    ctx_img_num,
                    # single_opacities
                    # primitive_mask_hard,
                    # sampling_vis
                ) = self.model.encoder(
                    (batch["context"]["image"] + 1) / 2,
                    batch["context"]["index"],
                    global_step=self.global_step, 
                    # decoder = self.model.decoder, 
                    name = batch['scene'],
                    target_view_count=target_view_count,
                    return_refine_data=True,
                    # global_rank = self.global_rank
                )
                gaussians = self.model._refine_gaussians(
                    encoder_output,
                    (batch["context"]["image"] + 1) / 2,
                    pred_all_extrinsic,
                    encoder_output.pred_context_pose,
                    ctx_img_num,
                    0.01,
                    100.0,
                )
                if (
                    encoder_output.infos is not None
                    and "gs_refine_history_views" in encoder_output.infos
                ):
                    final_history = int(
                        encoder_output.infos["gs_refine_history_views"].item()
                    )
                    if final_history != ctx_img_num - 1:
                        raise RuntimeError(
                            "GS refiner test rollout did not consume all causal context views: "
                            f"history={final_history}, expected={ctx_img_num - 1}."
                        )
        # if self.global_rank == 0:
        # export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0].permute(0,2,1), single_opacities, Path(f"gaussians_{[x[:20] for x in batch['scene']]}.ply"))
        
#         import re    
#         prediction_path = '/openbayes/home/AnySplat'
#         img_folder = os.path.join(prediction_path, 'output_frames') 
#         video_save_path = os.path.join(prediction_path, 'output_video.mp4') 
#         fps = 30 
#         if self.global_rank == 0:
#             images = [img for img in os.listdir(img_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
#             if len(images) > 0:
#                 images.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else x)
#                 first_img_path = os.path.join(img_folder, images[0])
#                 first_img = cv2.imread(first_img_path)
#                 H, W, _ = first_img.shape

#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (W, H))

#                 print(f"开始将 {len(images)} 张图片合成视频...")

#                 for img_name in images:
#                     img_path = os.path.join(img_folder, img_name)
#                     frame = cv2.imread(img_path)

#                     if frame is None:
#                         print(f"警告：无法读取图片 {img_name}，跳过。")
#                         continue

#                     if (frame.shape[1], frame.shape[0]) != (W, H):
#                         frame = cv2.resize(frame, (W, H))

#                     video_writer.write(frame)

#                 video_writer.release()
#                 print(f"视频已成功保存至: {video_save_path}")
#             else:
#                 print(f"错误：文件夹 {img_folder} 中没有找到图片。")
                
        def save_depth_and_presence_for_eval(
            pred_depth,              # [b, v, h, w, 1]
            primitive_mask_hard,     # [b*v*h*w, 1]
            out_dir,
            batch_idx=0,
        ):
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            assert pred_depth.ndim == 5, pred_depth.shape
            assert pred_depth.shape[-1] == 1, pred_depth.shape

            b, v, h, w, _ = pred_depth.shape

            assert primitive_mask_hard.shape == (b * v * h * w, 1), (
                primitive_mask_hard.shape,
                (b * v * h * w, 1),
            )

            # depth: [b, v, h, w, 1] -> [b, v, h, w]
            depth_np = pred_depth.detach().float().cpu().numpy()[..., 0]

            # mask: [b*v*h*w, 1] -> [b, v, h, w]
            mask_np = (
                primitive_mask_hard
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(b, v, h, w)
            )
            mask_int = mask_np.astype("uint8")

            total_ones = int(mask_int.sum())
            total_pixels = int(mask_int.size)
            total_zeros = total_pixels - total_ones

            print(f"[MASK DEBUG] total pixels: {total_pixels}")
            print(f"[MASK DEBUG] total ones : {total_ones}")
            print(f"[MASK DEBUG] total zeros: {total_zeros}")
            print(f"[MASK DEBUG] keep ratio  : {total_ones / max(total_pixels, 1):.6f}")
            for view_id in range(v):
                depth = depth_np[batch_idx, view_id].astype("float32")  # [h, w]
                mask = mask_np[batch_idx, view_id].astype("uint8") * 255

                depth_path = out_dir / f"pred_depth_{view_id:03d}.exr"
                mask_path = out_dir / f"existence_{view_id:03d}.png"

                ok = cv2.imwrite(str(depth_path), depth)
                if not ok:
                    raise RuntimeError(f"Failed to save depth: {depth_path}")

                Image.fromarray(mask).save(mask_path)

            print(f"[DONE] saved {v} views to {out_dir}")
            
        # save_depth_and_presence_for_eval(
        #     pred_depth=depth_map,                  # [b, v, h, w, 1]
        #     primitive_mask_hard=primitive_mask_hard, # [b*v*h*w, 1]
        #     out_dir=f"/openbayes/home/AnySplat/{batch['scene']}",
        #     batch_idx=0,
        # )
        from .utils import make_sampling_heatmap_overlay_tensors
        num_context_view = ctx_img_num
        pred_all_target_extrinsic = pred_all_extrinsic[:, ctx_img_num:]
        render_view_count = pred_all_target_extrinsic.shape[1]
        render_device = gaussians.means.device
                
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.model.decoder.forward(
                gaussians,
                pred_all_target_extrinsic,
                intrinsic[:,0:1,:,:].repeat(1, render_view_count, 1, 1).float(),
                # intrinsic.float(),
                torch.ones(1, render_view_count, device=render_device) * 0.01,
                torch.ones(1, render_view_count, device=render_device) * 100,
                (h, w),
            )

        # depth_pred = vis_depth_map(output.depth[0])
        # model_depth_pred = depth_map.squeeze(-1)[0]
        # model_depth_pred = vis_depth_map(model_depth_pred)
        
        psnr = None
        with torch.no_grad():
            if self.test_cfg.compute_scores:
                rgb_pred = output.color[0]
                rgb_gt = target_image[0]
                psnr = compute_psnr(rgb_gt, rgb_pred).mean().item()
                all_metrics = {
                    f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean().item(),
                    f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean().item(),
                    f"psnr_ours": psnr,
                }
                methods = ["ours"]
                self.log_dict(all_metrics, prog_bar=True, sync_dist=True, on_epoch=True)
                self.print_preview_metrics(all_metrics, methods)

        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        
        # for i in range(len(batch["target"]["index"])):
        #     idx = batch["target"]["index"][i].item()  
        #     single_color = output.color[0][i] 
        #     res_path = path / f"{psnr}_{scene}" / "color" / f"{int(idx):0>6}.png"
        #     save_image(single_color, res_path)

        if self.test_cfg.save_compare:
            context_img = inverse_normalize(batch["context"]["image"][0])
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Rendered Target"),
                add_label(vcat(*model_depth_pred), "GS Depth"),
                add_label(vcat(*depth_pred), "Rendered Depth"),  
            )
            save_image(comparison, path / f"{psnr}_{scene}.png")
        
#         if self.test_cfg.save_compare:
#             context_img = inverse_normalize(batch["context"]["image"][0][:num_context_view])  # [Vc, 3, H, W]

#             compare_items = [
#                 add_label(vcat(*context_img), "Context"),
#                 add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
#                 add_label(vcat(*rgb_pred), "Rendered Target"),
#                 add_label(vcat(*model_depth_pred), "GS Depth"),
#                 add_label(vcat(*depth_pred), "Rendered Depth"),
#             ]

#             # --------------------------------------------------
#             # Sampling heatmap visualization
#             # --------------------------------------------------
#             if sampling_vis is not None:
#                 u_v_pixel = sampling_vis["u_v_pixel"]
#                 primitive_mask_hard = sampling_vis.get("primitive_mask_hard", None)

#                 b_vis = sampling_vis.get("b", batch["context"]["image"].shape[0])
#                 v_vis = sampling_vis.get("v", batch["context"]["image"].shape[1])

#                 # 注意：这里用 context_img，因为 sampling 是发生在输入/context views 上的
#                 # context_img: [Vc, 3, H, W] -> [1, Vc, 3, H, W]
#                 sampling_heatmaps = make_sampling_heatmap_overlay_tensors(
#                     image=context_img.unsqueeze(0),
#                     u_v_pixel=u_v_pixel,
#                     b=b_vis,
#                     v=v_vis,
#                     primitive_mask_hard=primitive_mask_hard,
#                     batch_idx=0,
#                     alpha=0.55,
#                     blur_ksize=21,
#                     blur_sigma=7.0,
#                 )

#                 overlay_all = sampling_heatmaps["overlay_all"]      # [Vc, 3, H, W]
#                 overlay_kept = sampling_heatmaps["overlay_kept"]    # [Vc, 3, H, W] or None
#                 heat_all = sampling_heatmaps["heat_all"]            # [Vc, 3, H, W]
#                 heat_kept = sampling_heatmaps["heat_kept"]          # [Vc, 3, H, W] or None

#                 compare_items.append(
#                     add_label(vcat(*overlay_all), "Sampled Positions")
#                 )

#                 compare_items.append(
#                     add_label(vcat(*heat_all), "Sample Heatmap")
#                 )

#                 if overlay_kept is not None:
#                     compare_items.append(
#                         add_label(vcat(*overlay_kept), "Kept After Mask")
#                     )

#                 if heat_kept is not None:
#                     compare_items.append(
#                         add_label(vcat(*heat_kept), "Kept Heatmap")
#                     )

#             comparison = hcat(*compare_items)
#             save_image(comparison, path / f"{psnr}_{scene}.png")

    def on_test_end(self) -> None:
        self.benchmarker.summarize()

    def print_preview_metrics(
        self,
        metrics: dict[str, float | Tensor],
        methods: list[str] | None = None,
        overlap_tag: str | None = None,
    ) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {
                    k: ((s * v) + metrics[k]) / (s + 1)
                    for k, v in self.running_metrics_sub[overlap_tag].items()
                }
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ["ours"]

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        
        
#     def configure_optimizers(self):

#             # 1. 参数分组
#             backbone_params = []
#             head_params = []

#             for name, param in self.named_parameters():
#                 if not param.requires_grad:
#                     continue
#                 # 根据你的模块命名习惯区分
#                 if "gs_head" in name or "output_conv2" in name or "input_merger" in name:
#                     head_params.append(param)
#                 else:
#                     backbone_params.append(param)

#             # 初始学习率设置
#             param_dicts = [
#                 {"params": backbone_params, "lr": 1e-4, "name": "backbone"},
#                 {"params": head_params, "lr": 2e-4, "name": "head"}
#             ]

#             optimizer = torch.optim.AdamW(
#                 param_dicts, weight_decay=0.1, betas=(0.9, 0.95)
#             )

#             # 2. 🌟 绝对解耦的学习率逻辑 🌟
#             phase_1_steps = 50 
#             phase_2_steps = 10000 
#             eta_min_ratio = 0.1

#             # Backbone 逻辑：前 1w 步 Cosine 衰减，后 1w 步彻底消失 (0.0)
#             def lr_lambda_backbone(current_step):
#                 if current_step < phase_1_steps:
#                     # 0 -> 10k: 从 1.0 衰减到 0.1
#                     progress = float(current_step) / float(phase_1_steps)
#                     return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))
#                 else:
#                     # 🌟 10k 之后彻底关闭 Backbone 训练
#                     return 0.0

#             # Head 逻辑：前 1w 步彻底冻结 (0.0)，后 1w 步重新开启 Cosine 衰减
#             def lr_lambda_head(current_step):
#                 if current_step < phase_1_steps:
#                     # 🌟 0 -> 10k: 彻底不练 Head
#                     return 0.0
#                 else:
#                     # 10k -> 20k: 重新算进度 (从 0.0 衰减到 0.1)
#                     phase_2_progress = float(current_step - phase_1_steps) / float(phase_2_steps)
#                     phase_2_progress = min(1.0, phase_2_progress)
#                     return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * phase_2_progress))

#             lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#                 optimizer,
#                 lr_lambda=[lr_lambda_backbone, lr_lambda_head] 
#             )

#             return {
#                 "optimizer": optimizer,
#                 "lr_scheduler": {
#                     "scheduler": lr_scheduler,
#                     "interval": "step",
#                     "frequency": 1,
#                 },
#             }
    def configure_optimizers(self):
        pretrained_lr = (
            self.optimizer_cfg.pretrained_lr
            if self.optimizer_cfg.pretrained_lr is not None
            else self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier
        )
        scratch_lr = (
            self.optimizer_cfg.scratch_lr
            if self.optimizer_cfg.scratch_lr is not None
            else self.optimizer_cfg.lr
        )

        pretrained_params, scratch_params, other_params = [], [], []
        pretrained_names, scratch_names, other_names = [], [], []

        pretrained_keys = (
            "model.encoder.gaussian_param_head",
            "model.encoder.gs_head",
        )
        scratch_keys = (
            "model.encoder.depth_refiner",
            "model.gs_residual_refiner",
        )

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in pretrained_keys):
                pretrained_params.append(param)
                pretrained_names.append(name)
            elif any(key in name for key in scratch_keys):
                scratch_params.append(param)
                scratch_names.append(name)
            else:
                other_params.append(param)
                other_names.append(name)

        param_dicts = []
        if pretrained_params:
            param_dicts.append(
                {
                    "params": pretrained_params,
                    "lr": pretrained_lr,
                    "name": "pretrained",
                }
            )
        if scratch_params:
            param_dicts.append(
                {
                    "params": scratch_params,
                    "lr": scratch_lr,
                    "name": "scratch",
                }
            )
        if other_params:
            param_dicts.append(
                {
                    "params": other_params,
                    "lr": self.optimizer_cfg.lr,
                    "name": "other",
                }
            )

        if self.global_rank == 0:
            print(
                "Optimizer parameter groups: "
                f"pretrained={len(pretrained_names)} params @ {pretrained_lr}, "
                f"scratch={len(scratch_names)} params @ {scratch_lr}, "
                f"other={len(other_names)} params @ {self.optimizer_cfg.lr}"
            )
        
        optimizer = torch.optim.AdamW(
            param_dicts, weight_decay=0.1, betas=(0.9, 0.95)
        )
        
        max_steps = get_cfg()["trainer"]["max_steps"]
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        eta_min_ratio = 0.1 

        def lr_lambda_main(current_step):
            # 1. Warmup 阶段
            if warm_up_steps > 0 and current_step < warm_up_steps:
                return float(current_step) / float(max(1, warm_up_steps))
            
            # 2. Cosine 衰减阶段
            decay_steps = max_steps - warm_up_steps
            if decay_steps <= 0:
                return 1.0 # 防止总步数小于预热步数的异常情况
                
            current_decay_step = current_step - warm_up_steps
            progress = float(current_decay_step) / float(decay_steps)
            progress = min(1.0, max(0.0, progress)) # 确保在 0~1 之间
            
            return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lr_lambda_main] * len(param_dicts)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

#     def configure_optimizers(self):
#         new_params, new_param_names = [], []
#         delayed_params, delayed_param_names = [], [] 
        
#         for name, param in self.named_parameters():
#             if not param.requires_grad:
#                 continue

#             if "gaussian_param_head.scratch.output_conv2" in name:
#                 new_params.append(param)
#                 new_param_names.append(name)
#             elif "gaussian_param_head" in name:
#                 delayed_params.append(param)
#                 delayed_param_names.append(name)
#             else:
#                 new_params.append(param)
#                 new_param_names.append(name)
                
#         param_dicts = [
#             {
#                 "params": new_params,
#                 "lr": self.optimizer_cfg.lr, 
#                 "name": "new_params"
#             },
#             {
#                 "params": delayed_params,
#                 "lr": self.optimizer_cfg.lr,  
#                 "name": "delayed_params"
#             },
#         ]
        
#         optimizer = torch.optim.AdamW(
#             param_dicts, weight_decay=0.1, betas=(0.9, 0.95)
#         )
        
#         max_steps = get_cfg()["trainer"]["max_steps"]
#         warm_up_steps = self.optimizer_cfg.warm_up_steps
#         delay_steps = 5000  
#         eta_min_ratio = 0.1 

#         def lr_lambda_main(current_step):
#             # 1. Warmup 阶段
#             if warm_up_steps > 0 and current_step < warm_up_steps:
#                 return float(current_step) / float(max(1, warm_up_steps))
            
#             # 2. Cosine 衰减阶段
#             decay_steps = max_steps - warm_up_steps
#             if decay_steps <= 0:
#                 return 1.0 # 防止总步数小于预热步数的异常情况
                
#             current_decay_step = current_step - warm_up_steps
#             progress = float(current_decay_step) / float(decay_steps)
#             progress = min(1.0, max(0.0, progress)) # 确保在 0~1 之间
            
#             # 余弦退火公式
#             return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

#         # ⚠️ 关键点 3：定义延迟参数的学习率曲线
#         def lr_lambda_delayed(current_step):
#             if current_step < delay_steps:
#                 return 0.0  # 前 5000 步，强制乘子为 0，学习率绝对为 0
#             else:
#                 # 5000 步之后，直接调用主参数的函数！
#                 # 这样就能保证苏醒时的 LR 和衰减轨迹与主参数一模一样
#                 return lr_lambda_main(current_step)

#         # ⚠️ 关键点 4：将两个规则分别应用到两个参数组
#         lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#             optimizer,
#             lr_lambda=[lr_lambda_main, lr_lambda_delayed] # 顺序必须和 param_dicts 一一对应
#         )

#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": lr_scheduler,
#                 "interval": "step",
#                 "frequency": 1,
#             },
#         }
