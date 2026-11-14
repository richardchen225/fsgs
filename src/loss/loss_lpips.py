from dataclasses import dataclass
import math

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from src.model.decoder.decoder import DecoderOutput
from src.dataset.types import BatchedExample
from src.misc.nn_module_tools import convert_to_buffer
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    view_group: str = "all"
    chunk_size: int = 0
    gradient_checkpointing: bool = False
    conf: bool = False
    alpha: bool = False
    mask: bool = False
    alpha_threshold: float = 1e-4


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | None,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        image = (batch["context"]["image"] + 1) / 2

        if self.cfg.weight == 0.0:
            return torch.nan_to_num(prediction.color).sum() * 0.0

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        view_group = str(self.cfg.view_group).lower()
        if view_group not in {"source", "heldout", "all"}:
            raise ValueError(
                "LPIPS view_group must be source, heldout, or all, got "
                f"{view_group!r}."
            )
        if depth_dict is None or "depth" not in depth_dict:
            raise ValueError("LPIPS needs depth_dict['depth'] to locate source views.")
        source_views = int(depth_dict["depth"].shape[1])
        total_views = int(prediction.color.shape[1])
        if not 0 < source_views <= total_views:
            raise ValueError(
                "Invalid LPIPS source/heldout split: "
                f"source={source_views}, total={total_views}."
            )
        view_slice = {
            "source": slice(0, source_views),
            "heldout": slice(source_views, total_views),
            "all": slice(0, total_views),
        }[view_group]
        prediction_color = prediction.color[:, view_slice]
        image = image[:, view_slice]
        if prediction_color.shape[1] == 0:
            return prediction.color.sum() * 0.0

        if self.cfg.mask or self.cfg.alpha or self.cfg.conf:
            if self.cfg.mask:
                mask = batch["context"]["valid_mask"][:, view_slice]
            elif self.cfg.alpha:
                if prediction.alpha is None:
                    raise ValueError(
                        "Alpha-masked LPIPS requires rendered alpha."
                    )
                threshold = float(self.cfg.alpha_threshold)
                if not math.isfinite(threshold) or threshold < 0.0:
                    raise ValueError(
                        "LPIPS alpha_threshold must be finite and "
                        f"non-negative, got {threshold}."
                    )
                mask = prediction.alpha[:, view_slice] > threshold
            elif self.cfg.conf:
                mask = depth_dict["conf_valid_mask"][:, view_slice]
            expected_shape = (
                prediction_color.shape[0],
                prediction_color.shape[1],
                prediction_color.shape[3],
                prediction_color.shape[4],
            )
            if tuple(mask.shape) != expected_shape:
                raise ValueError(
                    "LPIPS mask shape does not match selected rendered color: "
                    f"mask={tuple(mask.shape)}, expected={expected_shape}."
                )
            mask = mask.detach().to(
                device=prediction_color.device,
                dtype=prediction_color.dtype,
            )
            b, v, c, h, w = prediction_color.shape
            expanded_mask = mask.unsqueeze(2).expand(-1, -1, c, -1, -1)
            masked_pred = prediction_color * expanded_mask
            masked_img = image * expanded_mask
            predicted_flat = rearrange(masked_pred, "b v c h w -> (b v) c h w")
            image_flat = rearrange(masked_img, "b v c h w -> (b v) c h w")
        else:
            predicted_flat = rearrange(
                prediction_color, "b v c h w -> (b v) c h w"
            )
            image_flat = rearrange(image, "b v c h w -> (b v) c h w")

        chunk_size = int(self.cfg.chunk_size)
        if chunk_size <= 0:
            chunk_size = predicted_flat.shape[0]

        def evaluate(predicted: Tensor, target: Tensor) -> Tensor:
            return self.lpips.forward(predicted, target, normalize=True)

        loss_sum = predicted_flat.new_zeros(())
        for start in range(0, predicted_flat.shape[0], chunk_size):
            stop = min(start + chunk_size, predicted_flat.shape[0])
            predicted_chunk = predicted_flat[start:stop]
            image_chunk = image_flat[start:stop]
            if self.cfg.gradient_checkpointing and predicted_chunk.requires_grad:
                values = checkpoint(
                    evaluate,
                    predicted_chunk,
                    image_chunk,
                    use_reentrant=False,
                )
            else:
                values = evaluate(predicted_chunk, image_chunk)
            loss_sum = loss_sum + values.sum()
        loss = loss_sum / predicted_flat.shape[0]

        return self.cfg.weight * torch.nan_to_num(
            loss, nan=0.0, posinf=0.0, neginf=0.0
        )
