from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from jaxtyping import Float

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


# ====================================================================================
#  Original SSIM Logic (Helper Functions & Classes)
# ====================================================================================


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(
                out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C
            )
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )
    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    retrun_seprate: bool = False,
) -> Tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    K1, K2 = K
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)

    brightness = contrast = structure = torch.zeros_like(ssim_per_channel)
    if retrun_seprate:
        epsilon = torch.finfo(torch.float32).eps ** 2
        sigma1_sq = sigma1_sq.clamp(min=epsilon)
        sigma2_sq = sigma2_sq.clamp(min=epsilon)
        sigma12 = torch.sign(sigma12) * torch.minimum(
            torch.sqrt(sigma1_sq * sigma2_sq), torch.abs(sigma12)
        )

        C3 = C2 / 2
        sigma1_sigma2 = torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq)
        brightness_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast_map = (2 * sigma1_sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
        structure_map = (sigma12 + C3) / (sigma1_sigma2 + C3)

        contrast_map = contrast_map.clamp(max=0.98)
        structure_map = structure_map.clamp(max=0.98)

        brightness = brightness_map.flatten(2).mean(-1)
        contrast = contrast_map.flatten(2).mean(-1)
        structure = structure_map.flatten(2).mean(-1)

    return ssim_per_channel, cs, brightness, contrast, structure


def ssim_func(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
    retrun_seprate: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if not X.shape == Y.shape:
        raise ValueError(
            f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}."
        )

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(
            f"Input images should be 4-d or 5-d tensors, but got {X.shape}"
        )

    if win is not None:
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs, brightness, contrast, structure = _ssim(
        X,
        Y,
        data_range=data_range,
        win=win,
        size_average=False,
        K=K,
        retrun_seprate=retrun_seprate,
    )

    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return (
            ssim_per_channel.mean(),
            brightness.mean(),
            contrast.mean(),
            structure.mean(),
        )
    else:
        return (
            ssim_per_channel.mean(1),
            brightness.mean(1),
            contrast.mean(1),
            structure.mean(1),
        )


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = False,
    ) -> None:
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(
            [channel, 1] + [1] * spatial_dims
        )
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X: Tensor, Y: Tensor):
        # returns (ssim_score, brightness, contrast, structure)
        return ssim_func(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


# ====================================================================================
#  New Loss Implementation
# ====================================================================================


@dataclass
class LossSsimCfg:
    weight: float
    apply_after_step: int
    conf: bool = False
    alpha: bool = False
    mask: bool = False
    # SSIM specific config
    win_size: int = 11
    win_sigma: float = 1.5
    data_range: float = 1.0  # Usually 1.0 because images are normalized to [0, 1]
    channel: int = 3


@dataclass
class LossSsimCfgWrapper:
    ssim: LossSsimCfg


class LossSsim(Loss[LossSsimCfg, LossSsimCfgWrapper]):
    ssim: SSIM

    def __init__(self, cfg: LossSsimCfgWrapper) -> None:
        super().__init__(cfg)

        # Initialize the SSIM module from the provided helper class
        self.ssim = SSIM(
            data_range=self.cfg.data_range,
            size_average=True,
            win_size=self.cfg.win_size,
            win_sigma=self.cfg.win_sigma,
            channel=self.cfg.channel,
            spatial_dims=2,
            nonnegative_ssim=False,  # Standard SSIM allows negative, but usually clamped in loss
        )

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | None,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Convert [-1, 1] image to [0, 1]
        image = (batch["context"]["image"] + 1) / 2

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0.0, dtype=torch.float32, device=image.device)

        pred_color = prediction.color
        gt_image = image

        # Apply masks if configured
        if self.cfg.mask or self.cfg.alpha or self.cfg.conf:
            if self.cfg.mask:
                mask = batch["context"]["valid_mask"]
            elif self.cfg.alpha:
                mask = prediction.alpha
            elif self.cfg.conf:
                mask = depth_dict["conf_valid_mask"]
            # mask = (prediction.alpha> 0.05)
            b, v, c, h, w = pred_color.shape
            expanded_mask = mask.unsqueeze(2).expand(-1, -1, c, -1, -1)

            pred_color = pred_color * expanded_mask
            gt_image = gt_image * expanded_mask

        # Reshape for SSIM: (B * V, C, H, W)
        pred_flat = rearrange(pred_color, "b v c h w -> (b v) c h w")
        gt_flat = rearrange(gt_image, "b v c h w -> (b v) c h w")

        # SSIM module returns a tuple (ssim_val, brightness, contrast, structure)
        # We only need the ssim_val.
        ssim_val, _, _, _ = self.ssim(pred_flat, gt_flat)

        # SSIM is a similarity metric (1 is identical). We want to minimize loss.
        # Loss = 1 - SSIM
        loss = 1.0 - ssim_val

        # Return weighted loss
        return self.cfg.weight * torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
