import sys
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _as_bchw_mask(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 3:
        value = value[:, None]
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(
            f"{name} must be [B,H,W] or [B,1,H,W], "
            f"got {tuple(value.shape)}."
        )
    return value


def _as_bhwc_flow(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(f"{name} must be rank 4, got {tuple(value.shape)}.")
    if value.shape[-1] == 2:
        return value.float()
    if value.shape[1] == 2:
        return value.permute(0, 2, 3, 1).float()
    raise ValueError(
        f"{name} must be [B,H,W,2] or [B,2,H,W], "
        f"got {tuple(value.shape)}."
    )


def forward_backward_tracker_consistency(
    forward_flow: torch.Tensor,
    forward_visibility: torch.Tensor,
    backward_flow: torch.Tensor,
    backward_visibility: torch.Tensor,
    *,
    visibility_threshold: float = 0.8,
    absolute_threshold: float = 1.5,
    relative_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source-pixel cycle validity and forward-backward error."""
    forward = _as_bhwc_flow(forward_flow, "forward_flow")
    backward = _as_bhwc_flow(backward_flow, "backward_flow")
    if forward.shape != backward.shape:
        raise ValueError(
            "Forward and backward flow shapes must match, got "
            f"{tuple(forward.shape)} and {tuple(backward.shape)}."
        )
    batch, height, width, _ = forward.shape
    expected = (batch, 1, height, width)
    forward_visibility = _as_bchw_mask(
        forward_visibility, "forward_visibility"
    ).float()
    backward_visibility = _as_bchw_mask(
        backward_visibility, "backward_visibility"
    ).float()
    if (
        forward_visibility.shape != expected
        or backward_visibility.shape != expected
    ):
        raise ValueError(
            "Flow and visibility spatial shapes must match; got "
            f"flow={tuple(forward.shape)}, "
            f"forward_visibility={tuple(forward_visibility.shape)}, "
            f"backward_visibility={tuple(backward_visibility.shape)}."
        )
    visibility_threshold = float(visibility_threshold)
    absolute_threshold = float(absolute_threshold)
    relative_threshold = float(relative_threshold)
    if not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError("visibility_threshold must be in [0,1].")
    if not math.isfinite(absolute_threshold) or absolute_threshold < 0.0:
        raise ValueError("absolute_threshold must be finite and non-negative.")
    if not math.isfinite(relative_threshold) or relative_threshold < 0.0:
        raise ValueError("relative_threshold must be finite and non-negative.")

    y, x = torch.meshgrid(
        torch.arange(height, device=forward.device, dtype=torch.float32),
        torch.arange(width, device=forward.device, dtype=torch.float32),
        indexing="ij",
    )
    target_x = x[None] + forward[..., 0]
    target_y = y[None] + forward[..., 1]
    finite_target = torch.isfinite(target_x) & torch.isfinite(target_y)
    inside = (
        finite_target
        & (target_x >= 0.0)
        & (target_x <= float(width - 1))
        & (target_y >= 0.0)
        & (target_y <= float(height - 1))
    )
    safe_x = torch.where(finite_target, target_x, torch.zeros_like(target_x))
    safe_y = torch.where(finite_target, target_y, torch.zeros_like(target_y))
    grid_x = (
        2.0 * safe_x / float(width - 1) - 1.0
        if width > 1
        else torch.zeros_like(safe_x)
    )
    grid_y = (
        2.0 * safe_y / float(height - 1) - 1.0
        if height > 1
        else torch.zeros_like(safe_y)
    )
    sample_grid = torch.stack([grid_x, grid_y], dim=-1)
    sampled_backward = F.grid_sample(
        backward.permute(0, 3, 1, 2),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).permute(0, 2, 3, 1)
    sampled_backward_visibility = F.grid_sample(
        backward_visibility,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]

    cycle = forward + sampled_backward
    cycle_error = cycle.norm(dim=-1)
    motion_scale = (
        forward.norm(dim=-1) + sampled_backward.norm(dim=-1)
    )
    finite_cycle = torch.isfinite(cycle_error) & torch.isfinite(motion_scale)
    threshold = absolute_threshold + relative_threshold * motion_scale
    valid = (
        inside
        & finite_cycle
        & (forward_visibility[:, 0] >= visibility_threshold)
        & (sampled_backward_visibility >= visibility_threshold)
        & (cycle_error <= threshold)
    )
    safe_error = torch.where(finite_cycle, cycle_error, torch.zeros_like(cycle_error))
    return valid[:, None], safe_error[:, None]


def forward_splat_tracker_coverage(
    flow: torch.Tensor,
    visibility: torch.Tensor,
    source_support: torch.Tensor,
    *,
    visibility_threshold: float = 0.8,
    coverage_threshold: float = 0.05,
    dilation_radius: int = 1,
) -> torch.Tensor:
    """Forward-splat supported source pixels into a boolean target coverage map."""
    flow_xy = _as_bhwc_flow(flow, "flow")

    visibility = _as_bchw_mask(visibility, "visibility").float()
    source_support = _as_bchw_mask(source_support, "source_support").bool()
    batch, height, width, _ = flow_xy.shape
    expected = (batch, 1, height, width)
    if visibility.shape != expected or source_support.shape != expected:
        raise ValueError(
            "flow, visibility, and source_support spatial shapes must match; "
            f"got flow={tuple(flow_xy.shape)}, visibility={tuple(visibility.shape)}, "
            f"source_support={tuple(source_support.shape)}."
        )
    if not 0.0 <= float(visibility_threshold) <= 1.0:
        raise ValueError("visibility_threshold must be in [0,1].")
    if float(coverage_threshold) <= 0.0:
        raise ValueError("coverage_threshold must be positive.")
    dilation_radius = int(dilation_radius)
    if dilation_radius < 0:
        raise ValueError("dilation_radius must be non-negative.")

    device = flow_xy.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    target_x = x[None] + flow_xy[..., 0]
    target_y = y[None] + flow_xy[..., 1]
    finite = torch.isfinite(target_x) & torch.isfinite(target_y)
    source_valid = (
        source_support[:, 0]
        & finite
        & (visibility[:, 0] >= float(visibility_threshold))
    )

    safe_x = torch.where(finite, target_x, torch.zeros_like(target_x))
    safe_y = torch.where(finite, target_y, torch.zeros_like(target_y))
    x0 = torch.floor(safe_x)
    y0 = torch.floor(safe_y)
    dx = safe_x - x0
    dy = safe_y - y0
    x0 = x0.long()
    y0 = y0.long()

    score = torch.zeros(
        (batch, height * width), device=device, dtype=torch.float32
    )
    source_weight = source_valid.float()
    corners = (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x0 + 1, y0, dx * (1.0 - dy)),
        (x0, y0 + 1, (1.0 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    )
    for corner_x, corner_y, corner_weight in corners:
        inside = (
            (corner_x >= 0)
            & (corner_x < width)
            & (corner_y >= 0)
            & (corner_y < height)
        )
        flat_index = (
            corner_y.clamp(0, height - 1) * width
            + corner_x.clamp(0, width - 1)
        ).reshape(batch, -1)
        weight = (source_weight * inside * corner_weight).reshape(batch, -1)
        score.scatter_add_(1, flat_index, weight)

    coverage = score.reshape(batch, 1, height, width) >= float(
        coverage_threshold
    )
    if dilation_radius:
        kernel = 2 * dilation_radius + 1
        coverage = F.max_pool2d(
            coverage.float(), kernel_size=kernel, stride=1, padding=dilation_radius
        ).bool()
    return coverage


class RollingSPOTTracker:
    """Lazy, frozen pairwise SPOT inference without its streaming memory stack."""

    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        *,
        mixed_precision: bool = True,
        iterations: int = 4,
    ) -> None:
        self.repo_path = repo_path
        self.checkpoint_path = checkpoint_path
        self.mixed_precision = bool(mixed_precision)
        self.iterations = max(1, int(iterations))
        self._model = None
        self._device = None

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def _resolve(self, path: str) -> Path:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self._project_root() / resolved
        return resolved.resolve()

    def _load(self, device: torch.device):
        if self._model is not None:
            if self._device != device:
                self._model.to(device)
                self._device = device
            return self._model

        repo_path = self._resolve(self.repo_path)
        checkpoint_path = self._resolve(self.checkpoint_path)
        if not (repo_path / "spot" / "models" / "shelf").is_dir():
            raise FileNotFoundError(
                "SPOT source was not found at "
                f"{repo_path}. Place the official DQiaole/SPOT repository there."
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SPOT checkpoint was not found at {checkpoint_path}."
            )
        repo_string = str(repo_path)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)

        from spot.models.shelf import SPOT

        model_config = SimpleNamespace(
            patch_size=4,
            num_iter=self.iterations,
            refine_alpha=True,
            norm_fnet="instance",
            norm_cnet="instance",
        )
        runtime_config = SimpleNamespace(ablate_fuser=False)
        model = SPOT(
            model_config,
            mixed_precision=self.mixed_precision,
            cfg=runtime_config,
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        state_dict = checkpoint.get("model", checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "SPOT checkpoint is incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}."
            )
        model.requires_grad_(False)
        model.eval().to(device)
        self._model = model
        self._device = device
        return model

    @torch.inference_mode()
    def predict_pair(
        self,
        previous_image: torch.Tensor,
        current_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return previous-to-current full-resolution flow and visibility."""
        if previous_image.shape != current_image.shape:
            raise ValueError(
                "SPOT image shapes must match, got "
                f"{tuple(previous_image.shape)} and {tuple(current_image.shape)}."
            )
        if previous_image.ndim != 4 or previous_image.shape[1] != 3:
            raise ValueError(
                "SPOT images must be [B,3,H,W], got "
                f"{tuple(previous_image.shape)}."
            )
        model = self._load(previous_image.device)
        height, width = previous_image.shape[-2:]
        pad_height = (-height) % 8
        pad_width = (-width) % 8
        padding = (0, pad_width, 0, pad_height)
        # SPOT is trained and officially evaluated with RGB mapped from
        # [0, 1] to [-1, 1].
        previous = F.pad(
            previous_image.detach().float().clamp(0.0, 1.0) * 2.0 - 1.0,
            padding,
            mode="replicate",
        )
        current = F.pad(
            current_image.detach().float().clamp(0.0, 1.0) * 2.0 - 1.0,
            padding,
            mode="replicate",
        )

        net, context = model.encode_context(previous)
        coords0, coords1, source_features, alpha, _ = model.encode_features(
            previous
        )
        target_features = model.fnet(current).float()
        sensory_memory = torch.zeros(
            (source_features.shape[0], 128, *source_features.shape[-2:]),
            device=source_features.device,
            dtype=source_features.dtype,
        )
        prediction = model.predict_flow(
            net,
            context,
            coords0,
            coords1,
            source_features,
            target_features,
            sensory_memory,
            alpha,
            is_train=False,
            num_iter=self.iterations,
            attn=None,
        )
        flow = prediction["flow"][:, :height, :width].float()
        visibility = prediction["alpha"][:, :height, :width].float().sigmoid()
        if not torch.isfinite(flow).all() or not torch.isfinite(visibility).all():
            raise FloatingPointError(
                "SPOT produced non-finite flow or visibility for an image pair "
                f"with resolution {height}x{width}."
            )
        return flow, visibility
