import sys
from pathlib import Path

import torch


class RollingAllTracker:
    """Frozen pairwise AllTracker adapter with SPOT-compatible outputs."""

    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        *,
        mixed_precision: bool = True,
        iterations: int = 4,
        window_length: int = 16,
    ) -> None:
        self.repo_path = repo_path
        self.checkpoint_path = checkpoint_path
        self.mixed_precision = bool(mixed_precision)
        self.iterations = max(1, int(iterations))
        self.window_length = max(2, int(window_length))
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
        if device.type != "cuda":
            raise RuntimeError(
                "The official AllTracker inference path requires CUDA."
            )

        repo_path = self._resolve(self.repo_path)
        checkpoint_path = self._resolve(self.checkpoint_path)
        if not (repo_path / "nets" / "alltracker.py").is_file():
            raise FileNotFoundError(
                "AllTracker source was not found at "
                f"{repo_path}. Place the official aharley/alltracker repository "
                "there."
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "AllTracker checkpoint was not found at "
                f"{checkpoint_path}. Download alltracker.pth from "
                "https://huggingface.co/aharley/alltracker."
            )

        repo_parent = str(repo_path.parent)
        if repo_parent not in sys.path:
            sys.path.insert(0, repo_parent)
        try:
            from AllTracker.nets.alltracker import Net
        except ImportError as error:
            raise ImportError(
                "The configured AllTracker source must include the vendored "
                "package layout at AllTracker/nets/alltracker.py."
            ) from error

        # The checkpoint supplies every parameter, so avoid downloading the
        # ConvNeXt initialization that the training constructor normally uses.
        model = Net(seqlen=self.window_length, init_weights=False)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=True)
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
        """Return previous-to-current flow and visibility-confidence product."""
        if previous_image.shape != current_image.shape:
            raise ValueError(
                "AllTracker image shapes must match, got "
                f"{tuple(previous_image.shape)} and {tuple(current_image.shape)}."
            )
        if previous_image.ndim != 4 or previous_image.shape[1] != 3:
            raise ValueError(
                "AllTracker images must be [B,3,H,W], got "
                f"{tuple(previous_image.shape)}."
            )
        model = self._load(previous_image.device)
        images = torch.stack(
            [previous_image.detach(), current_image.detach()], dim=1
        ).float().clamp(0.0, 1.0) * 255.0
        with torch.autocast(
            device_type=previous_image.device.type,
            dtype=torch.bfloat16,
            enabled=(
                self.mixed_precision
                and previous_image.device.type == "cuda"
                and torch.cuda.is_bf16_supported()
            ),
        ):
            flow, visibility_confidence, _, _ = model(
                images,
                iters=self.iterations,
                sw=None,
                is_training=False,
            )
        flow = flow.to(previous_image.device).float()
        visibility_confidence = visibility_confidence.to(
            previous_image.device
        ).float()
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise RuntimeError(
                "AllTracker returned an invalid flow shape: "
                f"{tuple(flow.shape)}."
            )
        if visibility_confidence.shape != flow.shape:
            raise RuntimeError(
                "AllTracker visibility/confidence must match flow shape; got "
                f"{tuple(visibility_confidence.shape)} and {tuple(flow.shape)}."
            )
        # Official evaluation multiplies visibility (channel 0) by tracking
        # confidence (channel 1) before thresholding.
        visibility = (
            visibility_confidence[:, 0] * visibility_confidence[:, 1]
        ).clamp(0.0, 1.0)
        flow = flow.permute(0, 2, 3, 1)
        if not torch.isfinite(flow).all() or not torch.isfinite(visibility).all():
            raise FloatingPointError(
                "AllTracker produced non-finite flow or visibility-confidence."
            )
        return flow, visibility
