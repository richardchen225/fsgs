from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraPhaseSettings:
    sim3_alignment_mode: str
    geometry_camera_source: str
    source_render_camera: str
    heldout_test_camera: str


def resolve_camera_phase_settings(cfg, training: bool) -> CameraPhaseSettings:
    """Resolve phase-specific cameras with legacy fields as fallbacks."""
    phase = "training" if training else "evaluation"

    def resolve(name: str, default: str, valid: set[str]) -> str:
        value = getattr(cfg, f"{phase}_{name}", None)
        if value is None:
            value = getattr(cfg, name, default)
        value = str(value).lower()
        if value not in valid:
            choices = ", ".join(sorted(valid))
            raise ValueError(
                f"{phase}_{name} must be one of {choices}, got {value!r}."
            )
        return value

    camera_sources = {"gt", "predicted"}
    return CameraPhaseSettings(
        sim3_alignment_mode=resolve(
            "sim3_alignment_mode", "never", {"auto", "always", "never"}
        ),
        geometry_camera_source=resolve(
            "geometry_camera_source", "predicted", camera_sources
        ),
        source_render_camera=resolve(
            "source_render_camera", "predicted", camera_sources
        ),
        heldout_test_camera=resolve(
            "heldout_test_camera", "predicted", camera_sources
        ),
    )
