from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.model.camera_phase import resolve_camera_phase_settings


def test_dl3dv_camera_phases_and_append_ablation() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "experiment"
        / "dl3dv.yaml"
    )
    cfg = OmegaConf.load(config_path)
    encoder = cfg.model.encoder

    assert encoder.training_sim3_alignment_mode == "never"
    assert encoder.training_geometry_camera_source == "predicted"
    assert encoder.training_source_render_camera == "predicted"
    assert encoder.training_heldout_test_camera == "predicted"
    assert encoder.evaluation_sim3_alignment_mode == "never"
    assert encoder.evaluation_geometry_camera_source == "predicted"
    assert encoder.evaluation_source_render_camera == "predicted"
    assert encoder.evaluation_heldout_test_camera == "predicted"
    assert encoder.gir_append_mode == "all"
    assert cfg.train.camera_pose_loss_mode == "all_pairs"
    assert cfg.train.camera_global_trajectory_weight == 0.0


def test_phase_camera_overrides_separate_training_and_evaluation() -> None:
    cfg = SimpleNamespace(
        sim3_alignment_mode="auto",
        geometry_camera_source="gt",
        source_render_camera="gt",
        heldout_test_camera="gt",
        training_sim3_alignment_mode="always",
        training_geometry_camera_source="predicted",
        training_source_render_camera="gt",
        training_heldout_test_camera="gt",
        evaluation_sim3_alignment_mode="never",
        evaluation_geometry_camera_source="predicted",
        evaluation_source_render_camera="predicted",
        evaluation_heldout_test_camera="predicted",
    )

    training = resolve_camera_phase_settings(cfg, True)
    evaluation = resolve_camera_phase_settings(cfg, False)

    assert training.sim3_alignment_mode == "always"
    assert training.geometry_camera_source == "predicted"
    assert training.source_render_camera == "gt"
    assert training.heldout_test_camera == "gt"
    assert evaluation.sim3_alignment_mode == "never"
    assert evaluation.geometry_camera_source == "predicted"
    assert evaluation.source_render_camera == "predicted"
    assert evaluation.heldout_test_camera == "predicted"


def test_phase_camera_settings_fall_back_to_legacy_fields() -> None:
    cfg = SimpleNamespace(
        sim3_alignment_mode="auto",
        geometry_camera_source="gt",
        source_render_camera="predicted",
        heldout_test_camera="gt",
    )

    settings = resolve_camera_phase_settings(cfg, True)

    assert settings.sim3_alignment_mode == "auto"
    assert settings.geometry_camera_source == "gt"
    assert settings.source_render_camera == "predicted"
    assert settings.heldout_test_camera == "gt"


def test_phase_camera_settings_reject_invalid_values() -> None:
    cfg = SimpleNamespace(evaluation_sim3_alignment_mode="sometimes")

    with pytest.raises(ValueError, match="evaluation_sim3_alignment_mode"):
        resolve_camera_phase_settings(cfg, False)
