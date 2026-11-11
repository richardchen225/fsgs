from __future__ import annotations

import math

import torch


def sanitize_pose_encoding_for_geometry(
    pose_encoding: torch.Tensor,
    *,
    fov_epsilon: float = 1e-3,
) -> torch.Tensor:
    """Return a finite translation/quaternion/FOV encoding for geometry."""
    safe_translation = torch.nan_to_num(
        pose_encoding[..., :3],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    raw_quaternion = torch.nan_to_num(
        pose_encoding[..., 3:7],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    quaternion_norm = raw_quaternion.norm(dim=-1, keepdim=True)
    identity_quaternion = torch.zeros_like(raw_quaternion)
    identity_quaternion[..., 3] = 1.0
    safe_quaternion = raw_quaternion / quaternion_norm.clamp_min(1e-6)
    safe_quaternion = torch.where(
        quaternion_norm > 1e-6,
        safe_quaternion,
        identity_quaternion,
    )
    safe_fov = torch.nan_to_num(
        pose_encoding[..., 7:9],
        nan=math.pi / 2.0,
        posinf=math.pi - fov_epsilon,
        neginf=fov_epsilon,
    ).clamp(fov_epsilon, math.pi - fov_epsilon)
    return torch.cat(
        [safe_translation, safe_quaternion, safe_fov],
        dim=-1,
    )
