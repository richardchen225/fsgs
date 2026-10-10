from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, "gaussian"],
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    min_opacity: float = 0.005,
    max_scale: float = 0.01,
    filter_gaussians: bool = True,
):
    if harmonics.ndim != 3:
        raise ValueError(
            "PLY export expects harmonics with shape [N, 3, D] or [N, D, 3], "
            f"got {tuple(harmonics.shape)}."
        )
    if harmonics.shape[1] == 3:
        pass
    elif harmonics.shape[2] == 3:
        harmonics = harmonics.permute(0, 2, 1)
    else:
        raise ValueError(
            "PLY export could not identify the RGB dimension in harmonics "
            f"with shape {tuple(harmonics.shape)}."
        )

    if opacities.ndim == 2 and opacities.shape[-1] == 1:
        opacities = opacities.squeeze(-1)
    elif opacities.ndim != 1:
        raise ValueError(
            "PLY export expects opacities with shape [N] or [N, 1], "
            f"got {tuple(opacities.shape)}."
        )
    original_count = means.shape[0]

    if filter_gaussians:
        mask = (opacities > min_opacity) & (
            scales.max(dim=-1).values < max_scale
        )
        means = means[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        harmonics = harmonics[mask]
        opacities = opacities[mask]

    print(
        f"PLY Gaussian count: {means.shape[0]}/{original_count} "
        f"(filtering={'enabled' if filter_gaussians else 'disabled'})"
    )

    if shift_and_scale:
        means = means - means.median(dim=0).values
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)
    opacity_logits = torch.logit(opacities.clamp(1e-6, 1.0 - 1e-6))
    num_rest = 0 if save_sh_dc_only else f_rest.shape[1]
    dtype_full = [
        (attribute, "f4")
        for attribute in construct_list_of_attributes(num_rest)
    ]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacity_logits[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    ]
    if save_sh_dc_only:
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    if attributes.shape[1] != len(dtype_full):
        raise RuntimeError(
            "PLY attribute/schema mismatch: "
            f"values={attributes.shape[1]}, fields={len(dtype_full)}, "
            f"harmonics={tuple(harmonics.shape)}."
        )
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
