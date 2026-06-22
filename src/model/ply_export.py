from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
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
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    # 新增两个参数用于控制过滤强度
    min_opacity: float = 0.005,  # 透明度低于0.005的点会被删除
    max_scale: float = 0.01       # 某个轴超过5.0的大球会被删除
):
    # ================= 核心修改区域开始 =================
    print(f"原始点数量: {means.shape[0]}")
    
    # 1. 创建透明度掩码 (保留不那么透明的点)
    # 注意：这里假设输入的 opacities 已经是 [0, 1] 范围的数值
    # 如果输入是 logit (未经过 sigmoid)，你需要先 torch.sigmoid(opacities)
    def inverse_sigmoid(x, eps=1e-6):
        x = x.clamp(eps, 1 - eps)
        return torch.log(x / (1 - x))

    opacities = opacities.squeeze(-1)

    # 如果输入是 alpha [0, 1]
    alpha = opacities
    raw_opacity = inverse_sigmoid(alpha)

    # pruning 用 alpha
    mask_opacity = alpha > min_opacity
    
    # 2. 创建体积掩码 (保留体积正常的点，去除巨大的背景球)
    # scales 通常是世界坐标下的尺寸，如果有点大得离谱（比如天空球），就删掉
    mask_scale = scales.max(dim=-1).values < max_scale

    # 3. 合并掩码
    mask = mask_opacity & mask_scale
    
    # 4. 应用掩码，过滤所有属性
    means = means[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    harmonics = harmonics[mask]
    opacities = opacities[mask]
    
    print(f"过滤后剩余点数量: {means.shape[0]}")
    # ================= 核心修改区域结束 =================

    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0 if save_sh_dc_only else f_rest.shape[1])]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    ]
    if save_sh_dc_only:
        # remove f_rest from attributes
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
