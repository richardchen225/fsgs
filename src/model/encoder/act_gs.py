import torch
from einops import rearrange

    
def reg_dense_offsets(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    shift = torch.tensor(6.0, dtype=d.dtype, device=d.device)
    
    return xyz / d.clamp(min=1e-8) * (torch.exp(d - shift) - torch.exp(-shift))

def reg_dense_scales(scales):
    return scales.exp()

def reg_dense_rotation(rotations, eps=1e-8):
    return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

def reg_dense_sh(sh):
    return rearrange(sh, '... (d_sh xyz) -> ... d_sh xyz', xyz=3)

def reg_dense_opacities(opacities):
    return opacities.sigmoid()

def reg_dense_weights(weights):
    return weights.sigmoid()

def reg_dense_sb(sb):
    return sb.sigmoid()