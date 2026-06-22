import torch
import torch.nn.functional as F


def _scatter_uv_to_heatmap(uv, H, W, weights=None):
    """
    uv: [M, 2], pixel coordinate
    return: [H, W]
    """
    x = torch.round(uv[:, 0]).long()
    y = torch.round(uv[:, 1]).long()

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    x = x[valid]
    y = y[valid]

    if weights is None:
        vals = torch.ones_like(x, dtype=torch.float32)
    else:
        vals = weights[valid].float()

    heat = torch.zeros(H * W, device=uv.device, dtype=torch.float32)
    linear_idx = y * W + x
    heat.scatter_add_(0, linear_idx, vals)

    return heat.view(H, W)


def _gaussian_blur_heatmap(heat, kernel_size=21, sigma=7.0):
    """
    heat: [H, W]
    """
    if kernel_size is None or kernel_size <= 1:
        return heat

    if kernel_size % 2 == 0:
        kernel_size += 1

    device = heat.device
    dtype = heat.dtype

    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - (kernel_size - 1) / 2.0

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum().clamp_min(1e-8)

    kernel = g[:, None] * g[None, :]
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    heat_4d = heat.view(1, 1, heat.shape[0], heat.shape[1])
    heat_blur = F.conv2d(heat_4d, kernel, padding=kernel_size // 2)

    return heat_blur[0, 0]


def _heat_to_jet_rgb(heat_norm):
    """
    heat_norm: [H, W], range [0, 1]
    return: [3, H, W], range [0, 1]
    """
    x = heat_norm.clamp(0.0, 1.0)

    r = torch.clamp(1.5 - torch.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = torch.clamp(1.5 - torch.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = torch.clamp(1.5 - torch.abs(4.0 * x - 1.0), 0.0, 1.0)

    return torch.stack([r, g, b], dim=0)


def _make_overlay(rgb, heat, alpha=0.55, blur_ksize=21, blur_sigma=7.0, use_log=True):
    """
    rgb:  [3, H, W], range [0, 1]
    heat: [H, W]
    return:
        heat_rgb: [3, H, W]
        overlay:  [3, H, W]
    """
    heat_vis = heat

    if use_log:
        heat_vis = torch.log1p(heat_vis)

    # heat_vis = _gaussian_blur_heatmap(
    #     heat_vis,
    #     kernel_size=blur_ksize,
    #     sigma=blur_sigma,
    # )
    heat_vis = heat_vis.unsqueeze(0).unsqueeze(0)
    heat_vis = F.max_pool2d(heat_vis, kernel_size=5, stride=1, padding=2)
    heat_vis = heat_vis[0, 0]
    heat_vis = heat_vis / heat_vis.max().clamp_min(1e-8)

    heat_rgb = _heat_to_jet_rgb(heat_vis)

    alpha_map = heat_vis.unsqueeze(0) * alpha
    overlay = rgb * (1.0 - alpha_map) + heat_rgb * alpha_map
    overlay = overlay.clamp(0.0, 1.0)

    return heat_rgb, overlay


def _sampling_count_to_rgb(count):
    """
    count: [H, W], sampled primitive count per snapped pixel
    return: [3, H, W], range [0, 1]

    Color coding:
        count == 0: blue
        count == 1: yellow
        count >= 2: orange-to-red by log count
    """
    count = count.float()
    rgb = torch.zeros(3, *count.shape, device=count.device, dtype=count.dtype)

    zero = count <= 0
    single = count == 1
    multi = count >= 2

    blue = torch.tensor([0.08, 0.22, 1.00], device=count.device, dtype=count.dtype)
    yellow = torch.tensor([1.00, 0.82, 0.10], device=count.device, dtype=count.dtype)

    rgb[:, zero] = blue[:, None]
    rgb[:, single] = yellow[:, None]

    if multi.any():
        max_count = count[multi].max().clamp_min(2.0)
        strength = torch.log1p(count[multi]) / torch.log1p(max_count)
        # Low repeated count is orange; high repeated count approaches red.
        rgb[0, multi] = 1.00
        rgb[1, multi] = 0.45 * (1.0 - strength)
        rgb[2, multi] = 0.06 * (1.0 - strength)

    return rgb


def _make_sampling_count_overlay(rgb, count, alpha=0.55, marker_size=5, zero_alpha=0.12):
    """
    rgb:   [3, H, W], range [0, 1]
    count: [H, W], raw sampled primitive count per pixel
    return:
        count_rgb: [3, H, W]
        overlay:   [3, H, W]
    """
    count_vis = count.float()

    if marker_size is not None and marker_size > 1:
        if marker_size % 2 == 0:
            marker_size += 1
        count_vis = count_vis.unsqueeze(0).unsqueeze(0)
        count_vis = F.max_pool2d(
            count_vis,
            kernel_size=marker_size,
            stride=1,
            padding=marker_size // 2,
        )
        count_vis = count_vis[0, 0]

    count_rgb = _sampling_count_to_rgb(count_vis)

    alpha_map = torch.full_like(count_vis, zero_alpha)
    alpha_map[count_vis == 1] = alpha * 0.75
    alpha_map[count_vis >= 2] = alpha

    overlay = rgb * (1.0 - alpha_map.unsqueeze(0)) + count_rgb * alpha_map.unsqueeze(0)
    overlay = overlay.clamp(0.0, 1.0)

    return count_rgb, overlay


@torch.no_grad()
def make_sampling_heatmap_overlay_tensors(
    image,                 # [b, v, 3, h, w], 建议传 inverse_normalize 后的 [0,1] 图像
    u_v_pixel,             # [b*v, M, 2]
    b,
    v,
    primitive_mask_hard=None,  # [b*v*M, 1] or None
    batch_idx=0,
    alpha=0.55,
    blur_ksize=21,
    blur_sigma=7.0,
):
    """
    返回 tensor，不保存文件。

    返回:
        overlay_all:  [v, 3, h, w]
        overlay_kept: [v, 3, h, w] or None
        heat_all:     [v, 3, h, w]
        heat_kept:    [v, 3, h, w] or None
    """
    assert image.ndim == 5, image.shape
    assert u_v_pixel.ndim == 3 and u_v_pixel.shape[-1] == 2, u_v_pixel.shape

    _, _, _, H, W = image.shape

    bv, M, _ = u_v_pixel.shape
    assert bv == b * v, f"u_v_pixel first dim {bv} != b*v {b*v}"

    device = image.device

    rgb_views = image[batch_idx].float().clamp(0.0, 1.0)  # [v, 3, H, W]

    uv = u_v_pixel.detach().float().to(device).reshape(b, v, M, 2)[batch_idx]

    if primitive_mask_hard is not None:
        assert primitive_mask_hard.shape == (b * v * M, 1), (
            primitive_mask_hard.shape,
            (b * v * M, 1),
        )

        mask = primitive_mask_hard.detach().float().to(device).reshape(b, v, M)[batch_idx]
        mask = (mask > 0.5)
    else:
        mask = None

    overlay_all_list = []
    overlay_kept_list = []
    heat_all_list = []
    heat_kept_list = []

    for view_id in range(v):
        rgb = rgb_views[view_id]       # [3, H, W]
        uv_view = uv[view_id]          # [M, 2]

        # -----------------------------
        # all sampled positions
        # -----------------------------
        heat_all = _scatter_uv_to_heatmap(uv_view, H, W)
        heat_all_rgb, overlay_all = _make_sampling_count_overlay(
            rgb=rgb,
            count=heat_all,
            alpha=alpha,
        )

        heat_all_list.append(heat_all_rgb)
        overlay_all_list.append(overlay_all)

        # -----------------------------
        # mask 后保留的位置
        # -----------------------------
        if mask is not None:
            keep = mask[view_id]       # [M]
            uv_kept = uv_view[keep]

            heat_kept = _scatter_uv_to_heatmap(uv_kept, H, W)
            heat_kept_rgb, overlay_kept = _make_sampling_count_overlay(
                rgb=rgb,
                count=heat_kept,
                alpha=alpha,
            )

            heat_kept_list.append(heat_kept_rgb)
            overlay_kept_list.append(overlay_kept)

    overlay_all = torch.stack(overlay_all_list, dim=0)
    heat_all = torch.stack(heat_all_list, dim=0)

    if mask is not None:
        overlay_kept = torch.stack(overlay_kept_list, dim=0)
        heat_kept = torch.stack(heat_kept_list, dim=0)
    else:
        overlay_kept = None
        heat_kept = None

    return {
        "overlay_all": overlay_all,
        "overlay_kept": overlay_kept,
        "heat_all": heat_all,
        "heat_kept": heat_kept,
    }
