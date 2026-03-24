import torch
import torch.nn.functional as F


def make_pixel_grid(B, H, W, device, dtype=torch.float32):
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=0)  # [3,H,W]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,3,H,W]
    return grid


def invert_pose(c2w: torch.Tensor):
    return torch.inverse(c2w)


def scale_intrinsics_batch(K: torch.Tensor, src_hw, dst_hw):
    Hs, Ws = src_hw
    Hd, Wd = dst_hw
    sx = Wd / float(Ws)
    sy = Hd / float(Hs)

    K2 = K.clone()
    K2[:, 0, 0] *= sx
    K2[:, 1, 1] *= sy
    K2[:, 0, 2] *= sx
    K2[:, 1, 2] *= sy
    return K2


def unproject_depth(depth: torch.Tensor, K: torch.Tensor):
    """
    depth: [B,1,H,W]
    K:     [B,3,3]
    return [B,3,H,W] in camera coordinates
    """
    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    pix = make_pixel_grid(B, H, W, device, dtype)  # [B,3,H,W]
    pix = pix.reshape(B, 3, -1)  # [B,3,HW]

    Kinv = torch.inverse(K)  # [B,3,3]
    rays = Kinv @ pix  # [B,3,HW]
    X = rays * depth.reshape(B, 1, -1)  # [B,3,HW]

    return X.reshape(B, 3, H, W)


def cam_to_world_grid(X_cam: torch.Tensor, c2w: torch.Tensor):
    """
    X_cam: [B,3,H,W]
    c2w:   [B,4,4]
    """
    B, _, H, W = X_cam.shape

    X_flat = X_cam.reshape(B, 3, -1)
    ones = torch.ones(B, 1, H * W, device=X_cam.device, dtype=X_cam.dtype)
    Xh = torch.cat([X_flat, ones], dim=1)  # [B,4,HW]

    Xw = c2w @ Xh  # [B,4,HW]
    return Xw[:, :3].reshape(B, 3, H, W)


def world_to_cam_grid(X_world: torch.Tensor, w2c: torch.Tensor):
    """
    X_world: [B,3,H,W]
    w2c:     [B,4,4]
    """
    B, _, H, W = X_world.shape

    X_flat = X_world.reshape(B, 3, -1)
    ones = torch.ones(B, 1, H * W, device=X_world.device, dtype=X_world.dtype)
    Xh = torch.cat([X_flat, ones], dim=1)  # [B,4,HW]

    Xc = w2c @ Xh
    return Xc[:, :3].reshape(B, 3, H, W)


def project_points_grid(X_cam: torch.Tensor, K: torch.Tensor, eps=1e-6):
    """
    X_cam: [B,3,H,W]
    K:     [B,3,3]
    return uv: [B,H,W,2]
    """
    B, _, H, W = X_cam.shape

    x = X_cam[:, 0]
    y = X_cam[:, 1]
    z = X_cam[:, 2].clamp(min=eps)

    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    return torch.stack([u, v], dim=-1)  # [B,H,W,2]


def uv_to_grid(uv: torch.Tensor, H: int, W: int):
    """
    uv: [B,H,W,2] in pixel coordinates
    return grid for grid_sample: [B,H,W,2] in [-1,1]
    """
    u = uv[..., 0]
    v = uv[..., 1]

    gx = 2.0 * (u / max(W - 1, 1)) - 1.0
    gy = 2.0 * (v / max(H - 1, 1)) - 1.0

    return torch.stack([gx, gy], dim=-1)


def warp_feature_to_ref_plane(
    src_feat,
    depth_plane,
    K_ref,
    c2w_ref,
    K_src,
    c2w_src,
):
    """
    Warp src_feat into ref image plane at a given depth hypothesis.

    src_feat:    [B,C,H,W]
    depth_plane: [B,1,H,W]
    K_ref:       [B,3,3]
    c2w_ref:     [B,4,4]
    K_src:       [B,3,3]
    c2w_src:     [B,4,4]
    """

    B, C, H, W = src_feat.shape

    X_ref_cam = unproject_depth(depth_plane, K_ref)  # [B,3,H,W]
    X_world = cam_to_world_grid(X_ref_cam, c2w_ref)  # [B,3,H,W]

    w2c_src = invert_pose(c2w_src)
    X_src_cam = world_to_cam_grid(X_world, w2c_src)  # [B,3,H,W]

    uv_src = project_points_grid(X_src_cam, K_src)  # [B,H,W,2]
    grid = uv_to_grid(uv_src, H, W)  # [B,H,W,2]

    warped = F.grid_sample(
        src_feat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    z = X_src_cam[:, 2:3]  # [B,1,H,W]

    valid = (
        (grid[..., 0] >= -1.0)
        & (grid[..., 0] <= 1.0)
        & (grid[..., 1] >= -1.0)
        & (grid[..., 1] <= 1.0)
        & (z[:, 0] > 1e-6)
    ).unsqueeze(1).float()

    return warped, valid
