import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, d=1):
        super().__init__()
        groups = 8 if cout % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, padding=p, dilation=d),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


def _depth_to_world_points(depth, intrinsic, extrinsic):
    """
    Backproject depth into world-space points.

    Args:
        depth: [N, 1, H, W]
        intrinsic: [N, 3, 3]
        extrinsic: [N, 3, 4] or [N, 4, 4] world-to-camera

    Returns:
        world points with shape [N, 3, H, W]
    """
    n, _, h, w = depth.shape
    device = depth.device
    dtype = depth.dtype

    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(1, 3, h * w)
    pixels = pixels.expand(n, -1, -1)

    inv_k = torch.inverse(intrinsic.to(dtype))
    cam_points = inv_k @ pixels
    cam_points = cam_points * depth.reshape(n, 1, h * w)

    cam_points_homo = torch.cat(
        [cam_points, torch.ones(n, 1, h * w, device=device, dtype=dtype)],
        dim=1,
    )
    extrinsic = extrinsic.to(dtype)
    if extrinsic.shape[-2:] == (3, 4):
        extrinsic_h = torch.zeros(n, 4, 4, device=device, dtype=dtype)
        extrinsic_h[:, :3, :4] = extrinsic
        extrinsic_h[:, 3, 3] = 1.0
        extrinsic = extrinsic_h
    elif extrinsic.shape[-2:] != (4, 4):
        raise ValueError(f"Expected extrinsic shape [N,3,4] or [N,4,4], got {tuple(extrinsic.shape)}")

    inv_e = torch.inverse(extrinsic)
    world_points = inv_e @ cam_points_homo
    return world_points[:, :3, :].reshape(n, 3, h, w)


class DepthAnchoredGaussianHead(nn.Module):
    """
    Gaussian head where VGGT depth initializes Gaussian means directly.

    This head learns appearance and basic Gaussian parameters from image features:
    - `means3D` comes from backprojected depth
    - `d_xyz` is fixed to zero
    - `quat` is fixed to the identity rotation
    - learned outputs are `scales`, `opacity`, and `sh_coeffs`
    """

    def __init__(
        self,
        feat_dim,
        hidden=256,
        sh_degree=1,
        num_surfaces=1,
        min_scale=0.002,
        max_scale=0.03,
    ):
        super().__init__()

        self.sh_degree = sh_degree
        self.sh_coeff_dim = (sh_degree + 1) ** 2
        self.sh_out_dim = 3 * self.sh_coeff_dim
        self.num_surfaces = num_surfaces
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Per surface:
        #   3 scale channels
        #   1 opacity channel
        #   3 * num_sh_coeffs SH channels
        self.per_surface_dim = 3 + 1 + self.sh_out_dim
        out_dim = self.per_surface_dim * self.num_surfaces

        self.net = nn.Sequential(
            ConvBlock(feat_dim, hidden, p=1, d=1),
            ConvBlock(hidden, hidden, p=2, d=2),
            ConvBlock(hidden, hidden, p=4, d=4),
            ConvBlock(hidden, hidden, p=1, d=1),
        )
        self.upsample_refine = nn.Sequential(
            ConvBlock(hidden, hidden, p=1, d=1),
            ConvBlock(hidden, hidden, p=1, d=1),
        )
        self.out = nn.Conv2d(hidden, out_dim, 1)

    def forward(self, feat, depth, intrinsic, extrinsic, conf=None, output_size=None):
        h = self.net(feat)

        if output_size is not None and h.shape[-2:] != output_size:
            h = F.interpolate(h, size=output_size, mode="bilinear", align_corners=False)
            h = self.upsample_refine(h)
            if depth.shape[-2:] != output_size:
                depth = F.interpolate(depth, size=output_size, mode="bilinear", align_corners=False)
            if conf is not None and conf.shape[-2:] != output_size:
                conf = F.interpolate(conf, size=output_size, mode="bilinear", align_corners=False)

        raw = self.out(h)
        batch_size, _, height, width = raw.shape
        raw = raw.view(batch_size, self.num_surfaces, self.per_surface_dim, height, width)

        cursor = 0
        s_raw = raw[:, :, cursor:cursor + 3]
        cursor += 3
        a_raw = raw[:, :, cursor:cursor + 1]
        cursor += 1
        sh_raw = raw[:, :, cursor:cursor + self.sh_out_dim]

        base_scales = self.min_scale + self.max_scale * torch.sigmoid(s_raw)
        opacity = torch.sigmoid(a_raw)

        if conf is not None:
            conf = conf.to(raw.dtype)
            conf_expanded = conf.unsqueeze(1)
            scales = base_scales * (1.25 - 0.75 * conf_expanded)
            opacity = opacity * (0.25 + 0.75 * conf_expanded)
        else:
            scales = base_scales

        sh_coeffs = sh_raw.view(
            batch_size,
            self.num_surfaces,
            3,
            self.sh_coeff_dim,
            height,
            width,
        )

        base_means = _depth_to_world_points(depth, intrinsic, extrinsic)
        means3D = base_means.unsqueeze(1).expand(-1, self.num_surfaces, -1, -1, -1).contiguous()

        d_xyz = torch.zeros(
            batch_size,
            self.num_surfaces,
            3,
            height,
            width,
            device=raw.device,
            dtype=raw.dtype,
        )

        quat = torch.zeros(
            batch_size,
            self.num_surfaces,
            4,
            height,
            width,
            device=raw.device,
            dtype=raw.dtype,
        )
        quat[:, :, 0] = 1.0

        return {
            "means3D": means3D,
            "d_xyz": d_xyz,
            "scales": scales,
            "quat": quat,
            "opacity": opacity,
            "sh_coeffs": sh_coeffs,
            "sh_degree": self.sh_degree,
        }
