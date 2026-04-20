import torch
import torch.nn as nn
import torch.nn.functional as F

from debug_util import DEBUG

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

    intrinsic = intrinsic.to(dtype).clone()
    max_f = torch.max(intrinsic[..., 0, 0].abs().amax(), intrinsic[..., 1, 1].abs().amax())
    max_c = torch.max(intrinsic[..., 0, 2].abs().amax(), intrinsic[..., 1, 2].abs().amax())
    if max_f < 10.0 and max_c <= 2.0:
        intrinsic[..., 0, 0] = intrinsic[..., 0, 0] * w
        intrinsic[..., 1, 1] = intrinsic[..., 1, 1] * h
        intrinsic[..., 0, 2] = intrinsic[..., 0, 2] * w
        intrinsic[..., 1, 2] = intrinsic[..., 1, 2] * h

    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(1, 3, h * w)
    pixels = pixels.expand(n, -1, -1)

    cam_points = torch.linalg.solve(intrinsic, pixels)
    cam_points = cam_points * depth.reshape(n, 1, h * w)

    extrinsic = extrinsic.to(dtype)
    if extrinsic.shape[-2:] == (3, 4):
        rotation = extrinsic[:, :3, :3]
        translation = extrinsic[:, :3, 3:].contiguous()
    elif extrinsic.shape[-2:] != (4, 4):
        raise ValueError(f"Expected extrinsic shape [N,3,4] or [N,4,4], got {tuple(extrinsic.shape)}")
    else:
        rotation = extrinsic[:, :3, :3]
        translation = extrinsic[:, :3, 3:].contiguous()

    cam_to_world = rotation.transpose(1, 2)
    world_points = cam_to_world @ (cam_points - translation)
    return world_points.reshape(n, 3, h, w)


def _normalize_confidence(conf):
    conf = torch.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    conf_min = float(conf.detach().amin())
    conf_max = float(conf.detach().amax())
    if conf_min < 0.0 or conf_max > 1.0:
        conf = torch.sigmoid(conf)
    return conf.clamp(0.0, 1.0)


def _relative_depth_gradient(depth):
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_safe = depth.clamp_min(1e-6)
    grad_x = F.pad(torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1]), (1, 0, 0, 0))
    grad_y = F.pad(torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :]), (0, 0, 1, 0))
    return torch.maximum(grad_x, grad_y) / depth_safe


class DepthAnchoredGaussianHead(nn.Module):
    """
    Gaussian head where VGGT depth initializes Gaussian means directly.

    This head learns appearance and basic Gaussian parameters from image features:
    - `means3D` comes from backprojected depth plus a learned small `delta_xyz`
    - `quat` is predicted and normalized
    - learned outputs are `delta_xyz`, `scales`, `quat`, `opacity`, and `sh_coeffs`
    """

    def __init__(
        self,
        feat_dim,
        hidden=256,
        sh_degree=0,
        num_surfaces=1,
        min_scale=0.001,
        max_scale=0.02,
        init_dc_bias=0.5,
    ):
        super().__init__()

        self.sh_degree = sh_degree
        self.sh_coeff_dim = (sh_degree + 1) ** 2
        self.sh_out_dim = 3 * self.sh_coeff_dim
        self.num_surfaces = num_surfaces
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.init_dc_bias = init_dc_bias

        # Per surface:
        #   3 delta xyz channels
        #   3 scale channels
        #   4 quaternion channels
        #   1 opacity channel
        #   3 * num_sh_coeffs SH channels
        self.per_surface_dim = 3 + 3 + 4 + 1 + self.sh_out_dim
        out_dim = self.per_surface_dim * self.num_surfaces

        self.net = nn.Sequential(
            ConvBlock(feat_dim, hidden, p=1, d=1),
            ConvBlock(hidden, hidden, p=2, d=2),
            ConvBlock(hidden, hidden, p=4, d=4),
            ConvBlock(hidden, hidden, p=1, d=1),
        )
        self.out = nn.Conv2d(hidden, out_dim, 1)
        self._init_output_layer()

    def _init_output_layer(self):
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        with torch.no_grad():
            for surface_idx in range(self.num_surfaces):
                base = surface_idx * self.per_surface_dim
                quat_base = base + 6
                self.out.bias[quat_base] = 1.0
                opacity_base = base + 10
                self.out.bias[opacity_base] = -2.0
                sh_base = base + 11  # 3 dxyz + 3 scale + 4 quat + 1 opacity
                for color_idx in range(3):
                    dc_index = sh_base + color_idx * self.sh_coeff_dim
                    self.out.bias[dc_index] = self.init_dc_bias

    def forward(self, feat, depth, intrinsic, extrinsic, conf=None):
        h = self.net(feat)

        raw = self.out(h)
        batch_size, _, height, width = raw.shape
        raw = raw.view(batch_size, self.num_surfaces, self.per_surface_dim, height, width)

        cursor = 0
        dxyz_raw = raw[:, :, cursor:cursor + 3]
        cursor += 3
        s_raw = raw[:, :, cursor:cursor + 3]
        cursor += 3
        q_raw = raw[:, :, cursor:cursor + 4]
        cursor += 4
        a_raw = raw[:, :, cursor:cursor + 1]
        cursor += 1
        sh_raw = raw[:, :, cursor:cursor + self.sh_out_dim]

        finite_depth = torch.isfinite(depth).to(raw.dtype)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_for_points = depth.clamp_min(1e-3)
        depth_positive_gate = torch.sigmoid(50.0 * (depth - 1e-3))
        depth_edge = _relative_depth_gradient(depth_for_points)
        edge_gate = (0.5 + 0.5 * torch.exp(-2.0 * depth_edge)).clamp(0.5, 1.0)

        if conf is not None:
            conf = _normalize_confidence(conf.to(raw.dtype))
        else:
            conf = torch.ones_like(depth, dtype=raw.dtype)

        valid_expanded = finite_depth.unsqueeze(1)
        depth_positive_expanded = depth_positive_gate.unsqueeze(1)
        conf_gate = (0.35 + 0.65 * conf).unsqueeze(1)
        edge_expanded = edge_gate.unsqueeze(1)

        offset_gate = valid_expanded * conf_gate * edge_expanded * depth_positive_expanded
        opacity_gate = valid_expanded * conf_gate * (0.25 + 0.75 * depth_positive_expanded)
        scale_gate = 0.75 + 0.25 * conf_gate

        d_xyz = 0.001 * torch.tanh(dxyz_raw) * offset_gate
        base_scales = torch.exp(s_raw - 6.0).clamp(min=self.min_scale, max=self.max_scale)
        quat = F.normalize(q_raw, dim=2, eps=1e-6)
        opacity = torch.sigmoid(a_raw) * opacity_gate
        scales = (base_scales * scale_gate).clamp(min=self.min_scale, max=self.max_scale)

        sh_coeffs = sh_raw.view(
            batch_size,
            self.num_surfaces,
            3,
            self.sh_coeff_dim,
            height,
            width,
        )

        base_means = _depth_to_world_points(depth_for_points, intrinsic, extrinsic)
        means3D = base_means.unsqueeze(1) + d_xyz

        if DEBUG.is_first_batch():
            DEBUG.log_debuge_csv(
                "gaussian_head_forward",
                feat=feat,
                depth=depth,
                depth_for_points=depth_for_points,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                conf=conf,
                finite_depth=finite_depth,
                depth_positive_gate=depth_positive_gate,
                depth_edge=depth_edge,
                edge_gate=edge_gate,
                conf_gate=conf_gate,
                offset_gate=offset_gate,
                opacity_gate=opacity_gate,
                scale_gate=scale_gate,
                base_means=base_means,
                d_xyz=d_xyz,
                scales=scales,
                quat=quat,
                opacity=opacity,
                sh_coeffs=sh_coeffs,
                means3D=means3D,
            )

        return {
            "means3D": means3D,
            "d_xyz": d_xyz,
            "scales": scales,
            "quat": quat,
            "opacity": opacity,
            "sh_coeffs": sh_coeffs,
            "sh_degree": self.sh_degree,
        }
