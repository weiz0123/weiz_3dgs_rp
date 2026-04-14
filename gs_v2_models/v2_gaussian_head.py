import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        groups = 8 if cout % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class V2GaussianHead(nn.Module):
    def __init__(self, feat_dim, hidden=128, sh_degree=1, num_surfaces=1):
        super().__init__()

        self.sh_degree = sh_degree
        self.sh_coeff_dim = (sh_degree + 1) ** 2
        self.sh_out_dim = 3 * self.sh_coeff_dim
        self.num_surfaces = num_surfaces
        self.per_surface_dim = 3 + 3 + 4 + 1 + self.sh_out_dim

        in_dim = feat_dim + 1 + 1 + 3
        out_dim = self.per_surface_dim * self.num_surfaces

        self.net = nn.Sequential(
            ConvBlock(in_dim, hidden),
            ConvBlock(hidden, hidden),
            ConvBlock(hidden, hidden),
        )
        self.out = nn.Conv2d(hidden, out_dim, 1)

    def forward(self, feat, depth, conf, rgb):
        x = torch.cat([feat, depth, conf, rgb], dim=1)
        h = self.net(x)
        raw = self.out(h)

        batch_size, _, height, width = raw.shape
        raw = raw.view(batch_size, self.num_surfaces, self.per_surface_dim, height, width)

        cursor = 0
        d_xyz = 0.05 * torch.tanh(raw[:, :, cursor:cursor + 3])
        cursor += 3

        s_raw = raw[:, :, cursor:cursor + 3]
        cursor += 3

        q_raw = raw[:, :, cursor:cursor + 4]
        cursor += 4

        a_raw = raw[:, :, cursor:cursor + 1]
        cursor += 1

        sh_raw = raw[:, :, cursor:cursor + self.sh_out_dim]

        quat = F.normalize(q_raw, dim=2, eps=1e-6)
        conf_expanded = conf.unsqueeze(1)
        base_scales = 0.002 + 0.03 * torch.sigmoid(s_raw)
        opacity_raw = torch.sigmoid(a_raw)

        scales = base_scales * (1.25 - 0.75 * conf_expanded)
        opacity = opacity_raw * (0.25 + 0.75 * conf_expanded)
        sh_coeffs = sh_raw.view(batch_size, self.num_surfaces, 3, self.sh_coeff_dim, height, width)

        return {
            "d_xyz": d_xyz,
            "scales": scales,
            "quat": quat,
            "opacity": opacity,
            "sh_coeffs": sh_coeffs,
            "sh_degree": self.sh_degree,
        }
