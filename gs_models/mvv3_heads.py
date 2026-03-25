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


class GaussianHead(nn.Module):
    def __init__(self, ref_feat_dim, mv_feat_dim, fused_dim, hidden=128):
        super().__init__()
        in_dim = ref_feat_dim + mv_feat_dim + fused_dim + 1 + 1 + 3

        self.net = nn.Sequential(
            ConvBlock(in_dim, hidden),
            ConvBlock(hidden, hidden),
            ConvBlock(hidden, hidden),
        )

        self.out = nn.Conv2d(hidden, 14, 1)

    def forward(self, ref_feat, mv_feat, fused_feat, depth, conf, rgb):
        x = torch.cat([ref_feat, mv_feat, fused_feat, depth, conf, rgb], dim=1)
        h = self.net(x)
        raw = self.out(h)

        d_xyz = 0.05 * torch.tanh(raw[:, 0:3])
        s_raw = raw[:, 3:6]
        q_raw = raw[:, 6:10]
        a_raw = raw[:, 10:11]
        c_raw = raw[:, 11:14]

        quat = F.normalize(q_raw, dim=1, eps=1e-6)
        base_scales = 0.002 + 0.03 * torch.sigmoid(s_raw)
        opacity_raw = torch.sigmoid(a_raw + 1.0)
        color = (rgb + 0.25 * torch.tanh(c_raw)).clamp(0.0, 1.0)

        scales = base_scales * (1.25 - 0.75 * conf)
        opacity = opacity_raw * (0.35 + 0.65 * conf)

        return d_xyz, scales, quat, opacity, color
