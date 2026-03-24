import torch
import torch.nn as nn
import torch.nn.functional as F

from .mvv2_geometry import warp_feature_to_ref_plane


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DepthConfidenceHead(nn.Module):
    """
    Takes:
      cost volume [B,D,H,W]
      ref feature [B,C,H,W]

    Predicts:
      depth     [B,1,H,W]
      conf      [B,1,H,W]
      depth_pdf [B,D,H,W]
      fused     [B,F,H,W]
    """

    def __init__(self, feat_dim, num_depth_bins, hidden=128):
        super().__init__()
        self.num_depth_bins = num_depth_bins

        self.ref_proj = nn.Conv2d(feat_dim, hidden, 1)
        self.cost_proj = nn.Conv2d(num_depth_bins, hidden, 3, padding=1)

        self.fuser = nn.Sequential(
            ConvBlock(hidden * 2, hidden),
            ConvBlock(hidden, hidden),
            ConvBlock(hidden, hidden),
        )

        self.depth_logits = nn.Conv2d(hidden, num_depth_bins, 1)

        self.conf_head = nn.Sequential(
            ConvBlock(hidden, hidden),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, ref_feat, cost_volume, depth_values):
        """
        depth_values: [D]
        """
        ref_p = self.ref_proj(ref_feat)
        cost_p = self.cost_proj(cost_volume)

        x = torch.cat([ref_p, cost_p], dim=1)
        fused = self.fuser(x)

        logits = self.depth_logits(fused)  # [B,D,H,W]
        pdf = torch.softmax(logits, dim=1)

        depth_values = depth_values.view(1, -1, 1, 1)  # [1,D,1,1]
        depth = (pdf * depth_values).sum(dim=1, keepdim=True)

        conf = pdf.max(dim=1, keepdim=True).values
        conf = torch.sigmoid(4.0 * (conf - 0.5))

        return depth, conf, pdf, fused


def aggregate_src_features_at_depth(
    src_feats, depth, K_ref, c2w_ref, K_srcs, c2w_srcs
):
    """
    Warp each src feature into ref view at predicted depth and average.

    src_feats: [B,Vs,C,H,W]
    depth:     [B,1,H,W]

    returns:
        agg_feat [B,C,H,W]
        valid    [B,1,H,W]
    """
    B, Vs, C, H, W = src_feats.shape
    warped_all = []
    valid_all = []

    for s in range(Vs):
        warped, valid = warp_feature_to_ref_plane(
            src_feat=src_feats[:, s],
            depth_plane=depth,
            K_ref=K_ref,
            c2w_ref=c2w_ref,
            K_src=K_srcs[:, s],
            c2w_src=c2w_srcs[:, s],
        )
        warped_all.append(warped)
        valid_all.append(valid)

    warped_all = torch.stack(warped_all, dim=1)  # [B,Vs,C,H,W]
    valid_all = torch.stack(valid_all, dim=1)  # [B,Vs,1,H,W]

    denom = valid_all.sum(dim=1).clamp(min=1.0)
    agg = (warped_all * valid_all).sum(dim=1) / denom
    valid = (valid_all.sum(dim=1) > 0).float()

    return agg, valid


class GaussianHead(nn.Module):
    """
    Predict Gaussian parameters per pixel.
    """

    def __init__(self, ref_feat_dim, fused_dim, hidden=128):
        super().__init__()
        in_dim = ref_feat_dim + ref_feat_dim + fused_dim + 1 + 1 + 3

        self.net = nn.Sequential(
            ConvBlock(in_dim, hidden),
            ConvBlock(hidden, hidden),
            ConvBlock(hidden, hidden),
        )

        # dx,dy,dz, sx,sy,sz, qx,qy,qz,qw, alpha, r,g,b
        self.out = nn.Conv2d(hidden, 14, 1)

    def forward(self, ref_feat, src_agg_feat, fused_feat, depth, conf, rgb):
        x = torch.cat([ref_feat, src_agg_feat, fused_feat, depth, conf, rgb], dim=1)
        h = self.net(x)
        raw = self.out(h)

        d_xyz = 0.05 * torch.tanh(raw[:, 0:3])
        s_raw = raw[:, 3:6]
        q_raw = raw[:, 6:10]
        a_raw = raw[:, 10:11]
        c_raw = raw[:, 11:14]

        quat = F.normalize(q_raw, dim=1, eps=1e-6)
        base_scales = 0.002 + 0.03 * torch.sigmoid(s_raw)
        opacity_raw = torch.sigmoid(a_raw)
        color = torch.sigmoid(c_raw)

        # Confidence-aware shaping keeps reliable points tighter and more visible.
        scales = base_scales * (1.25 - 0.75 * conf)
        opacity = opacity_raw * (0.25 + 0.75 * conf)

        return d_xyz, scales, quat, opacity, color
