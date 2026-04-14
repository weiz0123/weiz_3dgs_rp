import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, d=1):
        super().__init__()
        groups = 8 if cout % 8 == 0 else 1
        self.block = nn.Sequential(
            # Dilation 'd' exponentially increases the receptive field
            nn.Conv2d(cin, cout, k, s, padding=p, dilation=d),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)

class GaussianHead(nn.Module):
    """
    Advanced 3DGS Head predicting MULTIPLE Gaussians per pixel, 
    with a dilated receptive field for global context.
    """
    def __init__(
        self,
        ref_feat_dim,
        mv_feat_dim,
        fused_dim,
        hidden=256,        
        sh_degree=3,
        num_surfaces=2,    # SOTA trick: Predict 2 Gaussians per pixel (Foreground/Background)
    ):
        super().__init__()

        self.sh_degree = sh_degree
        self.sh_coeff_dim = (sh_degree + 1) ** 2
        self.sh_out_dim = 3 * self.sh_coeff_dim
        self.num_surfaces = num_surfaces

        in_dim = ref_feat_dim + mv_feat_dim + fused_dim + 1 + 1
        
        # Calculate parameters needed for ONE Gaussian (No semantics here)
        self.per_surface_dim = 3 + 3 + 4 + 1 + self.sh_out_dim
        
        # Total output channels needed for ALL Gaussians at this pixel
        out_dim = self.per_surface_dim * self.num_surfaces

        # Upgraded Backbone: Dilated Convolutions
        self.net = nn.Sequential(
            ConvBlock(in_dim, hidden, p=1, d=1),
            ConvBlock(hidden, hidden, p=2, d=2), # Dilation 2
            ConvBlock(hidden, hidden, p=4, d=4), # Dilation 4 (sees much further)
            ConvBlock(hidden, hidden, p=1, d=1),
        )
        self.upsample_refine = nn.Sequential(
            ConvBlock(hidden, hidden, p=1, d=1),
            ConvBlock(hidden, hidden, p=1, d=1),
        )

        self.out = nn.Conv2d(hidden, out_dim, 1)

    def forward(self, ref_feat, mv_feat, fused_feat, depth, conf, output_size=None):
        x = torch.cat([ref_feat, mv_feat, fused_feat, depth, conf], dim=1)
        h = self.net(x)

        if output_size is not None and h.shape[-2:] != output_size:
            h = F.interpolate(h, size=output_size, mode="bilinear", align_corners=False)
            h = self.upsample_refine(h)
            conf = F.interpolate(conf, size=output_size, mode="bilinear", align_corners=False)

        raw = self.out(h)

        batch_size, _, height, width = raw.shape

        # Reshape to separate the multiple surfaces
        # Shape becomes: [Batch, Num_Surfaces, Features_Per_Surface, Height, Width]
        raw = raw.view(batch_size, self.num_surfaces, self.per_surface_dim, height, width)

        cursor = 0

        # 1. Position Offsets (XYZ)
        d_xyz = 0.05 * torch.tanh(raw[:, :, cursor:cursor + 3])
        cursor += 3

        # 2. Scales
        s_raw = raw[:, :, cursor:cursor + 3]
        cursor += 3

        # 3. Rotations (Quaternions)
        q_raw = raw[:, :, cursor:cursor + 4]
        cursor += 4

        # 4. Opacity
        a_raw = raw[:, :, cursor:cursor + 1]
        cursor += 1

        # 5. Spherical Harmonics (Color)
        sh_raw = raw[:, :, cursor:cursor + self.sh_out_dim]

        # --- Apply Activations & Constraints ---
        quat = F.normalize(q_raw, dim=2, eps=1e-6)
        
        # Expand confidence to match the num_surfaces dimension
        # Shape goes from [B, 1, H, W] -> [B, 1, 1, H, W] to broadcast across surfaces
        conf_expanded = conf.unsqueeze(1) 
        
        base_scales = 0.002 + 0.03 * torch.sigmoid(s_raw)
        opacity_raw = torch.sigmoid(a_raw)

        # Confidence shaping applied to all surfaces
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
