import torch
import torch.nn as nn
import torch.nn.functional as F

from .mvv3_heads import ConvBlock


def _build_2d_sincos_pos_embed(h, w, dim, device, dtype):
    if dim % 4 != 0:
        raise ValueError(f"Expected embedding dim divisible by 4, got {dim}")

    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim // 4 - 1, 1)))

    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)

    pos = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
    return pos.to(dtype=dtype)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = ConvBlock(dim, dim)
        self.conv2 = ConvBlock(dim, dim)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class FeatureFusionBlock(nn.Module):
    def __init__(self, dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.skip = nn.Conv2d(dim, dim, 1)
        self.res1 = ResidualConvUnit(dim)
        self.res2 = ResidualConvUnit(dim)

    def forward(self, x, skip=None):
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = x + self.skip(skip)
        x = self.res1(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.res2(x)
        return x


class DPTDepthHead(nn.Module):
    def __init__(self, feat_dim, hidden=128, depth_min=0.5, depth_max=15.0):
        super().__init__()
        self.depth_min = depth_min
        self.depth_max = depth_max

        self.level1 = ConvBlock(feat_dim, hidden)
        self.level2 = ConvBlock(hidden, hidden, s=2)
        self.level3 = ConvBlock(hidden, hidden, s=2)
        self.level4 = ConvBlock(hidden, hidden, s=2)

        self.adapt4 = nn.Conv2d(hidden, hidden, 1)
        self.fuse3 = FeatureFusionBlock(hidden, upsample=True)
        self.fuse2 = FeatureFusionBlock(hidden, upsample=True)
        self.fuse1 = FeatureFusionBlock(hidden, upsample=False)

        self.out_proj = ConvBlock(hidden, hidden)
        self.depth_head = nn.Sequential(
            ConvBlock(hidden, hidden),
            nn.Conv2d(hidden, 1, 1),
        )
        self.conf_head = nn.Sequential(
            ConvBlock(hidden, hidden),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, ref_feat, target_hw):
        l1 = self.level1(ref_feat)
        l2 = self.level2(l1)
        l3 = self.level3(l2)
        l4 = self.level4(l3)

        x = self.adapt4(l4)
        x = self.fuse3(x, l3)
        x = self.fuse2(x, l2)
        x = self.fuse1(x, l1)

        fused = self.out_proj(
            F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        )

        depth = self.depth_min + torch.sigmoid(self.depth_head(fused)) * (
            self.depth_max - self.depth_min
        )
        conf = torch.sigmoid(self.conf_head(fused))

        return depth, conf, fused


class MiniVGGTDepthModule(nn.Module):
    """
    DINO features in, mini-VGGT depth/context out.
    """

    def __init__(
        self,
        in_dim,
        feat_dim=128,
        depth_min=0.5,
        depth_max=15.0,
        transformer_depth=4,
        transformer_heads=8,
        mlp_ratio=4.0,
        max_views=8,
        dropout=0.0,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.in_proj = nn.Conv2d(in_dim, feat_dim, 1)
        self.view_embed = nn.Embedding(max_views, feat_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=feat_dim,
                    num_heads=transformer_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(transformer_depth)
            ]
        )
        self.norm = nn.LayerNorm(feat_dim)
        self.depth_head = DPTDepthHead(
            feat_dim=feat_dim,
            hidden=feat_dim,
            depth_min=depth_min,
            depth_max=depth_max,
        )

    def forward(self, dino_feat_stack, ref_idx, target_hw):
        b, v, _, h, w = dino_feat_stack.shape
        if v > self.view_embed.num_embeddings:
            raise ValueError(
                f"MiniVGGTDepthModule supports at most {self.view_embed.num_embeddings} views, got {v}"
            )

        feat_stack_low = self.in_proj(
            dino_feat_stack.reshape(b * v, dino_feat_stack.shape[2], h, w)
        ).reshape(b, v, self.feat_dim, h, w)

        tokens = feat_stack_low.permute(0, 1, 3, 4, 2).reshape(b, v, h * w, self.feat_dim)

        pos = _build_2d_sincos_pos_embed(
            h, w, self.feat_dim, feat_stack_low.device, feat_stack_low.dtype
        ).view(1, 1, h * w, self.feat_dim)

        view_ids = torch.arange(v, device=feat_stack_low.device)
        view_pos = self.view_embed(view_ids).view(1, v, 1, self.feat_dim).to(
            feat_stack_low.dtype
        )

        x = (tokens + pos + view_pos).reshape(b, v * h * w, self.feat_dim)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        fused_low = x.reshape(b, v, h, w, self.feat_dim).permute(0, 1, 4, 2, 3).contiguous()
        fused_full = F.interpolate(
            fused_low.reshape(b * v, self.feat_dim, h, w),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).reshape(b, v, self.feat_dim, target_hw[0], target_hw[1])

        ref_feat_low = fused_low[:, ref_idx]
        ref_feat_full = fused_full[:, ref_idx]
        mv_context_full = fused_full.mean(dim=1)

        depth, conf, depth_fused = self.depth_head(ref_feat_low, target_hw)

        return {
            "ref_feat_full": ref_feat_full,
            "mv_context_full": mv_context_full,
            "depth": depth,
            "confidence": conf,
            "fused_feat": depth_fused,
            "token_h": h,
            "token_w": w,
        }
