import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


# =========================================================
# Geometry utils
# =========================================================

def make_pixel_grid(B, H, W, device, dtype=torch.float32):
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=0)   # [3,H,W]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) # [B,3,H,W]
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

    pix = make_pixel_grid(B, H, W, device, dtype)   # [B,3,H,W]
    pix = pix.reshape(B, 3, -1)                     # [B,3,HW]

    Kinv = torch.inverse(K)                         # [B,3,3]
    rays = Kinv @ pix                               # [B,3,HW]
    X = rays * depth.reshape(B, 1, -1)              # [B,3,HW]

    return X.reshape(B, 3, H, W)


def cam_to_world_grid(X_cam: torch.Tensor, c2w: torch.Tensor):
    """
    X_cam: [B,3,H,W]
    c2w:   [B,4,4]
    """
    B, _, H, W = X_cam.shape

    X_flat = X_cam.reshape(B, 3, -1)
    ones = torch.ones(B, 1, H * W, device=X_cam.device, dtype=X_cam.dtype)
    Xh = torch.cat([X_flat, ones], dim=1)          # [B,4,HW]

    Xw = c2w @ Xh                                  # [B,4,HW]
    return Xw[:, :3].reshape(B, 3, H, W)


def world_to_cam_grid(X_world: torch.Tensor, w2c: torch.Tensor):
    """
    X_world: [B,3,H,W]
    w2c:     [B,4,4]
    """
    B, _, H, W = X_world.shape

    X_flat = X_world.reshape(B, 3, -1)
    ones = torch.ones(B, 1, H * W, device=X_world.device, dtype=X_world.dtype)
    Xh = torch.cat([X_flat, ones], dim=1)          # [B,4,HW]

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

    return torch.stack([u, v], dim=-1)   # [B,H,W,2]


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

    X_ref_cam = unproject_depth(depth_plane, K_ref)       # [B,3,H,W]
    X_world = cam_to_world_grid(X_ref_cam, c2w_ref)       # [B,3,H,W]

    w2c_src = invert_pose(c2w_src)
    X_src_cam = world_to_cam_grid(X_world, w2c_src)       # [B,3,H,W]

    uv_src = project_points_grid(X_src_cam, K_src)        # [B,H,W,2]
    grid = uv_to_grid(uv_src, H, W)                       # [B,H,W,2]

    warped = F.grid_sample(
        src_feat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    z = X_src_cam[:, 2:3]                                 # [B,1,H,W]

    valid = (
        (grid[..., 0] >= -1.0) &
        (grid[..., 0] <= 1.0) &
        (grid[..., 1] >= -1.0) &
        (grid[..., 1] <= 1.0) &
        (z[:, 0] > 1e-6)
    ).unsqueeze(1).float()

    return warped, valid


# =========================================================
# DINOv2 encoder
# =========================================================

class DinoV2DenseEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", freeze=True):
        super().__init__()

        self.backbone = Dinov2Model.from_pretrained(
            model_name,
            output_hidden_states=True
        )

        self.hidden_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False
        )

    def forward(self, x):
        """
        x: [B,3,H,W]
        return:
            feat: [B,C,H//patch,W//patch]
            cls:  [B,C]
        """
        B, _, H, W = x.shape

        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x, output_hidden_states=True)

        hs = out.hidden_states[-1]   # [B,1+N,C]
        cls = hs[:, 0]
        patch = hs[:, 1:]

        B2, N, C = patch.shape
        gh = H // self.patch_size
        gw = W // self.patch_size

        if gh * gw != N:
            raise ValueError(
                f"DINO reshape mismatch H={H} W={W} gh={gh} gw={gw} N={N}"
            )

        feat = patch.reshape(B2, gh, gw, C).permute(0, 3, 1, 2).contiguous()
        return feat, cls


# =========================================================
# Cost volume
# =========================================================

class PlaneSweepCostVolume(nn.Module):
    """
    Full-resolution plane sweep over upsampled features.
    """

    def __init__(self, num_depth_bins=128, depth_min=0.5, depth_max=15.0):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max

    def get_depth_values(self, device, dtype):
        """
        Returns inverse-depth sampled depth bins: [D]
        """
        inv_min = 1.0 / self.depth_max
        inv_max = 1.0 / self.depth_min

        inv = torch.linspace(
            inv_min,
            inv_max,
            self.num_depth_bins,
            device=device,
            dtype=dtype
        )

        depth = 1.0 / inv   # [D]
        return depth

    def forward(self, ref_feat, src_feats, K_ref, c2w_ref, K_srcs, c2w_srcs):
        """
        ref_feat:  [B,C,H,W]
        src_feats: [B,Vs,C,H,W]
        K_ref:     [B,3,3]
        c2w_ref:   [B,4,4]
        K_srcs:    [B,Vs,3,3]
        c2w_srcs:  [B,Vs,4,4]

        returns:
            cost_volume:  [B,D,H,W]
            depth_values: [D]
        """
        B, C, H, W = ref_feat.shape
        Vs = src_feats.shape[1]

        device = ref_feat.device
        dtype = ref_feat.dtype

        depth_values = self.get_depth_values(device, dtype)   # [D]
        ref_feat_n = F.normalize(ref_feat, dim=1)

        cost_slices = []

        for d in range(self.num_depth_bins):
            plane_val = depth_values[d]
            plane = plane_val.view(1, 1, 1, 1).expand(B, 1, H, W)

            sims = []
            for s in range(Vs):
                warped_src, valid = warp_feature_to_ref_plane(
                    src_feat=src_feats[:, s],
                    depth_plane=plane,
                    K_ref=K_ref,
                    c2w_ref=c2w_ref,
                    K_src=K_srcs[:, s],
                    c2w_src=c2w_srcs[:, s],
                )

                warped_src_n = F.normalize(warped_src, dim=1)
                sim = (ref_feat_n * warped_src_n).sum(dim=1, keepdim=True)
                sim = sim * valid
                sims.append(sim)

            sim_stack = torch.stack(sims, dim=1)   # [B,Vs,1,H,W]
            sim_mean = sim_stack.mean(dim=1)       # [B,1,H,W]
            cost_slices.append(sim_mean)

        cost_volume = torch.cat(cost_slices, dim=1)  # [B,D,H,W]
        return cost_volume, depth_values


# =========================================================
# Depth/confidence head
# =========================================================

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
            nn.Conv2d(hidden, 1, 1)
        )

    def forward(self, ref_feat, cost_volume, depth_values):
        """
        depth_values: [D]
        """
        ref_p = self.ref_proj(ref_feat)
        cost_p = self.cost_proj(cost_volume)

        x = torch.cat([ref_p, cost_p], dim=1)
        fused = self.fuser(x)

        logits = self.depth_logits(fused)               # [B,D,H,W]
        pdf = torch.softmax(logits, dim=1)

        depth_values = depth_values.view(1, -1, 1, 1)   # [1,D,1,1]
        depth = (pdf * depth_values).sum(dim=1, keepdim=True)

        conf = pdf.max(dim=1, keepdim=True).values
        conf = torch.sigmoid(4.0 * (conf - 0.5))

        return depth, conf, pdf, fused


# =========================================================
# Feature aggregation at expected depth
# =========================================================

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

    warped_all = torch.stack(warped_all, dim=1)   # [B,Vs,C,H,W]
    valid_all = torch.stack(valid_all, dim=1)     # [B,Vs,1,H,W]

    denom = valid_all.sum(dim=1).clamp(min=1.0)
    agg = (warped_all * valid_all).sum(dim=1) / denom
    valid = (valid_all.sum(dim=1) > 0).float()

    return agg, valid


# =========================================================
# Gaussian head
# =========================================================

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

        # confidence-aware shaping
        scales = base_scales * (1.25 - 0.75 * conf)
        opacity = opacity_raw * (0.25 + 0.75 * conf)

        return d_xyz, scales, quat, opacity, color


# =========================================================
# Main model
# =========================================================

class MultiViewDinoDepthToGaussians(nn.Module):
    """
    Inputs:
      imgs   [B,V,3,H_img,W_img]
      Ks     [B,V,3,3]    in original image pixel coordinates
      c2ws   [B,V,4,4]
      ref_idx: int
      emit_stride: downsample pixels when emitting gaussians

    Full-resolution cost-volume version:
      - DINO extracts low-res features
      - reduced features are upsampled to full image resolution
      - cost volume is constructed at full image resolution
      - depth is predicted at full image resolution
    """
    def __init__(
        self,
        dino_name="facebook/dinov2-base",
        freeze_dino=True,
        num_depth_bins=128,
        depth_min=0.5,
        depth_max=15.0,
        feat_reduce_dim=128,
        use_full_res_cost_volume=True,
    ):
        super().__init__()

        self.encoder = DinoV2DenseEncoder(dino_name, freeze=freeze_dino)
        C = self.encoder.hidden_dim
        self.use_full_res_cost_volume = use_full_res_cost_volume

        self.feat_reduce = nn.Conv2d(C, feat_reduce_dim, 1)

        self.cost_volume = PlaneSweepCostVolume(
            num_depth_bins=num_depth_bins,
            depth_min=depth_min,
            depth_max=depth_max,
        )

        self.depth_head = DepthConfidenceHead(
            feat_dim=feat_reduce_dim,
            num_depth_bins=num_depth_bins,
            hidden=128,
        )

        self.gaussian_head = GaussianHead(
            ref_feat_dim=feat_reduce_dim,
            fused_dim=128,
            hidden=128,
        )

    def forward(self, imgs, Ks, c2ws, ref_idx=0, emit_stride=2):
        B, V, _, H_img, W_img = imgs.shape

        # -------------------------------------------------
        # 1) DINO low-res features per view
        # -------------------------------------------------
        feats = []
        for v in range(V):
            f, _ = self.encoder(imgs[:, v])   # [B,C,Hd,Wd]
            f = self.feat_reduce(f)           # [B,Cr,Hd,Wd]
            feats.append(f)

        feat_stack_low = torch.stack(feats, dim=1)   # [B,V,Cr,Hd,Wd]
        _, _, Cr, Hd, Wd = feat_stack_low.shape

        # -------------------------------------------------
        # 2) Build features used by cost volume
        # -------------------------------------------------
        if self.use_full_res_cost_volume:
            feat_stack = F.interpolate(
                feat_stack_low.reshape(B * V, Cr, Hd, Wd),
                size=(H_img, W_img),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, V, Cr, H_img, W_img)

            Ks_used = Ks
            Hcv, Wcv = H_img, W_img
        else:
            feat_stack = feat_stack_low
            Ks_used = scale_intrinsics_batch(
                Ks.reshape(B * V, 3, 3),
                src_hw=(H_img, W_img),
                dst_hw=(Hd, Wd),
            ).reshape(B, V, 3, 3)

            Hcv, Wcv = Hd, Wd

        # -------------------------------------------------
        # 3) Split ref/src
        # -------------------------------------------------
        ref_feat = feat_stack[:, ref_idx]                  # [B,Cr,Hcv,Wcv]

        if Hcv == H_img and Wcv == W_img:
            ref_img = imgs[:, ref_idx]
        else:
            ref_img = F.interpolate(
                imgs[:, ref_idx],
                size=(Hcv, Wcv),
                mode="bilinear",
                align_corners=False,
            )

        K_ref = Ks_used[:, ref_idx]
        c2w_ref = c2ws[:, ref_idx]

        src_indices = [i for i in range(V) if i != ref_idx]
        src_feats = feat_stack[:, src_indices]             # [B,Vs,Cr,Hcv,Wcv]
        K_srcs = Ks_used[:, src_indices]
        c2w_srcs = c2ws[:, src_indices]

        # -------------------------------------------------
        # 4) Full-resolution cost volume + depth
        # -------------------------------------------------
        cost_vol, depth_values = self.cost_volume(
            ref_feat=ref_feat,
            src_feats=src_feats,
            K_ref=K_ref,
            c2w_ref=c2w_ref,
            K_srcs=K_srcs,
            c2w_srcs=c2w_srcs,
        )

        depth, conf, depth_pdf, fused_feat = self.depth_head(
            ref_feat=ref_feat,
            cost_volume=cost_vol,
            depth_values=depth_values,
        )

        # keep full-res confidence for return
        conf_full = conf
        depth_full = depth

        # -------------------------------------------------
        # 5) Aggregate source features at predicted depth
        # -------------------------------------------------
        src_agg_feat, valid = aggregate_src_features_at_depth(
            src_feats=src_feats,
            depth=depth_full,
            K_ref=K_ref,
            c2w_ref=c2w_ref,
            K_srcs=K_srcs,
            c2w_srcs=c2w_srcs,
        )

        # -------------------------------------------------
        # 6) Gaussian params per pixel
        # -------------------------------------------------
        d_xyz, scales_full, quat_full, opacity_full, color_full = self.gaussian_head(
            ref_feat=ref_feat,
            src_agg_feat=src_agg_feat,
            fused_feat=fused_feat,
            depth=depth_full,
            conf=conf_full,
            rgb=ref_img,
        )

        # -------------------------------------------------
        # 7) Unproject full-res depth to world coordinates
        # -------------------------------------------------
        X_ref_cam = unproject_depth(depth_full, K_ref)   # [B,3,Hcv,Wcv]
        X_ref_cam = X_ref_cam + d_xyz
        X_world_full = cam_to_world_grid(X_ref_cam, c2w_ref)

        # -------------------------------------------------
        # 8) Emit gaussians on strided full-res grid
        # -------------------------------------------------
        X_world = X_world_full[:, :, ::emit_stride, ::emit_stride]
        scales = scales_full[:, :, ::emit_stride, ::emit_stride]
        quat = quat_full[:, :, ::emit_stride, ::emit_stride]
        opacity = opacity_full[:, :, ::emit_stride, ::emit_stride]
        color = color_full[:, :, ::emit_stride, ::emit_stride]
        conf_emit = conf_full[:, :, ::emit_stride, ::emit_stride]

        B2, _, Hg, Wg = X_world.shape
        M = Hg * Wg

        means3D = X_world.reshape(B2, 3, M).permute(0, 2, 1).contiguous()
        scales = scales.reshape(B2, 3, M).permute(0, 2, 1).contiguous()
        rotations = quat.reshape(B2, 4, M).permute(0, 2, 1).contiguous()
        opacities = opacity.reshape(B2, 1, M).permute(0, 2, 1).contiguous()
        colors = color.reshape(B2, 3, M).permute(0, 2, 1).contiguous()
        conf_flat = conf_emit.reshape(B2, 1, M).permute(0, 2, 1).contiguous()

        return {
            "depth": depth_full,             # [B,1,H_img,W_img] if full-res mode
            "confidence": conf_full,         # [B,1,H_img,W_img]
            "depth_pdf": depth_pdf,          # [B,D,H_img,W_img] if full-res mode
            "means3D": means3D,              # [B,M,3]
            "scales": scales,                # [B,M,3]
            "rotations": rotations,          # [B,M,4]
            "opacities": opacities,          # [B,M,1]
            "colors": colors,                # [B,M,3]
            "gaussian_conf": conf_flat,      # [B,M,1]
            "feat_h": Hcv,
            "feat_w": Wcv,
            "dino_h": Hd,
            "dino_w": Wd,
            "emit_stride": emit_stride,
            "ref_idx": ref_idx,
        }