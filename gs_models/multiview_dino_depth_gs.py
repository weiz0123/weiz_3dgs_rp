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
    """
    c2w: [B,4,4]
    returns w2c: [B,4,4]
    """
    return torch.inverse(c2w)


def scale_intrinsics_batch(K: torch.Tensor, src_hw, dst_hw):
    """
    K: [B,3,3]
    """
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
    returns X_cam: [B,3,H,W]
    """
    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    pix = make_pixel_grid(B, H, W, device, dtype)  # [B,3,H,W]
    pix = pix.view(B, 3, -1)                        # [B,3,HW]

    Kinv = torch.inverse(K)                         # [B,3,3]
    rays = Kinv @ pix                               # [B,3,HW]
    X = rays * depth.view(B, 1, -1)
    return X.view(B, 3, H, W)


def cam_to_world_grid(X_cam: torch.Tensor, c2w: torch.Tensor):
    """
    X_cam: [B,3,H,W]
    c2w:   [B,4,4]
    returns X_world: [B,3,H,W]
    """
    B, _, H, W = X_cam.shape
    X_flat = X_cam.view(B, 3, -1)
    ones = torch.ones(B, 1, H * W, device=X_cam.device, dtype=X_cam.dtype)
    Xh = torch.cat([X_flat, ones], dim=1)          # [B,4,HW]
    Xw = c2w @ Xh                                  # [B,4,HW]
    return Xw[:, :3].view(B, 3, H, W)


def world_to_cam_grid(X_world: torch.Tensor, w2c: torch.Tensor):
    """
    X_world: [B,3,H,W]
    w2c:     [B,4,4]
    returns X_cam: [B,3,H,W]
    """
    B, _, H, W = X_world.shape
    X_flat = X_world.view(B, 3, -1)
    ones = torch.ones(B, 1, H * W, device=X_world.device, dtype=X_world.dtype)
    Xh = torch.cat([X_flat, ones], dim=1)
    Xc = w2c @ Xh
    return Xc[:, :3].view(B, 3, H, W)


def project_points_grid(X_cam: torch.Tensor, K: torch.Tensor, eps=1e-6):
    """
    X_cam: [B,3,H,W]
    K:     [B,3,3]
    returns uv: [B,H,W,2] in pixels
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
    uv: [B,H,W,2] in pixels
    returns grid for grid_sample: [B,H,W,2] in [-1,1]
    """
    u = uv[..., 0]
    v = uv[..., 1]

    gx = 2.0 * (u / max(W - 1, 1)) - 1.0
    gy = 2.0 * (v / max(H - 1, 1)) - 1.0
    return torch.stack([gx, gy], dim=-1)


def warp_feature_to_ref_plane(
    src_feat: torch.Tensor,
    depth_plane: torch.Tensor,
    K_ref: torch.Tensor,
    c2w_ref: torch.Tensor,
    K_src: torch.Tensor,
    c2w_src: torch.Tensor,
):
    """
    Warp src_feat into reference view under a fixed reference depth plane.

    src_feat:    [B,C,H,W]
    depth_plane: [B,1,H,W] or scalar expanded
    returns:
        warped_src_feat [B,C,H,W]
        valid_mask      [B,1,H,W]
    """
    B, C, H, W = src_feat.shape

    # ref pixels + hypothetical depth -> ref camera 3D
    X_ref_cam = unproject_depth(depth_plane, K_ref)              # [B,3,H,W]

    # ref cam -> world -> src cam
    X_world = cam_to_world_grid(X_ref_cam, c2w_ref)
    w2c_src = invert_pose(c2w_src)
    X_src_cam = world_to_cam_grid(X_world, w2c_src)

    # project into source
    uv_src = project_points_grid(X_src_cam, K_src)               # [B,H,W,2]
    grid = uv_to_grid(uv_src, H, W)

    warped = F.grid_sample(
        src_feat, grid, mode="bilinear",
        padding_mode="zeros", align_corners=True
    )

    z = X_src_cam[:, 2:3]
    valid = (
        (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) &
        (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0) &
        (z[:, 0] > 1e-6)
    ).unsqueeze(1).float()

    return warped, valid


# =========================================================
# DINOv2 dense backbone
# =========================================================
class DinoV2DenseEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", freeze=True):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name, output_hidden_states=True)
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
        x: [B,3,H,W] in [0,1]
        returns:
            feat_map [B,C,H,W]
            cls      [B,C]
        """
        B, _, H, W = x.shape

        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x, output_hidden_states=True)

        hs = out.hidden_states[-1]   # [B, 1+N, C]
        cls = hs[:, 0]               # [B, C]
        patch = hs[:, 1:]            # [B, N, C]

        B2, N, C = patch.shape

        gh = H // self.patch_size
        gw = W // self.patch_size

        if gh * gw != N:
            raise ValueError(
                f"DINO token reshape mismatch: H={H}, W={W}, "
                f"patch={self.patch_size}, gh={gh}, gw={gw}, "
                f"gh*gw={gh*gw}, but N={N}"
            )

        feat = patch.view(B2, gh, gw, C).permute(0, 3, 1, 2).contiguous()

        return feat, cls
# =========================================================
# Cost volume builder
# =========================================================

class PlaneSweepCostVolume(nn.Module):
    def __init__(self, num_depth_bins=32, depth_min=0.5, depth_max=15.0):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max

    def get_depth_planes(self, B, H, W, device, dtype):
        # inverse-depth sampling is usually better
        inv_min = 1.0 / self.depth_max
        inv_max = 1.0 / self.depth_min
        inv = torch.linspace(inv_min, inv_max, self.num_depth_bins, device=device, dtype=dtype)
        depth = 1.0 / inv
        depth = depth.view(1, self.num_depth_bins, 1, 1).expand(B, self.num_depth_bins, H, W)
        return depth

    def forward(self, ref_feat, src_feats, K_ref, c2w_ref, K_srcs, c2w_srcs):
        """
        ref_feat:  [B,C,H,W]
        src_feats: [B,Vs,C,H,W]
        K_srcs:    [B,Vs,3,3]
        c2w_srcs:  [B,Vs,4,4]

        returns:
            cost_volume   [B,D,H,W]
            src_agg_feat  [B,C,H,W]  # feature warped at estimated depth later, placeholder now
            depth_planes  [B,D,H,W]
        """
        B, C, H, W = ref_feat.shape
        Vs = src_feats.shape[1]
        device = ref_feat.device
        dtype = ref_feat.dtype

        depth_planes = self.get_depth_planes(B, H, W, device, dtype)   # [B,D,H,W]

        ref_feat_n = F.normalize(ref_feat, dim=1)
        cost_slices = []

        for d in range(self.num_depth_bins):
            plane = depth_planes[:, d:d+1]  # [B,1,H,W]

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
                sim = (ref_feat_n * warped_src_n).sum(dim=1, keepdim=True)  # [B,1,H,W]
                sim = sim * valid
                sims.append(sim)

            sim_stack = torch.stack(sims, dim=1)    # [B,Vs,1,H,W]
            sim_mean = sim_stack.mean(dim=1)        # [B,1,H,W]
            cost_slices.append(sim_mean)

        cost_volume = torch.cat(cost_slices, dim=1)  # [B,D,H,W]
        return cost_volume, depth_planes


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
      ref feature  [B,C,H,W]
    Predicts:
      depth     [B,1,H,W]
      conf      [B,1,H,W]
      depth_pdf [B,D,H,W]
      fused     [B,F,H,W]  # feature for Gaussian head
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

    def forward(self, ref_feat, cost_volume, depth_planes):
        ref_p = self.ref_proj(ref_feat)
        cost_p = self.cost_proj(cost_volume)

        x = torch.cat([ref_p, cost_p], dim=1)
        fused = self.fuser(x)

        logits = self.depth_logits(fused)          # [B,D,H,W]
        pdf = torch.softmax(logits, dim=1)

        depth = (pdf * depth_planes).sum(dim=1, keepdim=True)   # expectation
        conf = pdf.max(dim=1, keepdim=True).values              # simple confidence

        conf = torch.sigmoid(4.0 * (conf - 0.5))                # sharpen a bit
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

        # ---- tiny innovation: confidence-aware Gaussian shaping ----
        # high conf -> tighter + more opaque
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
      Ks     [B,V,3,3]    in pixel coordinates for original image
      c2ws   [B,V,4,4]
      ref_idx: int
      emit_stride: downsample pixels when emitting gaussians

    Returns:
      dict with depth/conf + per-pixel Gaussian params
    """
    def __init__(
        self,
        dino_name="facebook/dinov2-base",
        freeze_dino=True,
        num_depth_bins=32,
        depth_min=0.5,
        depth_max=15.0,
        feat_reduce_dim=128,
    ):
        super().__init__()

        self.encoder = DinoV2DenseEncoder(dino_name, freeze=freeze_dino)
        C = self.encoder.hidden_dim

        self.feat_reduce = nn.Conv2d(C, feat_reduce_dim, 1)
        self.cost_volume = PlaneSweepCostVolume(
            num_depth_bins=num_depth_bins,
            depth_min=depth_min,
            depth_max=depth_max,
        )
        self.depth_head = DepthConfidenceHead(
            feat_dim=feat_reduce_dim,
            num_depth_bins=num_depth_bins,
            hidden=128
        )
        self.gaussian_head = GaussianHead(
            ref_feat_dim=feat_reduce_dim,
            fused_dim=128,
            hidden=128
        )

    def forward(self, imgs, Ks, c2ws, ref_idx=0, emit_stride=2):
        B, V, _, H_img, W_img = imgs.shape
        device = imgs.device

        # ---------------------------------------
        # dense features per view
        # ---------------------------------------
        feats = []
        for v in range(V):
            f, _ = self.encoder(imgs[:, v])          # [B,C,H,W]
            f = self.feat_reduce(f)                  # [B,Cr,H,W]
            feats.append(f)

        feat_stack = torch.stack(feats, dim=1)       # [B,V,Cr,H,W]
        _, _, Cr, Hf, Wf = feat_stack.shape

        # intrinsics to feature resolution
        Ks_f = scale_intrinsics_batch(
            Ks.view(B * V, 3, 3),
            src_hw=(H_img, W_img),
            dst_hw=(Hf, Wf)
        ).view(B, V, 3, 3)

        # split ref/src
        ref_feat = feat_stack[:, ref_idx]            # [B,Cr,Hf,Wf]
        ref_img = F.interpolate(imgs[:, ref_idx], size=(Hf, Wf), mode="bilinear", align_corners=False)
        K_ref = Ks_f[:, ref_idx]
        c2w_ref = c2ws[:, ref_idx]

        src_indices = [i for i in range(V) if i != ref_idx]
        src_feats = feat_stack[:, src_indices]       # [B,Vs,Cr,Hf,Wf]
        K_srcs = Ks_f[:, src_indices]
        c2w_srcs = c2ws[:, src_indices]

        # ---------------------------------------
        # depth via cost volume
        # ---------------------------------------
        cost_vol, depth_planes = self.cost_volume(
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
            depth_planes=depth_planes,
        )

        # ---------------------------------------
        # aggregate source features at estimated depth
        # ---------------------------------------
        src_agg_feat, valid = aggregate_src_features_at_depth(
            src_feats=src_feats,
            depth=depth,
            K_ref=K_ref,
            c2w_ref=c2w_ref,
            K_srcs=K_srcs,
            c2w_srcs=c2w_srcs,
        )

        # ---------------------------------------
        # gaussian params per pixel
        # ---------------------------------------
        d_xyz, scales, quat, opacity, color = self.gaussian_head(
            ref_feat=ref_feat,
            src_agg_feat=src_agg_feat,
            fused_feat=fused_feat,
            depth=depth,
            conf=conf,
            rgb=ref_img,
        )

        # ---------------------------------------
        # unproject to 3D in ref camera and convert to world
        # ---------------------------------------
        X_ref_cam = unproject_depth(depth, K_ref)   # [B,3,Hf,Wf]
        X_ref_cam = X_ref_cam + d_xyz
        X_world = cam_to_world_grid(X_ref_cam, c2w_ref)   # [B,3,Hf,Wf]

        # ---------------------------------------
        # emit gaussians on strided grid
        # ---------------------------------------
        X_world = X_world[:, :, ::emit_stride, ::emit_stride]
        scales = scales[:, :, ::emit_stride, ::emit_stride]
        quat = quat[:, :, ::emit_stride, ::emit_stride]
        opacity = opacity[:, :, ::emit_stride, ::emit_stride]
        color = color[:, :, ::emit_stride, ::emit_stride]
        conf = conf[:, :, ::emit_stride, ::emit_stride]

        B2, _, Hg, Wg = X_world.shape
        M = Hg * Wg

        means3D = X_world.view(B2, 3, M).permute(0, 2, 1).contiguous()
        scales = scales.view(B2, 3, M).permute(0, 2, 1).contiguous()
        rotations = quat.view(B2, 4, M).permute(0, 2, 1).contiguous()
        opacities = opacity.view(B2, 1, M).permute(0, 2, 1).contiguous()
        colors = color.view(B2, 3, M).permute(0, 2, 1).contiguous()
        conf_flat = conf.view(B2, 1, M).permute(0, 2, 1).contiguous()

        return {
            "depth": depth,                 # [B,1,Hf,Wf]
            "confidence": conf,             # [B,1,Hf,Wf] before flattening stride
            "depth_pdf": depth_pdf,         # [B,D,Hf,Wf]
            "means3D": means3D,             # [B,M,3]
            "scales": scales,               # [B,M,3]
            "rotations": rotations,         # [B,M,4]
            "opacities": opacities,         # [B,M,1]
            "colors": colors,               # [B,M,3]
            "gaussian_conf": conf_flat,     # [B,M,1]
            "feat_h": Hf,
            "feat_w": Wf,
            "emit_stride": emit_stride,
            "ref_idx": ref_idx,
        }