import torch
import torch.nn as nn
import torch.nn.functional as F

from .mvv2_cost_volume import PlaneSweepCostVolume
from .mvv2_encoder import DinoV2DenseEncoder
from .mvv2_geometry import (
    cam_to_world_grid,
    scale_intrinsics_batch,
    unproject_depth,
)
from .mvv2_heads import (
    DepthConfidenceHead,
    GaussianHead,
    aggregate_src_features_at_depth,
)


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

        # 1) DINO low-res features per view
        feats = []
        for v in range(V):
            f, _ = self.encoder(imgs[:, v])  # [B,C,Hd,Wd]
            f = self.feat_reduce(f)  # [B,Cr,Hd,Wd]
            feats.append(f)

        feat_stack_low = torch.stack(feats, dim=1)  # [B,V,Cr,Hd,Wd]
        _, _, Cr, Hd, Wd = feat_stack_low.shape

        # 2) Build features used by cost volume
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

        # 3) Split ref/src
        ref_feat = feat_stack[:, ref_idx]  # [B,Cr,Hcv,Wcv]

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
        src_feats = feat_stack[:, src_indices]  # [B,Vs,Cr,Hcv,Wcv]
        K_srcs = Ks_used[:, src_indices]
        c2w_srcs = c2ws[:, src_indices]

        # 4) Full-resolution cost volume + depth
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

        conf_full = conf
        depth_full = depth

        # 5) Aggregate source features at predicted depth
        src_agg_feat, valid = aggregate_src_features_at_depth(
            src_feats=src_feats,
            depth=depth_full,
            K_ref=K_ref,
            c2w_ref=c2w_ref,
            K_srcs=K_srcs,
            c2w_srcs=c2w_srcs,
        )

        # 6) Gaussian params per pixel
        d_xyz, scales_full, quat_full, opacity_full, color_full = self.gaussian_head(
            ref_feat=ref_feat,
            src_agg_feat=src_agg_feat,
            fused_feat=fused_feat,
            depth=depth_full,
            conf=conf_full,
            rgb=ref_img,
        )

        # 7) Unproject full-res depth to world coordinates
        X_ref_cam = unproject_depth(depth_full, K_ref)  # [B,3,Hcv,Wcv]
        X_ref_cam = X_ref_cam + d_xyz
        X_world_full = cam_to_world_grid(X_ref_cam, c2w_ref)

        # 8) Emit gaussians on strided full-res grid
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
            "depth": depth_full,
            "confidence": conf_full,
            "depth_pdf": depth_pdf,
            "means3D": means3D,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
            "colors": colors,
            "gaussian_conf": conf_flat,
            "feat_h": Hcv,
            "feat_w": Wcv,
            "dino_h": Hd,
            "dino_w": Wd,
            "emit_stride": emit_stride,
            "ref_idx": ref_idx,
        }
