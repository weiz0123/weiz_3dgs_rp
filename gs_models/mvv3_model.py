import torch
import torch.nn as nn

from .mvv3_encoder import DinoV2DenseEncoder
from .mvv3_geometry import cam_to_world_grid, unproject_depth
from .mvv3_heads import GaussianHead
from .mvv3_mini_vggt import MiniVGGTDepthModule


class MultiViewDinoDepthToGaussians(nn.Module):
    """
    MVV3: DINOv2 features + mini-VGGT depth/context module + Gaussian emission head.
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
        transformer_depth=4,
        transformer_heads=8,
        max_views=8,
    ):
        super().__init__()

        self.encoder = DinoV2DenseEncoder(
            model_name=dino_name,
            freeze=freeze_dino,
        )
        self.mini_vggt = MiniVGGTDepthModule(
            in_dim=self.encoder.hidden_dim,
            feat_dim=feat_reduce_dim,
            depth_min=depth_min,
            depth_max=depth_max,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            max_views=max_views,
        )
        self.gaussian_head = GaussianHead(
            ref_feat_dim=feat_reduce_dim,
            mv_feat_dim=feat_reduce_dim,
            fused_dim=feat_reduce_dim,
            hidden=feat_reduce_dim,
        )

    def forward(self, imgs, Ks, c2ws, ref_idx=0, emit_stride=2):
        b, v, _, h_img, w_img = imgs.shape
        if v < 2:
            raise ValueError("MVV3 requires at least 2 views")

        dino_feats = []
        for view_idx in range(v):
            feat, _ = self.encoder(imgs[:, view_idx])
            dino_feats.append(feat)
        dino_feat_stack = torch.stack(dino_feats, dim=1)

        mini_vggt = self.mini_vggt(
            dino_feat_stack=dino_feat_stack,
            ref_idx=ref_idx,
            target_hw=(h_img, w_img),
        )

        ref_feat_full = mini_vggt["ref_feat_full"]
        mv_context_full = mini_vggt["mv_context_full"]
        ref_img = imgs[:, ref_idx]
        K_ref = Ks[:, ref_idx]
        c2w_ref = c2ws[:, ref_idx]
        depth_full = mini_vggt["depth"]
        conf_full = mini_vggt["confidence"]
        fused_feat = mini_vggt["fused_feat"]

        d_xyz, scales_full, quat_full, opacity_full, color_full = self.gaussian_head(
            ref_feat=ref_feat_full,
            mv_feat=mv_context_full,
            fused_feat=fused_feat,
            depth=depth_full,
            conf=conf_full,
            rgb=ref_img,
        )

        x_ref_cam = unproject_depth(depth_full, K_ref)
        x_ref_cam = x_ref_cam + d_xyz
        x_world_full = cam_to_world_grid(x_ref_cam, c2w_ref)

        x_world = x_world_full[:, :, ::emit_stride, ::emit_stride]
        scales = scales_full[:, :, ::emit_stride, ::emit_stride]
        quat = quat_full[:, :, ::emit_stride, ::emit_stride]
        opacity = opacity_full[:, :, ::emit_stride, ::emit_stride]
        color = color_full[:, :, ::emit_stride, ::emit_stride]
        conf_emit = conf_full[:, :, ::emit_stride, ::emit_stride]

        b2, _, h_g, w_g = x_world.shape
        m = h_g * w_g

        means3D = x_world.reshape(b2, 3, m).permute(0, 2, 1).contiguous()
        scales = scales.reshape(b2, 3, m).permute(0, 2, 1).contiguous()
        rotations = quat.reshape(b2, 4, m).permute(0, 2, 1).contiguous()
        opacities = opacity.reshape(b2, 1, m).permute(0, 2, 1).contiguous()
        colors = color.reshape(b2, 3, m).permute(0, 2, 1).contiguous()
        conf_flat = conf_emit.reshape(b2, 1, m).permute(0, 2, 1).contiguous()

        return {
            "depth": depth_full,
            "confidence": conf_full,
            "depth_pdf": None,
            "means3D": means3D,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
            "colors": colors,
            "gaussian_conf": conf_flat,
            "feat_h": h_img,
            "feat_w": w_img,
            "dino_h": mini_vggt["token_h"],
            "dino_w": mini_vggt["token_w"],
            "emit_stride": emit_stride,
            "ref_idx": ref_idx,
        }
