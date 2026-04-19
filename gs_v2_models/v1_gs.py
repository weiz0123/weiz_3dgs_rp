import torch.nn.functional as F
import torch
import torch.nn as nn

from .dense_transformer import CrossAttention, DenseFusionTransformer, SelfAttention
from .v1_dino_encoder import DinoV3DenseEncoder
from .v1_vggt_encoder import V1VGGTEncoder
from .v1_gaussian_head import DepthAnchoredGaussianHead
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def _infer_token_grid(num_tokens, aspect_ratio):
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")

    best_h = 1
    best_w = num_tokens
    best_err = float("inf")

    for h in range(1, int(num_tokens ** 0.5) + 1):
        if num_tokens % h != 0:
            continue
        w = num_tokens // h
        err = abs((w / h) - aspect_ratio)
        if err < best_err:
            best_h = h
            best_w = w
            best_err = err

    return best_h, best_w



class V1GSModel(nn.Module):
    def __init__(self, num_view=8, gaussian_per_pixel=2, sh_degree=2, config=None):
        super().__init__()

        self.num_view = num_view
        self.gaussian_per_pixel = gaussian_per_pixel
        self.sh_degree = sh_degree
        self.config = config
        self.feature_dim = 2048  # 
        self.patch_h = 14
        self.patch_w = 14


        self.vggt = V1VGGTEncoder(
            config=self.config,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
        )
        self.dino = DinoV3DenseEncoder(
            model_name=self.config.model.dino_name,
            freeze=self.config.model.freeze_dino,
        )
        self.fusion_transformer = DenseFusionTransformer(
            vggt_dim=2048, 
            dino_dim=4096,  
            depth=2, 
            num_heads=8
        )

        self.gaussian_head = DepthAnchoredGaussianHead(
            feat_dim=4096, # Output dimension of dino features
            hidden=256,
            sh_degree=self.sh_degree,
            num_surfaces=self.gaussian_per_pixel,
            min_scale=0.01,
            max_scale=0.05,
            init_dc_bias=0.5,
            )

    def forward(self, inputs):
        if inputs.ndim == 4:
            inputs = inputs.unsqueeze(0)

        if inputs.ndim != 5:
            raise ValueError(
                f"Expected training images with shape [V, 3, H, W] or [B, V, 3, H, W], got {tuple(inputs.shape)}"
            )

        batch_size, num_view, channels, height, width = inputs.shape

        if num_view != self.num_view or channels != 3:
            raise ValueError(
                f"Expected training images with {self.num_view} RGB views, but got {tuple(inputs.shape)}"
            )

        # DINO features
        dino_features, _ = self.dino(inputs)
        feat_h, feat_w = dino_features.shape[-2:]
        flat_features = dino_features.reshape(batch_size * num_view, dino_features.shape[2], feat_h, feat_w)

        # VGGT outputs
        vggt_outputs = self.vggt(inputs)
        depth_all = vggt_outputs["depth"]
        extrinsic_all = vggt_outputs["estimated_extrinsics"]
        intrinsic_all = vggt_outputs["estimated_intrinsics"]
        flat_depth = depth_all.reshape(batch_size * num_view, 1, height, width)
        flat_extrinsics = extrinsic_all.reshape(batch_size * num_view, extrinsic_all.shape[-2], extrinsic_all.shape[-1])
        flat_intrinsics = intrinsic_all.reshape(batch_size * num_view, 3, 3)

        depth_low = F.interpolate(
            flat_depth,
            size=(feat_h, feat_w),
            mode="bilinear",
            align_corners=False,
        )
        conf_low = torch.ones_like(depth_low)
        conf_full = torch.ones_like(flat_depth)

        outputs = self.gaussian_head(
                feat=flat_features,
                depth=flat_depth,
                intrinsic=flat_intrinsics,
                extrinsic=flat_extrinsics,
                conf=conf_full,
                output_size=(height, width),
            )

        

        return {
            "guaussian_outputs": outputs,
            "features": dino_features,
            "fused_map": flat_features,
            "depth": depth_all,
            "depth_low": depth_low,
            "conf_low": conf_low,
            "estimated_extrinsics": extrinsic_all.float(),
            "estimated_intrinsics": intrinsic_all.float(),
        }
