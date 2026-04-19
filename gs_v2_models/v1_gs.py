import torch.nn.functional as F
import torch
import torch.nn as nn

from debug_util import DEBUG
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
        self._printed_intrinsics_debug = False


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
            min_scale=0.001,
            max_scale=0.02,
            init_dc_bias=0.5,
            )

    def forward(
        self,
        inputs,
        train_intrinsics=None,
        train_poses=None,
    ):
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

        if train_intrinsics is None or train_poses is None:
            raise ValueError("train_intrinsics and train_poses must be provided for GT-camera lifting")

        if train_intrinsics.ndim == 3:
            train_intrinsics = train_intrinsics.unsqueeze(0)
        if train_poses.ndim == 3:
            train_poses = train_poses.unsqueeze(0)

        # DINO features
        dino_features, _ = self.dino(inputs)
        feat_h, feat_w = dino_features.shape[-2:]
        flat_features = dino_features.reshape(batch_size * num_view, dino_features.shape[2], feat_h, feat_w)

        # VGGT outputs
        vggt_outputs = self.vggt(inputs)
        depth_all = vggt_outputs["depth"]
        depth_conf_all = vggt_outputs["depth_conf"]
        extrinsic_all = vggt_outputs["estimated_extrinsics"] # not used currently
        intrinsic_all = vggt_outputs["estimated_intrinsics"] # not used currently

        flat_depth = depth_all.reshape(batch_size * num_view, 1, height, width)
        flat_depth_conf = depth_conf_all.reshape(batch_size * num_view, 1, height, width)
        train_w2c = torch.inverse(train_poses)
        flat_extrinsics = train_w2c.reshape(batch_size * num_view, train_w2c.shape[-2], train_w2c.shape[-1])
        flat_intrinsics = train_intrinsics.reshape(batch_size * num_view, 3, 3)
        if not self._printed_intrinsics_debug:
            intr0 = flat_intrinsics[0].detach().cpu()
            looks_normalized = (
                intr0[0, 0].abs() < 10
                and intr0[1, 1].abs() < 10
                and intr0[0, 2].abs() <= 2
                and intr0[1, 2].abs() <= 2
            )
            print("flat_intrinsics[0]:")
            print(intr0)
            print(f"looks_normalized={looks_normalized}")
            self._printed_intrinsics_debug = True

        depth_low = F.interpolate(
            flat_depth,
            size=(feat_h, feat_w),
            mode="bilinear",
            align_corners=False,
        )
        conf_low = F.interpolate(
            flat_depth_conf,
            size=(feat_h, feat_w),
            mode="bilinear",
            align_corners=False,
        )

        outputs = self.gaussian_head(
                feat=flat_features,
                depth=depth_low,
                intrinsic=flat_intrinsics,
                extrinsic=flat_extrinsics,
                conf=conf_low,
            )

        if DEBUG.is_first_batch():
            DEBUG.log_debuge_csv(
                "v1_gs_forward",
                inputs=inputs,
                dino_features=dino_features,
                flat_features=flat_features,
                depth_all=depth_all,
                flat_depth=flat_depth,
                depth_conf_all=depth_conf_all,
                flat_intrinsics=flat_intrinsics,
                flat_extrinsics=flat_extrinsics,
                depth_low=depth_low,
                conf_low=conf_low,
                gaussian_means=outputs.get("means3D"),
                gaussian_scales=outputs.get("scales"),
                gaussian_opacity=outputs.get("opacity"),
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
