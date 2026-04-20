import torch.nn.functional as F
import torch
import torch.nn as nn

from debug_util import DEBUG
from .dense_transformer import DenseFusionTransformer
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


def _compute_padded_hw(height, width, patch_h, patch_w):
    pad_h = (patch_h - (height % patch_h)) % patch_h
    pad_w = (patch_w - (width % patch_w)) % patch_w
    return height + pad_h, width + pad_w


def _extract_vggt_spatial_map(tokens, padded_hw, patch_h, patch_w):
    token_tensor = tokens[-1] if isinstance(tokens, (list, tuple)) else tokens
    if token_tensor.ndim != 4:
        raise ValueError(
            f"Expected VGGT tokens with shape [B, V, N, C], got {tuple(token_tensor.shape)}"
        )

    padded_h, padded_w = padded_hw
    grid_h = padded_h // patch_h
    grid_w = padded_w // patch_w
    spatial_token_count = grid_h * grid_w
    total_tokens = token_tensor.shape[2]
    prefix_tokens = total_tokens - spatial_token_count

    if prefix_tokens < 0:
        raise ValueError(
            f"VGGT token count {total_tokens} is smaller than inferred spatial grid "
            f"{grid_h}x{grid_w} ({spatial_token_count})"
        )

    spatial_tokens = token_tensor[:, :, prefix_tokens:, :]
    if spatial_tokens.shape[2] != spatial_token_count:
        raise ValueError(
            f"Expected {spatial_token_count} spatial tokens, got {spatial_tokens.shape[2]}"
        )

    spatial_map = spatial_tokens.reshape(
        token_tensor.shape[0],
        token_tensor.shape[1],
        grid_h,
        grid_w,
        token_tensor.shape[-1],
    ).permute(0, 1, 4, 2, 3).contiguous()

    return token_tensor, spatial_map, prefix_tokens



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
        self.vggt_to_dino = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 4096),
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
        dino_token_grid = dino_features.permute(0, 1, 3, 4, 2).reshape(
            batch_size,
            num_view,
            feat_h * feat_w,
            dino_features.shape[2],
        )

        # VGGT outputs
        vggt_outputs = self.vggt(inputs)
        vggt_tokens_all = vggt_outputs["tokens"]
        depth_all = vggt_outputs["depth"]
        depth_conf_all = vggt_outputs["depth_conf"]
        extrinsic_all = vggt_outputs["estimated_extrinsics"] # not used currently
        intrinsic_all = vggt_outputs["estimated_intrinsics"] # not used currently

        padded_hw = _compute_padded_hw(height, width, self.patch_h, self.patch_w)
        vggt_token_tensor, vggt_spatial_map, vggt_prefix_tokens = _extract_vggt_spatial_map(
            vggt_tokens_all,
            padded_hw=padded_hw,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
        )
        vggt_spatial_low = F.interpolate(
            vggt_spatial_map.reshape(batch_size * num_view, vggt_spatial_map.shape[2], vggt_spatial_map.shape[3], vggt_spatial_map.shape[4]),
            size=(feat_h, feat_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, num_view, vggt_spatial_map.shape[2], feat_h, feat_w)
        vggt_token_grid = vggt_spatial_low.permute(0, 1, 3, 4, 2).reshape(
            batch_size,
            num_view,
            feat_h * feat_w,
            vggt_spatial_low.shape[2],
        )
        fused_vggt_tokens = self.fusion_transformer(
            vggt_tokens=vggt_token_grid,
            dino_tokens=dino_token_grid,
        )
        fused_features = dino_token_grid + self.vggt_to_dino(fused_vggt_tokens)
        fused_map = fused_features.reshape(
            batch_size,
            num_view,
            feat_h,
            feat_w,
            dino_features.shape[2],
        ).permute(0, 1, 4, 2, 3).contiguous()
        flat_features = fused_map.reshape(batch_size * num_view, fused_map.shape[2], feat_h, feat_w)

        flat_depth = depth_all.reshape(batch_size * num_view, 1, height, width).detach()
        flat_depth_conf = depth_conf_all.reshape(batch_size * num_view, 1, height, width).detach()
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
                dino_token_grid=dino_token_grid,
                vggt_token_tensor=vggt_token_tensor,
                vggt_prefix_tokens=vggt_prefix_tokens,
                vggt_spatial_map=vggt_spatial_map,
                vggt_spatial_low=vggt_spatial_low,
                fused_vggt_tokens=fused_vggt_tokens,
                flat_features=flat_features,
                fused_map=fused_map,
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
            "fused_map": fused_map,
            "depth": depth_all,
            "depth_low": depth_low,
            "conf_low": conf_low,
            "estimated_extrinsics": extrinsic_all.float(),
            "estimated_intrinsics": intrinsic_all.float(),
        }
