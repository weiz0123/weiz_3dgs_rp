import torch.nn.functional as F
import torch
import torch.nn as nn

from debug_util import DEBUG
from gs_models.mvv2_geometry import scale_intrinsics_batch, warp_feature_to_ref_plane
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


def _select_reference_view_index(num_view):
    if num_view <= 1:
        return 0
    if num_view % 2 == 0:
        return max(0, (num_view // 2) - 1)
    return num_view // 2



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
        self.emission_grid_upsample = max(
            1,
            int(getattr(self.config.model, "emission_grid_upsample", 1)),
        )
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
        self.reference_view_idx = _select_reference_view_index(self.num_view)
        self.warp_feat_dim = 256
        self.dino_to_warp = nn.Conv2d(4096, self.warp_feat_dim, kernel_size=1)
        self.warp_to_dino = nn.Conv2d(self.warp_feat_dim, 4096, kernel_size=1)
        self.vggt_map_to_dino = nn.Conv2d(2048, 4096, kernel_size=1)
        self.src_agg_logit = nn.Parameter(torch.tensor(-2.5))
        self.vggt_ref_logit = nn.Parameter(torch.tensor(-4.0))

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

        dino_features, _ = self.dino(inputs)
        feat_h, feat_w = dino_features.shape[-2:]
        emit_h = feat_h * self.emission_grid_upsample
        emit_w = feat_w * self.emission_grid_upsample
        if self.emission_grid_upsample > 1:
            dino_emission_map = F.interpolate(
                dino_features.reshape(batch_size * num_view, dino_features.shape[2], feat_h, feat_w),
                size=(emit_h, emit_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, num_view, dino_features.shape[2], emit_h, emit_w)
        else:
            dino_emission_map = dino_features

        vggt_outputs = self.vggt(inputs)
        vggt_tokens_all = vggt_outputs["tokens"]
        depth_all = vggt_outputs["depth"]
        depth_conf_all = vggt_outputs["depth_conf"]
        extrinsic_all = vggt_outputs["estimated_extrinsics"]
        intrinsic_all = vggt_outputs["estimated_intrinsics"]

        padded_hw = _compute_padded_hw(height, width, self.patch_h, self.patch_w)
        vggt_token_tensor, vggt_spatial_map, vggt_prefix_tokens = _extract_vggt_spatial_map(
            vggt_tokens_all,
            padded_hw=padded_hw,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
        )
        vggt_spatial_low = F.interpolate(
            vggt_spatial_map.reshape(
                batch_size * num_view,
                vggt_spatial_map.shape[2],
                vggt_spatial_map.shape[3],
                vggt_spatial_map.shape[4],
            ),
            size=(emit_h, emit_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, num_view, vggt_spatial_map.shape[2], emit_h, emit_w)

        scaled_intrinsics = scale_intrinsics_batch(
            train_intrinsics.reshape(batch_size * num_view, 3, 3),
            src_hw=(height, width),
            dst_hw=(emit_h, emit_w),
        ).reshape(batch_size, num_view, 3, 3)
        dino_warp_map = self.dino_to_warp(
            dino_emission_map.reshape(batch_size * num_view, dino_emission_map.shape[2], emit_h, emit_w)
        ).reshape(batch_size, num_view, self.warp_feat_dim, emit_h, emit_w)

        depth_low_all = F.interpolate(
            depth_all.reshape(batch_size * num_view, 1, height, width),
            size=(emit_h, emit_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, num_view, 1, emit_h, emit_w)
        conf_low_all = F.interpolate(
            depth_conf_all.reshape(batch_size * num_view, 1, height, width),
            size=(emit_h, emit_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, num_view, 1, emit_h, emit_w)

        src_agg_weight = torch.sigmoid(self.src_agg_logit)
        vggt_ref_weight = torch.sigmoid(self.vggt_ref_logit)
        fused_maps = []
        src_valid_fractions = []
        src_agg_feature_maps = []
        projected_vggt_refs = []
        for ref_view_idx in range(num_view):
            ref_feature_map = dino_emission_map[:, ref_view_idx]
            ref_vggt_map = vggt_spatial_low[:, ref_view_idx]
            projected_vggt_ref = self.vggt_map_to_dino(ref_vggt_map)
            projected_vggt_refs.append(projected_vggt_ref)

            warped_feature_sum = torch.zeros(
                batch_size,
                self.warp_feat_dim,
                emit_h,
                emit_w,
                device=inputs.device,
                dtype=ref_feature_map.dtype,
            )
            warped_valid_sum = torch.zeros(
                batch_size,
                1,
                emit_h,
                emit_w,
                device=inputs.device,
                dtype=ref_feature_map.dtype,
            )
            ref_depth_low = depth_low_all[:, ref_view_idx]
            for src_view_idx in range(num_view):
                if src_view_idx == ref_view_idx:
                    continue
                warped_src, warped_valid = warp_feature_to_ref_plane(
                    src_feat=dino_warp_map[:, src_view_idx],
                    depth_plane=ref_depth_low,
                    K_ref=scaled_intrinsics[:, ref_view_idx],
                    c2w_ref=train_poses[:, ref_view_idx],
                    K_src=scaled_intrinsics[:, src_view_idx],
                    c2w_src=train_poses[:, src_view_idx],
                )
                warped_feature_sum = warped_feature_sum + warped_src * warped_valid
                warped_valid_sum = warped_valid_sum + warped_valid

            src_valid_fraction = (warped_valid_sum / max(num_view - 1, 1)).clamp(0.0, 1.0)
            src_agg_feature_map = self.warp_to_dino(
                warped_feature_sum / warped_valid_sum.clamp(min=1.0)
            )
            fused_ref_map = (
                ref_feature_map
                + src_agg_weight * src_valid_fraction * src_agg_feature_map
                + vggt_ref_weight * projected_vggt_ref
            )
            fused_maps.append(fused_ref_map)
            src_valid_fractions.append(src_valid_fraction)
            src_agg_feature_maps.append(src_agg_feature_map)

        fused_map = torch.stack(fused_maps, dim=1)
        src_valid_fraction = torch.stack(src_valid_fractions, dim=1)
        src_agg_feature_map = torch.stack(src_agg_feature_maps, dim=1)
        projected_vggt_ref = torch.stack(projected_vggt_refs, dim=1)
        head_feature_map = fused_map
        flat_features = fused_map.reshape(batch_size * num_view, fused_map.shape[2], emit_h, emit_w)

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

        depth_low = depth_low_all.reshape(batch_size * num_view, 1, emit_h, emit_w).detach()
        conf_low = conf_low_all.reshape(batch_size * num_view, 1, emit_h, emit_w).detach()
        flat_depth = depth_low

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
                dino_emission_map=dino_emission_map,
                dino_warp_map=dino_warp_map,
                vggt_token_tensor=vggt_token_tensor,
                vggt_prefix_tokens=vggt_prefix_tokens,
                vggt_spatial_map=vggt_spatial_map,
                vggt_spatial_low=vggt_spatial_low,
                emission_reference_views=list(range(num_view)),
                projected_vggt_ref=projected_vggt_ref,
                src_agg_feature_map=src_agg_feature_map,
                src_valid_fraction=src_valid_fraction,
                src_agg_weight=src_agg_weight,
                vggt_ref_weight=vggt_ref_weight,
                emission_grid_upsample=self.emission_grid_upsample,
                emission_hw=[emit_h, emit_w],
                head_feature_map=head_feature_map,
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
