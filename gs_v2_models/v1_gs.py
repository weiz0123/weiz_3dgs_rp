from contextlib import nullcontext

import torch.nn.functional as F
import torch
import torch.nn as nn

from configs.re10k_experiment import (
    _configure_cache_dirs,
    _import_vggt_class,
    _resolve_cache_root,
)
from .dense_transformer import CrossAttention, DenseFusionTransformer, SelfAttention
from .v1_dino_encoder import DinoV3DenseEncoder
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from .v1_gaussian_head import GaussianHead
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def _normalize_depth_tensor(x, batch_size, num_views):
    if x is None:
        return None

    if x.ndim == 5:
        if x.shape[0] == batch_size and x.shape[1] == num_views and x.shape[2] <= 4:
            return x
        if x.shape[0] == batch_size and x.shape[1] == num_views and x.shape[-1] <= 4:
            return x.permute(0, 1, 4, 2, 3).contiguous()
        if x.shape[0] == batch_size and x.shape[2] == num_views:
            return x.permute(0, 2, 1, 3, 4).contiguous()

    if x.ndim == 4:
        if x.shape[0] == batch_size and x.shape[1] == num_views:
            return x.unsqueeze(2)
        if x.shape[0] == batch_size * num_views:
            return x.reshape(batch_size, num_views, *x.shape[1:])
        if x.shape[0] == num_views:
            return x.unsqueeze(0).unsqueeze(2)
        if x.shape[0] == batch_size and x.shape[1] == 1:
            return x.unsqueeze(1)

    if x.ndim == 3 and x.shape[0] == num_views:
        return x.unsqueeze(0).unsqueeze(2)

    raise ValueError(f"Unsupported VGGT depth/conf shape: {tuple(x.shape)}")


def _pad_images_to_patch_multiple(imgs, patch_h, patch_w):
    h, w = imgs.shape[-2:]
    pad_h = (patch_h - (h % patch_h)) % patch_h
    pad_w = (patch_w - (w % patch_w)) % patch_w

    if pad_h == 0 and pad_w == 0:
        return imgs, (h, w)

    if imgs.ndim == 5:
        b, v, c, _, _ = imgs.shape
        imgs_4d = imgs.reshape(b * v, c, h, w)
        padded_4d = torch.nn.functional.pad(
            imgs_4d,
            (0, pad_w, 0, pad_h),
            mode="replicate",
        )
        padded = padded_4d.reshape(b, v, c, h + pad_h, w + pad_w)
        return padded, (h, w)

    padded = torch.nn.functional.pad(imgs, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (h, w)


def _crop_predictions_to_original(x, original_hw):
    if x is None:
        return None
    h, w = original_hw
    return x[..., :h, :w]



class V1GSModel(nn.Module):
    def __init__(self, num_view=8, gaussian_per_pixel=2, sh_degree=2, config=None):
        super().__init__()

        self.num_view = num_view
        self.gaussian_per_pixel = gaussian_per_pixel
        self.sh_degree = sh_degree
        self.config = config
        self.feature_dim = 2048
        self.patch_h = 14
        self.patch_w = 14


        self.vggt = self.freezed_vggt()
        self.dino = DinoV3DenseEncoder(
            model_name=self.config.model.dino_name,
            freeze=self.config.model.freeze_dino,
        )
        self.fusion_transformer = DenseFusionTransformer(
            vggt_dim=2048, 
            dino_dim=768,  
            depth=2, 
            num_heads=8
        )

        self.gaussian_head = GaussianHead(
            ref_feat_dim=2048,
            mv_feat_dim=2048,
            fused_dim=2048,
            hidden=512,
            sh_degree=self.sh_degree,       
            num_surfaces=self.gaussian_per_pixel,    
            )


    def freezed_vggt(self):
        cache_root = _resolve_cache_root(self.config.model.vggt_cache_dir)
        checkpoints_dir = _configure_cache_dirs(cache_root)

        VGGT, _ = _import_vggt_class(self.config.model.vggt_repo_path)
        vggt = VGGT()

        if self.config.model.vggt_checkpoint_path:
            state_dict = torch.load(
                self.config.model.vggt_checkpoint_path,
                map_location="cpu",
            )
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                self.config.model.vggt_weights_url,
                model_dir=checkpoints_dir,
                map_location="cpu",
                progress=True,
            )

        vggt.load_state_dict(state_dict)

        if self.config.model.freeze_vggt:
            vggt.eval()
            for param in vggt.parameters():
                param.requires_grad = False

        return vggt


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
        dino_features = []
        for view_idx in range(num_view):
            feat, _ = self.dino(inputs[:, view_idx])
            dino_features.append(feat)
        dino_features = torch.stack(dino_features, dim=1)

        # VGGT pose and depth
        imgs_for_vggt, original_hw = _pad_images_to_patch_multiple(
            inputs,
            self.patch_h,
            self.patch_w,
        )

        vggt_grad = torch.no_grad() if self.config.model.freeze_vggt else nullcontext()
        with vggt_grad:

            # Global and Frame transformer (ed) token
            tokens, ps_idx = self.vggt.aggregator(imgs_for_vggt) #size of 24

            # camera head
            pose_enc = self.vggt.camera_head(tokens)[-1]
            extrinsic_all, intrinsic_all = pose_encoding_to_extri_intri(
                pose_enc,
                original_hw,
            )
            
            # raw depth
            depth_all, _ = self.vggt.depth_head(tokens, imgs_for_vggt, ps_idx)

        # normalized detph
        depth_all = depth_all.permute(0, 1, 4, 2, 3).contiguous() # B, V, C, H, W
        depth_all = _crop_predictions_to_original(depth_all, original_hw)


        last_tokens = tokens[-1] 
        
        vggt_spatial = last_tokens[:, :, 1:, :]

        # [B, V, C, H, W] -> [B, V, N, C]
        dino_tokens = dino_features.permute(0, 1, 3, 4, 2).reshape(
            batch_size,
            num_view,
            -1,
            dino_features.shape[2],
        )

        fused_spatial = self.fusion_transformer(vggt_spatial, dino_tokens)

        # 1. Reshape fused_spatial [B, V, 1200, 2048] to 4D [B*V, 2048, H, W]
        # 1200 tokens usually means 30x40 patches for a 420x560 image
        feat_h, feat_w = height // self.patch_h, width // self.patch_w
        
        # [B, V, N, C] -> [B*V, N, C] -> [B*V, C, H, W]
        fused_map = fused_spatial.view(-1, feat_h, feat_w, self.feature_dim).permute(0, 3, 1, 2).contiguous()

        # 2. Resize Depth and Confidence to match the Feature Map (e.g., 30x40)
        # depth_all is [B, V, 1, H, W] -> flatten to [B*V, 1, H, W]
        flat_depth = depth_all.reshape(-1, 1, height, width)
        
        # Downsample to match the 2048-channel feature map resolution
        depth_low = F.interpolate(flat_depth, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
        
        # For confidence, if you don't have it from VGGT, use a constant 1.0
        conf_low = torch.ones_like(depth_low)
    

        outputs = self.gaussian_head(
                ref_feat=fused_map,
                mv_feat=fused_map,
                fused_feat=fused_map,
                depth=depth_low,
                conf=conf_low
            )    

        

        return {
            "guaussian_outputs": outputs,
            "features": dino_features,
            "depth": depth_all,
            "estimated_extrinsics": extrinsic_all.float(),
            "estimated_intrinsics": intrinsic_all.float(),
        }
