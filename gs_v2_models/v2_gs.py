from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.re10k_experiment import (
    _configure_cache_dirs,
    _import_vggt_class,
    _resolve_cache_root,
)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .v2_dino_encoder import V2DinoDenseEncoder
from .v2_gaussian_head import V2GaussianHead


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
        padded_4d = F.pad(imgs_4d, (0, pad_w, 0, pad_h), mode="replicate")
        padded = padded_4d.reshape(b, v, c, h + pad_h, w + pad_w)
        return padded, (h, w)

    padded = F.pad(imgs, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (h, w)


def _crop_predictions_to_original(x, original_hw):
    if x is None:
        return None
    h, w = original_hw
    return x[..., :h, :w]


class V2GSModel(nn.Module):
    def __init__(self, num_view=8, gaussian_per_pixel=1, sh_degree=1, emit_stride=4, config=None):
        super().__init__()

        self.num_view = num_view
        self.gaussian_per_pixel = gaussian_per_pixel
        self.sh_degree = sh_degree
        self.emit_stride = emit_stride
        self.config = config
        self.patch_h = 14
        self.patch_w = 14
        self.feat_reduce_dim = 128

        self.vggt = self._build_vggt()
        self.dino = V2DinoDenseEncoder(
            model_name=self.config.model.dino_name,
            freeze=self.config.model.freeze_dino,
        )
        self.feat_reduce = nn.Conv2d(self.dino.hidden_dim, self.feat_reduce_dim, 1)
        self.gaussian_head = V2GaussianHead(
            feat_dim=self.feat_reduce_dim,
            hidden=128,
            sh_degree=self.sh_degree,
            num_surfaces=self.gaussian_per_pixel,
        )

    def _build_vggt(self):
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

        dino_features = []
        reduced_features = []
        for view_idx in range(num_view):
            feat, _ = self.dino(inputs[:, view_idx])
            dino_features.append(feat)
            reduced = self.feat_reduce(feat)
            reduced = F.interpolate(
                reduced,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            reduced_features.append(reduced)

        dino_features = torch.stack(dino_features, dim=1)
        reduced_stack = torch.stack(reduced_features, dim=1)
        fused_map = reduced_stack.reshape(batch_size * num_view, self.feat_reduce_dim, height, width)

        imgs_for_vggt, original_hw = _pad_images_to_patch_multiple(inputs, self.patch_h, self.patch_w)

        vggt_grad = torch.no_grad() if self.config.model.freeze_vggt else nullcontext()
        with vggt_grad:
            tokens, ps_idx = self.vggt.aggregator(imgs_for_vggt)
            pose_enc = self.vggt.camera_head(tokens)[-1]
            extrinsic_all, intrinsic_all = pose_encoding_to_extri_intri(
                pose_enc,
                original_hw,
            )
            depth_all, conf_all = self.vggt.depth_head(tokens, imgs_for_vggt, ps_idx)

        depth_all = _normalize_depth_tensor(depth_all, batch_size, num_view).float()
        depth_all = _crop_predictions_to_original(depth_all, original_hw)

        if conf_all is None:
            conf_all = torch.ones_like(depth_all)
        else:
            conf_all = _normalize_depth_tensor(conf_all, batch_size, num_view).float()
            conf_all = _crop_predictions_to_original(conf_all, original_hw)

        depth_low = F.interpolate(
            depth_all.reshape(batch_size * num_view, 1, height, width),
            size=dino_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        conf_low = F.interpolate(
            conf_all.reshape(batch_size * num_view, 1, height, width),
            size=dino_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        gaussian_outputs = self.gaussian_head(
            feat=fused_map,
            depth=depth_all.reshape(batch_size * num_view, 1, height, width),
            conf=conf_all.reshape(batch_size * num_view, 1, height, width),
            rgb=inputs.reshape(batch_size * num_view, 3, height, width),
        )
        gaussian_outputs["emit_stride"] = self.emit_stride

        return {
            "guaussian_outputs": gaussian_outputs,
            "features": dino_features,
            "fused_map": fused_map,
            "depth": depth_all,
            "depth_low": depth_low,
            "conf_low": conf_low,
            "estimated_extrinsics": extrinsic_all.float(),
            "estimated_intrinsics": intrinsic_all.float(),
        }
