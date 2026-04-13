from contextlib import nullcontext

import torch
import torch.nn as nn

from configs.re10k_experiment import (
    _configure_cache_dirs,
    _import_vggt_class,
    _resolve_cache_root,
)
from gs_models.mvv3_encoder import DinoV2DenseEncoder


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


def _normalize_patch_size(patch_size):
    if isinstance(patch_size, int):
        return patch_size, patch_size
    if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        return int(patch_size[0]), int(patch_size[1])
    raise ValueError(f"Unsupported patch size format: {patch_size}")


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
    def __init__(self, num_view, config):
        super().__init__()

        self.num_view = num_view
        self.config = config

        self.dino = self._build_dino_model()
        self.vggt = self._build_vggt_model()
        self.patch_h, self.patch_w = self._resolve_patch_size()

    def _build_dino_model(self):
        dino = DinoV2DenseEncoder(
            model_name=self.config.model.dino_name,
            freeze=self.config.model.freeze_dino,
        )

        return dino

    def _build_vggt_model(self):
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

    def _resolve_patch_size(self):
        candidates = [
            getattr(getattr(self.vggt, "aggregator", None), "patch_size", None),
            getattr(getattr(getattr(self.vggt, "aggregator", None), "patch_embed", None), "patch_size", None),
            getattr(getattr(self.vggt, "patch_embed", None), "patch_size", None),
        ]

        for candidate in candidates:
            if candidate is not None:
                return _normalize_patch_size(candidate)

        return 14, 14

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
        for view_idx in range(num_view):
            feat, _ = self.dino(inputs[:, view_idx])
            dino_features.append(feat)
        dino_features = torch.stack(dino_features, dim=1)

        imgs_for_vggt, original_hw = _pad_images_to_patch_multiple(
            inputs,
            self.patch_h,
            self.patch_w,
        )

        vggt_grad_ctx = torch.no_grad() if self.config.model.freeze_vggt else nullcontext()
        with vggt_grad_ctx:
            tokens, ps_idx = self.vggt.aggregator(imgs_for_vggt)
            depth_all, _ = self.vggt.depth_head(tokens, imgs_for_vggt, ps_idx)

        depth_all = _normalize_depth_tensor(depth_all, batch_size, num_view).float()
        depth_all = _crop_predictions_to_original(depth_all, original_hw)

        return dino_features, depth_all
