from contextlib import nullcontext

import torch
import torch.nn as nn

from configs.re10k_experiment import (
    _configure_cache_dirs,
    _import_vggt_class,
    _resolve_cache_root,
)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


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


class V1VGGTEncoder(nn.Module):
    def __init__(self, config, patch_h=14, patch_w=14):
        super().__init__()
        self.config = config
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.vggt = self._build_vggt()

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
        imgs_for_vggt, original_hw = _pad_images_to_patch_multiple(
            inputs,
            self.patch_h,
            self.patch_w,
        )

        vggt_grad = torch.no_grad() if self.config.model.freeze_vggt else nullcontext()
        with vggt_grad:
            tokens, ps_idx = self.vggt.aggregator(imgs_for_vggt)
            pose_enc = self.vggt.camera_head(tokens)[-1]
            extrinsic_all, intrinsic_all = pose_encoding_to_extri_intri(
                pose_enc,
                original_hw,
            )
            depth_all, _ = self.vggt.depth_head(tokens, imgs_for_vggt, ps_idx)

        depth_all = depth_all.permute(0, 1, 4, 2, 3).contiguous()
        depth_all = _crop_predictions_to_original(depth_all, original_hw)

        return {
            "tokens": tokens,
            "depth": depth_all,
            "estimated_extrinsics": extrinsic_all.float(),
            "estimated_intrinsics": intrinsic_all.float(),
            "original_hw": original_hw,
        }
