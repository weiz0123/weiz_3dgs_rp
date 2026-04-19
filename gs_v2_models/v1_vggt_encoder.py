from contextlib import nullcontext

import torch
import torch.nn as nn

from configs.re10k_experiment import (
    _configure_cache_dirs,
    _import_vggt_class,
    _resolve_cache_root,
)
from debug_util import DEBUG
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


def _set_requires_grad(module, requires_grad):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def _find_block_container(module):
    if module is None:
        return None

    for attr_name in (
        "blocks",
        "layers",
        "transformer_blocks",
        "stages",
        "stage_blocks",
    ):
        child = getattr(module, attr_name, None)
        if isinstance(child, (nn.ModuleList, nn.Sequential)) and len(child) > 0:
            return child

    for _, child in module.named_children():
        if isinstance(child, (nn.ModuleList, nn.Sequential)) and len(child) > 0:
            return child

    return None


class V1VGGTEncoder(nn.Module):
    def __init__(self, config, patch_h=14, patch_w=14):
        super().__init__()
        self.config = config
        self.patch_h = patch_h
        self.patch_w = patch_w
        self._vggt_has_trainable_params = False
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
            for param in vggt.parameters():
                param.requires_grad = False

            if self.config.model.vggt_unfreeze_heads:
                _set_requires_grad(getattr(vggt, "camera_head", None), True)
                _set_requires_grad(getattr(vggt, "depth_head", None), True)

            num_tail_blocks = max(0, int(self.config.model.vggt_unfreeze_last_blocks))
            if num_tail_blocks > 0:
                block_container = _find_block_container(getattr(vggt, "aggregator", None))
                if block_container is not None:
                    for block in list(block_container)[-num_tail_blocks:]:
                        _set_requires_grad(block, True)
        else:
            for param in vggt.parameters():
                param.requires_grad = True

        self._vggt_has_trainable_params = any(param.requires_grad for param in vggt.parameters())
        if not self._vggt_has_trainable_params:
            vggt.eval()

        total_params = sum(param.numel() for param in vggt.parameters())
        trainable_params = sum(param.numel() for param in vggt.parameters() if param.requires_grad)
        DEBUG.log_debuge_csv(
            "vggt_build",
            freeze_vggt=self.config.model.freeze_vggt,
            vggt_unfreeze_heads=self.config.model.vggt_unfreeze_heads,
            vggt_unfreeze_last_blocks=self.config.model.vggt_unfreeze_last_blocks,
            total_params=total_params,
            trainable_params=trainable_params,
            has_trainable_params=self._vggt_has_trainable_params,
        )

        return vggt

    def forward(self, inputs):
        imgs_for_vggt, original_hw = _pad_images_to_patch_multiple(
            inputs,
            self.patch_h,
            self.patch_w,
        )

        vggt_grad = nullcontext() if self._vggt_has_trainable_params else torch.no_grad()
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

        if DEBUG.is_first_batch():
            DEBUG.log_debuge_csv(
                "vggt_forward",
                inputs=inputs,
                padded_inputs=imgs_for_vggt,
                tokens=tokens,
                depth=depth_all,
                estimated_extrinsics=extrinsic_all,
                estimated_intrinsics=intrinsic_all,
                original_hw=original_hw,
            )

        return {
            "tokens": tokens,
            "depth": depth_all,
            "estimated_extrinsics": extrinsic_all.float(),
            "estimated_intrinsics": intrinsic_all.float(),
            "original_hw": original_hw,
        }
