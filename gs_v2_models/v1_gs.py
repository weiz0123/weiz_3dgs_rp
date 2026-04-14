from contextlib import nullcontext

import torch
import torch.nn as nn

from configs.re10k_experiment import (
    _configure_cache_dirs,
    _import_vggt_class,
    _resolve_cache_root,
)
from .v1_dino_encoder import DinoV3DenseEncoder
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


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


class CrossAttention(nn.Module):
    """
    Minimal cross-attention block for later token fusion between
    DINO and VGGT token sequences.

    query_tokens:   [B, Nq, Cq]
    context_tokens: [B, Nk, Ck]
    output:         [B, Nq, Cq]
    """

    def __init__(self, query_dim, context_dim, num_heads=8, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()

        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim={query_dim} must be divisible by num_heads={num_heads}")

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.out_dropout = nn.Dropout(proj_dropout)
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_ctx = nn.LayerNorm(context_dim)

    def forward(self, query_tokens, context_tokens, attention_mask=None, return_attention=False):
        if query_tokens.ndim != 3:
            raise ValueError(
                f"Expected query_tokens [B, Nq, Cq], got {tuple(query_tokens.shape)}"
            )
        if context_tokens.ndim != 3:
            raise ValueError(
                f"Expected context_tokens [B, Nk, Ck], got {tuple(context_tokens.shape)}"
            )

        batch_size, num_query, _ = query_tokens.shape
        _, num_context, _ = context_tokens.shape

        query_tokens = self.norm_q(query_tokens)
        context_tokens = self.norm_ctx(context_tokens)

        q = self.q_proj(query_tokens)
        k = self.k_proj(context_tokens)
        v = self.v_proj(context_tokens)

        q = q.view(batch_size, num_query, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_context, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_context, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_query, self.query_dim)
        output = self.out_proj(attended)
        output = self.out_dropout(output)

        if return_attention:
            return output, attn_weights
        return output


class SelfAttention(nn.Module):
    """
    Minimal self-attention block for token refinement within a single
    token sequence.

    tokens: [B, N, C]
    output: [B, N, C]
    """

    def __init__(self, embed_dim, num_heads=8, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(proj_dropout)

    def forward(self, tokens, attention_mask=None, return_attention=False):
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B, N, C], got {tuple(tokens.shape)}")

        batch_size, num_tokens, _ = tokens.shape
        tokens = self.norm(tokens)

        qkv = self.qkv_proj(tokens)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.embed_dim)
        output = self.out_proj(attended)
        output = self.out_dropout(output)

        if return_attention:
            return output, attn_weights
        return output


class V1GSModel(nn.Module):
    def __init__(self, num_view, config):
        super().__init__()

        self.num_view = num_view
        self.config = config

        self.dino = self.freezed_dino()
        self.vggt = self.freezed_vggt()

        # Defautl VGGT, retrain required if changed
        self.patch_h = 14
        self.patch_w = 14

    def freezed_dino(self):
        dino = DinoV3DenseEncoder(
            model_name=self.config.model.dino_name,
            freeze=self.config.model.freeze_dino,
        )

        return dino

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
            tokens, ps_idx = self.vggt.aggregator(imgs_for_vggt)

            # camera head
            pose_enc = self.vggt.camera_head(tokens)[-1]
            extrinsic_all, intrinsic_all = pose_encoding_to_extri_intri(
                pose_enc,
                original_hw,
            )
            
            # raw depth
            depth_all, _ = self.vggt.depth_head(tokens, imgs_for_vggt, ps_idx)
            print(tokens.shape, depth_all.shape, ps_idx.shape)
        # normalized detph
        depth_all = depth_all.permute(0, 1, 4, 2, 3).contiguous() # B, V, C, H, W
        depth_all = _crop_predictions_to_original(depth_all, original_hw)

        return {
            
            "features": dino_features,
            "depth": depth_all,
            "estimated_extrinsics": extrinsic_all.float(),
            "estimated_intrinsics": intrinsic_all.float(),
        }
