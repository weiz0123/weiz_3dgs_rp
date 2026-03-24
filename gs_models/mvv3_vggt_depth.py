import os
import sys
from contextlib import nullcontext

import torch
import torch.nn as nn

from .mvv3_heads import ConvBlock


def _resolve_cache_root(explicit_cache_dir=None):
    if explicit_cache_dir:
        return explicit_cache_dir

    scratch_cache = "/scratch/huggingface"
    if os.name != "nt" and os.path.isdir(scratch_cache):
        return scratch_cache
    return None


def _configure_cache_dirs(cache_root):
    if not cache_root:
        return None

    os.makedirs(cache_root, exist_ok=True)
    hub_dir = os.path.join(cache_root, "hub")
    checkpoints_dir = os.path.join(cache_root, "vggt")
    os.makedirs(hub_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = hub_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_dir
    os.environ["TRANSFORMERS_CACHE"] = hub_dir
    os.environ["TORCH_HOME"] = cache_root

    return checkpoints_dir


def _maybe_add_repo_path(repo_path):
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _normalize_depth_tensor(x, batch_size, num_views):
    if x is None:
        return None

    if x.ndim == 5:
        if x.shape[0] == batch_size and x.shape[1] == num_views:
            return x
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


class DepthFeatureAdapter(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(feat_dim * 2 + 2, feat_dim),
            ConvBlock(feat_dim, feat_dim),
        )

    def forward(self, ref_feat, mv_feat, depth, conf):
        return self.net(torch.cat([ref_feat, mv_feat, depth, conf], dim=1))


class OfficialVGGTDepthModule(nn.Module):
    """
    Wrapper around the official VGGT depth branch using the README load pattern:
      model = VGGT()
      state_dict = torch.hub.load_state_dict_from_url(...)
      model.load_state_dict(state_dict)
    """

    def __init__(
        self,
        feat_dim,
        depth_min=0.5,
        depth_max=15.0,
        vggt_model_name="facebook/VGGT-1B",
        vggt_repo_path=None,
        vggt_cache_dir=None,
        vggt_checkpoint_path=None,
        vggt_weights_url="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        freeze_vggt=True,
    ):
        super().__init__()
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.freeze_vggt = freeze_vggt
        self.vggt_model_name = vggt_model_name
        self.cache_root = _resolve_cache_root(vggt_cache_dir)
        self.checkpoints_dir = _configure_cache_dirs(self.cache_root)
        self.checkpoint_path = vggt_checkpoint_path
        self.weights_url = vggt_weights_url

        _maybe_add_repo_path(vggt_repo_path)

        try:
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise ImportError(
                "Official VGGT is not installed. Install the VGGT package on the HPC "
                "or provide its repo path in the config."
            ) from exc

        self.vggt = VGGT()
        state_dict = self._load_state_dict()
        self.vggt.load_state_dict(state_dict)

        if self.freeze_vggt:
            self.vggt.eval()
            for p in self.vggt.parameters():
                p.requires_grad = False

        self.feature_adapter = DepthFeatureAdapter(feat_dim)

    def _load_state_dict(self):
        if self.checkpoint_path:
            return torch.load(self.checkpoint_path, map_location="cpu")

        state_dict = torch.hub.load_state_dict_from_url(
            self.weights_url,
            model_dir=self.checkpoints_dir,
            map_location="cpu",
            progress=True,
        )

        if self.checkpoints_dir is not None:
            downloaded_path = os.path.join(
                self.checkpoints_dir,
                os.path.basename(self.weights_url),
            )
            if os.path.exists(downloaded_path):
                self.checkpoint_path = downloaded_path

        return state_dict

    def forward(self, imgs, ref_idx, ref_feat_full, mv_context_full):
        batch_size, num_views = imgs.shape[:2]

        grad_ctx = torch.no_grad() if self.freeze_vggt else nullcontext()
        with grad_ctx:
            tokens, ps_idx = self.vggt.aggregator(imgs)
            depth_all, conf_all = self.vggt.depth_head(tokens, imgs, ps_idx)

        depth_all = _normalize_depth_tensor(depth_all, batch_size, num_views).float()
        if conf_all is None:
            conf_all = torch.ones_like(depth_all)
        else:
            conf_all = _normalize_depth_tensor(conf_all, batch_size, num_views).float()

        depth_ref = depth_all[:, ref_idx]
        conf_ref = conf_all[:, ref_idx]

        if conf_ref.shape[1] != 1:
            conf_ref = conf_ref.mean(dim=1, keepdim=True)
        if depth_ref.shape[1] != 1:
            depth_ref = depth_ref.mean(dim=1, keepdim=True)

        depth_ref = depth_ref.clamp(min=self.depth_min, max=self.depth_max)
        if conf_ref.min() < 0.0 or conf_ref.max() > 1.0:
            conf_ref = conf_ref.sigmoid()

        fused_feat = self.feature_adapter(ref_feat_full, mv_context_full, depth_ref, conf_ref)

        return {
            "depth": depth_ref,
            "confidence": conf_ref,
            "fused_feat": fused_feat,
        }
