import torch
import torch.nn as nn
import torch.nn.functional as F

from .mvv2_geometry import warp_feature_to_ref_plane


class PlaneSweepCostVolume(nn.Module):
    """
    Full-resolution plane sweep over upsampled features.
    """

    def __init__(self, num_depth_bins=128, depth_min=0.5, depth_max=15.0):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max

    def get_depth_values(self, device, dtype):
        """
        Returns inverse-depth sampled depth bins: [D]
        """
        inv_min = 1.0 / self.depth_max
        inv_max = 1.0 / self.depth_min

        inv = torch.linspace(
            inv_min,
            inv_max,
            self.num_depth_bins,
            device=device,
            dtype=dtype,
        )

        depth = 1.0 / inv  # [D]
        return depth

    def forward(self, ref_feat, src_feats, K_ref, c2w_ref, K_srcs, c2w_srcs):
        """
        ref_feat:  [B,C,H,W]
        src_feats: [B,Vs,C,H,W]
        K_ref:     [B,3,3]
        c2w_ref:   [B,4,4]
        K_srcs:    [B,Vs,3,3]
        c2w_srcs:  [B,Vs,4,4]

        returns:
            cost_volume:  [B,D,H,W]
            depth_values: [D]
        """
        B, C, H, W = ref_feat.shape
        Vs = src_feats.shape[1]

        device = ref_feat.device
        dtype = ref_feat.dtype

        depth_values = self.get_depth_values(device, dtype)  # [D]
        ref_feat_n = F.normalize(ref_feat, dim=1)

        cost_slices = []

        for d in range(self.num_depth_bins):
            plane_val = depth_values[d]
            plane = plane_val.view(1, 1, 1, 1).expand(B, 1, H, W)

            sims = []
            for s in range(Vs):
                warped_src, valid = warp_feature_to_ref_plane(
                    src_feat=src_feats[:, s],
                    depth_plane=plane,
                    K_ref=K_ref,
                    c2w_ref=c2w_ref,
                    K_src=K_srcs[:, s],
                    c2w_src=c2w_srcs[:, s],
                )

                warped_src_n = F.normalize(warped_src, dim=1)
                sim = (ref_feat_n * warped_src_n).sum(dim=1, keepdim=True)
                sim = sim * valid
                sims.append(sim)

            sim_stack = torch.stack(sims, dim=1)  # [B,Vs,1,H,W]
            sim_mean = sim_stack.mean(dim=1)  # [B,1,H,W]
            cost_slices.append(sim_mean)

        cost_volume = torch.cat(cost_slices, dim=1)  # [B,D,H,W]
        return cost_volume, depth_values
