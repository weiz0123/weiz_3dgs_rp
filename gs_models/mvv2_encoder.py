import torch
import torch.nn as nn
from transformers import Dinov2Model


class DinoV2DenseEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", freeze=True):
        super().__init__()

        self.backbone = Dinov2Model.from_pretrained(
            model_name,
            output_hidden_states=True,
        )

        self.hidden_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x):
        """
        x: [B,3,H,W]
        return:
            feat: [B,C,H//patch,W//patch]
            cls:  [B,C]
        """
        B, _, H, W = x.shape

        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x, output_hidden_states=True)

        hs = out.hidden_states[-1]  # [B,1+N,C]
        cls = hs[:, 0]
        patch = hs[:, 1:]

        B2, N, C = patch.shape
        gh = H // self.patch_size
        gw = W // self.patch_size

        if gh * gw != N:
            raise ValueError(
                f"DINO reshape mismatch H={H} W={W} gh={gh} gw={gw} N={N}"
            )

        feat = patch.reshape(B2, gh, gw, C).permute(0, 3, 1, 2).contiguous()
        return feat, cls
