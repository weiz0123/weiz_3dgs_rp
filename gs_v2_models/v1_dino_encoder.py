import torch
import torch.nn as nn
from transformers import Dinov2Model


class DinoV2DenseEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-large", freeze=True):
        super().__init__()

        self.backbone = Dinov2Model.from_pretrained(
            model_name,
            output_hidden_states=True,
        )

        self.hidden_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size

        if freeze:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

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
        batch_size, _, height, width = x.shape

        x = (x - self.mean) / self.std
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]
        cls_token = hidden_states[:, 0]
        patch_tokens = hidden_states[:, 1:]

        _, num_patches, channels = patch_tokens.shape
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        if grid_h * grid_w != num_patches:
            raise ValueError(
                f"DINOv2 reshape mismatch H={height} W={width} gh={grid_h} gw={grid_w} N={num_patches}"
            )

        features = patch_tokens.reshape(batch_size, grid_h, grid_w, channels)
        features = features.permute(0, 3, 1, 2).contiguous()
        return features, cls_token
