import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DinoV3DenseEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", freeze=True):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )

        self.hidden_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size
        self.num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)

        if freeze:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        image_mean = self.processor.image_mean
        image_std = self.processor.image_std

        self.register_buffer(
            "mean",
            torch.tensor(image_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(image_std).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape

        x = (x - self.mean) / self.std
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)

        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0]
        patch_tokens = last_hidden_state[:, 1 + self.num_register_tokens :]

        _, num_patches, channels = patch_tokens.shape
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        if grid_h * grid_w != num_patches:
            raise ValueError(
                f"DINOv3 reshape mismatch H={height} W={width} gh={grid_h} gw={grid_w} N={num_patches}"
            )

        features = patch_tokens.reshape(batch_size, grid_h, grid_w, channels)
        features = features.permute(0, 3, 1, 2).contiguous()
        return features, cls_token
