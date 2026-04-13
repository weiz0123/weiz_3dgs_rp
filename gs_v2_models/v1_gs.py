import torch
import torch.nn as nn
from transformers import Dinov2Model

from configs.re10k_experiment import (
    _configure_cache_dirs,
    _import_vggt_class,
    _resolve_cache_root,
)


class V1GSModel(nn.Module):
    def __init__(self, num_view, config):
        super().__init__()

        self.num_view = num_view
        self.config = config

        self.dino = self._build_dino_model()
        self.vggt = self._build_vggt_model()

    def _build_dino_model(self):
        dino = Dinov2Model.from_pretrained(
            self.config.model.dino_name,
            output_hidden_states=True,
        )

        if self.config.model.freeze_dino:
            dino.eval()
            for param in dino.parameters():
                param.requires_grad = False

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

    def forward(self, inputs):
        batch_size, num_view, channels, height, width = inputs.shape

        if batch_size != 1 or num_view != self.num_view or channels != 3:
            raise ValueError(
                f"Expected input (training images) shape (1, {self.num_view}, 3, H, W), but got {inputs.shape}"
            )

        flat_inputs = inputs.reshape(batch_size * num_view, channels, height, width)
        dino_outputs = self.dino(pixel_values=flat_inputs, output_hidden_states=True)

        vggt_grad_ctx = torch.no_grad() if self.config.model.freeze_vggt else torch.enable_grad()
        with vggt_grad_ctx:
            vggt_outputs = self.vggt(inputs)

        return {
            "dino_outputs": dino_outputs,
            "vggt_outputs": vggt_outputs,
            "num_view": self.num_view,
            "input_shape": tuple(inputs.shape),
        }
