import os
from dataclasses import dataclass, field


def _default_vggt_cache_dir():
    candidates = [
        os.environ.get("VGGT_CACHE_DIR"),
        "/home/weiz/links/scratch/huggingface",
        "/lustre10/scratch/weiz/huggingface",
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return None


def _default_vggt_checkpoint_path():
    candidates = [
        os.environ.get("VGGT_CHECKPOINT_PATH"),
        "/home/weiz/links/scratch/huggingface/vggt/model.pt",
        "/lustre10/scratch/weiz/huggingface/vggt/model.pt",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _default_vggt_repo_path():
    candidates = [
        os.environ.get("VGGT_REPO_PATH"),
        "/home/weiz/links/scratch/vggt",
        "/lustre10/scratch/weiz/vggt",
        "/lustre10/scratch/weiz/repos/vggt",
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return None


@dataclass
class DataConfig:
    data_root: str = "datasets/realestate10k_subset"
    batch_size: int = 1
    num_workers: int = 2
    shuffle: bool = True
    pin_memory: bool = False
    n_input_views: int = 10
    min_input_views: int = 10
    input_view_sampling: str = "pose_sparse"
    num_target_views: int = 1
    target_mode: str = "middle"
    exclude_target: bool = True


@dataclass
class ModelConfig:
    model_version: str = "mvv3"
    dino_name: str = "facebook/dinov2-base"
    freeze_dino: bool = True
    freeze_vggt: bool = True
    num_depth_bins: int = 48
    depth_min: float = 0.5
    depth_max: float = 20.0
    feat_reduce_dim: int = 128
    use_full_res_cost_volume: bool = True
    transformer_depth: int = 4
    transformer_heads: int = 8
    max_views: int = 12
    vggt_model_name: str = "facebook/VGGT-1B"
    vggt_repo_path: str | None = field(default_factory=_default_vggt_repo_path)
    vggt_cache_dir: str | None = field(default_factory=_default_vggt_cache_dir)
    vggt_checkpoint_path: str | None = field(default_factory=_default_vggt_checkpoint_path)
    vggt_weights_url: str = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


@dataclass
class TrainingConfig:
    device: str = "auto"
    epochs: int = 400
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    emit_stride: int = 1
    max_scenes_per_epoch: int | None = 200
    save_dir: str = "outputs/re10k_debug"
    run_id: int = 2
    resume: bool = True
    resume_mode: str = "latest"
    metric_every: int = 1
    save_best_by: str = "loss"
    log_every_n_steps: int = 10
    visualize_every_n_epochs: int = 2
    visualize_every_n_steps: int = 20


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()
