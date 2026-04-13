import os
import sys
from dataclasses import dataclass, field
'''
re10k_experiment.py consists of:

    1. ExperimentConfig that encapulates
        a. DataConfig: All data-related settings (paths, batch size, num workers, etc.)
        b. ModelConfig: All model-related settings (architecture choices, pretrained model paths, etc
        c. TrainingConfig: All training-related settings (epochs, learning rate, logging, etc.)

    2. get_default_config() function that returns an ExperimentConfig with all default settings filled


'''
# TODO: The following are all private method for env setup
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


def _resolve_cache_root(explicit_cache_dir=None):
    if explicit_cache_dir:
        return explicit_cache_dir

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parent_root = os.path.abspath(os.path.join(repo_root, ".."))
    user = os.environ.get("USER") or os.environ.get("USERNAME")

    candidates = [
        "/home/weiz/links/scratch/huggingface",
        os.path.join(parent_root, "huggingface"),
        os.path.join(repo_root, "huggingface"),
        "/scratch/huggingface",
    ]

    if user:
        candidates.extend(
            [
                os.path.join("/home", user, "links", "scratch", "huggingface"),
                os.path.join("/lustre10", "scratch", user, "huggingface"),
                os.path.join("/scratch", user, "huggingface"),
            ]
        )

    for candidate in candidates:
        if os.name != "nt" and os.path.isdir(candidate):
            return candidate
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


def _candidate_vggt_repo_paths(explicit_repo_path=None):
    candidates = []

    if explicit_repo_path:
        candidates.append(explicit_repo_path)

    env_repo = os.environ.get("VGGT_REPO_PATH")
    if env_repo:
        candidates.append(env_repo)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parent_root = os.path.abspath(os.path.join(repo_root, ".."))
    user = os.environ.get("USER") or os.environ.get("USERNAME")

    candidates.extend(
        [
            os.path.join("/home", user, "links", "scratch", "vggt") if user else None,
            os.path.join(repo_root, "vggt"),
            os.path.join(repo_root, "external", "vggt"),
            os.path.join(parent_root, "vggt"),
            os.path.join(parent_root, "external", "vggt"),
        ]
    )

    if user:
        candidates.extend(
            [
                os.path.join("/home", user, "links", "scratch", "repos", "vggt"),
                os.path.join("/lustre10", "scratch", user, "vggt"),
                os.path.join("/lustre10", "scratch", user, "repos", "vggt"),
                os.path.join("/scratch", user, "vggt"),
            ]
        )

    seen = set()
    ordered = []
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _import_vggt_class(explicit_repo_path=None):
    last_exc = None

    try:
        from vggt.models.vggt import VGGT

        return VGGT, None
    except ImportError as exc:
        last_exc = exc

    for candidate in _candidate_vggt_repo_paths(explicit_repo_path):
        if not os.path.isdir(candidate):
            continue
        _maybe_add_repo_path(candidate)
        try:
            from vggt.models.vggt import VGGT

            return VGGT, candidate
        except ImportError as exc:
            last_exc = exc

    searched = "\n".join(f"  - {p}" for p in _candidate_vggt_repo_paths(explicit_repo_path))
    raise ImportError(
        "Official VGGT code could not be imported.\n"
        "Provide the repo path via `config.model.vggt_repo_path` or set `VGGT_REPO_PATH`.\n"
        "Searched:\n"
        f"{searched}"
    ) from last_exc


@dataclass
class DataConfig:
    data_root: str = "datasets/realestate10k_subset"
    batch_size: int = 1
    num_workers: int = 2
    shuffle: bool = True
    pin_memory: bool = False
    n_input_views: int = 8
    min_input_views: int = 4
    input_view_sampling: str = "pose_sparse"
    num_target_views: int = 1
    target_mode: str = "middle"
    exclude_target: bool = True


@dataclass
class ModelConfig:
    model_version: str = "v(-1)_gs"
    dino_name: str = "facebook/dinov2-large"
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
    save_every_n_epochs: int = 10
    emit_stride: int = 1
    max_scenes_per_epoch: int | None = 200
    save_dir: str = "outputs/re10k_debug"
    run_id: int = 2
    resume: bool = True
    resume_mode: str = "latest"
    metric_every: int = 5
    save_best_by: str = "loss"
    log_every_n_steps: int = 10
    enable_tensorboard: bool = True
    tensorboard_every_n_steps: int = 10
    tensorboard_log_dir: str | None = None
    visualize_every_n_epochs: int = 2
    visualize_every_n_steps: int = 20


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()
