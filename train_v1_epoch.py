import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


_HAS_VISUALIZED = False


def _to_image_numpy(image_tensor):
    image = image_tensor.detach().cpu()
    if image.ndim == 3:
        image = image.permute(1, 2, 0)
    return image.numpy().clip(0.0, 1.0)


def _to_feature_numpy(feature_tensor):
    feature = feature_tensor.detach().cpu().float()
    if feature.ndim != 3:
        raise ValueError(f"Expected feature tensor [C, H, W], got {tuple(feature.shape)}")
    feature = feature.mean(dim=0)
    feature = feature.numpy()
    feature = feature - feature.min()
    denom = feature.max()
    if denom > 0:
        feature = feature / denom
    return feature


def _to_depth_numpy(depth_tensor):
    depth = depth_tensor.detach().cpu().float()
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    elif depth.ndim != 2:
        raise ValueError(f"Expected depth tensor [1, H, W] or [H, W], got {tuple(depth.shape)}")

    depth = depth.numpy()
    depth = depth - depth.min()
    denom = depth.max()
    if denom > 0:
        depth = depth / denom
    return depth


def visualize_model_outputs(training_data, features, depth, save_path=None):
    num_views = training_data["train_images"].shape[0]
    fig, axes = plt.subplots(
        num_views,
        3,
        figsize=(15, 3.5 * num_views),
        constrained_layout=True,
    )

    if num_views == 1:
        axes = np.array([axes])

    for view_idx in range(num_views):
        image_ax = axes[view_idx, 0]
        feature_ax = axes[view_idx, 1]
        depth_ax = axes[view_idx, 2]

        train_idx = training_data["train_indices"][view_idx]
        timestamp = float(training_data["train_timestamps"][view_idx])

        image_ax.imshow(_to_image_numpy(training_data["train_images"][view_idx]))
        image_ax.set_title(
            f"train idx={train_idx} | t={timestamp:.0f}",
            fontsize=10,
            loc="left",
        )
        image_ax.axis("off")

        feature_ax.imshow(_to_feature_numpy(features[0, view_idx]), cmap="viridis")
        feature_ax.set_title("dino feature mean", fontsize=10, loc="left")
        feature_ax.axis("off")

        depth_ax.imshow(_to_depth_numpy(depth[0, view_idx]), cmap="plasma")
        depth_ax.set_title("vggt depth", fontsize=10, loc="left")
        depth_ax.axis("off")

    fig.suptitle(
        f"Scene: {training_data['scene']} | target idx={training_data['target_idx']}",
        fontsize=14,
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def train_epoch(model, data_manager, dataloader, optimizer, device, config=None, output_dir=None):
    global _HAS_VISUALIZED

    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="train_epoch", leave=False):
        scene = {
            "scene": batch["scene"][0],
            "images": batch["images"][0],
            "intrinsics": batch["intrinsics"][0],
            "poses": batch["poses"][0],
            "timestamps": batch["timestamps"][0],
        }

        training_data = data_manager.build_training_data(
            scene,
            config.data.n_input_views,
        )

        inputs = training_data["train_images"].to(device)

        with torch.no_grad():
            features, depth = model(inputs)

        if not _HAS_VISUALIZED:
            save_path = None
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, "train_features_depth_preview.png")
            visualize_model_outputs(training_data, features, depth, save_path=save_path)
            _HAS_VISUALIZED = True

        steps += 1

    steps = max(steps, 1)
    return {
        "loss_total": total_loss / steps,
        "loss_mse": total_mse / steps,
        "loss_l1": total_l1 / steps,
        "num_steps": steps,
    }
