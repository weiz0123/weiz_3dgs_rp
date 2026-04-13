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


def _to_dino_pca_numpy(feature_tensor):
    feature = feature_tensor.detach().cpu().float()
    if feature.ndim != 3:
        raise ValueError(f"Expected feature tensor [C, H, W], got {tuple(feature.shape)}")

    channels, height, width = feature.shape
    patch_tokens = feature.permute(1, 2, 0).reshape(height * width, channels)
    patch_tokens = patch_tokens - patch_tokens.mean(dim=0, keepdim=True)

    q = min(3, patch_tokens.shape[0], patch_tokens.shape[1])
    if q == 0:
        raise ValueError("Cannot run PCA on empty DINO feature tensor")

    _, _, v = torch.pca_lowrank(patch_tokens, q=q)
    pca_features = patch_tokens @ v[:, :q]

    if q < 3:
        padded = torch.zeros((pca_features.shape[0], 3), dtype=pca_features.dtype)
        padded[:, :q] = pca_features
        pca_features = padded

    pca_features = pca_features.reshape(height, width, 3)
    pca_features = pca_features - pca_features.amin(dim=(0, 1), keepdim=True)
    denom = pca_features.amax(dim=(0, 1), keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    pca_features = pca_features / denom

    return pca_features.numpy().clip(0.0, 1.0)


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


def _format_matrix_text(matrix_tensor):
    matrix = matrix_tensor.detach().cpu().float().numpy()
    return np.array2string(
        matrix,
        precision=4,
        suppress_small=True,
        max_line_width=120,
    )


def visualize_model_outputs(training_data, model_outputs, save_path=None):
    features = model_outputs["features"]
    depth = model_outputs["depth"]
    estimated_extrinsics = model_outputs["estimated_extrinsics"]
    num_views = training_data["train_images"].shape[0]
    fig, axes = plt.subplots(
        num_views,
        4,
        figsize=(21, 3.8 * num_views),
        constrained_layout=True,
    )

    if num_views == 1:
        axes = np.array([axes])

    for view_idx in range(num_views):
        image_ax = axes[view_idx, 0]
        feature_ax = axes[view_idx, 1]
        depth_ax = axes[view_idx, 2]
        pose_ax = axes[view_idx, 3]

        train_idx = training_data["train_indices"][view_idx]
        timestamp = float(training_data["train_timestamps"][view_idx])

        image_ax.imshow(_to_image_numpy(training_data["train_images"][view_idx]))
        image_ax.set_title(
            f"train idx={train_idx} | t={timestamp:.0f}",
            fontsize=10,
            loc="left",
        )
        image_ax.axis("off")

        feature_ax.imshow(_to_dino_pca_numpy(features[0, view_idx]))
        feature_ax.set_title("dino pca rgb", fontsize=10, loc="left")
        feature_ax.axis("off")

        depth_ax.imshow(_to_depth_numpy(depth[0, view_idx]), cmap="plasma")
        depth_ax.set_title("vggt depth", fontsize=10, loc="left")
        depth_ax.axis("off")

        gt_c2w = training_data["train_poses"][view_idx]
        gt_w2c = torch.linalg.inv(gt_c2w)
        est_w2c = estimated_extrinsics[0, view_idx]
        pose_ax.text(
            0.01,
            0.98,
            "gt extrinsic (w2c):\n"
            f"{_format_matrix_text(gt_w2c)}\n\n"
            "estimated extrinsic (w2c):\n"
            f"{_format_matrix_text(est_w2c)}",
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.5,
        )
        pose_ax.set_title("camera pose", fontsize=10, loc="left")
        pose_ax.axis("off")

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
            model_outputs = model(inputs)

        if not _HAS_VISUALIZED:
            save_path = None
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, "train_features_depth_preview.png")
            visualize_model_outputs(training_data, model_outputs, save_path=save_path)
            _HAS_VISUALIZED = True

        steps += 1

    steps = max(steps, 1)
    return {
        "loss_total": total_loss / steps,
        "loss_mse": total_mse / steps,
        "loss_l1": total_l1 / steps,
        "num_steps": steps,
    }
