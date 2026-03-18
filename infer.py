import os
import random
import torch
import torchvision.utils as vutils

from pipeline.data_loader import RealEstate10KDataset
from gs_models.mvv2 import MultiViewDinoDepthToGaussians
from gs_models.render_utils import rasterize_gaussians_single

print("🚀 inference script started")
import random
import torch

print("random test:", random.random())
print("torch test:", torch.rand(1))

def intrinsics_to_pixel(K: torch.Tensor, H: int, W: int):
    Kp = K.clone()
    Kp[:, 0, 0] *= W
    Kp[:, 1, 1] *= H
    Kp[:, 0, 2] *= W
    Kp[:, 1, 2] *= H
    return Kp


def scene_to_model_inputs_random_scene_consistent(
    scene,
    device="cuda",
    n_input=4,
    target_mode="random",   # "random" or "middle"
    exclude_target=True,
):
    """
    Same preprocessing logic as training:
    - convert normalized intrinsics to pixel intrinsics
    - choose target
    - choose nearest input frames to target
    """
    images = scene["images"]          # [T,3,H,W]
    Ks = scene["intrinsics"]          # normalized intrinsics
    poses = scene["poses"]            # [T,4,4]
    timestamps = scene["timestamps"]

    T, _, H, W = images.shape
    Ks = intrinsics_to_pixel(Ks, H, W)

    if target_mode == "middle":
        target_id = T // 2
    elif target_mode == "random":
        target_id = random.randint(0, T - 1)
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    if exclude_target:
        candidate_ids = [i for i in range(T) if i != target_id]
    else:
        candidate_ids = list(range(T))

    # Keep this consistent with training
    candidate_ids = sorted(candidate_ids, key=lambda i: abs(i - target_id))

    if len(candidate_ids) >= n_input:
        input_ids = candidate_ids[:n_input]
    else:
        if len(candidate_ids) == 0:
            input_ids = [target_id] * n_input
        else:
            input_ids = candidate_ids + [candidate_ids[-1]] * (n_input - len(candidate_ids))

    input_imgs = images[input_ids].unsqueeze(0).to(device)   # [1,V,3,H,W]
    input_Ks = Ks[input_ids].unsqueeze(0).to(device)         # [1,V,3,3]
    input_poses = poses[input_ids].unsqueeze(0).to(device)   # [1,V,4,4]

    target_img = images[target_id].unsqueeze(0).to(device)   # [1,3,H,W]
    target_K = Ks[target_id].to(device)                      # [3,3]
    target_pose = poses[target_id].to(device)                # [4,4]

    meta = {
        "scene_name": scene["scene"],
        "target_id": target_id,
        "input_ids": input_ids,
        "target_timestamp": timestamps[target_id],
        "image_hw": (H, W),
    }

    return input_imgs, input_Ks, input_poses, target_img, target_K, target_pose, meta


@torch.no_grad()
def run_inference(
    ckpt_path,
    data_root="datasets/realestate10k_subset",
    save_dir="outputs/inference_debug",
    n_input=4,
    num_samples=20,
    device="cuda",
    seed=None,                 # set to int for reproducibility
    target_mode="random",      # "random" or "middle"
):
    os.makedirs(save_dir, exist_ok=True)

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    model = MultiViewDinoDepthToGaussians(
        dino_name="facebook/dinov2-base",
        freeze_dino=True,
        num_depth_bins=48,
        depth_min=0.5,
        depth_max=20.0,
        feat_reduce_dim=128,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("✅ Loaded model:", ckpt_path)

    dataset = RealEstate10KDataset(data_root)
    dataset_size = len(dataset)
    print("📦 Dataset size:", dataset_size)

    if dataset_size == 0:
        print("❌ Dataset is empty")
        return

    # Random scenes
    all_indices = list(range(dataset_size))
    if num_samples >= dataset_size:
        indices = all_indices
        random.shuffle(indices)
    else:
        indices = random.sample(all_indices, k=num_samples)

    print("🎯 Selected scene indices:", indices)

    for sample_i, idx in enumerate(indices):
        print(f"\n🔥 [{sample_i+1}/{len(indices)}] Processing scene index {idx}")

        scene = dataset[idx]

        (
            input_imgs,
            input_Ks,
            input_poses,
            target_img,
            target_K,
            target_pose,
            meta,
        ) = scene_to_model_inputs_random_scene_consistent(
            scene,
            device=device,
            n_input=n_input,
            target_mode=target_mode,
            exclude_target=True,
        )

        print("Scene name:", meta["scene_name"])
        print("Target frame:", meta["target_id"])
        print("Input frames:", meta["input_ids"])
        print("Input imgs shape:", tuple(input_imgs.shape))
        print("Target K shape:", tuple(target_K.shape))
        print("Target pose shape:", tuple(target_pose.shape))

        out = model(
            imgs=input_imgs,
            Ks=input_Ks,
            c2ws=input_poses,
            ref_idx=0,
            emit_stride=1,
        )

        means3D = out["means3D"][0]
        scales = out["scales"][0]
        rotations = out["rotations"][0]
        opacities = out["opacities"][0]
        colors = out["colors"][0]

        pose = target_pose.view(4, 4).float()
        K_mat = target_K.view(3, 3).float()

        rendered = rasterize_gaussians_single(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            colors=colors,
            pose_c2w=pose,
            K=K_mat,
            H=target_img.shape[-2],
            W=target_img.shape[-1],
        )

        if rendered is None:
            print("❌ Rasterization failed")
            continue

        # Save with sample_i too, so nothing gets overwritten ambiguously
        render_path = os.path.join(save_dir, f"{sample_i:03d}_scene_{idx}_render.png")
        target_path = os.path.join(save_dir, f"{sample_i:03d}_scene_{idx}_target.png")
        input_grid_path = os.path.join(save_dir, f"{sample_i:03d}_scene_{idx}_inputs.png")

        vutils.save_image(rendered.clamp(0, 1), render_path)
        vutils.save_image(target_img[0], target_path)

        # save a grid of inputs too
        vutils.save_image(
            input_imgs[0],
            input_grid_path,
            nrow=min(n_input, 4),
        )

        print("💾 Saved:", render_path)
        print("💾 Saved:", target_path)
        print("💾 Saved:", input_grid_path)

    print("\n🎉 Inference done.")


if __name__ == "__main__":
    print("🔥 Running inference...")

    run_inference(
        ckpt_path=r"C:\Users\zhouw\Desktop\3DGS\weiz_3dgs_rp\outputs\model_epoch_226.pth",
        data_root="datasets/realestate10k_subset",
        save_dir="outputs/inference_debug",
        n_input=4,
        num_samples=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=None,          # use 42 if you want repeatable debugging
        target_mode="random"
    )