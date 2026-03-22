import os
import random
import math
import torch
import imageio
import torchvision.utils as vutils

from pipeline.data_loader import RealEstate10KDataset
from gs_models.mvv2 import MultiViewDinoDepthToGaussians
from gs_models.render_utils import rasterize_gaussians_single

print("🚀 Multi-trial VIDEO inference started")


# -------------------------------------------------
# Intrinsics conversion
# -------------------------------------------------
def intrinsics_to_pixel(K, H, W):
    Kp = K.clone()
    Kp[:, 0, 0] *= W
    Kp[:, 1, 1] *= H
    Kp[:, 0, 2] *= W
    Kp[:, 1, 2] *= H
    return Kp


# -------------------------------------------------
# RANDOM WIDE BASELINE INPUT
# -------------------------------------------------
def scene_to_inputs_random(scene, device="cuda", n_input=4):
    images = scene["images"]
    Ks = scene["intrinsics"]
    poses = scene["poses"]

    T, _, H, W = images.shape
    Ks = intrinsics_to_pixel(Ks, H, W)

    target_id = T // 2
    candidate_ids = [i for i in range(T) if i != target_id]

    input_ids = random.sample(candidate_ids, min(n_input, len(candidate_ids)))

    input_imgs = images[input_ids].unsqueeze(0).to(device)
    input_Ks = Ks[input_ids].unsqueeze(0).to(device)
    input_poses = poses[input_ids].unsqueeze(0).to(device)

    target_img = images[target_id].unsqueeze(0).to(device)
    target_K = Ks[target_id].to(device)
    target_pose = poses[target_id].to(device)

    return input_imgs, input_Ks, input_poses, target_img, target_K, target_pose, {
        "H": H,
        "W": W,
        "target_id": target_id,
        "input_ids": input_ids,
    }


# -------------------------------------------------
# SPIRAL CAMERA
# -------------------------------------------------
def generate_spiral(center_pose, n_frames=120):
    poses = []
    center = center_pose[:3, 3]
    base_R = center_pose[:3, :3]

    for i in range(n_frames):
        t = i / n_frames
        theta = 2 * math.pi * t

        x = 0.4 * math.cos(theta)
        z = 0.4 * math.sin(theta)
        y = 0.2 * math.sin(2 * theta)
        d = 0.15 * math.sin(theta)

        offset = torch.tensor([x, y, z + d], device=center_pose.device)

        new_pose = center_pose.clone()
        new_pose[:3, 3] = center + offset

        # small rotation
        angle = 0.1 * math.sin(theta)
        rot = torch.tensor([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)]
        ], device=center_pose.device)

        new_pose[:3, :3] = rot @ base_R
        poses.append(new_pose)

    return poses


# -------------------------------------------------
# MAIN
# -------------------------------------------------
@torch.no_grad()
def run(
    ckpt_path,
    data_root="datasets/realestate10k_subset",
    save_dir="outputs/multi_trial_video",
    n_input=4,
    num_trials=5,
    num_scenes=3,
    device="cuda",
):
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Load model
    # -------------------------
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

    print("✅ Model loaded")

    dataset = RealEstate10KDataset(data_root)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_scenes]

    print("🎯 Scenes:", indices)

    for scene_idx in indices:
        print(f"\n🔥 Scene {scene_idx}")

        scene = dataset[scene_idx]

        best_score = -1
        best_gaussians = None
        best_pose = None
        best_K = None
        best_meta = None

        # -------------------------
        # MULTI TRIAL
        # -------------------------
        for trial in range(num_trials):
            print(f"   🔁 Trial {trial}")

            (
                input_imgs,
                input_Ks,
                input_poses,
                target_img,
                target_K,
                target_pose,
                meta,
            ) = scene_to_inputs_random(scene, device, n_input)

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

            score = opacities.mean().item()
            print("      score:", score)

            if score > best_score:
                best_score = score
                best_gaussians = (means3D, scales, rotations, opacities, colors)
                best_pose = target_pose
                best_K = target_K
                best_meta = meta

        # -------------------------
        # VIDEO FROM BEST
        # -------------------------
        print("🏆 Using best trial")

        means3D, scales, rotations, opacities, colors = best_gaussians

        pose = best_pose.view(4, 4).float()
        K_mat = best_K.view(3, 3).float()

        cam_poses = generate_spiral(pose)

        frames = []

        for cam_pose in cam_poses:
            rendered = rasterize_gaussians_single(
                means3D=means3D,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                colors=colors,
                pose_c2w=cam_pose,
                K=K_mat,
                H=best_meta["H"],
                W=best_meta["W"],
            )

            if rendered is None:
                continue

            frame = rendered.clamp(0, 1)
            frame = frame.permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype("uint8")

            frames.append(frame)

        video_path = os.path.join(save_dir, f"scene_{scene_idx}.mp4")

        writer = imageio.get_writer(video_path, fps=20, codec="libx264")
        for f in frames:
            writer.append_data(f)
        writer.close()

        print("🎥 Saved:", video_path)

    print("\n🎉 DONE")


# -------------------------------------------------
# ENTRY
# -------------------------------------------------
if __name__ == "__main__":
    print("asdf")
    run(
        ckpt_path=r"/home/weiz/links/scratch/weiz_3dgs_rp/outputs/re10k_debug/run_2/model_epoch_226.pth",
        data_root="datasets/realestate10k_subset",
        save_dir="outputs/multi_trial_video",
        n_input=4,
        num_trials=5,
        num_scenes=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )