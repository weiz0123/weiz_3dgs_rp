import os
import random
import torch
import imageio

from pipeline.data_loader import RealEstate10KDataset
from gs_models.mvv2 import MultiViewDinoDepthToGaussians
from gs_models.render_utils import rasterize_gaussians_single

print("🚀 Dataset-pose VIDEO inference started", flush=True)


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
# RANDOM INPUT SELECTION
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
        "input_ids": input_ids,
    }


# -------------------------------------------------
# MAIN
# -------------------------------------------------
@torch.no_grad()
def run(
    ckpt_path,
    data_root="datasets/realestate10k_subset",
    save_dir="outputs/dataset_pose_video",
    n_input=4,
    num_trials=2,
    num_scenes=1,
    device="cuda",
):
    os.makedirs(save_dir, exist_ok=True)

    print("📦 Loading model...", flush=True)
    model = MultiViewDinoDepthToGaussians(
        dino_name="facebook/dinov2-base",
        freeze_dino=True,
        num_depth_bins=48,
        depth_min=0.5,
        depth_max=20.0,
        feat_reduce_dim=128,
    ).to(device)

    print("📥 Loading checkpoint...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("✅ Model ready", flush=True)

    print("📚 Loading dataset...", flush=True)
    dataset = RealEstate10KDataset(data_root)

    scene_idx = random.randint(0, len(dataset) - 1)
    print("🎯 Scene:", scene_idx, flush=True)

    scene = dataset[scene_idx]

    best_score = -1
    best_pack = None

    # -------------------------
    # MULTI-TRIAL
    # -------------------------
    for trial in range(num_trials):
        print(f"🔁 Trial {trial}", flush=True)

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

        score = out["opacities"][0].mean().item()
        print("   score:", score, flush=True)

        if score > best_score:
            best_score = score
            best_pack = (out, meta)

    print("🏆 Best trial selected", flush=True)

    out, meta = best_pack

    # -------------------------
    # Move to device
    # -------------------------
    means3D = out["means3D"][0].to(device)
    scales = out["scales"][0].to(device)
    rotations = out["rotations"][0].to(device)
    opacities = out["opacities"][0].to(device)
    colors = out["colors"][0].to(device)

    # -------------------------
    # 🔥 Reduce Gaussians (VERY IMPORTANT)
    # -------------------------
    K = 40000  # safe for laptop GPU

    scores = opacities.squeeze()
    idx = torch.topk(scores, min(K, scores.shape[0])).indices

    means3D = means3D[idx]
    scales = scales[idx]
    rotations = rotations[idx]
    opacities = opacities[idx]
    colors = colors[idx]

    print("🔧 Using", len(idx), "gaussians", flush=True)

    # -------------------------
    # Dataset poses
    # -------------------------
    all_ids = list(range(len(scene["poses"])))
    render_ids = [i for i in all_ids if i not in meta["input_ids"]]

    cam_poses = scene["poses"][render_ids].to(device)
    cam_Ks = scene["intrinsics"][render_ids].to(device)
    timestamps = scene["timestamps"][render_ids]

    order = torch.argsort(timestamps)
    cam_poses = cam_poses[order]
    cam_Ks = cam_Ks[order]

    cam_Ks = intrinsics_to_pixel(cam_Ks, meta["H"], meta["W"])

    # -------------------------
    # 🔥 Limit frames (VERY IMPORTANT)
    # -------------------------
    max_frames = 60
    cam_poses = cam_poses[:max_frames]
    cam_Ks = cam_Ks[:max_frames]

    print("🎬 Rendering", len(cam_poses), "frames...", flush=True)

    frames = []

    for i, (cam_pose, cam_K) in enumerate(zip(cam_poses, cam_Ks)):
        print(f"   frame {i}", flush=True)

        rendered = rasterize_gaussians_single(
            means3D,
            scales,
            rotations,
            opacities,
            colors,
            cam_pose,
            cam_K,
            meta["H"],
            meta["W"],
        )

        if rendered is None:
            continue

        frame = rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype("uint8")
        frames.append(frame)

        # 🔥 CRITICAL: prevent OOM
        del rendered
        torch.cuda.empty_cache()

    video_path = os.path.join(save_dir, "result.mp4")

    print("💾 Writing video...", flush=True)
    writer = imageio.get_writer(video_path, fps=20, codec="libx264")

    for f in frames:
        writer.append_data(f)

    writer.close()

    print("🎥 Saved:", video_path, flush=True)
    print("\n🎉 DONE", flush=True)


# -------------------------------------------------
# ENTRY
# -------------------------------------------------
if __name__ == "__main__":
    print("asdf")
    run(
        ckpt_path=r"/home/weiz/links/scratch/weiz_3dgs_rp/outputs/re10k_debug/run_2/model_epoch_226.pth",
        data_root="datasets/realestate10k_subset",
        save_dir="outputs/dataset_pose_video",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )