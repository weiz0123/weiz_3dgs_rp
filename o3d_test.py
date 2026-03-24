import os
import random
import torch
import imageio
import numpy as np
import open3d as o3d

from pipeline.data_loader import RealEstate10KDataset
from gs_models.mvv3 import MultiViewDinoDepthToGaussians
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
        "target_id": target_id,
    }


# -------------------------------------------------
# Open3D helpers
# -------------------------------------------------
def build_colored_point_cloud(means3D, colors, opacities=None, opacity_thresh=0.05):
    pts = means3D.detach().float().cpu()
    cols = colors.detach().float().cpu()

    if cols.ndim == 3 and cols.shape[1] == 1:
        cols = cols.squeeze(1)

    cols = cols.clamp(0, 1)

    if opacities is not None:
        opa = opacities.detach().float().cpu().view(-1)
        mask = opa > opacity_thresh
        pts = pts[mask]
        cols = cols[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.numpy())
    pcd.colors = o3d.utility.Vector3dVector(cols.numpy())
    return pcd


def create_camera_frustum_lineset(c2w, color=(1.0, 0.0, 0.0), scale=0.15):
    """
    Build a simple camera frustum from a camera-to-world pose.
    Assumes +Z forward in camera local space for visualization purposes.
    """
    if isinstance(c2w, torch.Tensor):
        c2w = c2w.detach().cpu().numpy()

    c2w = np.asarray(c2w, dtype=np.float64)

    origin = np.array([0.0, 0.0, 0.0, 1.0])
    p1 = np.array([-0.5, -0.3, 1.0, 1.0])
    p2 = np.array([ 0.5, -0.3, 1.0, 1.0])
    p3 = np.array([ 0.5,  0.3, 1.0, 1.0])
    p4 = np.array([-0.5,  0.3, 1.0, 1.0])

    cam_pts = np.stack([origin, p1, p2, p3, p4], axis=0)
    cam_pts[:, :3] *= scale

    world_pts = (c2w @ cam_pts.T).T[:, :3]

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    colors = [list(color) for _ in lines]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(world_pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def create_camera_trajectory_lineset(c2ws, color=(0.0, 1.0, 0.0)):
    centers = []
    for pose in c2ws:
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        centers.append(np.asarray(pose[:3, 3], dtype=np.float64))

    if len(centers) < 2:
        return None

    lines = [[i, i + 1] for i in range(len(centers) - 1)]
    colors = [list(color) for _ in lines]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.stack(centers, axis=0))
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def show_scene_o3d(
    means3D,
    colors,
    opacities,
    cam_poses,
    input_cam_poses=None,
    target_pose=None,
    max_vis_points=30000,
):
    pts = means3D.detach().float().cpu()
    cols = colors.detach().float().cpu()

    if cols.ndim == 3 and cols.shape[1] == 1:
        cols = cols.squeeze(1)

    cols = cols.clamp(0, 1)
    opa = opacities.detach().float().cpu().view(-1)

    mask = opa > 0.05
    pts = pts[mask]
    cols = cols[mask]
    opa = opa[mask]

    if len(opa) > max_vis_points:
        idx = torch.topk(opa, max_vis_points).indices
        pts = pts[idx]
        cols = cols[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.numpy())
    pcd.colors = o3d.utility.Vector3dVector(cols.numpy())

    geometries = [pcd]

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord)

    traj = create_camera_trajectory_lineset(cam_poses, color=(0.0, 1.0, 0.0))
    if traj is not None:
        geometries.append(traj)

    for i, pose in enumerate(cam_poses):
        frustum = create_camera_frustum_lineset(
            pose,
            color=(0.0, 1.0, 0.0),
            scale=0.12
        )
        geometries.append(frustum)

    if input_cam_poses is not None:
        for pose in input_cam_poses:
            frustum = create_camera_frustum_lineset(
                pose,
                color=(1.0, 0.0, 0.0),
                scale=0.16
            )
            geometries.append(frustum)

    if target_pose is not None:
        geometries.append(
            create_camera_frustum_lineset(
                target_pose,
                color=(0.0, 0.0, 1.0),
                scale=0.2
            )
        )

    print("🧠 Opening Open3D viewer...", flush=True)
    print("   Green  = render trajectory cameras", flush=True)
    print("   Red    = input cameras", flush=True)
    print("   Blue   = target camera", flush=True)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="3DGS Point Cloud + Camera Poses",
        width=1400,
        height=900,
    )


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
    show_o3d=True,
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
            best_pack = (
                out,
                meta,
                input_poses.detach().clone(),
                target_pose.detach().clone(),
            )

    print("🏆 Best trial selected", flush=True)

    out, meta, best_input_poses, best_target_pose = best_pack

    means3D = out["means3D"][0].to(device)
    scales = out["scales"][0].to(device)
    rotations = out["rotations"][0].to(device)
    opacities = out["opacities"][0].to(device)
    colors = out["colors"][0].to(device)

    K = 40000
    scores = opacities.squeeze()
    idx = torch.topk(scores, min(K, scores.shape[0])).indices

    means3D = means3D[idx]
    scales = scales[idx]
    rotations = rotations[idx]
    opacities = opacities[idx]
    colors = colors[idx]

    print("🔧 Using", len(idx), "gaussians", flush=True)

    all_ids = list(range(len(scene["poses"])))
    render_ids = [i for i in all_ids if i not in meta["input_ids"]]

    cam_poses = scene["poses"][render_ids].to(device)
    cam_Ks = scene["intrinsics"][render_ids].to(device)
    timestamps = scene["timestamps"][render_ids]

    order = torch.argsort(timestamps)
    cam_poses = cam_poses[order]
    cam_Ks = cam_Ks[order]

    cam_Ks = intrinsics_to_pixel(cam_Ks, meta["H"], meta["W"])

    max_frames = 60
    cam_poses = cam_poses[:max_frames]
    cam_Ks = cam_Ks[:max_frames]

    if show_o3d:
        show_scene_o3d(
            means3D=means3D,
            colors=colors,
            opacities=opacities,
            cam_poses=cam_poses,
            input_cam_poses=best_input_poses[0],
            target_pose=best_target_pose,
            max_vis_points=30000,
        )

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

        del rendered
        if torch.cuda.is_available():
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
    run(
        ckpt_path=r"C:\Users\zhouw\Desktop\3DGS\weiz_3dgs_rp\outputs\model_epoch_226.pth",
        data_root="datasets/realestate10k_subset",
        save_dir="outputs/dataset_pose_video",
        device="cuda" if torch.cuda.is_available() else "cpu",
        show_o3d=True,
    )
