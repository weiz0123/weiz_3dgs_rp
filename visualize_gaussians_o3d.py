import argparse
import json
import os
import random

import numpy as np
import torch
import open3d as o3d
import open3d.visualization.gui as gui
from PIL import Image

from configs.re10k_experiment import get_default_config
from eval_re10k_utils import compute_psnr
from gs_models.render_utils import rasterize_gaussians_single
from pipeline.data_loader import RealEstate10KDataset
from train import build_model, resolve_device
from train_re10k_utils import scene_to_model_inputs


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _default_ckpt_path(config):
    save_dir = os.path.join(config.training.save_dir, f"run_{config.training.run_id}")
    best_path = os.path.join(save_dir, "model_best.pth")
    latest_path = os.path.join(save_dir, "model_latest.pth")
    if os.path.isfile(best_path):
        return best_path
    return latest_path


def _make_point_cloud(means3d, colors, opacities, opacity_thresh=0.05, max_points=80000):
    pts = means3d.detach().float().cpu()
    cols = colors.detach().float().cpu().clamp(0.0, 1.0)
    opa = opacities.detach().float().cpu().view(-1)

    mask = opa >= opacity_thresh
    pts = pts[mask]
    cols = cols[mask]
    opa = opa[mask]

    if pts.shape[0] == 0:
        raise RuntimeError("No gaussians survived the opacity threshold for visualization.")

    if pts.shape[0] > max_points:
        idx = torch.topk(opa, k=max_points).indices
        pts = pts[idx]
        cols = cols[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.numpy())
    pcd.colors = o3d.utility.Vector3dVector(cols.numpy())
    return pcd, pts


def _create_camera_frustum(c2w, color, scale=0.12):
    c2w = _to_numpy(c2w).astype(np.float64)

    cam_pts = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [-0.5, -0.3, 1.0, 1.0],
            [0.5, -0.3, 1.0, 1.0],
            [0.5, 0.3, 1.0, 1.0],
            [-0.5, 0.3, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cam_pts[:, :3] *= scale
    world_pts = (c2w @ cam_pts.T).T[:, :3]

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]
    colors = [list(color) for _ in lines]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(world_pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _create_camera_marker(c2w, color, radius):
    center = _to_numpy(c2w)[:3, 3].astype(np.float64)
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.asarray(color, dtype=np.float64))
    mesh.translate(center)
    return mesh, center


def _create_gaussian_spheres(means3d, scales, colors, opacities, num_spheres, bbox_diag):
    if num_spheres <= 0:
        return [], None

    pts = means3d.detach().float().cpu()
    scl = scales.detach().float().cpu()
    cols = colors.detach().float().cpu().clamp(0.0, 1.0)
    opa = opacities.detach().float().cpu().view(-1)

    k = min(num_spheres, pts.shape[0])
    idx = torch.topk(opa, k=k).indices

    min_radius = max(1e-4, 0.002 * bbox_diag)
    max_radius = max(min_radius, 0.03 * bbox_diag)

    meshes = []
    merged_mesh = None
    for rank, i in enumerate(idx.tolist()):
        radius = float(scl[i].mean().item()) * 6.0
        radius = float(np.clip(radius, min_radius, max_radius))
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(cols[i].numpy())
        mesh.translate(pts[i].numpy())
        meshes.append((f"gaussian_sphere_{rank}", mesh))
        if merged_mesh is None:
            merged_mesh = mesh
        else:
            merged_mesh += mesh
    return meshes, merged_mesh


def _save_preview(preview_dir, rendered, target, meta, psnr):
    os.makedirs(preview_dir, exist_ok=True)

    rendered_np = rendered.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    target_np = target.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()

    Image.fromarray((rendered_np * 255).astype(np.uint8)).save(os.path.join(preview_dir, "rendered.png"))
    Image.fromarray((target_np * 255).astype(np.uint8)).save(os.path.join(preview_dir, "target.png"))

    with open(os.path.join(preview_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "scene_name": meta["scene_name"],
                "input_ids": list(meta["input_ids"]),
                "target_id": int(meta["target_id"]),
                "image_hw": list(meta["image_hw"]),
                "psnr": float(psnr),
            },
            f,
            indent=2,
        )


def _show_with_labels(named_geometries, labels):
    try:
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("3D Gaussians + Cameras", 1600, 960)
        vis.show_skybox(False)
        vis.show_ground = True

        for name, geom in named_geometries:
            vis.add_geometry(name, geom)

        for pos, text in labels:
            vis.add_3d_label(pos, text)

        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()
    except Exception as exc:
        print(f"Open3D label viewer unavailable, falling back to draw_geometries: {exc}")
        print("Camera labels:")
        for pos, text in labels:
            p = np.asarray(pos)
            print(f"  {text}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
        o3d.visualization.draw_geometries(
            [geom for _, geom in named_geometries],
            window_name="3D Gaussians + Cameras",
            width=1600,
            height=960,
        )


def _write_line_set(path, line_set):
    ok = o3d.io.write_line_set(path, line_set)
    if not ok:
        raise RuntimeError(f"Failed to write line set: {path}")


def _write_exports(export_dir, point_cloud, sphere_mesh, input_pose_records, target_pose_record, bbox_diag, labels):
    os.makedirs(export_dir, exist_ok=True)

    gaussian_ply = os.path.join(export_dir, "gaussians.ply")
    if not o3d.io.write_point_cloud(gaussian_ply, point_cloud):
        raise RuntimeError(f"Failed to write point cloud: {gaussian_ply}")

    if sphere_mesh is not None:
        sphere_ply = os.path.join(export_dir, "gaussian_spheres.ply")
        if not o3d.io.write_triangle_mesh(sphere_ply, sphere_mesh):
            raise RuntimeError(f"Failed to write mesh: {sphere_ply}")

    camera_points = []
    camera_colors = []
    camera_meta = []

    for role, frame_id, pose, color in input_pose_records:
        center = _to_numpy(pose)[:3, 3].astype(np.float64)
        camera_points.append(center)
        camera_colors.append(np.asarray(color, dtype=np.float64))
        camera_meta.append({"role": role, "frame_id": int(frame_id), "center": center.tolist()})
        frustum = _create_camera_frustum(pose, color=color, scale=max(0.05, 0.03 * bbox_diag))
        _write_line_set(os.path.join(export_dir, f"{role}_frame_{int(frame_id):04d}_frustum.ply"), frustum)

    role, frame_id, pose, color = target_pose_record
    center = _to_numpy(pose)[:3, 3].astype(np.float64)
    camera_points.append(center)
    camera_colors.append(np.asarray(color, dtype=np.float64))
    camera_meta.append({"role": role, "frame_id": int(frame_id), "center": center.tolist()})
    frustum = _create_camera_frustum(pose, color=color, scale=max(0.06, 0.035 * bbox_diag))
    _write_line_set(os.path.join(export_dir, f"{role}_frame_{int(frame_id):04d}_frustum.ply"), frustum)

    centers_pcd = o3d.geometry.PointCloud()
    centers_pcd.points = o3d.utility.Vector3dVector(np.stack(camera_points, axis=0))
    centers_pcd.colors = o3d.utility.Vector3dVector(np.stack(camera_colors, axis=0))
    centers_ply = os.path.join(export_dir, "camera_centers.ply")
    if not o3d.io.write_point_cloud(centers_ply, centers_pcd):
        raise RuntimeError(f"Failed to write point cloud: {centers_ply}")

    with open(os.path.join(export_dir, "camera_labels.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "cameras": camera_meta,
                "labels": [{"text": text, "position": np.asarray(pos).tolist()} for pos, text in labels],
            },
            f,
            indent=2,
        )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Visualize emitted 3D gaussians and camera poses with Open3D.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Checkpoint path. Defaults to model_best/model_latest under config save_dir.")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root. Defaults to config.data.data_root.")
    parser.add_argument("--scene-idx", type=int, default=None, help="Scene index to visualize. Defaults to a random scene.")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/auto. Defaults to config.training.device.")
    parser.add_argument("--emit-stride", type=int, default=None, help="Override emit stride for visualization.")
    parser.add_argument("--opacity-thresh", type=float, default=0.05, help="Opacity threshold for point cloud display.")
    parser.add_argument("--max-points", type=int, default=80000, help="Maximum number of gaussian centers to show in the point cloud.")
    parser.add_argument("--num-spheres", type=int, default=128, help="How many top-opacity gaussians to render as colored spheres.")
    parser.add_argument("--preview-dir", type=str, default="outputs/o3d_inspect", help="Directory to save rendered/target previews and metadata.")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory to export Open3D geometry files. Defaults to <preview-dir>/geometry.")
    parser.add_argument("--no-gui", action="store_true", help="Skip opening the Open3D GUI and only export files.")
    args = parser.parse_args()

    config = get_default_config()
    device = resolve_device(args.device or config.training.device)
    data_root = args.data_root or config.data.data_root
    ckpt_path = args.ckpt_path or _default_ckpt_path(config)
    emit_stride = args.emit_stride if args.emit_stride is not None else config.training.emit_stride

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dataset = RealEstate10KDataset(data_root)
    scene_idx = random.randrange(len(dataset)) if args.scene_idx is None else args.scene_idx
    scene = dataset[scene_idx]

    model = build_model(config, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    (
        input_imgs,
        input_Ks,
        input_poses,
        target_img,
        target_K,
        target_pose,
        meta,
    ) = scene_to_model_inputs(
        scene,
        device=device,
        target_mode=config.data.target_mode,
        exclude_target=config.data.exclude_target,
        n_input=config.data.n_input_views,
        min_input_views=config.data.min_input_views,
        input_view_sampling=config.data.input_view_sampling,
    )

    out = model(
        imgs=input_imgs,
        Ks=input_Ks,
        c2ws=input_poses,
        ref_idx=0,
        emit_stride=emit_stride,
    )

    means3d = out["means3D"][0]
    scales = out["scales"][0]
    rotations = out["rotations"][0]
    opacities = out["opacities"][0]
    colors = out["colors"][0]
    _ = rotations  # retained for future ellipsoid visualization and debugging

    rendered = rasterize_gaussians_single(
        means3D=means3d,
        scales=scales,
        rotations=out["rotations"][0],
        opacities=opacities,
        colors=colors,
        pose_c2w=target_pose,
        K=target_K,
        H=target_img.shape[-2],
        W=target_img.shape[-1],
    )
    if rendered is None:
        raise RuntimeError("Rasterization failed for visualization preview.")

    psnr = compute_psnr(rendered.unsqueeze(0).clamp(0, 1), target_img.clamp(0, 1))
    _save_preview(args.preview_dir, rendered, target_img[0], meta, psnr)

    pcd, pcd_pts = _make_point_cloud(
        means3d,
        colors,
        opacities,
        opacity_thresh=args.opacity_thresh,
        max_points=args.max_points,
    )
    bbox_min = pcd_pts.min(dim=0).values
    bbox_max = pcd_pts.max(dim=0).values
    bbox_diag = float(torch.norm(bbox_max - bbox_min, p=2).item())
    camera_radius = max(1e-4, 0.004 * bbox_diag)

    named_geometries = [("gaussian_points", pcd)]
    named_geometries.append(("world_frame", o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.1, 0.05 * bbox_diag))))
    sphere_geometries, merged_sphere_mesh = _create_gaussian_spheres(
        means3d, scales, colors, opacities, args.num_spheres, bbox_diag
    )
    named_geometries.extend(sphere_geometries)

    labels = []
    input_pose_records = []

    for local_idx, (frame_id, pose) in enumerate(zip(meta["input_ids"], input_poses[0])):
        color = (1.0, 0.1, 0.1)
        frustum = _create_camera_frustum(pose, color=color, scale=max(0.05, 0.03 * bbox_diag))
        named_geometries.append((f"input_frustum_{local_idx}", frustum))
        marker, center = _create_camera_marker(pose, color=color, radius=camera_radius)
        named_geometries.append((f"input_marker_{local_idx}", marker))
        role = "ref" if local_idx == 0 else "input"
        labels.append((center, f"{role}[{local_idx}] frame={frame_id}"))
        input_pose_records.append((role, frame_id, pose, color))

    target_color = (0.1, 0.3, 1.0)
    target_frustum = _create_camera_frustum(target_pose, color=target_color, scale=max(0.06, 0.035 * bbox_diag))
    named_geometries.append(("target_frustum", target_frustum))
    target_marker, target_center = _create_camera_marker(target_pose, color=target_color, radius=1.2 * camera_radius)
    named_geometries.append(("target_marker", target_marker))
    labels.append((target_center, f"target frame={meta['target_id']}"))

    export_dir = args.export_dir or os.path.join(args.preview_dir, "geometry")
    _write_exports(
        export_dir=export_dir,
        point_cloud=pcd,
        sphere_mesh=merged_sphere_mesh,
        input_pose_records=input_pose_records,
        target_pose_record=("target", meta["target_id"], target_pose, target_color),
        bbox_diag=bbox_diag,
        labels=labels,
    )

    print(f"Scene: {meta['scene_name']}")
    print(f"Inputs: {meta['input_ids']}  Target: {meta['target_id']}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Preview dir: {args.preview_dir}")
    print(f"Geometry export dir: {export_dir}")
    print(f"Preview PSNR: {psnr:.4f}")
    print(f"Visible gaussian centers: {len(pcd.points)}")
    print(f"Top gaussian spheres: {args.num_spheres}")

    if not args.no_gui:
        _show_with_labels(named_geometries, labels)
    else:
        print("GUI skipped due to --no-gui. Exported geometry files only.")


if __name__ == "__main__":
    main()
