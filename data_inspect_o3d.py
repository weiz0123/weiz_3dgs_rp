import argparse

import cv2
import numpy as np
import open3d as o3d

from configs.re10k_experiment import get_default_config
from pipeline.data_loader import RealEstate10KDataset


def _to_numpy_image(image):
    if hasattr(image, "detach"):
        image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = np.asarray(image, dtype=np.float32)
    return np.clip(image, 0.0, 1.0)


def _select_frame_indices(num_frames, max_frames):
    if max_frames is None or max_frames <= 0 or num_frames <= max_frames:
        return list(range(num_frames))

    positions = np.linspace(0, num_frames - 1, num=max_frames)
    indices = sorted({int(round(pos)) for pos in positions.tolist()})
    return indices


def _camera_center_and_axes(c2w):
    c2w = np.asarray(c2w, dtype=np.float64)
    center = c2w[:3, 3]
    rot = c2w[:3, :3]
    right = rot[:, 0]
    up = rot[:, 1]
    forward = rot[:, 2]
    return center, right, up, forward


def _make_camera_frustum(c2w, color=(0.1, 0.6, 1.0), scale=0.12):
    center, right, up, forward = _camera_center_and_axes(c2w)

    plane_center = center + forward * scale
    half_w = scale * 0.6
    half_h = scale * 0.4

    tl = plane_center - right * half_w + up * half_h
    tr = plane_center + right * half_w + up * half_h
    br = plane_center + right * half_w - up * half_h
    bl = plane_center - right * half_w - up * half_h

    points = np.stack([center, tl, tr, br, bl], axis=0)
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1))
    )
    return line_set


def _make_camera_axes(c2w, size=0.08):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(np.asarray(c2w, dtype=np.float64))
    return frame


def _make_image_plane(c2w, image, plane_scale=0.18, stride=8):
    image = _to_numpy_image(image)
    h, w, _ = image.shape

    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)

    center, right, up, forward = _camera_center_and_axes(c2w)
    plane_center = center + forward * plane_scale

    aspect = h / max(w, 1)
    plane_w = plane_scale * 1.4
    plane_h = plane_w * aspect

    points = []
    colors = []

    for y in ys:
        v = (y / max(h - 1, 1)) - 0.5
        for x in xs:
            u = (x / max(w - 1, 1)) - 0.5
            world_point = (
                plane_center
                + right * (u * plane_w)
                - up * (v * plane_h)
            )
            points.append(world_point)
            colors.append(image[y, x])

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return cloud


def _resolve_scene_index(dataset, scene_name):
    if scene_name is None:
        return 0

    scene_names = [scene_path.name for scene_path in dataset.scenes]
    if scene_name not in scene_names:
        raise ValueError(
            f"Scene '{scene_name}' not found. First 10 available scenes: {scene_names[:10]}"
        )
    return scene_names.index(scene_name)


def main():
    parser = argparse.ArgumentParser(description="Inspect RealEstate10K cameras in Open3D")
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Exact scene name to visualize. Defaults to the first scene.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to visualize. Defaults to config.data.n_input_views. Use 0 or a negative value for all frames.",
    )
    parser.add_argument(
        "--image_stride",
        type=int,
        default=10,
        help="Subsampling stride for image-plane points.",
    )
    args = parser.parse_args()

    config = get_default_config()
    dataset = RealEstate10KDataset(config.data.data_root)
    max_frames = config.data.n_input_views if args.max_frames is None else args.max_frames

    scene_idx = _resolve_scene_index(dataset, args.scene_name)
    scene = dataset[scene_idx]

    frame_indices = _select_frame_indices(scene["images"].shape[0], max_frames)

    print(f"Scene: {scene['scene']}")
    print(f"Total frames in scene: {scene['images'].shape[0]}")
    print(f"Visualized frame count: {len(frame_indices)}")
    print(f"Frame indices: {frame_indices}")

    geometries = []
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    geometries.append(world_frame)

    for frame_idx in frame_indices:
        pose = scene["poses"][frame_idx].detach().cpu().numpy()
        image = scene["images"][frame_idx]

        geometries.append(_make_camera_axes(pose, size=0.06))
        geometries.append(_make_camera_frustum(pose, scale=0.10))
        geometries.append(
            _make_image_plane(
                pose,
                image,
                plane_scale=0.16,
                stride=max(args.image_stride, 1),
            )
        )

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"RealEstate10K Scene: {scene['scene']}",
    )


if __name__ == "__main__":
    main()
