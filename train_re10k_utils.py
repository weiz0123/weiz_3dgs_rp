import random
import torch
import torchvision.utils as vutils
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils
from PIL import Image
from pipeline.data_loader import RealEstate10KDataset
'''
used to manipulate loaded data into training model
'''

def intrinsics_to_pixel(K: torch.Tensor, H: int, W: int):
    '''
        fx ≈ 0.49
        fy ≈ 0.87
        cx ≈ 0.5
        cy ≈ 0.5
            need to be noramlized
        fx_pixel = fx * W
        fy_pixel = fy * H
        cx_pixel = cx * W
        cy_pixel = cy * H
    '''
    Kp = K.clone()
    Kp[:, 0, 0] *= W
    Kp[:, 1, 1] *= H
    Kp[:, 0, 2] *= W
    Kp[:, 1, 2] *= H
    return Kp


def _evenly_sample_ids(ids, num_samples):
    if num_samples <= 0 or len(ids) == 0:
        return []
    if len(ids) <= num_samples:
        return list(ids)

    positions = torch.linspace(0, len(ids) - 1, steps=num_samples)
    selected = []
    used = set()

    for pos in positions.tolist():
        idx = int(round(pos))
        idx = max(0, min(idx, len(ids) - 1))

        if idx in used:
            for offset in range(1, len(ids)):
                left = idx - offset
                right = idx + offset
                if left >= 0 and left not in used:
                    idx = left
                    break
                if right < len(ids) and right not in used:
                    idx = right
                    break

        used.add(idx)
        selected.append(ids[idx])

    return selected


def _camera_centers_from_poses(poses):
    return poses[:, :3, 3]


def _prioritize_reference_view(input_ids, target_id, poses=None):
    if len(input_ids) <= 1:
        return list(input_ids)

    if poses is None:
        best_id = min(input_ids, key=lambda idx: abs(idx - target_id))
    else:
        centers = _camera_centers_from_poses(poses)
        target_center = centers[target_id]
        best_id = min(
            input_ids,
            key=lambda idx: torch.norm(centers[idx] - target_center, p=2).item(),
        )

    ordered = [best_id]
    ordered.extend(idx for idx in input_ids if idx != best_id)
    return ordered


def _select_pose_sparse_ids(candidate_ids, poses, target_id, n_input):
    if len(candidate_ids) < n_input:
        raise ValueError(
            f"Need at least {n_input} candidate input views, got {len(candidate_ids)}"
        )

    centers = _camera_centers_from_poses(poses)
    target_center = centers[target_id]
    candidate_centers = centers[candidate_ids]

    target_dist = torch.norm(candidate_centers - target_center.unsqueeze(0), dim=1)
    seed_order = torch.argsort(target_dist, descending=True)

    selected_local = [int(seed_order[0].item())]
    remaining = set(range(len(candidate_ids))) - set(selected_local)

    while len(selected_local) < n_input and remaining:
        best_idx = None
        best_score = None

        for idx in remaining:
            center = candidate_centers[idx]
            dist_to_target = torch.norm(center - target_center, p=2).item()
            diversity = min(
                torch.norm(center - candidate_centers[j], p=2).item()
                for j in selected_local
            )
            score = 0.5 * dist_to_target + 0.5 * diversity

            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        selected_local.append(best_idx)
        remaining.remove(best_idx)

    selected_ids = [candidate_ids[idx] for idx in selected_local]
    return sorted(selected_ids)


def _select_input_ids(candidate_ids, target_id, n_input, sampling, poses=None):
    if len(candidate_ids) < n_input:
        raise ValueError(
            f"Need at least {n_input} candidate input views, got {len(candidate_ids)}"
        )

    if sampling == "nearest":
        ordered = sorted(candidate_ids, key=lambda i: abs(i - target_id))
        return ordered[:n_input]

    if sampling == "sparse":
        before = [i for i in candidate_ids if i < target_id]
        after = [i for i in candidate_ids if i > target_id]

        left_quota = n_input // 2
        right_quota = n_input - left_quota

        left_ids = _evenly_sample_ids(before, min(left_quota, len(before)))
        right_ids = _evenly_sample_ids(after, min(right_quota, len(after)))

        remaining = n_input - len(left_ids) - len(right_ids)
        if remaining > 0:
            leftovers = [i for i in candidate_ids if i not in set(left_ids + right_ids)]
            extra = _evenly_sample_ids(leftovers, remaining)
        else:
            extra = []

        return sorted(left_ids + right_ids + extra)

    if sampling == "pose_sparse":
        if poses is None:
            raise ValueError("poses are required for pose_sparse input sampling")
        return _select_pose_sparse_ids(
            candidate_ids=candidate_ids,
            poses=poses,
            target_id=target_id,
            n_input=n_input,
        )

    raise ValueError(f"Unknown input sampling mode: {sampling}")


def scene_to_model_inputs(
    scene,
    device="cuda",
    target_mode="middle",
    exclude_target=True,
    n_input=3,
    min_input_views=None,
    input_view_sampling="nearest",
):

    # -------------------------------------------------
    # Handle batch automatically
    # -------------------------------------------------

    if isinstance(scene, list):

        imgs = []
        Ks_out = []
        poses_out = []
        targets = []
        target_Ks = []
        target_poses = []
        metas = []

        for s in scene:

            images = s["images"]
            Ks = s["intrinsics"]
            poses = s["poses"]
            timestamps = s["timestamps"]

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

            required_inputs = n_input if min_input_views is None else max(n_input, min_input_views)
            if len(candidate_ids) < required_inputs:
                raise ValueError(
                    f"Skipping scene {s['scene']}: need at least {required_inputs} candidate views, got {len(candidate_ids)}"
                )
            input_ids = _select_input_ids(
                candidate_ids=candidate_ids,
                target_id=target_id,
                n_input=n_input,
                sampling=input_view_sampling,
                poses=poses,
            )
            input_ids = _prioritize_reference_view(input_ids, target_id, poses=poses)

            imgs.append(images[input_ids])
            Ks_out.append(Ks[input_ids])
            poses_out.append(poses[input_ids])

            targets.append(images[target_id])
            target_Ks.append(Ks[target_id])
            target_poses.append(poses[target_id])

            metas.append({
                "scene_name": s["scene"],
                "target_id": target_id,
                "input_ids": input_ids,
                "target_timestamp": timestamps[target_id],
                "image_hw": (H, W),
            })

        input_imgs = torch.stack(imgs).to(device)
        input_Ks = torch.stack(Ks_out).to(device)
        input_poses = torch.stack(poses_out).to(device)

        target_img = torch.stack(targets).to(device)
        target_K = torch.stack(target_Ks).to(device)
        target_pose = torch.stack(target_poses).to(device)

        return input_imgs, input_Ks, input_poses, target_img, target_K, target_pose, metas

    # -------------------------------------------------
    # Original single-scene logic
    # -------------------------------------------------

    images = scene["images"]
    Ks = scene["intrinsics"]
    poses = scene["poses"]
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

    required_inputs = n_input if min_input_views is None else max(n_input, min_input_views)
    if len(candidate_ids) < required_inputs:
        raise ValueError(
            f"Skipping scene {scene['scene']}: need at least {required_inputs} candidate views, got {len(candidate_ids)}"
        )

    input_ids = _select_input_ids(
        candidate_ids=candidate_ids,
        target_id=target_id,
        n_input=n_input,
        sampling=input_view_sampling,
        poses=poses,
    )
    input_ids = _prioritize_reference_view(input_ids, target_id, poses=poses)

    input_imgs = images[input_ids].unsqueeze(0).to(device)
    input_Ks = Ks[input_ids].unsqueeze(0).to(device)
    input_poses = poses[input_ids].unsqueeze(0).to(device)

    target_img = images[target_id].unsqueeze(0).to(device)
    target_K = Ks[target_id].to(device)
    target_pose = poses[target_id].to(device)

    meta = {
        "scene_name": scene["scene"],
        "target_id": target_id,
        "input_ids": input_ids,
        "target_timestamp": timestamps[target_id],
        "image_hw": (H, W),
    }

    return input_imgs, input_Ks, input_poses, target_img, target_K, target_pose, meta

def save_visuals(save_dir, epoch, scene_idx,
                 input_imgs, target_img, rendered, depth):
    """
    Save input views, target view and rendered output.
    """

    os.makedirs(save_dir, exist_ok=True)

    folder = os.path.join(save_dir, f"epoch_{epoch:03d}_scene_{scene_idx:04d}")
    os.makedirs(folder, exist_ok=True)

    # save input views
    inputs = input_imgs[0]  # [V,3,H,W]
    for i in range(inputs.shape[0]):
        vutils.save_image(
            inputs[i],
            os.path.join(folder, f"input_{i}.png")
        )

    # target image
    vutils.save_image(target_img[0],
        os.path.join(folder, "target.png")
    )

    # rendered image
    vutils.save_image(rendered[0].clamp(0,1),
        os.path.join(folder, "rendered.png")
    )

    # depth (normalize for visualization)
    d = depth[0,0]
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    Image.fromarray((d.cpu().numpy()*255).astype("uint8")).save(
        os.path.join(folder, "depth.png")
    )
