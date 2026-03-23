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


def scene_to_model_inputs(
    scene,
    device="cuda",
    target_mode="middle",
    exclude_target=True,
    n_input=3,
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

            candidate_ids = sorted(candidate_ids, key=lambda i: abs(i - target_id))
            input_ids = candidate_ids[: min(n_input, len(candidate_ids))]

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

    candidate_ids = sorted(candidate_ids, key=lambda i: abs(i - target_id))
    if len(candidate_ids) >= n_input:
        input_ids = candidate_ids[:n_input]
    else:
        # repeat frames if scene is too short
        input_ids = candidate_ids + [candidate_ids[-1]] * (n_input - len(candidate_ids))

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