import random
import torch

def intrinsics_to_pixel(K: torch.Tensor, H: int, W: int):
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

    # choose only a few nearest frames
    candidate_ids = sorted(candidate_ids, key=lambda i: abs(i - target_id))
    input_ids = candidate_ids[: min(n_input, len(candidate_ids))]

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