import random
import torch

def intrinsics_to_pixel(K: torch.Tensor, H: int, W: int):
    Kp = K.clone()
    Kp[:, 0, 0] *= W
    Kp[:, 1, 1] *= H
    Kp[:, 0, 2] *= W
    Kp[:, 1, 2] *= H
    return Kp


def scene_to_model_inputs(scene, device="cuda", target_mode="middle", exclude_target=True):
    """
    Directly converts one scene dict from RealEstate10KDataset
    into model-ready tensors.
    """
    images = scene["images"]         # [T,3,H,W]
    Ks = scene["intrinsics"]         # [T,3,3] normalized
    poses = scene["poses"]           # [T,4,4]
    timestamps = scene["timestamps"] # [T]

    T, _, H, W = images.shape
    Ks = intrinsics_to_pixel(Ks, H, W)

    # choose target frame
    if target_mode == "middle":
        target_id = T // 2
    elif target_mode == "random":
        target_id = random.randint(0, T - 1)
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    if exclude_target:
        input_ids = [i for i in range(T) if i != target_id]
    else:
        input_ids = list(range(T))

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