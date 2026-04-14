import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from eval_metrics_v1 import compute_psnr, compute_ssim, compute_lpips


def get_world_points(depth, intrinsic, extrinsic):
    """
    Converts a depth map to world-space 3D points.
    depth: [V, 1, H, W]
    intrinsic: [V, 3, 3]
    extrinsic: [V, 4, 4] (World-to-Camera)
    """
    v, _, h, w = depth.shape
    device = depth.device
    
    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=depth.dtype),
        torch.arange(w, device=device, dtype=depth.dtype),
        indexing='ij',
    )
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(1, -1, 3) # [1, H*W, 3]
    pixels = pixels.expand(v, -1, -1).permute(0, 2, 1) # [V, 3, H*W]

    # Matrix multiply: inv(K) @ pixels * depth
    inv_K = torch.inverse(intrinsic)
    cam_points = inv_K @ pixels # [V, 3, H*W]
    cam_points = cam_points * depth.reshape(v, 1, -1) # Scale by depth

    # Transform to World Space: inv(Extrinsic) @ cam_points
    # Note: Extrinsics are usually World-to-Cam, so we invert to get Cam-to-World
    cam_points_homo = torch.cat([cam_points, torch.ones(v, 1, h*w, device=device)], dim=1)
    inv_E = torch.inverse(extrinsic)
    world_points = inv_E @ cam_points_homo # [V, 4, H*W]
    
    return world_points[:, :3, :].permute(0, 2, 1).reshape(v, h, w, 3)

def get_projection_matrix(znear, zfar, fovX, fovY, device):
    tanHalfFovY = torch.tan(fovY / 2)
    tanHalfFovX = torch.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def render_scene(outputs, depth_all, source_extrinsics, source_intrinsics, target_extrinsic, target_intrinsic, H, W, sh_degree):
    device = source_extrinsics.device
    
    # --- 1. Get Base World Points ---
    # depth_all: [1, 8, 1, H, W], intrin/extrin: [1, 8, ...]
    base_xyz = get_world_points(depth_all[0], source_intrinsics[0], source_extrinsics[0])

    d_xyz = outputs["d_xyz"]
    scales_out = outputs["scales"]
    quat_out = outputs["quat"]
    opacity_out = outputs["opacity"]
    sh_out = outputs["sh_coeffs"]

    if d_xyz.ndim == 5:
        num_views = depth_all.shape[1]
        num_surfaces = d_xyz.shape[1]
        d_xyz = d_xyz.view(num_views, num_surfaces, 3, d_xyz.shape[-2], d_xyz.shape[-1])
        scales_out = scales_out.view(num_views, num_surfaces, 3, scales_out.shape[-2], scales_out.shape[-1])
        quat_out = quat_out.view(num_views, num_surfaces, 4, quat_out.shape[-2], quat_out.shape[-1])
        opacity_out = opacity_out.view(num_views, num_surfaces, 1, opacity_out.shape[-2], opacity_out.shape[-1])
        sh_out = sh_out.view(
            num_views,
            num_surfaces,
            3,
            sh_out.shape[3],
            sh_out.shape[-2],
            sh_out.shape[-1],
        )
    elif d_xyz.ndim == 6:
        d_xyz = d_xyz[0]
        scales_out = scales_out[0]
        quat_out = quat_out[0]
        opacity_out = opacity_out[0]
        sh_out = sh_out[0]
    else:
        raise ValueError(f"Unsupported Gaussian output shape: {tuple(d_xyz.shape)}")
    
    # --- 2. Apply Offsets & Flatten ---
    offsets = d_xyz.permute(0, 1, 3, 4, 2) # [V, S, H, W, 3]
    means3D = base_xyz.unsqueeze(1) + offsets           # [8, 2, H, W, 3]
    
    means3D = means3D.reshape(-1, 3)
    opacity = opacity_out.reshape(-1, 1)
    scales  = scales_out.reshape(-1, 3)
    rotations = quat_out.reshape(-1, 4)
    # SH Coeffs: [V, S, 3, SH, H, W] -> [N, SH, 3]
    shs = sh_out.permute(0, 1, 4, 5, 3, 2).reshape(-1, sh_out.shape[3], 3)

    # --- 3. Compute View-Specific Parameters ---
    K = target_intrinsic[0]
    # IMPORTANT: Original 3DGS expects row-major matrices for the Rasterizer
    view_matrix = target_extrinsic[0].transpose(-1, -2)
    
    fx, fy = K[0, 0], K[1, 1]
    
    # Calculate tanFoV
    tanfovX = W / (2.0 * fx)
    tanfovY = H / (2.0 * fy)
    
    # Convert tanFoV back to FOV angles for our projection helper
    fovX = 2 * torch.atan(tanfovX)
    fovY = 2 * torch.atan(tanfovY)
    
    # Get the 4x4 Projection Matrix
    # znear/zfar can be standard 0.01 / 100.0 for most scenes
    proj_mat_4d = get_projection_matrix(0.01, 100.0, fovX, fovY, device)
    
    # The 'full_proj_matrix' is (ViewMatrix @ ProjectionMatrix)
    # We use .T because the CUDA code expects row-major
    full_proj_matrix = (view_matrix @ proj_mat_4d).T
    
    # --- 4. Setup Rasterizer ---
    settings = GaussianRasterizationSettings(
        image_height=int(H),
        image_width=int(W),
        tanfovx=tanfovX.item(),
        tanfovy=tanfovY.item(),
        bg=torch.tensor([0, 0, 0], device=device, dtype=torch.float32),
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=full_proj_matrix,
        sh_degree=sh_degree,
        campos=torch.inverse(view_matrix.transpose(-1, -2))[:3, 3],
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=settings)
    
    # --- 5. Final Render ---
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = torch.zeros_like(means3D, device=device, requires_grad=True),
        shs = shs.transpose(1, 2), # Now [N, 3, 16]
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )
    
    return rendered_image



def train_epoch(model, data_manager, dataloader, optimizer, device, config=None, output_dir=None):
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="train_epoch", leave=False):
        scene = {
            "scene": batch["scene"][0],
            "images": batch["images"][0],
            "intrinsics": batch["intrinsics"][0],
            "poses": batch["poses"][0],
            "timestamps": batch["timestamps"][0],
        }

        training_data = data_manager.build_training_data(
            scene,
            config.data.n_input_views,
        )

        inputs = training_data["train_images"].to(device)
        target_image = training_data["target_image"].to(device)

        optimizer.zero_grad(set_to_none=True)
        model_outputs = model(inputs)

        gaussian_head = model_outputs["guaussian_outputs"]
        dino_feat  = model_outputs["features"]
        fused_map = model_outputs["fused_map"]
        vggt_depth = model_outputs["depth"]
        depth_low = model_outputs["depth_low"]
        conf_low = model_outputs["conf_low"]


        estimated_image = render_scene(
            gaussian_head,
            model_outputs["depth"],
            training_data["train_poses"].unsqueeze(0).to(device),
            training_data["train_intrinsics"].unsqueeze(0).to(device),
            training_data["target_pose"].unsqueeze(0).to(device),
            training_data["target_intrinsics"].unsqueeze(0).to(device),
            H=inputs.shape[-2],
            W=inputs.shape[-1],
            sh_degree=2,
        )
        estimated_extrinsics = model_outputs["estimated_extrinsics"]
        estimated_intrinsics = model_outputs["estimated_intrinsics"]

       
        # Loss Computation:
        mse_loss = torch.nn.functional.mse_loss(
            estimated_image,
            target_image,
        )
        mae_loss = torch.nn.functional.l1_loss(
            estimated_image,
            target_image,
        )
        total_batch_loss = mse_loss + mae_loss
        total_batch_loss.backward()
        optimizer.step()

        psnr = compute_psnr(estimated_image, target_image)
        ssim = compute_ssim(estimated_image, target_image)
        lpips = compute_lpips(estimated_image, target_image)

        total_loss += total_batch_loss.item()
        total_mse += mse_loss.item()
        total_l1 += mae_loss.item()
        total_psnr += float(psnr)
        total_ssim += float(ssim)
        total_lpips += float(lpips)
        steps += 1

    steps = max(steps, 1)
    return {
        "dino_features": dino_feat.detach().cpu(),
        "fused_map": fused_map.detach().cpu(),
        "vggt_depth": vggt_depth.detach().cpu(),
        "depth_low": depth_low.detach().cpu(),
        "conf_low": conf_low.detach().cpu(),
        "estimated_image": estimated_image.detach().cpu(),
        "estimated_extrinsics": estimated_extrinsics.detach().cpu(),
        "estimated_intrinsics": estimated_intrinsics.detach().cpu(),
        "target_image": training_data["target_image"].detach().cpu(),
        "train_poses": training_data["train_poses"].detach().cpu(),
        "train_intrinsics": training_data["train_intrinsics"].detach().cpu(),
        "loss_total": total_loss / steps,
        "loss_mse": total_mse / steps,
        "loss_l1": total_l1 / steps,
        "psnr": total_psnr / steps,
        "ssim": total_ssim / steps,
        "lpips": total_lpips / steps,
        "num_steps": steps,
    }
