import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from debug_util import DEBUG
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from eval_metrics_v1 import compute_psnr, compute_ssim, compute_lpips


def _to_pixel_intrinsics(intrinsic, h, w):
    """
    RealEstate10K intrinsics are often normalized:
      fx, cx are in width units
      fy, cy are in height units

    Convert them to pixel-space when they look normalized.
    """
    k = intrinsic.clone()
    if k.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics [...,3,3], got {tuple(k.shape)}")

    max_f = torch.max(k[..., 0, 0].abs().amax(), k[..., 1, 1].abs().amax())
    max_c = torch.max(k[..., 0, 2].abs().amax(), k[..., 1, 2].abs().amax())

    if max_f < 10.0 and max_c <= 2.0:
        k[..., 0, 0] = k[..., 0, 0] * w
        k[..., 1, 1] = k[..., 1, 1] * h
        k[..., 0, 2] = k[..., 0, 2] * w
        k[..., 1, 2] = k[..., 1, 2] * h

    return k


def _to_homogeneous_4x4(extrinsic):
    if extrinsic.shape[-2:] == (4, 4):
        return extrinsic
    if extrinsic.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsic [...,3,4] or [...,4,4], got {tuple(extrinsic.shape)}")

    out = torch.zeros(*extrinsic.shape[:-2], 4, 4, device=extrinsic.device, dtype=extrinsic.dtype)
    out[..., :3, :4] = extrinsic
    out[..., 3, 3] = 1.0
    return out


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
    intrinsic = _to_pixel_intrinsics(intrinsic, h, w)
    extrinsic = _to_homogeneous_4x4(extrinsic)

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


def _non_black_fraction(image, threshold=1e-4):
    return (image.detach() > threshold).any(dim=0).float().mean().item()


def _simple_ssim(x, y, c1=0.01**2, c2=0.03**2):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean()


def _image_gradient_loss(pred, target):
    pred_b = pred.unsqueeze(0)
    target_b = target.unsqueeze(0)

    pred_dx = pred_b[:, :, :, 1:] - pred_b[:, :, :, :-1]
    pred_dy = pred_b[:, :, 1:, :] - pred_b[:, :, :-1, :]
    target_dx = target_b[:, :, :, 1:] - target_b[:, :, :, :-1]
    target_dy = target_b[:, :, 1:, :] - target_b[:, :, :-1, :]

    loss_x = torch.abs(pred_dx - target_dx).mean()
    loss_y = torch.abs(pred_dy - target_dy).mean()
    return loss_x + loss_y


def _gaussian_regularizers(outputs):
    scales = outputs["scales"]
    opacity = outputs["opacity"]

    scale_reg = (scales ** 2).mean()
    opacity_reg = opacity.mean()
    return scale_reg, opacity_reg


def _camera_space_point_stats(points_world, target_extrinsic):
    target_w2c = _to_homogeneous_4x4(target_extrinsic)
    rotation = target_w2c[:3, :3]
    translation = target_w2c[:3, 3]
    points_cam = points_world @ rotation.transpose(0, 1) + translation
    depth = points_cam[:, 2]
    in_front = depth > 1e-4
    return {
        "camera_depth": depth,
        "in_front_fraction": in_front.float().mean(),
        "camera_depth_positive": depth[in_front],
    }


def render_scene(
    outputs,
    depth_all,
    source_extrinsics,
    source_intrinsics,
    target_extrinsic,
    target_intrinsic,
    H,
    W,
    sh_degree,
    config=None,
):
    device = source_extrinsics.device

    d_xyz = outputs["d_xyz"]
    scales_out = outputs["scales"]
    quat_out = outputs["quat"]
    opacity_out = outputs["opacity"]
    sh_out = outputs.get("sh_coeffs")
    color_out = outputs.get("colors")
    means3d_out = outputs.get("means3D")

    if d_xyz.ndim == 5:
        num_views = depth_all.shape[1]
        num_surfaces = d_xyz.shape[1]
        d_xyz = d_xyz.view(num_views, num_surfaces, 3, d_xyz.shape[-2], d_xyz.shape[-1])
        scales_out = scales_out.view(num_views, num_surfaces, 3, scales_out.shape[-2], scales_out.shape[-1])
        quat_out = quat_out.view(num_views, num_surfaces, 4, quat_out.shape[-2], quat_out.shape[-1])
        opacity_out = opacity_out.view(num_views, num_surfaces, 1, opacity_out.shape[-2], opacity_out.shape[-1])
        if sh_out is not None:
            sh_out = sh_out.view(
                num_views,
                num_surfaces,
                3,
                sh_out.shape[3],
                sh_out.shape[-2],
                sh_out.shape[-1],
            )
        if color_out is not None:
            color_out = color_out.view(
                num_views,
                num_surfaces,
                3,
                color_out.shape[-2],
                color_out.shape[-1],
            )
        if means3d_out is not None:
            means3d_out = means3d_out.view(
                num_views,
                num_surfaces,
                3,
                means3d_out.shape[-2],
                means3d_out.shape[-1],
            )
    elif d_xyz.ndim == 6:
        d_xyz = d_xyz[0]
        scales_out = scales_out[0]
        quat_out = quat_out[0]
        opacity_out = opacity_out[0]
        if sh_out is not None:
            sh_out = sh_out[0]
        if color_out is not None:
            color_out = color_out[0]
        if means3d_out is not None:
            means3d_out = means3d_out[0]
    else:
        raise ValueError(f"Unsupported Gaussian output shape: {tuple(d_xyz.shape)}")

    # --- 2. Build Means & Flatten ---
    if means3d_out is not None:
        means3D = means3d_out.permute(0, 1, 3, 4, 2)
    else:
        source_w2c = torch.inverse(_to_homogeneous_4x4(source_extrinsics[0]))
        base_xyz = get_world_points(depth_all[0], source_intrinsics[0], source_w2c)
        offsets = d_xyz.permute(0, 1, 3, 4, 2)
        means3D = base_xyz.unsqueeze(1) + offsets
    
    means3D = means3D.reshape(-1, 3)
    opacity = opacity_out.reshape(-1, 1)
    scales  = scales_out.reshape(-1, 3)
    rotations = quat_out.reshape(-1, 4)
    shs = None
    colors_precomp = None
    if sh_out is not None:
        # SH Coeffs: [V, S, 3, SH, H, W] -> [N, SH, 3]
        shs = sh_out.permute(0, 1, 4, 5, 3, 2).reshape(-1, sh_out.shape[3], 3)
    if color_out is not None:
        # RGB colors: [V, S, 3, H, W] -> [N, 3]
        colors_precomp = color_out.permute(0, 1, 3, 4, 2).reshape(-1, 3)
    num_gaussians_total = int(means3D.shape[0])

    opacity_threshold = 0.0
    topk_gaussians = None
    if config is not None:
        opacity_threshold = float(getattr(config.training, "render_opacity_threshold", 0.0))
        topk_gaussians = getattr(config.training, "render_topk_gaussians", None)

    keep_mask = torch.ones(num_gaussians_total, dtype=torch.bool, device=device)
    kept_after_threshold = num_gaussians_total
    if opacity_threshold > 0.0:
        keep_mask = opacity.squeeze(-1) > opacity_threshold
        kept_after_threshold = int(keep_mask.sum().item())

    if keep_mask.sum().item() == 0:
        keep_mask = torch.ones_like(keep_mask)
        kept_after_threshold = num_gaussians_total

    kept_after_topk = kept_after_threshold
    if topk_gaussians is not None and keep_mask.sum().item() > int(topk_gaussians):
        keep_indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)
        keep_opacity = opacity.squeeze(-1)[keep_indices]
        _, top_local = torch.topk(
            keep_opacity,
            k=int(topk_gaussians),
            largest=True,
            sorted=False,
        )
        top_indices = keep_indices[top_local]
        top_mask = torch.zeros_like(keep_mask)
        top_mask[top_indices] = True
        keep_mask = top_mask
        kept_after_topk = int(keep_mask.sum().item())

    means3D = means3D[keep_mask]
    opacity = opacity[keep_mask]
    scales = scales[keep_mask]
    rotations = rotations[keep_mask]
    if shs is not None:
        shs = shs[keep_mask]
    if colors_precomp is not None:
        colors_precomp = colors_precomp[keep_mask]
    num_gaussians_kept = int(means3D.shape[0])
    camera_stats = _camera_space_point_stats(means3D, target_extrinsic[0])
    depth_positive = camera_stats["camera_depth_positive"]
    positive_depth_min = float(depth_positive.min().item()) if depth_positive.numel() > 0 else None
    positive_depth_max = float(depth_positive.max().item()) if depth_positive.numel() > 0 else None
    positive_depth_mean = float(depth_positive.mean().item()) if depth_positive.numel() > 0 else None

    if DEBUG.is_first_batch():
        DEBUG.log_debuge_csv(
            "render_scene_inputs",
            means3D=means3D,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
            shs=shs,
            colors_precomp=colors_precomp,
            num_gaussians_total=num_gaussians_total,
            kept_after_threshold=kept_after_threshold,
            kept_after_topk=kept_after_topk,
            num_gaussians_kept=num_gaussians_kept,
            in_front_fraction=camera_stats["in_front_fraction"],
            camera_depth=camera_stats["camera_depth"],
            positive_depth_min=positive_depth_min,
            positive_depth_max=positive_depth_max,
            positive_depth_mean=positive_depth_mean,
            target_intrinsic=target_intrinsic,
            target_extrinsic=target_extrinsic,
        )

    # --- 3. Compute View-Specific Parameters ---
    K = _to_pixel_intrinsics(target_intrinsic[0], H, W)
    target_w2c = torch.inverse(_to_homogeneous_4x4(target_extrinsic[0]))
    # Original 3DGS expects row-major matrices for the rasterizer.
    view_matrix = target_w2c.transpose(-1, -2)
    
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
        sh_degree=0 if colors_precomp is not None else sh_degree,
        campos=torch.inverse(view_matrix.transpose(-1, -2))[:3, 3],
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=settings)
    
    # --- 5. Final Render ---
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = torch.zeros_like(means3D, device=device, requires_grad=True),
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )

    non_black_fraction = _non_black_fraction(rendered_image)
    positive_radii_fraction = (radii > 0).float().mean().item()

    if DEBUG.is_first_batch() or non_black_fraction == 0.0:
        DEBUG.log_debuge_csv(
            "render_scene_outputs",
            rendered_image=rendered_image,
            radii=radii,
            num_gaussians_total=num_gaussians_total,
            kept_after_threshold=kept_after_threshold,
            kept_after_topk=kept_after_topk,
            num_gaussians_kept=num_gaussians_kept,
            positive_radii_fraction=positive_radii_fraction,
            non_black_fraction=non_black_fraction,
        )

    if non_black_fraction == 0.0:
        DEBUG.log_debuge_csv(
            "render_scene_black_debug",
            means3D=means3D,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
            shs=shs,
            num_gaussians_total=num_gaussians_total,
            kept_after_threshold=kept_after_threshold,
            kept_after_topk=kept_after_topk,
            num_gaussians_kept=num_gaussians_kept,
            in_front_fraction=camera_stats["in_front_fraction"],
            camera_depth=camera_stats["camera_depth"],
            positive_depth_min=positive_depth_min,
            positive_depth_max=positive_depth_max,
            positive_depth_mean=positive_depth_mean,
            target_intrinsic=target_intrinsic,
            target_extrinsic=target_extrinsic,
        )
    
    return rendered_image



def train_epoch(model, data_manager, dataloader, optimizer, device, config=None, output_dir=None, epoch_idx=None):
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_ssim_loss = 0.0
    total_gradient_loss = 0.0
    total_scale_reg = 0.0
    total_opacity_reg = 0.0
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

        DEBUG.set_context(
            epoch=epoch_idx,
            batch_idx=steps,
            scene=str(scene["scene"]),
            phase="train",
        )
        DEBUG.log_debuge_csv(
            "scene_batch",
            scene_name=str(scene["scene"]),
            num_images=scene["images"].shape[0],
            intrinsics=scene["intrinsics"],
            poses=scene["poses"],
            timestamps=scene["timestamps"],
        )

        training_data = data_manager.build_training_data(
            scene,
            config.data.n_input_views,
        )

        DEBUG.log_debuge_csv(
            "training_data",
            target_idx=training_data["target_idx"],
            train_indices=training_data["train_indices"],
            train_indices_before=training_data["train_indices_before"],
            train_indices_after=training_data["train_indices_after"],
            train_images=training_data["train_images"],
            target_image=training_data["target_image"],
            train_intrinsics=training_data["train_intrinsics"],
            target_intrinsics=training_data["target_intrinsics"],
            train_poses=training_data["train_poses"],
            target_pose=training_data["target_pose"],
        )

        inputs = training_data["train_images"].to(device)
        target_image = training_data["target_image"].to(device)

        optimizer.zero_grad(set_to_none=True)
        model_outputs = model(
            inputs,
            train_intrinsics=training_data["train_intrinsics"].to(device),
            train_poses=training_data["train_poses"].to(device),
        )

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
            sh_degree=gaussian_head["sh_degree"],
            config=config,
        )
        estimated_extrinsics = model_outputs["estimated_extrinsics"]
        estimated_intrinsics = model_outputs["estimated_intrinsics"]

        if DEBUG.is_first_batch():
            DEBUG.log_debuge_csv(
                "model_outputs",
                dino_features=dino_feat,
                fused_map=fused_map,
                fusion_map_coarse=model_outputs.get("fusion_map_coarse"),
                vggt_depth=vggt_depth,
                depth_low=depth_low,
                conf_low=conf_low,
                gaussian_outputs=gaussian_head,
                estimated_extrinsics=estimated_extrinsics,
                estimated_intrinsics=estimated_intrinsics,
            )

       
        # Loss Computation:
        mse_loss = torch.nn.functional.mse_loss(
            estimated_image,
            target_image,
        )
        mae_loss = torch.nn.functional.l1_loss(
            estimated_image,
            target_image,
        )
        ssim_loss = 1.0 - _simple_ssim(
            estimated_image.unsqueeze(0),
            target_image.unsqueeze(0),
        )
        gradient_loss = _image_gradient_loss(estimated_image, target_image)
        scale_reg, opacity_reg = _gaussian_regularizers(gaussian_head)

        lambda_ssim = float(getattr(config.training, "lambda_ssim", 0.2)) if config is not None else 0.2
        lambda_gradient = float(getattr(config.training, "lambda_gradient", 0.1)) if config is not None else 0.1
        lambda_scale_reg = float(getattr(config.training, "lambda_scale_reg", 1e-4)) if config is not None else 1e-4
        lambda_opacity_reg = float(getattr(config.training, "lambda_opacity_reg", 5e-5)) if config is not None else 5e-5

        total_batch_loss = (
            mse_loss
            + mae_loss
            + lambda_ssim * ssim_loss
            + lambda_gradient * gradient_loss
            + lambda_scale_reg * scale_reg
            + lambda_opacity_reg * opacity_reg
        )
        total_batch_loss.backward()
        optimizer.step()

        psnr = compute_psnr(estimated_image, target_image)
        ssim = compute_ssim(estimated_image, target_image)
        lpips = 0.0

        total_loss += total_batch_loss.item()
        total_mse += mse_loss.item()
        total_l1 += mae_loss.item()
        total_ssim_loss += ssim_loss.item()
        total_gradient_loss += gradient_loss.item()
        total_scale_reg += scale_reg.item()
        total_opacity_reg += opacity_reg.item()
        total_psnr += float(psnr)
        total_ssim += float(ssim)
        total_lpips += float(lpips)
        steps += 1

        DEBUG.log_debuge_csv(
            "batch_metrics",
            loss_total=total_batch_loss.item(),
            loss_mse=mse_loss.item(),
            loss_l1=mae_loss.item(),
            loss_ssim=ssim_loss.item(),
            loss_gradient=gradient_loss.item(),
            loss_scale_reg=scale_reg.item(),
            loss_opacity_reg=opacity_reg.item(),
            psnr=float(psnr),
            ssim=float(ssim),
            lpips=float(lpips),
            estimated_image=estimated_image,
            target_image=target_image,
            non_black_fraction=_non_black_fraction(estimated_image),
        )

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
        "train_images": training_data["train_images"].detach().cpu(),
        "train_poses": training_data["train_poses"].detach().cpu(),
        "train_intrinsics": training_data["train_intrinsics"].detach().cpu(),
        "loss_total": total_loss / steps,
        "loss_mse": total_mse / steps,
        "loss_l1": total_l1 / steps,
        "loss_ssim": total_ssim_loss / steps,
        "loss_gradient": total_gradient_loss / steps,
        "loss_scale_reg": total_scale_reg / steps,
        "loss_opacity_reg": total_opacity_reg / steps,
        "psnr": total_psnr / steps,
        "ssim": total_ssim / steps,
        "lpips": total_lpips / steps,
        "num_steps": steps,
    }
