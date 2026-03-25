import torch
import torch.nn as nn
import torch.nn.functional as F


def l1_loss(pred, target):
    return (pred - target).abs().mean()


def simple_ssim(x, y, C1=0.01**2, C2=0.03**2):
    """
    Lightweight SSIM approximation for training.
    x, y: [B,3,H,W]
    """
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean()


def depth_smoothness_loss(depth, image):
    """
    edge-aware depth smoothness
    depth: [B,1,H,W]
    image: [B,3,H,W]
    """
    dx_depth = (depth[:, :, :, 1:] - depth[:, :, :, :-1]).abs()
    dy_depth = (depth[:, :, 1:, :] - depth[:, :, :-1, :]).abs()

    dx_img = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(1, keepdim=True)
    dy_img = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(1, keepdim=True)

    loss_x = dx_depth * torch.exp(-dx_img)
    loss_y = dy_depth * torch.exp(-dy_img)
    return loss_x.mean() + loss_y.mean()


def gaussian_regs(scales, opacities):
    reg_scale = (scales ** 2).mean()
    reg_opacity = opacities.mean()
    return reg_scale, reg_opacity


def total_loss(
    rendered,
    target,
    depth,
    ref_img_small,
    scales,
    opacities,
    lambda_l1=1.0,
    lambda_ssim=0.2,
    lambda_smooth=0.05,
    lambda_scale=1e-3,
    lambda_opacity=0.0,
):
    l1 = l1_loss(rendered, target)
    ssim_term = 1.0 - simple_ssim(rendered, target)
    smooth = depth_smoothness_loss(depth, ref_img_small)
    reg_scale, reg_opacity = gaussian_regs(scales, opacities)

    total = (
        lambda_l1 * l1 +
        lambda_ssim * ssim_term +
        lambda_smooth * smooth +
        lambda_scale * reg_scale +
        lambda_opacity * reg_opacity
    )

    stats = {
        "loss_total": total.item(),
        "loss_l1": l1.item(),
        "loss_ssim": ssim_term.item(),
        "loss_smooth": smooth.item(),
        "loss_scale": reg_scale.item(),
        "loss_opacity": reg_opacity.item(),
    }
    return total, stats
