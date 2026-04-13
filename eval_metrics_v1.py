import math

import torch
import torch.nn.functional as F

try:
    import lpips
except ImportError:  # pragma: no cover - depends on local environment
    lpips = None


_LPIPS_MODELS = {}


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """
    Accept [3,H,W] or [B,3,H,W] and return [B,3,H,W].
    """
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected tensor with 3 or 4 dims, got shape {tuple(x.shape)}")


def _validate_pair(pred: torch.Tensor, target: torch.Tensor):
    pred = _ensure_4d(pred).float()
    target = _ensure_4d(target).float()

    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.shape[1] != 3:
        raise ValueError(f"Expected RGB tensors with 3 channels, got shape {tuple(pred.shape)}")

    return pred, target


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    pred, target: [B,3,H,W] or [3,H,W], assumed in [0,1]
    """
    pred, target = _validate_pair(pred, target)
    mse = F.mse_loss(pred, target, reduction="mean").item()
    return -10.0 * math.log10(max(mse, eps))


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    Lightweight global SSIM approximation for evaluation.
    pred, target: [B,3,H,W] or [3,H,W], assumed in [0,1]
    """
    pred, target = _validate_pair(pred, target)

    mu_pred = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    sigma_pred = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_pred * mu_pred
    sigma_target = F.avg_pool2d(target * target, 3, 1, 1) - mu_target * mu_target
    sigma_cross = F.avg_pool2d(pred * target, 3, 1, 1) - mu_pred * mu_target

    numerator = (2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)
    denominator = (
        (mu_pred * mu_pred + mu_target * mu_target + C1)
        * (sigma_pred + sigma_target + C2)
    )
    ssim_map = numerator / denominator.clamp(min=1e-8)
    return float(ssim_map.mean().item())


def _to_lpips_range(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def _get_lpips_model(net: str, device: str):
    if lpips is None:
        raise ImportError(
            "lpips is not installed. Install the `lpips` package before using compute_lpips."
        )

    key = (net, device)
    if key not in _LPIPS_MODELS:
        model = lpips.LPIPS(net=net)
        model = model.to(device)
        model.eval()
        _LPIPS_MODELS[key] = model
    return _LPIPS_MODELS[key]


@torch.no_grad()
def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
) -> float:
    """
    pred, target: [B,3,H,W] or [3,H,W], assumed in [0,1]
    """
    pred, target = _validate_pair(pred, target)
    device = pred.device.type

    model = _get_lpips_model(net=net, device=device)
    pred_lpips = _to_lpips_range(pred)
    target_lpips = _to_lpips_range(target)
    value = model(pred_lpips, target_lpips)
    return float(value.mean().item())


def compute_image_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_net: str = "alex",
):
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "lpips": compute_lpips(pred, target, net=lpips_net),
    }
