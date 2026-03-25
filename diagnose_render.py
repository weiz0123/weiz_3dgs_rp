import argparse
import importlib
import inspect
import os
import random

import torch
import torchvision.utils as vutils

from configs.re10k_experiment import get_default_config
from gs_models.render_utils import rasterize_gaussians_single
from pipeline.data_loader import RealEstate10KDataset
from train_re10k_utils import scene_to_model_inputs


def _camera_space_points(points_world: torch.Tensor, pose_c2w: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(
        points_world.shape[0],
        1,
        device=points_world.device,
        dtype=points_world.dtype,
    )
    points_h = torch.cat([points_world, ones], dim=1)
    w2c = torch.inverse(pose_c2w)
    points_cam = (w2c @ points_h.t()).t()[:, :3]
    return points_cam


def _camera_stats(
    points_world: torch.Tensor,
    pose_c2w: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    z_thresh: float = 0.2,
):
    points_cam = _camera_space_points(points_world, pose_c2w)
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    total = max(int(z.numel()), 1)
    positive_mask = z > 0.0
    safe_mask = z > z_thresh

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z_safe = z.clamp(min=1e-6)
    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy

    in_frame = (
        positive_mask
        & (u >= 0.0)
        & (u <= float(W - 1))
        & (v >= 0.0)
        & (v <= float(H - 1))
    )
    safe_in_frame = (
        safe_mask
        & (u >= 0.0)
        & (u <= float(W - 1))
        & (v >= 0.0)
        & (v <= float(H - 1))
    )

    positive_z = z[positive_mask]
    if positive_z.numel() > 0:
        depth_summary = {
            "min": float(positive_z.min().item()),
            "p05": float(torch.quantile(positive_z, 0.05).item()),
            "median": float(torch.quantile(positive_z, 0.50).item()),
            "p95": float(torch.quantile(positive_z, 0.95).item()),
            "max": float(positive_z.max().item()),
        }
    else:
        depth_summary = {
            "min": float("nan"),
            "p05": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    return {
        "total_points": total,
        "frac_z_gt_0": float(positive_mask.float().mean().item()),
        "frac_z_gt_thresh": float(safe_mask.float().mean().item()),
        "frac_in_frame": float(in_frame.float().mean().item()),
        "frac_safe_in_frame": float(safe_in_frame.float().mean().item()),
        "depth_summary": depth_summary,
    }


def _render_stats(rendered: torch.Tensor | None):
    if rendered is None:
        return {
            "is_none": True,
            "mean": float("nan"),
            "max": float("nan"),
            "frac_nonblack": float("nan"),
        }

    return {
        "is_none": False,
        "mean": float(rendered.mean().item()),
        "max": float(rendered.max().item()),
        "frac_nonblack": float((rendered > 1e-3).float().mean().item()),
    }


def _format_camera_report(title: str, stats: dict) -> str:
    depth = stats["depth_summary"]
    lines = [
        title,
        (
            f"  visible z>0: {stats['frac_z_gt_0']:.3f}, "
            f"z>0.2: {stats['frac_z_gt_thresh']:.3f}, "
            f"in-frame: {stats['frac_in_frame']:.3f}, "
            f"safe in-frame: {stats['frac_safe_in_frame']:.3f}"
        ),
        (
            f"  positive-depth stats: min={depth['min']:.4f}, "
            f"p05={depth['p05']:.4f}, median={depth['median']:.4f}, "
            f"p95={depth['p95']:.4f}, max={depth['max']:.4f}"
        ),
    ]
    return "\n".join(lines)


def _format_render_report(title: str, stats: dict) -> str:
    if stats["is_none"]:
        return f"{title}\n  render failed"
    return (
        f"{title}\n"
        f"  mean={stats['mean']:.6f}, max={stats['max']:.6f}, "
        f"frac_nonblack={stats['frac_nonblack']:.6f}"
    )


def _save_render(rendered: torch.Tensor | None, path: str):
    if rendered is None:
        return
    vutils.save_image(rendered.clamp(0, 1), path)


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def load_model_class(model_version: str):
    module = importlib.import_module(f"gs_models.{model_version}")
    return module.MultiViewDinoDepthToGaussians


def build_model_from_config(device: str):
    config = get_default_config()
    model_cls = load_model_class(config.model.model_version)
    model_kwargs = {
        "dino_name": config.model.dino_name,
        "freeze_dino": config.model.freeze_dino,
        "num_depth_bins": config.model.num_depth_bins,
        "depth_min": config.model.depth_min,
        "depth_max": config.model.depth_max,
        "feat_reduce_dim": config.model.feat_reduce_dim,
        "use_full_res_cost_volume": config.model.use_full_res_cost_volume,
        "transformer_depth": config.model.transformer_depth,
        "transformer_heads": config.model.transformer_heads,
        "max_views": config.model.max_views,
        "freeze_vggt": config.model.freeze_vggt,
        "vggt_model_name": config.model.vggt_model_name,
        "vggt_repo_path": config.model.vggt_repo_path,
        "vggt_cache_dir": config.model.vggt_cache_dir,
        "vggt_checkpoint_path": config.model.vggt_checkpoint_path,
        "vggt_weights_url": config.model.vggt_weights_url,
    }
    valid_params = inspect.signature(model_cls.__init__).parameters
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}
    return model_cls(**filtered_kwargs).to(device)


def load_model_for_diagnostics(ckpt_path: str, device: str):
    model = build_model_from_config(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def run_diagnostics(args):
    device = resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    dataset = RealEstate10KDataset(args.data_root)
    model = load_model_for_diagnostics(args.ckpt_path, device)

    if args.scene_idx is None:
        scene_idx = random.randint(0, len(dataset) - 1)
    else:
        scene_idx = args.scene_idx

    scene = dataset[scene_idx]
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
        target_mode=args.target_mode,
        exclude_target=True,
        n_input=args.n_input,
        min_input_views=args.n_input,
        input_view_sampling=args.input_view_sampling,
    )

    out = model(
        imgs=input_imgs,
        Ks=input_Ks,
        c2ws=input_poses,
        ref_idx=0,
        emit_stride=args.emit_stride,
    )

    means3D = out["means3D"][0].reshape(-1, 3).contiguous()
    scales = out["scales"][0].reshape(-1, 3).contiguous()
    rotations = out["rotations"][0].reshape(-1, 4).contiguous()
    opacities = out["opacities"][0].reshape(-1, 1).contiguous()
    colors = out["colors"][0].reshape(-1, 3).contiguous()

    target_pose_as_c2w = target_pose.float()
    target_pose_as_w2c = torch.inverse(target_pose.float())
    H = int(target_img.shape[-2])
    W = int(target_img.shape[-1])

    stats_c2w = _camera_stats(means3D, target_pose_as_c2w, target_K, H, W)
    stats_w2c = _camera_stats(means3D, target_pose_as_w2c, target_K, H, W)

    rendered_c2w = rasterize_gaussians_single(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        pose_c2w=target_pose_as_c2w,
        K=target_K,
        H=H,
        W=W,
    )
    rendered_w2c = rasterize_gaussians_single(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        pose_c2w=target_pose_as_w2c,
        K=target_K,
        H=H,
        W=W,
    )

    render_stats_c2w = _render_stats(rendered_c2w)
    render_stats_w2c = _render_stats(rendered_w2c)

    basename = f"scene_{scene_idx:04d}_target_{meta['target_id']}"
    input_path = os.path.join(args.save_dir, f"{basename}_inputs.png")
    target_path = os.path.join(args.save_dir, f"{basename}_target.png")
    render_c2w_path = os.path.join(args.save_dir, f"{basename}_render_pose_as_c2w.png")
    render_w2c_path = os.path.join(args.save_dir, f"{basename}_render_pose_as_w2c.png")
    report_path = os.path.join(args.save_dir, f"{basename}_report.txt")

    vutils.save_image(input_imgs[0], input_path, nrow=min(args.n_input, 5))
    vutils.save_image(target_img[0], target_path)
    _save_render(rendered_c2w, render_c2w_path)
    _save_render(rendered_w2c, render_w2c_path)

    report_lines = [
        f"scene_idx: {scene_idx}",
        f"scene_name: {meta['scene_name']}",
        f"target_id: {meta['target_id']}",
        f"input_ids: {meta['input_ids']}",
        f"image_hw: {(H, W)}",
        f"emit_stride: {args.emit_stride}",
        "",
        _format_camera_report("Assumption A: metadata pose is c2w", stats_c2w),
        "",
        _format_render_report("Render A: metadata pose is c2w", render_stats_c2w),
        "",
        _format_camera_report("Assumption B: metadata pose is w2c, so invert once", stats_w2c),
        "",
        _format_render_report("Render B: metadata pose is w2c, so invert once", render_stats_w2c),
        "",
        "Heuristic:",
        (
            "  If Assumption B has much higher safe in-frame coverage and a much less black render, "
            "the dataset pose is likely being interpreted with the wrong convention."
        ),
        (
            "  If both assumptions have poor z>0.2 or safe in-frame coverage, the problem is more likely "
            "depth scale/range or badly aligned image-pose pairs."
        ),
    ]
    report_text = "\n".join(report_lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print("")
    print(f"Saved inputs to: {input_path}")
    print(f"Saved target to: {target_path}")
    print(f"Saved pose-as-c2w render to: {render_c2w_path}")
    print(f"Saved pose-as-w2c render to: {render_w2c_path}")
    print(f"Saved report to: {report_path}")


def build_argparser():
    config = get_default_config()
    parser = argparse.ArgumentParser(description="Diagnose 3DGS black renders and pose convention issues.")
    parser.add_argument(
        "--ckpt-path",
        default=r"C:\Users\zhouw\Desktop\3DGS\weiz_3dgs_rp\outputs\model_epoch_226.pth",
    )
    parser.add_argument("--data-root", default="datasets/realestate10k_subset")
    parser.add_argument("--save-dir", default="outputs/pose_diagnostics")
    parser.add_argument("--scene-idx", type=int, default=None)
    parser.add_argument("--n-input", type=int, default=config.data.n_input_views)
    parser.add_argument("--emit-stride", type=int, default=config.training.emit_stride)
    parser.add_argument(
        "--target-mode",
        default=config.data.target_mode,
        choices=["random", "middle"],
    )
    parser.add_argument(
        "--input-view-sampling",
        default=config.data.input_view_sampling,
        choices=["nearest", "sparse", "pose_sparse"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    return parser


if __name__ == "__main__":
    run_diagnostics(build_argparser().parse_args())
