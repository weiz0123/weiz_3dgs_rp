import csv
import importlib
import inspect
import json
import os
from dataclasses import asdict
from tqdm import tqdm
import traceback

from PIL import Image
from pipeline.data_loader import RealEstate10KDataset

from configs.re10k_experiment import ExperimentConfig, get_default_config
from gs_models.losses import total_loss
from gs_models.render_utils import rasterize_gaussians_single

from train_re10k_utils import scene_to_model_inputs
from train_re10k_utils import save_visuals
from eval_re10k_utils import compute_psnr

import cv2
cv2.setNumThreads(0)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def scene_collate(batch):
    return batch


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def load_model_class(model_version: str):
    module = importlib.import_module(f"gs_models.{model_version}")
    return module.MultiViewDinoDepthToGaussians


def build_model(config: ExperimentConfig, device: str):
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
    }

    valid_params = inspect.signature(model_cls.__init__).parameters
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}
    return model_cls(**filtered_kwargs).to(device)


def train_one_step(
    model,
    optimizer,
    scene_batch,
    device="cuda",
    n_input=3,
    min_input_views=None,
    input_view_sampling="nearest",
    emit_stride=1,
    target_mode="middle",
    exclude_target=True,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_inputs = []
    batch_Ks = []
    batch_poses = []
    batch_targets = []
    batch_target_K = []
    batch_target_pose = []
    metas = []

    for scene in scene_batch:

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
            target_mode=target_mode,
            exclude_target=exclude_target,
            n_input=n_input,
            min_input_views=min_input_views,
            input_view_sampling=input_view_sampling,
        )

        batch_inputs.append(input_imgs)
        batch_Ks.append(input_Ks)
        batch_poses.append(input_poses)
        batch_targets.append(target_img)
        batch_target_K.append(target_K)
        batch_target_pose.append(target_pose)
        metas.append(meta)

    input_imgs = torch.cat(batch_inputs, dim=0)
    input_Ks = torch.cat(batch_Ks, dim=0)
    input_poses = torch.cat(batch_poses, dim=0)
    target_img = torch.cat(batch_targets, dim=0)
    target_K = torch.stack(batch_target_K)
    target_pose = torch.stack(batch_target_pose)

    out = model(
        imgs=input_imgs,
        Ks=input_Ks,
        c2ws=input_poses,
        ref_idx=0,
        emit_stride=emit_stride,
    )

    B = input_imgs.shape[0]

    rendered_list = []

    for b in range(B):

        means3D = out["means3D"][b].reshape(-1, 3).contiguous()
        scales = out["scales"][b].reshape(-1, 3).contiguous()
        rotations = out["rotations"][b].reshape(-1, 4).contiguous()
        opacities = out["opacities"][b].reshape(-1, 1).contiguous()
        colors = out["colors"][b].reshape(-1, 3).contiguous()

        _, _, H_full, W_full = target_img[b:b+1].shape
        depth = out["depth"][b:b+1]
        Hf, Wf = depth.shape[-2:]

        rendered = rasterize_gaussians_single(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            colors=colors,
            pose_c2w=target_pose[b],
            K=target_K[b],
            H=H_full,
            W=W_full,
        )

        if rendered is None:
            raise RuntimeError("Rasterization failed")
        else:
            rendered = rendered.unsqueeze(0)

        rendered_list.append(rendered)

    rendered = torch.cat(rendered_list, dim=0)

    depth = out["depth"]

    _, _, Hf, Wf = depth.shape

    ref_small = F.interpolate(
        input_imgs[:, 0],
        size=(Hf, Wf),
        mode="bilinear",
        align_corners=False,
    )

    loss, stats = total_loss(
        rendered=rendered,
        target=target_img,
        depth=depth,
        ref_img_small=ref_small,
        scales=out["scales"].reshape(-1,3),
        opacities=out["opacities"].reshape(-1,1),
    )

    loss.backward()
    optimizer.step()

    return stats, metas[0], rendered.detach(), target_img.detach(), depth.detach(), input_imgs.detach()

def train_re10k(config: ExperimentConfig):
    if config.data.num_target_views != 1:
        raise ValueError("Current training pipeline supports exactly 1 target view")

    device = resolve_device(config.training.device)
    save_dir = os.path.join(config.training.save_dir, f"run_{config.training.run_id}")
    os.makedirs(save_dir, exist_ok=True)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    dataset = RealEstate10KDataset(config.data.data_root)
    print("loading datasets...\n")
    # scene {frames:ins:pos}
  
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=scene_collate,
    )

    print("loading model...\n")
    model = build_model(config, device)

    print("init training...\n")

##############################
# Optimizer
##############################
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], # keep track of parameter that needs to be tuned
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    start_epoch = 0
    best_metric = None

    latest_ckpt_path = os.path.join(save_dir, "model_latest.pth")
    best_ckpt_path = os.path.join(save_dir, "model_best.pth")
    csv_log_path = os.path.join(save_dir, "train_log.csv")

##############################
# Pre Load
##############################
    if config.training.resume:
        chosen_ckpt = None

        if config.training.resume_mode == "best" and os.path.exists(best_ckpt_path):
            chosen_ckpt = best_ckpt_path
        elif config.training.resume_mode == "latest" and os.path.exists(latest_ckpt_path):
            chosen_ckpt = latest_ckpt_path
        else:
            ckpts = [f for f in os.listdir(save_dir) if f.startswith("model_epoch_") and f.endswith(".pth")]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                chosen_ckpt = os.path.join(save_dir, ckpts[-1])

        if chosen_ckpt is not None:
            print("Resuming from:", chosen_ckpt)

            ckpt = torch.load(chosen_ckpt, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"]
            best_metric = ckpt.get("best_metric", None)

    if not os.path.exists(csv_log_path):
        with open(csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "avg_loss",
                "avg_l1",
                "avg_ssim",
                "avg_smooth",
                "avg_psnr",
                "num_steps",
                "best_metric",
                "saved_as_best"
            ])

    print("start training...\n")
##############################
# Training Loop
##############################
    for ep in range(start_epoch, config.training.epochs):
        total_loss_val = 0.0
        total_l1_val = 0.0
        total_ssim_val = 0.0
        total_smooth_val = 0.0
        total_psnr_val = 0.0
        psnr_count = 0
        steps = 0

        do_metric = ((ep + 1) % config.training.metric_every == 0)

        for i, scene_batch in enumerate(tqdm(loader, desc=f"Epoch {ep+1}")):
            if (
                config.training.max_scenes_per_epoch is not None
                and i >= config.training.max_scenes_per_epoch
            ):
                break
            ##############################
            # 
            ##############################
            try:
                stats, meta, rendered, target_out, depth, input_imgs = train_one_step(
                    model=model,
                    optimizer=optimizer,
                    scene_batch=scene_batch,
                    device=device,
                    n_input=config.data.n_input_views,
                    min_input_views=config.data.min_input_views,
                    input_view_sampling=config.data.input_view_sampling,
                    emit_stride=config.training.emit_stride,
                    target_mode=config.data.target_mode,
                    exclude_target=config.data.exclude_target,
                )

            ##############################
            #  save model
            ##############################

                total_loss_val += float(stats["loss_total"])
                total_l1_val += float(stats["loss_l1"])
                total_ssim_val += float(stats["loss_ssim"])
                total_smooth_val += float(stats["loss_smooth"])
                steps += 1

                if do_metric:
                    with torch.no_grad():
                        psnr_val = compute_psnr(
                            rendered.detach().clamp(0, 1),
                            target_out.detach().clamp(0, 1)
                        )
                    total_psnr_val += psnr_val
                    psnr_count += 1
                else:
                    psnr_val = None
                
                if i % config.training.log_every_n_steps == 0:
                    msg = (
                        f"[Epoch {ep+1} | Scene {i}] "
                        f"{meta['scene_name']} | "
                        f"inputs={meta['input_ids']} target={meta['target_id']} | "
                        f"loss={stats['loss_total']:.4f} "
                        f"l1={stats['loss_l1']:.4f} "
                        f"ssim={stats['loss_ssim']:.4f} "
                        f"smooth={stats['loss_smooth']:.4f}"
                    )
                    if psnr_val is not None:
                        msg += f" psnr={psnr_val:.4f}"
                    print(msg)

            ##############################
            #  save inference
            ##############################
                # save visual result
                if (
                    ep % config.training.visualize_every_n_epochs == 0
                    and i % config.training.visualize_every_n_steps == 0
                ):
                    save_visuals(
                        save_dir=os.path.join(save_dir, "visuals"),
                        epoch=ep,
                        scene_idx=i,
                        input_imgs=input_imgs.cpu(),
                        target_img=target_out.cpu(),
                        rendered=rendered.cpu(),
                        depth=depth.cpu(),
                    )
            except Exception as e:
                print(f"Skipping scene {i} : {e}")
                torch.cuda.empty_cache()
                traceback.print_exc()
                continue
         
        avg = total_loss_val / max(steps, 1)
        avg_l1 = total_l1_val / max(steps, 1)
        avg_ssim = total_ssim_val / max(steps, 1)
        avg_smooth = total_smooth_val / max(steps, 1)
        avg_psnr = total_psnr_val / max(psnr_count, 1) if do_metric and psnr_count > 0 else None

        print(f"\nEpoch {ep+1} avg loss: {avg:.6f}\n")
        if avg_psnr is not None:
            print(f"Epoch {ep+1} avg psnr: {avg_psnr:.4f}\n")

        ##############################
        #  save Model
        ##############################
        latest_payload = {
            "epoch": ep + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "avg_loss": avg,
            "avg_l1": avg_l1,
            "avg_ssim": avg_ssim,
            "avg_smooth": avg_smooth,
            "avg_psnr": avg_psnr,
            "best_metric": best_metric,
        }

        torch.save(latest_payload, latest_ckpt_path)

        saved_as_best = False

        if config.training.save_best_by == "loss":
            current_metric = avg
            is_better = (best_metric is None) or (current_metric < best_metric)
        elif config.training.save_best_by == "psnr":
            current_metric = avg_psnr
            is_better = (current_metric is not None) and ((best_metric is None) or (current_metric > best_metric))
        else:
            raise ValueError("save_best_by must be either 'loss' or 'psnr'")

        if is_better:
            best_metric = current_metric
            best_payload = {
                "epoch": ep + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "avg_loss": avg,
                "avg_l1": avg_l1,
                "avg_ssim": avg_ssim,
                "avg_smooth": avg_smooth,
                "avg_psnr": avg_psnr,
                "best_metric": best_metric,
            }
            torch.save(best_payload, best_ckpt_path)
            saved_as_best = True
            print(
                f"New best model saved at epoch {ep+1} | "
                f"{config.training.save_best_by}={best_metric}"
            )

        with open(csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep + 1,
                avg,
                avg_l1,
                avg_ssim,
                avg_smooth,
                avg_psnr if avg_psnr is not None else "",
                steps,
                best_metric if best_metric is not None else "",
                int(saved_as_best)
            ])

    return model

if __name__ == "__main__":
    print("start...\n")
    config = get_default_config()
    print(f"training model version: {config.model.model_version}")
    train_re10k(config)
