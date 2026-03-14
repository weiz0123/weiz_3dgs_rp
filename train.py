import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline.data_loader import RealEstate10KDataset
from gs_models.multiview_dino_depth_gs import (
    MultiViewDinoDepthToGaussians,
    scale_intrinsics_batch,
)
from gs_models.losses import total_loss
from gs_models.render_utils import rasterize_gaussians_single
from train_re10k_utils import scene_to_model_inputs


def identity_collate(batch):
    return batch[0]


def train_one_step(model, optimizer, scene, device="cuda", n_input=3, emit_stride=8):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    (
        input_imgs,
        input_Ks,
        input_poses,
        target_img,
        target_K,
        target_pose,
        meta,
    ) = scene_to_model_inputs(scene, device=device, n_input=n_input)

    out = model(
        imgs=input_imgs,
        Ks=input_Ks,
        c2ws=input_poses,
        ref_idx=0,
        emit_stride=emit_stride,
    )

    means3D = out["means3D"][0]
    scales = out["scales"][0]
    rotations = out["rotations"][0]
    opacities = out["opacities"][0]
    colors = out["colors"][0]
    depth = out["depth"]

    _, _, H_full, W_full = target_img.shape
    Hf, Wf = depth.shape[-2:]

    target_small = F.interpolate(
        target_img,
        size=(Hf, Wf),
        mode="bilinear",
        align_corners=False,
    )

    target_K_small = scale_intrinsics_batch(
        target_K.unsqueeze(0),
        src_hw=(H_full, W_full),
        dst_hw=(Hf, Wf),
    )[0]

    rendered = rasterize_gaussians_single(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        pose_c2w=target_pose,
        K=target_K_small,
        H=Hf,
        W=Wf,
    ).unsqueeze(0)

    ref_small = F.interpolate(
        input_imgs[:, 0],
        size=(Hf, Wf),
        mode="bilinear",
        align_corners=False,
    )

    loss, stats = total_loss(
        rendered=rendered,
        target=target_small,
        depth=depth,
        ref_img_small=ref_small,
        scales=scales,
        opacities=opacities,
    )

    loss.backward()
    optimizer.step()

    del out, means3D, scales, rotations, opacities, colors, depth
    del rendered, target_small, ref_small
    torch.cuda.empty_cache()

    return stats, meta


def train_re10k(
    data_root="datasets/realestate10k_subset",
    epochs=5,
    device="cuda",
    lr=1e-4,
    n_input=3,
    emit_stride=8,
    max_scenes_per_epoch=None,
    save_dir="outputs/re10k_debug",
    run_id=0,
    resume=False,
):
    save_dir = os.path.join(save_dir, f"run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    dataset = RealEstate10KDataset(data_root)
    print("loading datasets...\n")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=identity_collate,
    )

    print("loading model...\n")
    model = MultiViewDinoDepthToGaussians(
        dino_name="facebook/dinov2-base",
        freeze_dino=True,
        num_depth_bins=8,
        depth_min=0.5,
        depth_max=20.0,
        feat_reduce_dim=64,
    ).to(device)

    print("init training...\n")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    start_epoch = 0

    if resume:
        ckpts = [f for f in os.listdir(save_dir) if f.endswith(".pth")]
        if ckpts:
            ckpts.sort()
            latest = os.path.join(save_dir, ckpts[-1])
            print("Resuming from:", latest)

            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"]

    print("start training...\n")

    for ep in range(start_epoch, epochs):
        total_loss_val = 0.0
        steps = 0

        for i, scene in enumerate(tqdm(loader, desc=f"Epoch {ep+1}")):
            if max_scenes_per_epoch is not None and i >= max_scenes_per_epoch:
                break

            try:
                stats, meta = train_one_step(
                    model=model,
                    optimizer=optimizer,
                    scene=scene,
                    device=device,
                    n_input=n_input,
                    emit_stride=emit_stride,
                )

                total_loss_val += stats["loss_total"]
                steps += 1

                if i % 10 == 0:
                    print(
                        f"[Epoch {ep+1} | Scene {i}] "
                        f"{meta['scene_name']} | "
                        f"inputs={meta['input_ids']} target={meta['target_id']} | "
                        f"loss={stats['loss_total']:.4f} "
                        f"l1={stats['loss_l1']:.4f} "
                        f"ssim={stats['loss_ssim']:.4f} "
                        f"smooth={stats['loss_smooth']:.4f}"
                    )

            except Exception as e:
                print(f"Skipping scene {i} ({scene['scene']}): {e}")
                torch.cuda.empty_cache()
                continue

        avg = total_loss_val / max(steps, 1)
        print(f"\nEpoch {ep+1} avg loss: {avg:.6f}\n")

        ckpt_path = os.path.join(save_dir, f"model_epoch_{ep+1}.pth")
        torch.save(
            {
                "epoch": ep + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "avg_loss": avg,
            },
            ckpt_path,
        )

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("start...\n")

    train_re10k(
        data_root="datasets/realestate10k_subset",
        epochs=10,
        device=device,
        lr=1e-4,
        n_input=3,
        emit_stride=8,
        max_scenes_per_epoch=100,
        save_dir="outputs/re10k_debug",
        run_id=0,
        resume=False,
    )