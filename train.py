import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pipeline.data_loader import RealEstate10KDataset

from gs_models.multiview_dino_depth_gs import (
    MultiViewDinoDepthToGaussians,
    scale_intrinsics_batch,
)
from gs_models.losses import total_loss
from gs_models.render_utils import rasterize_gaussians_single
from train_re10k_utils import scene_to_model_inputs


def identity_collate(batch):
    """
    Since each dataset item is already one full scene dict,
    and scenes can have variable number of frames,
    keep batch_size=1 and unwrap directly.
    """
    return batch[0]


def train_one_step(model, optimizer, scene, device="cuda", n_input=3, emit_stride=2):
    model.train()
    optimizer.zero_grad()

    (
        input_imgs,
        input_Ks,
        input_poses,
        target_img,
        target_K,
        target_pose,
        meta,
    ) = scene_to_model_inputs(scene)

    # forward
    out = model(
        imgs=input_imgs,      # [1,V,3,H,W]
        Ks=input_Ks,          # [1,V,3,3]
        c2ws=input_poses,     # [1,V,4,4]
        ref_idx=0,
        emit_stride=emit_stride,
    )

    means3D = out["means3D"][0]       # [M,3]
    scales = out["scales"][0]         # [M,3]
    rotations = out["rotations"][0]   # [M,4]
    opacities = out["opacities"][0]   # [M,1]
    colors = out["colors"][0]         # [M,3]
    depth = out["depth"]              # [1,1,Hf,Wf]

    # render at feature resolution first
    _, _, H_full, W_full = target_img.shape
    Hf, Wf = depth.shape[-2:]

    target_small = F.interpolate(
        target_img, size=(Hf, Wf),
        mode="bilinear", align_corners=False
    )

    target_K_small = scale_intrinsics_batch(
        target_K.unsqueeze(0),
        src_hw=(H_full, W_full),
        dst_hw=(Hf, Wf)
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
    ).unsqueeze(0)  # [1,3,Hf,Wf]

    ref_small = F.interpolate(
        input_imgs[:, 0], size=(Hf, Wf),
        mode="bilinear", align_corners=False
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

    return stats, rendered.detach(), target_small.detach(), out, meta


def train_re10k(
    data_root="datasets/realestate10k_subset",
    epochs=5,
    device="cuda",
    lr=1e-4,
    n_input=3,
    emit_stride=2,
    max_scenes_per_epoch=None,
    save_dir="outputs/re10k_debug",
):
    os.makedirs(save_dir, exist_ok=True)

    dataset = RealEstate10KDataset(data_root)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,     # start simple
        collate_fn=identity_collate,
    )

    model = MultiViewDinoDepthToGaussians(
        dino_name="facebook/dinov2-base",
        freeze_dino=True,
        num_depth_bins=32,
        depth_min=0.5,
        depth_max=20.0,
        feat_reduce_dim=128,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    for ep in range(epochs):
        total_loss_val = 0.0
        steps = 0

        for i, scene in enumerate(loader):
            if max_scenes_per_epoch is not None and i >= max_scenes_per_epoch:
                break

            try:
                stats, rendered, target, out, meta = train_one_step(
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
                continue

        avg = total_loss_val / max(steps, 1)
        print(f"\nEpoch {ep+1} avg loss: {avg:.6f}\n")

        ckpt_path = os.path.join(save_dir, f"model_epoch_{ep+1}.pth")
        torch.save({
            "epoch": ep + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "avg_loss": avg,
        }, ckpt_path)

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_re10k(
        data_root="datasets/realestate10k_subset",
        epochs=10,
        device=device,
        lr=1e-4,
        n_input=3,
        emit_stride=2,
        max_scenes_per_epoch=100,   # good for debugging
        save_dir="outputs/re10k_debug",
    )