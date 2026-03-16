import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils
import traceback
from PIL import Image
from pipeline.data_loader import RealEstate10KDataset
from gs_models.multiview_dino_depth_gs import (
    MultiViewDinoDepthToGaussians,
)
from gs_models.losses import total_loss
from gs_models.render_utils import rasterize_gaussians_single
from train_re10k_utils import scene_to_model_inputs
from train_re10k_utils import save_visuals

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
def scene_collate(batch):
    return batch

def train_one_step(model, optimizer, scene_batch, device="cuda", n_input=3, emit_stride=1):
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
        ) = scene_to_model_inputs(scene, device=device, n_input=n_input)

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

def train_re10k(
    data_root="datasets/realestate10k_subset",
    epochs=5,
    device="cuda",
    lr=1e-4,
    n_input=3,
    emit_stride=1,
    max_scenes_per_epoch=None,
    save_dir="outputs/re10k_debug",
    run_id=0,
    resume=True,
):
    save_dir = os.path.join(save_dir, f"run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    dataset = RealEstate10KDataset(data_root)
    print("loading datasets...\n")
    # scene {frames:ins:pos}
  
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=scene_collate,
    )

    print("loading model...\n")
    model = MultiViewDinoDepthToGaussians(
        dino_name="facebook/dinov2-base",
        freeze_dino=True,
        num_depth_bins=96,
        depth_min=0.5,
        depth_max=20.0,
        feat_reduce_dim=128,
    ).to(device)

    print("init training...\n")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], # keep track of parameter that needs to be tuned
        lr=lr,
        weight_decay=1e-4,
    )

    start_epoch = 0

    if resume:
        ckpts = [f for f in os.listdir(save_dir) if f.startswith("model_epoch_") and f.endswith(".pth")]

        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
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

        for i, scene_batch in enumerate(tqdm(loader, desc=f"Epoch {ep+1}")):
            if max_scenes_per_epoch is not None and i >= max_scenes_per_epoch:
                break

            try:
                stats, meta, rendered, target_out, depth, input_imgs = train_one_step(
                    model=model,
                    optimizer=optimizer,
                    scene_batch=scene_batch,
                    device=device,
                    n_input=n_input, # number of input view
                    emit_stride=emit_stride,  # one guassian / emit_stride pixels
                )



                total_loss_val += stats["loss_total"]
                steps += 1
                
                # save model
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
                # save visual result
                if ep % 2 == 0 and i % 20 == 0:
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
        epochs=400,
        device=device,
        lr=1e-4,
        n_input=4,
        emit_stride=1,
        max_scenes_per_epoch=200,
        save_dir="outputs/re10k_debug",
        run_id=0,
        resume=True,
    )