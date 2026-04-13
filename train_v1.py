# std library
import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# configs
from configs.re10k_experiment import get_default_config

# utils
from train_v1_epoch import train_epoch

# models
from gs_v2_models.v1_vggt_gs import BasicMLP
from pipeline.data_loader import RealEstate10KDataset

def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    avg_loss,
    best_metric,
    config,
):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "avg_loss": avg_loss,
        "best_metric": best_metric,
        "config": asdict(config),
    }
    torch.save(payload, path)

def configure_environment(args):
    """Dynamically configures PyTorch and CPU/GPU settings based on location."""
    
    if args.location == 'hpc':
        print("🚀 Configuring for HPC: Maximizing CPU & GPU utilization...")
        
        # HPC: Let PyTorch use all available CPU cores for dataloading/ops
        # We don't restrict set_num_threads here.
        
        # Enable TF32 for huge speedups on Ampere (A100) and Hopper (H100) GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable CuDNN benchmark: PyTorch will find the fastest convolution algorithms 
        # for your specific hardware. (Highly recommended for A100/H100).
        torch.backends.cudnn.benchmark = True 
        
        # Prevent OpenCV from eating up all CPU cores, which conflicts with PyTorch Dataloaders
        cv2.setNumThreads(0) 

        # If requesting multiple GPUs on HPC
        if args.num_gpu > 1:
            print(f"🌟 Multi-GPU requested: {args.num_gpu} GPUs.")
            # Note: You will still need to wrap your PyTorch model in 
            # torch.nn.DataParallel(model) or DistributedDataParallel (DDP) later in the script!

    else:
        print("💻 Configuring for Local: Restricting resources to keep PC responsive...")
        
        # Local: Restrict threads so your local machine doesn't freeze
        cv2.setNumThreads(0)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        # Disable heavy optimizations that might not be supported on consumer GPUs
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        
        # Force the script to only see the specific local GPU you requested
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

def main():
    parser = argparse.ArgumentParser(description="3DGS Training Script")

    # Adjusted types and added choices/defaults for safety
    parser.add_argument('--location', type=str, choices=['local', 'hpc'], default='local', help='local/hpc')
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--version', type=str, help='vx_x') # Changed to string since 'vx_x' contains text
    
    # Python keyword fix: Use dest='resume'
    parser.add_argument('--continue', dest='resume', type=int, choices=[0, 1], default=0, help='0/1 to resume training')
    
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to utilize')
    parser.add_argument('--gpu', type=int, default=0, help='Specific GPU ID to use (mainly for local)')
    parser.add_argument('--num_view', type=int, default=None, help='Number of input views / demo input features')
    parser.add_argument('--num_workers', type=int, default=None, help='Override config dataloader num_workers')
    parser.add_argument('--pin_memory', type=int, choices=[0, 1], default=None, help='Override config dataloader pin_memory with 0/1')
    parser.add_argument('--save_every_n_epochs', type=int, default=None, help='Override config checkpoint save frequency')
    
    args = parser.parse_args()

    # 1. Apply the hardware optimizations
    configure_environment(args)

    config = get_default_config()
    config.training.resume = bool(args.resume)
    if args.num_view is not None:
        config.data.n_input_views = args.num_view
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.pin_memory is not None:
        config.data.pin_memory = bool(args.pin_memory)
    if args.save_every_n_epochs is not None:
        config.training.save_every_n_epochs = args.save_every_n_epochs
    device = resolve_device(config.training.device)

    
    run_model_name = args.model_name or "gs_default_model_name"
    run_version = args.version or "v0_0"
    save_dir = os.path.join(config.training.save_dir, f"{run_model_name}_{run_version}")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    #TODO arg/device check
    print(f"Running model: {args.model_name}, Version: {args.version}")
    print(f"Using device: {device}")
    print(f"Using num_view: {config.data.n_input_views}")
    print(f"Using num_workers: {config.data.num_workers}")
    print(f"Using pin_memory: {config.data.pin_memory}")
    print(f"Saving epoch checkpoints every {config.training.save_every_n_epochs} epochs")
    
    # TODO: HERE we initilize the model
    model = BasicMLP(
        input_size=config.data.n_input_views,
        hidden_size=32,
        output_size=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    latest_ckpt_path = os.path.join(save_dir, "model_latest.pth")
    best_ckpt_path = os.path.join(save_dir, "model_best.pth")
    final_ckpt_path = os.path.join(save_dir, "model_final.pth")
    csv_log_path = os.path.join(save_dir, "train_log.csv")

    # TODO Tensor board set up
    tb_log_dir = (
        config.training.tensorboard_log_dir
        if config.training.tensorboard_log_dir is not None
        else os.path.join(save_dir, "tensorboard")
    )
    tb_writer = SummaryWriter(log_dir=tb_log_dir) if config.training.enable_tensorboard else None

    # TODO Dataset load
    dataset_manager = RealEstate10KDataset(config.data.data_root)
    dataset_manager.filter_re10k_scenes(config.data.data_root, config.data.n_input_views)
    dataset = dataset_manager
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    start_epoch = 0
    best_metric = None
    # TODO: model resume
    if args.resume:
        chosen_ckpt = None

        if config.training.resume_mode == "best" and os.path.exists(best_ckpt_path):
            chosen_ckpt = best_ckpt_path
        elif config.training.resume_mode == "latest" and os.path.exists(latest_ckpt_path):
            chosen_ckpt = latest_ckpt_path

        if chosen_ckpt is not None:
            print(f"Resuming training from previous checkpoint: {chosen_ckpt}")
            ckpt = torch.load(chosen_ckpt, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0)
            best_metric = ckpt.get("best_metric", None)
        else:
            print("Resume requested, but no checkpoint was found.")

    #TODO: Log CSV
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "avg_loss",
                "avg_mse",
                "avg_l1",
                "learning_rate",
                "num_steps",
                "best_metric",
                "saved_as_best",
            ])


    #TODO: During loop, we call train_epoch() which will do one epoch of training and return the metrics, loss, etc. We log those to tensorboard and also save checkpoints based on the config settings.
    for ep in range(start_epoch, config.training.epochs):
        stats = train_epoch(
            model=model,
            data_manager=dataset_manager,
            dataloader=loader,
            optimizer=optimizer,
            device=device,
            config=config,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        avg_loss = stats["loss_total"]

        print(
            f"[Epoch {ep + 1}] "
            f"loss={stats['loss_total']:.6f} "
            f"mse={stats['loss_mse']:.6f} "
            f"l1={stats['loss_l1']:.6f} "
            f"lr={current_lr:.6e}"
        )

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss_total", stats["loss_total"], ep + 1)
            tb_writer.add_scalar("train/loss_mse", stats["loss_mse"], ep + 1)
            tb_writer.add_scalar("train/loss_l1", stats["loss_l1"], ep + 1)
            tb_writer.add_scalar("train/learning_rate", current_lr, ep + 1)

        if (
            config.training.save_every_n_epochs > 0
            and (ep + 1) % config.training.save_every_n_epochs == 0
        ):
            epoch_ckpt_path = os.path.join(save_dir, f"model_epoch_{ep + 1}.pth")
            save_checkpoint(
                path=epoch_ckpt_path,
                epoch=ep + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                avg_loss=avg_loss,
                best_metric=best_metric,
                config=config,
            )
            print(f"Saved periodic checkpoint: {epoch_ckpt_path}")

        saved_as_best = False
        current_metric = avg_loss
        is_better = (best_metric is None) or (current_metric < best_metric)

        if is_better:
            best_metric = current_metric
            save_checkpoint(
                path=best_ckpt_path,
                epoch=ep + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                avg_loss=avg_loss,
                best_metric=best_metric,
                config=config,
            )
            saved_as_best = True
            print(f"New best checkpoint saved at epoch {ep + 1} | loss={best_metric:.6f}")

        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep + 1,
                stats["loss_total"],
                stats["loss_mse"],
                stats["loss_l1"],
                current_lr,
                stats["num_steps"],
                "" if best_metric is None else best_metric,
                int(saved_as_best),
            ])

        if scheduler is not None:
            scheduler.step()



    #TODO: Finally we save the model as checkpoint
    save_checkpoint(
        path=final_ckpt_path,
        epoch=config.training.epochs,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        avg_loss=best_metric,
        best_metric=best_metric,
        config=config,
    )

    if tb_writer is not None:
        tb_writer.close()

    print(f"Final checkpoint saved to: {final_ckpt_path}")



if __name__ == "__main__":
    main()
