import torch.nn.functional as F
from tqdm import tqdm


def train_epoch(model, data_manager, dataloader, optimizer, device, config=None):
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    steps = 0

    for batch in dataloader:
        scene = {
            "scene": batch["scene"][0],
            "images": batch["images"][0],
            "intrinsics": batch["intrinsics"][0],
            "poses": batch["poses"][0],
            "timestamps": batch["timestamps"][0],
        }

        training_data = data_manager.build_training_data(scene, config.data.n_input_views)


        optimizer.zero_grad(set_to_none=True)
        inputs = training_data["train_images"].to(device)
        target = training_data["target_image"].to(device)
        intrinsics = training_data["target_intrinsics"].to(device)
        pose=training_data["target_pose"].to(device)

        preds = model(inputs)
        mse = F.mse_loss(preds, target)
        l1 = F.l1_loss(preds, target)
        loss = mse

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_mse += float(mse.item())
        total_l1 += float(l1.item())
        steps += 1

    steps = max(steps, 1)
    return {
        "loss_total": total_loss / steps,
        "loss_mse": total_mse / steps,
        "loss_l1": total_l1 / steps,
        "num_steps": steps,
    }
