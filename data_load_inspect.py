import argparse

from configs.re10k_experiment import get_default_config
from pipeline.data_loader import RealEstate10KDataset
'''
scene["scene"]
Scene name: 0000cc6d8b108390

images shape: torch.Size([218, 3, 360, 640])
intrinsics shape: torch.Size([218, 3, 3])
poses shape: torch.Size([218, 4, 4])

First 5 timestamps:
tensor([52553000., 52586000., 52619000., 52653000., 52686000.])

First pixel value of first image:
tensor([0.4784, 0.3922, 0.3098])

First camera intrinsics:
tensor([[0.5110, 0.0000, 0.5000],
        [0.0000, 0.9084, 0.5000],
        [0.0000, 0.0000, 1.0000]])

Second camera intrinsics:
tensor([[0.5110, 0.0000, 0.5000],
        [0.0000, 0.9084, 0.5000],
        [0.0000, 0.0000, 1.0000]])

First camera pose:
tensor([[ 0.9999, -0.0043,  0.0107,  0.1400],
        [ 0.0044,  1.0000, -0.0029, -0.0237],
        [-0.0106,  0.0030,  0.9999,  0.3350],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])

Second camera pose:
tensor([[ 0.9999, -0.0043,  0.0108,  0.1416],
        [ 0.0043,  1.0000, -0.0034, -0.0248],
        [-0.0108,  0.0034,  0.9999,  0.3474],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])

'''
def main():
    parser = argparse.ArgumentParser(description="Inspect one RealEstate10K scene")
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Exact scene name to visualize. Defaults to the first filtered scene.",
    )
    args = parser.parse_args()

    config = get_default_config()

    dataset = RealEstate10KDataset(config.data.data_root)
    dataset.filter_re10k_scenes(config.data.data_root, config.data.n_input_views)

    scene_idx = 0
    if args.scene_name is not None:
        scene_names = [scene_path.name for scene_path in dataset.scenes]
        if args.scene_name not in scene_names:
            raise ValueError(
                f"Scene '{args.scene_name}' not found in filtered dataset. "
                f"First 10 available scenes: {scene_names[:10]}"
            )
        scene_idx = scene_names.index(args.scene_name)

    scene = dataset[scene_idx]

    print("Scene name:", scene["scene"])

    print("images shape:", scene["images"].shape)
    print("intrinsics shape:", scene["intrinsics"].shape)
    print("poses shape:", scene["poses"].shape)

    # -------------------------------------------------
    # Example values
    # -------------------------------------------------

    print("\nFirst 5 timestamps:")
    print(scene["timestamps"][:5])

    print("\nFirst pixel value of first image:")
    print(scene["images"][0, :, 0, 0])

    print("\nFirst camera intrinsics:")
    print(scene["intrinsics"][0])

    print("\nSecond camera intrinsics:")
    print(scene["intrinsics"][1])

    print("\nFirst camera pose:")
    print(scene["poses"][0])

    print("\nSecond camera pose:")
    print(scene["poses"][1])

    training_data = dataset.build_training_data(
        scene,
        num_input_views=config.data.n_input_views,
    )

    print("\nConfigured data root:")
    print(config.data.data_root)

    print("\nConfigured n_input_views:")
    print(config.data.n_input_views)

    print("\nTraining data target index:")
    print(training_data["target_idx"])

    print("\nTraining images shape:")
    print(training_data["train_images"].shape)

    print("\nSelected training indices:")
    print(training_data["train_indices"])

    print("\nTarget image shape:")
    print(training_data["target_image"].shape)

    dataset.visualize_training_data(training_data)


if __name__ == "__main__":
    main()
