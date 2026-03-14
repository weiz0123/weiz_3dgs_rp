from pipeline.data_loader import RealEstate10KDataset

dataset = RealEstate10KDataset("datasets/realestate10k_subset")

scene = dataset[0]

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