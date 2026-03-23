'''
this will scan through raw RE10K dataset and extract a subset video/scene id into a subsets.txt
'''

from pathlib import Path
import random

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

train_dir = ROOT / "datasets/realestate10k_subset/RealEstate10K/train"
subset_file = ROOT / "datasets/realestate10k_subset/subset_train.txt"

# number of scenes to sample
NUM_SCENES = 2000

# -------------------------------------------------------
# COLLECT ALL SCENES
# -------------------------------------------------------

all_scenes = list(train_dir.glob("*.txt"))

print("Total scenes available:", len(all_scenes))

# -------------------------------------------------------
# RANDOM SAMPLE
# -------------------------------------------------------

subset = random.sample(all_scenes, min(NUM_SCENES, len(all_scenes)))

# -------------------------------------------------------
# WRITE FILE
# -------------------------------------------------------

with open(subset_file, "w") as f:

    for scene in subset:
        f.write(scene.name + "\n")

print("Subset written to:", subset_file)
print("Scenes in subset:", len(subset))
