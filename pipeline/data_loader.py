import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import random


class RealEstate10KDataset(Dataset):

    def __init__(self, root, transform=None):

        self.root = Path(root)
        self.scenes = sorted((self.root / "scenes").glob("*"))
        self.transform = transform

        if len(self.scenes) == 0:
            raise RuntimeError(f"No scenes found in {self.root}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):

        scene_path = self.scenes[idx]
        scene_name = scene_path.name

        frames_path = scene_path / "frames"
        metadata_path = scene_path / "metadata.txt"

        images = sorted(frames_path.glob("*.jpg"))

        if not metadata_path.exists():
            # skip broken scene
            return self.__getitem__(random.randint(0, len(self.scenes)-1))

        with open(metadata_path) as f:
            lines = f.readlines()[1:]  # skip header

        # -------------------------------------------------
        # IMPORTANT FIX:
        # metadata lines and images may not match
        # -------------------------------------------------

        n = min(len(lines), len(images))

        if n == 0:
            # skip invalid scene
            return self.__getitem__(random.randint(0, len(self.scenes)-1))

        imgs = []
        intrinsics = []
        poses = []
        timestamps = []

        for i in range(n):

            line = lines[i]
            vals = line.split()

            try:

                # -------------------
                # timestamp
                # -------------------
                timestamp = float(vals[0])
                timestamps.append(timestamp)

                # -------------------
                # intrinsics
                # -------------------
                fx = float(vals[1])
                fy = float(vals[2])
                cx = float(vals[3])
                cy = float(vals[4])

                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)

                # -------------------
                # pose
                # -------------------
                R = np.array(vals[7:16], dtype=np.float32).reshape(3,3)
                t = np.array(vals[16:19], dtype=np.float32)

                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R
                pose[:3, 3] = t

                # -------------------
                # image
                # -------------------
                img = cv2.imread(str(images[i]))

                if img is None:
                    # corrupted image
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if self.transform:
                    img = self.transform(img)

                imgs.append(img)
                intrinsics.append(K)
                poses.append(pose)

            except Exception:
                # skip problematic frame
                continue

        # -------------------------------------------------
        # Safety check: ensure we have at least 1 frame
        # -------------------------------------------------

        if len(imgs) == 0:
            return self.__getitem__(random.randint(0, len(self.scenes)-1))

        imgs = torch.tensor(np.stack(imgs)).permute(0,3,1,2).float() / 255
        intrinsics = torch.tensor(np.stack(intrinsics))
        poses = torch.tensor(np.stack(poses))
        timestamps = torch.tensor(timestamps[:len(imgs)])

        return {
            "scene": scene_name,
            "timestamps": timestamps,
            "images": imgs,
            "intrinsics": intrinsics,
            "poses": poses
        }