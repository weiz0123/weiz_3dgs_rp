import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import random

'''
Dataset Loader
'''
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

        images = sorted(
            frames_path.glob("*.jpg"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
        )

        if not metadata_path.exists():
            # skip broken scene
            return self.__getitem__(random.randint(0, len(self.scenes)-1))

        with open(metadata_path) as f:
            lines = f.readlines()[1:]  # skip header

        if len(lines) == 0 or len(images) == 0:
            # skip invalid scene
            return self.__getitem__(random.randint(0, len(self.scenes)-1))

        imgs = []
        intrinsics = []
        poses = []
        timestamps = []

        for image_path in images:
            stem = image_path.stem
            if not stem.isdigit():
                continue

            meta_idx = int(stem)
            if meta_idx < 0 or meta_idx >= len(lines):
                continue

            line = lines[meta_idx]
            vals = line.split()

            try:
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
                # RealEstate10K stores a full 3x4 extrinsic matrix row-wise after
                # the intrinsics/distortion fields:
                #   [r00 r01 r02 t0  r10 r11 r12 t1  r20 r21 r22 t2]
                # The rest of this codebase consistently expects camera-to-world
                # transforms, so we parse the matrix correctly as w2c first and
                # invert it once here.
                extrinsic = np.array(vals[7:19], dtype=np.float32).reshape(3, 4)

                w2c = np.eye(4, dtype=np.float32)
                w2c[:3, :4] = extrinsic
                pose = np.linalg.inv(w2c).astype(np.float32)

                # -------------------
                # image
                # -------------------
                img = cv2.imread(str(image_path))

                if img is None:
                    # corrupted image
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if self.transform:
                    img = self.transform(img)

                # -------------------
                # timestamp
                # -------------------
                timestamp = float(vals[0])

                imgs.append(img)
                timestamps.append(timestamp)
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
