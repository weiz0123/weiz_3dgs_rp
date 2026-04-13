import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

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


    def filter_re10k_scenes(self, data_root, num_view):
        """
        Inspect RealEstate10K scenes using the same scene/frame checks as
        RealEstate10KDataset and keep only scenes with at least num_view + 1
        usable frames.

        We require num_view input frames plus 1 target frame.
        """
        root = Path(data_root)
        scene_paths = sorted((root / "scenes").glob("*"))

        min_required_frames = num_view + 1
        kept_scenes = []
        skipped_scenes = []

        for scene_path in tqdm(scene_paths, desc="filter_re10k_scenes", leave=False):
            frames_path = scene_path / "frames"
            metadata_path = scene_path / "metadata.txt"

            images = sorted(
                frames_path.glob("*.jpg"),
                key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
            )

            if not metadata_path.exists():
                skipped_scenes.append((scene_path, "missing_metadata"))
                continue

            with open(metadata_path, encoding="utf-8") as f:
                lines = f.readlines()[1:]  # skip header

            if len(lines) == 0 or len(images) == 0:
                skipped_scenes.append((scene_path, "empty_metadata_or_images"))
                continue

            valid_frame_count = 0

            for image_path in tqdm(images, desc=f"scan_frames:{scene_path.name}", leave=False):
                stem = image_path.stem
                if not stem.isdigit():
                    continue

                meta_idx = int(stem)
                if meta_idx < 0 or meta_idx >= len(lines):
                    continue

                vals = lines[meta_idx].split()

                try:
                    float(vals[0])  # timestamp
                    float(vals[1])  # fx
                    float(vals[2])  # fy
                    float(vals[3])  # cx
                    float(vals[4])  # cy
                    _ = [float(v) for v in vals[7:19]]

                    img = cv2.imread(str(image_path))
                    if img is None:
                        continue

                    valid_frame_count += 1
                except Exception:
                    continue

            if valid_frame_count >= min_required_frames:
                kept_scenes.append(scene_path)
            else:
                skipped_scenes.append((scene_path, f"too_few_valid_frames:{valid_frame_count}"))

        self.scenes = kept_scenes
        self.filtered_out_scenes = skipped_scenes
        return self


    def build_training_data(self, scene, num_input_views):
        """
        Split a loaded scene into:
          - one randomly selected target frame
          - num_input_views randomly sampled training frames from both
            before and after the target so the target stays temporally
            in the middle of the selected context
        """
        images = scene["images"]
        intrinsics = scene["intrinsics"]
        poses = scene["poses"]
        timestamps = scene["timestamps"]

        num_frames = images.shape[0]

        if num_frames < num_input_views + 1:
            raise ValueError(
                f"Need at least {num_input_views + 1} frames, got {num_frames}"
            )

        before_quota = num_input_views // 2
        after_quota = num_input_views // 2

        if num_input_views % 2 == 1:
            if random.random() < 0.5:
                before_quota += 1
            else:
                after_quota += 1

        valid_target_indices = [
            idx
            for idx in range(num_frames)
            if idx >= before_quota and (num_frames - 1 - idx) >= after_quota
        ]

        if len(valid_target_indices) == 0:
            raise ValueError(
                "Could not choose a target frame with enough frames on both sides "
                f"for num_input_views={num_input_views} and num_frames={num_frames}"
            )

        target_idx = random.choice(valid_target_indices)

        before_candidates = list(range(0, target_idx))
        after_candidates = list(range(target_idx + 1, num_frames))

        selected_before = sorted(random.sample(before_candidates, before_quota))
        selected_after = sorted(random.sample(after_candidates, after_quota))
        train_indices = selected_before + selected_after

        return {
            "scene": scene["scene"],
            "num_frames": num_frames,
            "target_idx": target_idx,
            "num_input_views": num_input_views,
            "num_before_target": before_quota,
            "num_after_target": after_quota,
            "train_indices_before": selected_before,
            "train_indices_after": selected_after,
            "train_indices": train_indices,
            "target_image": images[target_idx],
            "target_intrinsics": intrinsics[target_idx],
            "target_pose": poses[target_idx],
            "target_timestamp": timestamps[target_idx],
            "train_images": images[train_indices],
            "train_intrinsics": intrinsics[train_indices],
            "train_poses": poses[train_indices],
            "train_timestamps": timestamps[train_indices],
        }


    def format_matrix_text(self, name, matrix):
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
        matrix_text = np.array2string(
            matrix,
            precision=4,
            suppress_small=True,
            max_line_width=120,
        )
        return matrix_text


    def visualize_training_data(self, training_data):
        """
        Show target/train images and their matrices with matplotlib.
        """
        frames = [
            {
                "label": f"target idx={training_data['target_idx']}",
                "image": training_data["target_image"],
                "intrinsics": training_data["target_intrinsics"],
                "pose": training_data["target_pose"],
                "timestamp": training_data["target_timestamp"],
            }
        ]

        for local_idx, frame_idx in tqdm(
            enumerate(training_data["train_indices"]),
            total=len(training_data["train_indices"]),
            desc="build_visual_frames",
            leave=False,
        ):
            frames.append(
                {
                    "label": f"train idx={frame_idx}",
                    "image": training_data["train_images"][local_idx],
                    "intrinsics": training_data["train_intrinsics"][local_idx],
                    "pose": training_data["train_poses"][local_idx],
                    "timestamp": training_data["train_timestamps"][local_idx],
                }
            )

        num_rows = len(frames)
        fig, axes = plt.subplots(
            num_rows,
            3,
            figsize=(18, 3.2 * num_rows),
            gridspec_kw={"width_ratios": [1.8, 1.0, 1.4]},
            constrained_layout=True,
        )

        if num_rows == 1:
            axes = np.array([axes])

        for row_idx, frame in tqdm(
            enumerate(frames),
            total=len(frames),
            desc="draw_visual_frames",
            leave=False,
        ):
            image_ax = axes[row_idx, 0]
            intrinsics_ax = axes[row_idx, 1]
            pose_ax = axes[row_idx, 2]

            image = frame["image"]
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().permute(1, 2, 0).numpy()

            image_ax.imshow(image)
            image_ax.set_title(
                f"{frame['label']} | t={float(frame['timestamp']):.0f}",
                fontsize=11,
                loc="left",
                pad=8,
            )
            image_ax.axis("off")

            intrinsics_ax.text(
                0.02,
                0.98,
                self.format_matrix_text("intrinsics", frame["intrinsics"]),
                va="top",
                ha="left",
                family="monospace",
                fontsize=9,
            )
            intrinsics_ax.set_title("intrinsics", fontsize=10, loc="left", pad=6)
            intrinsics_ax.axis("off")

            pose_ax.text(
                0.02,
                0.98,
                self.format_matrix_text("pose", frame["pose"]),
                va="top",
                ha="left",
                family="monospace",
                fontsize=9,
            )
            pose_ax.set_title("pose", fontsize=10, loc="left", pad=6)
            pose_ax.axis("off")

        fig.suptitle(
            f"Scene: {training_data['scene']} | frames={training_data['num_frames']}",
            fontsize=14,
        )
        plt.show()


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

        for image_path in tqdm(images, desc=f"load_scene:{scene_name}", leave=False):
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
