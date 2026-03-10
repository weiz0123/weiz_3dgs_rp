import subprocess
from pathlib import Path
import cv2
import shutil

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

subset_file = ROOT / "datasets/realestate10k_subset/subset_train.txt"
train_dir = ROOT / "datasets/realestate10k_subset/RealEstate10K/train"

video_dir = ROOT / "datasets/realestate10k_subset/videos"
scene_dir = ROOT / "datasets/realestate10k_subset/scenes"

video_dir.mkdir(parents=True, exist_ok=True)
scene_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# STEP 1: READ SUBSET
# -------------------------------------------------------

video_entries = []

with open(subset_file, "r", encoding="utf-8-sig") as f:

    for line in f:

        filename = line.strip()
        if not filename:
            continue

        metadata = train_dir / filename

        if not metadata.exists():
            print("Missing metadata:", metadata)
            continue

        with open(metadata) as tf:
            url = tf.readline().strip()

        video_entries.append((url, metadata))

print("Scenes found:", len(video_entries))


# -------------------------------------------------------
# STEP 2: DOWNLOAD VIDEO (once per YouTube ID)
# -------------------------------------------------------

def download_video(url):

    video_id = url.split("v=")[-1]
    output = video_dir / f"{video_id}.mp4"

    if output.exists():
        print("Video exists:", video_id)
        return output

    print("Downloading:", video_id)

    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "-o", str(video_dir / "%(id)s.%(ext)s"),
        url
    ]

    subprocess.run(cmd)

    return output


# -------------------------------------------------------
# STEP 3: EXTRACT FRAMES USING METADATA TIMESTAMPS
# -------------------------------------------------------

def extract_scene_frames(video_path, metadata_path):

    scene_id = metadata_path.stem

    scene_path = scene_dir / scene_id
    frames_path = scene_path / "frames"

    frames_path.mkdir(parents=True, exist_ok=True)

    shutil.copy(metadata_path, scene_path / "metadata.txt")

    print("Processing scene:", scene_id)

    # read metadata timestamps
    with open(metadata_path) as f:
        lines = f.readlines()

    timestamps = []

    for line in lines[1:]:
        vals = line.split()
        timestamps.append(float(vals[0]) / 1000.0)  # microseconds → milliseconds

    cap = cv2.VideoCapture(str(video_path))

    saved = 0

    for i, t in enumerate(timestamps):

        # seek to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, t)

        ret, frame = cap.read()

        if not ret:
            print(f"Warning: failed to read frame at {t} ms")
            continue

        out_file = frames_path / f"{saved:05d}.jpg"
        cv2.imwrite(str(out_file), frame)

        saved += 1

    cap.release()

    print("Saved frames:", saved)


# -------------------------------------------------------
# PIPELINE
# -------------------------------------------------------

videos = {}

for url, metadata in video_entries:

    video_id = url.split("v=")[-1]

    if video_id not in videos:

        video_path = download_video(url)
        videos[video_id] = video_path

    else:

        video_path = videos[video_id]

    if video_path.exists():

        extract_scene_frames(video_path, metadata)

print("\nDataset preparation complete.")


# -------------------------------------------------------
# CLEANUP
# -------------------------------------------------------

print("\nDeleting videos...")

for video_path in videos.values():

    if video_path.exists():
        video_path.unlink()
        print("Deleted:", video_path.name)

print("\nDataset preparation complete.")