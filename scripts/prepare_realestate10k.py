import subprocess
from pathlib import Path
import cv2

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

subset_file = ROOT / "datasets/realestate10k_subset/subset_train.txt"
train_dir = ROOT / "datasets/realestate10k_subset/RealEstate10K/train"

video_dir = ROOT / "datasets/realestate10k_subset/videos"
frame_dir = ROOT / "datasets/realestate10k_subset/frames"

video_dir.mkdir(parents=True, exist_ok=True)
frame_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# STEP 1: Collect URLs
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

print("Videos found:", len(video_entries))


# -------------------------------------------------------
# STEP 2: Download video
# -------------------------------------------------------

def download_video(url):

    video_id = url.split("v=")[-1]
    output = video_dir / f"{video_id}.mp4"

    if output.exists():
        print("Already downloaded:", video_id)
        return output

    print("Downloading:", url)

    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "-o", str(video_dir / "%(id)s.%(ext)s"),
        url
    ]

    subprocess.run(cmd)

    return output


# -------------------------------------------------------
# STEP 3: Convert video → frames (OpenCV)
# -------------------------------------------------------

def video_to_frames(video_path):

    video_id = video_path.stem
    out_dir = frame_dir / video_id
    out_dir.mkdir(exist_ok=True)

    print("Extracting frames:", video_id)

    cap = cv2.VideoCapture(str(video_path))

    frame_index = 0
    saved_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # save every 10th frame (~3fps if video ≈30fps)
        if frame_index % 10 == 0:

            out_path = out_dir / f"{saved_index:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_index += 1

        frame_index += 1

    cap.release()

    print("Saved frames:", saved_index)


# -------------------------------------------------------
# PIPELINE
# -------------------------------------------------------

videos = []

for url, metadata in video_entries:

    video = download_video(url)

    if video.exists():
        videos.append(video)

print("Downloaded videos:", len(videos))


for video in videos:
    video_to_frames(video)

print("Dataset preparation complete.")