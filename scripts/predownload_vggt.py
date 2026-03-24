import argparse
import os
import shutil

import torch


DEFAULT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def main():
    parser = argparse.ArgumentParser(description="Pre-download official VGGT weights")
    parser.add_argument("--url", default=DEFAULT_URL, help="VGGT checkpoint URL")
    parser.add_argument(
        "--cache-root",
        default="/scratch/huggingface",
        help="Root cache directory on HPC",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path for model.pt",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_root, exist_ok=True)
    os.environ["TORCH_HOME"] = args.cache_root

    checkpoints_dir = os.path.join(args.cache_root, "vggt")
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"Downloading VGGT weights from: {args.url}")
    state_dict = torch.hub.load_state_dict_from_url(
        args.url,
        model_dir=checkpoints_dir,
        map_location="cpu",
        progress=True,
    )

    downloaded_path = os.path.join(checkpoints_dir, os.path.basename(args.url))
    final_output = args.output or downloaded_path

    if downloaded_path != final_output:
        os.makedirs(os.path.dirname(final_output), exist_ok=True)
        shutil.copy2(downloaded_path, final_output)

    print(f"Saved VGGT checkpoint to: {final_output}")
    print(f"Loaded {len(state_dict)} tensors")


if __name__ == "__main__":
    main()
