#!/bin/bash
#SBATCH --job-name=3dgs_train
#SBATCH --account=def-vislearn
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:30:00
#SBATCH --output=logs/%x-%j.out

echo "===================================="
echo "Job started on $(hostname)"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"
echo "===================================="

# Load modules
module load gcc
module load scipy-stack/2024a
module load arrow/18.1.0
module load opencv/4.10.0
module load cuda/12.2

# GPU arch for H100
export TORCH_CUDA_ARCH_LIST="9.0"

# HuggingFace cache
export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_DATASETS_CACHE=$SCRATCH/huggingface

# Avoid ~/.local interference
export PYTHONNOUSERSITE=1

# Move to project directory
cd ~/links/scratch/weiz_3dgs_rp

# Activate Python environment
source weiz_3dgs/bin/activate

echo "GPU info:"
nvidia-smi

echo "===================================="
echo "Starting training..."
echo "===================================="

python -u video_infer.py

echo "===================================="
echo "Training finished at $(date)"
echo "===================================="