import os
import csv
import math
import traceback
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

#TODO ARCHIEVE for train only not train_v1.py
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """
    pred, target: [B,3,H,W] or [3,H,W], assumed in [0,1]
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    mse = F.mse_loss(pred, target, reduction="mean").item()
    psnr = -10.0 * math.log10(max(mse, eps))
    return psnr