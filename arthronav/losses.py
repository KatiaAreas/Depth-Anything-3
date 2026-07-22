"""
Loss functions for training. Starting simple: plain L1 on valid depth
pixels. We can swap in something fancier (scale-invariant log loss,
edge-aware terms) once this baseline is confirmed working.
"""

import torch


def masked_l1_loss(pred, gt, mask):
    """
    pred, gt: (B, H, W)
    mask: (B, H, W) bool, True where gt is valid
    """
    if mask.sum() == 0:
        return pred.sum() * 0.0  # no valid pixels in this batch
    return (pred[mask] - gt[mask]).abs().mean()
