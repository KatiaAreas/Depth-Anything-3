"""
Standard depth-estimation metrics, matching the convention used across
the endoscopic-depth literature (EndoDAC, Endo-FASt3r, etc.) so numbers
are directly comparable.
"""

import torch


def abs_rel(pred, gt, mask):
    """Mean absolute relative error: mean(|pred - gt| / gt), valid pixels only."""
    if mask.sum() == 0:
        return torch.tensor(float("nan"))
    return (torch.abs(pred[mask] - gt[mask]) / gt[mask]).mean()


def rmse(pred, gt, mask):
    """Root mean squared error, valid pixels only."""
    if mask.sum() == 0:
        return torch.tensor(float("nan"))
    return torch.sqrt(((pred[mask] - gt[mask]) ** 2).mean())


def compute_all_metrics(pred, gt, mask):
    return {
        "AbsRel": abs_rel(pred, gt, mask).item(),
        "RMSE": rmse(pred, gt, mask).item(),
    }
