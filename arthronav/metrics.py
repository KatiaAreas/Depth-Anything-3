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


def abs_error_stats(pred, gt, mask):
    """Raw absolute error in meters: min, max, mean, over valid pixels only."""
    if mask.sum() == 0:
        return {"min_error_m": float("nan"), "max_error_m": float("nan"), "mean_error_m": float("nan")}
    errors = (pred[mask] - gt[mask]).abs()
    return {
        "min_error_m": errors.min().item(),
        "max_error_m": errors.max().item(),
        "mean_error_m": errors.mean().item(),
    }
