"""
Run validation: evaluate depth predictions on held-out SCARED frames
(val_frames -- never seen during training), report AbsRel and RMSE.

Usage:
    python -m arthronav.validate_scared --subset-fraction 0.05
"""

import argparse

import torch
#torch.backends.cudnn.enabled = False

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from arthronav.lora import inject_lora
from arthronav.metrics import compute_all_metrics
from arthronav.scared_io import build_frame_list, split_frames
from arthronav.scared_dataset import SCAREDDataset
from arthronav.train_scared import prepare_batch, H5_ROOT, JSON_ROOT

import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--subset-fraction", type=float, default=1.0,
                     help="fraction of val frames to use, for a quicker check")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="path to a saved LoRA checkpoint (.pt); omit to evaluate the base pretrained model")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = model.model
    inject_lora(net, rank=args.lora_rank)
    net = net.to(device)
    net.eval()

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint given -- evaluating base pretrained model (no fine-tuning).")

    print("Building frame list...")
    frames = build_frame_list(H5_ROOT, JSON_ROOT)
    train_frames, val_frames = split_frames(frames)

    if args.subset_fraction < 1.0:
        rng = random.Random(42)
        n_keep = max(1, int(len(val_frames) * args.subset_fraction))
        val_frames = rng.sample(val_frames, n_keep)

    print(f"Validating on {len(val_frames)} frames")
    val_ds = SCAREDDataset(val_frames)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_abs_rel = []
    all_rmse = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="validating"):
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)
            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            metrics = compute_all_metrics(pred, depth_gt, valid_mask)
            all_abs_rel.append(metrics["AbsRel"])
            all_rmse.append(metrics["RMSE"])

    mean_abs_rel = sum(all_abs_rel) / len(all_abs_rel)
    mean_rmse = sum(all_rmse) / len(all_rmse)

    print(f"\nValidation results ({len(val_frames)} frames):")
    print(f"  AbsRel: {mean_abs_rel:.4f}")
    print(f"  RMSE:   {mean_rmse:.4f}")


if __name__ == "__main__":
    main()
