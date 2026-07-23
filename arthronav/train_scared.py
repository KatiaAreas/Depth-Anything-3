"""
Fine-tune DA3METRIC-LARGE on SCARED with LoRA adapters.

Usage:
    python -m arthronav.train_scared --epochs 5 --batch-size 2
    python -m arthronav.train_scared --epochs 1 --batch-size 1 --subset-fraction 0.2

Notes:
    - batch-size 1 is currently the max this GPU (24GB) can hold at full
      1022x1274 resolution with backprop. Batch size 2 runs out of memory.
    - Full training set is 15,846 frames -> ~5.5h/epoch at batch-size 1.
      Use --subset-fraction (e.g. 0.2) to train on a random subset for
      faster iteration while still validating the pipeline; drop it (or
      set to 1.0) for a real full-dataset training run.
    - Only the trainable subset (LoRA adapters) gets checkpointed each
      epoch, not the full 336M-parameter model -- a few MB instead of
      over a gigabyte.
    - Each run gets its own timestamped checkpoint folder by default
      (checkpoints/run_<timestamp>), so different runs never overwrite
      each other's checkpoints. Pass --checkpoint-dir to override.
    - Every step's loss is logged to loss_log.csv inside that folder,
      so loss curves can be plotted later without depending on terminal
      scrollback.
"""

import argparse
import csv
import os
import random
from datetime import datetime

import torch
#torch.backends.cudnn.enabled = False  # workaround for the cuDNN version conflict on this machine

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from arthronav.lora import inject_lora
from arthronav.losses import masked_l1_loss
from arthronav.scared_io import build_frame_list, split_frames
from arthronav.scared_dataset import SCAREDDataset

H5_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything"
JSON_ROOT = "/mnt/areas_nas/SLAM/scared_dataset_full_copy/frame_trajectory_data"
TARGET_SIZE = (1022, 1274)  # nearest multiples of 14 below the native 1024x1280


def prepare_batch(batch, device):
    rgb = F.interpolate(batch["rgb"], size=TARGET_SIZE, mode="bilinear", align_corners=False)
    rgb = rgb.unsqueeze(1).to(device)  # (B, 1, 3, H, W)

    depth_gt = F.interpolate(batch["depth"].unsqueeze(1), size=TARGET_SIZE, mode="nearest")
    depth_gt = depth_gt.squeeze(1).to(device)  # (B, H, W)

    valid_mask = depth_gt > 1e-4
    return rgb, depth_gt, valid_mask


def save_trainable_checkpoint(net, path):
    """Saves only parameters with requires_grad=True (the LoRA adapters)."""
    trainable_names = {name for name, p in net.named_parameters() if p.requires_grad}
    state = {name: tensor for name, tensor in net.state_dict().items() if name in trainable_names}
    torch.save(state, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--subset-fraction", type=float, default=1.0,
                     help="fraction of training frames to use (0 < f <= 1), for faster iteration")
    ap.add_argument("--checkpoint-dir", type=str, default=None,
                     help="directory to save checkpoints; defaults to checkpoints/run_<timestamp>")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = model.model
    adapted = inject_lora(net, rank=args.lora_rank)
    net = net.to(device)
    net.train()
    print(f"LoRA injected into {len(adapted)} layers")

    if args.checkpoint_dir is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        args.checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    log_path = os.path.join(args.checkpoint_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "loss"])

    print("Building frame list...")
    frames = build_frame_list(H5_ROOT, JSON_ROOT)
    train_frames, val_frames = split_frames(frames)

    if args.subset_fraction < 1.0:
        rng = random.Random(42)
        n_keep = max(1, int(len(train_frames) * args.subset_fraction))
        train_frames = rng.sample(train_frames, n_keep)
        print(f"Using subset: {n_keep} / {len(frames)} frames ({args.subset_fraction:.0%})")

    train_ds = SCAREDDataset(train_frames)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)

            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)

            loss = masked_l1_loss(pred, depth_gt, valid_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, n_batches, loss.item()])

        print(f"epoch {epoch} | avg loss: {epoch_loss / n_batches:.4f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        save_trainable_checkpoint(net, ckpt_path)
        print(f"saved checkpoint: {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
