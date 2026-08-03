"""
Benchmark training throughput across a range of batch sizes, in one run,
so results are directly comparable (same warmup, same measurement window,
same machine state) instead of manually running separate commands.

Usage:
    python -m arthronav.benchmark_batch_size --batch-sizes 1 2 3 4 6 8
"""

import argparse
import time

import torch
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3

from arthronav.lora import inject_lora
from arthronav.losses import masked_l1_loss
from arthronav.scared_io import build_frame_list, split_frames
from arthronav.scared_dataset import SCAREDDataset
from torch.utils.data import DataLoader

H5_ROOT = "/data/scared_dataset_full_copy/depth_anything_preprocessed_data/train_depth_anything"
JSON_ROOT = "/data/scared_dataset_full_copy/frame_trajectory_data"
TARGET_SIZE = (1022, 1274)


def prepare_batch(batch, device):
    rgb = F.interpolate(batch["rgb"], size=TARGET_SIZE, mode="bilinear", align_corners=False)
    rgb = rgb.unsqueeze(1).to(device)
    depth_gt = F.interpolate(batch["depth"].unsqueeze(1), size=TARGET_SIZE, mode="nearest")
    depth_gt = depth_gt.squeeze(1).to(device)
    valid_mask = depth_gt > 1e-4
    return rgb, depth_gt, valid_mask


def build_fresh_model(device, lora_rank=16):
    """Fresh model + optimizer per batch size, so no state leaks between tests."""
    wrapper = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    net = wrapper.model
    inject_lora(net, rank=lora_rank)
    net = net.to(device)
    net.train()
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    return net, optimizer


def benchmark_one_batch_size(batch_size, loader_source, device, warmup_steps=2, timed_steps=5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    net, optimizer = build_fresh_model(device)
    loader = DataLoader(loader_source, batch_size=batch_size, shuffle=True,
                         num_workers=4, drop_last=True)

    step = 0
    times = []
    try:
        for batch in loader:
            rgb, depth_gt, valid_mask = prepare_batch(batch, device)

            torch.cuda.synchronize()
            t0 = time.time()

            output = net(rgb, export_feat_layers=[])
            pred = output.depth.squeeze(1)
            loss = masked_l1_loss(pred, depth_gt, valid_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            t1 = time.time()

            step += 1
            if step > warmup_steps:
                times.append(t1 - t0)
            if step >= warmup_steps + timed_steps:
                break

        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        avg_time = sum(times) / len(times)
        it_per_sec = 1.0 / avg_time
        samples_per_sec = it_per_sec * batch_size

        return {
            "batch_size": batch_size,
            "status": "OK",
            "peak_mem_gb": round(peak_mem_gb, 2),
            "it_per_sec": round(it_per_sec, 3),
            "samples_per_sec": round(samples_per_sec, 2),
        }

    except torch.OutOfMemoryError:
        return {
            "batch_size": batch_size,
            "status": "OOM",
            "peak_mem_gb": None,
            "it_per_sec": None,
            "samples_per_sec": None,
        }
    finally:
        del net, optimizer
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--subset-fraction", type=float, default=0.02)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU total memory: {total_mem_gb:.1f} GB")

    print("Building frame list...")
    frames = build_frame_list(H5_ROOT, JSON_ROOT)
    train_frames, _ = split_frames(frames)

    import random
    rng = random.Random(42)
    n_keep = max(64, int(len(train_frames) * args.subset_fraction))
    subset = rng.sample(train_frames, min(n_keep, len(train_frames)))
    ds = SCAREDDataset(subset, bad_files_path="bad_h5_files.txt")
    print(f"Benchmark subset size: {len(ds)} frames")

    results = []
    for bs in args.batch_sizes:
        print(f"\nTesting batch size {bs}...")
        result = benchmark_one_batch_size(bs, ds, device)
        results.append(result)
        print(result)

    print("\n" + "=" * 70)
    print(f"{'Batch':>6} | {'Status':>6} | {'Peak Mem (GB)':>14} | {'it/s':>8} | {'samples/s':>10}")
    print("-" * 70)
    for r in results:
        mem = f"{r['peak_mem_gb']:.2f}" if r["peak_mem_gb"] is not None else "-"
        its = f"{r['it_per_sec']:.3f}" if r["it_per_sec"] is not None else "-"
        sps = f"{r['samples_per_sec']:.2f}" if r["samples_per_sec"] is not None else "-"
        print(f"{r['batch_size']:>6} | {r['status']:>6} | {mem:>14} | {its:>8} | {sps:>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
