"""
SCARED dataset (h5-backed). Wraps the frame list from scared_io.py.

Resilience to corrupted files, added after a truncated .h5 file silently
killed a multi-day training run with no recovery:
  1. If a bad-files list (from scan_h5_integrity.py) is passed in, those
     frames are excluded up front, before training even starts.
  2. As a safety net for anything not caught by the scan (e.g. a file that
     goes bad mid-run), __getitem__ catches read errors, logs them once to
     a file, and retries with a different random index instead of crashing
     the whole DataLoader.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from arthronav.scared_io import load_frame_h5, load_frame_json


class SCAREDDataset(Dataset):
    def __init__(self, frame_list, bad_files_path: str | None = None,
                 runtime_error_log: str = "skipped_frames.log", max_retries: int = 5):
        self.frames = list(frame_list)
        self.runtime_error_log = runtime_error_log
        self.max_retries = max_retries

        if bad_files_path is not None:
            with open(bad_files_path) as f:
                bad_paths = {line.split("\t")[0].strip() for line in f if line.strip()}
            before = len(self.frames)
            self.frames = [fr for fr in self.frames if fr["h5_path"] not in bad_paths]
            removed = before - len(self.frames)
            print(f"Excluded {removed} known-bad frames (from {bad_files_path})")

    def __len__(self):
        return len(self.frames)

    def _load(self, idx):
        entry = self.frames[idx]
        rgb, depth = load_frame_h5(entry["h5_path"])
        pose, K = load_frame_json(entry["json_path"])

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, H, W)
        depth = torch.from_numpy(depth).float()                # (H, W)
        pose = torch.from_numpy(pose).float()                  # (4, 4)
        K = torch.from_numpy(K).float()                         # (3, 3)

        return {
            "rgb": rgb,
            "depth": depth,
            "pose": pose,
            "intrinsics": K,
        }

    def __getitem__(self, idx):
        last_error = None
        tried_idx = idx

        for attempt in range(self.max_retries):
            try:
                return self._load(tried_idx)
            except Exception as e:
                last_error = e
                bad_path = self.frames[tried_idx]["h5_path"]
                with open(self.runtime_error_log, "a") as f:
                    f.write(f"idx={tried_idx} path={bad_path} error={e}\n")
                tried_idx = random.randrange(len(self.frames))

        raise RuntimeError(
            f"Failed to load a valid frame after {self.max_retries} attempts "
            f"starting from idx={idx}. Last error: {last_error}"
        )
