import json
import numpy as np
import h5py
from pathlib import Path
import random


def load_frame_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

    pose = np.array(data["camera-pose"], dtype=np.float32)          # (4, 4)
    K = np.array(data["camera-calibration"]["KL"], dtype=np.float32)  # (3, 3)

    return pose, K
    

def load_frame_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        rgb = f["rgb"][:]      # (H, W, 3) float32, [0, 1]
        depth = f["depth"][:]  # (H, W) float32, NOT meters -- raw_mm / 256 (see README note), multiply by 0.256 for real meters

    return rgb, depth
    


EXCLUDED_DATASETS = {"dataset_4", "dataset_5"}


def build_frame_list(h5_root, json_root, exclude=EXCLUDED_DATASETS):
    """
    Returns a list of dicts: {h5_path, json_path, dataset, keyframe, frame_idx}
    """
    h5_root = Path(h5_root)
    json_root = Path(json_root)
    frames = []

    for dataset_dir in sorted(h5_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in exclude:
            continue
        for keyframe_dir in sorted(dataset_dir.iterdir()):
            if not keyframe_dir.is_dir():
                continue
            for h5_file in sorted(keyframe_dir.glob("*.h5")):
                frame_idx = h5_file.stem  # e.g. "000000"
                json_file = json_root / dataset_dir.name / keyframe_dir.name / f"frame_data{frame_idx}.json"
                if not json_file.exists():
                    continue  # no matching pose (e.g. keyframes with no interpolation)
                frames.append({
                    "h5_path": str(h5_file),
                    "json_path": str(json_file),
                    "dataset": dataset_dir.name,
                    "keyframe": keyframe_dir.name,
                    "frame_idx": frame_idx,
                })

    return frames
    
    



def split_frames(frames, val_fraction=0.1, seed=42):
    """
    Splits by keyframe (not by individual frame) so consecutive video
    frames from the same keyframe don't leak across train/val.
    """
    keyframes = sorted(set((f["dataset"], f["keyframe"]) for f in frames))
    rng = random.Random(seed)
    rng.shuffle(keyframes)

    n_val = max(1, int(len(keyframes) * val_fraction))
    val_keyframes = set(keyframes[:n_val])

    train_frames = [f for f in frames if (f["dataset"], f["keyframe"]) not in val_keyframes]
    val_frames = [f for f in frames if (f["dataset"], f["keyframe"]) in val_keyframes]

    return train_frames, val_frames
