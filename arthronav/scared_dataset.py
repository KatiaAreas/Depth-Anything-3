import torch
from torch.utils.data import Dataset

from arthronav.scared_io import load_frame_h5, load_frame_json


class SCAREDDataset(Dataset):
    def __init__(self, frame_list):
        self.frames = frame_list

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
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
