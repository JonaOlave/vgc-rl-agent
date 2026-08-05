"""
PyTorch Dataset for behavioral cloning from parsed VGC replays.
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ReplayDataset(Dataset):
    """
    Dataset of (observation, action) pairs extracted from PS replays.

    Each item:
        obs    : FloatTensor (925,)
        action : LongTensor  (2,)   — [action_slot0, action_slot1]
    """

    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        if not samples:
            raise ValueError("Empty sample list")
        obs_list, act_list = zip(*samples)
        self.obs = torch.from_numpy(np.stack(obs_list)).float()
        self.actions = torch.from_numpy(np.stack(act_list)).long()

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.obs[idx], self.actions[idx]

    @staticmethod
    def from_directory(replay_dir: str) -> "ReplayDataset":
        """Parse all *.json replays in a directory and build a dataset."""
        import json
        from pathlib import Path
        from .parser import parse_replay

        path = Path(replay_dir)
        files = sorted(path.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No replay JSON files found in {path}")

        all_samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for f in files:
            try:
                data = json.loads(f.read_text())
                samples = parse_replay(data)
                all_samples.extend(samples)
            except Exception as e:
                print(f"  Skipping {f.name}: {e}")

        print(f"Loaded {len(all_samples)} samples from {len(files)} replays")
        return ReplayDataset(all_samples)
