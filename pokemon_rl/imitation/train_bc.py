"""
Behavioral Cloning (BC) pre-training from VGC replays.

Trains the PPO policy in a supervised fashion before RL fine-tuning.
The saved model can be loaded directly as a PPO starting checkpoint.

Usage:
    # Download replays + train:
    python -m pokemon_rl.imitation.train_bc --download --n-download 200

    # Train from already-downloaded replays:
    python -m pokemon_rl.imitation.train_bc --replay-dir data/replays

    # Then fine-tune with RL:
    python -m pokemon_rl.vgc.train --format gen9vgc2025regg --bc-model models/bc_pretrained
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, random_split

from .dataset import ReplayDataset
from .downloader import download_replays
from .parser import parse_replay


# ─────────────────────────────────────────────────────────────────────────────
# Dummy env — only used to build the SB3 policy architecture offline
# ─────────────────────────────────────────────────────────────────────────────

def _build_dummy_env():
    import gymnasium as gym
    from gymnasium import spaces

    class _DummyDoublesEnv(gym.Env):
        observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-1.0, 1.0, (925,), np.float32),
                "action_mask": spaces.Box(0, 1, (214,), np.int64),
            }
        )
        action_space = spaces.MultiDiscrete([107, 107])

        def reset(self, **kwargs):
            obs = {
                "observation": np.zeros(925, dtype=np.float32),
                "action_mask": np.ones(214, dtype=np.int64),
            }
            return obs, {}

        def step(self, action):
            return self.reset()[0], 0.0, True, False, {}

    return _DummyDoublesEnv()


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _obs_batch_to_tensor(obs_np: torch.Tensor, policy, device: torch.device):
    """
    Convert a batch of raw observation vectors to the dict format that
    SB3's MultiInputPolicy expects, then run obs_to_tensor.
    """
    batch_size = obs_np.shape[0]
    obs_dict_np = {
        "observation": obs_np.cpu().numpy(),
        "action_mask": np.ones((batch_size, 214), dtype=np.int64),
    }
    obs_tensor, _ = policy.obs_to_tensor(obs_dict_np)
    return obs_tensor


def _forward_logits(policy, obs_tensor):
    """
    Run the policy network and return raw action logits.

    For MultiDiscrete([107, 107]) the action_net output has shape
    (batch, 214) = (batch, 107 + 107).
    """
    features = policy.extract_features(obs_tensor, policy.features_extractor)
    if policy.share_features_extractor:
        latent_pi, _ = policy.mlp_extractor(features)
    else:
        pi_features, _ = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
    return policy.action_net(latent_pi)  # (batch, 214)


def _run_epoch(
    policy,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
) -> dict:
    training = optimizer is not None
    policy.train(training)

    total_loss = 0.0
    correct0 = correct1 = total = 0

    for obs_batch, act_batch in loader:
        obs_batch = obs_batch.to(device)
        act_batch = act_batch.to(device)

        obs_tensor = _obs_batch_to_tensor(obs_batch, policy, device)
        logits = _forward_logits(policy, obs_tensor)
        logits0, logits1 = logits[:, :107], logits[:, 107:]

        loss = (criterion(logits0, act_batch[:, 0]) + criterion(logits1, act_batch[:, 1])) / 2

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred0 = logits0.argmax(dim=1)
        pred1 = logits1.argmax(dim=1)
        correct0 += (pred0 == act_batch[:, 0]).sum().item()
        correct1 += (pred1 == act_batch[:, 1]).sum().item()
        total += act_batch.shape[0]

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "acc0": correct0 / max(total, 1),
        "acc1": correct1 / max(total, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_bc(
    replay_dir: str = "data/replays",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    val_fraction: float = 0.1,
    save_path: str = "models/bc_pretrained",
    download: bool = False,
    n_download: int = 200,
    format_id: str = "gen9vgc2025regg",
) -> PPO:
    """
    Pre-train a PPO policy via behavioral cloning on VGC replays.

    Parameters
    ----------
    replay_dir    : Directory containing (or to contain) replay JSON files
    epochs        : Training epochs
    batch_size    : Mini-batch size
    lr            : Learning rate
    val_fraction  : Fraction of data used for validation
    save_path     : Where to save the BC-pretrained model (without .zip)
    download      : Whether to download replays before training
    n_download    : Number of replays to download if download=True
    format_id     : PS format for replay download/search

    Returns
    -------
    PPO model with BC-pretrained policy weights
    """
    if download:
        print(f"Downloading {n_download} replays for {format_id}...")
        download_replays(
            format_id=format_id,
            n_replays=n_download,
            output_dir=replay_dir,
        )

    # Load dataset
    print(f"\nParsing replays from {replay_dir}/ ...")
    try:
        dataset = ReplayDataset.from_directory(replay_dir)
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("Run with --download to fetch replays first.")
        raise

    print(f"Dataset: {len(dataset)} samples")

    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Build PPO model on dummy env to get the right architecture
    print("\nBuilding PPO policy architecture...")
    dummy_env = _build_dummy_env()
    model = PPO(
        policy="MultiInputPolicy",
        env=dummy_env,
        verbose=0,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
    )

    policy = model.policy
    device = policy.device
    print(f"Policy device: {device}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nTraining BC: {n_train} train, {n_val} val, {epochs} epochs\n")
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_stats = _run_epoch(policy, train_loader, device, optimizer, criterion)
        with torch.no_grad():
            val_stats = _run_epoch(policy, val_loader, device, None, criterion)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_stats['loss']:.4f}  "
            f"val_loss={val_stats['loss']:.4f}  "
            f"val_acc=[{val_stats['acc0']:.3f}, {val_stats['acc1']:.3f}]"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

    # Restore best weights
    if best_state is not None:
        policy.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\nRestored best weights (val_loss={best_val_loss:.4f})")

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"BC model saved → {save_path}.zip")

    return model


def parse_args():
    p = argparse.ArgumentParser(description="BC pre-training from VGC replays")
    p.add_argument("--replay-dir", default="data/replays",
                   help="Directory with replay JSON files")
    p.add_argument("--download", action="store_true",
                   help="Download replays before training")
    p.add_argument("--n-download", type=int, default=200,
                   help="Number of replays to download")
    p.add_argument("--format", default="gen9vgc2025regg",
                   dest="format_id",
                   help="PS format for replay download")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-path", default="models/bc_pretrained")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_bc(
        replay_dir=args.replay_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        download=args.download,
        n_download=args.n_download,
        format_id=args.format_id,
    )
