#!/usr/bin/env python3
"""Train a tiny behavior-cloning baseline on simple arm HDF5 demos.

This script does not start Isaac Sim:

    python3 examples/examples_urdf_all_train/05_train_bc_simple_arm.py \
      --dataset_file /tmp/simple_arm_demos.hdf5 --epochs 100 --output /tmp/simple_arm_bc.pt
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn

OBS_TERM_ORDER = ["actions", "joint_pos", "joint_vel", "eef_pos", "eef_quat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC on simple URDF arm demos.")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="/tmp/simple_arm_bc.pt")
    return parser.parse_args()


def flatten_obs(obs_group: h5py.Group) -> np.ndarray:
    parts = [np.asarray(obs_group[term]) for term in OBS_TERM_ORDER if term in obs_group]
    if not parts:
        raise ValueError(f"obs group has no known terms. keys={list(obs_group.keys())}")
    return np.concatenate(parts, axis=-1)


def load_dataset(dataset_file: str) -> tuple[np.ndarray, np.ndarray, int]:
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    num_demos = 0
    with h5py.File(dataset_file, "r") as f:
        data = f["data"]
        for name in sorted(data):
            if not name.startswith("demo_") or "actions" not in data[name] or "obs" not in data[name]:
                continue
            demo = data[name]
            act = np.asarray(demo["actions"])
            obs = flatten_obs(demo["obs"])
            length = min(len(act), len(obs))
            obs_list.append(obs[:length])
            act_list.append(act[:length])
            num_demos += 1
            print(f"{name}: steps={length}, obs_dim={obs.shape[-1]}, act_dim={act.shape[-1]}")
    if not obs_list:
        raise RuntimeError(f"No usable demos found in {dataset_file}")
    return np.concatenate(obs_list, axis=0), np.concatenate(act_list, axis=0), num_demos


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def main() -> None:
    args = parse_args()
    obs_all, act_all, num_demos = load_dataset(args.dataset_file)
    print(f"[INFO] demos={num_demos}, samples={len(obs_all)}, obs_dim={obs_all.shape[-1]}, act_dim={act_all.shape[-1]}")

    obs_t = torch.from_numpy(obs_all.astype(np.float32))
    act_t = torch.from_numpy(act_all.astype(np.float32))
    perm = torch.randperm(len(obs_t))
    obs_t = obs_t[perm]
    act_t = act_t[perm]
    n_train = max(1, int(0.8 * len(obs_t)))
    train_obs, val_obs = obs_t[:n_train], obs_t[n_train:]
    train_act, val_act = act_t[:n_train], act_t[n_train:]

    model = BCPolicy(obs_all.shape[-1], act_all.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for start in range(0, len(train_obs), args.batch_size):
            b_obs = train_obs[start : start + args.batch_size]
            b_act = train_act[start : start + args.batch_size]
            loss = criterion(model(b_obs), b_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b_obs)
        train_loss = total / len(train_obs)
        if len(val_obs) > 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(val_obs), val_act).item()
        else:
            val_loss = float("nan")
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch + 1:03d}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": obs_all.shape[-1],
            "act_dim": act_all.shape[-1],
            "obs_term_order": OBS_TERM_ORDER,
        },
        args.output,
    )
    print(f"[INFO] saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
