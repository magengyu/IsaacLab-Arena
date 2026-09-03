"""使用 SpaceMouse HDF5 示范训练一个最小行为克隆策略。

该脚本只依赖 PyTorch、NumPy 和 h5py，不需要启动 Isaac Sim：

    python3 examples/examples_teleop/example01_teleop_spacemouse_train_bc.py \
        --dataset_file /tmp/franka_spacemouse_demos.hdf5 \
        --epochs 100 --output /tmp/franka_spacemouse_bc.pt
"""

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn

OBS_TERM_ORDER = ["actions", "joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_pos"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC on SpaceMouse demonstrations.")
    parser.add_argument("--dataset_file", type=str, required=True, help="SpaceMouse HDF5 数据集。")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数。")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小。")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--output", type=str, default="./franka_spacemouse_bc.pt", help="模型输出路径。")
    return parser.parse_args()


def flatten_obs(obs_group: h5py.Group) -> np.ndarray:
    parts = [np.asarray(obs_group[term]) for term in OBS_TERM_ORDER if term in obs_group]
    if not parts:
        raise ValueError(f"obs 组中没有已知观测项，实际包含：{list(obs_group.keys())}")
    return np.concatenate(parts, axis=-1)


def load_dataset(dataset_file: str) -> tuple[np.ndarray, np.ndarray, int]:
    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    num_demos = 0

    with h5py.File(dataset_file, "r") as file:
        if "data" not in file:
            raise RuntimeError(f"{dataset_file} 中没有 data 组。")
        for name in sorted(file["data"]):
            demo = file["data"][name]
            if not name.startswith("demo_") or "actions" not in demo or "obs" not in demo:
                continue
            actions = np.asarray(demo["actions"])
            observations = flatten_obs(demo["obs"])
            length = min(len(actions), len(observations))
            if length == 0:
                continue
            action_list.append(actions[:length])
            obs_list.append(observations[:length])
            num_demos += 1
            print(
                f"  {name}: {length} 步, obs_dim={observations.shape[-1]}, "
                f"act_dim={actions.shape[-1]}, success={bool(demo.attrs.get('success', False))}"
            )

    if not obs_list:
        raise RuntimeError(f"{dataset_file} 中没有可用 demo，请先运行 SpaceMouse 录制脚本。")
    return np.concatenate(obs_list), np.concatenate(action_list), num_demos


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    observations, actions, num_demos = load_dataset(args.dataset_file)
    if len(observations) < 2:
        raise RuntimeError("数据至少需要两个时间步，才能拆分训练集和验证集。")

    obs_tensor = torch.from_numpy(observations.astype(np.float32))
    action_tensor = torch.from_numpy(actions.astype(np.float32))
    permutation = torch.randperm(len(obs_tensor))
    obs_tensor = obs_tensor[permutation]
    action_tensor = action_tensor[permutation]
    num_train = min(max(int(0.8 * len(obs_tensor)), 1), len(obs_tensor) - 1)
    train_obs, val_obs = obs_tensor[:num_train], obs_tensor[num_train:]
    train_actions, val_actions = action_tensor[:num_train], action_tensor[num_train:]

    model = BCPolicy(observations.shape[1], actions.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for start in range(0, len(train_obs), args.batch_size):
            batch_obs = train_obs[start : start + args.batch_size]
            batch_actions = train_actions[start : start + args.batch_size]
            loss = criterion(model(batch_obs), batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_obs)

        train_loss = total_loss / len(train_obs)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_obs), val_actions).item()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch + 1:3d}/{args.epochs}: train={train_loss:.6f}, val={val_loss:.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": observations.shape[1],
            "act_dim": actions.shape[1],
            "obs_term_order": OBS_TERM_ORDER,
            "num_demos": num_demos,
            "source_dataset": os.path.abspath(args.dataset_file),
        },
        args.output,
    )
    print(f"训练完成：{num_demos} 条 demo，模型已保存到 {args.output}")


if __name__ == "__main__":
    main()
