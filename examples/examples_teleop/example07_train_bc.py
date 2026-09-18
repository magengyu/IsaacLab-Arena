"""Level 7：行为克隆（BC）训练 — 用 example05 录制的示范训练 Franka 策略。

纯 PyTorch + h5py，**不需要 Isaac Sim**，任意装了 torch/h5py 的 Python 都能跑（宿主机
python3 或容器内 /isaac-sim/python.sh 均可）：

    python3 examples/examples_teleop/example07_train_bc.py \\
        --dataset_file /path/to/franka_demos.hdf5 --epochs 100 --output /path/to/franka_bc.pt

原理：把遥操作录制得到的 (obs, action) 对当作监督学习数据，训练一个 MLP
      π(a|s)：输入观测 obs → 输出预测动作，最小化 ||π(obs) - action||²。
      本质就是监督学习的 MSE Loss（跟 MNIST 一样）。

关键点：
  - HDF5 里每条 demo（data/demo_N/）存了 actions 和 obs。
  - obs 是 dict（joint_pos / eef_pos / eef_quat / gripper_pos / ...），要按
    OBS_TERM_ORDER 固定顺序拼成一个向量。
  - 训练出的 policy 部署方式：把 obs 拼成同样的向量喂给网络，拿输出当 action
    喂回 env.step()。
"""

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn

# obs 各分量的拼接顺序（与 FrankaObservationsCfg 的 policy 组一致）。
OBS_TERM_ORDER = ["actions", "joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_pos"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Train a BC policy on recorded Franka demos.")
    parser.add_argument("--dataset_file", type=str, required=True, help="example05 录制的 HDF5 路径。")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数。")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小。")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率。")
    parser.add_argument("--output", type=str, default="./franka_bc.pt", help="checkpoint 输出路径。")
    return parser.parse_args()


def flatten_obs(obs_group: h5py.Group) -> np.ndarray:
    """把 HDF5 里的 obs dict（各 term 数组）按 OBS_TERM_ORDER 拼成一个向量。

    Args:
        obs_group: HDF5 里某条 demo 的 ``obs`` 组。

    Returns:
        shape (T, obs_dim) 的数组。
    """
    parts = [np.asarray(obs_group[term]) for term in OBS_TERM_ORDER if term in obs_group]
    if not parts:
        raise ValueError(f"obs 组里没有任何已知分量，实际有：{list(obs_group.keys())}")
    return np.concatenate(parts, axis=-1)


def load_dataset(dataset_file: str) -> tuple[np.ndarray, np.ndarray, int]:
    """读 HDF5，汇总所有 demo 的 (obs, action) 对。

    Returns:
        (obs_all, act_all, num_demos)：obs_all/act_all 都是 (总步数, dim)。
    """
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    num_demos = 0

    with h5py.File(dataset_file, "r") as f:
        data = f["data"]
        for name in sorted(data):
            if not name.startswith("demo_") or "actions" not in data[name]:
                continue
            demo = data[name]
            act = np.asarray(demo["actions"])  # (T, act_dim)
            if "obs" not in demo:
                continue
            obs = flatten_obs(demo["obs"])
            # 对齐步数（两者都应是 T 步）。
            length = min(len(act), len(obs))
            act_list.append(act[:length])
            obs_list.append(obs[:length])
            num_demos += 1
            success = bool(demo.attrs.get("success", False))
            print(f"  {name}: {length} 步, obs_dim={obs.shape[-1]}, act_dim={act.shape[-1]}, success={success}")

    if num_demos == 0:
        raise RuntimeError(f"{dataset_file} 里没有可用 demo，请先运行 example05_teleop_quest_record.py 录制。")

    return np.concatenate(obs_list, axis=0), np.concatenate(act_list, axis=0), num_demos


class BCPolicy(nn.Module):
    """输入 obs 向量 → 输出预测动作的小 MLP。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    """加载数据、训练 MLP、保存 checkpoint。"""
    args = parse_args()

    print("=" * 60)
    print("1. 加载录制数据...")
    print("=" * 60)
    obs_all, act_all, num_demos = load_dataset(args.dataset_file)
    print(f"\n共 {num_demos} 条 demo，{obs_all.shape[0]} 步。")
    print(f"  obs 维度:  {obs_all.shape}")
    print(f"  action 维度: {act_all.shape}")

    # 转 float32 tensor，按 80/20 拆 train/val。
    obs_t = torch.from_numpy(obs_all.astype(np.float32))
    act_t = torch.from_numpy(act_all.astype(np.float32))
    n = len(obs_t)
    n_train = int(0.8 * n)
    perm = torch.randperm(n)
    obs_t, act_t = obs_t[perm], act_t[perm]
    train_obs, train_act = obs_t[:n_train], act_t[:n_train]
    val_obs, val_act = obs_t[n_train:], act_t[n_train:]

    print("=" * 60)
    print("2. 训练 BC 网络（监督学习）...")
    print("=" * 60)
    model = BCPolicy(obs_all.shape[1], act_all.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    print(f"网络结构:\n{model.net}")

    train_losses, val_losses = [], []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for i in range(0, len(train_obs), args.batch_size):
            b_obs = train_obs[i : i + args.batch_size]
            b_act = train_act[i : i + args.batch_size]
            pred = model(b_obs)
            loss = criterion(pred, b_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b_obs)
        train_losses.append(epoch_loss / len(train_obs))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_obs), val_act).item()
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch + 1:3d}/{args.epochs}: train_loss={train_losses[-1]:.6f}  val_loss={val_losses[-1]:.6f}")

    # 保存 checkpoint（含维度信息，部署时复用）。
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": obs_all.shape[1],
            "act_dim": act_all.shape[1],
            "obs_term_order": OBS_TERM_ORDER,
        },
        args.output,
    )
    print(f"\n训练完成，checkpoint 已保存到: {args.output}")
    print(f"最终 train_loss={train_losses[-1]:.6f}, val_loss={val_losses[-1]:.6f}")
    print("\n✅ BC 训练完成！核心收获：")
    print("   1. 行为克隆 = 收集示范 (obs, action) → 监督学习 → 预测动作")
    print("   2. 部署时把 obs 按同样顺序拼成向量喂给网络，输出当 action 喂回 env.step()")
    print("   3. 下一步可加 rollout 评估（example08），看策略能不能自己把方块放进目标区")


if __name__ == "__main__":
    main()
