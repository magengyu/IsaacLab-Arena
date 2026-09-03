"""Level 6：回放录制的示范（replay HDF5）。

把 example05 录制的 HDF5 在 Isaac Sim 里逐帧回放，视觉确认录制的轨迹对不对。
**不需要 Quest / 遥操作设备**：重建同样的场景，把录制的 action 逐帧喂回 env.step。

运行（容器内，单终端）：
  /isaac-sim/python.sh examples/examples_teleop/example06_teleop_quest_replay.py \
    --dataset_file /tmp/franka_demos.hdf5

参数：
  --dataset_file      要回放的 HDF5 路径（example05 的产物）
  --select_episodes   只回放指定编号的 demo（如 0 1 2），默认回放全部
  --step_hz           回放步率（默认 30，数值越大回放越快）
"""

import argparse
import contextlib
import time

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import (
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--dataset_file", type=str, required=True, help="example05 录制的 HDF5 路径。")
    parser.add_argument(
        "--select_episodes", type=int, nargs="+", default=[], help="只回放这些编号的 demo（空=全部）。"
    )
    parser.add_argument("--step_hz", type=int, default=30, help="回放步率，数值越大回放越快。")
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """重建 example05 的场景，逐帧回放录制的 action。"""
    print("[REPLAY] Isaac Sim 已启动，开始重建 Arena 环境。", flush=True)

    import torch

    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    asset_registry = AssetRegistry()

    # 重建与 example05 完全一致的场景（回放不需要遥操作设备 + recorder）。
    table = asset_registry.get_asset_by_name("table")()
    robot = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=False)
    cube = asset_registry.get_asset_by_name("dex_cube")()
    light = asset_registry.get_asset_by_name("light")()

    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
    cube.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.04)))

    scene = Scene([table, light, cube])
    env_cfg = IsaacLabArenaEnvironment(
        name="teleop_table_cube_replay",
        embodiment=robot,
        scene=scene,
        task=NoTask(),
    )
    env_builder = ArenaEnvBuilder(env_cfg, builder_cfg)
    env = env_builder.make_registered()
    env.reset()
    print("[REPLAY] 环境创建完成。", flush=True)

    # 打开数据集。
    dataset = HDF5DatasetFileHandler()
    dataset.open(args_cli.dataset_file, "r")
    episode_names = list(dataset.get_episode_names())
    if args_cli.select_episodes:
        episode_names = [n for i, n in enumerate(episode_names) if i in args_cli.select_episodes]
    print(f"[REPLAY] 共 {len(episode_names)} 条 demo 待回放：{episode_names}", flush=True)

    sleep_dt = 1.0 / args_cli.step_hz
    try:
        with torch.inference_mode():
            for name in episode_names:
                episode = dataset.load_episode(name, device=str(env.unwrapped.device))
                initial_state = episode.get_initial_state()
                if initial_state is None:
                    print(f"[WARN] {name} 没有 initial_state，跳过。", flush=True)
                    continue

                # 回到录制的初始状态（相对坐标）。
                env.unwrapped.reset_to(
                    initial_state, torch.tensor([0], device=env.unwrapped.device), is_relative=True
                )

                print(f"[REPLAY] 正在回放 {name} ...", flush=True)
                step = 0
                while simulation_app.is_running():
                    action = episode.get_next_action()
                    if action is None:
                        break
                    env.step(action.unsqueeze(0))
                    step += 1
                    time.sleep(sleep_dt)
                print(f"[REPLAY] {name} 回放完成，共 {step} 步。", flush=True)
    finally:
        dataset.close()
        env.close()


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
