"""回放 SpaceMouse 遥操作录制的 Franka HDF5 示范。

回放只使用 episode 的初始状态和动作，不需要连接 SpaceMouse：

    /isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse_replay.py \
        --dataset_file /tmp/franka_spacemouse_demos.hdf5
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
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--dataset_file", type=str, required=True, help="SpaceMouse 录制的 HDF5 路径。")
    parser.add_argument(
        "--select_episodes",
        type=int,
        nargs="+",
        default=[],
        help="只回放指定编号的 demo，例如 0 1 2；默认回放全部。",
    )
    parser.add_argument("--step_hz", type=int, default=30, help="回放步率。")
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
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

    table = asset_registry.get_asset_by_name("table")()
    robot = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=False)
    cube = asset_registry.get_asset_by_name("dex_cube")()
    light = asset_registry.get_asset_by_name("light")()

    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
    cube.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.04)))

    env_cfg = IsaacLabArenaEnvironment(
        name="teleop_table_cube_spacemouse_replay",
        embodiment=robot,
        scene=Scene([table, light, cube]),
        task=NoTask(),
    )
    env = ArenaEnvBuilder(env_cfg, builder_cfg).make_registered()
    env.reset()

    dataset = HDF5DatasetFileHandler()
    dataset.open(args_cli.dataset_file, "r")
    episode_names = list(dataset.get_episode_names())
    if args_cli.select_episodes:
        episode_names = [name for index, name in enumerate(episode_names) if index in args_cli.select_episodes]
    print(f"[REPLAY] 将回放 {len(episode_names)} 条 demo：{episode_names}")

    sleep_dt = 1.0 / args_cli.step_hz
    try:
        with torch.inference_mode():
            for name in episode_names:
                episode = dataset.load_episode(name, device=str(env.unwrapped.device))
                initial_state = episode.get_initial_state()
                if initial_state is None:
                    print(f"[WARN] {name} 没有 initial_state，跳过。")
                    continue

                env.unwrapped.reset_to(
                    initial_state,
                    torch.tensor([0], device=env.unwrapped.device),
                    is_relative=True,
                )
                step = 0
                print(f"[REPLAY] 正在回放 {name} ...")
                while simulation_app.is_running():
                    action = episode.get_next_action()
                    if action is None:
                        break
                    env.step(action.unsqueeze(0))
                    step += 1
                    time.sleep(sleep_dt)
                print(f"[REPLAY] {name} 回放完成，共 {step} 步。")
    finally:
        dataset.close()
        env.close()


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
