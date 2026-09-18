"""Level 1b（最简）SpaceMouse 遥操作：一张桌子 + 一个机械臂 + 一个方块。

用 3Dconnexion SpaceMouse 控制 Franka 末端，场景与 example01（键盘）完全一致，
只是把输入设备从键盘换成 SpaceMouse。**原生设备，单终端，不需要 CloudXR runtime**。

操作（Se3SpaceMouse）：
    6DoF 帽（推/拉/平移/旋转）  末端 delta pose
    左键                        夹爪开/合切换
    右键                        重置环境

前置：SpaceMouse 已插好且 hidraw 有读权限（见
    .talisman/dev_docs/2026-08/task05_3dxware遥操录制.md）。

运行（单终端）：
    /isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse.py
"""

import argparse
import contextlib
import time
import traceback

from isaaclab.app import AppLauncher
from isaaclab.devices.teleop_device_factory import create_teleop_device

from isaaclab_arena.cli.isaaclab_arena_cli import (
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，默认启用 Kit 可视化窗口。"""
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50000,
        help="最大仿真步数。",
    )
    parser.add_argument(
        "--keep_open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="演示结束后保持 Kit 窗口打开。",
    )
    parser.add_argument(
        "--pos_sensitivity",
        type=float,
        default=0.4,
        help="位置灵敏度，越大移动越快（IsaacLab 原生默认 0.4，Arena 默认 0.05 太小）。",
    )
    parser.add_argument(
        "--rot_sensitivity",
        type=float,
        default=0.8,
        help="旋转灵敏度，越大转得越快（IsaacLab 原生默认 0.8）。",
    )
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """构建最简场景（桌子 + 机械臂 + 方块）并用 SpaceMouse 遥操作。"""
    import torch

    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    # 资产：桌子 + 机械臂 + 方块 + 灯光（与 example01 一致）
    table = asset_registry.get_asset_by_name("table")()
    robot = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=False)
    cube = asset_registry.get_asset_by_name("dex_cube")()
    light = asset_registry.get_asset_by_name("light")()

    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
    cube.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.04)))

    # ★ SpaceMouse 遥操作设备（区别于 example01 的 keyboard）
    #    显式传灵敏度，覆盖 Arena 默认的 0.05/0.05（太小，移动缓慢）。
    teleop_device = device_registry.get_device_by_name("spacemouse")(
        sim_device=builder_cfg.device,
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=args_cli.rot_sensitivity,
    )

    scene = Scene([table, light, cube])
    env_cfg = IsaacLabArenaEnvironment(
        name="teleop_table_cube_spacemouse",
        embodiment=robot,
        scene=scene,
        task=NoTask(),
        teleop_device=teleop_device,
    )

    env_builder = ArenaEnvBuilder(env_cfg, builder_cfg)
    env = env_builder.make_registered()
    env.reset()
    simulation_app.update()

    try:
        should_reset = False

        def request_reset() -> None:
            nonlocal should_reset
            should_reset = True

        teleop_interface = create_teleop_device(
            "spacemouse",
            env.unwrapped.cfg.teleop_devices.devices,
            callbacks={"R": request_reset},
        )
        teleop_interface.reset()
        print(teleop_interface)
        print("SpaceMouse 遥操作已启动：6DoF 帽控制末端，左键开合夹爪，右键重置。")

        step = 0
        while simulation_app.is_running():
            if should_reset:
                should_reset = False
                try:
                    print("[INFO] 正在重置环境...")
                    with torch.inference_mode():
                        env.reset()
                    teleop_interface.reset()
                    simulation_app.update()
                    print("[INFO] 环境重置完成。")
                except Exception:
                    print("[ERROR] 环境重置失败：")
                    traceback.print_exc()
                continue

            try:
                with torch.inference_mode():
                    action = teleop_interface.advance().repeat(env.unwrapped.num_envs, 1)
                    env.step(action)
                simulation_app.update()
                step += 1
            except Exception:
                print("[ERROR] 遥操作循环执行失败：")
                traceback.print_exc()
                while simulation_app.is_running():
                    simulation_app.update()
                    time.sleep(0.01)
                break

            if not args_cli.keep_open and step >= args_cli.num_steps:
                break
            time.sleep(env.unwrapped.step_dt)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
