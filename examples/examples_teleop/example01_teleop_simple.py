"""Level 1（最简）键盘遥操作：一张桌子 + 一个机械臂 + 一个方块。

只验证遥操作链路（键盘 → 末端增量 → IK → Franka），不含任何任务/成功判定。
场景最简：table + franka_ik + procedural_cube，启动快（无 kitchen，无需等 AWS 大场景下载）。

键位（Se3Keyboard）：
    W/S、A/D、Q/E   末端平移 (x/y/z)
    Z/X、T/G、C/V   末端旋转 (roll/pitch/yaw)
    K               夹爪开/合切换
    R               重置环境

运行（交互式，需要显示器 + 键盘焦点）：
    /isaac-sim/python.sh examples/examples_teleop/example01_teleop_simple.py

headless 冒烟（无显示器，仅验证环境能起）：
    /isaac-sim/python.sh examples/examples_teleop/example01_teleop_simple.py \
        --headless --num_steps 50 --no-keep_open
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
        default=500,
        help="最大仿真步数。",
    )
    parser.add_argument(
        "--keep_open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="演示结束后保持 Kit 窗口打开。",
    )
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """构建最简场景（桌子 + 机械臂 + 方块）并在 Kit 窗口中运行键盘遥操作。"""
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

    # 资产：桌子 + 机械臂 + 方块 + 灯光
    table = asset_registry.get_asset_by_name("table")()
    robot = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=False)
    cube = asset_registry.get_asset_by_name("dex_cube")()
    light = asset_registry.get_asset_by_name("light")()

    # 桌子固定在原点，机械臂在桌子左侧，方块放在桌面上
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
    cube.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.04)))

    # 键盘遥操作设备
    teleop_device = device_registry.get_device_by_name("keyboard")(sim_device=builder_cfg.device)

    # 组装场景（无 Task，只遥操作）
    scene = Scene([table, light, cube])
    env_cfg = IsaacLabArenaEnvironment(
        name="teleop_table_cube",
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

        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("omni.appwindow", True)
        import carb.input  # noqa: F401
        import omni.appwindow  # noqa: F401

        teleop_interface = create_teleop_device(
            "keyboard",
            env.unwrapped.cfg.teleop_devices.devices,
            callbacks={"R": request_reset},
        )
        teleop_interface.reset()
        print(teleop_interface)
        print("遥操作已启动：W/S、A/D、Q/E 平移，Z/X、T/G、C/V 旋转，K 开合夹爪，R 重置。")

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
    except Exception:
        print("[ERROR] example01 teleop startup failed:")
        traceback.print_exc()
    finally:
        simulation_app.close()
