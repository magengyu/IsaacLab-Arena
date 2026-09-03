"""Level 3：Quest XR 遥操作（带锚点修正）— 一张桌子 + 一个机械臂 + 一个方块。

在 example02 基础上，按「方式 B」给 XrCfg 锚点补上站位和朝向，修正两个问题：
  1. 桌子贴地          → anchor_pos 的 Z 调高度（眼睛在桌面上方）
  2. 机械臂朝向不匹配  → anchor_rot 绕 Z 转 -90°（把视线对齐到机器人前伸 +X）
  3. anchor_prim_path 锚到 panda_link0（动态跟随机器人）

运行步骤同 example02（两个终端）：
  1. 终端 1：/isaac-sim/python.sh -m isaacteleop.cloudxr --host-client
  2. 终端 2：source ~/.cloudxr/run/cloudxr.env
             /isaac-sim/python.sh examples/examples_teleop/example03_teleop_quest_anchor.py --xr
  3. Quest 浏览器 Connect → XR 标签页 Start Session → Play

注意：anchor_pos 的 Z（-0.9）和 anchor_rot 的 yaw（-0.70711）需按你的桌高/坐姿实测微调，
      yaw 方向反了改成 +0.70711（差 180°）。
"""

import argparse
import contextlib
import time
import traceback

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import (
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，要求通过 ``--xr`` 启用 OpenXR。"""
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
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False)
    args = parser.parse_args()
    assert args.xr, "请添加 --xr 以启用 Quest OpenXR 遥操作"
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """构建最简场景（桌子 + 机械臂 + 方块）并通过 Quest 右手柄遥操作（带锚点修正）。"""
    print("[XR] Isaac Sim 已启动，开始构建 Arena 环境。", flush=True)

    import torch

    from isaaclab_teleop import create_isaac_teleop_device
    from isaaclab_teleop.xr_cfg import XrAnchorRotationMode

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
    # XR 遥操作使用头显视图，不需要额外初始化 Franka wrist camera
    table = asset_registry.get_asset_by_name("table")()
    robot = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=False)
    cube = asset_registry.get_asset_by_name("dex_cube")()
    light = asset_registry.get_asset_by_name("light")()

    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
    cube.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.04)))

    # Quest 右手柄位姿控制末端，Trigger 控制夹爪
    teleop_device = device_registry.get_device_by_name("openxr")(sim_device=builder_cfg.device)

    scene = Scene([table, light, cube])
    env_cfg = IsaacLabArenaEnvironment(
        name="teleop_table_cube_xr",
        embodiment=robot,
        scene=scene,
        task=NoTask(),
        teleop_device=teleop_device,
    )

    print("[XR] 正在创建 Franka 环境。", flush=True)
    env_builder = ArenaEnvBuilder(env_cfg, builder_cfg)
    env = env_builder.make_registered()
    print("[XR] 环境创建完成，正在 reset。", flush=True)
    env.reset()
    print("[XR] 环境 reset 完成，正在创建 IsaacTeleop 设备。", flush=True)

    try:
        should_reset = False

        # 该独立示例没有实现远程 START/STOP 控制。关闭 control channel 后，
        # controller pipeline 会在 Session 建立后直接逐帧运行。
        env.unwrapped.cfg.isaac_teleop.control_channel_uuid = None

        # ★ 方式 B：补上 XrCfg 锚点（站位 + 朝向），修正「桌子贴地 + 朝向不匹配」。
        #    数值按你的桌高/坐姿实测微调。
        env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_pos = (0.0, 0.0, -0.75)              # 高度：眼睛在桌面上方
        env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_rot = (0.0, 0.0, -0.70711, 0.70711)  # 绕 Z -90° 对齐朝向
        env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_prim_path = "/World/envs/env_0/Robot/panda_link0"
        env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
        env.unwrapped.cfg.isaac_teleop.xr_cfg.fixed_anchor_height = True

        def request_reset() -> None:
            nonlocal should_reset
            should_reset = True

        teleop_interface = create_isaac_teleop_device(
            env.unwrapped.cfg.isaac_teleop,
            sim_device=str(env.unwrapped.device),
            callbacks={"R": request_reset},
        )
        print("[XR] IsaacTeleop 设备创建完成，正在启动 Teleop Session。", flush=True)
        with teleop_interface:
            print("[XR] Teleop Session 已启动。", flush=True)
            teleop_interface.reset()
            print(teleop_interface)
            print(
                "XR 遥操作已启动：移动 Quest 右手柄控制末端，按下 Trigger 闭合夹爪，松开 Trigger 打开。",
                flush=True,
            )

            step = 0
            while simulation_app.is_running():
                if should_reset:
                    should_reset = False
                    try:
                        print("[INFO] 正在重置环境...")
                        with torch.inference_mode():
                            env.reset()
                        teleop_interface.reset()
                        print("[INFO] 环境重置完成。")
                    except Exception:
                        print("[ERROR] 环境重置失败：")
                        traceback.print_exc()
                    continue

                try:
                    with torch.inference_mode():
                        action = teleop_interface.advance()
                        if action is None:
                            # 保持 Kit 响应，同时等待 WebXR Session 提供控制器数据。
                            env.unwrapped.sim.render()
                        else:
                            if step == 0:
                                print("[INFO] 已收到 WebXR 控制器数据，开始执行遥操作。", flush=True)
                            env.step(action.repeat(env.unwrapped.num_envs, 1))
                            step += 1
                except Exception:
                    print("[ERROR] 遥操作循环执行失败：")
                    traceback.print_exc()
                    break

                if not args_cli.keep_open and step >= args_cli.num_steps:
                    break
    finally:
        env.close()


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
