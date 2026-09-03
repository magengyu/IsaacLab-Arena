"""Level 5b：Quest XR 遥操作 + 录制示范（带 START/STOP 控制）。

在 example05 基础上，保留控制通道（control_channel_uuid 用默认值），支持 Quest 端
START / STOP / RESET 交互，行为对齐官方 record_demos.py：

  - START：开始录制（机械臂响应 + 逐帧记录 (obs, action)）
  - STOP：暂停录制（机械臂不响应）
  - 成功：连续 num_success_steps 步成功 → 自动导出该 demo 并提示
  - RESET：重置这一条（不计成功）

与 example05 的区别：example05 显式关闭控制通道（control_channel_uuid = None）、改成
「成功自动导出」的简化逻辑；本脚本保留控制通道，Quest 端有开始/停止/重置按钮。

运行步骤同 example03（两个终端）：
  1. 终端 1：/isaac-sim/python.sh -m isaacteleop.cloudxr --host-client
  2. 终端 2：source ~/.cloudxr/run/cloudxr.env
             /isaac-sim/python.sh examples/examples_teleop/example05_teleop_quest_record_startstop.py --xr
  3. Quest 浏览器 Connect → XR 标签页 Start Session → Play
  4. Quest 端 START 开始录制 → 操作 → 成功后自动保存 → RESET 录下一条

参数：
  --dataset_file       录制输出 HDF5（默认 ./datasets/franka_demos.hdf5）
  --num_demos          要录制的成功 demo 数（默认 0 = 无限）
  --num_success_steps  连续多少步成功判定 demo 完成（默认 2）
"""

import argparse
import contextlib
import os
import traceback

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import (
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)

# ★ 目标区：方块放进这个区域（x/y/z 都在区间内）就算成功。按你的桌高/坐姿实测微调。
TARGET_X_RANGE = (0.4, 0.6)
TARGET_Y_RANGE = (-0.15, 0.15)
TARGET_Z_RANGE = (0.02, 0.3)


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
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="./datasets/franka_demos.hdf5",
        help="录制输出 HDF5 路径。",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=0,
        help="要录制的成功 demo 数，0 表示无限。",
    )
    parser.add_argument(
        "--num_success_steps",
        type=int,
        default=2,
        help="连续多少步成功判定 demo 完成。",
    )
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False)
    args = parser.parse_args()
    assert args.xr, "请添加 --xr 以启用 Quest OpenXR 遥操作"
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """构建 Franka 目标位姿场景，Quest 遥操作并录制（带 START/STOP 控制）。"""
    print("[XR] Isaac Sim 已启动，开始构建 Arena 环境。", flush=True)

    import torch

    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    from isaaclab_teleop import create_isaac_teleop_device, poll_control_events
    from isaaclab_teleop.xr_cfg import XrAnchorRotationMode

    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
    from isaaclab_arena.utils.pose import Pose

    # 解析输出目录 / 文件名（去掉 .hdf5 后缀），recorder 用这两个字段落盘。
    output_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    # 闭包暂存 success 终止项，供主循环手动判定。
    success_term_holder: dict = {}

    def env_cfg_callback(env_cfg):
        """在 env 构建前注入 recorder 配置，并禁用 success 自动 reset。"""
        # 暂存 success 项，置 None 避免 env 成功后自动 reset（改由脚本手动判定）。
        success_term_holder["term"] = env_cfg.terminations.success
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None
        # 注入 action/state/obs recorder，只在成功时导出。
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        return env_cfg

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    # 资产：桌子 + 机械臂 + 方块 + 灯光（与 example03 一致）。
    table = asset_registry.get_asset_by_name("table")()
    robot = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=False)
    cube = asset_registry.get_asset_by_name("dex_cube")()
    light = asset_registry.get_asset_by_name("light")()

    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
    cube.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.04)))

    teleop_device = device_registry.get_device_by_name("openxr")(sim_device=builder_cfg.device)

    scene = Scene([table, light, cube])
    # ★ 把方块放进目标区算成功（区别于 example03 的 NoTask）。
    task = GoalPoseTask(
        cube,
        target_x_range=TARGET_X_RANGE,
        target_y_range=TARGET_Y_RANGE,
        target_z_range=TARGET_Z_RANGE,
    )
    env_cfg = IsaacLabArenaEnvironment(
        name="teleop_table_cube_record_startstop",
        embodiment=robot,
        scene=scene,
        task=task,
        teleop_device=teleop_device,
        env_cfg_callback=env_cfg_callback,
    )

    print("[XR] 正在创建 Franka 录制环境。", flush=True)
    env_builder = ArenaEnvBuilder(env_cfg, builder_cfg)
    env = env_builder.make_registered()
    print("[XR] 环境创建完成，正在 reset。", flush=True)
    env.reset()
    print(f"[XR] 环境 reset 完成，示范将录制到：{args_cli.dataset_file}", flush=True)

    # ★ 可视化目标区：spawn 一个绿色半透明盒子，标出 TARGET_X/Y/Z_RANGE 覆盖的区域。
    from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
    from isaaclab.sim.spawners.shapes import spawn_cuboid
    from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg

    target_center = (
        (TARGET_X_RANGE[0] + TARGET_X_RANGE[1]) / 2,
        (TARGET_Y_RANGE[0] + TARGET_Y_RANGE[1]) / 2,
        (TARGET_Z_RANGE[0] + TARGET_Z_RANGE[1]) / 2,
    )
    target_size = (
        TARGET_X_RANGE[1] - TARGET_X_RANGE[0],
        TARGET_Y_RANGE[1] - TARGET_Y_RANGE[0],
        TARGET_Z_RANGE[1] - TARGET_Z_RANGE[0],
    )
    spawn_cuboid(
        "/World/envs/env_0/TargetZone",
        CuboidCfg(
            size=target_size,
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.9, 0.1),
                emissive_color=(0.0, 0.3, 0.0),
                opacity=0.2,
                roughness=0.4,
                metallic=0.0,
            ),
        ),
        translation=target_center,
    )
    print(
        f"[XR] 已 spawn 目标区可视化盒子（绿色半透明）：中心 {target_center}，尺寸 {target_size}。",
        flush=True,
    )

    try:
        should_reset = False
        success_step_count = 0
        # XR 下默认「暂停」，等 Quest 端发 START 才开始录制（与官方 record_demos.py 一致）。
        running_recording_instance = False

        # ★ 注意：不关闭 control_channel_uuid（保持默认 TELEOP_CONTROL_CHANNEL_UUID），
        #    这样 poll_control_events 才能收到 Quest 端的 START/STOP/RESET。

        # ★ 方式 B：锚点修正（同 example03）。
        env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_pos = (0.0, 0.0, -0.75)
        env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_rot = (0.0, 0.0, -0.70711, 0.70711)
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
                "XR 遥操作录制已启动：Quest 端 START 开始录制，STOP 暂停，把方块放进绿色目标区成功后自动保存。",
                flush=True,
            )

            step = 0
            while simulation_app.is_running():
                if should_reset:
                    should_reset = False
                    try:
                        print("[INFO] 重置环境（不计成功）...")
                        env.unwrapped.sim.reset()
                        env.unwrapped.recorder_manager.reset()
                        with torch.inference_mode():
                            env.reset()
                        teleop_interface.reset()
                        success_step_count = 0
                        print("[INFO] 环境重置完成。")
                    except Exception:
                        print("[ERROR] 环境重置失败：")
                        traceback.print_exc()
                    continue

                try:
                    with torch.inference_mode():
                        action = teleop_interface.advance()

                        # ★ 轮询控制事件：START/STOP 切换录制状态，RESET 触发重置。
                        ctrl = poll_control_events(teleop_interface)
                        if ctrl.is_active is not None and ctrl.is_active != running_recording_instance:
                            running_recording_instance = ctrl.is_active
                            print(
                                "[INFO] 录制已开始。" if running_recording_instance else "[INFO] 录制已暂停。",
                                flush=True,
                            )
                        if ctrl.should_reset:
                            should_reset = True

                        if action is None:
                            # 等待 WebXR Session 提供控制器数据。
                            env.unwrapped.sim.render()
                            continue

                        if not running_recording_instance:
                            # 暂停状态：机械臂不响应，只渲染。
                            env.unwrapped.sim.render()
                            continue

                        if step == 0:
                            print("[INFO] 已收到 WebXR 控制器数据，开始录制。", flush=True)
                        env.step(action.repeat(env.unwrapped.num_envs, 1))
                        step += 1

                        # ★ 成功判定：连续 num_success_steps 步命中 → 导出 demo 并 reset。
                        success_term = success_term_holder["term"]
                        if bool(success_term.func(env.unwrapped, **success_term.params)[0]):
                            success_step_count += 1
                            if success_step_count >= args_cli.num_success_steps:
                                recorder = env.unwrapped.recorder_manager
                                recorder.record_pre_reset([0], force_export_or_skip=False)
                                recorder.set_success_to_episodes(
                                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.unwrapped.device)
                                )
                                recorder.export_episodes([0])
                                print(
                                    f"[INFO] ✅ 已录制第 {recorder.exported_successful_episode_count} 条成功 demo。",
                                    flush=True,
                                )
                                if (
                                    args_cli.num_demos > 0
                                    and recorder.exported_successful_episode_count >= args_cli.num_demos
                                ):
                                    print("[INFO] 已录满目标 demo 数，退出。", flush=True)
                                    break
                                # reset，准备下一条 demo。
                                env.unwrapped.sim.reset()
                                recorder.reset()
                                with torch.inference_mode():
                                    env.reset()
                                teleop_interface.reset()
                                success_step_count = 0
                        else:
                            success_step_count = 0
                except Exception:
                    print("[ERROR] 录制循环执行失败：")
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
