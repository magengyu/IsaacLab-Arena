"""SpaceMouse 遥操作录制 + reset 时随机化方块初始位姿。

这个脚本基于 ``example01_teleop_spacemouse_record.py``，重点演示 Isaac Lab / Arena
里最常用的一类随机化：episode reset 时随机改变仿真状态。

运行：
    /isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse_record_randomized.py \
        --dataset_file /tmp/franka_spacemouse_randomized_demos.hdf5 \
        --num_demos 5 \
        --randomize_cube_pose

操作：6DoF 帽控制末端，左键切换夹爪，右键放弃当前 episode 并重置。
每次成功或手动重置后，方块会在指定范围内重新采样初始位置。
"""

import argparse
import contextlib
import os
import time
import traceback

from isaaclab.app import AppLauncher
from isaaclab.devices.teleop_device_factory import create_teleop_device

from isaaclab_arena.cli.isaaclab_arena_cli import (
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)

TARGET_X_RANGE = (0.4, 0.6)
TARGET_Y_RANGE = (-0.15, 0.15)
TARGET_Z_RANGE = (0.02, 0.3)


def parse_args() -> argparse.Namespace:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--num_steps", type=int, default=50000, help="最大仿真步数。")
    parser.add_argument(
        "--keep_open", action=argparse.BooleanOptionalAction, default=True, help="达到最大步数后是否保持窗口。"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="./datasets/franka_spacemouse_randomized_demos.hdf5",
        help="输出 HDF5 路径。",
    )
    parser.add_argument("--num_demos", type=int, default=0, help="成功 demo 数，0 表示持续录制。")
    parser.add_argument("--num_success_steps", type=int, default=2, help="连续成功多少步后导出 demo。")
    parser.add_argument("--pos_sensitivity", type=float, default=0.4, help="SpaceMouse 平移灵敏度。")
    parser.add_argument("--rot_sensitivity", type=float, default=0.8, help="SpaceMouse 旋转灵敏度。")
    parser.add_argument(
        "--randomize_cube_pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在每次 reset 时随机化方块初始位置和 yaw。",
    )
    parser.add_argument("--cube_x_min", type=float, default=0.22, help="方块 reset 随机 x 下界。")
    parser.add_argument("--cube_x_max", type=float, default=0.42, help="方块 reset 随机 x 上界。")
    parser.add_argument("--cube_y_min", type=float, default=-0.18, help="方块 reset 随机 y 下界。")
    parser.add_argument("--cube_y_max", type=float, default=0.18, help="方块 reset 随机 y 上界。")
    parser.add_argument("--cube_z", type=float, default=0.04, help="方块 reset 高度，桌面小方块通常保持固定。")
    parser.add_argument("--cube_yaw_min", type=float, default=-3.14159, help="方块 reset yaw 下界，单位 rad。")
    parser.add_argument("--cube_yaw_max", type=float, default=3.14159, help="方块 reset yaw 上界，单位 rad。")
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import torch

    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode

    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
    from isaaclab_arena.utils.pose import Pose, PoseRange

    output_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    success_term_holder: dict = {}

    def env_cfg_callback(env_cfg):
        # 手动处理成功与重置，才能在 reset 前明确标记并导出当前 episode。
        success_term_holder["term"] = env_cfg.terminations.success
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        return env_cfg

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    table = asset_registry.get_asset_by_name("table")()
    robot = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=False)
    cube = asset_registry.get_asset_by_name("dex_cube")()
    light = asset_registry.get_asset_by_name("light")()

    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))

    if args_cli.randomize_cube_pose:
        cube_pose_range = PoseRange(
            position_xyz_min=(args_cli.cube_x_min, args_cli.cube_y_min, args_cli.cube_z),
            position_xyz_max=(args_cli.cube_x_max, args_cli.cube_y_max, args_cli.cube_z),
            rpy_min=(0.0, 0.0, args_cli.cube_yaw_min),
            rpy_max=(0.0, 0.0, args_cli.cube_yaw_max),
        )
        cube.set_initial_pose(cube_pose_range)
    else:
        cube.set_initial_pose(Pose(position_xyz=(0.3, 0.0, args_cli.cube_z)))

    teleop_device = device_registry.get_device_by_name("spacemouse")(
        sim_device=builder_cfg.device,
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=args_cli.rot_sensitivity,
    )
    scene = Scene([table, light, cube])
    task = GoalPoseTask(
        cube,
        target_x_range=TARGET_X_RANGE,
        target_y_range=TARGET_Y_RANGE,
        target_z_range=TARGET_Z_RANGE,
    )
    env_cfg = IsaacLabArenaEnvironment(
        name="teleop_table_cube_spacemouse_record_randomized",
        embodiment=robot,
        scene=scene,
        task=task,
        teleop_device=teleop_device,
        env_cfg_callback=env_cfg_callback,
    )
    env = ArenaEnvBuilder(env_cfg, builder_cfg).make_registered()
    env.reset()
    simulation_app.update()

    from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
    from isaaclab.sim.spawners.shapes import spawn_cuboid
    from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg

    target_center = tuple((low + high) / 2 for low, high in (TARGET_X_RANGE, TARGET_Y_RANGE, TARGET_Z_RANGE))
    target_size = tuple(high - low for low, high in (TARGET_X_RANGE, TARGET_Y_RANGE, TARGET_Z_RANGE))
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

    should_reset = False
    success_step_count = 0

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
    print(f"SpaceMouse 随机化录制已启动，数据输出到 {args_cli.dataset_file}")
    print(f"目标区：中心={target_center}，尺寸={target_size}")
    if args_cli.randomize_cube_pose:
        print(
            "[INFO] 方块 reset 随机范围："
            f"x=({args_cli.cube_x_min}, {args_cli.cube_x_max}), "
            f"y=({args_cli.cube_y_min}, {args_cli.cube_y_max}), "
            f"z={args_cli.cube_z}, yaw=({args_cli.cube_yaw_min}, {args_cli.cube_yaw_max})"
        )

    def reset_env_and_report(reason: str) -> None:
        nonlocal success_step_count
        print(f"[INFO] {reason}，正在 reset 环境并重新采样随机项...")
        env.unwrapped.sim.reset()
        env.unwrapped.recorder_manager.reset()
        with torch.inference_mode():
            env.reset()
        teleop_interface.reset()
        success_step_count = 0
        simulation_app.update()
        if args_cli.randomize_cube_pose:
            cube_pose = cube.get_object_pose(env)
            print(f"[INFO] 本轮方块初始 pose xyz={cube_pose[0, :3].detach().cpu().tolist()}")

    try:
        step = 0
        while simulation_app.is_running():
            if should_reset:
                should_reset = False
                try:
                    reset_env_and_report("放弃当前 episode")
                except Exception:
                    print("[ERROR] 环境重置失败：")
                    traceback.print_exc()
                continue

            try:
                with torch.inference_mode():
                    action = teleop_interface.advance().repeat(env.unwrapped.num_envs, 1)
                    env.step(action)
                    step += 1

                    success_term = success_term_holder["term"]
                    if bool(success_term.func(env.unwrapped, **success_term.params)[0]):
                        success_step_count += 1
                    else:
                        success_step_count = 0

                    if success_step_count >= args_cli.num_success_steps:
                        recorder = env.unwrapped.recorder_manager
                        recorder.record_pre_reset([0], force_export_or_skip=False)
                        recorder.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.unwrapped.device)
                        )
                        recorder.export_episodes([0])
                        count = recorder.exported_successful_episode_count
                        print(f"[INFO] 已录制第 {count} 条成功 demo。")
                        if args_cli.num_demos > 0 and count >= args_cli.num_demos:
                            break
                        reset_env_and_report("成功 demo 已导出")

                simulation_app.update()
            except Exception:
                print("[ERROR] 遥操作录制循环失败：")
                traceback.print_exc()
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
