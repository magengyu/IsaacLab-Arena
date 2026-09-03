"""SpaceMouse 遥操作 + GoalPoseTask 成功判定 + HDF5 示范录制。

运行：
    /isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse_record.py \
        --dataset_file /tmp/franka_spacemouse_demos.hdf5 --num_demos 5

操作：6DoF 帽控制末端，左键切换夹爪，右键放弃当前 episode 并重置。
把方块稳定放入绿色目标区后，脚本导出一条成功 demo 并自动开始下一条。
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
        "--dataset_file", type=str, default="./datasets/franka_spacemouse_demos.hdf5", help="输出 HDF5 路径。"
    )
    parser.add_argument("--num_demos", type=int, default=0, help="成功 demo 数，0 表示持续录制。")
    parser.add_argument("--num_success_steps", type=int, default=2, help="连续成功多少步后导出 demo。")
    parser.add_argument("--pos_sensitivity", type=float, default=0.4, help="SpaceMouse 平移灵敏度。")
    parser.add_argument("--rot_sensitivity", type=float, default=0.8, help="SpaceMouse 旋转灵敏度。")
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
    from isaaclab_arena.utils.pose import Pose

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
    cube.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.04)))

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
        name="teleop_table_cube_spacemouse_record",
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
    print(f"SpaceMouse 录制已启动，数据输出到 {args_cli.dataset_file}")
    print(f"请把方块放入绿色目标区：中心={target_center}，尺寸={target_size}")

    try:
        step = 0
        while simulation_app.is_running():
            if should_reset:
                should_reset = False
                try:
                    print("[INFO] 放弃当前 episode，正在重置环境...")
                    env.unwrapped.sim.reset()
                    env.unwrapped.recorder_manager.reset()
                    with torch.inference_mode():
                        env.reset()
                    teleop_interface.reset()
                    success_step_count = 0
                    simulation_app.update()
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
                        env.unwrapped.sim.reset()
                        recorder.reset()
                        env.reset()
                        teleop_interface.reset()
                        success_step_count = 0

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
