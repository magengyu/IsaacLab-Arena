"""Record fixed-length SpaceMouse demos for the tiny URDF arm.

Run:
    /isaac-sim/python.sh examples/examples_urdf_all_train/04_record_spacemouse_simple_arm.py \
      --dataset_file /tmp/simple_arm_demos.hdf5 --num_demos 3 --steps_per_demo 200
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


def parse_args() -> argparse.Namespace:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--dataset_file", type=str, default="/tmp/simple_arm_demos.hdf5")
    parser.add_argument("--num_demos", type=int, default=3)
    parser.add_argument("--steps_per_demo", type=int, default=200)
    parser.add_argument("--pos_sensitivity", type=float, default=0.10)
    parser.add_argument("--rot_sensitivity", type=float, default=0.25)
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
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    output_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    def env_cfg_callback(env_cfg):
        env_cfg.terminations.time_out = None
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        return env_cfg

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    robot = asset_registry.get_asset_by_name("simple_urdf_arm_ik")(enable_cameras=False)
    light = asset_registry.get_asset_by_name("light")()
    cube = asset_registry.get_asset_by_name("dex_cube")()

    robot.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    cube.set_initial_pose(Pose(position_xyz=(0.45, 0.0, 0.04)))

    teleop_device = device_registry.get_device_by_name("spacemouse")(
        sim_device=builder_cfg.device,
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=args_cli.rot_sensitivity,
    )

    env_cfg = IsaacLabArenaEnvironment(
        name="simple_urdf_arm_spacemouse_record",
        embodiment=robot,
        scene=Scene([light, cube]),
        task=NoTask(),
        teleop_device=teleop_device,
        env_cfg_callback=env_cfg_callback,
    )

    env = ArenaEnvBuilder(env_cfg, builder_cfg).make_registered()
    env.reset()
    simulation_app.update()

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
    print(f"[INFO] Recording to {args_cli.dataset_file}")
    print(f"[INFO] Each demo exports after {args_cli.steps_per_demo} steps. Right button abandons current demo.")

    demo_step = 0
    try:
        while simulation_app.is_running():
            if should_reset:
                should_reset = False
                env.unwrapped.sim.reset()
                env.unwrapped.recorder_manager.reset()
                with torch.inference_mode():
                    env.reset()
                teleop_interface.reset()
                demo_step = 0
                simulation_app.update()
                continue

            try:
                with torch.inference_mode():
                    action = teleop_interface.advance()[:6].repeat(env.unwrapped.num_envs, 1)
                    env.step(action)
                    demo_step += 1

                    if demo_step >= args_cli.steps_per_demo:
                        recorder = env.unwrapped.recorder_manager
                        recorder.record_pre_reset([0], force_export_or_skip=False)
                        recorder.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.unwrapped.device)
                        )
                        recorder.export_episodes([0])
                        print(f"[INFO] Exported demo {recorder.exported_successful_episode_count}")
                        if recorder.exported_successful_episode_count >= args_cli.num_demos:
                            break
                        env.unwrapped.sim.reset()
                        recorder.reset()
                        env.reset()
                        teleop_interface.reset()
                        demo_step = 0

                simulation_app.update()
            except Exception:
                print("[ERROR] Recording loop failed:")
                traceback.print_exc()
                break

            time.sleep(env.unwrapped.step_dt)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    except Exception:
        print("[ERROR] Simple arm recording startup failed:")
        traceback.print_exc()
    finally:
        simulation_app.close()
