"""SpaceMouse teleoperation for the tiny URDF arm.

Run:
    /isaac-sim/python.sh examples/examples_urdf_all_train/03_teleop_spacemouse_simple_arm.py
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
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--keep_open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pos_sensitivity", type=float, default=0.10)
    parser.add_argument("--rot_sensitivity", type=float, default=0.25)
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
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
        name="simple_urdf_arm_spacemouse",
        embodiment=robot,
        scene=Scene([light, cube]),
        task=NoTask(),
        teleop_device=teleop_device,
    )

    print("[INFO] Building simple URDF arm environment...")
    env = ArenaEnvBuilder(env_cfg, builder_cfg).make_registered()
    env.reset()
    simulation_app.update()
    with contextlib.suppress(Exception):
        robot_articulation = env.unwrapped.scene["robot"]
        print(f"[INFO] joints: {robot_articulation.data.joint_names}")
        print(f"[INFO] bodies: {robot_articulation.data.body_names}")

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
    print("[INFO] Simple arm teleop started. Cap controls 6D relative IK; right button resets.")

    try:
        step = 0
        while simulation_app.is_running():
            if should_reset:
                should_reset = False
                with torch.inference_mode():
                    env.reset()
                teleop_interface.reset()
                simulation_app.update()
                continue

            try:
                with torch.inference_mode():
                    raw_action = teleop_interface.advance()
                    action = raw_action[:6].repeat(env.unwrapped.num_envs, 1)
                    if step % 120 == 0:
                        print(f"[DEBUG] action: {raw_action[:6].detach().cpu().tolist()}")
                    env.step(action)
                simulation_app.update()
                step += 1
            except Exception:
                print("[ERROR] Teleoperation loop failed:")
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
    except Exception:
        print("[ERROR] Simple arm teleop startup failed:")
        traceback.print_exc()
    finally:
        simulation_app.close()
