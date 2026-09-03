"""Newton keyboard teleoperation demo for ABB IRB1200 with a simplified two-finger gripper.

This script tests the Newton-friendly gripper route:
    - keep the ABB IRB1200 articulation simple;
    - do not load the full Robotiq 2F-140 closed-chain/mimic articulation;
    - attach lightweight visual/collision fingers under link_6;
    - toggle the finger opening target with K and move the fingers gradually.

The fingers in this example are kinematic USD meshes. They are useful for checking size, placement,
and keyboard flow, but they are not driven by Newton joints and will not produce reliable grasping
contacts. Use example25 for a physical prismatic-joint gripper route.

Run:
    /isaac-sim/python.sh examples/examples_teleop/example24_teleop_keyboard_abb_irb1200_simple_gripper_newton.py
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
    parser.add_argument("--num_steps", type=int, default=50000, help="Maximum simulation steps.")
    parser.add_argument(
        "--keep_open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the Kit window open after the demo step limit.",
    )
    parser.add_argument("--pos_sensitivity", type=float, default=0.12, help="Keyboard translation sensitivity.")
    parser.add_argument("--rot_sensitivity", type=float, default=0.06, help="Keyboard rotation sensitivity.")
    parser.add_argument(
        "--joint_step_scale",
        type=float,
        default=0.1,
        help="Radians added to a driven ABB joint per keyboard tick.",
    )
    parser.add_argument(
        "--gripper_open_width",
        type=float,
        default=0.045,
        help="Simplified gripper half opening in meters.",
    )
    parser.add_argument(
        "--gripper_close_width",
        type=float,
        default=0.032,
        help="Simplified gripper half closing in meters.",
    )
    parser.add_argument(
        "--gripper_speed",
        type=float,
        default=0.004,
        help="Simplified gripper half-width change per simulation step in meters.",
    )
    parser.add_argument("--debug_joints", action="store_true", help="Print live joint positions.")
    parser.add_argument("--debug_actions", action="store_true", help="Print raw keyboard and processed joint actions.")
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False, presets="newton")
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


_ABB_ARM_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")


def _make_or_set_translate(xformable, xyz):
    from pxr import Gf, UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*xyz))
            return op
    return xformable.AddTranslateOp().Set(Gf.Vec3d(*xyz))


def _define_box(stage, path: str, translate, dims, color):
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    half_x = dims[0] * 0.5
    half_y = dims[1] * 0.5
    half_z = dims[2] * 0.5
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(-half_x, -half_y, -half_z),
                Gf.Vec3f(half_x, -half_y, -half_z),
                Gf.Vec3f(half_x, half_y, -half_z),
                Gf.Vec3f(-half_x, half_y, -half_z),
                Gf.Vec3f(-half_x, -half_y, half_z),
                Gf.Vec3f(half_x, -half_y, half_z),
                Gf.Vec3f(half_x, half_y, half_z),
                Gf.Vec3f(-half_x, half_y, half_z),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr(
        [
            0,
            1,
            2,
            3,
            4,
            7,
            6,
            5,
            0,
            4,
            5,
            1,
            1,
            5,
            6,
            2,
            2,
            6,
            7,
            3,
            3,
            7,
            4,
            0,
        ]
    )
    mesh.CreateExtentAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(-half_x, -half_y, -half_z),
                Gf.Vec3f(half_x, half_y, half_z),
            ]
        )
    )
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    _make_or_set_translate(mesh, translate)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def _create_simple_gripper(stage, parent_path: str):
    from pxr import Sdf, UsdGeom

    stage.SetEditTarget(stage.GetRootLayer())
    root_path = f"{parent_path}/simple_newton_gripper"
    root = UsdGeom.Xform.Define(stage, root_path)
    _make_or_set_translate(root, (0.055, 0.0, 0.0))

    _define_box(
        stage,
        f"{root_path}/palm",
        translate=(0.0, 0.0, 0.0),
        dims=(0.025, 0.04, 0.025),
        color=(0.18, 0.2, 0.22),
    )
    left = _define_box(
        stage,
        f"{root_path}/left_finger",
        translate=(0.065, args_cli.gripper_open_width, 0.0),
        dims=(0.08, 0.006, 0.014),
        color=(0.05, 0.05, 0.05),
    )
    right = _define_box(
        stage,
        f"{root_path}/right_finger",
        translate=(0.065, -args_cli.gripper_open_width, 0.0),
        dims=(0.08, 0.006, 0.014),
        color=(0.05, 0.05, 0.05),
    )
    assert stage.GetPrimAtPath(Sdf.Path(root_path)).IsValid(), f"Failed to create {root_path}"
    print("[CHECK] Simple gripper mesh dims: palm=(0.025,0.040,0.025), finger=(0.080,0.006,0.014) m")
    return left, right


def _set_simple_gripper_width(left_finger, right_finger, width: float) -> None:
    _make_or_set_translate(left_finger, (0.065, width, 0.0))
    _make_or_set_translate(right_finger, (0.065, -width, 0.0))


def _move_width_towards(current_width: float, target_width: float, max_step: float) -> float:
    if current_width < target_width:
        return min(current_width + max_step, target_width)
    if current_width > target_width:
        return max(current_width - max_step, target_width)
    return current_width


def main() -> None:
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    import isaaclab.sim as sim_utils
    from omni.usd import get_context

    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    robot = asset_registry.get_asset_by_name("abb_irb1200_ik")(enable_cameras=False)
    robot.scene_config.robot.spawn = robot.scene_config.robot.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    )
    robot.scene_config.robot.actuators["arm"] = ImplicitActuatorCfg(
        joint_names_expr=["joint_[1-6]"],
        effort_limit_sim=3000.0,
        velocity_limit_sim=4.0,
        stiffness=2500.0,
        damping=250.0,
        armature=1e-3,
    )
    robot.scene_config.robot.init_state = robot.scene_config.robot.init_state.replace(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": -0.35,
            "joint_3": 0.65,
            "joint_4": 0.0,
            "joint_5": 0.75,
            "joint_6": 0.0,
        }
    )

    table = asset_registry.get_asset_by_name("table")()
    cube = asset_registry.get_asset_by_name("dex_cube")()
    light = asset_registry.get_asset_by_name("light")()

    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    robot.set_initial_pose(Pose(position_xyz=(-0.35, 0.0, 0.0)))
    cube.set_initial_pose(Pose(position_xyz=(0.25, 0.0, 0.04)))

    teleop_device = device_registry.get_device_by_name("keyboard")(
        sim_device=builder_cfg.device,
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=args_cli.rot_sensitivity,
    )

    scene = Scene([table, light, cube])
    env_cfg = IsaacLabArenaEnvironment(
        name="teleop_abb_irb1200_keyboard_simple_gripper_newton",
        embodiment=robot,
        scene=scene,
        task=NoTask(),
        teleop_device=teleop_device,
    )

    print("[INFO] Building ABB IRB1200 Newton keyboard environment with simplified gripper...")
    env_builder = ArenaEnvBuilder(env_cfg, builder_cfg)
    manager_env_cfg, env_kwargs = env_builder.compose_manager_cfg()
    manager_env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    if manager_env_cfg.sim.physics is not None:
        manager_env_cfg.sim.physics.num_substeps = 6
        solver_cfg = getattr(manager_env_cfg.sim.physics, "solver_cfg", None)
        if solver_cfg is not None:
            solver_cfg.iterations = 150
            solver_cfg.ls_iterations = 25

    env = env_builder.make_registered(manager_env_cfg, env_kwargs)
    stage = get_context().get_stage()
    link6_path = "/World/envs/env_0/Robot/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6"
    left_finger, right_finger = _create_simple_gripper(stage, link6_path)
    print(f"[CHECK] Simple Newton gripper created under: {link6_path}")
    print("[INFO] Environment built. Resetting...")
    env.reset()
    simulation_app.update()

    robot_articulation = env.unwrapped.scene["robot"]
    joint_name_to_index = {name: idx for idx, name in enumerate(robot_articulation.data.joint_names)}
    arm_joint_ids = [joint_name_to_index[name] for name in _ABB_ARM_JOINTS]
    direct_joint_targets = robot_articulation.data.joint_pos.torch[:, arm_joint_ids].clone()
    print(f"[INFO] ABB IRB1200 joints: {robot_articulation.data.joint_names}")

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
        print(
            "ABB IRB1200 Newton simple-gripper teleop started: W/S A/D Q/E drive joints, "
            "Z/X T/G C/V rotate joints, K toggles simplified gripper, R resets."
        )

        step = 0
        gripper_closed = False
        gripper_width = args_cli.gripper_open_width
        gripper_target_width = args_cli.gripper_open_width
        while simulation_app.is_running():
            if should_reset:
                should_reset = False
                try:
                    print("[INFO] Resetting environment...")
                    with torch.inference_mode():
                        env.reset()
                    direct_joint_targets = robot_articulation.data.joint_pos.torch[:, arm_joint_ids].clone()
                    teleop_interface.reset()
                    gripper_closed = False
                    gripper_width = args_cli.gripper_open_width
                    gripper_target_width = args_cli.gripper_open_width
                    _set_simple_gripper_width(left_finger, right_finger, gripper_width)
                    simulation_app.update()
                    print("[INFO] Reset complete.")
                except Exception:
                    print("[ERROR] Environment reset failed:")
                    traceback.print_exc()
                continue

            try:
                with torch.inference_mode():
                    raw_action = teleop_interface.advance()
                    action = raw_action[:6].repeat(env.unwrapped.num_envs, 1)
                    joint_action = torch.zeros_like(action)
                    joint_action[:, 0] = action[:, 1]
                    joint_action[:, 1] = -action[:, 0]
                    joint_action[:, 2] = action[:, 2]
                    joint_action[:, 3] = action[:, 3]
                    joint_action[:, 4] = action[:, 4]
                    joint_action[:, 5] = action[:, 5]

                    env.step(torch.zeros(env.unwrapped.num_envs, 6, device=env.unwrapped.device))
                    direct_action = joint_action.to(device=direct_joint_targets.device)
                    direct_joint_targets = direct_joint_targets + torch.sign(direct_action) * args_cli.joint_step_scale
                    robot_articulation.write_joint_position_to_sim_index(
                        position=direct_joint_targets,
                        joint_ids=arm_joint_ids,
                    )
                    robot_articulation.write_joint_velocity_to_sim_index(
                        velocity=torch.zeros_like(direct_joint_targets),
                        joint_ids=arm_joint_ids,
                    )

                    gripper_closed_now = raw_action[6].item() < 0.0
                    if gripper_closed_now != gripper_closed:
                        gripper_closed = gripper_closed_now
                        gripper_target_width = args_cli.gripper_close_width if gripper_closed else args_cli.gripper_open_width
                        print(
                            f"[INFO] Simplified kinematic gripper target "
                            f"{'closed' if gripper_closed else 'opened'}: width={gripper_target_width:.3f} m"
                        )
                    next_width = _move_width_towards(gripper_width, gripper_target_width, args_cli.gripper_speed)
                    if next_width != gripper_width:
                        gripper_width = next_width
                        _set_simple_gripper_width(left_finger, right_finger, gripper_width)

                    if step % 120 == 0:
                        print(f"[DEBUG] Keyboard raw 7D action: {raw_action[:7].detach().cpu().tolist()}")
                    if args_cli.debug_actions and bool(torch.any(torch.abs(raw_action[:6]) > 1e-6).item()):
                        raw_debug = [round(float(v), 5) for v in raw_action[:7].detach().cpu().tolist()]
                        joint_debug = [round(float(v), 5) for v in joint_action[0].detach().cpu().tolist()]
                        print(f"[DEBUG] raw={raw_debug} joint_direct={joint_debug}")
                    if args_cli.debug_joints and step % 30 == 0:
                        joint_pos = robot_articulation.data.joint_pos.torch[0].detach().cpu()
                        joints = {
                            name: round(float(joint_pos[joint_name_to_index[name]]), 4)
                            for name in _ABB_ARM_JOINTS
                            if name in joint_name_to_index
                        }
                        print(f"[DEBUG] Joint positions rad: {joints}")

                simulation_app.update()
                step += 1
            except Exception:
                print("[ERROR] Keyboard teleoperation loop failed:")
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
        print("[ERROR] ABB IRB1200 Newton simple-gripper teleop startup failed:")
        traceback.print_exc()
    finally:
        simulation_app.close()
