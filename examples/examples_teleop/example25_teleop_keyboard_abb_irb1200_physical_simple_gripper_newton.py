"""Newton keyboard teleoperation for ABB IRB1200 with a physical simplified prismatic gripper.

Unlike example24, this script loads a USD that contains two prismatic finger joints and collision
fingers under link_6. The K key changes joint position targets instead of editing visual transforms.
"""

import argparse
import contextlib
from pathlib import Path
import time
import traceback

from isaaclab.app import AppLauncher
from isaaclab.devices.teleop_device_factory import create_teleop_device

from isaaclab_arena.cli.isaaclab_arena_cli import (
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)


def _find_simple_gripper_usd_path() -> str:
    relative_path = Path("isaaclab_arena/assets/robots/abb/irb1200_7_70_simple_gripper_newton/irb1200_7_70.usda")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"ABB IRB1200 Newton simple gripper USD not found: {relative_path}")


def parse_args() -> argparse.Namespace:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--num_steps", type=int, default=50000, help="Maximum simulation steps.")
    parser.add_argument("--keep_open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pos_sensitivity", type=float, default=0.12)
    parser.add_argument("--rot_sensitivity", type=float, default=0.06)
    parser.add_argument("--joint_step_scale", type=float, default=0.025)
    parser.add_argument(
        "--arm_control_mode",
        choices=("direct", "target"),
        default="direct",
        help="Use direct joint-state writes for responsive Newton testing, or actuator position targets.",
    )
    parser.add_argument("--gripper_open_width", type=float, default=0.045)
    parser.add_argument("--gripper_close_width", type=float, default=0.024)
    parser.add_argument("--gripper_speed", type=float, default=0.0005)
    parser.add_argument(
        "--world_gravity",
        type=float,
        default=-9.81,
        help="World gravity along z. The robot is gravity-disabled separately as a Newton hold workaround.",
    )
    parser.add_argument("--debug_joints", action="store_true")
    parser.add_argument("--debug_actions", action="store_true")
    parser.add_argument(
        "--grasp_assist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hold the cube at the gripper center after it is closed around the cube.",
    )
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False, presets="newton")
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


_ABB_ARM_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
_GRIPPER_JOINTS = ("left_finger_joint", "right_finger_joint")


def _apply_arm_command(robot_articulation, arm_joint_ids, joint_targets, control_mode: str) -> None:
    import torch

    if control_mode == "target":
        robot_articulation.set_joint_position_target_index(
            target=joint_targets,
            joint_ids=arm_joint_ids,
        )
    else:
        robot_articulation.write_joint_position_to_sim_index(
            position=joint_targets,
            joint_ids=arm_joint_ids,
        )
        robot_articulation.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(joint_targets),
            joint_ids=arm_joint_ids,
        )


def _manual_env_step(
    env,
    robot_articulation,
    arm_joint_ids,
    arm_joint_targets,
    arm_control_mode: str,
    gripper_joint_ids,
    gripper_targets,
) -> None:
    for _ in range(env.unwrapped.cfg.decimation):
        _apply_arm_command(robot_articulation, arm_joint_ids, arm_joint_targets, arm_control_mode)
        robot_articulation.set_joint_position_target_index(
            target=gripper_targets,
            joint_ids=gripper_joint_ids,
        )
        env.unwrapped.scene.write_data_to_sim()
        env.unwrapped.sim.step(render=False)
        env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
        _apply_arm_command(robot_articulation, arm_joint_ids, arm_joint_targets, arm_control_mode)
    if env.unwrapped.sim.is_rendering:
        env.unwrapped.sim.render(skip_app_pumping=False)


def _get_grasp_pose_w(robot_articulation, link_6_body_id: int, num_envs: int):
    import torch
    from isaaclab.utils.math import quat_apply

    link_pos_w = robot_articulation.data.body_pos_w.torch[:, link_6_body_id]
    link_quat_w = robot_articulation.data.body_quat_w.torch[:, link_6_body_id]
    local_offset = torch.tensor((0.12, 0.0, 0.0), device=link_pos_w.device, dtype=link_pos_w.dtype).repeat(num_envs, 1)
    return link_pos_w + quat_apply(link_quat_w, local_offset), link_quat_w


def _maybe_update_grasp_assist(
    env,
    robot_articulation,
    cube_asset,
    link_6_body_id: int,
    gripper_closed: bool,
    grasp_assist_active: bool,
) -> bool:
    import torch

    if not args_cli.grasp_assist:
        return False

    num_envs = env.unwrapped.num_envs
    cube_pose = cube_asset.data.root_pose_w.torch.clone()
    cube_pos = cube_pose[:, :3]
    grasp_pos, grasp_quat = _get_grasp_pose_w(robot_articulation, link_6_body_id, num_envs)
    distance = torch.linalg.norm(cube_pos - grasp_pos, dim=1)

    if gripper_closed and (grasp_assist_active or bool(torch.all(distance < 0.075).item())):
        cube_pose[:, :3] = grasp_pos
        cube_pose[:, 3:7] = grasp_quat
        cube_asset.write_root_pose_to_sim(cube_pose)
        cube_asset.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=cube_pose.device, dtype=cube_pose.dtype))
        return True
    return False


def _bind_high_friction_material(stage, sim_utils) -> None:
    material_path = "/World/Physics/HighFrictionGraspMaterial"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=8.0,
        dynamic_friction=6.0,
        restitution=0.0,
    )
    material_cfg.func(material_path, material_cfg)

    bound_paths = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()
        if prim_name in {"left_finger", "right_finger"} or "newton_grip_cube" in prim_path.lower():
            try:
                if sim_utils.bind_physics_material(prim_path, material_path, stage=stage):
                    bound_paths.append(prim_path)
            except ValueError:
                continue
    print(f"[CHECK] Bound high-friction grasp material to {len(bound_paths)} prims.")


def main() -> None:
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
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
    simple_gripper_usd_path = _find_simple_gripper_usd_path()
    robot.scene_config.robot.spawn = robot.scene_config.robot.spawn.replace(
        usd_path=simple_gripper_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    )
    robot.scene_config.robot.actuators["arm"] = ImplicitActuatorCfg(
        joint_names_expr=["joint_[1-6]"],
        effort_limit_sim=5000.0,
        velocity_limit_sim=3.0,
        stiffness=6000.0,
        damping=600.0,
        armature=2e-2,
    )
    robot.scene_config.robot.actuators["simple_gripper"] = ImplicitActuatorCfg(
        joint_names_expr=["left_finger_joint", "right_finger_joint"],
        effort_limit_sim=600.0,
        velocity_limit_sim=0.08,
        stiffness=2400.0,
        damping=260.0,
    )
    robot.scene_config.robot.init_state = robot.scene_config.robot.init_state.replace(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": -0.35,
            "joint_3": 0.65,
            "joint_4": 0.0,
            "joint_5": 0.75,
            "joint_6": 0.0,
            "left_finger_joint": args_cli.gripper_open_width,
            "right_finger_joint": -args_cli.gripper_open_width,
        }
    )

    table = asset_registry.get_asset_by_name("table")()
    cube = Object(
        name="newton_grip_cube",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.015),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=8.0,
                dynamic_friction=6.0,
                restitution=0.0,
            ),
        ),
    )
    cube.object_cfg.spawn = cube.object_cfg.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.015),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=8.0,
            dynamic_friction=6.0,
            restitution=0.0,
        ),
    )
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
        name="teleop_abb_irb1200_keyboard_physical_simple_gripper_newton",
        embodiment=robot,
        scene=scene,
        task=NoTask(),
        teleop_device=teleop_device,
    )

    print("[INFO] Building ABB IRB1200 Newton physical simple-gripper environment...")
    print(f"[INFO] USD: {simple_gripper_usd_path}")
    env_builder = ArenaEnvBuilder(env_cfg, builder_cfg)
    manager_env_cfg, env_kwargs = env_builder.compose_manager_cfg()
    manager_env_cfg.sim.gravity = (0.0, 0.0, args_cli.world_gravity)
    if manager_env_cfg.sim.physics is not None:
        manager_env_cfg.sim.physics.num_substeps = 6
        solver_cfg = getattr(manager_env_cfg.sim.physics, "solver_cfg", None)
        if solver_cfg is not None:
            solver_cfg.iterations = 150
            solver_cfg.ls_iterations = 25

    env = env_builder.make_registered(manager_env_cfg, env_kwargs)
    from omni.usd import get_context

    _bind_high_friction_material(get_context().get_stage(), sim_utils)
    print("[INFO] Environment built. Resetting...")
    env.reset()
    simulation_app.update()

    robot_articulation = env.unwrapped.scene["robot"]
    cube_asset = env.unwrapped.scene["newton_grip_cube"]
    joint_name_to_index = {name: idx for idx, name in enumerate(robot_articulation.data.joint_names)}
    body_name_to_index = {name: idx for idx, name in enumerate(robot_articulation.data.body_names)}
    arm_joint_ids = [joint_name_to_index[name] for name in _ABB_ARM_JOINTS]
    gripper_joint_ids = [joint_name_to_index[name] for name in _GRIPPER_JOINTS]
    link_6_body_id = body_name_to_index["link_6"]
    direct_joint_targets = robot_articulation.data.joint_pos.torch[:, arm_joint_ids].clone()
    print(f"[INFO] ABB IRB1200 joints: {robot_articulation.data.joint_names}")
    print(f"[INFO] Simple gripper joints: {_GRIPPER_JOINTS}")
    _apply_arm_command(robot_articulation, arm_joint_ids, direct_joint_targets, args_cli.arm_control_mode)
    robot_articulation.set_joint_position_target_index(
        target=torch.tensor(
            [[args_cli.gripper_open_width, -args_cli.gripper_open_width]],
            device=direct_joint_targets.device,
            dtype=direct_joint_targets.dtype,
        ).repeat(env.unwrapped.num_envs, 1),
        joint_ids=gripper_joint_ids,
    )
    env.unwrapped.scene.write_data_to_sim()

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
            "ABB IRB1200 Newton physical simple-gripper teleop started: W/S A/D Q/E drive joints, "
            "Z/X T/G C/V rotate joints, K toggles physical simplified gripper, R resets."
        )

        step = 0
        gripper_closed = False
        grasp_assist_active = False
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
                    grasp_assist_active = False
                    gripper_width = args_cli.gripper_open_width
                    gripper_target_width = args_cli.gripper_open_width
                    _apply_arm_command(robot_articulation, arm_joint_ids, direct_joint_targets, args_cli.arm_control_mode)
                    robot_articulation.set_joint_position_target_index(
                        target=torch.tensor(
                            [[gripper_width, -gripper_width]],
                            device=direct_joint_targets.device,
                            dtype=direct_joint_targets.dtype,
                        ).repeat(env.unwrapped.num_envs, 1),
                        joint_ids=gripper_joint_ids,
                    )
                    env.unwrapped.scene.write_data_to_sim()
                    simulation_app.update()
                    print("[INFO] Reset complete.")
                except Exception:
                    print("[ERROR] Environment reset failed:")
                    traceback.print_exc()
                continue

            try:
                simulation_app.update()
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

                    direct_action = joint_action.to(device=direct_joint_targets.device)
                    direct_joint_targets = direct_joint_targets + torch.sign(direct_action) * args_cli.joint_step_scale
                    _apply_arm_command(robot_articulation, arm_joint_ids, direct_joint_targets, args_cli.arm_control_mode)

                    gripper_closed_now = raw_action[6].item() < 0.0
                    if gripper_closed_now != gripper_closed:
                        gripper_closed = gripper_closed_now
                        if not gripper_closed:
                            grasp_assist_active = False
                        gripper_target_width = args_cli.gripper_close_width if gripper_closed else args_cli.gripper_open_width
                        print(
                            f"[INFO] Physical simple gripper target "
                            f"{'closed' if gripper_closed else 'opened'}: width={gripper_target_width:.3f} m"
                        )
                    if gripper_width < gripper_target_width:
                        gripper_width = min(gripper_width + args_cli.gripper_speed, gripper_target_width)
                    elif gripper_width > gripper_target_width:
                        gripper_width = max(gripper_width - args_cli.gripper_speed, gripper_target_width)
                    gripper_targets = torch.tensor(
                        [[gripper_width, -gripper_width]],
                        device=direct_joint_targets.device,
                        dtype=direct_joint_targets.dtype,
                    ).repeat(env.unwrapped.num_envs, 1)
                    robot_articulation.set_joint_position_target_index(
                        target=gripper_targets,
                        joint_ids=gripper_joint_ids,
                    )
                    grasp_assist_active = _maybe_update_grasp_assist(
                        env,
                        robot_articulation,
                        cube_asset,
                        link_6_body_id,
                        gripper_closed,
                        grasp_assist_active,
                    )
                    _manual_env_step(
                        env,
                        robot_articulation,
                        arm_joint_ids,
                        direct_joint_targets,
                        args_cli.arm_control_mode,
                        gripper_joint_ids,
                        gripper_targets,
                    )
                    _apply_arm_command(robot_articulation, arm_joint_ids, direct_joint_targets, args_cli.arm_control_mode)
                    robot_articulation.set_joint_position_target_index(
                        target=gripper_targets,
                        joint_ids=gripper_joint_ids,
                    )
                    grasp_assist_active = _maybe_update_grasp_assist(
                        env,
                        robot_articulation,
                        cube_asset,
                        link_6_body_id,
                        gripper_closed,
                        grasp_assist_active,
                    )

                    if step % 120 == 0:
                        print(f"[DEBUG] Keyboard raw 7D action: {raw_action[:7].detach().cpu().tolist()}")
                    if args_cli.debug_actions and bool(torch.any(torch.abs(raw_action[:6]) > 1e-6).item()):
                        raw_debug = [round(float(v), 5) for v in raw_action[:7].detach().cpu().tolist()]
                        joint_debug = [round(float(v), 5) for v in joint_action[0].detach().cpu().tolist()]
                        target_debug = [round(float(v), 4) for v in direct_joint_targets[0].detach().cpu().tolist()]
                        print(f"[DEBUG] raw={raw_debug} joint_direct={joint_debug} arm_target={target_debug}")
                    if args_cli.debug_joints and step % 30 == 0:
                        joint_pos = robot_articulation.data.joint_pos.torch[0].detach().cpu()
                        joints = {
                            name: round(float(joint_pos[joint_name_to_index[name]]), 4)
                            for name in (*_ABB_ARM_JOINTS, *_GRIPPER_JOINTS)
                            if name in joint_name_to_index
                        }
                        print(f"[DEBUG] Joint positions: {joints}")

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
        print("[ERROR] ABB IRB1200 Newton physical simple-gripper teleop startup failed:")
        traceback.print_exc()
    finally:
        simulation_app.close()
