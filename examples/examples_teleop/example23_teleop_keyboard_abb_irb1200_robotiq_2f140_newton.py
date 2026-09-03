"""Newton keyboard teleoperation demo for ABB IRB1200-7/0.70 with a stabilized Robotiq 2F-140.

Scene: a table, an ABB IRB1200, a cube, and a dome light.

Run:
    /isaac-sim/python.sh examples/examples_teleop/example23_teleop_keyboard_abb_irb1200_robotiq_2f140_newton.py

Controls follow Isaac Lab's Se3Keyboard convention:
    W/S, A/D, Q/E   translate end-effector along x/y/z
    Z/X, T/G, C/V   rotate end-effector around roll/pitch/yaw
    K               toggle gripper open/close with Newton direct joint targets
    R               reset environment
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
    parser.add_argument("--rot_sensitivity", type=float, default=0.04, help="Keyboard rotation sensitivity.")
    parser.add_argument(
        "--lock_orientation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Lock end-effector rotation commands for stable Newton IK teleoperation.",
    )
    parser.add_argument(
        "--hold_gripper_open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hold all Robotiq joints open for Newton stability instead of commanding gripper open/close.",
    )
    parser.add_argument(
        "--direct_gripper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drive Robotiq joints directly in Newton instead of relying on mimic/finger_joint only.",
    )
    parser.add_argument(
        "--debug_joints",
        action="store_true",
        help="Print live joint positions while teleoperating.",
    )
    parser.add_argument(
        "--debug_actions",
        action="store_true",
        help="Print raw keyboard actions, processed actions, and link_6 motion diagnostics.",
    )
    parser.add_argument(
        "--control_mode",
        choices=("joint_direct", "joint_relative", "ik"),
        default="joint_direct",
        help="Newton arm control mode. joint_direct writes joint positions directly; ik uses task-space differential IK.",
    )
    parser.add_argument(
        "--joint_step_scale",
        type=float,
        default=0.025,
        help="Radians added to a driven ABB joint per keyboard tick in joint_direct mode.",
    )
    parser.set_defaults(num_envs=1, visualizer=["kit"], enable_cameras=False, presets="newton")
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _find_irb1200_robotiq_usd_path() -> str:
    from isaaclab_arena.embodiments.abb.abb_irb1200 import find_irb1200_robotiq_2f140_usd_path

    return find_irb1200_robotiq_2f140_usd_path(require_exists=True)


_ROBOTIQ_2F140_JOINTS = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_outer_finger_joint",
    "right_outer_finger_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_pad_joint",
    "right_inner_finger_pad_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
]

_ROBOTIQ_2F140_DIRECT_JOINTS = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_outer_finger_joint",
    "right_outer_finger_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_pad_joint",
    "right_inner_finger_pad_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
]

_ROBOTIQ_2F140_DIRECT_OPEN_POSE = {
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "left_outer_finger_joint": 0.0,
    "right_outer_finger_joint": 0.0,
    "left_inner_finger_joint": 0.0,
    "right_inner_finger_joint": 0.0,
    "left_inner_finger_pad_joint": 0.0,
    "right_inner_finger_pad_joint": 0.0,
    "left_inner_knuckle_joint": 0.0,
    "right_inner_knuckle_joint": 0.0,
}

_ROBOTIQ_2F140_DIRECT_CLOSE_POSE = {
    "finger_joint": 0.8,
    "right_outer_knuckle_joint": 0.8,
    "left_outer_finger_joint": 0.0,
    "right_outer_finger_joint": 0.0,
    "left_inner_finger_joint": 0.0,
    "right_inner_finger_joint": 0.0,
    "left_inner_finger_pad_joint": 0.0,
    "right_inner_finger_pad_joint": 0.0,
    "left_inner_knuckle_joint": -0.8,
    "right_inner_knuckle_joint": -0.8,
}


def _write_joint_pose_and_zero_velocity(robot_articulation, joint_ids, joint_targets) -> None:
    import torch

    robot_articulation.write_joint_position_to_sim_index(
        position=joint_targets,
        joint_ids=joint_ids,
    )
    robot_articulation.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(joint_targets),
        joint_ids=joint_ids,
    )


def main() -> None:
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, RelativeJointPositionActionCfg
    from isaaclab.managers import ActionTermCfg
    import isaaclab.sim as sim_utils
    from isaaclab.utils.configclass import configclass
    from omni.usd import get_context

    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    robot = asset_registry.get_asset_by_name("abb_irb1200_ik")(enable_cameras=False)

    @configclass
    class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
        class_type = BinaryJointPositionZeroToOneAction

    @configclass
    class ABBIRB1200RobotiqActionCfg:
        arm_action: ActionTermCfg = (
            RelativeJointPositionActionCfg(
                asset_name="robot",
                joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
                scale=0.25,
                preserve_order=True,
            )
            if args_cli.control_mode == "joint_relative"
            else robot.action_config.arm_action
        )
        gripper_action: ActionTermCfg = BinaryJointPositionZeroToOneActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": torch.pi / 4},
        )

    robotiq_usd_path = _find_irb1200_robotiq_usd_path()
    robot.scene_config.robot.spawn = robot.scene_config.robot.spawn.replace(
        usd_path=robotiq_usd_path,
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
    if args_cli.hold_gripper_open or args_cli.direct_gripper:
        robot.scene_config.robot.actuators["gripper_hold_open"] = ImplicitActuatorCfg(
            joint_names_expr=_ROBOTIQ_2F140_DIRECT_JOINTS,
            effort_limit_sim=120.0,
            velocity_limit_sim=1.5,
            stiffness=80.0,
            damping=18.0,
        )
    else:
        robot.scene_config.robot.actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=5.0,
            stiffness=80.0,
            damping=8.0,
        )
    robot.scene_config.robot.init_state = robot.scene_config.robot.init_state.replace(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": -0.35,
            "joint_3": 0.65,
            "joint_4": 0.0,
            "joint_5": 0.75,
            "joint_6": 0.0,
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            "right_outer_finger_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "left_inner_finger_pad_joint": 0.0,
            "right_inner_finger_pad_joint": 0.0,
            "left_inner_knuckle_joint": 0.0,
            "right_inner_knuckle_joint": 0.0,
        }
    )
    robot.action_config = ABBIRB1200RobotiqActionCfg()
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
        name="teleop_abb_irb1200_keyboard_robotiq_newton",
        embodiment=robot,
        scene=scene,
        task=NoTask(),
        teleop_device=teleop_device,
    )

    print("[INFO] Building ABB IRB1200 Newton keyboard Arena environment...")
    print(f"[INFO] ABB IRB1200 controllable Robotiq 2F-140 USD: {robotiq_usd_path}")
    if args_cli.hold_gripper_open:
        print("[INFO] Newton debug mode: Robotiq 2F-140 joints are held open for articulation stability.")
    if args_cli.control_mode == "joint_relative":
        print("[INFO] Newton debug mode: keyboard XYZ/RPY commands are mapped to relative ABB joint targets.")
    elif args_cli.control_mode == "joint_direct":
        print("[INFO] Newton debug mode: keyboard commands directly update ABB joint positions.")
    env_builder = ArenaEnvBuilder(env_cfg, builder_cfg)
    manager_env_cfg, env_kwargs = env_builder.compose_manager_cfg()
    manager_env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    if manager_env_cfg.sim.physics is not None:
        manager_env_cfg.sim.physics.num_substeps = 6
        solver_cfg = getattr(manager_env_cfg.sim.physics, "solver_cfg", None)
        if solver_cfg is not None:
            solver_cfg.iterations = 150
            solver_cfg.ls_iterations = 25
    print("[INFO] Newton debug mode: global gravity disabled for stable IK teleoperation.")
    env = env_builder.make_registered(manager_env_cfg, env_kwargs)
    stage = get_context().get_stage()
    print("[INFO] Environment built. Resetting...")
    env.reset()
    simulation_app.update()

    gripper_prim = stage.GetPrimAtPath(
        "/World/envs/env_0/Robot/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6/robotiq_2f140_mount"
    )
    robotiq_base_prim = stage.GetPrimAtPath(
        "/World/envs/env_0/Robot/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6/"
        "robotiq_2f140_mount/robotiq_base_link"
    )
    robotiq_finger_joint_prim = stage.GetPrimAtPath(
        "/World/envs/env_0/Robot/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6/"
        "robotiq_2f140_mount/finger_joint"
    )
    print(f"[CHECK] robotiq_2f140_mount exists: {gripper_prim.IsValid()}")
    print(f"[CHECK] Robotiq base_link exists: {robotiq_base_prim.IsValid()}")
    print(f"[CHECK] Robotiq finger_joint prim exists: {robotiq_finger_joint_prim.IsValid()}")

    robot_articulation = None
    joint_name_to_index = {}
    link6_body_index = None
    last_link6_pos = None
    arm_joint_ids = None
    direct_joint_targets = None
    gripper_joint_ids = None
    gripper_open_targets = None
    gripper_close_targets = None
    with contextlib.suppress(Exception):
        robot_articulation = env.unwrapped.scene["robot"]
        joint_name_to_index = {name: idx for idx, name in enumerate(robot_articulation.data.joint_names)}
        if "link_6" in robot_articulation.data.body_names:
            link6_body_index = robot_articulation.data.body_names.index("link_6")
        arm_joint_ids = [joint_name_to_index[name] for name in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")]
        direct_joint_targets = robot_articulation.data.joint_pos.torch[:, arm_joint_ids].clone()
        gripper_joint_names = [name for name in _ROBOTIQ_2F140_DIRECT_JOINTS if name in joint_name_to_index]
        gripper_joint_ids = [joint_name_to_index[name] for name in gripper_joint_names]
        gripper_open_targets = torch.tensor(
            [_ROBOTIQ_2F140_DIRECT_OPEN_POSE[name] for name in gripper_joint_names],
            device=direct_joint_targets.device,
            dtype=direct_joint_targets.dtype,
        ).repeat(env.unwrapped.num_envs, 1)
        gripper_close_targets = torch.tensor(
            [_ROBOTIQ_2F140_DIRECT_CLOSE_POSE[name] for name in gripper_joint_names],
            device=direct_joint_targets.device,
            dtype=direct_joint_targets.dtype,
        ).repeat(env.unwrapped.num_envs, 1)
        print(f"[INFO] ABB IRB1200 joints: {robot_articulation.data.joint_names}")
        print(f"[INFO] ABB IRB1200 bodies: {robot_articulation.data.body_names}")
        print(f"[INFO] Robotiq direct joints: {gripper_joint_names}")
        if gripper_joint_ids and gripper_open_targets is not None:
            _write_joint_pose_and_zero_velocity(robot_articulation, gripper_joint_ids, gripper_open_targets)
        if arm_joint_ids and direct_joint_targets is not None and args_cli.control_mode == "joint_direct":
            _write_joint_pose_and_zero_velocity(robot_articulation, arm_joint_ids, direct_joint_targets)
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
        if args_cli.hold_gripper_open:
            gripper_msg = "Gripper is held open."
        elif args_cli.direct_gripper:
            gripper_msg = "K toggles Robotiq direct joint targets."
        else:
            gripper_msg = "K toggles Robotiq finger_joint."
        print(
            "ABB IRB1200 Newton keyboard teleop started: W/S A/D Q/E drive joints, "
            f"Z/X T/G C/V rotate joints, R resets. {gripper_msg}"
        )

        step = 0
        while simulation_app.is_running():
            if should_reset:
                should_reset = False
                try:
                    print("[INFO] Resetting environment...")
                    with torch.inference_mode():
                        env.reset()
                    if robot_articulation is not None and arm_joint_ids is not None:
                        direct_joint_targets = robot_articulation.data.joint_pos.torch[:, arm_joint_ids].clone()
                    if robot_articulation is not None and gripper_joint_ids is not None and gripper_open_targets is not None:
                        _write_joint_pose_and_zero_velocity(robot_articulation, gripper_joint_ids, gripper_open_targets)
                    teleop_interface.reset()
                    simulation_app.update()
                    print("[INFO] Reset complete.")
                except Exception:
                    print("[ERROR] Environment reset failed:")
                    traceback.print_exc()
                continue

            try:
                with torch.inference_mode():
                    raw_action = teleop_interface.advance()
                    action = raw_action[:7].repeat(env.unwrapped.num_envs, 1)
                    if args_cli.lock_orientation:
                        action[:, 3:6] = 0.0
                    if args_cli.hold_gripper_open or args_cli.direct_gripper:
                        action[:, 6] = 0.0
                    if args_cli.control_mode == "joint_relative":
                        se3_action = action.clone()
                        action[:, :6] = 0.0
                        action[:, 0] = se3_action[:, 1]
                        action[:, 1] = -se3_action[:, 0]
                        action[:, 2] = se3_action[:, 2]
                        action[:, 3] = se3_action[:, 3]
                        action[:, 4] = se3_action[:, 4]
                        action[:, 5] = se3_action[:, 5]
                    elif args_cli.control_mode == "joint_direct":
                        se3_action = action.clone()
                        action[:, :6] = 0.0
                        action[:, 0] = se3_action[:, 1]
                        action[:, 1] = -se3_action[:, 0]
                        action[:, 2] = se3_action[:, 2]
                        action[:, 3] = se3_action[:, 3]
                        action[:, 4] = se3_action[:, 4]
                        action[:, 5] = se3_action[:, 5]
                    action_is_nonzero = bool(torch.any(torch.abs(raw_action[:6]) > 1e-6).item())
                    if step % 120 == 0:
                        print(f"[DEBUG] Keyboard raw 7D action: {raw_action[:7].detach().cpu().tolist()}")
                    if args_cli.debug_actions and (action_is_nonzero or step % 60 == 0):
                        raw_debug = [round(float(v), 5) for v in raw_action[:7].detach().cpu().tolist()]
                        action_debug = [round(float(v), 5) for v in action[0, :7].detach().cpu().tolist()]
                        print(f"[DEBUG] raw={raw_debug} processed={action_debug}")
                    if args_cli.debug_joints and robot_articulation is not None and step % 30 == 0:
                        joint_pos = robot_articulation.data.joint_pos.torch[0].detach().cpu()
                        joint_debug = {
                            name: round(float(joint_pos[joint_name_to_index[name]]), 4)
                            for name in (
                                "joint_1",
                                "joint_2",
                                "joint_3",
                                "joint_4",
                                "joint_5",
                                "joint_6",
                                "finger_joint",
                                "left_outer_finger_joint",
                                "left_inner_finger_joint",
                                "right_outer_finger_joint",
                                "right_inner_finger_joint",
                            )
                            if name in joint_name_to_index
                        }
                        print(f"[DEBUG] Joint positions rad: {joint_debug}")
                    if (
                        args_cli.control_mode == "joint_direct"
                        and robot_articulation is not None
                        and arm_joint_ids is not None
                        and direct_joint_targets is not None
                    ):
                        direct_action = action[:, :6].to(device=direct_joint_targets.device)
                        direct_joint_targets = direct_joint_targets + torch.sign(direct_action) * args_cli.joint_step_scale
                        _write_joint_pose_and_zero_velocity(robot_articulation, arm_joint_ids, direct_joint_targets)
                    if (
                        (args_cli.hold_gripper_open or args_cli.direct_gripper)
                        and robot_articulation is not None
                        and gripper_joint_ids is not None
                        and gripper_open_targets is not None
                    ):
                        if args_cli.hold_gripper_open:
                            gripper_targets = gripper_open_targets
                        elif raw_action[6].item() < 0.0 and gripper_close_targets is not None:
                            gripper_targets = gripper_close_targets
                        else:
                            gripper_targets = gripper_open_targets
                        robot_articulation.set_joint_position_target_index(
                            target=gripper_targets,
                            joint_ids=gripper_joint_ids,
                        )
                        _write_joint_pose_and_zero_velocity(robot_articulation, gripper_joint_ids, gripper_targets)
                    env.step(action)
                    if (
                        args_cli.control_mode == "joint_direct"
                        and robot_articulation is not None
                        and arm_joint_ids is not None
                        and direct_joint_targets is not None
                    ):
                        _write_joint_pose_and_zero_velocity(robot_articulation, arm_joint_ids, direct_joint_targets)
                    if (
                        (args_cli.hold_gripper_open or args_cli.direct_gripper)
                        and robot_articulation is not None
                        and gripper_joint_ids is not None
                        and gripper_open_targets is not None
                    ):
                        _write_joint_pose_and_zero_velocity(robot_articulation, gripper_joint_ids, gripper_targets)
                    if args_cli.debug_actions and robot_articulation is not None and link6_body_index is not None:
                        link6_pos = robot_articulation.data.body_pos_w.torch[0, link6_body_index].detach().clone()
                        if last_link6_pos is None:
                            last_link6_pos = link6_pos
                        if action_is_nonzero or step % 60 == 0:
                            delta = link6_pos - last_link6_pos
                            pos_debug = [round(float(v), 5) for v in link6_pos.cpu().tolist()]
                            delta_debug = [round(float(v), 5) for v in delta.cpu().tolist()]
                            print(f"[DEBUG] link_6_pos_w={pos_debug} delta_since_last_debug={delta_debug}")
                            last_link6_pos = link6_pos
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
        print("[ERROR] ABB IRB1200 Newton keyboard teleop startup failed:")
        traceback.print_exc()
    finally:
        simulation_app.close()
