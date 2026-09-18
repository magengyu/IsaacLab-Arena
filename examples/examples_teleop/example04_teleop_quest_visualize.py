"""Level 4：Quest XR 遥操作 + 手柄位姿可视化。

在 example03（锚点修正版）基础上，额外把「右手柄在世界系的位姿」画出来，
用于直观检查锚点（anchor_pos / anchor_rot）对不对：
  - 红色小球 = 手柄位置
  - 三条坐标轴（RGB）= 手柄朝向（X 红 / Y 绿 / Z 蓝）

原理：IsaacTeleop 管线里 `transformed_controllers.output(ControllersSource.RIGHT)`
就是手柄经过 world_T_anchor 变换后的世界系位姿。这里自定义了一条管线，把
这个位姿和 action 一起输出，再用 omni.isaac.debug_draw 画出来。

运行步骤同 example03（两个终端）：
  1. 终端 1：/isaac-sim/python.sh -m isaacteleop.cloudxr --host-client
  2. 终端 2：source ~/.cloudxr/run/cloudxr.env
             /isaac-sim/python.sh examples/examples_teleop/example04_teleop_quest_visualize.py --xr
  3. Quest 浏览器 Connect → XR 标签页 Start Session → Play
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


def build_teleop_pipeline_with_controller_pose():
    """在 example03 的 action 之外，额外输出右手柄的世界系位姿。"""
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3RelRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    ee_delta = Se3RelRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            delta_pos_scale_factor=5.0,
            delta_rot_scale_factor=1.0,
            alpha_pos=0.5,
            alpha_rot=0.3,
        ),
        name="right_controller_ee_delta",
    ).connect({ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)})

    gripper = GripperRetargeter(
        GripperRetargeterConfig(hand_side="right", controller_threshold=0.5),
        name="right_controller_gripper",
    ).connect({
        HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
    })

    pose_elements = ["dx", "dy", "dz", "rx", "ry", "rz"]
    gripper_elements = ["gripper"]
    action = TensorReorderer(
        input_config={"ee_delta": pose_elements, "gripper": gripper_elements},
        output_order=pose_elements + gripper_elements,
        input_types={"ee_delta": "array", "gripper": "scalar"},
        name="single_arm_action",
    ).connect({
        "ee_delta": ee_delta.output("ee_delta"),
        "gripper": gripper.output("gripper_command"),
    })

    # ★ 额外输出右手柄世界系位姿，供可视化
    return OutputCombiner({
        "action": action.output("output"),
        "controller_pose": transformed_controllers.output(ControllersSource.RIGHT),
    })


def _extract_controller_pose(controller_pose) -> tuple:
    """从 ControllerInput 张量组里取出位置 (3,) 和朝向 quat xyzw (4,)。

    与 isaacteleop 的 se3_retargeter 同款取法：用 np.from_dlpack 把 DLPack 张量转 numpy。
    """
    import numpy as np

    from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex

    grip_pos = np.from_dlpack(controller_pose[ControllerInputIndex.GRIP_POSITION])  # (3,)
    grip_ori = np.from_dlpack(controller_pose[ControllerInputIndex.GRIP_ORIENTATION])  # (4,) xyzw
    return grip_pos, grip_ori


def main() -> None:
    """构建最简场景并通过 Quest 右手柄遥操作，同时可视化手柄位姿。"""
    print("[XR] Isaac Sim 已启动，开始构建 Arena 环境。", flush=True)

    import numpy as np
    import torch

    from isaacsim.util.debug_draw import _debug_draw

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

    table = asset_registry.get_asset_by_name("table")()
    robot = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=False)
    cube = asset_registry.get_asset_by_name("dex_cube")()
    light = asset_registry.get_asset_by_name("light")()

    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
    cube.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.04)))

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

        env.unwrapped.cfg.isaac_teleop.control_channel_uuid = None

        # ★ 方式 B：锚点修正（同 example03）
        env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_pos = (0.0, 0.0, -0.7)
        env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_rot = (0.0, 0.0, -0.70711, 0.70711)
        env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_prim_path = "/World/envs/env_0/Robot/panda_link0"
        env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
        env.unwrapped.cfg.isaac_teleop.xr_cfg.fixed_anchor_height = True

        # ★ 换用自定义管线，额外输出手柄位姿
        env.unwrapped.cfg.isaac_teleop.pipeline_builder = build_teleop_pipeline_with_controller_pose

        def request_reset() -> None:
            nonlocal should_reset
            should_reset = True

        teleop_interface = create_isaac_teleop_device(
            env.unwrapped.cfg.isaac_teleop,
            sim_device=str(env.unwrapped.device),
            callbacks={"R": request_reset},
        )
        draw = _debug_draw.acquire_debug_draw_interface()
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
                        result = teleop_interface.advance()
                        if result is None:
                            env.unwrapped.sim.render()
                            continue

                        # 自定义管线返回 dict：{"action": ..., "controller_pose": ...}
                        if isinstance(result, dict):
                            action = result["action"]
                            pos, quat = _extract_controller_pose(result["controller_pose"])
                            pos = np.asarray(pos, dtype=np.float64).reshape(-1)
                            quat = np.asarray(quat, dtype=np.float64).reshape(-1)
                            if pos.shape[0] == 3 and quat.shape[0] == 4:
                                from scipy.spatial.transform import Rotation

                                # 位置：红色点（debug_draw 接受 (x,y,z) 元组）
                                p = tuple(float(v) for v in pos)
                                draw.clear_points()
                                draw.draw_points([p], [(1.0, 0.0, 0.0, 1.0)], [12.0])

                                # 朝向：三条轴（X 红 / Y 绿 / Z 蓝），quat 是 xyzw
                                rot = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_matrix()
                                scale = 0.15
                                starts = [p, p, p]
                                ends = [tuple(float(v) for v in (pos + rot[:, i] * scale)) for i in range(3)]
                                axis_colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
                                draw.clear_lines()
                                draw.draw_lines(starts, ends, axis_colors, [2.0, 2.0, 2.0])
                        else:
                            action = result

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
        _debug_draw.release_debug_draw_interface(draw)
        env.close()


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
