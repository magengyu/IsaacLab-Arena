# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton 刚体仿真：本地水壶和目标容器置于静态台面上。

使用方法：

    .venv/bin/python examples/newton/example02_buildscene.py --viz [newton, kit, rerun, viser, none]
"""

import argparse
import contextlib
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Newton 水壶与容器场景示例。")
parser.add_argument("--log_every", type=int, default=1000, help="每隔多少物理步打印一次水壶位置。")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["newton"], device="cuda:0")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """创建 Newton 场景并执行固定数量的物理步。"""
    import newton
    import newton.solvers

    # IsaacLab Newton 0.13.6 imports the pre-1.5 name. Newton 1.5 moved the
    # same bit-mask enum to ``newton.ModelFlags``. This must run after Kit
    # starts so Newton does not preload a conflicting pxr binding.
    if not hasattr(newton.solvers, "SolverNotifyFlags"):
        newton.solvers.SolverNotifyFlags = newton.ModelFlags

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from pxr import UsdGeom

    physics_cfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(solver="newton", integrator="implicitfast"),
        default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.0),
        num_substeps=5,
    )
    assert args_cli.log_every > 0, "--log_every 必须为正整数。"
    sim_cfg = SimulationCfg(device=args_cli.device, dt=1.0 / 1200.0, physics=physics_cfg)
    visualizers = [] if args_cli.visualizer == ["none"] else args_cli.visualizer

    with build_simulation_context(
        sim_cfg=sim_cfg,
        add_lighting=True,
        visualizers=visualizers,
    ) as sim:
        camera_eye = (1.6, -1.6, 1.2)
        camera_target = (0.45, 0.0, 0.1)

        def set_z_up_camera() -> None:
            """Set the interactive camera with the world Z axis pointing upward."""
            UsdGeom.SetStageUpAxis(sim.stage, UsdGeom.Tokens.z)
            sim.set_camera_view(eye=camera_eye, target=camera_target)

        # PhysX contactOffset is converted to Newton shape_gap. Explicitly
        # remove it so contacts become active at the geometric surface.
        hard_collision_cfg = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.0,
            rest_offset=0.0,
        )
        hard_contact_material = sim_utils.PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            compliant_contact_stiffness=1.0e6,
            compliant_contact_damping=2.0e3,
        )

        table = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Table",
                spawn=sim_utils.CuboidCfg(
                    size=(1.2, 1.2, 0.1),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=hard_collision_cfg,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
                    physics_material=hard_contact_material,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
            )
        )
        jug = RigidObject(
            RigidObjectCfg(
                prim_path="/World/FStyleJug",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/home/magengyu/IsaacLab-Arena/scene/fstylejug_a01/fstylejug_a01_inst_physx.usd",
                    collision_props=hard_collision_cfg,
                    physics_material=hard_contact_material,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.25, 0.01)),
            )
        )
        container = RigidObject(
            RigidObjectCfg(
                prim_path="/World/TargetContainer",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=(
                        "/home/magengyu/IsaacLab-Arena/scene/Container_B04_40x30x12cm/"
                        "Container_B04_40x30x12cm_PR_V_NVD_01.usd"
                    ),
                    scale=(0.01, 0.01, 0.01),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=hard_collision_cfg,
                    physics_material=hard_contact_material,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.22, 0.0)),
            )
        )

        sim.reset()
        table.reset()
        jug.reset()
        container.reset()
        # Kit/RTX creates its viewport resources during reset/render. Apply the
        # camera afterwards so its Z-up roll is not replaced by the startup pose.
        sim.render()
        set_z_up_camera()
        sim.render()
        print("[INFO] Newton 场景已就绪，开始持续执行物理步。按 Ctrl+C 或关闭 visualizer 停止。", flush=True)

        step = 0
        while True:
            if sim.visualizers and not any(
                visualizer.is_running() and not visualizer.is_closed for visualizer in sim.visualizers
            ):
                print(f"[INFO] visualizer 在第 {step} 步前关闭。", flush=True)
                break
            table.write_data_to_sim()
            jug.write_data_to_sim()
            container.write_data_to_sim()
            sim.step()
            if step == 0:
                # Newton's deferred CUDA graph and Cubric/Fabric synchronization
                # initialize on the first step and may replace the active Kit pose.
                set_z_up_camera()
                sim.render()
            table.update(sim.get_physics_dt())
            jug.update(sim.get_physics_dt())
            container.update(sim.get_physics_dt())
            step += 1
            if step % args_cli.log_every == 0:
                position = jug.data.root_pos_w.torch[0].detach().cpu().tolist()
                print(
                    f"[INFO] step={step:04d}, 水壶世界坐标: "
                    f"x={position[0]:+.5f}, y={position[1]:+.5f}, z={position[2]:+.5f}",
                    flush=True,
                )
            time.sleep(sim.get_physics_dt())

        print("[INFO] Newton 仿真结束。", flush=True)


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
