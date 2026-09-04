# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""最小 Newton 刚体仿真：一个本地生成的饼干盒落到静态台面上。

运行示例：

    python examples/newton/example02_buildscene.py --viz newton
    python examples/newton/example02_buildscene.py --viz kit
    python examples/newton/example02_buildscene.py --viz none --num_steps 240 --log_every 30
"""

import argparse
import contextlib
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Newton 最小刚体场景示例。")
parser.add_argument("--log_every", type=int, default=1, help="每隔多少物理步打印一次饼干盒位置。")
parser.add_argument("--num_steps", type=int, default=0, help="最大物理步数；0 表示持续运行。")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["newton"], device="cuda:0")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """创建 Newton 场景并执行物理步。"""
    import newton
    import newton.solvers

    # 兼容 Newton 将 SolverNotifyFlags 移到顶层 ModelFlags 的版本。
    if not hasattr(newton.solvers, "SolverNotifyFlags"):
        newton.solvers.SolverNotifyFlags = newton.ModelFlags

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context

    assert args_cli.log_every > 0, "--log_every 必须为正整数。"
    assert args_cli.num_steps >= 0, "--num_steps 不能为负数。"

    physics_cfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(solver="newton", integrator="implicitfast"),
        num_substeps=5,
    )
    sim_cfg = SimulationCfg(device=args_cli.device, dt=1.0 / 120.0, physics=physics_cfg)
    visualizers = [] if args_cli.visualizer == ["none"] else args_cli.visualizer

    with build_simulation_context(
        sim_cfg=sim_cfg,
        add_lighting=True,
        visualizers=visualizers,
    ) as sim:
        sim.set_camera_view(eye=(2.0, -2.0, 1.5), target=(0.0, 0.0, 0.3))

        table = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Table",
                spawn=sim_utils.CuboidCfg(
                    size=(1.2, 1.2, 0.1),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
            )
        )
        cracker_box = RigidObject(
            RigidObjectCfg(
                prim_path="/World/CrackerBox",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
            )
        )

        sim.reset()
        table.reset()
        cracker_box.reset()
        print("[INFO] Newton 场景已就绪。按 Ctrl+C 或关闭 visualizer 停止。", flush=True)

        step = 0
        while True:
            if sim.visualizers and not any(
                visualizer.is_running() and not visualizer.is_closed for visualizer in sim.visualizers
            ):
                print(f"[INFO] visualizer 在第 {step} 步前关闭。", flush=True)
                break

            table.write_data_to_sim()
            cracker_box.write_data_to_sim()
            sim.step()
            table.update(sim.get_physics_dt())
            cracker_box.update(sim.get_physics_dt())
            step += 1

            if step % args_cli.log_every == 0:
                position = cracker_box.data.root_pos_w.torch[0].detach().cpu().tolist()
                print(
                    f"[INFO] step={step:04d}, CrackerBox 世界坐标: "
                    f"x={position[0]:+.5f}, y={position[1]:+.5f}, z={position[2]:+.5f}",
                    flush=True,
                )

            if args_cli.num_steps > 0 and step >= args_cli.num_steps:
                print(f"[INFO] 已完成指定的 {args_cli.num_steps} 个 Newton 物理步。", flush=True)
                break
            time.sleep(sim.get_physics_dt())

        print("[INFO] Newton 仿真结束。", flush=True)


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
