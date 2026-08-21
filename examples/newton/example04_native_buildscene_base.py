# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton 原生基准场景：静态台面和使用简化盒碰撞体的 YCB 饼干盒。

该场景不经过 IsaacLab。饼干盒的 USD 网格只用于显示，物理接触使用单一
长方体碰撞模型，用于和 ``example03_native_buildscene.py`` 的 Physics USD
碰撞模型进行对比。

运行方法：

    .venv/bin/python examples/newton/example04_native_buildscene_base.py
    .venv/bin/python examples/newton/example04_native_buildscene_base.py --log-every 10
"""

import warnings

import warp as wp

import newton
import newton.examples


ISAAC_ASSET_ROOT_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac"
CRACKER_BOX_USD_URL = f"{ISAAC_ASSET_ROOT_URL}/Props/YCB/Axis_Aligned/003_cracker_box.usd"
CRACKER_BOX_TEXTURE_URL = (
    f"{ISAAC_ASSET_ROOT_URL}/Props/YCB/Axis_Aligned/Materials/Textures/003_cracker_box_COLOR.png"
)

TABLE_HALF_EXTENTS = (0.6, 0.6, 0.05)
TABLE_POSITION = (0.0, 0.0, -0.05)
CRACKER_BOX_HALF_EXTENTS = (0.082018, 0.106719, 0.0359)
CRACKER_BOX_POSITION = (0.0, 0.0, 0.8)
CRACKER_BOX_MASS = 0.411


class Example:
    """Build and simulate the simplified native-Newton baseline scene."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True

        self.viewer = viewer
        self.log_every = args.log_every
        self.step_count = 0
        self.frame_dt = 1.0 / 120.0
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 1.0

        # Static world shape: 1.2 x 1.2 x 0.1 m, with its top surface at z=0.
        table_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(*TABLE_POSITION), q=wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=wp.vec3(0.2, 0.2, 0.2),
            label="table",
        )

        # Use one box for contact; the YCB mesh below is visual-only.
        cracker_box_mesh = newton.Mesh.create_from_usd(
            CRACKER_BOX_USD_URL,
            load_normals=True,
            load_uvs=True,
        )
        cracker_box_mesh.texture = CRACKER_BOX_TEXTURE_URL

        self.cracker_box_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(*CRACKER_BOX_POSITION), q=wp.quat_identity()),
            label="cracker_box",
        )
        hx, hy, hz = CRACKER_BOX_HALF_EXTENTS
        collider_cfg = newton.ModelBuilder.ShapeConfig(
            density=CRACKER_BOX_MASS / (8.0 * hx * hy * hz),
            is_visible=False,
        )
        builder.add_shape_box(
            self.cracker_box_body,
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=collider_cfg,
            label="cracker_box_collision",
        )
        visual_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        builder.add_shape_mesh(
            self.cracker_box_body,
            mesh=cracker_box_mesh,
            cfg=visual_cfg,
            label="cracker_box_visual",
        )

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            iterations=100,
            ls_iterations=50,
            ccd_iterations=35,
            impratio=1.0,
            cone="pyramidal",
            tolerance=1.0e-6,
            use_mujoco_cpu=False,
            use_mujoco_contacts=True,
            update_data_interval=1,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(2.0, -2.0, 1.5),
            pitch=-15.0,
            yaw=135.0,
        )
        self.graph = self._capture()

    def _simulate(self) -> None:
        """Advance one displayed frame using five MuJoCo-Warp substeps."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture(self):
        """Capture the simulation step when the active device supports it."""
        if not wp.get_device().is_cuda:
            return None
        try:
            with wp.ScopedCapture() as capture:
                self._simulate()
            return capture.graph
        except Exception as exc:
            warnings.warn(f"CUDA graph capture failed; using eager stepping: {exc}", stacklevel=2)
            return None

    def step(self) -> None:
        """Advance the simulation and print the cracker-box world position."""
        if self.graph is None:
            self._simulate()
        else:
            wp.capture_launch(self.graph)

        self.step_count += 1
        self.sim_time += self.frame_dt
        if self.step_count % self.log_every == 0:
            position = self.state_0.body_q.numpy()[self.cracker_box_body, :3]
            print(
                f"[INFO] step={self.step_count:04d}, CrackerBox 世界坐标: "
                f"x={position[0]:+.5f}, y={position[1]:+.5f}, z={position[2]:+.5f}",
                flush=True,
            )

    def render(self) -> None:
        """Render the current Newton state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--log-every", type=int, default=1, help="每隔多少帧打印一次饼干盒位置。")
    viewer, args = newton.examples.init(parser)
    assert args.log_every > 0, "--log-every 必须为正整数。"
    newton.examples.run(Example(viewer, args), args)
