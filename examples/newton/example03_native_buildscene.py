# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton 原生对照场景：运动学长方体台面和一个自由落体的 YCB 饼干盒。

该场景不经过 IsaacLab，但使用与 ``example02_buildscene.py`` 相同的运动学
台面和 ``Axis_Aligned_Physics`` USD 资产，用于隔离 IsaacLab 集成层的影响。

运行方法：

    .venv/bin/python examples/newton/example03_native_buildscene.py
    .venv/bin/python examples/newton/example03_native_buildscene.py --log-every 10
"""

import urllib.request
import warnings
from pathlib import Path

import warp as wp

import newton
import newton.examples
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx


ISAAC_ASSET_ROOT_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac"
CRACKER_BOX_PHYSICS_USD_URL = f"{ISAAC_ASSET_ROOT_URL}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
CRACKER_BOX_VISUAL_USD_URL = f"{ISAAC_ASSET_ROOT_URL}/Props/YCB/Axis_Aligned/003_cracker_box.usd"
CRACKER_BOX_TEXTURE_URL = (
    f"{ISAAC_ASSET_ROOT_URL}/Props/YCB/Axis_Aligned/Materials/Textures/003_cracker_box_COLOR.png"
)

TABLE_HALF_EXTENTS = (0.6, 0.6, 0.05)
TABLE_POSITION = (0.0, 0.0, -0.05)
CRACKER_BOX_POSITION = (0.0, 0.0, 0.8)


def _download_ycb_asset(url: str, relative_path: str) -> Path:
    """Download one YCB asset while preserving its USD-relative directory layout."""
    asset_path = Path(__file__).resolve().parents[2] / ".usd_cache" / "isaac_ycb" / relative_path
    if asset_path.is_file():
        return asset_path

    asset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = asset_path.with_suffix(f"{asset_path.suffix}.part")
    try:
        urllib.request.urlretrieve(url, temporary_path)
        temporary_path.replace(asset_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return asset_path


def _get_cracker_box_physics_usd() -> Path:
    """Cache the physics USD and all relative assets required to compose it."""
    physics_usd = _download_ycb_asset(
        CRACKER_BOX_PHYSICS_USD_URL,
        "Axis_Aligned_Physics/003_cracker_box.usd",
    )
    _download_ycb_asset(
        CRACKER_BOX_VISUAL_USD_URL,
        "Axis_Aligned/003_cracker_box.usd",
    )
    _download_ycb_asset(
        CRACKER_BOX_TEXTURE_URL,
        "Axis_Aligned/Materials/Textures/003_cracker_box_COLOR.png",
    )
    return physics_usd


class Example:
    """Build and simulate the IsaacLab-equivalent scene with native Newton APIs."""

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

        # Match example02: the table is a kinematic rigid body rather than a world shape.
        table_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(*TABLE_POSITION), q=wp.quat_identity()),
            is_kinematic=True,
            label="table",
        )
        builder.add_shape_box(
            body=table_body,
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            color=wp.vec3(0.2, 0.2, 0.2),
            label="table",
        )

        # Match example02: import the complete Physics USD instead of replacing
        # its authored collision mesh with a hand-built box collider.
        cracker_box_result = builder.add_usd(
            str(_get_cracker_box_physics_usd()),
            xform=wp.transform(p=wp.vec3(*CRACKER_BOX_POSITION), q=wp.quat_identity()),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        cracker_box_bodies = list(cracker_box_result["path_body_map"].values())
        assert len(cracker_box_bodies) == 1, "饼干盒 Physics USD 应当只包含一个刚体。"
        self.cracker_box_body = cracker_box_bodies[0]

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
